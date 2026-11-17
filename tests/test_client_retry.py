"""Retry, backoff, Retry-After, timeouts and the total deadline."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest
from fakes import FAKE_KEYS, FakeClock, ScriptedProviders, err, google, make_client, ok, openrouter

from llm_toolkit.client import (
    AllProvidersFailed,
    ChatRequest,
    DeadlineExceeded,
    Message,
    RetryPolicy,
    parse_retry_after,
)

REQ = ChatRequest(messages=[Message.user("hi")], deadline_s=60)


def test_backoff_is_exponential_with_full_jitter():
    top = RetryPolicy(base_s=0.5, cap_s=8.0, rand=lambda: 1.0)
    assert [top.backoff(n) for n in range(6)] == [0.5, 1.0, 2.0, 4.0, 8.0, 8.0]
    half = RetryPolicy(base_s=0.5, cap_s=8.0, rand=lambda: 0.5)
    assert half.backoff(3) == 2.0

    # Full jitter: samples spread over [0, ceiling], never above it.
    rng = random.Random(7)
    real = RetryPolicy(base_s=0.5, cap_s=8.0, rand=rng.random)
    samples = [real.backoff(3) for _ in range(2000)]
    assert min(samples) >= 0 and max(samples) <= 4.0
    assert min(samples) < 0.2 and max(samples) > 3.8


def test_retry_after_seconds_and_http_date_are_honoured():
    now = datetime(2026, 11, 17, 12, 0, 0, tzinfo=UTC)
    assert parse_retry_after("7") == 7.0
    assert parse_retry_after(format_datetime(now + timedelta(seconds=12), usegmt=True), now=now) == 12.0
    assert parse_retry_after("garbage") is None
    assert parse_retry_after(None) is None

    clock = FakeClock()
    script = ScriptedProviders(clock).add("google", err(503, headers={"Retry-After": "3"}), ok())
    client = make_client(script, providers=[google(FAKE_KEYS)], clock=clock)
    res = client.chat(REQ)
    assert res.provider == "google"
    # Jitter is 0 in tests, so the only reason to wait 3 s is the header.
    assert clock.sleeps == [3.0]


def test_zero_retries_means_single_attempt():
    script = ScriptedProviders().add("google", err(503))
    spec = google(FAKE_KEYS)
    spec = type(spec)(**{**spec.__dict__, "max_retries": 0})
    client = make_client(script, providers=[spec])
    with pytest.raises(AllProvidersFailed):
        client.chat(REQ)
    assert script.count("google") == 1


def test_5xx_is_retried_with_growing_waits_then_succeeds():
    clock = FakeClock()
    script = ScriptedProviders(clock).add("openrouter", err(500), err(502), ok())
    client = make_client(
        script,
        providers=[openrouter(FAKE_KEYS)],
        clock=clock,
        retry=RetryPolicy(base_s=1.0, rand=lambda: 1.0),
    )
    res = client.chat(REQ)
    assert res.provider == "openrouter"
    assert clock.sleeps == [1.0, 2.0]
    assert [a.outcome for a in res.attempts] == ["retryable", "retryable", "ok"]


def test_timeout_and_transport_errors_are_retryable():
    script = (
        ScriptedProviders()
        .add(
            "google",
            httpx.ReadTimeout("slow"),
            httpx.ConnectError("refused"),
        )
        .add("openrouter", ok())
    )
    client = make_client(script)
    res = client.chat(REQ)
    assert res.provider == "openrouter"
    assert [a.detail.split(" ")[0] for a in res.attempts[:2]] == ["timeout", "transport"]


def test_connect_and_read_timeouts_are_separate():
    seen: list[dict] = []

    def capture(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"])
        return ok()

    script = ScriptedProviders().add("openrouter", capture)
    client = make_client(script, providers=[openrouter(FAKE_KEYS)])
    client.chat(REQ)
    assert seen[0]["connect"] == 5.0
    assert seen[0]["read"] == 30.0
    assert seen[0]["write"] == 10.0


def test_read_timeout_never_exceeds_remaining_deadline():
    clock = FakeClock()
    seen: list[float] = []

    def slow_then_fail(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"]["read"])
        clock.advance(4.0)
        raise httpx.ReadTimeout("slow")

    script = ScriptedProviders(clock).add("openrouter", slow_then_fail, slow_then_fail, slow_then_fail)
    client = make_client(script, providers=[openrouter(FAKE_KEYS)], clock=clock)
    with pytest.raises((DeadlineExceeded, AllProvidersFailed)):
        client.chat(ChatRequest(messages=[Message.user("hi")], deadline_s=10))
    # Spec read timeout is 30 s, but the request only has 10 s, then 6 s, then 2 s left.
    assert seen == [10.0, 6.0, 2.0]


def test_total_deadline_caps_retries_and_failover():
    """Prototype worst case was ~225 s (MP) or providers x 60 s (MT). Here the whole call, including
    retries, backoff waits and failover, stays inside the 30 s budget."""
    clock = FakeClock()

    def hang(request: httpx.Request) -> httpx.Response:
        clock.advance(request.extensions["timeout"]["read"])  # simulate hitting the read timeout
        raise httpx.ReadTimeout("hung")

    script = ScriptedProviders(clock).add("google", *[hang] * 5).add("openrouter", *[hang] * 5)
    client = make_client(script, clock=clock, retry=RetryPolicy(base_s=2.0, rand=lambda: 1.0))
    start = clock()
    with pytest.raises(DeadlineExceeded) as exc:
        client.chat(ChatRequest(messages=[Message.user("hi")], deadline_s=30))
    assert clock() - start <= 30.0
    assert exc.value.attempts, "the error carries the attempt list"


def test_min_interval_paces_requests_within_the_deadline():
    clock = FakeClock()
    script = ScriptedProviders(clock).add("google", ok(), ok(), ok())
    client = make_client(script, providers=[google(FAKE_KEYS)], clock=clock, min_interval_s=6.0)
    client.chat(REQ)
    clock.advance(2.0)
    client.chat(REQ)
    assert clock.sleeps == [4.0], "waits only the rest of the gap"
    with pytest.raises(DeadlineExceeded):
        client.chat(ChatRequest(messages=[Message.user("hi")], deadline_s=3))
    assert script.count("google") == 2, "a pacing wait that does not fit the deadline sends nothing"


def test_wait_that_would_cross_deadline_fails_over_instead():
    clock = FakeClock()
    script = (
        ScriptedProviders(clock)
        .add("google", err(503, headers={"Retry-After": "50"}))
        .add("openrouter", ok())
    )
    client = make_client(script, clock=clock)
    res = client.chat(ChatRequest(messages=[Message.user("hi")], deadline_s=20))
    assert res.provider == "openrouter"
    assert clock.sleeps == []
    assert "no time left" in res.attempts[0].detail
