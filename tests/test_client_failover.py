"""Failover, error classes, cooldown and gateway failures."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fakes import (
    FAKE_KEYS,
    FakeClock,
    ScriptedProviders,
    err,
    google,
    make_client,
    ok,
    openai_body,
    openrouter,
)

from llm_toolkit.client import (
    AllProvidersFailed,
    ChatRequest,
    CooldownStore,
    Message,
    ProviderRequestError,
)
from llm_toolkit.client.gateway import GATEWAY_ERROR_PATTERNS

REQ = ChatRequest(messages=[Message.user("hi")])


# ---- 4xx split ----------------------------------------------------------------------------


@pytest.mark.parametrize("status", [400, 422])
def test_bad_request_400_and_422_raise_without_failover(status):
    script = ScriptedProviders().add("google", err(status, '{"error": "bad field"}')).add("openrouter", ok())
    client = make_client(script)
    with pytest.raises(ProviderRequestError) as exc:
        client.chat(REQ)
    assert exc.value.status == status
    assert exc.value.provider == "google"
    assert script.count("google") == 1, "not retried"
    assert script.count("openrouter") == 0, "not failed over"
    assert [a.outcome for a in exc.value.attempts] == ["request_error"]


@pytest.mark.parametrize("status", [401, 403, 404])
def test_auth_and_model_errors_fail_over_with_error_log(status, caplog):
    script = ScriptedProviders().add("google", err(status, "denied")).add("openrouter", ok())
    client = make_client(script)
    with caplog.at_level(logging.ERROR, logger="llm_toolkit"):
        res = client.chat(REQ)
    assert res.provider == "openrouter"
    assert script.count("google") == 1, "not retried on the same provider"
    first = res.attempts[0]
    assert (first.provider, first.outcome, first.status) == ("google", "provider_error", status)
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors and f"HTTP {status}" in errors[0].getMessage() and "google" in errors[0].getMessage()


def test_provider_errors_on_every_provider_raise_all_failed():
    script = ScriptedProviders().add("google", err(401)).add("openrouter", err(404))
    client = make_client(script)
    with pytest.raises(AllProvidersFailed) as exc:
        client.chat(REQ)
    assert [(a.provider, a.outcome, a.status) for a in exc.value.attempts] == [
        ("google", "provider_error", 401),
        ("openrouter", "provider_error", 404),
    ]


def test_other_4xx_are_treated_as_request_errors():
    script = ScriptedProviders().add("google", err(413, "too large"))
    with pytest.raises(ProviderRequestError):
        make_client(script).chat(REQ)
    assert script.count("openrouter") == 0


# ---- cooldown -------------------------------------------------------------------------------


def test_quota_429_cools_down_provider_for_next_request():
    clock = FakeClock()
    script = (
        ScriptedProviders(clock)
        .add("google", err(429, '{"error": {"details": [{"quotaId": "GenerateRequestsPerDayPerProject"}]}}'))
        .add("openrouter", ok(), ok())
    )
    client = make_client(script, clock=clock)
    first = client.chat(REQ)
    assert first.provider == "openrouter"
    assert script.count("google") == 1, "quota 429 is not retried"
    second = client.chat(REQ)
    assert second.provider == "openrouter"
    assert script.count("google") == 1, "provider in cooldown is skipped"
    assert second.attempts[0].outcome == "skipped_cooldown"
    clock.advance(3601)
    assert not client.cooldowns.active("google")


def test_cooldown_from_retry_after_without_body_text():
    """The prototype only recognised Google's 'PerDay' text. A long Retry-After is enough here."""
    script = ScriptedProviders().add("openrouter", err(429, "slow down", headers={"Retry-After": "120"}))
    script.add("google", ok())
    client = make_client(script, providers=[openrouter(FAKE_KEYS), google(FAKE_KEYS)])
    client.chat(REQ)
    assert 119 < client.cooldowns.remaining("openrouter") <= 120


def test_short_429_is_retried_not_cooled_down():
    clock = FakeClock()
    script = ScriptedProviders(clock).add("google", err(429, "busy", headers={"Retry-After": "2"}), ok())
    client = make_client(script, clock=clock)
    res = client.chat(REQ)
    assert res.provider == "google" and clock.sleeps == [2.0]
    assert not client.cooldowns.active("google")


def test_cooldown_survives_restart_with_file_store(tmp_path: Path):
    path = tmp_path / "cooldowns.json"
    wall = [1_000_000.0]
    store = CooldownStore(path, monotonic=lambda: 50.0, wall=lambda: wall[0])
    store.set("google", 600)
    # New process: different monotonic origin, 100 s of wall time later.
    wall[0] += 100
    restarted = CooldownStore(path, monotonic=lambda: 7.0, wall=lambda: wall[0])
    assert restarted.active("google")
    assert restarted.remaining("google") == pytest.approx(500)


def test_corrupt_cooldown_file_is_ignored(tmp_path: Path):
    path = tmp_path / "cooldowns.json"
    path.write_text("{not json", encoding="utf-8")
    assert not CooldownStore(path).active("google")


# ---- gateway failures -------------------------------------------------------------------------


def test_disguised_gateway_error_fails_over():
    script = (
        ScriptedProviders()
        .add("google", ok(openai_body("Upstream error: the model returned no content.")))
        .add("openrouter", ok(openai_body("Real answer")))
    )
    res = make_client(script).chat(REQ)
    assert res.message.content == "Real answer"
    assert res.attempts[0].outcome == "gateway"
    assert script.count("google") == 1


def test_non_json_200_body_fails_over():
    script = ScriptedProviders().add("google", ok("<html>proxy page</html>")).add("openrouter", ok())
    res = make_client(script).chat(REQ)
    assert res.provider == "openrouter"
    assert res.attempts[0].outcome == "gateway"


def test_error_object_inside_200_fails_over():
    script = (
        ScriptedProviders().add("google", ok({"error": {"message": "overloaded"}})).add("openrouter", ok())
    )
    assert make_client(script).chat(REQ).provider == "openrouter"


def test_empty_answer_fails_over():
    script = ScriptedProviders().add("google", ok(openai_body(None))).add("openrouter", ok())
    res = make_client(script).chat(REQ)
    assert res.provider == "openrouter"
    assert "empty answer" in res.attempts[0].detail


def test_gateway_failure_on_last_provider_raises_all_failed():
    script = ScriptedProviders().add("google", ok(openai_body(""))).add("openrouter", ok("not json"))
    with pytest.raises(AllProvidersFailed):
        make_client(script).chat(REQ)


def test_long_answer_mentioning_bad_gateway_is_not_discarded():
    text = "A 502 Bad Gateway means the proxy got an invalid reply. " * 10
    script = ScriptedProviders().add("google", ok(openai_body(text)))
    assert make_client(script).chat(REQ).provider == "google"


def test_gateway_error_patterns_have_one_definition():
    """The prototype had two copies with different patterns. Exactly one module may define them."""
    src = Path(__file__).resolve().parents[1] / "src" / "llm_toolkit"
    hits = [
        p.relative_to(src).as_posix()
        for p in src.rglob("*.py")
        if any(pattern in p.read_text(encoding="utf-8") for pattern in GATEWAY_ERROR_PATTERNS)
    ]
    assert hits == ["client/gateway.py"]


# ---- keys ---------------------------------------------------------------------------------------


def test_missing_key_skips_provider_without_error():
    script = ScriptedProviders().add("openrouter", ok())
    env = {"OPENROUTER_API_KEY": FAKE_KEYS["OPENROUTER_API_KEY"]}
    client = make_client(script, providers=[google(env), openrouter(env)], env=env)
    res = client.chat(REQ)
    assert res.provider == "openrouter"
    assert res.attempts[0].outcome == "skipped_no_key"
    assert script.count("google") == 0


def test_no_keys_at_all_raises_all_failed():
    client = make_client(ScriptedProviders(), env={})
    with pytest.raises(AllProvidersFailed) as exc:
        client.chat(REQ)
    assert {a.outcome for a in exc.value.attempts} == {"skipped_no_key"}


def test_keys_are_read_at_call_time(monkeypatch):
    script = ScriptedProviders().add("openrouter", ok())
    from llm_toolkit.client import LLMClient

    client = LLMClient([openrouter({})], transport=script.transport)  # env=None -> os.environ
    with pytest.raises(AllProvidersFailed):
        client.chat(REQ)
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEYS["OPENROUTER_API_KEY"])
    assert client.chat(REQ).provider == "openrouter"
