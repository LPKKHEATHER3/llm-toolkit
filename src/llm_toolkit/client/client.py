"""Multi-provider chat client: retry, cooldown, failover and a total deadline per request.

Failure classes for one attempt
-------------------------------
* **retryable** - 429, 5xx (incl. Anthropic 529), timeouts, transport errors: retried on the same
  provider with exponential backoff + full jitter (at least ``Retry-After``), then failed over.
* **quota** - a 429 with a long ``Retry-After`` or a quota body: provider goes into cooldown, the
  client fails over at once.
* **gateway** - a 2xx that is not a usable answer (non-JSON, no message, empty, or a gateway error
  disguised as content): counted as a failure, the client fails over. Never raised directly.
* **provider_error** - 401, 402, 403, 404 (bad key, no credit, no permission, unknown model on that
  provider): logged at ERROR, recorded, failed over.
* **request_error** - 400, 422 and any other 4xx: the request itself is wrong, so it is raised at
  once without retry or failover.

All waiting and every timeout is bounded by ``ChatRequest.deadline_s``.
"""

from __future__ import annotations

import os
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence

import httpx

from ..redact import get_logger, redact
from .adapters import ADAPTERS
from .cooldown import CooldownStore
from .gateway import looks_like_gateway_error
from .providers import ProviderSpec
from .retry import RetryPolicy, parse_retry_after
from .types import (
    AllProvidersFailed,
    Attempt,
    ChatRequest,
    ChatResponse,
    DeadlineExceeded,
    ProviderRequestError,
    RequestBudgetExhausted,
    ResponseFormatError,
)
from .usage import UsageLog

log = get_logger(__name__)

PROVIDER_SPECIFIC_STATUSES = frozenset({401, 402, 403, 404})
MIN_ATTEMPT_S = 0.5  # do not start an attempt with less budget than this


class LLMClient:
    def __init__(
        self,
        providers: Sequence[ProviderSpec],
        *,
        transport: httpx.BaseTransport | None = None,
        cooldowns: CooldownStore | None = None,
        usage_log: UsageLog | None = None,
        retry: RetryPolicy | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        env: Mapping[str, str] | None = None,
        max_requests: int | None = None,
        min_interval_s: float = 0.0,
    ):
        if not providers:
            raise ValueError("at least one provider is required")
        self.providers = list(providers)
        self._http = httpx.Client(transport=transport)
        # The cooldown store must share the client's clock, or cooldowns and deadlines disagree.
        self.cooldowns = cooldowns or CooldownStore(monotonic=clock)
        self.usage_log = usage_log or UsageLog(None)
        self.retry = retry or RetryPolicy()
        self._clock = clock
        self._sleep = sleep
        self._env = env  # None: read os.environ at call time, so keys are never cached
        self._warned_no_key: set[str] = set()
        self.request_count = 0  # HTTP requests actually sent
        self.max_requests = max_requests  # hard cap over the client's lifetime (None: unlimited)
        self._count_lock = threading.Lock()
        self.min_interval_s = min_interval_s  # minimum gap between HTTP requests (0: no pacing)
        self._last_sent = float("-inf")

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- helpers -----------------------------------------------------------------------------

    def _key(self, spec: ProviderSpec) -> str:
        env = os.environ if self._env is None else self._env
        return env.get(spec.key_env, "")

    def _quota_cooldown(self, spec: ProviderSpec, retry_after: float | None, body: str) -> float | None:
        if retry_after is not None:
            return retry_after if retry_after > spec.cooldown_threshold_s else None
        for pattern, seconds in spec.quota_patterns:
            if re.search(pattern, body):
                return seconds
        return None

    def _timeout(self, spec: ProviderSpec, remaining: float) -> httpx.Timeout:
        return httpx.Timeout(
            connect=min(spec.connect_timeout, remaining),
            read=min(spec.read_timeout, remaining),
            write=min(spec.write_timeout, remaining),
            pool=min(spec.connect_timeout, remaining),
        )

    # ---- main entry point ----------------------------------------------------------------------

    def chat(self, req: ChatRequest) -> ChatResponse:
        start = self._clock()
        deadline = start + req.deadline_s
        attempts: list[Attempt] = []

        def remaining() -> float:
            return deadline - self._clock()

        for spec in self.providers:
            key = self._key(spec)
            if not key:
                attempts.append(Attempt(provider=spec.name, model=spec.model, outcome="skipped_no_key"))
                if spec.name not in self._warned_no_key:
                    self._warned_no_key.add(spec.name)
                    log.info("skipping provider %s: %s is not set", spec.name, spec.key_env)
                continue
            if self.cooldowns.active(spec.name):
                attempts.append(
                    Attempt(
                        provider=spec.name,
                        model=spec.model,
                        outcome="skipped_cooldown",
                        detail=f"{self.cooldowns.remaining(spec.name):.0f}s left",
                    )
                )
                continue

            adapter = ADAPTERS[spec.kind]
            url, headers, body = adapter.build(spec, req, key)
            tries = 1 + spec.max_retries
            for attempt_no in range(tries):
                if remaining() < MIN_ATTEMPT_S:
                    raise DeadlineExceeded(f"deadline of {req.deadline_s:.1f}s exceeded", attempts)
                t0 = self._clock()
                retry_after: float | None = None

                def record(
                    outcome: str,
                    status: int | None = None,
                    detail: str = "",
                    _t0: float = t0,
                    _spec: ProviderSpec = spec,
                    _key: str = key,
                ) -> Attempt:
                    a = Attempt(
                        provider=_spec.name,
                        model=_spec.model,
                        outcome=outcome,  # type: ignore[arg-type]
                        status=status,
                        detail=redact(detail, extra_secrets=[_key])[:300],
                        elapsed_s=round(self._clock() - _t0, 3),
                    )
                    attempts.append(a)
                    return a

                with self._count_lock:
                    if self.max_requests is not None and self.request_count >= self.max_requests:
                        raise RequestBudgetExhausted(
                            f"request budget of {self.max_requests} HTTP requests is used up", attempts
                        )
                    # Client-side pacing (free tiers allow a few requests per minute). Never wait
                    # past the deadline: if the gap does not fit, stop with DeadlineExceeded.
                    gap = self._last_sent + self.min_interval_s - self._clock()
                    if gap > 0:
                        if gap >= remaining() - MIN_ATTEMPT_S:
                            raise DeadlineExceeded("pacing wait would cross the deadline", attempts)
                        self._sleep(gap)
                    self._last_sent = self._clock()
                    self.request_count += 1
                try:
                    resp = self._http.post(
                        url, headers=headers, json=body, timeout=self._timeout(spec, remaining())
                    )
                except httpx.TimeoutException as e:
                    current = record("retryable", detail=f"timeout ({type(e).__name__})")
                except httpx.TransportError as e:
                    current = record("retryable", detail=f"transport error ({type(e).__name__}: {e})")
                else:
                    status = resp.status_code
                    text = resp.text
                    if 200 <= status < 300:
                        try:
                            msg, usage = adapter.parse(spec, resp.json())
                        except (ValueError, ResponseFormatError) as e:
                            record("gateway", status, f"unusable 2xx body: {e}")
                            log.warning("provider %s returned an unusable 2xx body; failing over", spec.name)
                            break
                        if looks_like_gateway_error(msg.content):
                            record("gateway", status, f"gateway error disguised as content: {msg.content!r}")
                            log.warning(
                                "provider %s returned a gateway error as content; failing over", spec.name
                            )
                            break
                        if not (msg.content or "").strip() and not msg.tool_calls:
                            record("gateway", status, "empty answer (no text and no tool calls)")
                            log.warning("provider %s returned an empty answer; failing over", spec.name)
                            break
                        record("ok", status)
                        latency_ms = (self._clock() - start) * 1000
                        self.usage_log.record(
                            provider=spec.name,
                            model=spec.model,
                            prompt_tokens=usage.prompt_tokens,
                            completion_tokens=usage.completion_tokens,
                            cost_usd=usage.cost_usd,
                            latency_ms=latency_ms,
                        )
                        return ChatResponse(
                            message=msg,
                            usage=usage,
                            provider=spec.name,
                            model=spec.model,
                            latency_ms=latency_ms,
                            attempts=attempts,
                        )
                    retry_after = parse_retry_after(resp.headers.get("retry-after"))
                    if status == 429:
                        cool = self._quota_cooldown(spec, retry_after, text)
                        if cool is not None:
                            self.cooldowns.set(spec.name, cool)
                            record("retryable", status, f"quota exhausted, cooldown {cool:.0f}s")
                            log.warning("provider %s quota exhausted; cooling down %.0fs", spec.name, cool)
                            break
                        current = record("retryable", status, text[:200])
                    elif status >= 500:
                        current = record("retryable", status, text[:200])
                    elif status in PROVIDER_SPECIFIC_STATUSES:
                        record("provider_error", status, text[:200])
                        log.error(
                            "provider %s (model %s) returned HTTP %d; failing over. Body: %s",
                            spec.name,
                            spec.model,
                            status,
                            redact(text[:200], extra_secrets=[key]),
                        )
                        break
                    else:
                        record("request_error", status, text[:200])
                        raise ProviderRequestError(
                            f"{spec.name} rejected the request with HTTP {status}: "
                            f"{redact(text[:200], extra_secrets=[key])}",
                            status=status,
                            provider=spec.name,
                            attempts=attempts,
                        )

                # Only retryable failures reach this point.
                if attempt_no == tries - 1:
                    break
                wait = self.retry.backoff(attempt_no, retry_after)
                if wait >= remaining() - MIN_ATTEMPT_S:
                    current.detail = (current.detail + " | no time left to wait, failing over").strip(" |")
                    break
                current.waited_s = round(wait, 3)
                log.info("retrying %s in %.2fs (attempt %d/%d)", spec.name, wait, attempt_no + 2, tries)
                self._sleep(wait)

        if remaining() <= 0:
            raise DeadlineExceeded(f"deadline of {req.deadline_s:.1f}s exceeded", attempts)
        summary = ", ".join(
            f"{a.provider}:{a.outcome}{'/' + str(a.status) if a.status else ''}" for a in attempts
        )
        raise AllProvidersFailed(f"all providers failed ({summary or 'no providers'})", attempts)
