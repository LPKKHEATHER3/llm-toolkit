"""A small key-holding chat proxy on the standard library's HTTP server.

The API keys stay in this process's environment. Browsers talk to ``POST /api/chat``; the proxy
owns the system prompt and sends no tools, so a caller can only add user/assistant text.

Checks, in order, with real status codes (an error is never a 200):

=====================================  ====================================================
``Host`` not in the allowlist           421 (blocks DNS-rebinding: attacker domain -> 127.0.0.1)
``OPTIONS`` (no CORS is offered)        405
``/api/*`` without ``Origin``           403 (non-browser callers must not spend the key)
``Origin`` not in the allowlist         403
no / bad ``Content-Length``             411 / 400
body larger than the byte limit         413 (decided from the header, before reading)
not ``application/json``                415 (a cross-site text/plain POST needs no preflight)
bad JSON / shape / roles / extra keys   400
daily request cap reached               429 + Retry-After
upstream deadline exceeded              504
upstream failed / rejected the request  502
no provider configured / budget used    503
static path traversal, dotfiles, ...    400 / 404
=====================================  ====================================================
"""

from __future__ import annotations

import json
import os
import re
import socket
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..agent.loop import ChatClient
from ..client.types import (
    AllProvidersFailed,
    ChatRequest,
    DeadlineExceeded,
    LLMError,
    Message,
    ProviderRequestError,
    RequestBudgetExhausted,
)
from ..redact import get_logger

log = get_logger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
# After rejecting a request whose body was not read, discard at most this much of it (see
# ProxyHandler._lingering_close) before closing the connection.
LINGER_MAX_BYTES = 1 << 20
LINGER_TIMEOUT_S = 2.0
STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}
DEFAULT_SYSTEM_PROMPT = (
    "You are the help desk of Pinewick Supply, a fictional shop. Answer briefly and politely. "
    "If you do not know a store policy, say so instead of guessing."
)
_BAD_PERCENT = re.compile(r"%(?![0-9A-Fa-f]{2})")


@dataclass
class ProxyConfig:
    host: str = "127.0.0.1"
    port: int = 8787
    allowed_origins: frozenset[str] = frozenset()
    allowed_hosts: frozenset[str] = frozenset()
    max_body_bytes: int = 64 * 1024
    max_messages: int = 20
    max_message_chars: int = 4000
    daily_cap: int = 500
    deadline_s: float = 30.0
    socket_timeout_s: float = 15.0
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    static_dir: Path = field(default=STATIC_DIR)

    def with_port(self, port: int) -> ProxyConfig:
        """Fill the default allowlists for the actual port (useful when binding to port 0)."""
        origins = self.allowed_origins or frozenset({f"http://127.0.0.1:{port}", f"http://localhost:{port}"})
        hosts = self.allowed_hosts or frozenset({f"127.0.0.1:{port}", f"localhost:{port}"})
        return ProxyConfig(
            **{**self.__dict__, "port": port, "allowed_origins": origins, "allowed_hosts": hosts}
        )


class InMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatIn(BaseModel):
    """Only text turns. ``system``, ``tools``, ``model`` and any other key are rejected."""

    model_config = ConfigDict(extra="forbid")

    messages: list[InMessage] = Field(min_length=1)

    @model_validator(mode="after")
    def _last_is_user(self) -> ChatIn:
        if self.messages[-1].role != "user":
            raise ValueError("the last message must be from the user")
        return self


class DailyCounter:
    """Atomic check-and-increment under a lock: concurrent requests cannot all pass the check before
    any of them counts (an earlier prototype's count-then-insert race). Attempts are counted
    before the model is called, so failing requests are not free."""

    def __init__(self, cap: int, now: Any = lambda: datetime.now(UTC)):
        self.cap = cap
        self._now = now
        self._lock = threading.Lock()
        self._day = self._now().date()
        self._count = 0

    def try_acquire(self) -> bool:
        with self._lock:
            today = self._now().date()
            if today != self._day:
                self._day, self._count = today, 0
            if self._count >= self.cap:
                return False
            self._count += 1
            return True

    def seconds_until_reset(self) -> int:
        now = self._now()
        tomorrow = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
        return max(1, int((tomorrow - now).total_seconds()))


class HttpError(Exception):
    def __init__(self, status: int, message: str, headers: dict[str, str] | None = None):
        super().__init__(message)
        self.status, self.message, self.headers = status, message, headers or {}


class ProxyHandler(BaseHTTPRequestHandler):
    server_version = "llm-toolkit-proxy/0.1"
    sys_version = ""
    config: ProxyConfig
    client: ChatClient
    counter: DailyCounter

    def setup(self) -> None:
        self.timeout = self.config.socket_timeout_s  # slow clients cannot hold a thread forever
        super().setup()

    def log_message(self, fmt: str, *args: Any) -> None:
        log.info("%s %s", self.address_string(), fmt % args)

    # ---- responses ---------------------------------------------------------------------------

    def _send(self, status: int, body: bytes, ctype: str, headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, data: Any, headers: dict[str, str] | None = None) -> None:
        self._send(
            status,
            json.dumps(data, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            headers,
        )

    # ---- dispatch ------------------------------------------------------------------------------

    def _handle(self) -> None:
        self._body_read = False
        try:
            self._dispatch()
        finally:
            if self._has_unread_body():
                self._lingering_close()

    def _dispatch(self) -> None:
        try:
            self._check_host()
            if self.command == "OPTIONS":
                raise HttpError(405, "CORS is not offered by this proxy", {"Allow": "GET, POST"})
            path = urlsplit(self.path).path
            if path.startswith("/api/"):
                self._check_origin()
                if path == "/api/chat":
                    if self.command != "POST":
                        raise HttpError(405, "use POST", {"Allow": "POST"})
                    return self._chat()
                raise HttpError(404, "no such endpoint")
            if self.command not in ("GET", "HEAD"):
                raise HttpError(405, "method not allowed", {"Allow": "GET"})
            return self._static(path)
        except HttpError as e:
            headers = dict(e.headers)
            if self._has_unread_body():
                headers["Connection"] = "close"  # the unread body must not be parsed as a request
                self.close_connection = True
            self._json(e.status, {"error": e.message}, headers)
        except Exception:  # never leak a stack trace
            log.exception("unhandled error in proxy")
            self._json(500, {"error": "internal error"})

    do_GET = do_HEAD = do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _handle  # type: ignore[assignment]

    # ---- early rejections ------------------------------------------------------------------------

    def _has_unread_body(self) -> bool:
        if self._body_read:
            return False
        if self.headers.get("Transfer-Encoding"):
            return True
        return (self.headers.get("Content-Length") or "0").strip() != "0"

    def _lingering_close(self) -> None:
        """Close a connection whose request body was never read, without resetting it.

        When a request is rejected before its body is read (bad or oversized ``Content-Length``,
        wrong ``Host`` or ``Origin``), the body is still waiting in the socket. Closing a socket
        with unread input makes the OS send a TCP reset, which can reach the client before it has
        read the error response (on Windows it regularly does). So, after the response is written:
        stop sending (FIN), discard what the client already sent, bounded in bytes and time, and
        only then close. The connection is never reused, because the body's framing is unknown.
        """
        self.close_connection = True
        try:
            self.wfile.flush()
            self.connection.shutdown(socket.SHUT_WR)
            self.connection.settimeout(LINGER_TIMEOUT_S)
            remaining = LINGER_MAX_BYTES
            while remaining > 0:
                chunk = self.rfile.read1(min(64 * 1024, remaining))
                if not chunk:  # the client closed its side: nothing left to discard
                    break
                remaining -= len(chunk)
        except OSError:  # timeout or peer already gone: closing now is the best we can do
            pass

    # ---- checks --------------------------------------------------------------------------------

    def _check_host(self) -> None:
        host = (self.headers.get("Host") or "").strip().lower()
        if host not in self.config.allowed_hosts:
            raise HttpError(421, "host not allowed")

    def _check_origin(self) -> None:
        origin = self.headers.get("Origin")
        if not origin:
            raise HttpError(403, "Origin header required")
        if origin.strip().lower().rstrip("/") not in self.config.allowed_origins:
            raise HttpError(403, "origin not allowed")

    def _read_body(self) -> bytes:
        raw_len = self.headers.get("Content-Length")
        if raw_len is None:
            raise HttpError(411, "Content-Length required")
        try:
            n = int(raw_len)
        except ValueError:
            raise HttpError(400, "bad Content-Length") from None
        if n < 0:
            raise HttpError(400, "bad Content-Length")
        if n > self.config.max_body_bytes:
            raise HttpError(413, f"body larger than {self.config.max_body_bytes} bytes")
        body = self.rfile.read(n)
        self._body_read = True
        return body

    # ---- /api/chat -------------------------------------------------------------------------------

    def _chat(self) -> None:
        body = self._read_body()
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            raise HttpError(415, "Content-Type must be application/json")
        try:
            data = json.loads(body)
            req = ChatIn.model_validate(data)
        except (ValueError, ValidationError) as e:
            detail = (
                "; ".join(
                    f"{'.'.join(map(str, x['loc'])) or 'body'}: {x['msg']}"
                    for x in e.errors(include_url=False)
                )
                if isinstance(e, ValidationError)
                else "body is not valid JSON"
            )
            raise HttpError(400, detail) from None
        if len(req.messages) > self.config.max_messages:
            raise HttpError(400, f"at most {self.config.max_messages} messages")
        if any(len(m.content) > self.config.max_message_chars for m in req.messages):
            raise HttpError(400, f"each message is limited to {self.config.max_message_chars} characters")
        if not self.counter.try_acquire():
            raise HttpError(
                429, "daily request limit reached", {"Retry-After": str(self.counter.seconds_until_reset())}
            )

        messages = [Message.system(self.config.system_prompt)] + [
            Message(role=m.role, content=m.content) for m in req.messages
        ]
        try:
            resp = self.client.chat(ChatRequest(messages=messages, deadline_s=self.config.deadline_s))
        except DeadlineExceeded:
            raise HttpError(504, "the model did not answer in time") from None
        except RequestBudgetExhausted:
            raise HttpError(503, "request budget used up") from None
        except AllProvidersFailed as e:
            if e.attempts and all(a.outcome == "skipped_no_key" for a in e.attempts):
                raise HttpError(503, "no model provider is configured") from None
            raise HttpError(502, "all model providers failed") from None
        except ProviderRequestError as e:
            log.error("upstream rejected the request: %s", e)
            raise HttpError(502, "the model provider rejected the request") from None
        except LLMError:
            raise HttpError(502, "model call failed") from None
        self._json(
            200,
            {
                "message": resp.message.content or "",
                "provider": resp.provider,
                "usage": resp.usage.model_dump(),
            },
        )

    # ---- static ----------------------------------------------------------------------------------

    def _static(self, raw_path: str) -> None:
        if _BAD_PERCENT.search(raw_path):
            raise HttpError(400, "malformed percent-encoding")
        try:
            path = unquote(raw_path, errors="strict")
        except UnicodeDecodeError:
            raise HttpError(400, "path is not valid UTF-8") from None
        if "\x00" in path or "\\" in path:
            raise HttpError(400, "bad path")
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        root = self.config.static_dir.resolve()
        target = (root / rel).resolve()
        root_s, target_s = os.path.normcase(str(root)), os.path.normcase(str(target))
        if not target_s.startswith(root_s + os.sep):
            raise HttpError(404, "not found")
        parts = target.relative_to(root).parts
        ctype = STATIC_TYPES.get(target.suffix.lower())
        if any(p.startswith(".") for p in parts) or ctype is None or not target.is_file():
            raise HttpError(404, "not found")
        self._send(200, target.read_bytes(), ctype)


class ProxyServer(ThreadingHTTPServer):
    """ThreadingHTTPServer with a listen backlog sized for bursts.

    socketserver's default backlog is 5. Under a burst of simultaneous connections, Linux drops
    the overflow from the accept queue even though the client already sees the connection as open,
    so the client's request is answered with a reset instead of a status code.
    """

    request_queue_size = 128
    daemon_threads = True


def make_proxy(client: ChatClient, config: ProxyConfig | None = None) -> ThreadingHTTPServer:
    """Create (not start) the server. Binding to port 0 picks a free port; the allowlists default to
    that port on 127.0.0.1 and localhost."""
    config = config or ProxyConfig()
    httpd = ProxyServer((config.host, config.port), ProxyHandler)
    final = config.with_port(httpd.server_address[1])  # fills only the allowlists left empty
    handler = type(
        "BoundProxyHandler",
        (ProxyHandler,),
        {"config": final, "client": client, "counter": DailyCounter(final.daily_cap)},
    )
    httpd.RequestHandlerClass = handler
    return httpd


def allowed(values: Iterable[str]) -> frozenset[str]:
    return frozenset(v.strip().lower().rstrip("/") for v in values if v.strip())
