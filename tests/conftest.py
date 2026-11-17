"""Test-wide guards: no API keys, no network.

Every test runs with the provider key variables removed and with sockets restricted to loopback, so
a test that accidentally reaches a real provider fails loudly instead of spending money.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from llm_toolkit.redact import KEY_ENV_VARS

_LOOPBACK = {"127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1"}


def _host_of(address: object) -> str:
    if isinstance(address, tuple) and address:
        return str(address[0])
    return str(address)


@pytest.fixture(autouse=True)
def _scrub_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (*KEY_ENV_VARS, "LLM_TOOLKIT_PROVIDERS", "INVENTORY_API_URL"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def check(address: object) -> None:
        if isinstance(address, str | bytes):  # AF_UNIX path
            return
        if _host_of(address) not in _LOOPBACK:
            raise RuntimeError(f"network access is blocked in tests: {address!r}")

    def connect(self: socket.socket, address: object) -> None:
        check(address)
        return real_connect(self, address)

    def connect_ex(self: socket.socket, address: object) -> int:
        check(address)
        return real_connect_ex(self, address)

    def getaddrinfo(host: object, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        if host not in (None, *_LOOPBACK, b"localhost", b"127.0.0.1"):
            raise RuntimeError(f"DNS lookup is blocked in tests: {host!r}")
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
