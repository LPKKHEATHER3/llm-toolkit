"""Run the key-holding proxy.

    python -m llm_toolkit.proxy --port 8787
    python -m llm_toolkit.proxy --port 8787 --origin http://127.0.0.1:8787 --daily-cap 200

Keys are read from the environment (GEMINI_API_KEY, OPENROUTER_API_KEY, ANTHROPIC_API_KEY) and never
sent to the browser.
"""

from __future__ import annotations

import argparse
import logging
import sys

from ..client import LLMClient, UsageLog, default_providers
from .server import ProxyConfig, allowed, make_proxy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m llm_toolkit.proxy")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--origin", action="append", default=[], help="allowed Origin (repeatable)")
    parser.add_argument(
        "--allowed-host", action="append", default=[], help="allowed Host header (repeatable)"
    )
    parser.add_argument("--daily-cap", type=int, default=500)
    parser.add_argument("--deadline", type=float, default=30.0, help="total seconds per chat request")
    parser.add_argument("--usage-log", default="logs/usage.jsonl")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        logging.warning("binding to %s: anyone who can reach this address can spend your API key", args.host)

    config = ProxyConfig(
        host=args.host,
        port=args.port,
        allowed_origins=allowed(args.origin),
        allowed_hosts=allowed(args.allowed_host),
        daily_cap=args.daily_cap,
        deadline_s=args.deadline,
    )
    client = LLMClient(default_providers(), usage_log=UsageLog(args.usage_log))
    httpd = make_proxy(client, config)
    cfg = httpd.RequestHandlerClass.config  # type: ignore[attr-defined]
    print(
        f"proxy on http://{cfg.host}:{cfg.port}  origins={sorted(cfg.allowed_origins)}  hosts={sorted(cfg.allowed_hosts)}"
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
