# llm-toolkit design notes

A small, dependency-light Python toolkit for building LLM features that keep working when providers
misbehave: a multi-provider client with retry, failover and deadlines; a tool-calling agent loop with
strict argument validation; a pure-Python BM25 RAG pipeline with a quote verifier; an append-only
evaluation harness; an MCP server over the `inventory-manager` REST API; a hardened key-holding proxy;
and a Claude Skill.

It is a Python rewrite of ideas from two earlier personal prototypes (prototype A: a JavaScript
single-page tool-calling assistant with a Node.js key-holding proxy and a benchmark script; prototype B:
a TypeScript chat feature with provider fallback). **No file is copied from them**; only the ideas and
the lessons from their weak points are reused. Numbers quoted about the prototypes were measured in a
private prototype and have no result file in this repository. All sample data is fictional.

---

## 1. Ground rules

| Area | Rule |
|---|---|
| Python | 3.11+ (CI: 3.11 and 3.12, ubuntu-latest and windows-latest) |
| Runtime deps | `httpx`, `pydantic` (v2), `mcp` (official Python SDK, 2.x). Nothing else |
| Dev deps | `pytest`, `hypothesis`, `ruff` |
| Providers | OpenRouter, Google AI Studio (OpenAI-compatible endpoint), Anthropic Messages API, each on its official endpoint |
| Secrets | Read from the environment only, never printed. Keys, `Authorization` values and JWTs are redacted before any log write |
| Tests | Offline, no keys: `httpx.MockTransport` or scripted fakes. A conftest guard blocks non-loopback sockets and DNS and unsets the key variables |
| Data | All fictional (a made-up store's policies and products) |
| Numbers | Every result number in the README comes from a run in this repo, with the raw result file committed |
| Language | Code comments, errors and docs in English |

---

## 2. Layout

```
llm-toolkit/
  pyproject.toml              # setuptools, deps, ruff + pytest config
  README.md  LICENSE  .gitignore
  docs/design.md              # this file
  .github/workflows/ci.yml
  src/llm_toolkit/
    __init__.py
    redact.py                 # redact(), RedactingFilter, get_logger(): used by every logger and the usage log
    client/
      __init__.py             # LLMClient, ProviderSpec, ChatRequest, ChatResponse, errors
      types.py                # internal message model (OpenAI-shaped): Message, ToolCall, Usage, Attempt, errors
      providers.py            # ProviderSpec, built-in specs for openrouter / google / anthropic, default models
      adapters.py             # OpenAI-compatible and Anthropic adapters (wire format <-> internal model)
      retry.py                # RetryPolicy: exponential backoff + full jitter, Retry-After parsing
      cooldown.py             # CooldownStore (in memory, optional JSON file), monotonic + wall clock
      gateway.py              # the ONE definition of "error disguised as model output"
      usage.py                # UsageLog: thread-safe JSONL writer + prompt-token alarm
      client.py               # LLMClient.chat(): ordering, cooldown, failover, total deadline, request budget
    tools/
      __init__.py             # build_default_registry()
      registry.py             # Tool, ToolArgs, ToolRegistry, ToolResult: the single tool definition point
      inventory.py            # InventoryAPI (httpx) + list_items / get_item / issue_stock / undo_last_change
      fake_inventory.py       # FakeInventoryBackend: in-memory inventory behind an httpx.MockTransport
      policies.py             # search_policies (wraps the RAG retriever)
    agent/
      __init__.py
      loop.py                 # Agent.run(): bounded tool-calling loop
      history.py              # trim_history(): token-budgeted, never splits tool_call/tool pairs
    rag/
      __init__.py             # default_index(), search()
      chunking.py             # heading-aware chunker with sentence overlap, stable chunk ids
      bm25.py                 # BM25Index (pure Python), Retriever protocol, EmbeddingRetriever
      answer.py               # answer_with_citations(): prompt, parse the JSON answer, verify
      verify.py               # verify_citations(): verbatim quote check, drop + count
    eval/
      __init__.py
      cases.py                # pydantic Case model, loader for JSON case files
      graders.py              # code graders, registered by type
      runner.py               # run(): repeats, request budget, outcome classes, append-only JSONL
      planted.py              # planted-defect dataset builder and verifier benchmark
      report.py               # per-model accuracy, p50/p95 latency, tokens, cost -> Markdown
      __main__.py             # python -m llm_toolkit.eval {check,run,report,verifier-bench,build-planted}
    mcp_server/
      __init__.py
      server.py               # MCP server on the SDK's low-level mcp.server.Server
      __main__.py             # python -m llm_toolkit.mcp_server [--fake-backend] (stdio)
    proxy/
      __init__.py
      server.py               # ThreadingHTTPServer proxy: Host/Origin checks, deadline, static files, daily cap
      static/index.html       # tiny demo page (the only thing served statically)
      __main__.py
  data/
    policies/*.md             # fictional store policies (RAG corpus)
    eval/cases/*.json         # eval cases
    eval/planted_citations.json  # answers with deliberately wrong quotes (verifier benchmark)
    prices.json               # USD per 1M tokens, used only when a provider reports no cost
  results/
    eval/*.jsonl              # append-only raw results (committed)
    eval/REPORT.md            # generated from the JSONL, never hand-written
  skills/inventory-stock/SKILL.md
  examples/
    policy_agent.py           # RAG Q&A over the policies with verified citations
    stock_agent.py            # agent that manages stock through the shared tools
    mcp_stdio_demo.py         # starts the MCP server over stdio and calls its tools
  tests/
    conftest.py               # network guard, key-env scrubbing
    fakes.py                  # scripted providers, fake clock, mock inventory backend
    scripted_llm.py           # a scripted chat client for agent and proxy tests
    test_client_*.py  test_agent_*.py  test_rag.py  test_eval.py
    test_mcp_server.py  test_proxy.py  test_tools_registry.py  test_examples_and_skill.py
```

`tools/` exists so that the agent, the eval harness and the MCP server all import the **same** tool
objects (see §4).

---

## 3. Module interfaces

### 3.1 `client/`

```python
@dataclass(frozen=True)
class ProviderSpec:
    name: str  # "openrouter" | "google" | "anthropic"
    kind: Literal["openai", "anthropic"]
    base_url: str  # https://openrouter.ai/api/v1
    # https://generativelanguage.googleapis.com/v1beta/openai
    # https://api.anthropic.com/v1
    model: str
    key_env: str  # OPENROUTER_API_KEY | GEMINI_API_KEY | ANTHROPIC_API_KEY
    connect_timeout: float = 5.0
    read_timeout: float = 30.0
    write_timeout: float = 10.0
    max_retries: int = 2  # 0 is valid and means "one attempt" (no `or` defaulting)
    cooldown_threshold_s: float = 30.0
    quota_patterns: tuple[tuple[str, float], ...] = ()  # (regex on the 429 body, cooldown seconds)
    extra_body: Mapping[str, Any] = {}
    extra_headers: Mapping[str, str] = {}


class ChatRequest(BaseModel):
    messages: list[Message]  # internal OpenAI-shaped model, system message allowed first
    tools: list[dict] | None = None  # OpenAI function schemas, produced by ToolRegistry
    temperature: float | None = 0.0
    max_tokens: int = 2048
    deadline_s: float = 60.0  # TOTAL budget across retries, backoff and failover


class ChatResponse(BaseModel):
    message: Message  # role="assistant", content: str | None, tool_calls: list[ToolCall]
    usage: Usage  # prompt_tokens, completion_tokens, cost_usd (if reported)
    provider: str
    model: str
    latency_ms: float
    attempts: list[Attempt]  # every try: provider, model, outcome, status, detail, waited_s, elapsed_s


class LLMClient:
    def __init__(
        self,
        providers: Sequence[ProviderSpec],
        *,
        transport: httpx.BaseTransport | None = None,
        cooldowns: CooldownStore | None = None,
        usage_log: UsageLog | None = None,
        retry: RetryPolicy | None = None,
        clock=time.monotonic,
        sleep=time.sleep,
        env: Mapping[str, str] | None = None,
        max_requests: int | None = None,  # hard cap on HTTP requests, retries included
        min_interval_s: float = 0.0,  # spacing between HTTP requests
    ): ...
    def chat(self, req: ChatRequest) -> ChatResponse: ...  # raises LLMError subclasses
```

Behaviour of `chat()`:

1. Walk providers in order. Skip a provider whose key variable is empty (logged once, not a failure) or
   that is in cooldown.
2. Per attempt, `httpx.Timeout(connect=c, read=min(r, remaining), write=w, pool=...)`, where `remaining`
   is what is left of the request deadline. If the remaining budget is smaller than a useful attempt,
   stop with `DeadlineExceeded`.
3. Classification of one attempt:
   - **retryable**: 429, 5xx (including Anthropic 529), `httpx.TimeoutException`, `httpx.TransportError`.
   - **gateway failure** (counts as a failed attempt and moves to the next provider, never raised
     straight away): non-JSON 200 body, 200 without a message, empty answer (no text and no tool calls),
     or content matching `gateway.looks_like_gateway_error()` (a short error string such as
     "upstream error" or "bad gateway" returned as if it were the answer).
   - **request error**: 400, 422 and any other 4xx not listed below (the request itself is wrong). No
     retry and **no failover**: the next provider would reject the same request, so failing over only
     spends money and hides the bug. Raised as `ProviderRequestError(status, provider, body_excerpt_redacted)`.
   - **provider-specific error**: 401, 402, 403, 404 (bad key, no credit, no permission, unknown model id
     on *that* provider). No retry on the same provider; logged at ERROR level with status and provider,
     recorded in the attempt list, and the client **fails over** to the next provider.
4. Backoff: `sleep = min(cap, base * 2**attempt) * random()` (full jitter). If the response has
   `Retry-After` (seconds or HTTP date), wait `max(jittered, retry_after)`. A wait that would cross the
   deadline is not taken; the client moves on (or fails) instead.
5. Quota 429: if `Retry-After` is above the provider's `cooldown_threshold_s`, or the body matches one of
   the provider's `quota_patterns`, the provider is put in cooldown for `retry_after` (or the pattern's
   default, e.g. 60 s / 1 h for per-minute / per-day quota) and the client fails over immediately.
6. When every provider fails: `AllProvidersFailed(attempts)`; when the deadline is hit:
   `DeadlineExceeded(attempts)`; when `max_requests` is used up: `RequestBudgetExhausted`. Each carries
   enough for callers (eval, proxy) to classify it as an infrastructure error.
7. On success: content parts (a list of `{type: "text"}`) are joined into a string; usage is appended to
   the `UsageLog`; if `prompt_tokens` exceeds the alarm threshold (default 8000) a warning is logged.

`UsageLog(path)`: one JSON object per line (`ts, provider, model, prompt_tokens, completion_tokens,
cost_usd, latency_ms`), written under a `threading.Lock` with a single `write()` call per line,
`ensure_ascii=False`, values passed through `redact()`.

`redact(text)`: removes (a) the current values of the known key variables, (b) `Bearer <token>` and
`x-api-key: <token>`, (c) `sk-...` / `sk-ant-...` / `AIza...`-shaped strings, (d) JWTs.
`RedactingFilter` is a `logging.Filter` installed on the `llm_toolkit` logger.

`CooldownStore`: `dict[str, float]` guarded by a lock, using `time.monotonic()` for in-process checks. The
optional JSON-file backend stores wall-clock expiry so a cooldown survives a restart.

The Anthropic adapter converts the internal model: `system` becomes the top-level `system` field;
assistant `tool_calls` become `tool_use` blocks; consecutive `tool` messages become one user message with
`tool_result` blocks; `input_tokens` / `output_tokens` map to prompt / completion tokens. Headers:
`x-api-key`, `anthropic-version: 2026-10-21` (`ANTHROPIC_VERSION` in `client/adapters.py`). Provider-specific fields on a tool call (for example
Gemini's thought signature) are replayed only to the provider that produced them.

### 3.2 `tools/`: one source of tool definitions

```python
class ToolArgs(BaseModel): ...  # base for argument models: extra="forbid", strict types

@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[ToolArgs]
    handler: Callable[[Any], Any]          # returns JSON-serialisable data
    def openai_schema(self) -> dict
    def anthropic_schema(self) -> dict
    def mcp_input_schema(self) -> dict

class ToolResult(BaseModel):
    ok: bool
    data: Any | None = None
    error: str | None = None               # human/model-readable

class ToolRegistry:
    def register(self, tool: Tool) -> None
    def schemas(self, fmt: Literal["openai", "anthropic"] = "openai") -> list[dict]
    def schema_hash(self) -> str           # stored with every eval result
    def call(self, name: str, raw_arguments: str | dict | None) -> ToolResult
```

`call()` never raises for model mistakes:

- unknown tool → `ToolResult(ok=False, error="unknown tool 'x'; available: ...")`
- invalid JSON → the decoder's message and position
- schema mismatch → one `field: message` line per pydantic error (missing, wrong type, unknown field).
  Nothing is replaced with `{}` and **no default is filled in for a field the model did not send**:
  required fields have no defaults, optional fields must default to `None` (checked when a `Tool` is
  created), and handlers receive only the fields the model actually set.
- handler raises `Exception` → `error="<ExceptionType>: <message>"`. `BaseException`
  (`KeyboardInterrupt`, `SystemExit`) is not caught.

`build_default_registry(inventory: InventoryAPI, retriever: Retriever | None = None) -> ToolRegistry` is
the only factory. Agent, eval and MCP server all call it.

### 3.3 `agent/`

```python
class Agent:
    def __init__(self, client: ChatClient, registry: ToolRegistry, *, system_prompt: str,
                 max_rounds: int = 5, history_token_budget: int = 6000,
                 temperature: float | None = 0.0, max_tokens: int = 2048, deadline_s: float = 60.0): ...
    def run(self, history: Sequence[Message], user_text: str) -> AgentResult

class AgentResult(BaseModel):
    messages: list[Message]            # full updated history, always API-valid
    final_text: str
    stop_reason: Literal["answer", "max_rounds", "llm_error"]
    rounds: int
    tool_calls: list[ToolCallRecord]   # round, name, raw arguments, ok, error, data (used by graders)
    usage: list[Usage]
    latency_ms: list[float]
    providers: list[str]
```

- Each round sends `[system, *trim_history(history, budget)]` with the registry's schemas.
- Tool results are appended as `{"role": "tool", "tool_call_id": id, "content": json(ToolResult)}`.
- When `max_rounds` is reached, a closing assistant message is appended ("stopped after N tool rounds"),
  so the history never ends with dangling tool results and the next turn is valid.

`trim_history(messages, budget_tokens, count_tokens=approx_tokens) -> list[Message]` groups messages into
atomic units (a `user` message, a plain `assistant` message, or an `assistant` with `tool_calls` **plus
all of its `tool` replies**), keeps the newest units that fit the token budget, always keeps the newest
user message, and makes the window start at a `user` message. It never returns a `tool` message whose
parent was cut, nor an assistant `tool_calls` message missing any of its replies.

### 3.4 `rag/`

```python
@dataclass(frozen=True)
class Chunk:  id: str; doc_id: str; section: str; text: str; start: int; end: int  # offsets into the doc

def chunk_document(doc_id: str, text: str, *, max_chars: int = 600, overlap: int = 1) -> list[Chunk]

class Retriever(Protocol):
    def search(self, query: str, k: int = 4) -> list[Hit]          # Hit: chunk, score

class BM25Index:            # k1=1.5, b=0.75, lowercase word tokenizer, small stopword list, tiny stemmer
    @classmethod
    def build(cls, chunks: Iterable[Chunk]) -> "BM25Index"

class EmbeddingRetriever:   # takes an `embed(texts) -> list[vector]` callable; cosine similarity

def answer_with_citations(client, retriever: Retriever, question: str, *, k: int = 4,
                          corpus: Mapping[str, Chunk] | None = None,
                          deadline_s: float = 60.0, max_tokens: int = 1024) -> CitedAnswer
def verify_citations(answer: RawAnswer, hits: Sequence[Hit], corpus=None, *,
                     min_quote_chars: int = 8) -> Verification
```

A heading always starts a new chunk and becomes the chunk's `section`; `overlap` sentences are repeated
at the start of the next chunk.

The model is asked for JSON: `{"answer": str, "citations": [{"chunk_id": str, "quote": str}]}`.
`verify_citations` keeps a citation only if its trimmed `quote` is at least `min_quote_chars` long and
appears **verbatim** (exact substring, no normalisation) in the text of the chunk it names, and that chunk
was among the retrieved hits. Dropped citations carry a reason (`empty`, `too_short`, `unknown_chunk`,
`not_retrieved`, `quote_not_found`) and are counted (`dropped_count`). An answer whose citations are all
dropped is flagged `grounded=False`.

### 3.5 `eval/`

Case file (JSON, one list per file; unknown fields are rejected):

```json
{
  "id": "issue-stock-by-sku",
  "suite": "agent",
  "critical": true,
  "input": "We just sold 2 packs of binder clips, SKU OSU-0001. Please record the sale.",
  "fixture": "inventory_default",
  "graders": [
    {"type": "tool_called", "name": "issue_stock", "args": {"sku": "OSU-0001", "quantity": 2}},
    {"type": "stock_equals", "sku": "OSU-0001", "qty": 38},
    {"type": "answer_contains_any", "values": ["38"]}
  ]
}
```

- Graders are plain functions registered by `type`: `tool_called`, `tool_not_called`, `stock_equals`,
  `no_movements`, `answer_contains_any`, `answer_mentions_all`, `answer_matches`, `grounded`,
  `cites_doc`, `abstains`. No grader matches on model wording when a structural check is possible.
- Outcome per line: `pass`, `fail` (model behaviour), `infra_error` (429, 5xx, timeouts, deadline, every
  provider down), `error` (our request was rejected, e.g. HTTP 400) and `not_run` (the request budget ran
  out first). Accuracy is computed over `pass + fail` only; the others are reported separately.
- The runner uses the real `Agent` with the same `max_rounds` and the registry from
  `build_default_registry()` against a fresh fake inventory per case.
- Results are appended to `results/eval/<suite>.jsonl` (open mode `"a"`, never `"w"`), one line per
  `(run_id, model, case_id, repeat)` with `git_sha`, `prompt_hash`, `tool_schema_hash`, `temperature`,
  latency, tokens, cost, outcome and grader notes.
- `--max-requests` (default and hard maximum **30**) is enforced by `LLMClient` in HTTP requests,
  retries included; the run stops cleanly and records the remaining cases as `not_run`.
- Live runs need a key in the environment; otherwise the command prints `skipped: no provider key in the
  environment` and exits 0 without writing.
- `report` reads all JSONL lines and writes `results/eval/REPORT.md`: per-model accuracy (pass / graded),
  critical pass, infra errors, p50/p95 latency, mean prompt/completion tokens, total cost (provider-reported
  cost when present, otherwise `data/prices.json`; "n/a" if neither).
- `verifier-bench` (offline, no LLM) runs the verifier on `data/eval/planted_citations.json`, built by
  `build-planted` from the fictional corpus. Each citation is labelled `genuine` or carries one planted
  defect: paraphrase, changed number, flipped negation, changed case, doubled space, fabricated sentence,
  wrong chunk, chunk not retrieved, unknown chunk, empty, too short. It reports how many defects were
  dropped, how many genuine quotes were kept and whether each drop had the expected reason, and appends
  to `results/eval/verifier.jsonl` (`--dry-run` writes nothing).

### 3.6 `mcp_server/`

Built on the official SDK's low-level `mcp.server.Server` (not `FastMCP`), so the advertised input schemas
are exactly the registry's pydantic-generated schemas and argument validation happens in one place (the
registry). `list_tools` returns the registry's tools; `call_tool` runs `registry.call` in a worker thread
(`anyio.to_thread.run_sync`). A `ToolResult(ok=False)` becomes a `CallToolResult` with `is_error=True` and
the error text as content, so the model reads the backend's reason instead of a protocol error.

Transport: stdio (`python -m llm_toolkit.mcp_server`). Backend URL from `INVENTORY_API_URL` (default
`http://127.0.0.1:8000`); `--fake-backend` serves the in-memory fictional inventory instead.

### 3.7 `proxy/`

Standard-library `ThreadingHTTPServer` (no extra dependency), bound to `127.0.0.1:8787` by default.

| Request | Check | Response |
|---|---|---|
| any | `Host` header not in `allowed_hosts` (default `127.0.0.1:<port>`, `localhost:<port>`) | **421** |
| `OPTIONS` | no CORS is offered | **405** with `Allow` |
| `/api/*` | `Origin` missing | **403** |
| `/api/*` | `Origin` not in `allowed_origins` (exact match, scheme+host+port) | **403** |
| `POST /api/chat` | `Content-Length` missing / invalid | **411 / 400** |
| | body larger than `max_body_bytes` (bytes, decided from the header before reading) | **413** |
| | not `application/json` | **415** |
| | not JSON / wrong shape / roles other than `user`/`assistant` / a `tools` or `system` field / too many or too long messages | **400** |
| | daily request cap reached (atomic counter under a lock) | **429** with `Retry-After` |
| | upstream rejected the request, or every provider failed | **502** with a generic message (detail only in the redacted log) |
| | no provider configured, or request budget used up | **503** |
| | request deadline exceeded | **504** |
| | success | **200** `{message, usage, provider}` |
| `GET` static | only files under `proxy/static/`, extension allowlist, no dotfiles; the resolved path must stay under the root (compared with `os.path.normcase` on Windows); malformed `%`-encoding detected by hand | **400 / 404** |
| other method | | **405** |

The server owns the system prompt and sends no tool list; clients can only send `user`/`assistant` text
messages (bounded count and length). Keys live in the server's environment and never leave it.

When a request is rejected before its body is read (bad or oversized `Content-Length`, wrong `Host` or
`Origin`), the response carries `Connection: close` and the server does a *lingering close*: it sends the
response, half-closes its side, discards what the client already sent (at most 1 MiB, at most 2 s) and
only then closes. Closing a socket with unread input makes the OS send a TCP reset, which can reach the
client before the error response does (`test_rejected_requests_with_unread_body_always_get_their_response`).

### 3.8 `skills/inventory-stock/SKILL.md`

A Claude Skill (YAML frontmatter `name`, `description`) that teaches: look up before acting
(`list_items` → `get_item`), confirm SKU and quantity with the user before `issue_stock`, how to read a
409 ("only N in stock") and what to offer instead, when and how to use `undo_last_change` with the
expected label, and to cite `search_policies` results for policy questions. A test checks the frontmatter
and that every tool name it mentions exists in the registry.

### 3.9 `examples/`

- `policy_agent.py "Can I return an opened item?"`: RAG answer with verified citations.
- `stock_agent.py`: agent loop over the same tools, against a running `inventory-manager`.
- `mcp_stdio_demo.py`: starts the MCP server over stdio with the fake backend and calls its tools.
- `policy_agent.py` and `stock_agent.py` accept `--offline` (scripted model + fake backend), so every
  example runs without keys.

---

## 4. Inventory-manager tools exposed through MCP

The backend (inventory-manager, a separate standard-library REST API repository) returns JSON errors with
real status codes: 400 validation (`details` per field), 404, 405, 409 conflict, 413.

| MCP tool | Arguments (required unless marked) | Backend call | Notes |
|---|---|---|---|
| `list_items` | `query: str` (optional, ≤80), `category: str` (optional), `low_stock_only: bool` (optional) | `GET /api/items?q=&category=&low=1` | Only fields the model sent become query params |
| `get_item` | `sku: str` matching `^[A-Z0-9]{1,8}-\d{1,8}$` | `GET /api/items/<sku>` | The SKU pattern also prevents path injection |
| `issue_stock` | `sku`, `quantity: int` (1 to 100,000), `reason: str` (1-120) | `POST /api/items/<sku>/move` with `delta = -quantity` | **Not retried** on timeout/5xx (not idempotent); a timeout returns "outcome unknown, check get_item before retrying". 409 text ("only 4 in stock, cannot issue 50") is returned verbatim as an error result |
| `undo_last_change` | `expected_label: str` (e.g. `-3 ELE-0007`) | `POST /api/undo` with `{"expected": label}` | The backend's undo stack is global. inventory-manager compares `expected` with the top of the stack under the same lock that performs the undo (409 with `current` on mismatch), so the model cannot undo someone else's change and there is no check-then-act race |
| `search_policies` | `query: str` (2-200), `k: int` 1-8 (optional) | BM25 index over `data/policies/` | Returns chunk ids and text so the model can quote verbatim |

GET calls are retried once on a connection error or 5xx; POST calls never are.

Not exposed on purpose: create / edit / delete / restore, CSV import / export and label cards. They are
either destructive in bulk or not useful to a chat agent.

`issue_stock` output includes the undo label the backend will use (`"-<qty> <sku>"`), so a follow-up
`undo_last_change` can pass it.

---

## 5. Weak points in the prototypes → fix → proving test

"A" = prototype A (Node.js proxy, browser agent loop, benchmark script); "B" = prototype B (TypeScript
provider fallback and chat action). Figures about A and B were measured in a private prototype.

### 5.1 LLM client

| # | Weak point in the prototype | Fix here | Test |
|---|---|---|---|
| C1 | B: no retry/backoff; A: linear 1.5 s × n waits | Exponential backoff with full jitter, bounded by a cap | `test_backoff_is_exponential_with_full_jitter` |
| C2 | Neither honours `Retry-After` | Wait at least `Retry-After` (seconds or HTTP date) | `test_retry_after_seconds_and_http_date_are_honoured` |
| C3 | B: every 4xx fails over silently (hides config errors, spends money on requests that cannot succeed) | 400/422: raised at once, no retry, no failover. 401/402/403/404: no retry, ERROR log, recorded attempt, failover; `AllProvidersFailed` if none succeeds | `test_bad_request_400_and_422_raise_without_failover`, `test_auth_and_model_errors_fail_over_with_error_log`, `test_provider_errors_on_every_provider_raise_all_failed` |
| C4 | B: no cooldown, every request hits the dead primary | Quota 429 → cooldown, later requests skip the provider | `test_quota_429_cools_down_provider_for_next_request` |
| C5 | A: quota detection relies on Google-only text (`PerDay`); cooldown lost on restart | `Retry-After` first, per-provider patterns second; optional file-backed store | `test_cooldown_from_retry_after_without_body_text`, `test_cooldown_survives_restart_with_file_store` |
| C6 | A: ~225 s worst case (measured in a private prototype); B: providers × 60 s | Total deadline per request across retries, backoff and failover; read timeout shrinks to the remaining budget | `test_total_deadline_caps_retries_and_failover`, `test_read_timeout_never_exceeds_remaining_deadline` |
| C7 | Single `timeout` value | Separate connect / read / write / pool timeouts | `test_connect_and_read_timeouts_are_separate` |
| C8 | A: an error disguised as content raises instead of failing over | Counted as a failed attempt → next provider | `test_disguised_gateway_error_fails_over` |
| C9 | A: the same check duplicated client/server with different patterns | One definition in `client/gateway.py` | `test_gateway_error_patterns_have_one_definition` (scans the package source) |
| C10 | B: a content array becomes `"[object Object]"` | Text parts joined; non-text parts ignored | `test_content_parts_are_joined_as_text` |
| C11 | B: no usage accounting | Thread-safe JSONL usage log | `test_usage_log_is_valid_jsonl_under_concurrent_writes` |
| C12 | A: token alarm only a console line | Alarm via logger with provider/model | `test_prompt_token_alarm_logs_warning` |
| C13 | A/B: a non-JSON 200 is a confusing error | Treated as a gateway failure → failover | `test_non_json_200_body_fails_over` |
| C14 | Port hazard: `retries or 2` turns 0 into 2 | `max_retries=0` means one attempt | `test_zero_retries_means_single_attempt` |
| C15 | Logs printed raw upstream text (could contain secrets) | `redact()` + `RedactingFilter` on every log path | `test_logs_redact_keys_bearer_and_jwt`, `test_usage_log_never_contains_key_values` |
| C16 | B: keys read once at import | Read from `env` at call time; a missing key skips the provider | `test_missing_key_skips_provider_without_error` |

### 5.2 Agent loop

| # | Weak point | Fix | Test |
|---|---|---|---|
| A1 | A: malformed tool-call JSON silently replaced with `{}` | Error with the decoder's message sent back to the model | `test_malformed_tool_json_is_reported_to_model` |
| A2 | A: wrong/missing args produce a vague error; JS number coercion | pydantic strict models, per-field errors returned | `test_schema_mismatch_lists_field_errors` |
| A3 | A: silent fallbacks for unsupported args (`units` ignored) | `extra="forbid"`; the handler gets only fields that were sent; no defaults invented | `test_unknown_argument_is_rejected`, `test_missing_required_field_is_not_defaulted` |
| A4 | A tool exception must not crash the turn | Exception → error text to the model; `BaseException` still propagates | `test_tool_exception_is_returned_as_error`, `test_keyboard_interrupt_is_not_swallowed` |
| A5 | A: at the round limit, history ends with tool results and no reply | Closing assistant message appended, `stop_reason="max_rounds"` | `test_round_limit_leaves_history_valid` |
| A6 | A: window counted in messages, not tokens (a 34-message window, measured in a private prototype) | Token-budgeted `trim_history` | `test_trim_history_respects_token_budget` |
| A7 | Trimming must never split `tool_calls` from its `tool` replies | Atomic units; property tests with Hypothesis, fixed `@seed` | `test_trim_history_never_orphans_tool_messages` (property), `test_trim_history_keeps_latest_user_message` (property) |
| A8 | A: protocol keys not in English | English JSON keys everywhere | `test_schemas_are_clean_for_every_format` |

### 5.3 Evaluation harness

| # | Weak point | Fix | Test |
|---|---|---|---|
| E1 | A: tool logic duplicated in the bench; its drift check compared schemas only and missed a behaviour drift (same input, different result) | One `build_default_registry()` used by agent, eval and MCP | `test_agent_eval_and_mcp_share_one_tool_definition` |
| E2 | A: results overwritten every run (headline numbers had no raw file) | Append-only JSONL; README numbers come from `REPORT.md`, generated from the JSONL | `test_results_are_appended_not_overwritten`, `test_report_is_generated_from_results_file` |
| E3 | A: 429/503 scored as model failures (9/20 and 40/40 in a saved run, measured in a private prototype) | `infra_error` outcome, excluded from accuracy | `test_infra_errors_are_excluded_from_accuracy` |
| E4 | A: no timeout in the bench | The bench goes through `LLMClient` (deadline) | covered by C6 |
| E5 | A: one run per case, no temperature control | `repeats` and `temperature` recorded with each result | `test_result_lines_record_run_settings` |
| E6 | A: "critical" detected by an emoji in a free-text description | `critical: bool` field in the case schema | `test_case_schema_rejects_unknown_fields` |
| E7 | A: the bench used 4 rounds, the app used 5 | The runner uses the real `Agent` and its settings | `test_runner_uses_agent_max_rounds` |
| E8 | Live runs must stay within 30 requests | Hard request budget, counted in HTTP requests | `test_run_stops_at_request_budget` |
| E9 | The verifier must be measured | Planted-defect benchmark | `test_verifier_bench_detects_every_planted_defect_type` |

### 5.4 RAG

| # | Weak point | Fix | Test |
|---|---|---|---|
| R1 | B: grounding was prompt-only, with no output check | Every quote must be a verbatim substring of a retrieved chunk; others are dropped and counted | `test_fabricated_quote_is_dropped_and_counted`, `test_quote_from_unretrieved_chunk_is_dropped` |
| R2 | The retrieval backend must be swappable | `Retriever` protocol; BM25 and the embedding retriever both satisfy it | `test_retrievers_share_the_protocol` |

### 5.5 Proxy

| # | Weak point in prototype A's proxy | Fix | Test |
|---|---|---|---|
| P1 | Errors returned as HTTP 200 | Real status codes (table in §3.7) | `test_upstream_failures_map_to_real_status_codes`, `test_bad_requests_get_4xx`, `test_other_methods_and_paths` |
| P2 | No `Origin` header → allowed | `Origin` required on `/api/*` and must be in the allowlist | `test_missing_origin_is_rejected`, `test_unlisted_origin_is_rejected` |
| P3 | `Host` not checked → DNS-rebinding shape accepted | `Host` allowlist → 421 | `test_dns_rebinding_host_is_rejected` |
| P4 | No overall deadline | Deadline passed to `LLMClient`; 504 on expiry | `test_deadline_exceeded_returns_504` |
| P5 | An error disguised as content was raised | Failover via the client | `test_proxy_fails_over_on_disguised_gateway_error` |
| P6 | Body cap counted characters, and the connection was dropped without a response | Byte cap from `Content-Length` before reading → 413 JSON, then a lingering close | `test_oversized_body_returns_413`, `test_rejected_requests_with_unread_body_always_get_their_response` |
| P7 | Client could send any system prompt and tool list | The server owns both; the client may send only user/assistant text | `test_client_cannot_send_system_or_tools` |
| P8 | Path traversal / everything served, including private files | Static root + allowlist + resolved-path check + normcase + dotfile block + bad `%` → 400 | `test_static_blocks_traversal` (15 cases: `../`, `%2e%2e%2f`, `..%5c`, absolute, dotfile, `.py`, bad `%`, ...) |
| P9 | B: count-then-insert daily cap race | Atomic counter under a lock | `test_daily_cap_is_atomic_under_concurrency` |
| P10 | B: client-supplied history trusted (forged assistant/tool turns) | Roles restricted, counts and lengths bounded | covered by the P7 test cases |

### 5.6 MCP server

| # | Requirement | Test |
|---|---|---|
| M1 | A backend rejection (409, not enough stock) reaches the model as an error result with the backend's text | `test_insufficient_stock_error_reaches_client`, `test_insufficient_stock_error_reaches_the_agents_model` |
| M2 | 404 / 400 details are surfaced | `test_unknown_sku_and_validation_errors_are_surfaced` |
| M3 | The non-idempotent `issue_stock` is not retried | `test_issue_stock_is_not_retried_on_timeout` |
| M4 | Undo refuses an unexpected label, atomically on the backend | `test_undo_refuses_unexpected_label`, `test_tools_against_real_inventory_manager` |
| M5 | The tool list and schemas match the registry | `test_mcp_lists_registry_tools` |

Not ported (app-specific, out of scope for a toolkit): a client-chosen restricted-answer mode, per-day chat
sessions, image stripping and browser-storage quota handling.

---

## 6. Test strategy

- `tests/conftest.py`: an autouse fixture deletes `OPENROUTER_API_KEY`, `GEMINI_API_KEY` and
  `ANTHROPIC_API_KEY` from the environment and patches `socket.socket.connect`, `connect_ex` and
  `socket.getaddrinfo` to refuse anything but loopback, so a test that tries the real network fails loudly.
- Time: `LLMClient` takes `clock` and `sleep`; tests use a fake clock, so backoff and deadline tests run
  instantly and assert exact waits.
- HTTP: `httpx.MockTransport` handlers scripted per test (status sequence, headers, bodies, raised
  `httpx.ReadTimeout`).
- Inventory: `FakeInventoryBackend`, a `MockTransport` that implements the subset of the inventory API with
  the same status codes and error bodies (409 `"only N in stock, cannot issue M"`). An optional integration
  test runs against a real `inventory-manager` checkout and is skipped unless `INVENTORY_MANAGER_PATH` is set.
- MCP: the SDK's in-memory `mcp.Client(server)` (in process, no subprocess); one example test starts the
  stdio server as a subprocess.
- Proxy: a real `ThreadingHTTPServer` on `127.0.0.1:0`, upstream through a scripted client.
- Hypothesis: `@seed(20261117)` and `settings(max_examples=500, deadline=None)` on the history property tests.

## 7. CI

`.github/workflows/ci.yml`: matrix `os: [ubuntu-latest, windows-latest]`, `python: ["3.11", "3.12"]`;
`pip install -e ".[dev]"`, `ruff check .`, `ruff format --check .`, `pytest`, then
`python -m llm_toolkit.eval check` and `python -m llm_toolkit.eval verifier-bench --dry-run` (offline,
writes nothing). CI uses no secrets.

## 8. Live evaluation

Runs only when a key is present in the environment, with at most 30 HTTP requests per run (enforced in
code). Without a key the command reports `skipped` and writes nothing. Results go to
`results/eval/*.jsonl`, and the README quotes `results/eval/REPORT.md`.

## 9. Trade-offs

1. **Non-429 4xx are split.** 400/422 raise immediately; 401/402/403/404 fail over with an ERROR log and a
   recorded attempt (see §3.1). Failing over on a malformed request would only repeat the mistake on every
   provider.
2. **`undo_last_change` requires `expected_label`.** The model must carry the label from `issue_stock`,
   which is more work for it, but an undo can never reverse a change it did not make. The alternative, an
   undo with no argument and a warning in the tool description, was rejected as unsafe with a shared stack.
3. **Eval cases are JSON**, not YAML, to avoid a YAML dependency.
4. **The proxy uses the standard library** (`http.server`), so no web framework is added.
5. **`tools/` is its own package** under `src/llm_toolkit/`, shared by the agent, the eval harness and the
   MCP server.
6. **Model ids are configuration, not code.** Defaults live in `client/providers.py` and can be overridden
   with `LLM_TOOLKIT_<PROVIDER>_MODEL`.
