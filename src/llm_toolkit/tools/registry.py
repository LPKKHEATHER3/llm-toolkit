"""The single place tools are defined.

The agent, the eval harness and the MCP server all get their tools from one registry, built by
``build_default_registry``. There is no second copy that could drift (the earlier prototype's bench
had its own copy of the tool logic and silently disagreed with the app).

Argument handling rules
-----------------------
* Arguments are parsed with ``json.loads`` first, so a malformed payload produces the decoder's
  message and position, which is sent back to the model. Nothing is replaced with ``{}``.
* Argument models are strict (no ``"12"`` -> ``12`` coercion) and forbid unknown fields.
* Optional fields must default to ``None``; handlers read :func:`provided` to see only what the
  model actually sent. A default the model did not ask for is never invented.
* A handler exception becomes an error result for the model. ``BaseException`` (KeyboardInterrupt,
  SystemExit) is not caught.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic_core import PydanticUndefined


class ToolArgs(BaseModel):
    """Base class for tool argument models."""

    model_config = ConfigDict(extra="forbid", strict=True)


def provided(args: BaseModel) -> dict[str, Any]:
    """Only the fields the model actually sent."""
    return args.model_dump(exclude_unset=True)


class ToolError(Exception):
    """An expected failure whose message is meant for the model (e.g. a backend rejection)."""


class ToolResult(BaseModel):
    ok: bool
    data: Any = None
    error: str | None = None

    def to_content(self) -> str:
        payload: dict[str, Any] = {"ok": self.ok}
        if self.ok:
            payload["data"] = self.data
        else:
            payload["error"] = self.error
        return json.dumps(payload, ensure_ascii=False, default=str)


def _inline_refs(schema: Any, defs: dict[str, Any]) -> Any:
    if isinstance(schema, dict):
        if "$ref" in schema:
            name = schema["$ref"].rsplit("/", 1)[-1]
            return _inline_refs(defs[name], defs)
        return {k: _inline_refs(v, defs) for k, v in schema.items() if k not in ("$defs", "title")}
    if isinstance(schema, list):
        return [_inline_refs(v, defs) for v in schema]
    return schema


def _drop_null_branches(schema: dict[str, Any], required: set[str]) -> dict[str, Any]:
    """``X | None = None`` renders as ``anyOf: [X, null]`` with ``default: null``. For the model, an
    optional field is simply one it may leave out, so show ``X`` and drop the default."""
    props = schema.get("properties", {})
    for name, prop in props.items():
        if name in required:
            continue
        prop.pop("default", None)
        branches = prop.get("anyOf")
        if isinstance(branches, list):
            non_null = [b for b in branches if b.get("type") != "null"]
            if len(non_null) == 1:
                prop.pop("anyOf")
                prop.update(non_null[0])
    return schema


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[ToolArgs]
    handler: Callable[[Any], Any]

    def __post_init__(self) -> None:
        for field_name, info in self.args_model.model_fields.items():
            has_default = info.default is not PydanticUndefined or info.default_factory is not None
            if has_default and info.default is not None:
                raise ValueError(
                    f"tool {self.name!r}: optional field {field_name!r} must default to None, "
                    "so no value is invented that the model did not send"
                )

    def parameters(self) -> dict[str, Any]:
        raw = self.args_model.model_json_schema()
        schema = _inline_refs(raw, raw.get("$defs", {}))
        schema = _drop_null_branches(schema, set(schema.get("required", [])))
        schema.setdefault("properties", {})
        schema["additionalProperties"] = False
        return schema

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters()},
        }

    def anthropic_schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.parameters()}

    def mcp_input_schema(self) -> dict[str, Any]:
        return self.parameters()


def _format_validation_error(err: ValidationError, model: type[BaseModel]) -> str:
    allowed = ", ".join(model.model_fields)
    parts = []
    for e in err.errors(include_url=False):
        loc = ".".join(str(x) for x in e["loc"]) or "(arguments)"
        if e["type"] == "missing":
            parts.append(f"{loc}: field required")
        elif e["type"] == "extra_forbidden":
            parts.append(f"{loc}: unknown field (allowed: {allowed})")
        else:
            parts.append(f"{loc}: {e['msg']} (got {e.get('input')!r})")
    return "invalid arguments: " + "; ".join(parts)


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()):
        self._tools: dict[str, Tool] = {}
        for t in tools:
            self.register(t)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool:
        return self._tools[name]

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self, fmt: Literal["openai", "anthropic"] = "openai") -> list[dict[str, Any]]:
        if fmt == "openai":
            return [t.openai_schema() for t in self._tools.values()]
        return [t.anthropic_schema() for t in self._tools.values()]

    def schema_hash(self) -> str:
        canonical = json.dumps(self.schemas("openai"), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def validate(self, name: str, raw_arguments: str | dict[str, Any] | None) -> ToolArgs | ToolResult:
        """Parse and validate arguments. Returns the args model, or an error ``ToolResult``."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                ok=False, error=f"unknown tool {name!r}; available tools: {', '.join(self._tools)}"
            )
        if isinstance(raw_arguments, dict):
            data: Any = raw_arguments
        else:
            text = raw_arguments if isinstance(raw_arguments, str) else ""
            try:
                data = json.loads(text)
            except json.JSONDecodeError as e:
                return ToolResult(
                    ok=False,
                    error=(
                        f"arguments are not valid JSON: {e.msg} at line {e.lineno} column {e.colno} "
                        f"(position {e.pos}). Send one JSON object that matches the {name} schema."
                    ),
                )
        if not isinstance(data, dict):
            return ToolResult(ok=False, error=f"arguments must be a JSON object, got {type(data).__name__}")
        try:
            return tool.args_model.model_validate(data)
        except ValidationError as e:
            return ToolResult(ok=False, error=_format_validation_error(e, tool.args_model))

    def call(self, name: str, raw_arguments: str | dict[str, Any] | None) -> ToolResult:
        """Run a tool. Never raises for model mistakes or handler failures (``Exception``)."""
        parsed = self.validate(name, raw_arguments)
        if isinstance(parsed, ToolResult):
            return parsed
        try:
            out = self._tools[name].handler(parsed)
        except ToolError as e:
            return ToolResult(ok=False, error=str(e))
        except Exception as e:  # deliberately not BaseException
            return ToolResult(ok=False, error=f"{type(e).__name__}: {e}")
        return out if isinstance(out, ToolResult) else ToolResult(ok=True, data=out)
