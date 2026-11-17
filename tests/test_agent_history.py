"""Property tests for history trimming (Hypothesis, fixed seed)."""

from __future__ import annotations

from hypothesis import HealthCheck, given, seed, settings
from hypothesis import strategies as st

from llm_toolkit.agent import approx_tokens, trim_history
from llm_toolkit.client import Message, ToolCall

SEED = 20261117
EXAMPLES = 500
PROPERTY = settings(
    max_examples=EXAMPLES,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
)

text = st.text(alphabet="abcdefghij ", min_size=0, max_size=120)


@st.composite
def turn(draw, turn_no: int) -> list[Message]:
    msgs = [Message.user(f"u{turn_no} " + draw(text))]
    for r in range(draw(st.integers(0, 3))):
        n = draw(st.integers(1, 3))
        ids = [f"t{turn_no}_{r}_{k}" for k in range(n)]
        msgs.append(
            Message(
                role="assistant",
                content=draw(st.one_of(st.none(), text)),
                tool_calls=[ToolCall(id=i, name="get_item", arguments='{"sku": "ELE-0001"}') for i in ids],
            )
        )
        msgs.extend(Message.tool(i, draw(text)) for i in ids)
    if draw(st.booleans()):
        msgs.append(Message.assistant(draw(text)))
    return msgs


@st.composite
def conversation(draw) -> list[Message]:
    n = draw(st.integers(1, 12))
    out: list[Message] = []
    for t in range(n):
        out.extend(draw(turn(t)))
    return out


@st.composite
def corrupted_conversation(draw) -> list[Message]:
    """A conversation whose head was cut badly by some earlier code: orphan tool replies and an
    assistant tool call with a missing reply."""
    msgs = draw(conversation())
    prefix = [Message.tool("orphan_1", "x"), Message.tool("orphan_2", "y")]
    if draw(st.booleans()):
        prefix.insert(
            0,
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(id="half_a", name="get_item", arguments="{}"),
                    ToolCall(id="half_b", name="get_item", arguments="{}"),
                ],
            ),
        )
        prefix.insert(1, Message.tool("half_a", "only one reply"))
    return prefix + msgs


def check_invariants(window: list[Message]) -> None:
    assert window, "the current turn is always kept"
    assert window[0].role == "user", "window starts at a user message"
    pending: set[str] = set()
    for m in window:
        if m.role == "tool":
            assert m.tool_call_id in pending, f"orphan tool message {m.tool_call_id}"
            pending.discard(m.tool_call_id)
        else:
            assert not pending, f"tool calls without replies: {pending}"
            if m.role == "assistant" and m.tool_calls:
                pending = {tc.id for tc in m.tool_calls}
    assert not pending, f"tool calls without replies at the end: {pending}"


def current_turn(msgs: list[Message]) -> list[Message]:
    last_user = max(i for i, m in enumerate(msgs) if m.role == "user")
    return msgs[last_user:]


@seed(SEED)
@PROPERTY
@given(msgs=conversation(), budget=st.integers(0, 3000))
def test_trim_history_never_orphans_tool_messages(msgs, budget):
    window = trim_history(msgs, budget)
    check_invariants(window)
    assert window == msgs[len(msgs) - len(window) :], "the window is a suffix of the conversation"


@seed(SEED)
@PROPERTY
@given(msgs=conversation(), budget=st.integers(0, 3000))
def test_trim_history_keeps_latest_user_message(msgs, budget):
    window = trim_history(msgs, budget)
    turn_msgs = current_turn(msgs)
    assert window[-len(turn_msgs) :] == turn_msgs


@seed(SEED)
@PROPERTY
@given(msgs=conversation(), budget=st.integers(0, 3000))
def test_trim_history_respects_token_budget(msgs, budget):
    window = trim_history(msgs, budget)
    used = sum(approx_tokens(m) for m in window)
    turn_cost = sum(approx_tokens(m) for m in current_turn(msgs))
    if turn_cost <= budget:
        assert used <= budget
    else:
        assert window == current_turn(msgs), "over budget: only the current turn is sent"


@seed(SEED)
@PROPERTY
@given(msgs=corrupted_conversation(), budget=st.integers(0, 3000))
def test_trim_history_repairs_corrupted_heads(msgs, budget):
    check_invariants(trim_history(msgs, budget))
