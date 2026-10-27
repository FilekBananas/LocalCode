"""How hard to squeeze: the difference between a 4B local model and a cloud one.

Everything in this project is shaped by one assumption — the model is small and
its window is tight.  Tool menus get shortened, tool output clipped, old tool
results masked, the conversation compacted at 80 % of the window.  All of that
is right for a 4B model with 8k tokens and wrong for a frontier model with two
hundred thousand: clipping a file to 400 lines there only makes the model ask
for the rest, and compaction costs a paid request to save room nobody needed.

So there are two profiles, and the model decides which one applies:

``local``
    what LocalCode has always done — the default, and what every model that
    runs on this Mac gets.
``strong``
    for models reached over an API key (:mod:`localcode.remote`): the full tool
    menu, generous tool output, a later compaction threshold and more room for
    the answer.

Nothing here is a preference dialog.  A model either runs on this machine or it
does not, and that single fact picks the profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: A window this size or smaller gets the short tool menu and compact schemas.
TINY_CTX = 8192


@dataclass(frozen=True)
class ModelProfile:
    """The knobs that differ between a small local model and a strong remote one."""

    name: str
    #: Short tool menu, schemas stripped of their prose (see ``agent/tools``).
    tiny_tools: bool
    #: Fraction of the window at which the conversation is compacted.
    compact_threshold: float
    #: Tokens kept free for the answer, when nothing else says otherwise.
    reserve_output: int
    #: Replace the body of old tool results with a one-line note.
    mask_old_tool_results: bool

    @property
    def is_strong(self) -> bool:
        return self.name == "strong"


LOCAL = ModelProfile(
    name="local",
    tiny_tools=True,                 # decided per window, see profile_for
    compact_threshold=0.80,
    reserve_output=1024,
    mask_old_tool_results=True,
)

STRONG = ModelProfile(
    name="strong",
    tiny_tools=False,
    # A frontier window is large enough that compacting early throws away
    # detail for nothing; and every compaction of a remote model is a paid
    # request, so it waits until the window is genuinely running out.
    compact_threshold=0.92,
    reserve_output=8192,
    mask_old_tool_results=False,
)


def is_remote_model(model: Any) -> bool:
    """True for a model reached over somebody's API key."""
    if model is None:
        return False
    if str(getattr(model, "format", "") or "") == "remote":
        return True
    if str(getattr(model, "source", "") or "") == "remote":
        return True
    key = str(getattr(model, "key", "") or model if isinstance(model, str) else
              getattr(model, "key", "") or "")
    return key.startswith("remote:")


def profile_for(model: Any = None, n_ctx: int | None = None, *,
                remote: bool | None = None) -> ModelProfile:
    """The profile for one model.

    ``remote`` overrides the detection (the tests and the chat service know it
    up front); otherwise it is read off the model row.
    """
    strong = remote if remote is not None else is_remote_model(model)
    if strong:
        return STRONG
    window = int(n_ctx or 0)
    # Local models keep exactly what they had: the short menu below 8k.
    return ModelProfile(
        name="local",
        tiny_tools=bool(window and window <= TINY_CTX),
        compact_threshold=LOCAL.compact_threshold,
        reserve_output=LOCAL.reserve_output,
        mask_old_tool_results=LOCAL.mask_old_tool_results,
    )
