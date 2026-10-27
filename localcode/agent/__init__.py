"""The LocalCode coding agent: the turn loop, its context budget, the
permission policy, the model-facing prompts and the session transcript.

``localcode.agent.tools`` holds the tools themselves and their registry
(``default_tools``/``tools_for_mode``/``get_tool``); the agent picks its tool
set from there lazily, per mode and context size.

Typical use (the TUI and the HTTP server do exactly this)::

    from localcode.agent import Agent, SessionStore

    store = SessionStore.create(cwd, model_key=key, mode="default")
    agent = Agent(runtime, key, cwd, settings, on_event, ask_permission, session=store)
    await agent.run_turn("Napraw failujący test")

Resuming an earlier session::

    store = SessionStore.resume(session_id)          # store.messages / store.items
    agent = Agent(runtime, store.session.model_key, store.session.cwd, settings,
                  on_event, ask_permission, session=store)
"""

from __future__ import annotations

from localcode.agent.context import ContextManager, limits_for, truncate
from localcode.agent.loop import Agent, AgentBusyError, dict_events
from localcode.agent.permissions import MODE_LABELS, MODES, PermissionPolicy, normalize_mode
from localcode.agent.session import (
    AgentSession,
    SessionError,
    SessionStore,
    delete_session,
    list_sessions,
    load_session,
)

__all__ = [
    "Agent",
    "AgentBusyError",
    "AgentSession",
    "ContextManager",
    "MODES",
    "MODE_LABELS",
    "PermissionPolicy",
    "SessionError",
    "SessionStore",
    "delete_session",
    "dict_events",
    "limits_for",
    "list_sessions",
    "load_session",
    "normalize_mode",
    "truncate",
]
