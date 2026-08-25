"""Pre-mutation idempotency behavior for id-bearing prompt submissions."""

from __future__ import annotations

import threading
import types

from tui_gateway import server
from tui_gateway.prompt_submission_contract import (
    CONTRACT_VERSION,
    DisplayKind,
    build_canonical_semantic_object,
    compute_semantic_fingerprint,
    compute_text_sha256,
)


def _session(**extra):
    return {
        "agent": types.SimpleNamespace(),
        "session_key": "session-key",
        "history": [
            {"_row_id": 101, "role": "user", "content": "first"},
            {"_row_id": 102, "role": "assistant", "content": "reply"},
        ],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        **extra,
    }


def _params(text: str) -> dict:
    canonical = build_canonical_semantic_object(
        text_sha256=compute_text_sha256(text),
        display_kind=DisplayKind.NORMAL,
        queued=False,
        interrupted=False,
        surface="",
        truncation_target="row_id:101",
        truncation_consent=True,
        attachments=[],
        replay_controls={"confirm_empty_truncate": True},
    )
    return {
        "session_id": "idempotent-sid",
        "text": text,
        "submission_id": "submission-1",
        "contract_version": CONTRACT_VERSION,
        "semantic_fingerprint": compute_semantic_fingerprint(canonical),
        "truncate_before_row_id": 101,
        "confirm_truncate": True,
        "confirm_empty_truncate": True,
    }


def test_duplicate_semantic_mismatch_refuses_before_rewind_mutation(monkeypatch):
    """A reused id with changed text must not re-run a confirmed rewind."""
    replaced = []

    class _UnstartedThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr(server.threading, "Thread", _UnstartedThread)
    session = _session()
    server._sessions["idempotent-sid"] = session

    class _DB:
        def replace_messages(self, key, messages, **kwargs):
            replaced.append((key, list(messages)))

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(server, "_start_agent_build", lambda *args: None)
    monkeypatch.setattr(server, "_start_inflight_turn", lambda *args: None)

    try:
        first = server.handle_request(
            {"id": "first", "method": "prompt.submit", "params": _params("rewrite")}
        )
        assert first["result"]["status"] == "streaming"
        assert len(replaced) == 1
        history_version = session["history_version"]

        second = server.handle_request(
            {
                "id": "second",
                "method": "prompt.submit",
                "params": _params("changed meaning"),
            }
        )

        assert second["error"]["code"] == 4091
        assert second["error"]["data"] == {
            "submission_id": "submission-1",
            "field_classes": ["semantic_fingerprint"],
        }
        assert len(replaced) == 1
        assert session["history_version"] == history_version
    finally:
        server._sessions.pop("idempotent-sid", None)


def test_matching_duplicate_returns_stored_status_before_busy_redirect(monkeypatch):
    """A lost rewind response must not redirect the already-admitted live turn."""
    redirects = []
    agent = types.SimpleNamespace(
        _supports_active_turn_redirect=True,
        redirect=lambda text: redirects.append(text) or True,
    )
    session = _session(agent=agent)
    server._sessions["idempotent-sid"] = session

    class _DB:
        def replace_messages(self, key, messages, **kwargs):
            pass

    class _UnstartedThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr(server.threading, "Thread", _UnstartedThread)
    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(server, "_start_agent_build", lambda *args: None)
    monkeypatch.setattr(server, "_start_inflight_turn", lambda *args: None)

    try:
        first = server.handle_request(
            {"id": "first", "method": "prompt.submit", "params": _params("rewrite")}
        )
        duplicate = server.handle_request(
            {"id": "retry", "method": "prompt.submit", "params": _params("rewrite")}
        )

        assert first["result"]["status"] == "streaming"
        assert duplicate["result"]["status"] == "streaming"
        assert redirects == []
    finally:
        server._sessions.pop("idempotent-sid", None)
