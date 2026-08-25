"""Pre-mutation idempotency behavior for id-bearing prompt submissions."""

from __future__ import annotations

import threading
import types

import pytest

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


def _params(text: str, *, submission_id="submission-1", truncate=True) -> dict:
    canonical = build_canonical_semantic_object(
        text_sha256=compute_text_sha256(text),
        display_kind=DisplayKind.NORMAL,
        queued=False,
        interrupted=False,
        surface="",
        truncation_target="row_id:101" if truncate else None,
        truncation_consent=True,
        attachments=[],
        replay_controls={"confirm_empty_truncate": True},
    )
    params = {
        "session_id": "idempotent-sid",
        "text": text,
        "submission_id": submission_id,
        "contract_version": CONTRACT_VERSION,
        "semantic_fingerprint": compute_semantic_fingerprint(canonical),
        "confirm_truncate": True,
        "confirm_empty_truncate": True,
    }
    if truncate:
        params["truncate_before_row_id"] = 101
    return params


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


def test_capacity_refusal_retry_returns_same_refusal_without_mutation(monkeypatch):
    """An id reserved before capacity refusal must replay that safe refusal."""
    session = _session()
    server._sessions["idempotent-sid"] = session
    original_history = list(session["history"])
    monkeypatch.setattr(
        server, "_ensure_active_session_slot", lambda *_args: "capacity exhausted"
    )
    monkeypatch.setattr(
        server, "_start_agent_build", lambda *_args: pytest.fail("must not build")
    )

    try:
        first = server.handle_request(
            {"id": "first", "method": "prompt.submit", "params": _params("rewrite")}
        )
        second = server.handle_request(
            {"id": "retry", "method": "prompt.submit", "params": _params("rewrite")}
        )

        assert first["error"]["code"] == 4090
        assert second == first | {"id": "retry"}
        assert session["history"] == original_history
        assert session["history_version"] == 0
        assert session["running"] is False
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


def test_id_bearing_typed_voice_stop_replays_saved_result_and_legacy_stays_unchanged(monkeypatch):
    """Typed voice stops save their local result only when an admission id exists."""
    session = _session()
    server._sessions["idempotent-sid"] = session
    monkeypatch.setenv("HERMES_VOICE", "1")
    monkeypatch.setenv("HERMES_VOICE_TTS", "1")
    monkeypatch.setitem(
        __import__("sys").modules,
        "tools.voice_mode",
        types.SimpleNamespace(is_voice_stop_phrase=lambda text: text.lower() == "stop"),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "hermes_cli.voice",
        types.SimpleNamespace(stop_continuous=lambda: None),
    )
    monkeypatch.setattr(server, "_tts_stream_stop", lambda **_kwargs: None)
    monkeypatch.setattr(server, "_voice_emit", lambda *_args: None)
    try:
        params = _params("stop", submission_id="voice-stop", truncate=False)
        first = server.handle_request(
            {"id": "first", "method": "prompt.submit", "params": params}
        )
        retry = server.handle_request(
            {"id": "retry", "method": "prompt.submit", "params": params}
        )
        assert first["result"] == {"voice_stopped": True}
        assert retry == first | {"id": "retry"}
        monkeypatch.setenv("HERMES_VOICE", "1")
        legacy = server.handle_request(
            {
                "id": "legacy",
                "method": "prompt.submit",
                "params": {"session_id": "idempotent-sid", "text": "stop"},
            }
        )
        assert legacy["result"] == {"voice_stopped": True}
    finally:
        server._sessions.pop("idempotent-sid", None)


def test_duplicate_during_provisional_admission_never_observes_streaming(monkeypatch):
    """A controlled first caller leaves duplicates pending until its refusal is known."""
    entered = threading.Event()
    release = threading.Event()
    session = _session()
    server._sessions["idempotent-sid"] = session

    def capacity_gate(*_args):
        entered.set()
        assert release.wait(timeout=2)
        return "capacity exhausted"

    monkeypatch.setattr(server, "_ensure_active_session_slot", capacity_gate)
    params = _params("rewrite", submission_id="interleaved")
    first_response = {}
    first_thread = threading.Thread(
        target=lambda: first_response.setdefault(
            "value",
            server.handle_request(
                {"id": "first", "method": "prompt.submit", "params": params}
            ),
        )
    )
    try:
        first_thread.start()
        assert entered.wait(timeout=2)
        duplicate = server.handle_request(
            {"id": "duplicate", "method": "prompt.submit", "params": params}
        )
        admission = session["_prompt_submission_admissions"]["interleaved"]

        assert duplicate["result"] == {"submission_id": "interleaved", "status": "pending"}
        assert admission["status"] == "pending"
        release.set()
        first_thread.join(timeout=2)
        assert first_response["value"]["error"]["code"] == 4090
    finally:
        release.set()
        first_thread.join(timeout=2)
        server._sessions.pop("idempotent-sid", None)


def test_invalid_truncate_refusal_replays_same_saved_error(monkeypatch):
    """A later validation refusal finalizes the reserved id without mutation."""
    session = _session()
    server._sessions["idempotent-sid"] = session
    params = _params("rewrite", submission_id="invalid-truncate", truncate=False)
    try:
        first = server.handle_request(
            {"id": "first", "method": "prompt.submit", "params": params}
        )
        retry = server.handle_request(
            {"id": "retry", "method": "prompt.submit", "params": params}
        )

        assert first["error"]["code"] == 4004
        assert retry == first | {"id": "retry"}
        assert session["history_version"] == 0
        assert session["running"] is False
    finally:
        server._sessions.pop("idempotent-sid", None)
