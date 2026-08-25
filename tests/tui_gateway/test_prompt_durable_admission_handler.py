"""PROVENANCE_UNVERIFIED — handler-level durable prompt admission tests.

These prove exactly-once durable acceptance/admission, not provider execution.
"""
from __future__ import annotations

import logging
import threading

import pytest

from hermes_state import SessionDB
from tui_gateway import server
from tui_gateway.prompt_submission_contract import (
    DisplayKind,
    build_canonical_semantic_object,
    compute_semantic_fingerprint,
    compute_text_sha256,
)


def _session(profile_home, session_id="ui-session"):
    return {
        "session_key": session_id,
        "profile_home": str(profile_home),
        "history": [],
        "history_lock": threading.RLock(),
        "running": False,
        "client_surface": "",
    }


def _fingerprint(*, text, queued=False, interrupted=False, surface=""):
    """Use the public v1 canonicalizer, never a test-only hash algorithm."""
    return compute_semantic_fingerprint(build_canonical_semantic_object(
        text_sha256=compute_text_sha256(text),
        display_kind=DisplayKind.NORMAL,
        queued=queued,
        interrupted=interrupted,
        surface="hud" if surface == "hud" else "",
        truncation_target=None,
        truncation_consent=False,
        attachments=[],
        replay_controls={"attachments": "unsupported", "truncation": "unsupported"},
    ))


def _request(submission_id="submission-1", fingerprint=None, **extra):
    params = {
        "session_id": "ui-session",
        "text": "private text",
        "submission_id": submission_id,
        "contract_version": "1",
        **extra,
    }
    params["semantic_fingerprint"] = fingerprint or _fingerprint(
        text=params["text"],
        queued=bool(params.get("queued")),
        interrupted=bool(params.get("interrupted")),
        surface=params.get("surface", ""),
    )
    return {"id": "request-1", "method": "prompt.submit", "params": params}


def _prepare(profile_home):
    profile_home.mkdir()
    db = SessionDB(db_path=profile_home / "state.db")
    db.create_session("ui-session", "tui")
    db.close()


def _count(profile_home):
    db = SessionDB(db_path=profile_home / "state.db")
    try:
        return db.prompt_submission_counts("ui-session")
    finally:
        db.close()


def test_handler_invalid_claimed_fingerprint_cannot_mutate_or_settle_original(monkeypatch, tmp_path):
    """A noncanonical same-id retry is rejected before durable or legacy mutation."""
    profile_home = tmp_path / "profile"
    _prepare(profile_home)
    session = _session(profile_home)
    server._sessions["ui-session"] = session
    mutations = []
    for name in (
        "_ensure_active_session_slot", "_handle_busy_submit", "_start_inflight_turn",
        "_start_agent_build", "_run_prompt_submit",
    ):
        monkeypatch.setattr(server, name, lambda *a, _name=name, **k: mutations.append(_name))
    monkeypatch.setattr(server, "_schedule_durable_prompt_projection", lambda *a, **k: None)
    try:
        accepted = server.handle_request(_request())
        assert accepted["result"]["durable_admission_status"] == "accepted"
        original = _count(profile_home)
        rejected = server.handle_request(_request(fingerprint="b" * 64))
        assert rejected["error"] == {
            "code": 4004, "message": "invalid durable prompt submission"
        }
        assert _count(profile_home) == original == (1, 1)
        assert mutations == []
        assert session["client_surface"] == "" and session["running"] is False
    finally:
        server._sessions.pop("ui-session", None)


def test_handler_matching_retry_while_running_replays_before_busy_rejection(monkeypatch, tmp_path):
    profile_home = tmp_path / "profile"
    _prepare(profile_home)
    session = _session(profile_home)
    server._sessions["ui-session"] = session
    projections = []
    monkeypatch.setattr(server, "_schedule_durable_prompt_projection", lambda *a, **k: projections.append(a))
    try:
        accepted = server.handle_request(_request())
        session["running"] = True
        replay = server.handle_request(_request())
        assert replay["result"] == accepted["result"]
        assert len(projections) == 1
        assert _count(profile_home) == (1, 1)
    finally:
        server._sessions.pop("ui-session", None)


def test_handler_rejects_unverified_v1_before_any_storage_write(monkeypatch, tmp_path):
    """Version, fingerprint grammar, and preimage mismatch fail closed pre-admission."""
    profile_home = tmp_path / "profile"
    _prepare(profile_home)
    session = _session(profile_home)
    server._sessions["ui-session"] = session
    attempted_writes = []
    monkeypatch.setattr(
        server, "_ensure_session_db_row", lambda *a, **k: attempted_writes.append("ensure")
    )
    try:
        for request in (
            _request(submission_id="bad-version", contract_version="2"),
            _request(submission_id="bad-grammar", fingerprint="A" * 64),
            _request(submission_id="bad-preimage", fingerprint="a" * 64),
        ):
            response = server.handle_request(request)
            assert response["error"] == {
                "code": 4004, "message": "invalid durable prompt submission"
            }
        assert attempted_writes == []
        assert _count(profile_home) == (0, 0)
    finally:
        server._sessions.pop("ui-session", None)


@pytest.mark.parametrize("field", ("queued", "interrupted"))
@pytest.mark.parametrize("wire_value", ("false", 1, None, [], {}))
def test_handler_rejects_non_boolean_durable_control_before_any_mutation(
    monkeypatch, tmp_path, field, wire_value
):
    """Present durable flags must be JSON booleans, never truthy wire values."""
    profile_home = tmp_path / "profile"
    _prepare(profile_home)
    session = _session(profile_home)
    server._sessions["ui-session"] = session
    attempted_writes, mutations = [], []
    monkeypatch.setattr(
        server, "_ensure_session_db_row", lambda *a, **k: attempted_writes.append("ensure")
    )
    for name in (
        "_ensure_active_session_slot", "_handle_busy_submit", "_start_inflight_turn",
        "_start_agent_build", "_run_prompt_submit",
    ):
        monkeypatch.setattr(server, name, lambda *a, _name=name, **k: mutations.append(_name))
    try:
        fingerprint = _fingerprint(
            text="private text",
            queued=bool(wire_value) if field == "queued" else False,
            interrupted=bool(wire_value) if field == "interrupted" else False,
        )
        response = server.handle_request(_request(
            submission_id=f"non-bool-{field}-{type(wire_value).__name__}",
            fingerprint=fingerprint,
            **{field: wire_value},
        ))
        assert response is not None
        assert response["error"] == {
            "code": 4004, "message": "invalid durable prompt submission"
        }
        assert attempted_writes == []
        assert mutations == []
        assert _count(profile_home) == (0, 0)
        assert session["client_surface"] == "" and session["running"] is False
    finally:
        server._sessions.pop("ui-session", None)


def test_handler_same_id_changed_semantics_never_replays_prior_ack(monkeypatch, tmp_path):
    """A claimed fingerprint cannot bind changed admitted text or flags to old work."""
    profile_home = tmp_path / "profile"
    _prepare(profile_home)
    server._sessions["ui-session"] = _session(profile_home)
    monkeypatch.setattr(server, "_schedule_durable_prompt_projection", lambda *a, **k: None)
    try:
        accepted = server.handle_request(_request())
        assert accepted["result"]["durable_admission_status"] == "accepted"
        claimed = _fingerprint(text="private text")
        for changed in (
            {"text": "unrelated second request"},
            {"queued": True},
            {"interrupted": True},
        ):
            rejected = server.handle_request(_request(fingerprint=claimed, **changed))
            assert rejected["error"] == {
                "code": 4004, "message": "invalid durable prompt submission"
            }
        assert _count(profile_home) == (1, 1)
    finally:
        server._sessions.pop("ui-session", None)


def test_handler_matching_retry_replays_durable_ack_without_mutation(monkeypatch, tmp_path):
    """A matching retry returns the stored safe status and never reprojects."""
    profile_home = tmp_path / "profile"
    _prepare(profile_home)
    server._sessions["ui-session"] = _session(profile_home)
    projections, mutations = [], []
    monkeypatch.setattr(server, "_schedule_durable_prompt_projection", lambda *a, **k: projections.append(a))
    for name in ("_ensure_active_session_slot", "_handle_busy_submit", "_start_inflight_turn", "_start_agent_build"):
        monkeypatch.setattr(server, name, lambda *a, _name=name, **k: mutations.append(_name))
    try:
        first = server.handle_request(_request())
        replay = server.handle_request(_request())
        assert first["result"] == replay["result"]
        assert first["result"]["invocation_status"] == "accepted"
        assert len(projections) == 1
        assert mutations == [] and _count(profile_home) == (1, 1)
    finally:
        server._sessions.pop("ui-session", None)


def test_handler_commit_checkpoint_is_recoverable_and_exact_retry_reads_it(monkeypatch, tmp_path):
    """A process death after commit before projection retains one recoverable intent."""
    profile_home = tmp_path / "profile"
    _prepare(profile_home)
    server._sessions["ui-session"] = _session(profile_home)
    monkeypatch.setattr(server, "_schedule_durable_prompt_projection", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("raw projection failure")))
    try:
        accepted = server.handle_request(_request())
        assert accepted["result"]["durable_admission_status"] == "accepted"
        assert _count(profile_home) == (1, 1)
        resumed = SessionDB(db_path=profile_home / "state.db")
        try:
            assert [(w["submission_id"], w["state"]) for w in resumed.recover_prompt_submission_work()] == [("submission-1", "ACCEPTED")]
        finally:
            resumed.close()
        replay = server.handle_request(_request())
        assert replay["result"] == accepted["result"]
    finally:
        server._sessions.pop("ui-session", None)


@pytest.mark.parametrize(
    ("control", "value"),
    (
        ("truncate_before_user_ordinal", 0),
        ("truncate_before_row_id", "row-1"),
        ("truncate_before_message_id", "message-1"),
        ("confirm_truncate", True),
        ("confirm_empty_truncate", True),
    ),
)
def test_handler_rejects_every_v1_truncation_control_before_admission(
    monkeypatch, tmp_path, control, value
):
    """Unsupported durable v1 rejects targets and orphan confirmations alike."""
    profile_home = tmp_path / "profile"
    _prepare(profile_home)
    server._sessions["ui-session"] = _session(profile_home)
    attempted_writes = []
    monkeypatch.setattr(
        server, "_ensure_session_db_row", lambda *a, **k: attempted_writes.append("ensure")
    )
    try:
        response = server.handle_request(_request(submission_id=f"control-{control}", **{control: value}))
        assert response["error"] == {
            "code": 4092,
            "message": "durable truncation is not available",
            "data": {"safe_action": "start_new_submission"},
        }
        assert attempted_writes == []
        assert _count(profile_home) == (0, 0)
    finally:
        server._sessions.pop("ui-session", None)


def test_handler_changed_truncation_control_same_id_never_replays_prior_ack(monkeypatch, tmp_path):
    """A retry cannot attach unsupported confirmation state to accepted work."""
    profile_home = tmp_path / "profile"
    _prepare(profile_home)
    server._sessions["ui-session"] = _session(profile_home)
    monkeypatch.setattr(server, "_schedule_durable_prompt_projection", lambda *a, **k: None)
    try:
        accepted = server.handle_request(_request())
        replay = server.handle_request(_request(
            fingerprint=_fingerprint(text="private text"), confirm_empty_truncate=True
        ))
        assert accepted["result"]["durable_admission_status"] == "accepted"
        assert replay["error"] == {
            "code": 4092,
            "message": "durable truncation is not available",
            "data": {"safe_action": "start_new_submission"},
        }
        assert _count(profile_home) == (1, 1)
    finally:
        server._sessions.pop("ui-session", None)


def test_id_bearing_unsafe_attachment_or_truncation_fails_before_admission(monkeypatch, tmp_path):
    """Legacy-only path/rewind inputs fail closed without receipt or mutation."""
    profile_home = tmp_path / "profile"
    _prepare(profile_home)
    session = _session(profile_home)
    session["attached_images"] = ["/private/path.png"]
    server._sessions["ui-session"] = session
    monkeypatch.setattr(server, "_schedule_durable_prompt_projection", lambda *a, **k: None)
    try:
        attachment = server.handle_request(_request())
        assert attachment["error"]["code"] == 4092
        assert attachment["error"]["data"] == {"safe_action": "reattach_then_continue"}
        session.pop("attached_images")
        truncate = server.handle_request(_request(submission_id="submission-2", truncate_before_user_ordinal=0, confirm_truncate=True))
        assert truncate["error"]["code"] == 4092
        assert truncate["error"]["data"] == {"safe_action": "start_new_submission"}
        assert _count(profile_home) == (0, 0) and session["running"] is False
    finally:
        server._sessions.pop("ui-session", None)


def test_projection_failure_never_leaks_or_changes_safe_replay(monkeypatch, tmp_path, caplog):
    """Post-commit projection errors stay out of logs and cannot rewrite the receipt."""
    profile_home = tmp_path / "profile"
    _prepare(profile_home)
    server._sessions["ui-session"] = _session(profile_home)
    raw = "hostile prompt=/private token=not-a-secret path=/private/never-log"
    monkeypatch.setattr(server, "_schedule_durable_prompt_projection", lambda *a, **k: (_ for _ in ()).throw(RuntimeError(raw)))
    caplog.set_level(logging.DEBUG, logger=server.__name__)
    try:
        accepted = server.handle_request(_request())
        replay = server.handle_request(_request())
        assert raw not in str(accepted) + str(replay)
        assert accepted["result"] == replay["result"]
        assert accepted["result"]["invocation_status"] == "accepted"
        assert _count(profile_home) == (1, 1)
        assert raw not in caplog.text
        assert all(record.exc_info is None for record in caplog.records)
    finally:
        server._sessions.pop("ui-session", None)
