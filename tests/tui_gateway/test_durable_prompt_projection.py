"""PROVENANCE_UNVERIFIED — vertical durable v1 text projection tests."""
from __future__ import annotations

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


def _db(tmp_path, *session_ids):
    db = SessionDB(db_path=tmp_path / "state.db")
    for session_id in session_ids:
        db.create_session(session_id, "tui")
    return db


def _work(db, session_id="session-a", submission_id="submission-a", text="private text"):
    return db.create_or_read_prompt_submission(
        session_id=session_id,
        submission_id=submission_id,
        contract_version="1",
        semantic_fingerprint="a" * 64,
        payload={"text": text},
    )


def _fingerprint(text, *, queued=False, interrupted=False):
    return compute_semantic_fingerprint(build_canonical_semantic_object(
        text_sha256=compute_text_sha256(text),
        display_kind=DisplayKind.NORMAL,
        queued=queued,
        interrupted=interrupted,
        surface="",
        truncation_target=None,
        truncation_consent=False,
        attachments=[],
        replay_controls={"attachments": "unsupported", "truncation": "unsupported"},
    ))


def _session(profile_home, session_id="session-a"):
    return {
        "session_key": session_id,
        "profile_home": str(profile_home),
        "history": [],
        "history_lock": threading.RLock(),
        "running": False,
        "client_surface": "",
    }


def test_claim_returns_only_claimed_canonical_payload_and_session_identity(tmp_path):
    db = _db(tmp_path, "session-a", "session-b")
    try:
        work = _work(db)
        claim = db.claim_prompt_submission_work(
            work["work_id"], owner_token="owner-a", session_id="session-a"
        )
        assert {key: claim[key] for key in ("work_id", "session_id", "owner_generation", "payload")} == {
            "work_id": work["work_id"],
            "session_id": "session-a",
            "owner_generation": 1,
            "payload": {
                "payload_version": 1,
                "text": "private text",
                "display_kind": "normal",
                "queued": False,
                "interrupted": False,
                "truncation_target": None,
                "truncation_consent": False,
                "attachments": [],
            },
        }
    finally:
        db.close()


def test_fresh_session_scoped_recovery_marks_only_its_invocation_unknown(tmp_path):
    db = _db(tmp_path, "session-a", "session-b")
    try:
        first, second = _work(db, "session-a", "first"), _work(db, "session-b", "second")
        for work, owner, attempt in ((first, "owner-a", "attempt-a"), (second, "owner-b", "attempt-b")):
            db.claim_prompt_submission_work(work["work_id"], owner_token=owner, session_id=work is first and "session-a" or "session-b")
            db.mark_prompt_submission_invoking(work["work_id"], owner_token=owner, attempt_token=attempt)
        assert db.recover_prompt_submission_work(session_id="session-a", now=0) == []
        assert db.create_or_read_prompt_submission(
            session_id="session-a", submission_id="first", contract_version="1", semantic_fingerprint="a" * 64, payload={"text": "private text"}
        )["ack"]["invocation_status"] == "unknown_outcome"
        assert db.create_or_read_prompt_submission(
            session_id="session-b", submission_id="second", contract_version="1", semantic_fingerprint="a" * 64, payload={"text": "private text"}
        )["ack"]["invocation_status"] == "invoking"
    finally:
        db.close()


def test_owner_attempt_fencing_rejects_stale_renewal_and_settlement(tmp_path):
    db = _db(tmp_path, "session-a")
    try:
        work = _work(db)
        db.claim_prompt_submission_work(work["work_id"], owner_token="owner-a", session_id="session-a")
        db.mark_prompt_submission_invoking(work["work_id"], owner_token="owner-a", attempt_token="attempt-a")
        assert db.renew_prompt_submission_lease(
            work["work_id"], owner_token="owner-b", attempt_token="attempt-a", lease_seconds=60
        ) is False
        assert db.renew_prompt_submission_lease(
            work["work_id"], owner_token="owner-a", attempt_token="attempt-b", lease_seconds=60
        ) is False
        assert db.complete_prompt_submission_work(
            work["work_id"], owner_token="owner-b", attempt_token="attempt-a", state="COMPLETED"
        ) is False
        assert db.renew_prompt_submission_lease(
            work["work_id"], owner_token="owner-a", attempt_token="attempt-a", lease_seconds=60
        ) is True
    finally:
        db.close()


@pytest.mark.parametrize("control", ("running", "queued", "interrupted", "attachments", "hidden", "truncation"))
def test_v1_unsupported_or_busy_admission_has_no_receipt_or_runtime_mutation(monkeypatch, tmp_path, control):
    profile = tmp_path / "profile"
    profile.mkdir()
    db = SessionDB(db_path=profile / "state.db")
    db.create_session("session-a", "tui")
    db.close()
    session = _session(profile)
    if control == "running":
        session["running"] = True
    if control == "attachments":
        session["attached_images"] = ["/not-used"]
    server._sessions["session-a"] = session
    calls = []
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda *_a, **_k: calls.append("ensure"))
    monkeypatch.setattr(server, "_schedule_durable_prompt_projection", lambda *_a, **_k: calls.append("project"))
    params = {
        "session_id": "session-a", "text": "private text", "submission_id": f"submission-{control}",
        "contract_version": "1", "semantic_fingerprint": _fingerprint("private text"),
    }
    if control in {"queued", "interrupted"}:
        params[control] = True
        params["semantic_fingerprint"] = _fingerprint("private text", **{control: True})
    elif control == "hidden":
        params["display_kind"] = "hidden"
    elif control == "truncation":
        params["confirm_truncate"] = True
    try:
        response = server.handle_request({"id": "r", "method": "prompt.submit", "params": params})
        assert response["error"]["code"] == 4092
        check = SessionDB(db_path=profile / "state.db")
        try:
            assert check.prompt_submission_counts("session-a") == (0, 0)
        finally:
            check.close()
        assert calls == []
    finally:
        server._sessions.pop("session-a", None)
