"""PROVENANCE_UNVERIFIED — durable prompt submission outbox integration tests.

These tests prove durable acceptance/admission, not exactly-once provider execution.
"""
from __future__ import annotations

from hermes_state import SessionDB
import threading



def test_conflicting_same_id_cannot_settle_original_pending_work(tmp_path):
    """A mismatch must not alter the original receipt or its later completion."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        created = db.create_or_read_prompt_submission(
            session_id="stored-session",
            submission_id="submission-1",
            contract_version="1",
            semantic_fingerprint="a" * 64,
            payload={"text": "private original prompt", "attachments": []},
        )
        assert created["created"] is True
        conflict = db.create_or_read_prompt_submission(
            session_id="stored-session",
            submission_id="submission-1",
            contract_version="1",
            semantic_fingerprint="b" * 64,
            payload={"text": "different private prompt", "attachments": []},
        )
        assert conflict == {"created": False, "conflict": True, "code": 4091}
        claim = db.claim_prompt_submission_work(created["work_id"], owner_token="owner-a")
        attempt = db.mark_prompt_submission_invoking(
            created["work_id"], owner_token="owner-a", attempt_token="attempt-a"
        )
        assert claim["state"] == "DISPATCHING"
        assert attempt["state"] == "INVOKING"
        assert db.complete_prompt_submission_work(
            created["work_id"], owner_token="owner-a", attempt_token="attempt-a", state="COMPLETED"
        ) is True
        replay = db.create_or_read_prompt_submission(
            session_id="stored-session",
            submission_id="submission-1",
            contract_version="1",
            semantic_fingerprint="a" * 64,
            payload={"text": "private original prompt", "attachments": []},
        )
        assert replay["created"] is False
        assert replay["conflict"] is False
        assert replay["ack"]["invocation_status"] == "completed"
    finally:
        db.close()


def test_real_sessiondb_barrier_race_creates_one_receipt_and_one_work(tmp_path):
    """The transaction barrier proves the losing same-fingerprint caller is read-only."""
    path = tmp_path / "state.db"
    entered, release = threading.Event(), threading.Event()
    first = SessionDB(db_path=path)
    second = SessionDB(db_path=path)
    results = []

    def first_writer():
        results.append(first.create_or_read_prompt_submission(
            session_id="stored-session", submission_id="race-1", contract_version="1",
            semantic_fingerprint="c" * 64, payload={"text": "private"},
            before_commit=lambda: (entered.set(), release.wait(2)),
        ))

    thread = threading.Thread(target=first_writer)
    thread.start()
    assert entered.wait(2)
    # A mismatch waits for the real SQLite transaction then returns only 4091.
    conflict = []
    mismatch = threading.Thread(target=lambda: conflict.append(second.create_or_read_prompt_submission(
        session_id="stored-session", submission_id="race-1", contract_version="1",
        semantic_fingerprint="d" * 64, payload={"text": "other"},
    )))
    mismatch.start()
    release.set(); thread.join(3); mismatch.join(3)
    assert results[0]["created"] is True
    assert conflict == [{"created": False, "conflict": True, "code": 4091}]
    assert first.prompt_submission_counts("stored-session") == (1, 1)
    first.close(); second.close()


def test_commit_before_runtime_admission_recovers_once_after_fresh_db_resume(tmp_path):
    """A committed record is discoverable without any runtime-side admission."""
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    accepted = db.create_or_read_prompt_submission(session_id="stored", submission_id="crash-1", contract_version="1", semantic_fingerprint="e" * 64, payload={"text": "recover exactly"})
    db.close()  # Simulated gateway death before dispatcher admission.
    resumed = SessionDB(db_path=path)
    eligible = resumed.recover_prompt_submission_work()
    assert [(item["submission_id"], item["state"]) for item in eligible] == [("crash-1", "ACCEPTED")]
    retry = resumed.create_or_read_prompt_submission(session_id="stored", submission_id="crash-1", contract_version="1", semantic_fingerprint="e" * 64, payload={"text": "recover exactly"})
    assert retry["created"] is False and retry["ack"]["invocation_status"] == "accepted"
    resumed.close()


def test_invocation_crashes_become_unknown_outcome_without_provider_retry(tmp_path):
    """INVOKING/RUNNING is invocation intent, not permission to rerun a provider."""
    db = SessionDB(db_path=tmp_path / "state.db")
    for suffix, reached_provider in (("before", 0), ("after", 1)):
        work = db.create_or_read_prompt_submission(session_id="stored", submission_id=f"invoke-{suffix}", contract_version="1", semantic_fingerprint=("f" if suffix == "before" else "a") * 64, payload={"text": "private"})
        db.claim_prompt_submission_work(work["work_id"], owner_token=f"owner-{suffix}")
        db.mark_prompt_submission_invoking(work["work_id"], owner_token=f"owner-{suffix}", attempt_token=f"attempt-{suffix}")
        if reached_provider:
            assert db.mark_prompt_submission_running(work["work_id"], owner_token=f"owner-{suffix}", attempt_token=f"attempt-{suffix}")
    assert db.recover_prompt_submission_work() == []
    for suffix in ("before", "after"):
        replay = db.create_or_read_prompt_submission(session_id="stored", submission_id=f"invoke-{suffix}", contract_version="1", semantic_fingerprint=("f" if suffix == "before" else "a") * 64, payload={"text": "private"})
        assert replay["ack"]["invocation_status"] == "unknown_outcome"
    db.close()


def test_hostile_persistence_error_never_enters_safe_receipt_or_replay(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    hostile = "prompt=steal https://secret.example/?token=abc traceback"
    work = db.create_or_read_prompt_submission(session_id="stored", submission_id="safe-1", contract_version="1", semantic_fingerprint="9" * 64, payload={"text": "private prompt"})
    db.claim_prompt_submission_work(work["work_id"], owner_token="owner")
    db.mark_prompt_submission_invoking(work["work_id"], owner_token="owner", attempt_token="attempt")
    assert db.complete_prompt_submission_work(work["work_id"], owner_token="owner", attempt_token="attempt", state="TERMINAL_ERROR", safe_terminal={"code": "storage_failed", "message": hostile, "traceback": hostile})
    replay = db.create_or_read_prompt_submission(session_id="stored", submission_id="safe-1", contract_version="1", semantic_fingerprint="9" * 64, payload={"text": "private prompt"})
    assert hostile not in str(replay)
    with db._lock:
        raw = db._conn.execute("SELECT safe_ack_json, safe_terminal_json FROM prompt_submission_receipts r JOIN prompt_accepted_work w ON w.work_id=r.work_id WHERE r.submission_id='safe-1'").fetchone()
    assert hostile not in "".join(raw)
    db.close()
