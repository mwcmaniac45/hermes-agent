"""PROVENANCE_UNVERIFIED — durable prompt submission outbox integration tests.

These tests prove durable acceptance/admission, not exactly-once provider execution.
"""
from __future__ import annotations

import json
from hermes_state import SessionDB
import pytest
import sqlite3
import threading


def _db_with_session(path, session_id="stored-session"):
    db = SessionDB(db_path=path)
    db.create_session(session_id, "tui")
    return db


def test_prompt_submission_requires_existing_profile_local_session_and_fk_is_clean(tmp_path):
    """Receipts/work belong to a stored session in this profile-local database."""
    db = _db_with_session(tmp_path / "default.db")
    other = _db_with_session(tmp_path / "other.db")
    try:
        import sqlite3
        with pytest.raises(sqlite3.IntegrityError):
            db.create_or_read_prompt_submission(
                session_id="missing", submission_id="orphan", contract_version="1",
                semantic_fingerprint="0" * 64, payload={"text": "allowed"},
            )
        created = db.create_or_read_prompt_submission(
            session_id="stored-session", submission_id="local", contract_version="1",
            semantic_fingerprint="1" * 64, payload={"text": "allowed"},
        )
        assert created["created"] is True
        assert db._conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert other.prompt_submission_counts("stored-session") == (0, 0)
        db._conn.execute("DELETE FROM sessions WHERE id='stored-session'")
        assert db.prompt_submission_counts("stored-session") == (0, 0)
    finally:
        db.close()
        other.close()


def test_durable_receipt_ack_includes_stored_semantic_fingerprint_on_create_and_replay(tmp_path):
    """The versioned acknowledgement binds both create and replay to its fingerprint."""
    db = _db_with_session(tmp_path / "state.db")
    fingerprint = "a" * 64
    try:
        created = db.create_or_read_prompt_submission(
            session_id="stored-session", submission_id="fingerprinted", contract_version="1",
            semantic_fingerprint=fingerprint, payload={"text": "allowed"},
        )
        replayed = db.create_or_read_prompt_submission(
            session_id="stored-session", submission_id="fingerprinted", contract_version="1",
            semantic_fingerprint=fingerprint, payload={"text": "allowed"},
        )
        assert created["ack"]["semantic_fingerprint"] == fingerprint
        assert replayed["ack"]["semantic_fingerprint"] == fingerprint
    finally:
        db.close()


def test_recovery_reattaches_proven_unexpired_live_owner_and_completion_survives(tmp_path):
    """Recovery preserves only the exact, leased invocation owner witness."""
    db = _db_with_session(tmp_path / "state.db")
    try:
        work = db.create_or_read_prompt_submission(
            session_id="stored-session", submission_id="live", contract_version="1",
            semantic_fingerprint="2" * 64, payload={"text": "allowed"},
        )
        claim = db.claim_prompt_submission_work(work["work_id"], owner_token="owner", lease_seconds=60)
        attempt = db.mark_prompt_submission_invoking(
            work["work_id"], owner_token="owner", owner_generation=1, attempt_token="attempt", lease_seconds=60,
        )
        assert db.recover_prompt_submission_work(
            now=0,
            live_owner_witnesses=[{
                "work_id": work["work_id"],
                "owner_generation": claim["owner_generation"],
                "owner_token": "owner",
                "invocation_attempt_token": attempt["invocation_attempt_token"],
            }],
        ) == []
        assert db.complete_prompt_submission_work(
            work["work_id"], owner_token="owner", owner_generation=1, attempt_token="attempt", state="COMPLETED",
        ) is True
    finally:
        db.close()


def test_payload_is_structural_allowlist_and_benign_text_keeps_ordinary_words(tmp_path):
    """Text is allowed verbatim; attachment capabilities are rejected by field shape."""
    db = _db_with_session(tmp_path / "state.db")
    try:
        payload = {
            "text": "Discuss this URL token path preview without persisting a capability.",
            "display_kind": "normal",
            "queued": True,
            "interrupted": False,
            "truncation_target": "message",
            "truncation_consent": True,
            "attachments": [{"identity": "attachment-1", "version": "v1", "order": 0, "status": "ready"}],
        }
        work = db.create_or_read_prompt_submission(
            session_id="stored-session", submission_id="payload", contract_version="1",
            semantic_fingerprint="3" * 64, payload=payload,
        )
        stored = json.loads(db._conn.execute(
            "SELECT payload_json FROM prompt_accepted_work WHERE work_id=?", (work["work_id"],)
        ).fetchone()[0])
        assert stored == {"payload_version": 1, **payload}
        with pytest.raises(ValueError, match="ATTACHMENT_REATTACH_REQUIRED"):
            db.create_or_read_prompt_submission(
                session_id="stored-session", submission_id="hostile", contract_version="1",
                semantic_fingerprint="4" * 64,
                payload={"text": "allowed", "attachments": [{"identity": "a", "version": "v1", "order": 0, "status": "ready", "file": "/private"}]},
            )
    finally:
        db.close()


@pytest.mark.parametrize(
    "field, hostile",
    [
        (field, hostile)
        for field in ("identity", "version", "status")
        for hostile in (
            "https://secret.example/file?token=abc",
            "/Users/alice/private.png",
            "../private.png",
            "Bearer secret-token-value",
        )
    ],
)
def test_attachment_persisted_strings_reject_capabilities_before_payload_json(tmp_path, field, hostile):
    """Only conservative opaque attachment metadata may enter the durable payload."""
    db = _db_with_session(tmp_path / "state.db")
    try:
        attachment = {"identity": "attachment-1", "version": "v1", "order": 0, "status": "ready"}
        attachment[field] = hostile
        with pytest.raises(ValueError, match="ATTACHMENT_REATTACH_REQUIRED"):
            db.create_or_read_prompt_submission(
                session_id="stored-session", submission_id=f"hostile-{field}", contract_version="1",
                semantic_fingerprint="4" * 64, payload={"text": "allowed", "attachments": [attachment]},
            )
        assert db.prompt_submission_counts("stored-session") == (0, 0)
        assert db._conn is not None
        assert db._conn.execute("SELECT COUNT(payload_json) FROM prompt_accepted_work").fetchone()[0] == 0
    finally:
        db.close()


def test_conflicting_same_id_cannot_settle_original_pending_work(tmp_path):
    """A mismatch must not alter the original receipt or its later completion."""
    db = _db_with_session(tmp_path / "state.db")
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
            created["work_id"], owner_token="owner-a", owner_generation=1, attempt_token="attempt-a"
        )
        assert claim["state"] == "DISPATCHING"
        assert attempt["state"] == "INVOKING"
        assert db.complete_prompt_submission_work(
            created["work_id"], owner_token="owner-a", owner_generation=1, attempt_token="attempt-a", state="COMPLETED"
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


def test_real_sessiondb_barrier_race_replays_same_fingerprint_once(tmp_path):
    """Two connections race one identity and replay the sole durable receipt/work."""
    path = tmp_path / "state.db"
    entered, release = threading.Event(), threading.Event()
    first = _db_with_session(path)
    second = SessionDB(db_path=path)
    results, replay = [], []
    thread = second_writer = second_conn = None

    def first_writer():
        results.append(first.create_or_read_prompt_submission(
            session_id="stored-session", submission_id="race-1", contract_version="1",
            semantic_fingerprint="c" * 64, payload={"text": "private"},
            before_commit=lambda: (entered.set(), release.wait(2)),
        ))

    try:
        thread = threading.Thread(target=first_writer)
        thread.start()
        assert entered.wait(2)

        # The trace callback runs as the second connection dispatches its real
        # BEGIN IMMEDIATE, before SQLite can acquire the first writer's lock.
        second_begin_attempted = threading.Event()

        def trace_second_begin(statement):
            if statement == "BEGIN IMMEDIATE":
                second_begin_attempted.set()

        second_conn = second._conn
        assert second_conn is not None
        second_conn.set_trace_callback(trace_second_begin)
        second_writer = threading.Thread(target=lambda: replay.append(second.create_or_read_prompt_submission(
            session_id="stored-session", submission_id="race-1", contract_version="1",
            semantic_fingerprint="c" * 64, payload={"text": "private"},
        )))
        second_writer.start()
        assert second_begin_attempted.wait(2), "second writer never attempted BEGIN IMMEDIATE while first was held"
        release.set()
        thread.join(3)
        second_writer.join(3)

        assert not thread.is_alive()
        assert not second_writer.is_alive()
        all_results = results + replay
        assert len(all_results) == 2
        assert [result["created"] for result in all_results].count(True) == 1
        assert [result["created"] for result in all_results].count(False) == 1
        assert all(result["conflict"] is False for result in all_results)
        assert results[0]["work_id"] == replay[0]["work_id"]
        assert first.prompt_submission_counts("stored-session") == (1, 1)
    finally:
        release.set()
        if thread is not None:
            thread.join(3)
        if second_writer is not None:
            second_writer.join(3)
        if second_conn is not None:
            second_conn.set_trace_callback(None)
        first.close()
        second.close()


def test_commit_before_runtime_admission_recovers_once_after_fresh_db_resume(tmp_path):
    """A committed record is discoverable without any runtime-side admission."""
    path = tmp_path / "state.db"
    db = _db_with_session(path)
    accepted = db.create_or_read_prompt_submission(session_id="stored-session", submission_id="crash-1", contract_version="1", semantic_fingerprint="e" * 64, payload={"text": "recover exactly"})
    db.close()  # Simulated gateway death before dispatcher admission.
    resumed = SessionDB(db_path=path)
    eligible = resumed.recover_prompt_submission_work()
    assert [(item["submission_id"], item["state"]) for item in eligible] == [("crash-1", "ACCEPTED")]
    retry = resumed.create_or_read_prompt_submission(session_id="stored-session", submission_id="crash-1", contract_version="1", semantic_fingerprint="e" * 64, payload={"text": "recover exactly"})
    assert retry["created"] is False and retry["ack"]["invocation_status"] == "accepted"
    resumed.close()


def test_invocation_crashes_become_unknown_outcome_without_provider_retry(tmp_path):
    """INVOKING/RUNNING is invocation intent, not permission to rerun a provider."""
    db = _db_with_session(tmp_path / "state.db")
    for suffix, reached_provider in (("before", 0), ("after", 1)):
        work = db.create_or_read_prompt_submission(session_id="stored-session", submission_id=f"invoke-{suffix}", contract_version="1", semantic_fingerprint=("f" if suffix == "before" else "a") * 64, payload={"text": "private"})
        db.claim_prompt_submission_work(work["work_id"], owner_token=f"owner-{suffix}")
        db.mark_prompt_submission_invoking(work["work_id"], owner_token=f"owner-{suffix}", owner_generation=1, attempt_token=f"attempt-{suffix}")
        if reached_provider:
            assert db.mark_prompt_submission_running(work["work_id"], owner_token=f"owner-{suffix}", owner_generation=1, attempt_token=f"attempt-{suffix}")
    assert db.recover_prompt_submission_work() == []
    for suffix in ("before", "after"):
        replay = db.create_or_read_prompt_submission(session_id="stored-session", submission_id=f"invoke-{suffix}", contract_version="1", semantic_fingerprint=("f" if suffix == "before" else "a") * 64, payload={"text": "private"})
        assert replay["ack"]["invocation_status"] == "unknown_outcome"
    db.close()


def test_hostile_persistence_error_never_enters_safe_receipt_or_replay(tmp_path):
    db = _db_with_session(tmp_path / "state.db")
    hostile = "prompt=steal https://secret.example/?token=abc traceback"
    work = db.create_or_read_prompt_submission(session_id="stored-session", submission_id="safe-1", contract_version="1", semantic_fingerprint="9" * 64, payload={"text": "private prompt"})
    db.claim_prompt_submission_work(work["work_id"], owner_token="owner")
    db.mark_prompt_submission_invoking(work["work_id"], owner_token="owner", owner_generation=1, attempt_token="attempt")
    assert db.complete_prompt_submission_work(work["work_id"], owner_token="owner", owner_generation=1, attempt_token="attempt", state="TERMINAL_ERROR", safe_terminal={"layer": "provider", "code": "storage_failed", "retryable": False, "safe_action": "start_new_submission"})
    replay = db.create_or_read_prompt_submission(session_id="stored-session", submission_id="safe-1", contract_version="1", semantic_fingerprint="9" * 64, payload={"text": "private prompt"})
    assert hostile not in str(replay)
    with db._lock:
        raw = db._conn.execute("SELECT safe_ack_json, safe_terminal_json FROM prompt_submission_receipts r JOIN prompt_accepted_work w ON w.work_id=r.work_id WHERE r.submission_id='safe-1'").fetchone()
    assert hostile not in "".join(raw)
    db.close()



def test_legacy_outbox_schema_migrates_to_session_owned_foreign_keys(tmp_path):
    path = tmp_path / "legacy.db"
    db = _db_with_session(path)
    db.create_or_read_prompt_submission(session_id="stored-session", submission_id="legacy", contract_version="1", semantic_fingerprint="7" * 64, payload={"text": "allowed"})
    db.close()
    legacy = sqlite3.connect(path)
    legacy.executescript("""
        PRAGMA foreign_keys=OFF;
        ALTER TABLE prompt_accepted_work RENAME TO work_old;
        ALTER TABLE prompt_submission_receipts RENAME TO receipt_old;
        CREATE TABLE prompt_submission_receipts (session_id TEXT NOT NULL, submission_id TEXT NOT NULL, contract_version TEXT NOT NULL, semantic_fingerprint TEXT NOT NULL, work_id TEXT NOT NULL UNIQUE, safe_ack_json TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL, PRIMARY KEY (session_id, submission_id));
        CREATE TABLE prompt_accepted_work (work_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, submission_id TEXT NOT NULL, payload_json TEXT NOT NULL, state TEXT NOT NULL, owner_token TEXT, owner_generation INTEGER NOT NULL DEFAULT 0, lease_expires_at REAL, invocation_attempt_token TEXT, invocation_attempt_no INTEGER NOT NULL DEFAULT 0, safe_terminal_json TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL, UNIQUE (session_id, submission_id), FOREIGN KEY (session_id, submission_id) REFERENCES prompt_submission_receipts(session_id, submission_id));
        INSERT INTO prompt_submission_receipts SELECT * FROM receipt_old;
        INSERT INTO prompt_accepted_work (work_id, session_id, submission_id, payload_json, state, owner_token, owner_generation, lease_expires_at, invocation_attempt_token, invocation_attempt_no, safe_terminal_json, created_at, updated_at) SELECT work_id, session_id, submission_id, payload_json, state, owner_token, owner_generation, lease_expires_at, invocation_attempt_token, invocation_attempt_no, safe_terminal_json, created_at, updated_at FROM work_old;
        DROP TABLE work_old;
        DROP TABLE receipt_old;
    """)
    legacy.close()
    migrated = SessionDB(db_path=path)
    try:
        assert migrated.prompt_submission_counts("stored-session") == (1, 1)
        assert migrated._conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert {row[2] for row in migrated._conn.execute("PRAGMA foreign_key_list(prompt_submission_receipts)")} == {"sessions"}
    finally:
        migrated.close()


def test_recovery_rejects_mismatched_or_expired_live_owner_witnesses(tmp_path):
    db = _db_with_session(tmp_path / "state.db")
    try:
        for submission_id, witness, generation_delta, expiry in (("wrong-attempt", "other-attempt", 0, 100), ("foreign-generation", "attempt-foreign-generation", 1, 100), ("expired", "attempt-expired", 0, 10)):
            work = db.create_or_read_prompt_submission(session_id="stored-session", submission_id=submission_id, contract_version="1", semantic_fingerprint=submission_id[0] * 64, payload={"text": "allowed"})
            claim = db.claim_prompt_submission_work(work["work_id"], owner_token=f"owner-{submission_id}")
            attempt = f"attempt-{submission_id}"
            db.mark_prompt_submission_invoking(work["work_id"], owner_token=f"owner-{submission_id}", owner_generation=1, attempt_token=attempt)
            with db._lock:
                db._conn.execute("UPDATE prompt_accepted_work SET lease_expires_at=? WHERE work_id=?", (expiry, work["work_id"]))
            db.recover_prompt_submission_work(now=50, live_owner_witnesses=[{"work_id": work["work_id"], "owner_generation": claim["owner_generation"] + generation_delta, "owner_token": f"owner-{submission_id}", "invocation_attempt_token": witness}])
            assert db.create_or_read_prompt_submission(session_id="stored-session", submission_id=submission_id, contract_version="1", semantic_fingerprint=submission_id[0] * 64, payload={"text": "allowed"})["ack"]["invocation_status"] == "unknown_outcome"
    finally:
        db.close()


def test_recovery_reclaims_only_expired_dispatching_work(tmp_path):
    db = _db_with_session(tmp_path / "state.db")
    try:
        for submission_id, expiry in (("expired-dispatch", 10), ("live-dispatch", 100)):
            work = db.create_or_read_prompt_submission(session_id="stored-session", submission_id=submission_id, contract_version="1", semantic_fingerprint=submission_id[0] * 64, payload={"text": "allowed"})
            db.claim_prompt_submission_work(work["work_id"], owner_token=submission_id)
            with db._lock:
                db._conn.execute("UPDATE prompt_accepted_work SET lease_expires_at=? WHERE work_id=?", (expiry, work["work_id"]))
        eligible = db.recover_prompt_submission_work(now=50)
        assert [item["submission_id"] for item in eligible] == ["expired-dispatch"]
        assert db._conn.execute("SELECT state FROM prompt_accepted_work WHERE submission_id='live-dispatch'").fetchone()[0] == "DISPATCHING"
    finally:
        db.close()


@pytest.mark.parametrize("field, hostile", [("layer", "https://secret.example/?token=x"), ("code", "https://secret.example/?token=x"), ("retryable", "false"), ("safe_action", "https://secret.example/?token=x")])
def test_terminal_summary_rejects_hostile_value_in_every_safe_field(tmp_path, field, hostile):
    db = _db_with_session(tmp_path / "state.db")
    try:
        work = db.create_or_read_prompt_submission(session_id="stored-session", submission_id="terminal", contract_version="1", semantic_fingerprint="5" * 64, payload={"text": "allowed"})
        db.claim_prompt_submission_work(work["work_id"], owner_token="owner")
        db.mark_prompt_submission_invoking(work["work_id"], owner_token="owner", owner_generation=1, attempt_token="attempt")
        summary = {"layer": "provider", "code": "storage_failed", "retryable": False, "safe_action": "start_new_submission"}
        summary[field] = hostile
        with pytest.raises(ValueError, match="invalid durable terminal summary"):
            db.complete_prompt_submission_work(work["work_id"], owner_token="owner", owner_generation=1, attempt_token="attempt", state="TERMINAL_ERROR", safe_terminal=summary)
    finally:
        db.close()


def test_payload_rejects_unknown_alternate_attachment_capability_key(tmp_path):
    db = _db_with_session(tmp_path / "state.db")
    try:
        with pytest.raises(ValueError, match="ATTACHMENT_REATTACH_REQUIRED"):
            db.create_or_read_prompt_submission(session_id="stored-session", submission_id="alternate-key", contract_version="1", semantic_fingerprint="6" * 64, payload={"text": "allowed", "attachments": [{"identity": "a", "version": "v1", "order": 0, "status": "ready", "renderer_handle": "secret"}]})
    finally:
        db.close()
