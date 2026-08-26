"""PROVENANCE_UNVERIFIED — PREPARING durable runtime boundaries."""
from hermes_state import SessionDB


def _db(path):
    db = SessionDB(db_path=path)
    db.create_session("s", "tui")
    return db


def _work(db, submission):
    return db.create_or_read_prompt_submission(
        session_id="s", submission_id=submission, contract_version="1",
        semantic_fingerprint=(submission[0] * 64), payload={"text": "private"},
    )


def test_preparing_is_fenced_and_recovers_to_accepted_not_unknown(tmp_path):
    db = _db(tmp_path / "state.db")
    try:
        work = _work(db, "preparing")
        claim = db.claim_prompt_submission_work(work["work_id"], owner_token="owner")
        preparing = db.mark_prompt_submission_preparing(
            work["work_id"], owner_token="owner", prepare_token="prepare", lease_seconds=60,
        )
        assert preparing == {"state": "PREPARING", "runtime_prepare_token": "prepare"}
        # A stale preparation is definitely pre-provider and is replayable.
        db.recover_prompt_submission_work(now=100)
        replay = db.create_or_read_prompt_submission(
            session_id="s", submission_id="preparing", contract_version="1",
            semantic_fingerprint="p" * 64, payload={"text": "private"},
        )
        assert replay["ack"]["invocation_status"] == "accepted"
        assert claim["owner_generation"] == 1
    finally:
        db.close()


def test_preparing_promotes_with_separate_tokens_and_exact_terminal_fence(tmp_path):
    db = _db(tmp_path / "state.db")
    try:
        work = _work(db, "promote")
        db.claim_prompt_submission_work(work["work_id"], owner_token="owner")
        db.mark_prompt_submission_preparing(work["work_id"], owner_token="owner", prepare_token="prepare")
        assert not db.complete_prompt_submission_preparing(
            work["work_id"], owner_token="owner", prepare_token="wrong", state="TERMINAL_ERROR",
            safe_terminal={"layer": "runtime", "code": "local_runtime_failed", "retryable": False, "safe_action": "start_new_submission"},
        )
        invoking = db.mark_prompt_submission_invoking_from_preparing(
            work["work_id"], owner_token="owner", prepare_token="prepare", attempt_token="attempt",
        )
        assert invoking == {"state": "INVOKING", "invocation_attempt_token": "attempt"}
        row = db._conn.execute(
            "SELECT runtime_prepare_token, invocation_attempt_token FROM prompt_accepted_work WHERE work_id=?", (work["work_id"],)
        ).fetchone()
        assert tuple(row) == ("prepare", "attempt")
        assert db.mark_prompt_submission_running(work["work_id"], owner_token="owner", attempt_token="attempt")
    finally:
        db.close()
