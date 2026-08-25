"""PROVENANCE_UNVERIFIED — durable prompt-submit receipt/outbox APIs.

This mixin deliberately owns durable state transitions. Generic RPC response
handling must never settle these records: only owner and attempt tokens can.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

_TERMINAL = {"COMPLETED", "TERMINAL_ERROR", "ATTACHMENT_REATTACH_REQUIRED", "UNKNOWN_OUTCOME"}
_SAFE_ACTION = {
    "ATTACHMENT_REATTACH_REQUIRED": "reattach_then_continue",
    "UNKNOWN_OUTCOME": "check_transcript_resume_or_start_new",
    "TERMINAL_ERROR": "start_new_submission",
}


class SessionSubmissionMixin:
    """SessionDB-owned receipt/work state machine; never uses state_meta."""

    @staticmethod
    def _submission_ack(submission_id: str, state: str) -> dict[str, Any]:
        return {"submission_id": submission_id, "contract_version": "1",
                "durable_admission_status": "accepted", "invocation_status": state.lower(),
                "safe_terminal_action": _SAFE_ACTION.get(state)}

    def create_or_read_prompt_submission(self, *, session_id: str, submission_id: str,
                                         contract_version: str, semantic_fingerprint: str,
                                         payload: dict[str, Any], before_commit=None) -> dict[str, Any]:
        """One BEGIN IMMEDIATE create-or-read transaction before runtime mutation."""
        if contract_version != "1":
            raise ValueError("unsupported prompt submission contract")
        # Current attachment data is renderer-visible capability, so never accept it durably.
        payload_wire = json.dumps(payload, sort_keys=True)
        if any(key in payload_wire.lower() for key in ("path", "url", "token", "preview")):
            raise ValueError("ATTACHMENT_REATTACH_REQUIRED")
        now = time.time()
        def write(conn):
            row = conn.execute("SELECT semantic_fingerprint, work_id FROM prompt_submission_receipts WHERE session_id=? AND submission_id=?", (session_id, submission_id)).fetchone()
            if row:
                if row[0] != semantic_fingerprint:
                    return {"created": False, "conflict": True, "code": 4091}
                state = conn.execute("SELECT state FROM prompt_accepted_work WHERE work_id=?", (row[1],)).fetchone()[0]
                return {"created": False, "conflict": False, "work_id": row[1], "ack": self._submission_ack(submission_id, state)}
            work_id, ack = str(uuid.uuid4()), self._submission_ack(submission_id, "ACCEPTED")
            conn.execute("INSERT INTO prompt_submission_receipts (session_id, submission_id, contract_version, semantic_fingerprint, work_id, safe_ack_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (session_id, submission_id, contract_version, semantic_fingerprint, work_id, json.dumps(ack, sort_keys=True), now, now))
            conn.execute("INSERT INTO prompt_accepted_work (work_id, session_id, submission_id, payload_json, state, owner_generation, invocation_attempt_no, created_at, updated_at) VALUES (?, ?, ?, ?, 'ACCEPTED', 0, 0, ?, ?)", (work_id, session_id, submission_id, payload_wire, now, now))
            # Test-only deterministic crash/race checkpoint; transaction remains real.
            if before_commit is not None:
                before_commit()
            return {"created": True, "conflict": False, "work_id": work_id, "ack": ack}
        return self._execute_write(write)

    def claim_prompt_submission_work(self, work_id: str, *, owner_token: str, lease_seconds: float = 30) -> dict[str, Any] | None:
        now = time.time()
        def write(conn):
            changed = conn.execute("UPDATE prompt_accepted_work SET state='DISPATCHING', owner_token=?, owner_generation=owner_generation+1, lease_expires_at=?, updated_at=? WHERE work_id=? AND state IN ('ACCEPTED','QUEUED')", (owner_token, now + lease_seconds, now, work_id)).rowcount
            if changed != 1: return None
            row = conn.execute("SELECT state, owner_generation FROM prompt_accepted_work WHERE work_id=?", (work_id,)).fetchone()
            return {"state": row[0], "owner_generation": row[1]}
        return self._execute_write(write)

    def mark_prompt_submission_invoking(self, work_id: str, *, owner_token: str, attempt_token: str, lease_seconds: float = 30) -> dict[str, Any] | None:
        now = time.time()
        def write(conn):
            changed = conn.execute("UPDATE prompt_accepted_work SET state='INVOKING', invocation_attempt_token=?, invocation_attempt_no=invocation_attempt_no+1, lease_expires_at=?, updated_at=? WHERE work_id=? AND owner_token=? AND state='DISPATCHING'", (attempt_token, now + lease_seconds, now, work_id, owner_token)).rowcount
            return {"state": "INVOKING", "invocation_attempt_token": attempt_token} if changed == 1 else None
        return self._execute_write(write)

    def mark_prompt_submission_running(self, work_id: str, *, owner_token: str, attempt_token: str) -> bool:
        return self._execute_write(lambda conn: conn.execute("UPDATE prompt_accepted_work SET state='RUNNING', updated_at=? WHERE work_id=? AND owner_token=? AND invocation_attempt_token=? AND state='INVOKING'", (time.time(), work_id, owner_token, attempt_token)).rowcount == 1)

    def complete_prompt_submission_work(self, work_id: str, *, owner_token: str, attempt_token: str, state: str, safe_terminal: dict[str, Any] | None = None) -> bool:
        if state not in _TERMINAL: raise ValueError("invalid durable terminal state")
        summary = {k: safe_terminal[k] for k in ("layer", "code", "retryable", "safe_action") if safe_terminal and k in safe_terminal}
        return self._execute_write(lambda conn: conn.execute("UPDATE prompt_accepted_work SET state=?, safe_terminal_json=?, owner_token=NULL, lease_expires_at=NULL, updated_at=? WHERE work_id=? AND owner_token=? AND invocation_attempt_token=? AND state IN ('INVOKING','RUNNING')", (state, json.dumps(summary, sort_keys=True), time.time(), work_id, owner_token, attempt_token)).rowcount == 1)

    def recover_prompt_submission_work(self, *, now: float | None = None) -> list[dict[str, Any]]:
        """Only pre-invocation work is eligible; prior provider intent is unknown."""
        now = time.time() if now is None else now
        def write(conn):
            conn.execute("UPDATE prompt_accepted_work SET state='UNKNOWN_OUTCOME', owner_token=NULL, lease_expires_at=NULL, safe_terminal_json=?, updated_at=? WHERE state IN ('INVOKING','RUNNING')", (json.dumps({"code": "unknown_outcome", "safe_action": _SAFE_ACTION["UNKNOWN_OUTCOME"]}), now))
            conn.execute("UPDATE prompt_accepted_work SET state='ACCEPTED', owner_token=NULL, lease_expires_at=NULL, updated_at=? WHERE state='DISPATCHING' AND lease_expires_at < ?", (now, now))
            return [dict(r) for r in conn.execute("SELECT work_id, session_id, submission_id, state FROM prompt_accepted_work WHERE state IN ('ACCEPTED','QUEUED') ORDER BY created_at, work_id").fetchall()]
        return self._execute_write(write)

    def prompt_submission_counts(self, session_id: str) -> tuple[int, int]:
        with self._lock:
            a = self._conn.execute("SELECT COUNT(*) FROM prompt_submission_receipts WHERE session_id=?", (session_id,)).fetchone()[0]
            b = self._conn.execute("SELECT COUNT(*) FROM prompt_accepted_work WHERE session_id=?", (session_id,)).fetchone()[0]
        return int(a), int(b)
