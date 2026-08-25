"""PROVENANCE_UNVERIFIED — durable prompt-submit receipt/outbox APIs.

This mixin deliberately owns durable state transitions. Generic RPC response
handling must never settle these records: only owner and attempt tokens can.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

_TERMINAL = {"COMPLETED", "TERMINAL_ERROR", "ATTACHMENT_REATTACH_REQUIRED", "UNKNOWN_OUTCOME"}
_SAFE_ACTION = {
    "ATTACHMENT_REATTACH_REQUIRED": "reattach_then_continue",
    "UNKNOWN_OUTCOME": "check_transcript_resume_or_start_new",
    "TERMINAL_ERROR": "start_new_submission",
}
_PAYLOAD_VERSION = 1
_PAYLOAD_FIELDS = frozenset({
    "text", "display_kind", "queued", "interrupted", "truncation_target",
    "truncation_consent", "attachments",
})
_ATTACHMENT_FIELDS = frozenset({"identity", "version", "order", "status"})
_DISPLAY_KINDS = frozenset({"normal", "voice", "continuation"})
_TRUNCATION_TARGETS = frozenset({"message", "prompt", "attachment"})
_ATTACHMENT_STATUSES = frozenset({"ready", "missing", "reattach_required"})
_SAFE_TERMINAL_FIELDS = frozenset({"layer", "code", "retryable", "safe_action"})
_SAFE_LAYERS = frozenset({"admission", "attachment", "provider", "recovery"})
_SAFE_CODES = frozenset({"storage_failed", "provider_failed", "attachment_reattach_required", "unknown_outcome"})
_SAFE_ACTIONS = frozenset(_SAFE_ACTION.values())
# Durable attachment strings are opaque local metadata, never renderer handles:
# identity is a 1-64 ASCII identifier; version is a v-prefixed numeric version.
_OPAQUE_ATTACHMENT_IDENTITY = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")
_OPAQUE_ATTACHMENT_VERSION = re.compile(r"v[0-9]+(?:\.[0-9]+){0,3}\Z")
_CAPABILITY_WORDS = frozenset({"bearer", "token", "secret", "authorization", "credential", "capability"})


def _is_safe_attachment_opaque(value: Any, pattern: re.Pattern[str]) -> bool:
    """Accept only bounded non-capability local attachment metadata."""
    return (
        isinstance(value, str)
        and pattern.fullmatch(value) is not None
        and not any(word in value.casefold() for word in _CAPABILITY_WORDS)
    )


def _canonical_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Persist only versioned replay semantics, never renderer capabilities."""
    if not isinstance(payload, dict) or set(payload) - _PAYLOAD_FIELDS:
        raise ValueError("ATTACHMENT_REATTACH_REQUIRED")
    text = payload.get("text")
    if not isinstance(text, str):
        raise ValueError("invalid durable prompt payload")
    display_kind = payload.get("display_kind", "normal")
    if display_kind not in _DISPLAY_KINDS:
        raise ValueError("invalid durable prompt payload")
    queued = payload.get("queued", False)
    interrupted = payload.get("interrupted", False)
    truncation_consent = payload.get("truncation_consent", False)
    if any(type(value) is not bool for value in (queued, interrupted, truncation_consent)):
        raise ValueError("invalid durable prompt payload")
    truncation_target = payload.get("truncation_target")
    if truncation_target is not None and truncation_target not in _TRUNCATION_TARGETS:
        raise ValueError("invalid durable prompt payload")
    attachments = payload.get("attachments", [])
    if not isinstance(attachments, list):
        raise ValueError("ATTACHMENT_REATTACH_REQUIRED")
    canonical_attachments = []
    for attachment in attachments:
        if not isinstance(attachment, dict) or set(attachment) != _ATTACHMENT_FIELDS:
            raise ValueError("ATTACHMENT_REATTACH_REQUIRED")
        identity, version, order, status = (
            attachment["identity"], attachment["version"], attachment["order"], attachment["status"]
        )
        if (
            not _is_safe_attachment_opaque(identity, _OPAQUE_ATTACHMENT_IDENTITY)
            or not _is_safe_attachment_opaque(version, _OPAQUE_ATTACHMENT_VERSION)
            or type(order) is not int
            or status not in _ATTACHMENT_STATUSES
        ):
            raise ValueError("ATTACHMENT_REATTACH_REQUIRED")
        canonical_attachments.append({"identity": identity, "version": version, "order": order, "status": status})
    return {
        "payload_version": _PAYLOAD_VERSION,
        "text": text,
        "display_kind": display_kind,
        "queued": queued,
        "interrupted": interrupted,
        "truncation_target": truncation_target,
        "truncation_consent": truncation_consent,
        "attachments": canonical_attachments,
    }


def _safe_terminal_summary(safe_terminal: dict[str, Any] | None) -> dict[str, Any]:
    if safe_terminal is None:
        return {}
    if not isinstance(safe_terminal, dict) or set(safe_terminal) - _SAFE_TERMINAL_FIELDS:
        raise ValueError("invalid durable terminal summary")
    summary = dict(safe_terminal)
    if "layer" in summary and summary["layer"] not in _SAFE_LAYERS:
        raise ValueError("invalid durable terminal summary")
    if "code" in summary and summary["code"] not in _SAFE_CODES:
        raise ValueError("invalid durable terminal summary")
    if "retryable" in summary and type(summary["retryable"]) is not bool:
        raise ValueError("invalid durable terminal summary")
    if "safe_action" in summary and summary["safe_action"] not in _SAFE_ACTIONS:
        raise ValueError("invalid durable terminal summary")
    return summary


class SessionSubmissionMixin:
    """SessionDB-owned receipt/work state machine; never uses state_meta."""

    @staticmethod
    def _submission_ack(submission_id: str, state: str) -> dict[str, Any]:
        return {"submission_id": submission_id, "contract_version": "1",
                "durable_admission_status": "accepted", "invocation_status": state.lower(),
                "safe_terminal_action": _SAFE_ACTION.get(state)}

    def read_prompt_submission_receipt(self, *, session_id: str, submission_id: str,
                                       semantic_fingerprint: str) -> dict[str, Any] | None:
        """Read an existing receipt without creating work or mutating runtime state."""
        with self._lock:
            row = self._conn.execute(
                "SELECT r.semantic_fingerprint, r.work_id, w.state "
                "FROM prompt_submission_receipts r JOIN prompt_accepted_work w ON w.work_id=r.work_id "
                "WHERE r.session_id=? AND r.submission_id=?",
                (session_id, submission_id),
            ).fetchone()
        if row is None:
            return None
        if row["semantic_fingerprint"] != semantic_fingerprint:
            return {"conflict": True}
        return {"conflict": False, "work_id": row["work_id"],
                "ack": self._submission_ack(submission_id, row["state"])}

    def create_or_read_prompt_submission(self, *, session_id: str, submission_id: str,
                                         contract_version: str, semantic_fingerprint: str,
                                         payload: dict[str, Any], before_commit=None) -> dict[str, Any]:
        """One BEGIN IMMEDIATE create-or-read transaction before runtime mutation."""
        if contract_version != "1":
            raise ValueError("unsupported prompt submission contract")
        payload_wire = json.dumps(_canonical_payload(payload), sort_keys=True, separators=(",", ":"))
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

    def claim_prompt_submission_work(self, work_id: str, *, owner_token: str,
                                     session_id: str | None = None,
                                     lease_seconds: float = 30) -> dict[str, Any] | None:
        """Atomically claim and return only the claimed row's canonical payload."""
        now = time.time()
        def write(conn):
            predicate = "work_id=? AND state IN ('ACCEPTED','QUEUED')"
            values: list[Any] = [owner_token, now + lease_seconds, now, work_id]
            if session_id is not None:
                predicate += " AND session_id=?"
                values.append(session_id)
            changed = conn.execute(
                f"UPDATE prompt_accepted_work SET state='DISPATCHING', owner_token=?, "
                f"owner_generation=owner_generation+1, lease_expires_at=?, updated_at=? "
                f"WHERE {predicate}", values,
            ).rowcount
            if changed != 1:
                return None
            row = conn.execute(
                "SELECT work_id, session_id, payload_json, state, owner_generation "
                "FROM prompt_accepted_work WHERE work_id=? AND owner_token=?", (work_id, owner_token)
            ).fetchone()
            if row is None:
                return None
            return {
                "work_id": row["work_id"], "session_id": row["session_id"],
                "payload": json.loads(row["payload_json"]), "state": row["state"],
                "owner_generation": row["owner_generation"],
            }
        return self._execute_write(write)

    def release_prompt_submission_dispatch(self, work_id: str, *, owner_token: str) -> bool:
        """Release only this owner's uninvoked DISPATCHING claim back to ACCEPTED."""
        return self._execute_write(lambda conn: conn.execute(
            "UPDATE prompt_accepted_work SET state='ACCEPTED', owner_token=NULL, "
            "lease_expires_at=NULL, updated_at=? WHERE work_id=? AND owner_token=? "
            "AND state='DISPATCHING' AND invocation_attempt_token IS NULL",
            (time.time(), work_id, owner_token),
        ).rowcount == 1)

    def mark_prompt_submission_invoking(self, work_id: str, *, owner_token: str, attempt_token: str, lease_seconds: float = 30) -> dict[str, Any] | None:
        now = time.time()
        def write(conn):
            changed = conn.execute("UPDATE prompt_accepted_work SET state='INVOKING', invocation_attempt_token=?, invocation_attempt_no=invocation_attempt_no+1, lease_expires_at=?, updated_at=? WHERE work_id=? AND owner_token=? AND state='DISPATCHING'", (attempt_token, now + lease_seconds, now, work_id, owner_token)).rowcount
            return {"state": "INVOKING", "invocation_attempt_token": attempt_token} if changed == 1 else None
        return self._execute_write(write)

    def mark_prompt_submission_running(self, work_id: str, *, owner_token: str,
                                       attempt_token: str, lease_seconds: float = 30) -> bool:
        now = time.time()
        return self._execute_write(lambda conn: conn.execute(
            "UPDATE prompt_accepted_work SET state='RUNNING', lease_expires_at=?, updated_at=? "
            "WHERE work_id=? AND owner_token=? AND invocation_attempt_token=? AND state='INVOKING'",
            (now + lease_seconds, now, work_id, owner_token, attempt_token),
        ).rowcount == 1)

    def renew_prompt_submission_lease(self, work_id: str, *, owner_token: str,
                                      attempt_token: str, lease_seconds: float = 30) -> bool:
        """Renew only the exact invocation owner/attempt currently at the boundary."""
        now = time.time()
        return self._execute_write(lambda conn: conn.execute(
            "UPDATE prompt_accepted_work SET lease_expires_at=?, updated_at=? "
            "WHERE work_id=? AND owner_token=? AND invocation_attempt_token=? "
            "AND state IN ('INVOKING','RUNNING')",
            (now + lease_seconds, now, work_id, owner_token, attempt_token),
        ).rowcount == 1)

    def complete_prompt_submission_work(self, work_id: str, *, owner_token: str, attempt_token: str, state: str, safe_terminal: dict[str, Any] | None = None) -> bool:
        if state not in _TERMINAL: raise ValueError("invalid durable terminal state")
        summary = _safe_terminal_summary(safe_terminal)
        return self._execute_write(lambda conn: conn.execute("UPDATE prompt_accepted_work SET state=?, safe_terminal_json=?, owner_token=NULL, lease_expires_at=NULL, updated_at=? WHERE work_id=? AND owner_token=? AND invocation_attempt_token=? AND state IN ('INVOKING','RUNNING')", (state, json.dumps(summary, sort_keys=True), time.time(), work_id, owner_token, attempt_token)).rowcount == 1)

    def recover_prompt_submission_work(self, *, session_id: str | None = None,
                                       now: float | None = None,
                                       live_owner_witnesses: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        """Recover a single session's work; invocation intent never auto-replays."""
        now = time.time() if now is None else now
        witnesses = set()
        for witness in live_owner_witnesses or []:
            if not isinstance(witness, dict) or set(witness) != {
                "work_id", "owner_generation", "owner_token", "invocation_attempt_token"
            }:
                raise ValueError("invalid recovered live-owner witness")
            work_id = witness["work_id"]
            generation = witness["owner_generation"]
            owner_token = witness["owner_token"]
            attempt_token = witness["invocation_attempt_token"]
            if (not isinstance(work_id, str) or type(generation) is not int
                    or not isinstance(owner_token, str) or not isinstance(attempt_token, str)):
                raise ValueError("invalid recovered live-owner witness")
            witnesses.add((work_id, generation, owner_token, attempt_token))

        def write(conn):
            scope = ""
            scope_values: list[Any] = []
            if session_id is not None:
                scope = " AND session_id=?"
                scope_values.append(session_id)
            invoking = conn.execute(
                "SELECT work_id, owner_generation, owner_token, invocation_attempt_token, lease_expires_at "
                "FROM prompt_accepted_work WHERE state IN ('INVOKING','RUNNING')" + scope,
                scope_values,
            ).fetchall()
            unknown_ids = [
                row["work_id"] for row in invoking
                if row["lease_expires_at"] is None or row["lease_expires_at"] <= now
                or (row["work_id"], row["owner_generation"], row["owner_token"], row["invocation_attempt_token"]) not in witnesses
            ]
            if unknown_ids:
                placeholders = ",".join("?" for _ in unknown_ids)
                conn.execute(
                    f"UPDATE prompt_accepted_work SET state='UNKNOWN_OUTCOME', owner_token=NULL, "
                    f"lease_expires_at=NULL, safe_terminal_json=?, updated_at=? "
                    f"WHERE work_id IN ({placeholders}) AND state IN ('INVOKING','RUNNING')",
                    (json.dumps({"code": "unknown_outcome", "safe_action": _SAFE_ACTION["UNKNOWN_OUTCOME"]}, sort_keys=True), now, *unknown_ids),
                )
            conn.execute(
                "UPDATE prompt_accepted_work SET state='ACCEPTED', owner_token=NULL, "
                "lease_expires_at=NULL, updated_at=? WHERE state='DISPATCHING' "
                "AND lease_expires_at <= ?" + scope,
                (now, now, *scope_values),
            )
            return [dict(r) for r in conn.execute(
                "SELECT work_id, session_id, submission_id, state FROM prompt_accepted_work "
                "WHERE state IN ('ACCEPTED','QUEUED')" + (" AND session_id=?" if session_id is not None else "") +
                " ORDER BY created_at, work_id", scope_values,
            ).fetchall()]
        return self._execute_write(write)

    def has_unresolved_prompt_submission_work(self, session_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM prompt_accepted_work WHERE session_id=? "
                "AND state NOT IN ('COMPLETED','TERMINAL_ERROR','ATTACHMENT_REATTACH_REQUIRED','UNKNOWN_OUTCOME') LIMIT 1",
                (session_id,),
            ).fetchone()
        return row is not None

    def prompt_submission_counts(self, session_id: str) -> tuple[int, int]:
        with self._lock:
            a = self._conn.execute("SELECT COUNT(*) FROM prompt_submission_receipts WHERE session_id=?", (session_id,)).fetchone()[0]
            b = self._conn.execute("SELECT COUNT(*) FROM prompt_accepted_work WHERE session_id=?", (session_id,)).fetchone()[0]
        return int(a), int(b)
