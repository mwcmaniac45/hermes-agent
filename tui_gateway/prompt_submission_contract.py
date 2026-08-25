"""
Versioned durable-admission contract for prompt submission recovery.

Contract version: 1
Execution guarantee: durable_acceptance_admission_only

'accepted' status means the owning profile state.db atomically committed
a receipt and replayable accepted-work row.  It does NOT mean the provider
ran, a user turn persisted, or any external effect occurred.  No API, UX,
documentation, or test name may claim exactly-once provider execution.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence

CONTRACT_VERSION = "1"
_EXECUTION_GUARANTEE = "durable_acceptance_admission_only"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DurableAdmissionStatus(str, Enum):
    ACCEPTED = "accepted"
    CONFLICT = "conflict"


class InvocationStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    DISPATCHING = "dispatching"
    INVOKING = "invoking"
    RUNNING = "running"
    COMPLETED = "completed"
    TERMINAL_ERROR = "terminal_error"
    ATTACHMENT_REATTACH_REQUIRED = "attachment_reattach_required"
    UNKNOWN_OUTCOME = "unknown_outcome"


class SafeTerminalAction(str, Enum):
    CHECK_TRANSCRIPT_RESUME_OR_START_NEW = "check_transcript_resume_or_start_new"
    REATTACH_THEN_CONTINUE = "reattach_then_continue"
    START_NEW_SUBMISSION = "start_new_submission"


class DisplayKind(str, Enum):
    NORMAL = "normal"
    VOICE = "voice"
    CONTINUATION = "continuation"


# ---------------------------------------------------------------------------
# Attachment identity (non-capability — identity/version/order only)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AttachmentIdentity:
    """Non-sensitive attachment identity for fingerprinting only.

    This MUST NOT contain paths, previewUrls, bearer tokens, or any
    renderer-readable capability.  The canonical dict is used only in the
    semantic fingerprint, never in logs.
    """
    order: int
    identity: str
    version: str

    def to_canonical_dict(self) -> dict:
        """Return sorted-key dict for deterministic fingerprinting."""
        return {
            "identity": self.identity,
            "order": self.order,
            "version": self.version,
        }


# ---------------------------------------------------------------------------
# Frozen contract dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PromptSubmissionV1:
    """Immutable replayable request semantics — spec-01 contract version 1.

    submission_id is an opaque UUID (crypto.randomUUID() on the client).
    It is NEVER a hash of prompt content.
    """
    submission_id: str
    contract_version: str
    semantic_fingerprint: str

    def __post_init__(self) -> None:
        if not self.submission_id:
            raise ValueError("submission_id is required")
        if not self.contract_version:
            raise ValueError("contract_version is required")
        if not self.semantic_fingerprint:
            raise ValueError("semantic_fingerprint is required")
        if self.contract_version != CONTRACT_VERSION:
            raise ValueError(
                f"contract_version mismatch: expected {CONTRACT_VERSION!r}, "
                f"got {self.contract_version!r}"
            )


@dataclass(frozen=True)
class PromptSubmissionAckV1:
    """Immutable acknowledgment — durable admission receipt only.

    'accepted' means the profile state.db committed the receipt row.
    It does NOT mean the provider ran or a user turn persisted.
    """
    submission_id: str
    contract_version: str
    semantic_fingerprint: str
    durable_admission_status: DurableAdmissionStatus
    invocation_status: InvocationStatus
    safe_terminal_action: Optional[SafeTerminalAction]

    def to_safe_log_dict(self) -> dict:
        """Return the only approved log surface for this ack.

        Contains ONLY: submission_id, contract_version,
        durable_admission_status, invocation_status, safe_terminal_action.
        Never: text, paths, error bodies, fingerprint preimage, raw errors.
        """
        return {
            "submission_id": self.submission_id,
            "contract_version": self.contract_version,
            "durable_admission_status": self.durable_admission_status.value,
            "invocation_status": self.invocation_status.value,
            "safe_terminal_action": (
                self.safe_terminal_action.value
                if self.safe_terminal_action is not None
                else None
            ),
        }


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------

def compute_text_sha256(text: str) -> str:
    """Hash raw prompt text.  The hash — not the text — enters the canonical object."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_canonical_semantic_object(
    *,
    text_sha256: str,
    display_kind: DisplayKind | str,
    queued: bool,
    interrupted: bool,
    surface: str,
    truncation_target: Optional[str],
    truncation_consent: bool,
    attachments: Sequence[AttachmentIdentity],
    display_metadata: Optional[Any] = None,
    replay_controls: Optional[Any] = None,
) -> dict:
    """Build the deterministic canonical object used for fingerprinting.

    Raw text is excluded.  Attachments are sorted by (order, identity, version)
    so insertion order cannot affect the fingerprint.
    Keys are in alphabetical order to match sorted-key JSON serialization.
    """
    kind_value = (
        display_kind.value
        if isinstance(display_kind, DisplayKind)
        else str(display_kind)
    )
    sorted_attachments = sorted(
        attachments, key=lambda a: (a.order, a.identity, a.version)
    )
    return {
        "attachments": [a.to_canonical_dict() for a in sorted_attachments],
        "display_kind": kind_value,
        "display_metadata": display_metadata,
        "interrupted": interrupted,
        "queued": queued,
        "replay_controls": replay_controls,
        "surface": surface,
        "text_sha256": text_sha256,
        "truncation_consent": truncation_consent,
        "truncation_target": truncation_target,
    }


def compute_semantic_fingerprint(canonical_object: dict) -> str:
    """SHA-256 of deterministically serialized canonical object.

    sort_keys=True ensures key order never affects the fingerprint.
    separators omit whitespace for a compact, unambiguous encoding.
    The resulting hex digest is the only fingerprint representation.
    Raw text, paths, or error bodies MUST NOT appear in canonical_object.
    """
    serialized = json.dumps(
        canonical_object, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_REQUIRED_REQUEST_FIELDS = frozenset(
    {"submission_id", "contract_version", "semantic_fingerprint"}
)

_REQUIRED_ACK_FIELDS = frozenset(
    {
        "submission_id",
        "contract_version",
        "semantic_fingerprint",
        "durable_admission_status",
        "invocation_status",
        "safe_terminal_action",
    }
)


def validate_request_from_dict(data: dict) -> PromptSubmissionV1:
    """Validate and construct a PromptSubmissionV1 from a raw dict.

    Raises ValueError on missing required fields or contract version mismatch.
    Does NOT mutate any runtime state.
    """
    missing = _REQUIRED_REQUEST_FIELDS - data.keys()
    if missing:
        raise ValueError(f"Missing required request fields: {sorted(missing)}")
    return PromptSubmissionV1(
        submission_id=str(data["submission_id"]),
        contract_version=str(data["contract_version"]),
        semantic_fingerprint=str(data["semantic_fingerprint"]),
    )


def validate_ack_from_dict(data: dict) -> PromptSubmissionAckV1:
    """Validate and construct a PromptSubmissionAckV1 from a raw dict.

    Raises ValueError on missing required fields or unknown enum values.
    Does NOT mutate any runtime state.
    """
    missing = _REQUIRED_ACK_FIELDS - data.keys()
    if missing:
        raise ValueError(f"Missing required ack fields: {sorted(missing)}")

    raw_action = data["safe_terminal_action"]
    safe_action: Optional[SafeTerminalAction] = (
        SafeTerminalAction(raw_action) if raw_action is not None else None
    )

    return PromptSubmissionAckV1(
        submission_id=str(data["submission_id"]),
        contract_version=str(data["contract_version"]),
        semantic_fingerprint=str(data["semantic_fingerprint"]),
        durable_admission_status=DurableAdmissionStatus(data["durable_admission_status"]),
        invocation_status=InvocationStatus(data["invocation_status"]),
        safe_terminal_action=safe_action,
    )
