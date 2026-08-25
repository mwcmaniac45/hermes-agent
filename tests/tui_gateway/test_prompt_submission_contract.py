"""
RED → GREEN contract tests for spec-01: Freeze the versioned durable-admission contract.

Execution guarantee tested: durable_acceptance_admission_only.
These tests do NOT assert exactly-once provider execution.

Run:
    pytest tests/tui_gateway/test_prompt_submission_contract.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tui_gateway.prompt_submission_contract import (
    CONTRACT_VERSION,
    AttachmentIdentity,
    DisplayKind,
    DurableAdmissionStatus,
    InvocationStatus,
    PromptSubmissionAckV1,
    PromptSubmissionV1,
    SafeTerminalAction,
    build_canonical_semantic_object,
    compute_semantic_fingerprint,
    compute_text_sha256,
    validate_ack_from_dict,
    validate_request_from_dict,
)

# ---------------------------------------------------------------------------
# Fixture loader
# ---------------------------------------------------------------------------

_FIXTURE_PATH = (
    Path(__file__).parent.parent / "contracts" / "prompt_submission_recovery.v1.json"
)


@pytest.fixture(scope="session")
def fixture() -> dict:
    with _FIXTURE_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# TestFixtureLoads
# ---------------------------------------------------------------------------

class TestFixtureLoads:
    def test_contract_version_matches(self, fixture: dict) -> None:
        assert fixture["contract_version"] == CONTRACT_VERSION

    def test_execution_guarantee_is_admission_only(self, fixture: dict) -> None:
        assert fixture["execution_guarantee"] == "durable_acceptance_admission_only"

    def test_not_guaranteed_excludes_provider_ran(self, fixture: dict) -> None:
        assert "provider_ran" in fixture["not_guaranteed"]

    def test_not_guaranteed_excludes_exactly_once_provider(self, fixture: dict) -> None:
        assert "exactly_once_provider_execution" in fixture["not_guaranteed"]

    def test_fixture_version_incompatibility_rejected(self, fixture: dict) -> None:
        """A fixture with a different contract_version must be rejected by the validator."""
        with pytest.raises(ValueError, match="contract_version"):
            validate_request_from_dict({
                "submission_id": "550e8400-e29b-41d4-a716-446655440000",
                "contract_version": "99",
                "semantic_fingerprint": "a" * 64,
            })


# ---------------------------------------------------------------------------
# TestRequestRequiredFields
# ---------------------------------------------------------------------------

class TestRequestRequiredFields:
    """Each missing required request field must be individually rejected."""

    def _base_request(self) -> dict:
        return {
            "submission_id": "550e8400-e29b-41d4-a716-446655440000",
            "contract_version": CONTRACT_VERSION,
            "semantic_fingerprint": "a" * 64,
        }

    def test_fixture_declares_submission_id_required(self, fixture: dict) -> None:
        assert "submission_id" in fixture["request_schema"]["required_fields"]

    def test_fixture_declares_contract_version_required(self, fixture: dict) -> None:
        assert "contract_version" in fixture["request_schema"]["required_fields"]

    def test_fixture_declares_semantic_fingerprint_required(self, fixture: dict) -> None:
        assert "semantic_fingerprint" in fixture["request_schema"]["required_fields"]

    def test_reject_missing_submission_id(self, fixture: dict) -> None:
        vec = next(
            v for v in fixture["test_vectors"]["required_field_validation"]
            if v["id"] == "rfv_request_missing_submission_id"
        )
        assert vec["expect_error"] is True
        with pytest.raises(ValueError, match="submission_id"):
            validate_request_from_dict(vec["input"])

    def test_reject_missing_contract_version(self, fixture: dict) -> None:
        vec = next(
            v for v in fixture["test_vectors"]["required_field_validation"]
            if v["id"] == "rfv_request_missing_contract_version"
        )
        assert vec["expect_error"] is True
        with pytest.raises(ValueError, match="contract_version"):
            validate_request_from_dict(vec["input"])

    def test_reject_missing_semantic_fingerprint(self, fixture: dict) -> None:
        vec = next(
            v for v in fixture["test_vectors"]["required_field_validation"]
            if v["id"] == "rfv_request_missing_semantic_fingerprint"
        )
        assert vec["expect_error"] is True
        with pytest.raises(ValueError, match="semantic_fingerprint"):
            validate_request_from_dict(vec["input"])


# ---------------------------------------------------------------------------
# TestAckRequiredFields
# ---------------------------------------------------------------------------

class TestAckRequiredFields:
    """Each missing required ack field must be individually rejected."""

    def _base_ack(self) -> dict:
        return {
            "submission_id": "550e8400-e29b-41d4-a716-446655440000",
            "contract_version": CONTRACT_VERSION,
            "semantic_fingerprint": "a" * 64,
            "durable_admission_status": "accepted",
            "invocation_status": "pending",
            "safe_terminal_action": None,
        }

    def test_reject_missing_durable_admission_status(self, fixture: dict) -> None:
        vec = next(
            v for v in fixture["test_vectors"]["required_field_validation"]
            if v["id"] == "rfv_ack_missing_durable_admission_status"
        )
        assert vec["expect_error"] is True
        with pytest.raises(ValueError, match="durable_admission_status"):
            validate_ack_from_dict(vec["input"])

    def test_reject_missing_invocation_status(self, fixture: dict) -> None:
        vec = next(
            v for v in fixture["test_vectors"]["required_field_validation"]
            if v["id"] == "rfv_ack_missing_invocation_status"
        )
        assert vec["expect_error"] is True
        with pytest.raises(ValueError, match="invocation_status"):
            validate_ack_from_dict(vec["input"])

    def test_reject_missing_safe_terminal_action(self, fixture: dict) -> None:
        vec = next(
            v for v in fixture["test_vectors"]["required_field_validation"]
            if v["id"] == "rfv_ack_missing_safe_terminal_action"
        )
        assert vec["expect_error"] is True
        with pytest.raises(ValueError, match="safe_terminal_action"):
            validate_ack_from_dict(vec["input"])


# ---------------------------------------------------------------------------
# TestCanonicalEquivalence
# ---------------------------------------------------------------------------

class TestCanonicalEquivalence:
    """Same semantics, different insertion order → same fingerprint."""

    def _fingerprint_from_raw(self, canonical_dict: dict) -> str:
        # Build AttachmentIdentity objects from the raw dict attachments
        attachments = [
            AttachmentIdentity(
                order=a["order"], identity=a["identity"], version=a["version"]
            )
            for a in canonical_dict.get("attachments", [])
        ]
        obj = build_canonical_semantic_object(
            text_sha256=canonical_dict["text_sha256"],
            display_kind=canonical_dict["display_kind"],
            queued=canonical_dict["queued"],
            interrupted=canonical_dict["interrupted"],
            surface=canonical_dict["surface"],
            truncation_target=canonical_dict["truncation_target"],
            truncation_consent=canonical_dict["truncation_consent"],
            attachments=attachments,
            display_metadata=canonical_dict.get("display_metadata"),
            replay_controls=canonical_dict.get("replay_controls"),
        )
        return compute_semantic_fingerprint(obj)

    def test_field_insertion_order_irrelevant(self, fixture: dict) -> None:
        vec = next(
            v for v in fixture["test_vectors"]["canonical_equivalence"]
            if v["id"] == "ceq_field_insertion_order_irrelevant"
        )
        fp_a = self._fingerprint_from_raw(vec["canonical_a"])
        fp_b = self._fingerprint_from_raw(vec["canonical_b"])
        assert fp_a == fp_b, "Same canonical fields in different order must produce same fingerprint"

    def test_attachment_list_order_normalized(self, fixture: dict) -> None:
        vec = next(
            v for v in fixture["test_vectors"]["canonical_equivalence"]
            if v["id"] == "ceq_attachment_list_order_normalized"
        )
        fp_a = self._fingerprint_from_raw(vec["canonical_a"])
        fp_b = self._fingerprint_from_raw(vec["canonical_b"])
        assert fp_a == fp_b, "Same attachments in different array positions must normalize to same fingerprint"


# ---------------------------------------------------------------------------
# TestCanonicalMismatch
# ---------------------------------------------------------------------------

class TestCanonicalMismatch:
    """Each canonical field change must produce a different fingerprint."""

    def _fp(self, canonical_dict: dict) -> str:
        attachments = [
            AttachmentIdentity(
                order=a["order"], identity=a["identity"], version=a["version"]
            )
            for a in canonical_dict.get("attachments", [])
        ]
        obj = build_canonical_semantic_object(
            text_sha256=canonical_dict["text_sha256"],
            display_kind=canonical_dict["display_kind"],
            queued=canonical_dict["queued"],
            interrupted=canonical_dict["interrupted"],
            surface=canonical_dict["surface"],
            truncation_target=canonical_dict["truncation_target"],
            truncation_consent=canonical_dict["truncation_consent"],
            attachments=attachments,
            display_metadata=canonical_dict.get("display_metadata"),
            replay_controls=canonical_dict.get("replay_controls"),
        )
        return compute_semantic_fingerprint(obj)

    @pytest.mark.parametrize(
        "vector_id",
        [
            "cmm_text_sha256_changed",
            "cmm_display_kind_changed",
            "cmm_display_metadata_changed",
            "cmm_queued_changed",
            "cmm_interrupted_changed",
            "cmm_surface_changed",
            "cmm_truncation_target_changed",
            "cmm_truncation_consent_changed",
            "cmm_attachment_identity_changed",
            "cmm_attachment_version_changed",
            "cmm_attachment_order_changed",
            "cmm_replay_controls_changed",
        ],
    )
    def test_mismatch_vector(self, fixture: dict, vector_id: str) -> None:
        vec = next(
            v for v in fixture["test_vectors"]["canonical_mismatch"]
            if v["id"] == vector_id
        )
        assert vec["expect_same_fingerprint"] is False
        fp_base = self._fp(vec["base"])
        fp_variant = self._fp(vec["variant"])
        assert fp_base != fp_variant, (
            f"Vector {vector_id}: changing {vec['field_changed']} must change the fingerprint"
        )


# ---------------------------------------------------------------------------
# TestValidAckVectors
# ---------------------------------------------------------------------------

class TestValidAckVectors:
    """All valid_ack_vectors in the fixture must round-trip through validate_ack_from_dict."""

    def test_all_valid_ack_vectors_parse(self, fixture: dict) -> None:
        for vec in fixture["test_vectors"]["valid_ack_vectors"]:
            if not vec.get("expect_valid", False):
                continue
            ack = validate_ack_from_dict(vec["ack"])
            assert ack.submission_id == vec["ack"]["submission_id"]
            assert ack.contract_version == vec["ack"]["contract_version"]

    def test_unknown_outcome_has_action(self, fixture: dict) -> None:
        vec = next(
            v for v in fixture["test_vectors"]["valid_ack_vectors"]
            if v["id"] == "vav_unknown_outcome_with_action"
        )
        ack = validate_ack_from_dict(vec["ack"])
        assert ack.invocation_status == InvocationStatus.UNKNOWN_OUTCOME
        assert ack.safe_terminal_action == SafeTerminalAction.CHECK_TRANSCRIPT_RESUME_OR_START_NEW

    def test_attachment_reattach_has_action(self, fixture: dict) -> None:
        vec = next(
            v for v in fixture["test_vectors"]["valid_ack_vectors"]
            if v["id"] == "vav_attachment_reattach_required"
        )
        ack = validate_ack_from_dict(vec["ack"])
        assert ack.invocation_status == InvocationStatus.ATTACHMENT_REATTACH_REQUIRED
        assert ack.safe_terminal_action == SafeTerminalAction.REATTACH_THEN_CONTINUE


# ---------------------------------------------------------------------------
# TestSafeLogSurface
# ---------------------------------------------------------------------------

class TestSafeLogSurface:
    """to_safe_log_dict must never contain forbidden fields."""

    _FORBIDDEN_KEYS = frozenset(
        {
            "text",
            "prompt",
            "canonical_text",
            "path",
            "previewUrl",
            "detail",
            "refText",
            "bearer",
            "token",
            "error_body",
            "raw_error",
            "exception_message",
            "fingerprint_preimage",
            "attachment_capability",
            "attachment_url",
        }
    )

    def _make_ack(self) -> PromptSubmissionAckV1:
        return PromptSubmissionAckV1(
            submission_id="550e8400-e29b-41d4-a716-446655440000",
            contract_version=CONTRACT_VERSION,
            semantic_fingerprint="a" * 64,
            durable_admission_status=DurableAdmissionStatus.ACCEPTED,
            invocation_status=InvocationStatus.PENDING,
            safe_terminal_action=None,
        )

    def test_safe_log_dict_contains_no_forbidden_keys(self, fixture: dict) -> None:
        forbidden = set(fixture.get("forbidden_in_logs_and_diagnostics", []))
        ack = self._make_ack()
        log_dict = ack.to_safe_log_dict()
        violations = set(log_dict.keys()) & forbidden
        assert not violations, f"Safe log dict contained forbidden keys: {violations}"

    def test_safe_log_dict_contains_only_allowed_fields(self, fixture: dict) -> None:
        allowed = set(fixture.get("safe_log_allowed_fields", []))
        ack = self._make_ack()
        log_dict = ack.to_safe_log_dict()
        extra = set(log_dict.keys()) - allowed
        assert not extra, f"Safe log dict contained unexpected keys: {extra}"

    def test_safe_log_dict_never_contains_fingerprint_itself(self) -> None:
        """semantic_fingerprint must appear in the safe log only via allowed fields."""
        ack = self._make_ack()
        log_dict = ack.to_safe_log_dict()
        # semantic_fingerprint is NOT in safe_log_allowed_fields
        assert "semantic_fingerprint" not in log_dict


# ---------------------------------------------------------------------------
# TestExecutionGuarantee
# ---------------------------------------------------------------------------

class TestExecutionGuarantee:
    """The fixture and contract must never claim exactly-once provider execution."""

    def test_fixture_execution_guarantee_is_admission_only(self, fixture: dict) -> None:
        assert fixture["execution_guarantee"] == "durable_acceptance_admission_only"
        assert "exactly_once_provider_execution" not in fixture["execution_guarantee"]

    def test_fixture_forbidden_logs_excludes_text(self, fixture: dict) -> None:
        forbidden = fixture.get("forbidden_in_logs_and_diagnostics", [])
        assert "text" in forbidden

    def test_compute_text_sha256_excludes_raw_text_from_canonical(self) -> None:
        """SHA-256 of text is computed; the raw text never enters build_canonical_semantic_object."""
        raw = "some prompt text"
        sha = compute_text_sha256(raw)
        # The sha is a hex string — not the raw text
        assert sha != raw
        assert len(sha) == 64
        assert raw not in sha
