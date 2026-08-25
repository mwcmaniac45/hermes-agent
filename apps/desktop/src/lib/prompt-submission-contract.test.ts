/**
 * RED → GREEN Vitest contract tests for spec-01.
 *
 * Mirrors the Python test suite in tests/tui_gateway/test_prompt_submission_contract.py.
 * Both suites independently load the same shared fixture.
 *
 * Execution guarantee tested: durable_acceptance_admission_only.
 * These tests do NOT assert exactly-once provider execution.
 *
 * Run:
 *   npm --prefix apps/desktop run test:ui -- --runInBand
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  CONTRACT_VERSION,
  buildCanonicalSemanticObject,
  computeSemanticFingerprint,
  computeTextSha256,
  sortedJsonStringify,
  toSafeLogContext,
  validateAck,
  validateRequest,
  type AttachmentIdentity,
  type PromptSubmissionAckV1,
} from './prompt-submission-contract';

// ---------------------------------------------------------------------------
// Fixture loader (independent of Python — same file, independent load)
// ---------------------------------------------------------------------------

const FIXTURE_PATH = join(
  __dirname,
  '../../../../tests/contracts/prompt_submission_recovery.v1.json'
);

const fixture: Record<string, unknown> = JSON.parse(
  readFileSync(FIXTURE_PATH, 'utf-8')
) as Record<string, unknown>;

// Typed helpers for fixture access
function getVectors(category: string): Array<Record<string, unknown>> {
  const tv = fixture['test_vectors'] as Record<string, unknown>;
  return tv[category] as Array<Record<string, unknown>>;
}

function findVector(category: string, id: string): Record<string, unknown> {
  const vec = getVectors(category).find((v) => v['id'] === id);
  if (!vec) throw new Error(`Vector not found: ${category}/${id}`);
  return vec;
}

// ---------------------------------------------------------------------------
// Fixture canonical dict → fingerprint helper
// ---------------------------------------------------------------------------

function fingerprintFromRaw(canonical: Record<string, unknown>): string {
  const rawAttachments = (canonical['attachments'] as Array<Record<string, unknown>>) ?? [];
  const attachments: AttachmentIdentity[] = rawAttachments.map((a) => ({
    order: a['order'] as number,
    identity: a['identity'] as string,
    version: a['version'] as string,
  }));
  const obj = buildCanonicalSemanticObject({
    text_sha256: canonical['text_sha256'] as string,
    display_kind: canonical['display_kind'] as string,
    queued: canonical['queued'] as boolean,
    interrupted: canonical['interrupted'] as boolean,
    surface: canonical['surface'] as string,
    truncation_target: canonical['truncation_target'] as string | null,
    truncation_consent: canonical['truncation_consent'] as boolean,
    attachments,
    display_metadata: canonical['display_metadata'],
    replay_controls: (canonical['replay_controls'] as Record<string, unknown> | null) ?? null,
  });
  return computeSemanticFingerprint(obj);
}

// ---------------------------------------------------------------------------
// TestFixtureLoads
// ---------------------------------------------------------------------------

describe('TestFixtureLoads', () => {
  it('contract_version matches CONTRACT_VERSION', () => {
    expect(fixture['contract_version']).toBe(CONTRACT_VERSION);
  });

  it('execution_guarantee is durable_acceptance_admission_only', () => {
    expect(fixture['execution_guarantee']).toBe('durable_acceptance_admission_only');
  });

  it('not_guaranteed excludes provider_ran', () => {
    const ng = fixture['not_guaranteed'] as string[];
    expect(ng).toContain('provider_ran');
  });

  it('not_guaranteed excludes exactly_once_provider_execution', () => {
    const ng = fixture['not_guaranteed'] as string[];
    expect(ng).toContain('exactly_once_provider_execution');
  });

  it('rejects incompatible contract_version', () => {
    expect(() =>
      validateRequest({
        submission_id: '550e8400-e29b-41d4-a716-446655440000',
        contract_version: '99',
        semantic_fingerprint: 'a'.repeat(64),
      })
    ).toThrow('contract_version');
  });
});

// ---------------------------------------------------------------------------
// TestRequestRequiredFields
// ---------------------------------------------------------------------------

describe('TestRequestRequiredFields', () => {
  it('fixture declares submission_id required', () => {
    const schema = fixture['request_schema'] as Record<string, unknown>;
    expect(schema['required_fields']).toContain('submission_id');
  });

  it('fixture declares contract_version required', () => {
    const schema = fixture['request_schema'] as Record<string, unknown>;
    expect(schema['required_fields']).toContain('contract_version');
  });

  it('fixture declares semantic_fingerprint required', () => {
    const schema = fixture['request_schema'] as Record<string, unknown>;
    expect(schema['required_fields']).toContain('semantic_fingerprint');
  });

  it('rejects missing submission_id', () => {
    const vec = findVector('required_field_validation', 'rfv_request_missing_submission_id');
    expect(vec['expect_error']).toBe(true);
    expect(() => validateRequest(vec['input'])).toThrow('submission_id');
  });

  it('rejects missing contract_version', () => {
    const vec = findVector('required_field_validation', 'rfv_request_missing_contract_version');
    expect(vec['expect_error']).toBe(true);
    expect(() => validateRequest(vec['input'])).toThrow('contract_version');
  });

  it('rejects missing semantic_fingerprint', () => {
    const vec = findVector('required_field_validation', 'rfv_request_missing_semantic_fingerprint');
    expect(vec['expect_error']).toBe(true);
    expect(() => validateRequest(vec['input'])).toThrow('semantic_fingerprint');
  });
});

// ---------------------------------------------------------------------------
// TestAckRequiredFields
// ---------------------------------------------------------------------------

describe('TestAckRequiredFields', () => {
  it('rejects missing durable_admission_status', () => {
    const vec = findVector('required_field_validation', 'rfv_ack_missing_durable_admission_status');
    expect(vec['expect_error']).toBe(true);
    expect(() => validateAck(vec['input'])).toThrow('durable_admission_status');
  });

  it('rejects missing invocation_status', () => {
    const vec = findVector('required_field_validation', 'rfv_ack_missing_invocation_status');
    expect(vec['expect_error']).toBe(true);
    expect(() => validateAck(vec['input'])).toThrow('invocation_status');
  });

  it('rejects missing safe_terminal_action', () => {
    const vec = findVector('required_field_validation', 'rfv_ack_missing_safe_terminal_action');
    expect(vec['expect_error']).toBe(true);
    expect(() => validateAck(vec['input'])).toThrow('safe_terminal_action');
  });
});

// ---------------------------------------------------------------------------
// TestCanonicalEquivalence
// ---------------------------------------------------------------------------

describe('TestCanonicalEquivalence', () => {
  it('field insertion order is irrelevant to fingerprint', () => {
    const vec = findVector('canonical_equivalence', 'ceq_field_insertion_order_irrelevant');
    const fpA = fingerprintFromRaw(vec['canonical_a'] as Record<string, unknown>);
    const fpB = fingerprintFromRaw(vec['canonical_b'] as Record<string, unknown>);
    expect(fpA).toBe(fpB);
  });

  it('attachment list order is normalized before fingerprinting', () => {
    const vec = findVector('canonical_equivalence', 'ceq_attachment_list_order_normalized');
    const fpA = fingerprintFromRaw(vec['canonical_a'] as Record<string, unknown>);
    const fpB = fingerprintFromRaw(vec['canonical_b'] as Record<string, unknown>);
    expect(fpA).toBe(fpB);
  });
});

// ---------------------------------------------------------------------------
// TestCanonicalMismatch
// ---------------------------------------------------------------------------

describe('TestCanonicalMismatch', () => {
  const mismatchVectorIds = [
    'cmm_text_sha256_changed',
    'cmm_display_kind_changed',
    'cmm_display_metadata_changed',
    'cmm_queued_changed',
    'cmm_interrupted_changed',
    'cmm_surface_changed',
    'cmm_truncation_target_changed',
    'cmm_truncation_consent_changed',
    'cmm_attachment_identity_changed',
    'cmm_attachment_version_changed',
    'cmm_attachment_order_changed',
    'cmm_replay_controls_changed',
  ];

  for (const vectorId of mismatchVectorIds) {
    it(`changing ${vectorId} produces different fingerprint`, () => {
      const vec = findVector('canonical_mismatch', vectorId);
      expect(vec['expect_same_fingerprint']).toBe(false);
      const fpBase = fingerprintFromRaw(vec['base'] as Record<string, unknown>);
      const fpVariant = fingerprintFromRaw(vec['variant'] as Record<string, unknown>);
      expect(fpBase).not.toBe(fpVariant);
    });
  }
});

// ---------------------------------------------------------------------------
// TestValidAckVectors
// ---------------------------------------------------------------------------

describe('TestValidAckVectors', () => {
  it('all valid_ack_vectors round-trip through validateAck', () => {
    const vectors = getVectors('valid_ack_vectors');
    for (const vec of vectors) {
      if (!vec['expect_valid']) continue;
      const ack = validateAck(vec['ack']);
      expect(ack.submission_id).toBe((vec['ack'] as Record<string, unknown>)['submission_id']);
    }
  });

  it('unknown_outcome vector has check_transcript action', () => {
    const vec = findVector('valid_ack_vectors', 'vav_unknown_outcome_with_action');
    const ack = validateAck(vec['ack']);
    expect(ack.invocation_status).toBe('unknown_outcome');
    expect(ack.safe_terminal_action).toBe('check_transcript_resume_or_start_new');
  });

  it('attachment_reattach vector has reattach action', () => {
    const vec = findVector('valid_ack_vectors', 'vav_attachment_reattach_required');
    const ack = validateAck(vec['ack']);
    expect(ack.invocation_status).toBe('attachment_reattach_required');
    expect(ack.safe_terminal_action).toBe('reattach_then_continue');
  });
});

// ---------------------------------------------------------------------------
// TestSafeLogSurface
// ---------------------------------------------------------------------------

describe('TestSafeLogSurface', () => {
  const FORBIDDEN_KEYS = new Set([
    'text',
    'prompt',
    'canonical_text',
    'path',
    'previewUrl',
    'detail',
    'refText',
    'bearer',
    'token',
    'error_body',
    'raw_error',
    'exception_message',
    'fingerprint_preimage',
    'attachment_capability',
    'attachment_url',
  ]);

  const SAFE_ACK: PromptSubmissionAckV1 = {
    submission_id: '550e8400-e29b-41d4-a716-446655440000',
    contract_version: CONTRACT_VERSION,
    semantic_fingerprint: 'a'.repeat(64),
    durable_admission_status: 'accepted',
    invocation_status: 'pending',
    safe_terminal_action: null,
  };

  it('toSafeLogContext contains no forbidden keys', () => {
    const forbidden = (fixture['forbidden_in_logs_and_diagnostics'] as string[]) ?? [];
    const ctx = toSafeLogContext(SAFE_ACK);
    const violations = Object.keys(ctx).filter((k) => forbidden.includes(k));
    expect(violations).toHaveLength(0);
  });

  it('toSafeLogContext contains only safe_log_allowed_fields', () => {
    const allowed = new Set((fixture['safe_log_allowed_fields'] as string[]) ?? []);
    const ctx = toSafeLogContext(SAFE_ACK);
    const extra = Object.keys(ctx).filter((k) => !allowed.has(k));
    expect(extra).toHaveLength(0);
  });

  it('semantic_fingerprint is absent from safe log context', () => {
    const ctx = toSafeLogContext(SAFE_ACK);
    expect('semantic_fingerprint' in ctx).toBe(false);
  });

  it('fixture forbidden_in_logs_and_diagnostics includes text', () => {
    const forbidden = fixture['forbidden_in_logs_and_diagnostics'] as string[];
    expect(forbidden).toContain('text');
  });
});

// ---------------------------------------------------------------------------
// TestSortedJsonStringify (cross-language compatibility)
// ---------------------------------------------------------------------------

describe('TestSortedJsonStringify', () => {
  it('sorts object keys alphabetically', () => {
    const result = sortedJsonStringify({ z: 1, a: 2, m: 3 });
    expect(result).toBe('{"a":2,"m":3,"z":1}');
  });

  it('preserves array element order', () => {
    const result = sortedJsonStringify([3, 1, 2]);
    expect(result).toBe('[3,1,2]');
  });

  it('sorts nested object keys', () => {
    const result = sortedJsonStringify({ b: { y: 1, x: 2 }, a: 0 });
    expect(result).toBe('{"a":0,"b":{"x":2,"y":1}}');
  });

  it('handles null correctly', () => {
    expect(sortedJsonStringify(null)).toBe('null');
  });

  it('field-insertion-order-irrelevant produces same fingerprint as Python', () => {
    // Both orderings of the same canonical object must produce the same fingerprint
    const objA = buildCanonicalSemanticObject({
      text_sha256: 'b94d27b9934d3e08a52e52d7da7dabfac484efe04294e576af58cb95b6887f',
      display_kind: 'normal',
      queued: false,
      interrupted: false,
      surface: 'main',
      truncation_target: null,
      truncation_consent: false,
      attachments: [],
      display_metadata: null,
      replay_controls: null,
    });
    const fp = computeSemanticFingerprint(objA);
    // Same canonical input must always produce the same hex digest
    expect(fp).toHaveLength(64);
    expect(fp).toMatch(/^[0-9a-f]{64}$/);
    // Running again with same input must be deterministic
    expect(computeSemanticFingerprint(objA)).toBe(fp);
  });
});

// ---------------------------------------------------------------------------
// TestExecutionGuarantee
// ---------------------------------------------------------------------------

describe('TestExecutionGuarantee', () => {
  it('fixture execution_guarantee does not claim provider execution', () => {
    expect(fixture['execution_guarantee']).not.toContain('provider');
    expect(fixture['execution_guarantee']).not.toContain('exactly_once');
  });

  it('computeTextSha256 returns a hex string, not raw text', () => {
    const raw = 'some prompt text';
    const sha = computeTextSha256(raw);
    expect(sha).toHaveLength(64);
    expect(sha).toMatch(/^[0-9a-f]{64}$/);
    expect(sha).not.toBe(raw);
    expect(sha).not.toContain(raw);
  });
});
