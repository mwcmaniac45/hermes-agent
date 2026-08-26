/**
 * Versioned durable-admission contract for prompt submission recovery.
 *
 * Contract version: 1
 * Execution guarantee: durable_acceptance_admission_only
 *
 * "accepted" status means the owning profile state.db atomically committed
 * a receipt and replayable accepted-work row.  It does NOT mean the provider
 * ran, a user turn persisted, or any external effect occurred.  No API, UX,
 * documentation, or test name may claim exactly-once provider execution.
 */

import { createHash } from 'node:crypto';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

export const CONTRACT_VERSION = '1' as const;
const EXECUTION_GUARANTEE = 'durable_acceptance_admission_only' as const;

// ---------------------------------------------------------------------------
// Enum string unions (kept as plain union types for strict TS + exhaustive checks)
// ---------------------------------------------------------------------------

export type DurableAdmissionStatus = 'accepted' | 'conflict';

export type InvocationStatus =
  | 'pending'
  | 'queued'
  | 'dispatching'
  | 'preparing'
  | 'invoking'
  | 'running'
  | 'completed'
  | 'terminal_error'
  | 'attachment_reattach_required'
  | 'unknown_outcome';

export type SafeTerminalAction =
  | 'check_transcript_resume_or_start_new'
  | 'reattach_then_continue'
  | 'start_new_submission';

export type DisplayKind = 'normal' | 'voice' | 'continuation';

// ---------------------------------------------------------------------------
// Attachment identity (non-capability — order/identity/version only)
// ---------------------------------------------------------------------------

export interface AttachmentIdentity {
  /** Declared position in the submission attachment list. */
  readonly order: number;
  /** Opaque stable identifier for this attachment (not a path or URL). */
  readonly identity: string;
  /** Version discriminator for attachment content. */
  readonly version: string;
}

// ---------------------------------------------------------------------------
// Contract types
// ---------------------------------------------------------------------------

export interface PromptSubmissionV1 {
  /** Opaque UUID from crypto.randomUUID(); never a hash of prompt content. */
  readonly submission_id: string;
  readonly contract_version: typeof CONTRACT_VERSION;
  /** SHA-256 hex of sorted-key JSON canonical object; raw text excluded. */
  readonly semantic_fingerprint: string;
}

export interface PromptSubmissionAckV1 {
  readonly submission_id: string;
  readonly contract_version: string;
  readonly semantic_fingerprint: string;
  readonly durable_admission_status: DurableAdmissionStatus;
  readonly invocation_status: InvocationStatus;
  /** Null when no action is required; non-null for terminal/ambiguous states. */
  readonly safe_terminal_action: SafeTerminalAction | null;
}

// ---------------------------------------------------------------------------
// Canonical semantic object
// ---------------------------------------------------------------------------

export interface ReplayControls {
  readonly allow_retry_after_unknown_outcome?: boolean;
  [key: string]: unknown;
}

export interface CanonicalSemanticObject {
  readonly attachments: ReadonlyArray<{
    readonly identity: string;
    readonly order: number;
    readonly version: string;
  }>;
  readonly display_kind: string;
  readonly display_metadata: unknown;
  readonly interrupted: boolean;
  readonly queued: boolean;
  readonly replay_controls: ReplayControls | null;
  readonly surface: string;
  readonly text_sha256: string;
  readonly truncation_consent: boolean;
  readonly truncation_target: string | null;
}

// ---------------------------------------------------------------------------
// Safe log surface
// ---------------------------------------------------------------------------

export interface SafeLogContext {
  readonly submission_id: string;
  readonly contract_version: string;
  readonly durable_admission_status: DurableAdmissionStatus;
  readonly invocation_status: InvocationStatus;
  readonly safe_terminal_action: SafeTerminalAction | null;
}

// ---------------------------------------------------------------------------
// Sorted-key JSON serialization (produces identical output to Python's sort_keys=True)
// ---------------------------------------------------------------------------

/**
 * Recursively serialize a value to JSON with object keys sorted.
 * Arrays preserve their element order.
 * Produces output identical to Python's json.dumps(sort_keys=True, separators=(',', ':'))
 */
export function sortedJsonStringify(value: unknown): string {
  if (value === null || value === undefined) {return JSON.stringify(value);}

  if (typeof value !== 'object') {return JSON.stringify(value);}

  if (Array.isArray(value)) {
    return '[' + value.map(sortedJsonStringify).join(',') + ']';
  }

  const obj = value as Record<string, unknown>;
  const sortedKeys = Object.keys(obj).sort();

  const pairs = sortedKeys.map(
    (k) => JSON.stringify(k) + ':' + sortedJsonStringify(obj[k])
  );

  return '{' + pairs.join(',') + '}';
}

// ---------------------------------------------------------------------------
// Hashing
// ---------------------------------------------------------------------------

/**
 * Hash raw prompt text.  The hash — not the text — enters the canonical object.
 * Raw prompt text must never appear in logs, diagnostics, or persisted contract records.
 */
export function computeTextSha256(text: string): string {
  return createHash('sha256').update(text, 'utf8').digest('hex');
}

// ---------------------------------------------------------------------------
// Canonicalization
// ---------------------------------------------------------------------------

/**
 * Build the deterministic canonical object used for fingerprinting.
 *
 * Attachments are sorted by (order, identity, version) so insertion order
 * cannot affect the fingerprint.  Keys are in alphabetical order.
 * Raw text is excluded — only text_sha256 appears.
 */
export function buildCanonicalSemanticObject(params: {
  text_sha256: string;
  display_kind: DisplayKind | string;
  queued: boolean;
  interrupted: boolean;
  surface: string;
  truncation_target: string | null;
  truncation_consent: boolean;
  attachments: ReadonlyArray<AttachmentIdentity>;
  display_metadata?: unknown;
  replay_controls?: ReplayControls | null;
}): CanonicalSemanticObject {
  const {
    text_sha256,
    display_kind,
    queued,
    interrupted,
    surface,
    truncation_target,
    truncation_consent,
    attachments,
    display_metadata = null,
    replay_controls = null,
  } = params;

  // Sort attachments by (order, identity, version) — stable, deterministic
  const sortedAttachments = [...attachments].sort((a, b) => {
    if (a.order !== b.order) {return a.order - b.order;}

    if (a.identity < b.identity) {return -1;}

    if (a.identity > b.identity) {return 1;}

    if (a.version < b.version) {return -1;}

    if (a.version > b.version) {return 1;}

    return 0;
  });

  // Return with keys in alphabetical order to match sorted-key serialization
  return {
    attachments: sortedAttachments.map((a) => ({
      identity: a.identity,
      order: a.order,
      version: a.version,
    })),
    display_kind,
    display_metadata: display_metadata ?? null,
    interrupted,
    queued,
    replay_controls: replay_controls ?? null,
    surface,
    text_sha256,
    truncation_consent,
    truncation_target,
  };
}

/**
 * SHA-256 of deterministically serialized canonical object.
 * sortedJsonStringify ensures key order never affects the fingerprint.
 * Produces output compatible with Python compute_semantic_fingerprint.
 */
export function computeSemanticFingerprint(canonical: CanonicalSemanticObject): string {
  const serialized = sortedJsonStringify(canonical);

  return createHash('sha256').update(serialized, 'utf8').digest('hex');
}

/** Build the immutable durable-v1 identity reused for every transport retry. */
export function buildPromptSubmissionV1(params: {
  readonly submission_id: string;
  readonly text: string;
  readonly queued: boolean;
  readonly interrupted: boolean;
  readonly surface: string;
}): PromptSubmissionV1 {
  const canonical = buildCanonicalSemanticObject({
    text_sha256: computeTextSha256(params.text),
    display_kind: 'normal',
    queued: params.queued,
    interrupted: params.interrupted,
    surface: params.surface,
    truncation_target: null,
    truncation_consent: false,
    attachments: [],
    replay_controls: { attachments: 'unsupported', truncation: 'unsupported' },
  });

  return {
    submission_id: params.submission_id,
    contract_version: CONTRACT_VERSION,
    semantic_fingerprint: computeSemanticFingerprint(canonical),
  };
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

const REQUIRED_REQUEST_FIELDS = [
  'submission_id',
  'contract_version',
  'semantic_fingerprint',
] as const;

const REQUIRED_ACK_FIELDS = [
  'submission_id',
  'contract_version',
  'semantic_fingerprint',
  'durable_admission_status',
  'invocation_status',
  'safe_terminal_action',
] as const;

const DURABLE_ADMISSION_STATUSES = new Set<DurableAdmissionStatus>(['accepted', 'conflict']);
const INVOCATION_STATUSES = new Set<InvocationStatus>([
  'pending', 'queued', 'dispatching', 'preparing', 'invoking', 'running', 'completed',
  'terminal_error', 'attachment_reattach_required', 'unknown_outcome',
]);
const SAFE_TERMINAL_ACTIONS = new Set<SafeTerminalAction>([
  'check_transcript_resume_or_start_new', 'reattach_then_continue', 'start_new_submission',
]);

/**
 * Validate and construct a PromptSubmissionV1 from an unknown value.
 * Throws on missing required fields or contract version mismatch.
 * Does NOT mutate any runtime state.
 */
export function validateRequest(data: unknown): PromptSubmissionV1 {
  if (typeof data !== 'object' || data === null) {
    throw new Error('Request must be a non-null object');
  }

  const obj = data as Record<string, unknown>;
  const missing = REQUIRED_REQUEST_FIELDS.filter((f) => !(f in obj));

  if (missing.length > 0) {
    throw new Error(`Missing required request fields: ${missing.join(', ')}`);
  }

  const contractVersion = String(obj['contract_version']);

  if (contractVersion !== CONTRACT_VERSION) {
    throw new Error(
      `contract_version mismatch: expected ${CONTRACT_VERSION}, got ${contractVersion}`
    );
  }

  return {
    submission_id: String(obj['submission_id']),
    contract_version: CONTRACT_VERSION,
    semantic_fingerprint: String(obj['semantic_fingerprint']),
  };
}

/**
 * Validate and construct a PromptSubmissionAckV1 from an unknown value.
 * Throws on missing required fields.
 * Does NOT mutate any runtime state.
 */
export function validateAck(data: unknown): PromptSubmissionAckV1 {
  if (typeof data !== 'object' || data === null) {
    throw new Error('Ack must be a non-null object');
  }

  const obj = data as Record<string, unknown>;
  const missing = REQUIRED_ACK_FIELDS.filter((f) => !(f in obj));

  if (missing.length > 0) {
    throw new Error(`Missing required ack fields: ${missing.join(', ')}`);
  }

  const durableAdmissionStatus = obj['durable_admission_status'];
  const invocationStatus = obj['invocation_status'];
  const safeTerminalAction = obj['safe_terminal_action'];
  if (typeof durableAdmissionStatus !== 'string' || !DURABLE_ADMISSION_STATUSES.has(durableAdmissionStatus as DurableAdmissionStatus)) {
    throw new Error('Unknown durable_admission_status');
  }
  if (typeof invocationStatus !== 'string' || !INVOCATION_STATUSES.has(invocationStatus as InvocationStatus)) {
    throw new Error('Unknown invocation_status');
  }
  if (safeTerminalAction !== null && (typeof safeTerminalAction !== 'string' || !SAFE_TERMINAL_ACTIONS.has(safeTerminalAction as SafeTerminalAction))) {
    throw new Error('Unknown safe_terminal_action');
  }

  return {
    submission_id: String(obj['submission_id']),
    contract_version: String(obj['contract_version']),
    semantic_fingerprint: String(obj['semantic_fingerprint']),
    durable_admission_status: durableAdmissionStatus as DurableAdmissionStatus,
    invocation_status: invocationStatus as InvocationStatus,
    safe_terminal_action: safeTerminalAction as SafeTerminalAction | null,
  };
}

/**
 * Return the only approved log surface for an ack.
 * Contains ONLY: submission_id, contract_version, durable_admission_status,
 * invocation_status, safe_terminal_action.
 * Never: text, paths, error bodies, fingerprint preimage, raw errors.
 */
export function toSafeLogContext(ack: PromptSubmissionAckV1): SafeLogContext {
  return {
    submission_id: ack.submission_id,
    contract_version: ack.contract_version,
    durable_admission_status: ack.durable_admission_status,
    invocation_status: ack.invocation_status,
    safe_terminal_action: ack.safe_terminal_action,
  };
}
