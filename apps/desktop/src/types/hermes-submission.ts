/**
 * Additive TypeScript type companion for hermes.ts — spec-01 contract types.
 *
 * This file is purely additive.  It introduces no runtime behavior changes.
 * Import from here for submission-recovery contract types throughout the Desktop app.
 *
 * Execution guarantee: durable_acceptance_admission_only.
 * "accepted" does NOT mean the provider ran or a user turn persisted.
 */

export type {
  AttachmentIdentity,
  CanonicalSemanticObject,
  DisplayKind,
  DurableAdmissionStatus,
  InvocationStatus,
  PromptSubmissionAckV1,
  PromptSubmissionV1,
  ReplayControls,
  SafeLogContext,
  SafeTerminalAction,
} from '../lib/prompt-submission-contract';

export {
  buildCanonicalSemanticObject,
  computeSemanticFingerprint,
  computeTextSha256,
  CONTRACT_VERSION,
  sortedJsonStringify,
  toSafeLogContext,
  validateAck,
  validateRequest,
} from '../lib/prompt-submission-contract';
