/**
 * zod schemas mirroring the backend's Pydantic models (backend/app/api/*.py, views.py,
 * actions/decisions.py). Each schema is the runtime check for one JSON shape; the
 * TypeScript types are inferred from them so they cannot drift apart.
 */
import { z } from 'zod'

const uuid = z.string()
const instant = z.string() // ISO 8601 with offset

export const reviewKind = z.enum(['notification', 'question', 'proposal'])
export const reviewStatus = z.enum(['pending', 'stale', 'resolved', 'cancelled'])
export const workflowStatus = z.enum([
  'running',
  'waiting_review',
  'paused',
  'done_no_action',
  'done_with_actions',
  'failed',
  'cancelled',
])

/** views.ReviewItemOut */
export const ReviewItemOut = z.object({
  id: uuid,
  kind: z.string(),
  status: z.string(),
  workflow_id: uuid,
  proposal_id: uuid.nullable(),
  interrupt_id: z.string().nullable(),
  sender: z.string(),
  subject: z.string(),
  received_at: instant,
  classification: z.string().nullable(),
  reason: z.string().nullable(),
  created_at: instant,
})
export type ReviewItemOut = z.infer<typeof ReviewItemOut>

/** views.WorkflowSummaryOut (the list rows of GET /workflows) */
export const WorkflowSummaryOut = z.object({
  id: uuid,
  status: z.string(),
  error_code: z.string().nullable(),
  classification: z.string().nullable(),
  reason: z.string().nullable(),
  sender: z.string(),
  subject: z.string(),
  received_at: instant,
  created_at: instant,
  updated_at: instant,
})
export type WorkflowSummaryOut = z.infer<typeof WorkflowSummaryOut>

/** pagination.Page[T] */
export const page = <T extends z.ZodType>(item: T) =>
  z.object({ items: z.array(item), next_cursor: z.string().nullable() })

export const ReviewPage = page(ReviewItemOut)
export const WorkflowPage = page(WorkflowSummaryOut)
export type ReviewPage = z.infer<typeof ReviewPage>
export type WorkflowPage = z.infer<typeof WorkflowPage>

const payload = z.record(z.string(), z.unknown())

export const RunOut = z.object({
  id: uuid,
  status: z.string(),
  error_code: z.string().nullable(),
  classification: z.string().nullable(),
  reason: z.string().nullable(),
  step_count: z.number(),
  graph_version: z.string(),
  provider_thread_id: z.string(),
  created_at: instant,
  updated_at: instant,
})

export const MessageOut = z.object({
  id: uuid,
  provider_message_id: z.string(),
  sender: z.string(),
  to: z.array(z.string()),
  cc: z.array(z.string()),
  subject: z.string(),
  body_text: z.string(),
  body_html: z.string().nullable(),
  received_at: instant,
})

export const ReviewItemDetailOut = z.object({
  id: uuid,
  kind: z.string(),
  status: z.string(),
  interrupt_id: z.string().nullable(),
  proposal_id: uuid.nullable(),
  payload,
  created_at: instant,
  resolved_at: instant.nullable(),
})

export const ProposalOut = z.object({
  id: uuid,
  kind: z.string(),
  status: z.string(),
  revision: z.number(),
  payload,
  payload_hash: z.string(),
  author: z.string(),
  created_at: instant,
})

export const ActionOut = z.object({
  id: uuid,
  proposal_id: uuid,
  kind: z.string(),
  status: z.string(),
  provider_ref: z.string().nullable(),
  error_code: z.string().nullable(),
  updated_at: instant,
})

/** views.EventOut: also the `data:` of every SSE message. */
export const EventOut = z.object({
  id: z.number(),
  workflow_id: uuid,
  type: z.string(),
  created_at: instant,
  payload,
})
export type EventOut = z.infer<typeof EventOut>

/** views.WorkflowOut (GET /workflows/{id}) */
export const WorkflowOut = z.object({
  run: RunOut,
  thread: z.array(MessageOut),
  review_items: z.array(ReviewItemDetailOut),
  proposals: z.array(ProposalOut),
  actions: z.array(ActionOut),
  events: z.array(EventOut),
})
export type WorkflowOut = z.infer<typeof WorkflowOut>

/** actions.decisions.DecisionIn (request body of POST /proposals/{id}/decision) */
export const DecisionIn = z.object({
  decision: z.string(),
  proposal_revision: z.number(),
  payload_hash: z.string(),
  interrupt_id: z.string(),
  request_id: z.string(),
  edited_payload: payload.nullish(),
  feedback: z.string().nullish(),
})
export type DecisionIn = z.infer<typeof DecisionIn>

export const DecisionOutcome = z.object({
  status: z.literal('accepted'),
  approval_id: uuid,
  job_id: uuid.nullable(),
})

/** system.FlagsOut */
export const FlagsOut = z.object({
  ingestion_paused: z.boolean(),
  outbound_paused: z.boolean(),
})
export type FlagsOut = z.infer<typeof FlagsOut>

/** auth.LoginOut / auth.MeOut */
export const LoginOut = z.object({ csrf_token: z.string() })
export const MeOut = z.object({ owner_id: uuid, email: z.string() })
export type MeOut = z.infer<typeof MeOut>

/** health.ready */
export const HealthOut = z.object({ status: z.string() })
export type HealthOut = z.infer<typeof HealthOut>
