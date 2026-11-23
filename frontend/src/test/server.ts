/** MSW server plus factories that build the backend's real response shapes. */
import { http, HttpResponse } from 'msw'
import { setupServer } from 'msw/node'
import type { ReviewItemOut, WorkflowOut, WorkflowSummaryOut } from '@/api/schemas'

export const review = (n: number, over: Partial<ReviewItemOut> = {}): ReviewItemOut => ({
  id: `00000000-0000-4000-8000-${String(n).padStart(12, '0')}`,
  kind: 'proposal',
  status: 'pending',
  workflow_id: `11111111-1111-4111-8111-${String(n).padStart(12, '0')}`,
  proposal_id: `22222222-2222-4222-8222-${String(n).padStart(12, '0')}`,
  interrupt_id: 'int-1',
  sender: `sender${n}@example.com`,
  subject: `Subject ${n}`,
  received_at: '2026-11-20T14:30:00+00:00',
  classification: 'needs_reply',
  reason: `Reason ${n}`,
  created_at: '2026-11-20T14:31:00+00:00',
  ...over,
})

export const workflow = (n: number, over: Partial<WorkflowSummaryOut> = {}): WorkflowSummaryOut => ({
  id: `33333333-3333-4333-8333-${String(n).padStart(12, '0')}`,
  status: 'failed',
  error_code: null,
  classification: 'needs_reply',
  reason: `Workflow reason ${n}`,
  sender: `wf${n}@example.com`,
  subject: `Workflow subject ${n}`,
  received_at: '2026-11-20T14:30:00+00:00',
  created_at: '2026-11-20T14:31:00+00:00',
  updated_at: '2026-11-20T14:32:00+00:00',
  ...over,
})

export const empty = { items: [], next_cursor: null }

export const me = { owner_id: '44444444-4444-4444-8444-444444444444', email: 'me@example.com' }

/** Handlers every page test needs; individual tests override with `server.use`. */
export const baseHandlers = [
  http.get('/api/v1/auth/me', () => HttpResponse.json(me)),
  http.get('/api/v1/health/ready', () => HttpResponse.json({ status: 'ok' })),
  http.get('/api/v1/flags', () => HttpResponse.json({ ingestion_paused: false, outbound_paused: false })),
  http.get('/api/v1/reviews', () => HttpResponse.json(empty)),
  http.get('/api/v1/workflows', () => HttpResponse.json(empty)),
]

export const unauthenticated = http.get('/api/v1/auth/me', () =>
  HttpResponse.json({ error: 'unauthenticated' }, { status: 401 }),
)

export const server = setupServer(...baseHandlers)

// ---- Workflow detail (GET /workflows/{id}) ----

export const WF_ID = '55555555-5555-4555-8555-000000000001'
export const PROPOSAL_ID = '66666666-6666-4666-8666-000000000001'
export const ITEM_ID = '77777777-7777-4777-8777-000000000001'

export const replyPayload = {
  provider_thread_id: 'thread-1',
  in_reply_to: '<m1@example.com>',
  references: ['<m1@example.com>'],
  to: ['alice@example.com'],
  cc: [],
  subject: 'Re: Lunch?',
  body_text: 'Tuesday works for me.',
  evidence_refs: ['message:m1'],
}

export const eventPayload = {
  calendar_id: 'primary',
  title: 'Project sync',
  start: '2026-11-24T13:00:00+00:00',
  end: '2026-11-24T13:30:00+00:00',
  time_zone: 'Europe/Paris',
  attendees: ['alice@example.com', 'bob@example.com'],
  location: 'Video call',
  send_updates: 'all',
  evidence_refs: ['availability:2026-11-24:2026-11-24'],
}

type Detail = WorkflowOut
type ProposalKind = 'send_reply' | 'create_event'

/** A run waiting on one pending proposal of `kind`; override any part with `over`. */
export function workflowDetail(kind: ProposalKind = 'send_reply', over: Partial<Detail> = {}): Detail {
  const payload = kind === 'send_reply' ? replyPayload : eventPayload
  return {
    run: {
      id: WF_ID,
      status: 'waiting_review',
      error_code: null,
      classification: 'needs_reply',
      reason: 'Alice asks about lunch',
      step_count: 3,
      graph_version: 'wf1',
      provider_thread_id: 'thread-1',
      created_at: '2026-11-20T14:31:00+00:00',
      updated_at: '2026-11-20T14:32:00+00:00',
    },
    thread: [
      {
        id: '88888888-8888-4888-8888-000000000001',
        provider_message_id: 'm1',
        sender: 'alice@example.com',
        to: ['me@example.com'],
        cc: [],
        subject: 'Lunch?',
        body_text: 'Are you free for lunch on Tuesday?',
        body_html: null,
        received_at: '2026-11-20T14:30:00+00:00',
      },
    ],
    review_items: [
      {
        id: ITEM_ID,
        kind: 'proposal',
        status: 'pending',
        interrupt_id: 'int-1',
        proposal_id: PROPOSAL_ID,
        payload:
          kind === 'create_event'
            ? {
                proposal_kind: kind,
                revision: 1,
                availability: [
                  {
                    key: 'availability:2026-11-24:2026-11-24',
                    time_zone: 'Europe/Paris',
                    retrieved_at: '2026-11-23T09:00:00+00:00',
                  },
                ],
              }
            : { proposal_kind: kind, revision: 1 },
        created_at: '2026-11-20T14:32:00+00:00',
        resolved_at: null,
      },
    ],
    proposals: [
      {
        id: PROPOSAL_ID,
        kind,
        status: 'awaiting_approval',
        revision: 1,
        payload,
        payload_hash: 'hash-1',
        author: 'model',
        created_at: '2026-11-20T14:32:00+00:00',
      },
    ],
    actions: [],
    events: [],
    ...over,
  }
}

export const detailHandler = (detail: Detail) =>
  http.get(`/api/v1/workflows/${detail.run.id}`, () => HttpResponse.json(detail))

export const accepted = { status: 'accepted', approval_id: '99999999-9999-4999-8999-000000000001', job_id: null, revision: 1 }
