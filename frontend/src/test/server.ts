/** MSW server plus factories that build the backend's real response shapes. */
import { http, HttpResponse } from 'msw'
import { setupServer } from 'msw/node'
import type { ReviewItemOut, WorkflowSummaryOut } from '@/api/schemas'

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
