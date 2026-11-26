import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse, delay } from 'msw'
import type { DecisionIn } from '@/api/schemas'
import { renderApp } from '@/test/render'
import { PROPOSAL_ID, WF_ID, accepted, detailHandler, server, workflowDetail } from '@/test/server'

function captureDecisions(
  response: () => Promise<Response> | Response = () => HttpResponse.json(accepted, { status: 202 }),
) {
  const bodies: DecisionIn[] = []
  server.use(
    http.post(`/api/v1/proposals/${PROPOSAL_ID}/decision`, async ({ request }) => {
      bodies.push((await request.json()) as DecisionIn)
      return response()
    }),
  )
  return bodies
}

describe('ReviewActions', () => {
  it('double click posts once', async () => {
    server.use(detailHandler(workflowDetail()))
    const bodies = captureDecisions(async () => {
      await delay(50)
      return HttpResponse.json(accepted, { status: 202 })
    })
    const user = userEvent.setup()
    renderApp(`/workflows/${WF_ID}`)
    const approve = await screen.findByRole('button', { name: 'Approve and send' })
    await user.dblClick(approve)
    expect(approve).toBeDisabled()
    await waitFor(() => expect(bodies).toHaveLength(1))
    await delay(100)
    expect(bodies).toHaveLength(1)
    expect(bodies[0]).toMatchObject({ decision: 'approve', interrupt_id: 'int-1', payload_hash: 'hash-1' })
    expect(bodies[0].edited_payload).toBeUndefined()
  })

  it('409 shows reload message', async () => {
    server.use(detailHandler(workflowDetail()))
    captureDecisions(() => HttpResponse.json({ error: 'conflict', reason: 'stale_revision' }, { status: 409 }))
    const user = userEvent.setup()
    renderApp(`/workflows/${WF_ID}`)
    await user.click(await screen.findByRole('button', { name: 'Approve and send' }))
    expect(await screen.findByRole('alert')).toHaveTextContent(
      'This proposal changed — reload to review the latest version',
    )
  })

  it('dismiss submits reject', async () => {
    server.use(detailHandler(workflowDetail()))
    const bodies = captureDecisions()
    const user = userEvent.setup()
    renderApp(`/workflows/${WF_ID}`)
    await user.click(await screen.findByRole('button', { name: 'Dismiss' }))
    await waitFor(() => expect(bodies).toHaveLength(1))
    expect(bodies[0].decision).toBe('reject')
  })

  it('give feedback opens a textarea and submits feedback', async () => {
    server.use(detailHandler(workflowDetail()))
    const bodies = captureDecisions()
    const user = userEvent.setup()
    renderApp(`/workflows/${WF_ID}`)
    await user.click(await screen.findByRole('button', { name: 'Give feedback' }))
    await user.type(screen.getByLabelText('Feedback for the assistant'), 'Make it shorter')
    await user.click(screen.getByRole('button', { name: 'Send feedback' }))
    await waitFor(() => expect(bodies).toHaveLength(1))
    expect(bodies[0]).toMatchObject({ decision: 'feedback', feedback: 'Make it shorter' })
  })

  it('cancel pending action posts to the workflow cancel endpoint', async () => {
    const ids: string[] = []
    server.use(
      detailHandler(workflowDetail()),
      http.post(`/api/v1/workflows/${WF_ID}/cancel`, async ({ request }) => {
        ids.push(((await request.json()) as { request_id: string }).request_id)
        return HttpResponse.json(
          {
            status: 'accepted',
            workflow_id: WF_ID,
            run_status: 'cancelled',
            cancelled_review_items: [],
            cancelled_proposals: [],
            completed_actions: [],
            job_id: null,
          },
          { status: 202 },
        )
      }),
    )
    const user = userEvent.setup()
    renderApp(`/workflows/${WF_ID}`)
    await user.click(await screen.findByRole('button', { name: 'Cancel pending action' }))
    await waitFor(() => expect(ids).toHaveLength(1))
  })
})

describe('ActionOutcome', () => {
  it('outcome unknown shows no-resend text', async () => {
    const base = workflowDetail()
    server.use(
      detailHandler({
        ...base,
        run: { ...base.run, status: 'failed', error_code: 'outcome_unknown' },
        review_items: [{ ...base.review_items[0], status: 'resolved', resolved_at: '2026-11-20T15:00:00+00:00' }],
        proposals: [{ ...base.proposals[0], status: 'approved' }],
        actions: [
          {
            id: 'aaaaaaaa-aaaa-4aaa-8aaa-000000000001',
            proposal_id: PROPOSAL_ID,
            kind: 'send_reply',
            status: 'outcome_unknown',
            provider_ref: null,
            error_code: 'timeout',
            updated_at: '2026-11-20T15:01:00+00:00',
          },
        ],
      }),
    )
    renderApp(`/workflows/${WF_ID}`)
    expect(await screen.findByText('Outcome unknown')).toBeInTheDocument()
    expect(screen.getByText('Check your Sent folder; the assistant will not resend automatically')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Approve and send' })).not.toBeInTheDocument()
  })
})
