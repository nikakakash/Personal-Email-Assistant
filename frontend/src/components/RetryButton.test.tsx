import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { renderApp } from '@/test/render'
import { WF_ID, detailHandler, server, workflowDetail } from '@/test/server'

function pausedDetail(error_code: string, status = 'paused', retryable = true) {
  const base = workflowDetail()
  return {
    ...base,
    run: { ...base.run, status, error_code, retryable },
    review_items: [{ ...base.review_items[0], status: 'resolved', resolved_at: '2026-11-20T15:00:00+00:00' }],
    proposals: [{ ...base.proposals[0], status: 'approved' }],
  }
}

describe('RetryButton', () => {
  it('a paused run explains why and posts one retry', async () => {
    const ids: string[] = []
    server.use(
      detailHandler(pausedDetail('outbound_disabled')),
      http.post(`/api/v1/workflows/${WF_ID}/retry`, async ({ request }) => {
        ids.push(((await request.json()) as { request_id: string }).request_id)
        return HttpResponse.json(
          { status: 'accepted', workflow_id: WF_ID, job_id: '99999999-9999-4999-8999-000000000001' },
          { status: 202 },
        )
      }),
    )
    const user = userEvent.setup()
    renderApp(`/workflows/${WF_ID}`)

    expect(await screen.findByText(/Sending is switched off/)).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Retry' }))
    await waitFor(() => expect(ids).toHaveLength(1))
  })

  it('409 says the run can no longer be retried', async () => {
    server.use(
      detailHandler(pausedDetail('budget')),
      http.post(`/api/v1/workflows/${WF_ID}/retry`, () =>
        HttpResponse.json({ error: 'conflict', reason: 'not_retryable' }, { status: 409 }),
      ),
    )
    const user = userEvent.setup()
    renderApp(`/workflows/${WF_ID}`)
    await user.click(await screen.findByRole('button', { name: 'Retry' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('This run can no longer be retried')
  })

  it('an unknown outcome offers no retry', async () => {
    server.use(detailHandler(pausedDetail('outcome_unknown', 'failed', false)))
    renderApp(`/workflows/${WF_ID}`)
    expect(await screen.findByText('Error: outcome_unknown')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Retry' })).not.toBeInTheDocument()
  })
})
