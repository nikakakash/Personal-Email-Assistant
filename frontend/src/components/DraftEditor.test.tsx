import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import type { DecisionIn } from '@/api/schemas'
import { renderApp } from '@/test/render'
import { PROPOSAL_ID, WF_ID, accepted, detailHandler, server, workflowDetail } from '@/test/server'

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/

describe('DraftEditor', () => {
  it('shows the subject read-only and the exact stored text', async () => {
    server.use(detailHandler(workflowDetail()))
    renderApp(`/workflows/${WF_ID}`)
    const subject = await screen.findByLabelText('Subject')
    expect(subject).toHaveAttribute('readonly')
    expect(subject).toHaveValue('Re: Lunch?')
    const preview = screen.getByRole('region', { name: 'Exact text to be sent' })
    expect(preview).toHaveTextContent('To: alice@example.com')
    expect(preview).toHaveTextContent('Subject: Re: Lunch?')
    expect(preview).toHaveTextContent('Tuesday works for me.')
  })

  it('approve disabled after edit; edit-and-approve posts edited_payload with new request_id', async () => {
    const bodies: DecisionIn[] = []
    let fail = true
    server.use(
      detailHandler(workflowDetail()),
      http.post(`/api/v1/proposals/${PROPOSAL_ID}/decision`, async ({ request }) => {
        bodies.push((await request.json()) as DecisionIn)
        if (fail) {
          fail = false
          return HttpResponse.json({ error: 'internal_error' }, { status: 500 })
        }
        return HttpResponse.json(accepted, { status: 202 })
      }),
    )
    const user = userEvent.setup()
    renderApp(`/workflows/${WF_ID}`)

    const approve = await screen.findByRole('button', { name: 'Approve and send' })
    const editApprove = screen.getByRole('button', { name: 'Edit and approve' })
    expect(approve).toBeEnabled()
    expect(editApprove).toBeDisabled()

    const body = screen.getByLabelText('Message')
    await user.clear(body)
    await user.type(body, 'Tuesday at noon works.')
    await user.type(screen.getByLabelText('Cc'), 'carol@example.com')

    expect(approve).toBeDisabled()
    expect(editApprove).toBeEnabled()

    await user.click(editApprove)
    await screen.findByRole('alert')
    await user.click(editApprove)
    await waitFor(() => expect(bodies).toHaveLength(2))

    const [first, second] = bodies
    expect(second).toMatchObject({
      decision: 'edit_approve',
      proposal_revision: 1,
      payload_hash: 'hash-1',
      interrupt_id: 'int-1',
      edited_payload: {
        to: ['alice@example.com'],
        cc: ['carol@example.com'],
        subject: 'Re: Lunch?',
        body_text: 'Tuesday at noon works.',
      },
    })
    expect(first.request_id).toMatch(UUID)
    expect(second.request_id).toMatch(UUID)
    expect(second.request_id).not.toBe(first.request_id)
  })
})
