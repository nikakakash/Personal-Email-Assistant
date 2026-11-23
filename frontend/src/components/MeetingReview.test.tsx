import { screen, within } from '@testing-library/react'
import { formatReceived } from '@/lib/utils'
import { renderApp } from '@/test/render'
import { WF_ID, detailHandler, eventPayload, server, workflowDetail } from '@/test/server'

describe('MeetingReview', () => {
  it('meeting review shows time zone and invitation behavior', async () => {
    server.use(detailHandler(workflowDetail('create_event')))
    renderApp(`/workflows/${WF_ID}`)
    const card = await screen.findByRole('region', { name: 'Meeting to create' })
    const m = within(card)
    expect(m.getByText('primary')).toBeInTheDocument()
    expect(m.getByText('Project sync')).toBeInTheDocument()
    expect(card).toHaveTextContent(/Tuesday, November 24, 2026/)
    // 13:00Z is 14:00 in Paris (CET, UTC+1) on that date.
    expect(card).toHaveTextContent(/14:00\s*–\s*14:30/)
    expect(card).toHaveTextContent('Europe/Paris (UTC+01:00)')
    expect(m.getByText('alice@example.com')).toBeInTheDocument()
    expect(m.getByText('bob@example.com')).toBeInTheDocument()
    expect(m.getByText('Video call')).toBeInTheDocument()
    expect(card).toHaveTextContent('Invitations will be sent: yes')
    expect(card).toHaveTextContent(`Availability checked ${formatReceived('2026-11-23T09:00:00+00:00')}`)
  })

  it('says no invitations when send_updates is none', async () => {
    const detail = workflowDetail('create_event')
    detail.proposals[0].payload = { ...eventPayload, send_updates: 'none' }
    server.use(detailHandler(detail))
    renderApp(`/workflows/${WF_ID}`)
    const card = await screen.findByRole('region', { name: 'Meeting to create' })
    expect(card).toHaveTextContent('Invitations will be sent: no')
  })
})
