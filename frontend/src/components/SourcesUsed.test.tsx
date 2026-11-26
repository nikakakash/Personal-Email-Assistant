import { screen, within } from '@testing-library/react'
import { renderApp } from '@/test/render'
import { WF_ID, detailHandler, server, workflowDetail } from '@/test/server'

const source = {
  ref: 'history:old-m1',
  sender: 'alice@example.com',
  date: '2026-10-01',
  subject: 'Earlier',
  snippet: 'Confirmed: the agreed price is 4500. <img src=x onerror=alert(1)>',
}

describe('SourcesUsed', () => {
  it('lists the past emails a reply draft cites, as plain text', async () => {
    const detail = workflowDetail('send_reply')
    detail.review_items[0].payload = { proposal_kind: 'send_reply', revision: 1, sources: [source] }
    server.use(detailHandler(detail))
    renderApp(`/workflows/${WF_ID}`)
    const card = await screen.findByRole('region', { name: 'Sources used' })
    const s = within(card)
    expect(s.getByText('Earlier')).toBeInTheDocument()
    expect(s.getByText('alice@example.com')).toBeInTheDocument()
    expect(card).toHaveTextContent('2026-10-01')
    // Untrusted email text is rendered as text, never as markup.
    expect(card).toHaveTextContent('<img src=x onerror=alert(1)>')
    expect(card.querySelector('img')).toBeNull()
  })

  it('shows sources on a meeting proposal too', async () => {
    const detail = workflowDetail('create_event')
    detail.review_items[0].payload = {
      ...(detail.review_items[0].payload as object),
      sources: [source],
    }
    server.use(detailHandler(detail))
    renderApp(`/workflows/${WF_ID}`)
    expect(await screen.findByRole('region', { name: 'Sources used' })).toBeInTheDocument()
    expect(screen.getByRole('region', { name: 'Meeting to create' })).toBeInTheDocument()
  })

  it('is absent when nothing was retrieved', async () => {
    server.use(detailHandler(workflowDetail('send_reply')))
    renderApp(`/workflows/${WF_ID}`)
    await screen.findByRole('region', { name: /draft/i })
    expect(screen.queryByRole('region', { name: 'Sources used' })).toBeNull()
  })
})
