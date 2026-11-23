import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { formatReceived } from '@/lib/utils'
import { renderApp } from '@/test/render'
import { review, server, unauthenticated, workflow } from '@/test/server'

describe('Queue', () => {
  it('Queue renders filters Reply ready / Needs information / Notifications / Failed / Completed', async () => {
    renderApp()
    const tabs = await screen.findAllByRole('tab')
    expect(tabs.map((t) => t.textContent?.replace(/\s*\(\d+\)/, ''))).toEqual([
      'Reply ready',
      'Needs information',
      'Notifications',
      'Failed',
      'Completed',
    ])
  })

  it('Queue shows sender, subject, received time, classification, reason', async () => {
    let params: URLSearchParams | null = null
    server.use(
      http.get('/api/v1/reviews', ({ request }) => {
        const q = new URL(request.url).searchParams
        if (q.get('kind') === 'proposal') params = q
        return HttpResponse.json({ items: q.get('kind') === 'proposal' ? [review(1)] : [], next_cursor: null })
      }),
    )
    renderApp()
    const link = await screen.findByRole('link', { name: /sender1@example.com/ })
    expect(link).toHaveAttribute('href', '/workflows/11111111-1111-4111-8111-000000000001')
    const row = within(link)
    expect(row.getByText('Subject 1')).toBeInTheDocument()
    expect(row.getByText(formatReceived('2026-11-20T14:30:00+00:00'))).toBeInTheDocument()
    expect(row.getByText('needs_reply')).toBeInTheDocument()
    expect(row.getByText('Reason 1')).toBeInTheDocument()
    expect(params!.get('status')).toBe('pending')
  })

  it('shows an empty state per tab', async () => {
    const user = userEvent.setup()
    renderApp()
    expect(await screen.findByText('No replies are waiting for review.')).toBeInTheDocument()
    await user.click(screen.getByRole('tab', { name: 'Failed' }))
    expect(await screen.findByText('No failed runs.')).toBeInTheDocument()
  })

  it('Completed requests both done statuses', async () => {
    let statuses: string[] = []
    server.use(
      http.get('/api/v1/workflows', ({ request }) => {
        const s = new URL(request.url).searchParams.getAll('status')
        if (s.includes('done_no_action')) statuses = s
        return HttpResponse.json({ items: [], next_cursor: null })
      }),
    )
    renderApp()
    await waitFor(() => expect(statuses).toEqual(['done_with_actions', 'done_no_action']))
  })

  it('Load more fetches next cursor', async () => {
    const user = userEvent.setup()
    const cursors: (string | null)[] = []
    server.use(
      http.get('/api/v1/workflows', ({ request }) => {
        const url = new URL(request.url)
        if (url.searchParams.get('status') !== 'failed') return HttpResponse.json({ items: [], next_cursor: null })
        const cursor = url.searchParams.get('cursor')
        cursors.push(cursor)
        return HttpResponse.json(
          cursor ? { items: [workflow(2)], next_cursor: null } : { items: [workflow(1)], next_cursor: 'CUR1' },
        )
      }),
    )
    renderApp()
    await user.click(await screen.findByRole('tab', { name: /Failed/ }))
    expect(await screen.findByText('Workflow subject 1')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Load more' }))
    expect(await screen.findByText('Workflow subject 2')).toBeInTheDocument()
    expect(screen.getByText('Workflow subject 1')).toBeInTheDocument()
    expect(cursors).toEqual([null, 'CUR1'])
    expect(screen.queryByRole('button', { name: 'Load more' })).not.toBeInTheDocument()
  })

  it('401 redirects to login', async () => {
    server.use(unauthenticated)
    renderApp('/')
    expect(await screen.findByRole('button', { name: 'Sign in' })).toBeInTheDocument()
  })

  it('status strip shows readiness, live state and pause flags', async () => {
    server.use(
      http.get('/api/v1/flags', () => HttpResponse.json({ ingestion_paused: true, outbound_paused: false })),
    )
    renderApp()
    expect(await screen.findByText('Backend: ok')).toBeInTheDocument()
    expect(screen.getByText('Live: connecting')).toBeInTheDocument()
    expect(await screen.findByText('Ingestion paused')).toBeInTheDocument()
    expect(screen.queryByText('Outbound paused')).not.toBeInTheDocument()
  })
})
