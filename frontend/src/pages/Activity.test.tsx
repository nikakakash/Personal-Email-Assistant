import { screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { renderApp } from '@/test/render'
import { WF_ID, server } from '@/test/server'

const event = (id: number, type: string, payload: Record<string, unknown> = {}) => ({
  id,
  workflow_id: WF_ID,
  type,
  created_at: '2026-11-20T14:31:00+00:00',
  payload,
})

describe('Activity', () => {
  it('lists workflow events as a timeline linking to each run, with load more', async () => {
    const cursors: (string | null)[] = []
    server.use(
      http.get('/api/v1/activity', ({ request }) => {
        const cursor = new URL(request.url).searchParams.get('cursor')
        cursors.push(cursor)
        return cursor
          ? HttpResponse.json({ items: [event(1, 'run_started')], next_cursor: null })
          : HttpResponse.json({
              items: [event(3, 'action_succeeded', { kind: 'send_reply' }), event(2, 'approval_required')],
              next_cursor: '2',
            })
      }),
    )
    const user = userEvent.setup()
    renderApp('/activity')
    const timeline = await screen.findByRole('list', { name: 'Activity timeline' })
    const items = within(timeline).getAllByRole('listitem')
    expect(items[0]).toHaveTextContent('action succeeded')
    expect(items[0]).toHaveTextContent('send_reply')
    expect(within(items[0]).getByRole('link')).toHaveAttribute('href', `/workflows/${WF_ID}`)
    await user.click(screen.getByRole('button', { name: 'Load more' }))
    expect(await screen.findByText('run started')).toBeInTheDocument()
    expect(cursors).toEqual([null, '2'])
  })
})
