import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { renderApp } from '@/test/render'
import { server } from '@/test/server'

const usage = {
  daily: [],
  today: { cost_usd: '0.50', budget_usd: '2.00', remaining_usd: '1.50' },
}

const scheduling = {
  namespace: 'scheduling',
  revision: 2,
  value: {
    time_zone: 'Europe/Paris',
    work_start: '09:00',
    work_end: '17:00',
    work_days: [0, 1, 2, 3, 4],
    buffer_minutes: 10,
    default_duration_minutes: 30,
  },
  created_at: '2026-11-20T14:31:00+00:00',
}

function common() {
  server.use(
    http.get('/api/v1/usage', () => HttpResponse.json(usage)),
    http.get('/api/v1/preferences', () => HttpResponse.json({ scheduling })),
    http.get('/api/v1/dev/fixtures', () => HttpResponse.json({ error: 'not_found' }, { status: 404 })),
  )
}

describe('Settings', () => {
  it('shows usage today against the budget', async () => {
    common()
    renderApp('/settings')
    expect(await screen.findByText('$0.50 of $2.00 used today')).toBeInTheDocument()
  })

  it('pause toggles patch the flags', async () => {
    common()
    const bodies: unknown[] = []
    server.use(
      http.patch('/api/v1/flags', async ({ request }) => {
        const body = (await request.json()) as Record<string, boolean>
        bodies.push(body)
        return HttpResponse.json({ ingestion_paused: false, outbound_paused: true, ...body })
      }),
    )
    const user = userEvent.setup()
    renderApp('/settings')
    await user.click(await screen.findByRole('checkbox', { name: 'Pause outgoing' }))
    await waitFor(() => expect(bodies).toEqual([{ outbound_paused: true }]))
  })

  it('saving a preference namespace sends the revision it was based on', async () => {
    common()
    const bodies: unknown[] = []
    server.use(
      http.patch('/api/v1/preferences/scheduling', async ({ request }) => {
        bodies.push(await request.json())
        return HttpResponse.json({ ...scheduling, revision: 3 })
      }),
    )
    const user = userEvent.setup()
    renderApp('/settings')
    const buffer = await screen.findByLabelText('Buffer between meetings (minutes)')
    await waitFor(() => expect(buffer).toHaveValue(10))
    await user.clear(buffer)
    await user.type(buffer, '15')
    await user.click(screen.getByRole('button', { name: 'Save scheduling preferences' }))
    await waitFor(() => expect(bodies).toHaveLength(1))
    expect(bodies[0]).toEqual({
      expected_revision: 2,
      value: { ...scheduling.value, buffer_minutes: 15 },
    })
  })

  it('shows a conflict when preferences changed elsewhere', async () => {
    common()
    server.use(
      http.patch('/api/v1/preferences/writing', () =>
        HttpResponse.json({ error: 'conflict', reason: 'revision_mismatch' }, { status: 409 }),
      ),
    )
    const user = userEvent.setup()
    renderApp('/settings')
    await user.type(await screen.findByLabelText('Tone'), 'warm')
    await user.click(screen.getByRole('button', { name: 'Save writing preferences' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('changed in another tab')
  })

  it('hides the dev tools when the backend has none', async () => {
    common()
    renderApp('/settings')
    await screen.findByText('$0.50 of $2.00 used today')
    expect(screen.queryByRole('heading', { name: 'Developer tools' })).not.toBeInTheDocument()
  })

  it('dev tools inject a fixture and set a fault', async () => {
    common()
    const injected: unknown[] = []
    const faults: unknown[] = []
    server.use(
      http.get('/api/v1/dev/fixtures', () =>
        HttpResponse.json([{ id: 'lunch', classification: 'respond', message_count: 1, subject: 'Lunch?' }]),
      ),
      http.post('/api/v1/dev/inject', async ({ request }) => {
        injected.push(await request.json())
        return HttpResponse.json({ fixture_id: 'lunch', results: [] })
      }),
      http.patch('/api/v1/dev/faults', async ({ request }) => {
        const body = (await request.json()) as Record<string, string>
        faults.push(body)
        return HttpResponse.json({ send: 'ok', calendar_create: 'ok', ...body })
      }),
    )
    const user = userEvent.setup()
    renderApp('/settings')
    await user.selectOptions(await screen.findByLabelText('Fixture'), 'lunch')
    await user.click(screen.getByRole('button', { name: 'Inject fixture' }))
    await waitFor(() => expect(injected).toEqual([{ fixture_id: 'lunch' }]))
    await user.selectOptions(screen.getByLabelText('Send fault'), 'timeout_after_accept')
    await waitFor(() => expect(faults).toEqual([{ send: 'timeout_after_accept' }]))
  })
})
