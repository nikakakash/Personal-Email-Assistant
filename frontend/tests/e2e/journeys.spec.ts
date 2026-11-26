/**
 * Spec §20 journeys 1–8 and 10 against the full stack (compose E2E overlay: scripted model,
 * mock mail and calendar, outbound execution on). Fixtures live in evals/e2e-fixtures; each
 * journey injects its own, so they only need a fresh database, not a particular order.
 */
import { execFileSync } from 'node:child_process'
import { expect, test, type APIRequestContext, type Page } from '@playwright/test'

/** `docker` arguments that address the E2E compose project (override with E2E_COMPOSE_ARGS). */
const COMPOSE_ARGS = (
  process.env.E2E_COMPOSE_ARGS ?? 'compose -p email-assistant-e2e -f ../compose.yaml -f ../compose.e2e.yaml'
).split(' ')

interface Workflow {
  id: string
  status: string
  subject: string
}
interface OutboxRow {
  payload: { provider_thread_id: string; body_text: string }
}

async function csrf(page: Page): Promise<string> {
  const cookie = (await page.context().cookies()).find((c) => c.name === 'csrf_token')
  if (!cookie) throw new Error('not logged in: no csrf_token cookie')
  return decodeURIComponent(cookie.value)
}

async function inject(page: Page, fixtureId: string): Promise<void> {
  const res = await page.request.post('/api/v1/dev/inject', {
    data: { fixture_id: fixtureId },
    headers: { 'X-CSRF-Token': await csrf(page) },
  })
  expect(res.status(), await res.text()).toBe(200)
  const body = (await res.json()) as { results: { status: string }[] }
  // "duplicate" here means the database is not fresh: run `down -v` before the suite.
  expect(body.results.map((r) => r.status)).not.toContain('duplicate')
}

async function workflows(request: APIRequestContext): Promise<Workflow[]> {
  const res = await request.get('/api/v1/workflows?limit=100')
  return ((await res.json()) as { items: Workflow[] }).items
}

/** Wait until the run for `subject` reaches `status`; returns its id. */
async function waitForRun(page: Page, subject: string, status = 'waiting_review'): Promise<string> {
  let id = ''
  await expect
    .poll(
      async () => {
        const run = (await workflows(page.request)).find((w) => w.subject === subject)
        id = run?.id ?? ''
        return run?.status
      },
      { timeout: 30_000 },
    )
    .toBe(status)
  return id
}

async function outbox(page: Page, threadId: string): Promise<OutboxRow[]> {
  const rows = (await (await page.request.get('/api/v1/dev/outbox')).json()) as OutboxRow[]
  return rows.filter((r) => r.payload.provider_thread_id === threadId)
}

async function openFromQueue(page: Page, tab: string, sender: string): Promise<void> {
  await page.goto('/')
  await page.getByRole('tab', { name: new RegExp(tab) }).click()
  await page.getByRole('link', { name: new RegExp(sender) }).click()
  await expect(page.getByRole('heading', { level: 1 })).toBeVisible()
}

const sent = (page: Page) => expect(page.getByText('Accepted by provider')).toBeVisible()

test('1–2: scheduling request shows time-zone-explicit options; approve sends exactly once', async ({ page }) => {
  await inject(page, 'e2e-sched')
  await waitForRun(page, 'Time for a quick call?')
  await openFromQueue(page, 'Reply ready', 'lena@example.org')

  const preview = page.getByRole('region', { name: 'Exact text to be sent' })
  await expect(preview).toContainText('09:00 UTC')
  await expect(preview).toContainText('To: lena@example.org')

  await page.getByRole('button', { name: 'Approve and send' }).click()
  await sent(page)
  const rows = await outbox(page, 'thr-e2e-sched')
  expect(rows).toHaveLength(1)
  expect(rows[0].payload.body_text).toContain('09:00 UTC')
})

test('3: edit and approve sends the edited text as revision 2', async ({ page }) => {
  await inject(page, 'e2e-edit')
  await waitForRun(page, 'Report draft')
  await openFromQueue(page, 'Reply ready', 'omar@example.org')

  const approve = page.getByRole('button', { name: 'Approve and send' })
  await page.getByLabel('Message').fill('Hi Omar,\n\nThe draft will be with you on Thursday.\n\nBest')
  await expect(approve).toBeDisabled()
  await page.getByRole('button', { name: 'Edit and approve' }).click()

  await sent(page)
  await expect(page.getByText('Reply · revision 2')).toBeVisible()
  const rows = await outbox(page, 'thr-e2e-edit')
  expect(rows).toHaveLength(1)
  expect(rows[0].payload.body_text).toContain('on Thursday')
})

test('4: dismiss sends nothing and closes the item visibly', async ({ page }) => {
  await inject(page, 'e2e-dismiss')
  await waitForRun(page, 'Quick favour')
  await openFromQueue(page, 'Reply ready', 'rita@example.org')

  await page.getByRole('button', { name: 'Dismiss' }).click()
  await expect(page.getByText('Dismissed', { exact: true }).first()).toBeVisible()
  await waitForRun(page, 'Quick favour', 'done_no_action')
  expect(await outbox(page, 'thr-e2e-dismiss')).toHaveLength(0)
})

test('5: a notification can be converted into a reply request', async ({ page }) => {
  await inject(page, 'e2e-notify')
  await waitForRun(page, 'Office closed Friday')
  await openFromQueue(page, 'Notifications', 'pat@example.org')

  await page.getByRole('button', { name: 'Draft a reply' }).click()
  await expect(page.getByRole('region', { name: 'Exact text to be sent' })).toContainText('Thanks for letting me know')
  await expect(page.getByRole('button', { name: 'Approve and send' })).toBeEnabled()
})

test('6: a meeting request produces an event in an explicit zone', async ({ page }) => {
  await inject(page, 'e2e-meeting')
  await waitForRun(page, 'Project kickoff')
  await openFromQueue(page, 'Reply ready', 'arjun@example.org')

  const meeting = page.getByRole('region', { name: 'Meeting to create' })
  await expect(meeting).toContainText('Asia/Kolkata (UTC+05:30)')
  await expect(meeting).toContainText(/14:30\s*–\s*15:00/)
  await expect(meeting).toContainText('Invitations will be sent: yes')
  await expect(meeting).toContainText('Availability checked')

  await page.getByRole('button', { name: 'Approve and send' }).click()
  await sent(page)
})

test('7: missing information asks a question; the answer resumes the same workflow', async ({ page }) => {
  await inject(page, 'e2e-question')
  const id = await waitForRun(page, 'Call next week')
  await openFromQueue(page, 'Needs information', 'kenji@example.org')

  await expect(page.getByText('Which time zone is Kenji in?')).toBeVisible()
  await page.getByLabel('Your answer').fill('Kenji is in Tokyo.')
  await page.getByRole('button', { name: 'Send answer' }).click()

  await expect(page.getByRole('region', { name: 'Exact text to be sent' })).toContainText('Tokyo time')
  expect(page.url()).toContain(id) // same workflow, not a new one
})

test('8: a worker restart keeps the pending draft; a double click sends once', async ({ page }) => {
  await inject(page, 'e2e-restart')
  await waitForRun(page, 'Dinner on Saturday?')

  execFileSync('docker', [...COMPOSE_ARGS, 'restart', 'worker'], { stdio: 'inherit' })

  await openFromQueue(page, 'Reply ready', 'noah@example.org')
  await page.getByRole('button', { name: 'Approve and send' }).dblclick()
  await sent(page)
  await waitForRun(page, 'Dinner on Saturday?', 'done_with_actions')
  expect(await outbox(page, 'thr-e2e-restart')).toHaveLength(1)
})

test('10: a manual reply in the thread makes the pending proposal stale', async ({ page }) => {
  await inject(page, 'e2e-stale')
  const id = await waitForRun(page, 'Book club')

  await inject(page, 'e2e-stale-followup')

  await page.goto(`/workflows/${id}`)
  await expect(page.getByText('Stale', { exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Approve and send' })).toHaveCount(0)
  expect(await outbox(page, 'thr-e2e-stale')).toHaveLength(0)
})
