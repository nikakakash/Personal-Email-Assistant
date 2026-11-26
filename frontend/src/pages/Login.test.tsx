import { screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { renderApp } from '@/test/render'
import { me, server, unauthenticated } from '@/test/server'

async function submit(passphrase = 'pw') {
  const user = userEvent.setup()
  await user.type(await screen.findByLabelText('Passphrase'), passphrase)
  await user.click(screen.getByRole('button', { name: 'Sign in' }))
}

describe('Login', () => {
  it.each([
    [401, { error: 'invalid_credentials' }, 'Wrong passphrase'],
    [429, { error: 'too_many_attempts' }, 'Too many attempts, wait a minute'],
    [503, { error: 'not_configured' }, 'Server not configured — set OWNER_PASSPHRASE'],
  ])('login shows %i message', async (status, body, message) => {
    server.use(unauthenticated, http.post('/api/v1/auth/login', () => HttpResponse.json(body, { status })))
    renderApp('/login')
    await submit()
    expect(await screen.findByRole('alert')).toHaveTextContent(message)
  })

  it('goes to the queue after a successful login', async () => {
    let loggedIn = false
    server.use(
      http.get('/api/v1/auth/me', () =>
        loggedIn ? HttpResponse.json(me) : HttpResponse.json({ error: 'unauthenticated' }, { status: 401 }),
      ),
      http.post('/api/v1/auth/login', () => {
        loggedIn = true
        return HttpResponse.json({ csrf_token: 'abc' })
      }),
    )
    renderApp('/login')
    await submit()
    expect(await screen.findByRole('tab', { name: /Reply ready/ })).toBeInTheDocument()
  })
})
