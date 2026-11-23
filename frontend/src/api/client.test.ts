import { http, HttpResponse } from 'msw'
import { z } from 'zod'
import { server } from '@/test/server'
import { ApiError, api, onUnauthorized } from './client'

describe('api client', () => {
  it('client sends X-CSRF-Token on POST', async () => {
    document.cookie = 'csrf_token=tok-123; path=/'
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
    let seen: { csrf: string | null; body: unknown } | null = null
    server.use(
      http.post('/api/v1/echo', async ({ request }) => {
        seen = {
          csrf: request.headers.get('X-CSRF-Token'),
          body: await request.json(),
        }
        return HttpResponse.json({ status: 'ok' })
      }),
    )
    const out = await api.post('/echo', { a: 1 }, z.object({ status: z.string() }))
    expect(out.status).toBe('ok')
    expect(seen).toEqual({ csrf: 'tok-123', body: { a: 1 } })
    expect(fetchSpy.mock.calls[0][1]?.credentials).toBe('include')
    fetchSpy.mockRestore()
  })

  it('does not send the CSRF header on GET', async () => {
    document.cookie = 'csrf_token=tok-123; path=/'
    let csrf: string | null = 'unset'
    server.use(
      http.get('/api/v1/thing', ({ request }) => {
        csrf = request.headers.get('X-CSRF-Token')
        return HttpResponse.json({ status: 'ok' })
      }),
    )
    await api.get('/thing', z.object({ status: z.string() }))
    expect(csrf).toBeNull()
  })

  it('client throws ApiError with status 409', async () => {
    server.use(
      http.post('/api/v1/proposals/x/decision', () =>
        HttpResponse.json({ error: 'conflict', reason: 'stale_revision' }, { status: 409 }),
      ),
    )
    const err = await api.post('/proposals/x/decision', {}, z.object({})).catch((e: unknown) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect((err as ApiError).status).toBe(409)
    expect((err as ApiError).body).toEqual({ error: 'conflict', reason: 'stale_revision' })
  })

  it('401 calls the registered unauthorized handler', async () => {
    const handler = vi.fn()
    onUnauthorized(handler)
    server.use(
      http.get('/api/v1/auth/me', () => HttpResponse.json({ error: 'unauthenticated' }, { status: 401 })),
    )
    const err = await api.get('/auth/me', z.object({})).catch((e: unknown) => e)
    onUnauthorized(null)
    expect((err as ApiError).status).toBe(401)
    expect(handler).toHaveBeenCalledOnce()
  })
})
