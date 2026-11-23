/**
 * Vitest setup: jest-dom matchers, DOM cleanup, and an MSW server shared by all tests.
 * Node's fetch rejects relative URLs, so they are resolved against the jsdom origin here.
 */
import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterAll, afterEach, beforeAll } from 'vitest'
import { server } from './server'

const realFetch = globalThis.fetch.bind(globalThis)
globalThis.fetch = (input: RequestInfo | URL, init?: RequestInit) =>
  realFetch(typeof input === 'string' && input.startsWith('/') ? new URL(input, location.origin) : input, init)

beforeAll(() => server.listen({ onUnhandledFrame: 'error' }))
afterEach(() => {
  cleanup()
  server.resetHandlers()
  document.cookie = 'csrf_token=; max-age=0; path=/'
})
afterAll(() => server.close())
