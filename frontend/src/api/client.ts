/**
 * Typed fetch wrapper for the backend API.
 *
 * Every call sends the session cookie (`credentials: "include"`), echoes the readable
 * `csrf_token` cookie in `X-CSRF-Token` on non-GET requests (double-submit CSRF), parses the
 * response with a zod schema, and throws `ApiError(status, body)` on a non-2xx answer.
 * A 401 calls the handler the app registered with `onUnauthorized` (the client itself knows
 * nothing about routing).
 */
import type { z } from 'zod'

export class ApiError extends Error {
  readonly status: number
  readonly body: unknown

  constructor(status: number, body: unknown) {
    super(`API error ${status}`)
    this.name = 'ApiError'
    this.status = status
    this.body = body
  }
}

let unauthorizedHandler: (() => void) | null = null

/** Register the callback run on any 401 (the app redirects to /login). */
export function onUnauthorized(handler: (() => void) | null): void {
  unauthorizedHandler = handler
}

const BASE = '/api/v1'

export function readCsrfToken(): string | null {
  for (const part of document.cookie.split('; ')) {
    if (part.startsWith('csrf_token=')) return decodeURIComponent(part.slice('csrf_token='.length))
  }
  return null
}

async function parseBody(res: Response): Promise<unknown> {
  const text = await res.text()
  if (!text) return null
  try {
    return JSON.parse(text)
  } catch {
    return text
  }
}

async function request<S extends z.ZodType>(
  method: string,
  path: string,
  schema: S,
  body?: unknown,
): Promise<z.infer<S>> {
  const headers: Record<string, string> = { Accept: 'application/json' }
  if (body !== undefined) headers['Content-Type'] = 'application/json'
  if (method !== 'GET') {
    const token = readCsrfToken()
    if (token) headers['X-CSRF-Token'] = token
  }
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers,
    credentials: 'include',
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  const parsed = await parseBody(res)
  if (!res.ok) {
    if (res.status === 401) unauthorizedHandler?.()
    throw new ApiError(res.status, parsed)
  }
  return schema.parse(parsed)
}

export const api = {
  get: <S extends z.ZodType>(path: string, schema: S) => request('GET', path, schema),
  post: <S extends z.ZodType>(path: string, body: unknown, schema: S) =>
    request('POST', path, schema, body),
  patch: <S extends z.ZodType>(path: string, body: unknown, schema: S) =>
    request('PATCH', path, schema, body),
}
