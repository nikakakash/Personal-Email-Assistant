/**
 * One owner action against the API: a fresh `request_id` per click, at most one request in
 * flight, and a readable error.
 *
 * The in-flight guard is a ref, not only state, so a double click that lands before React
 * re-renders the disabled button still posts once. After success the run, the queue lists and
 * the activity timeline are invalidated so they refetch the authoritative state.
 */
import { useQueryClient } from '@tanstack/react-query'
import { useRef, useState } from 'react'
import { ApiError } from '@/api/client'

export const CONFLICT_MESSAGE = 'This proposal changed — reload to review the latest version'

export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    const body = (error.body ?? {}) as { error?: string; errors?: unknown }
    if (error.status === 409) return CONFLICT_MESSAGE
    if (error.status === 422 && body.error === 'invalid_edit' && Array.isArray(body.errors)) {
      return `The edit was not accepted: ${body.errors.join(', ')}`
    }
    if (error.status === 422) return 'The server rejected this input. Check the fields and try again.'
    if (error.status === 404) return 'This item no longer exists. Reload the page.'
  }
  return 'Something went wrong. Try again.'
}

export function useSubmit(workflowId?: string) {
  const queryClient = useQueryClient()
  const inFlight = useRef(false)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function submit(send: (requestId: string) => Promise<unknown>): Promise<boolean> {
    if (inFlight.current) return false
    inFlight.current = true
    setPending(true)
    setError(null)
    try {
      await send(crypto.randomUUID())
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: workflowId ? ['workflow', workflowId] : ['workflow'] }),
        queryClient.invalidateQueries({ queryKey: ['reviews'] }),
        queryClient.invalidateQueries({ queryKey: ['workflows'] }),
      ])
      return true
    } catch (e) {
      setError(errorMessage(e))
      return false
    } finally {
      inFlight.current = false
      setPending(false)
    }
  }

  return { submit, pending, error }
}
