/**
 * Live updates: an `EventSource` on `/api/v1/events` that invalidates the cached queries.
 *
 * The stream is only a notification channel (the table is the record), so every message
 * simply marks the review and workflow queries stale and TanStack Query refetches the
 * authoritative state. Two safety nets: after a reconnect everything is refetched, and a
 * 30 s timer refetches regardless, because the backend stream can miss events committed
 * out of order. The browser's EventSource reconnects by itself and sends Last-Event-ID.
 */
import { useQueryClient } from '@tanstack/react-query'
import { useEffect, useState } from 'react'

export type ConnectionState = 'connecting' | 'open' | 'error'

export const SAFETY_REFETCH_MS = 30_000

export function useEventStream(workflowId?: string): ConnectionState {
  const queryClient = useQueryClient()
  const [state, setState] = useState<ConnectionState>('connecting')

  useEffect(() => {
    const invalidateAll = () => {
      void queryClient.invalidateQueries({ queryKey: ['reviews'] })
      void queryClient.invalidateQueries({ queryKey: ['workflows'] })
      void queryClient.invalidateQueries({
        queryKey: workflowId ? ['workflow', workflowId] : ['workflow'],
      })
    }

    const url = workflowId
      ? `/api/v1/events?workflow_id=${encodeURIComponent(workflowId)}`
      : '/api/v1/events'
    const source = new EventSource(url, { withCredentials: true })
    let wasDown = false

    source.onopen = () => {
      setState('open')
      if (wasDown) invalidateAll() // reconnected: re-read the authoritative state
      wasDown = false
    }
    source.onerror = () => {
      setState('error')
      wasDown = true
    }
    source.onmessage = () => invalidateAll()

    const timer = setInterval(invalidateAll, SAFETY_REFETCH_MS)
    return () => {
      clearInterval(timer)
      source.close()
    }
  }, [queryClient, workflowId])

  return state
}
