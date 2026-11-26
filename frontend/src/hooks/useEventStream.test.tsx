import { QueryClientProvider } from '@tanstack/react-query'
import { act, renderHook } from '@testing-library/react'
import type { ReactNode } from 'react'
import { FakeEventSource, installFakeEventSource, newClient } from '@/test/render'
import { SAFETY_REFETCH_MS, useEventStream } from './useEventStream'

function setup(workflowId?: string) {
  installFakeEventSource()
  const client = newClient()
  const spy = vi.spyOn(client, 'invalidateQueries')
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  )
  const hook = renderHook(() => useEventStream(workflowId), { wrapper })
  return { spy, ...hook, source: FakeEventSource.instances[0] }
}

describe('useEventStream', () => {
  it('SSE message invalidates reviews query', () => {
    const { spy, source } = setup('wf-1')
    expect(source.url).toBe('/api/v1/events?workflow_id=wf-1')
    expect(source.init).toEqual({ withCredentials: true })
    act(() => source.onmessage?.({ data: '{}' }))
    const keys = spy.mock.calls.map((c) => c[0]?.queryKey)
    expect(keys).toEqual([['reviews'], ['workflows'], ['workflow', 'wf-1']])
  })

  it('exposes connection state and closes on unmount', () => {
    const { result, source, unmount } = setup()
    expect(result.current).toBe('connecting')
    act(() => source.onopen?.())
    expect(result.current).toBe('open')
    act(() => source.onerror?.())
    expect(result.current).toBe('error')
    unmount()
    expect(source.closed).toBe(true)
  })

  it('refetches every 30 s as a safety net', () => {
    vi.useFakeTimers()
    try {
      const { spy } = setup()
      expect(spy).not.toHaveBeenCalled()
      act(() => void vi.advanceTimersByTime(SAFETY_REFETCH_MS))
      expect(spy.mock.calls.map((c) => c[0]?.queryKey)[0]).toEqual(['reviews'])
    } finally {
      vi.useRealTimers()
    }
  })
})
