/**
 * TanStack Query hooks over the API. Query keys start with the names the event stream
 * invalidates: `["reviews", ...]`, `["workflows", ...]`, `["workflow", id]`.
 */
import { useInfiniteQuery, useQuery } from '@tanstack/react-query'
import { api } from './client'
import { FlagsOut, HealthOut, MeOut, ReviewPage, WorkflowPage } from './schemas'

/** The pre-filtered lists behind the queue tabs. */
export type ReviewKind = 'proposal' | 'question' | 'notification'

export function useReviews(kind: ReviewKind) {
  return useInfiniteQuery({
    queryKey: ['reviews', kind],
    initialPageParam: null as string | null,
    queryFn: ({ pageParam }) => {
      const qs = new URLSearchParams({ kind, status: 'pending' })
      if (pageParam) qs.set('cursor', pageParam)
      return api.get(`/reviews?${qs}`, ReviewPage)
    },
    getNextPageParam: (last) => last.next_cursor,
  })
}

/** `statuses` repeat as `?status=a&status=b` (the API accepts a list). */
export function useWorkflows(statuses: string[]) {
  return useInfiniteQuery({
    queryKey: ['workflows', statuses],
    initialPageParam: null as string | null,
    queryFn: ({ pageParam }) => {
      const qs = new URLSearchParams()
      for (const s of statuses) qs.append('status', s)
      if (pageParam) qs.set('cursor', pageParam)
      return api.get(`/workflows?${qs}`, WorkflowPage)
    },
    getNextPageParam: (last) => last.next_cursor,
  })
}

export function useMe() {
  return useQuery({ queryKey: ['me'], queryFn: () => api.get('/auth/me', MeOut) })
}

export function useFlags() {
  return useQuery({
    queryKey: ['flags'],
    queryFn: () => api.get('/flags', FlagsOut),
    refetchInterval: 30_000,
  })
}

export function useHealth() {
  return useQuery({
    queryKey: ['health'],
    queryFn: () => api.get('/health/ready', HealthOut),
    refetchInterval: 30_000,
  })
}
