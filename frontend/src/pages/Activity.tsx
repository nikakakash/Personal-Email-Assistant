/**
 * Activity: every run's events, newest first, as a timeline. Each entry links to its run.
 * Only short scalar payload fields are shown (kind, decision, error code, …), never raw
 * model output.
 */
import { Link } from 'react-router-dom'
import { useActivity } from '@/api/queries'
import { Button } from '@/components/ui/button'
import { formatReceived } from '@/lib/utils'

const MAX_FIELDS = 4

function summary(payload: Record<string, unknown>): string {
  return Object.entries(payload)
    .filter(([, v]) => typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean')
    .filter(([k]) => !k.endsWith('_id') && k !== 'payload_hash')
    .slice(0, MAX_FIELDS)
    .map(([k, v]) => `${k.replaceAll('_', ' ')}: ${String(v).slice(0, 80)}`)
    .join(' · ')
}

export function Activity() {
  const query = useActivity()
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Activity</h1>
      {query.isPending && <p className="text-sm text-muted-foreground">Loading…</p>}
      {query.isError && (
        <p role="alert" className="text-sm text-destructive">
          Could not load activity.
        </p>
      )}
      {query.isSuccess && (
        <>
          {query.data.pages[0].items.length === 0 && <p className="text-sm text-muted-foreground">No activity yet.</p>}
          <ol aria-label="Activity timeline" className="relative space-y-3 border-l border-border pl-5">
            {query.data.pages
              .flatMap((p) => p.items)
              .map((e) => {
                const details = summary(e.payload)
                return (
                  <li key={e.id} className="relative">
                    <span aria-hidden className="absolute -left-[1.6rem] top-1.5 h-2.5 w-2.5 rounded-full bg-muted-foreground" />
                    <div className="flex flex-wrap items-baseline gap-x-3 text-sm">
                      <time dateTime={e.created_at} className="text-xs text-muted-foreground">
                        {formatReceived(e.created_at)}
                      </time>
                      <span className="font-medium">{e.type.replaceAll('_', ' ')}</span>
                      <Link to={`/workflows/${e.workflow_id}`} className="text-xs underline">
                        Open workflow
                      </Link>
                    </div>
                    {details && <p className="text-xs text-muted-foreground">{details}</p>}
                  </li>
                )
              })}
          </ol>
          {query.hasNextPage && (
            <Button variant="outline" onClick={() => void query.fetchNextPage()} disabled={query.isFetchingNextPage}>
              Load more
            </Button>
          )}
        </>
      )}
    </div>
  )
}
