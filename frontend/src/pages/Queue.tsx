/** The review queue: five tabs, each a cursor-paginated list linking to the thread workspace. */
import type { UseInfiniteQueryResult, InfiniteData } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { useReviews, useWorkflows } from '@/api/queries'
import type { ReviewPage, WorkflowPage } from '@/api/schemas'
import { StatusBadge } from '@/components/StatusBadge'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { formatReceived } from '@/lib/utils'

interface Row {
  key: string
  workflowId: string
  sender: string
  subject: string
  receivedAt: string
  classification: string | null
  reason: string | null
}

function QueueRow({ row }: { row: Row }) {
  return (
    <li>
      <Link
        to={`/workflows/${row.workflowId}`}
        className="block rounded-lg border border-border bg-card p-4 transition-colors hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
      >
        <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
          <span className="font-medium">{row.sender}</span>
          <time dateTime={row.receivedAt} className="text-xs text-muted-foreground">
            {formatReceived(row.receivedAt)}
          </time>
        </div>
        <div className="mt-1">{row.subject}</div>
        <div className="mt-2 flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
          {row.classification && <StatusBadge>{row.classification}</StatusBadge>}
          {row.reason && <span>{row.reason}</span>}
        </div>
      </Link>
    </li>
  )
}

function QueueList({
  query,
  toRows,
  empty,
}: {
  query: UseInfiniteQueryResult<InfiniteData<ReviewPage | WorkflowPage>>
  toRows: (page: ReviewPage | WorkflowPage) => Row[]
  empty: string
}) {
  if (query.isPending) return <p className="text-sm text-muted-foreground">Loading…</p>
  if (query.isError) return <p role="alert" className="text-sm text-destructive">Could not load this list.</p>
  const rows = query.data.pages.flatMap(toRows)
  if (rows.length === 0) return <p className="text-sm text-muted-foreground">{empty}</p>
  return (
    <div className="space-y-3">
      <ul className="space-y-2">
        {rows.map((row) => (
          <QueueRow key={row.key} row={row} />
        ))}
      </ul>
      {query.hasNextPage && (
        <Button variant="outline" onClick={() => void query.fetchNextPage()} disabled={query.isFetchingNextPage}>
          Load more
        </Button>
      )}
    </div>
  )
}

const reviewRows = (page: ReviewPage | WorkflowPage): Row[] =>
  (page as ReviewPage).items.map((r) => ({
    key: r.id,
    workflowId: r.workflow_id,
    sender: r.sender,
    subject: r.subject,
    receivedAt: r.received_at,
    classification: r.classification,
    reason: r.reason,
  }))

const workflowRows = (page: ReviewPage | WorkflowPage): Row[] =>
  (page as WorkflowPage).items.map((w) => ({
    key: w.id,
    workflowId: w.id,
    sender: w.sender,
    subject: w.subject,
    receivedAt: w.received_at,
    classification: w.classification,
    reason: w.reason ?? w.error_code,
  }))

const COMPLETED = ['done_with_actions', 'done_no_action']
const FAILED = ['failed']

export function Queue() {
  const proposals = useReviews('proposal')
  const questions = useReviews('question')
  const notifications = useReviews('notification')
  const failed = useWorkflows(FAILED)
  const completed = useWorkflows(COMPLETED)

  const count = (q: { data?: InfiniteData<ReviewPage | WorkflowPage> }) => {
    const n = q.data?.pages.reduce((sum, p) => sum + p.items.length, 0) ?? 0
    return n > 0 ? ` (${n})` : ''
  }

  return (
    <Tabs defaultValue="reply">
      <TabsList aria-label="Queue filters">
        <TabsTrigger value="reply">Reply ready{count(proposals)}</TabsTrigger>
        <TabsTrigger value="needs-info">Needs information{count(questions)}</TabsTrigger>
        <TabsTrigger value="notifications">Notifications{count(notifications)}</TabsTrigger>
        <TabsTrigger value="failed">Failed{count(failed)}</TabsTrigger>
        <TabsTrigger value="completed">Completed</TabsTrigger>
      </TabsList>
      <TabsContent value="reply">
        <QueueList query={proposals} toRows={reviewRows} empty="No replies are waiting for review." />
      </TabsContent>
      <TabsContent value="needs-info">
        <QueueList query={questions} toRows={reviewRows} empty="Nothing needs your input." />
      </TabsContent>
      <TabsContent value="notifications">
        <QueueList query={notifications} toRows={reviewRows} empty="No notifications to review." />
      </TabsContent>
      <TabsContent value="failed">
        <QueueList query={failed} toRows={workflowRows} empty="No failed runs." />
      </TabsContent>
      <TabsContent value="completed">
        <QueueList query={completed} toRows={workflowRows} empty="No completed runs yet." />
      </TabsContent>
    </Tabs>
  )
}
