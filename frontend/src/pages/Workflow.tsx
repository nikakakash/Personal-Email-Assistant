/**
 * The thread workspace for one run: the email thread beside whatever the assistant is waiting
 * on (a reply draft, a meeting, a question or a notification), the decision controls right
 * under it, the outcome of every proposal, and the run's recent events.
 *
 * On a narrow screen the two columns stack: thread first, then the draft with its controls
 * directly below it.
 */
import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import type { z } from 'zod'
import { ApiError } from '@/api/client'
import { useWorkflow } from '@/api/queries'
import {
  EventPayload,
  ProposalItemPayload,
  ReplyPayload,
  type ProposalOut,
  type ReviewItemDetailOut,
  type WorkflowOut,
} from '@/api/schemas'
import { ActionOutcome } from '@/components/ActionOutcome'
import { DraftEditor } from '@/components/DraftEditor'
import { MeetingReview } from '@/components/MeetingReview'
import { NotificationPanel } from '@/components/NotificationPanel'
import { QuestionPanel } from '@/components/QuestionPanel'
import { RetryButton } from '@/components/RetryButton'
import { SourcesUsed } from '@/components/SourcesUsed'
import { CancelButton, ReviewActions } from '@/components/ReviewActions'
import { StatusBadge } from '@/components/StatusBadge'
import { ThreadView } from '@/components/ThreadView'
import { editedPayload, editsOf, isDirty, type ReplyEdits } from '@/lib/draft'
import { formatReceived } from '@/lib/utils'

type Proposal = z.infer<typeof ProposalOut>
type Item = z.infer<typeof ReviewItemDetailOut>

const FINAL = new Set(['done_no_action', 'done_with_actions', 'failed', 'cancelled'])

/** The cited past emails stored on the review item (none when RAG is off). */
function sourcesOf(item: Item) {
  const context = ProposalItemPayload.safeParse(item.payload)
  return context.success ? (context.data.sources ?? []) : []
}

function ReplyReview({ workflowId, item, proposal }: { workflowId: string; item: Item; proposal: Proposal }) {
  const parsed = ReplyPayload.safeParse(proposal.payload)
  const [edits, setEdits] = useState<ReplyEdits | null>(parsed.success ? editsOf(parsed.data) : null)
  if (!parsed.success || !edits) return <p role="alert">This draft has an unexpected shape.</p>
  const dirty = isDirty(parsed.data, edits)
  return (
    <div className="space-y-4">
      <DraftEditor payload={parsed.data} edits={edits} onChange={setEdits} />
      <SourcesUsed sources={sourcesOf(item)} />
      <ReviewActions
        workflowId={workflowId}
        reviewed={{
          proposalId: proposal.id,
          revision: proposal.revision,
          payloadHash: proposal.payload_hash,
          interruptId: item.interrupt_id,
        }}
        dirty={dirty}
        editedPayload={dirty ? editedPayload(edits) : null}
      />
    </div>
  )
}

function EventReview({ workflowId, item, proposal }: { workflowId: string; item: Item; proposal: Proposal }) {
  const parsed = EventPayload.safeParse(proposal.payload)
  const context = ProposalItemPayload.safeParse(item.payload)
  if (!parsed.success) return <p role="alert">This meeting proposal has an unexpected shape.</p>
  return (
    <div className="space-y-4">
      <MeetingReview payload={parsed.data} availability={context.success ? context.data.availability : []} />
      <SourcesUsed sources={sourcesOf(item)} />
      <ReviewActions
        workflowId={workflowId}
        reviewed={{
          proposalId: proposal.id,
          revision: proposal.revision,
          payloadHash: proposal.payload_hash,
          interruptId: item.interrupt_id,
        }}
        dirty={false}
        editedPayload={null}
      />
    </div>
  )
}

function Pending({ detail }: { detail: WorkflowOut }) {
  const workflowId = detail.run.id
  const item = detail.review_items.find((i) => i.status === 'pending')
  if (!item) return null
  if (item.kind === 'question') return <QuestionPanel workflowId={workflowId} item={item} />
  if (item.kind === 'notification') return <NotificationPanel workflowId={workflowId} item={item} />
  const proposal = detail.proposals.find((p) => p.id === item.proposal_id)
  if (!proposal) return null
  // Keyed by revision hash: a new revision resets any unsaved edits.
  return proposal.kind === 'create_event' ? (
    <EventReview key={proposal.payload_hash} workflowId={workflowId} item={item} proposal={proposal} />
  ) : (
    <ReplyReview key={proposal.payload_hash} workflowId={workflowId} item={item} proposal={proposal} />
  )
}

function Outcomes({ detail }: { detail: WorkflowOut }) {
  const decided = detail.proposals.filter((p) => p.status !== 'awaiting_approval')
  if (decided.length === 0) return null
  return (
    <section aria-labelledby="outcomes-heading" className="space-y-2">
      <h2 id="outcomes-heading" className="text-base font-semibold">
        Actions
      </h2>
      {decided.map((p) => (
        <ActionOutcome key={p.id} proposal={p} action={detail.actions.find((a) => a.proposal_id === p.id)} />
      ))}
    </section>
  )
}

function RunEvents({ detail }: { detail: WorkflowOut }) {
  if (detail.events.length === 0) return null
  return (
    <section aria-labelledby="run-events-heading" className="space-y-2">
      <h2 id="run-events-heading" className="text-base font-semibold">
        Events
      </h2>
      <ol className="space-y-1 text-sm">
        {detail.events.map((e) => (
          <li key={e.id} className="flex flex-wrap gap-x-3">
            <time dateTime={e.created_at} className="text-xs text-muted-foreground">
              {formatReceived(e.created_at)}
            </time>
            <span>{e.type.replaceAll('_', ' ')}</span>
          </li>
        ))}
      </ol>
    </section>
  )
}

export function Workflow() {
  const { id = '' } = useParams()
  const query = useWorkflow(id)

  if (query.isPending) return <p className="text-sm text-muted-foreground">Loading…</p>
  if (query.isError) {
    const missing = query.error instanceof ApiError && (query.error.status === 404 || query.error.status === 422)
    return (
      <p role="alert" className="text-sm text-destructive">
        {missing ? 'This workflow does not exist.' : 'Could not load this workflow.'}{' '}
        <Link to="/" className="underline">
          Back to the queue
        </Link>
      </p>
    )
  }

  const detail = query.data
  const { run } = detail
  const latest = detail.thread[detail.thread.length - 1]
  const final = FINAL.has(run.status)
  const pendingProposal = detail.review_items.some((i) => i.status === 'pending' && i.kind === 'proposal')
  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <h1 className="text-xl font-semibold">{latest?.subject ?? 'Workflow'}</h1>
        <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
          <StatusBadge tone={run.status === 'failed' ? 'bad' : 'neutral'}>{run.status.replaceAll('_', ' ')}</StatusBadge>
          {run.classification && <StatusBadge>{run.classification}</StatusBadge>}
          {run.error_code && <span>Error: {run.error_code}</span>}
          {run.reason && <span>{run.reason}</span>}
        </div>
        {run.retryable && <RetryButton workflowId={run.id} errorCode={run.error_code} />}
        {!final && !pendingProposal && <CancelButton workflowId={run.id} label="Cancel workflow" />}
      </header>
      <div className="grid gap-6 lg:grid-cols-2">
        <ThreadView messages={detail.thread} />
        <div className="space-y-6">
          <Pending detail={detail} />
          <Outcomes detail={detail} />
          <RunEvents detail={detail} />
        </div>
      </div>
    </div>
  )
}
