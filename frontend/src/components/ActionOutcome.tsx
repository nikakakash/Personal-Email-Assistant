/**
 * Where one proposal stands: Proposed → Approved → Accepted by provider, or Failed, or
 * Outcome unknown. "Unknown" means the provider call timed out after it may have been
 * accepted; the assistant never retries that blindly, so the owner is told to check.
 */
import type { z } from 'zod'
import type { ActionOut, ProposalOut } from '@/api/schemas'
import { StatusBadge, type Tone } from '@/components/StatusBadge'
import { formatReceived } from '@/lib/utils'

type Proposal = z.infer<typeof ProposalOut>
type Action = z.infer<typeof ActionOut>

const UNKNOWN_HELP = 'Check your Sent folder; the assistant will not resend automatically'

interface Outcome {
  label: string
  tone: Tone
  detail?: string
}

function outcomeOf(proposal: Proposal, action: Action | undefined): Outcome {
  if (action) {
    switch (action.status) {
      case 'succeeded':
        return { label: 'Accepted by provider', tone: 'good' }
      case 'failed_definite':
        return { label: 'Failed', tone: 'bad', detail: action.error_code ?? undefined }
      case 'outcome_unknown':
        return { label: 'Outcome unknown', tone: 'warn', detail: UNKNOWN_HELP }
      default:
        return { label: 'Approved', tone: 'neutral', detail: 'Sending…' }
    }
  }
  switch (proposal.status) {
    case 'awaiting_approval':
      return { label: 'Proposed', tone: 'neutral' }
    case 'approved':
      return { label: 'Approved', tone: 'neutral' }
    case 'executed':
      return { label: 'Accepted by provider', tone: 'good' }
    case 'rejected':
      return { label: 'Dismissed', tone: 'neutral' }
    case 'superseded':
      return { label: 'Replaced by a newer draft', tone: 'neutral' }
    case 'stale':
      return { label: 'Stale', tone: 'warn', detail: 'The thread changed after this draft was written.' }
    case 'cancelled_before_execution':
      return { label: 'Cancelled', tone: 'neutral' }
    default:
      return { label: proposal.status, tone: 'neutral' }
  }
}

const KIND: Record<string, string> = { send_reply: 'Reply', create_event: 'Calendar event' }

export function ActionOutcome({ proposal, action }: { proposal: Proposal; action?: Action }) {
  const o = outcomeOf(proposal, action)
  return (
    <div className="space-y-1 rounded-md border border-border p-3 text-sm">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-medium">
          {KIND[proposal.kind] ?? proposal.kind} · revision {proposal.revision}
        </span>
        <StatusBadge tone={o.tone}>{o.label}</StatusBadge>
        {action && (
          <time dateTime={action.updated_at} className="text-xs text-muted-foreground">
            {formatReceived(action.updated_at)}
          </time>
        )}
      </div>
      {o.detail && <p className={o.tone === 'warn' ? 'font-medium' : 'text-muted-foreground'}>{o.detail}</p>}
    </div>
  )
}
