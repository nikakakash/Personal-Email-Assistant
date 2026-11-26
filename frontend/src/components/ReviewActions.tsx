/**
 * The owner's decision on one proposal revision.
 *
 * Every decision is bound to what the owner reviewed: the revision, its payload hash and the
 * interrupt id (spec §9). "Approve and send" approves the stored payload as is, so it is
 * disabled while the draft has unsaved edits; "Edit and approve" sends the edited payload and
 * is enabled only then. Each click gets its own request id (see `useSubmit`).
 */
import { useState } from 'react'
import { api } from '@/api/client'
import { CancelOutcome, DecisionOutcome, type DecisionIn } from '@/api/schemas'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { useSubmit } from '@/hooks/useSubmit'
import { toast } from 'sonner'

export interface ReviewedRevision {
  proposalId: string
  revision: number
  payloadHash: string
  interruptId: string | null
}

type Decision = 'approve' | 'edit_approve' | 'reject' | 'feedback'

export function CancelButton({
  workflowId,
  label = 'Cancel pending action',
}: {
  workflowId: string
  label?: string
}) {
  const { submit, pending, error } = useSubmit(workflowId)
  async function cancel() {
    const ok = await submit((request_id) =>
      api.post(`/workflows/${encodeURIComponent(workflowId)}/cancel`, { request_id }, CancelOutcome),
    )
    if (ok) toast.success('Cancelled')
  }
  return (
    <div className="space-y-1">
      <Button variant="ghost" onClick={() => void cancel()} disabled={pending}>
        {label}
      </Button>
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
    </div>
  )
}

export function ReviewActions({
  workflowId,
  reviewed,
  dirty,
  editedPayload,
}: {
  workflowId: string
  reviewed: ReviewedRevision
  dirty: boolean
  /** The full edited payload; only sent with edit_approve. */
  editedPayload: Record<string, unknown> | null
}) {
  const { submit, pending, error } = useSubmit(workflowId)
  const [feedbackOpen, setFeedbackOpen] = useState(false)
  const [feedback, setFeedback] = useState('')
  const ready = reviewed.interruptId !== null

  async function decide(decision: Decision, extra: Partial<DecisionIn> = {}) {
    if (!reviewed.interruptId) return
    const interrupt_id = reviewed.interruptId
    const ok = await submit((request_id) => {
      const body: DecisionIn = {
        decision,
        proposal_revision: reviewed.revision,
        payload_hash: reviewed.payloadHash,
        interrupt_id,
        request_id,
        ...extra,
      }
      return api.post(
        `/proposals/${encodeURIComponent(reviewed.proposalId)}/decision`,
        body,
        DecisionOutcome,
      )
    })
    if (ok) {
      toast.success(
        decision === 'reject' ? 'Dismissed' : decision === 'feedback' ? 'Feedback sent' : 'Approved',
      )
      setFeedbackOpen(false)
      setFeedback('')
    }
  }

  return (
    <section aria-label="Review actions" className="space-y-3">
      {!ready && <p className="text-sm text-muted-foreground">Waiting for the assistant to pause for review…</p>}
      <div className="flex flex-wrap gap-2">
        <Button onClick={() => void decide('approve')} disabled={!ready || pending || dirty}>
          Approve and send
        </Button>
        <Button
          variant="outline"
          onClick={() => editedPayload && void decide('edit_approve', { edited_payload: editedPayload })}
          disabled={!ready || pending || !dirty || !editedPayload}
        >
          Edit and approve
        </Button>
        <Button
          variant="outline"
          aria-expanded={feedbackOpen}
          aria-controls="feedback-form"
          onClick={() => setFeedbackOpen((v) => !v)}
          disabled={!ready || pending}
        >
          Give feedback
        </Button>
        <Button variant="outline" onClick={() => void decide('reject')} disabled={!ready || pending}>
          Dismiss
        </Button>
      </div>
      {feedbackOpen && (
        <form
          id="feedback-form"
          className="space-y-2"
          onSubmit={(e) => {
            e.preventDefault()
            if (feedback.trim()) void decide('feedback', { feedback: feedback.trim() })
          }}
        >
          <label htmlFor="feedback-text" className="text-sm font-medium">
            Feedback for the assistant
          </label>
          <Textarea id="feedback-text" rows={3} value={feedback} onChange={(e) => setFeedback(e.target.value)} />
          <Button type="submit" size="sm" disabled={pending || !feedback.trim()}>
            Send feedback
          </Button>
        </form>
      )}
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
      <CancelButton workflowId={workflowId} />
    </section>
  )
}
