/** An email worth knowing about but not answering by default: dismiss it, or ask for a reply. */
import type { z } from 'zod'
import { api } from '@/api/client'
import { ReviewOutcome, type ReviewItemDetailOut } from '@/api/schemas'
import { StatusBadge } from '@/components/StatusBadge'
import { Button } from '@/components/ui/button'
import { useSubmit } from '@/hooks/useSubmit'
import { toast } from 'sonner'

type Item = z.infer<typeof ReviewItemDetailOut>

const text = (v: unknown) => (typeof v === 'string' ? v : null)

export function NotificationPanel({ workflowId, item }: { workflowId: string; item: Item }) {
  const { submit, pending, error } = useSubmit(workflowId)
  const interruptId = item.interrupt_id
  const urgency = text(item.payload.urgency)
  const reason = text(item.payload.reason)

  async function decide(decision: 'dismiss' | 'respond') {
    if (!interruptId) return
    const ok = await submit((request_id) =>
      api.post(
        `/notifications/${encodeURIComponent(item.id)}/decision`,
        { decision, request_id, interrupt_id: interruptId },
        ReviewOutcome,
      ),
    )
    if (ok) toast.success(decision === 'dismiss' ? 'Dismissed' : 'The assistant will draft a reply')
  }

  return (
    <section aria-labelledby="notification-heading" className="space-y-3 rounded-lg border border-border bg-card p-4">
      <h2 id="notification-heading" className="flex items-center gap-2 text-base font-semibold">
        Notification
        {urgency && <StatusBadge tone={urgency === 'high' ? 'bad' : 'neutral'}>{urgency} urgency</StatusBadge>}
      </h2>
      {reason && <p className="text-sm">{reason}</p>}
      <div className="flex flex-wrap gap-2">
        <Button onClick={() => void decide('dismiss')} disabled={pending || !interruptId}>
          Dismiss notification
        </Button>
        <Button variant="outline" onClick={() => void decide('respond')} disabled={pending || !interruptId}>
          Draft a reply
        </Button>
      </div>
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
    </section>
  )
}
