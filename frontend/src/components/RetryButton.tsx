/**
 * Retry for a paused or recoverably failed run (POST /workflows/{id}/retry).
 *
 * The server decides what may be retried (`run.retryable`) and where the run continues: a
 * pause after approval re-runs every pre-send check with the same approval, so a retry never
 * sends something that is no longer valid. A send whose outcome is unknown is never offered.
 */
import { toast } from 'sonner'
import { api } from '@/api/client'
import { RetryOutcome } from '@/api/schemas'
import { Button } from '@/components/ui/button'
import { CONFLICT_MESSAGE, useSubmit } from '@/hooks/useSubmit'
import { PAUSE_EXPLANATIONS } from '@/lib/pause'

export function RetryButton({ workflowId, errorCode }: { workflowId: string; errorCode: string | null }) {
  const { submit, pending, error } = useSubmit(workflowId)
  async function retry() {
    const ok = await submit((request_id) =>
      api.post(`/workflows/${encodeURIComponent(workflowId)}/retry`, { request_id }, RetryOutcome),
    )
    if (ok) toast.success('Retrying')
  }
  const explanation = errorCode ? PAUSE_EXPLANATIONS[errorCode] : undefined
  return (
    <div className="space-y-2">
      {explanation && <p className="text-sm">{explanation}</p>}
      <Button onClick={() => void retry()} disabled={pending}>
        Retry
      </Button>
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error === CONFLICT_MESSAGE ? 'This run can no longer be retried. Reload the page.' : error}
        </p>
      )}
    </div>
  )
}
