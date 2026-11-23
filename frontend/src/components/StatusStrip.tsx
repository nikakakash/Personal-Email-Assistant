/** Header strip: backend readiness, live-stream connection and the two pause flags. */
import { useFlags, useHealth } from '@/api/queries'
import { StatusBadge } from '@/components/StatusBadge'
import { useEventStream } from '@/hooks/useEventStream'

export function StatusStrip() {
  const connection = useEventStream()
  const health = useHealth()
  const flags = useFlags()

  const backend = health.isError ? 'down' : health.data ? 'ok' : 'checking'
  return (
    <ul aria-label="System status" className="flex flex-wrap items-center gap-2 text-xs">
      <li>
        <StatusBadge tone={backend === 'ok' ? 'good' : backend === 'down' ? 'bad' : 'neutral'}>
          Backend: {backend}
        </StatusBadge>
      </li>
      <li>
        <StatusBadge tone={connection === 'open' ? 'good' : connection === 'error' ? 'bad' : 'neutral'}>
          Live: {connection}
        </StatusBadge>
      </li>
      {flags.data?.ingestion_paused && (
        <li>
          <StatusBadge tone="warn">Ingestion paused</StatusBadge>
        </li>
      )}
      {flags.data?.outbound_paused && (
        <li>
          <StatusBadge tone="warn">Outbound paused</StatusBadge>
        </li>
      )}
    </ul>
  )
}
