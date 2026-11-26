/**
 * The past emails a draft relies on (retrieved by RAG and cited as history:<id>), so the owner
 * can check those facts before approving. Their text is untrusted email content and is only
 * ever rendered as plain text.
 */
import type { HistorySource } from '@/api/schemas'

export function SourcesUsed({ sources }: { sources: HistorySource[] }) {
  if (sources.length === 0) return null
  return (
    <section
      aria-labelledby="sources-heading"
      className="space-y-2 rounded-lg border border-border bg-card p-4 text-sm"
    >
      <h2 id="sources-heading" className="text-base font-semibold">
        Sources used
      </h2>
      <p className="text-muted-foreground">
        Earlier emails this draft relies on. Check the facts before you approve.
      </p>
      <ul className="divide-y divide-border">
        {sources.map((s) => (
          <li key={s.ref} className="space-y-1 py-2">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <span className="font-medium break-words">{s.subject || '(no subject)'}</span>
              <time dateTime={s.date} className="text-xs text-muted-foreground">
                {s.date}
              </time>
            </div>
            <div className="text-xs text-muted-foreground break-all">{s.sender}</div>
            <blockquote className="border-l-2 border-border pl-3 whitespace-pre-wrap break-words">
              {s.snippet}
            </blockquote>
          </li>
        ))}
      </ul>
    </section>
  )
}
