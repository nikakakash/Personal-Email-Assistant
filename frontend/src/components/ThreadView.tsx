/**
 * The email thread, oldest first.
 *
 * HTML bodies arrive already sanitized by the server (no scripts, no remote images). They are
 * still shown only inside `<iframe sandbox="">`: no scripts, no forms, no same-origin access,
 * plus a Content-Security-Policy that blocks every network load. Messages without HTML show
 * their plain text.
 */
import type { z } from 'zod'
import type { MessageOut } from '@/api/schemas'
import { formatReceived } from '@/lib/utils'

type Message = z.infer<typeof MessageOut>

const CSP =
  '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'; img-src data:">'

function Body({ message }: { message: Message }) {
  if (message.body_html) {
    return (
      <iframe
        sandbox=""
        srcDoc={`${CSP}${message.body_html}`}
        title={`Message from ${message.sender}`}
        className="h-72 w-full rounded-md border border-border bg-white"
      />
    )
  }
  return <p className="whitespace-pre-wrap break-words text-sm">{message.body_text}</p>
}

export function ThreadView({ messages }: { messages: Message[] }) {
  return (
    <section aria-labelledby="thread-heading" className="space-y-3">
      <h2 id="thread-heading" className="text-base font-semibold">
        Thread ({messages.length})
      </h2>
      <ol className="space-y-3">
        {messages.map((m) => (
          <li key={m.id}>
            <article className="rounded-lg border border-border bg-card p-4">
              <header className="mb-3 space-y-0.5 text-sm">
                <div className="flex flex-wrap items-baseline justify-between gap-x-4">
                  <span className="font-medium">{m.sender}</span>
                  <time dateTime={m.received_at} className="text-xs text-muted-foreground">
                    {formatReceived(m.received_at)}
                  </time>
                </div>
                <div className="text-muted-foreground">
                  To: {m.to.join(', ') || '—'}
                  {m.cc.length > 0 && <> · Cc: {m.cc.join(', ')}</>}
                </div>
                <div className="font-medium">{m.subject}</div>
              </header>
              <Body message={m} />
            </article>
          </li>
        ))}
      </ol>
    </section>
  )
}
