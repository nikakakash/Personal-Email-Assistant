/**
 * The reply draft: To, Cc and the message are editable; the subject is not (the server keeps
 * it code-derived). "Exact text to be sent" shows the stored payload verbatim, which is what
 * a plain approval sends. Any edit makes the draft dirty; a dirty draft can only be submitted
 * as edit-and-approve, which creates a new revision that the server validates again.
 */
import type { ReplyPayload } from '@/api/schemas'
import { exactText, isDirty, type ReplyEdits } from '@/lib/draft'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'

export function DraftEditor({
  payload,
  edits,
  onChange,
  readOnly = false,
}: {
  payload: ReplyPayload
  edits: ReplyEdits
  onChange: (edits: ReplyEdits) => void
  readOnly?: boolean
}) {
  const dirty = isDirty(payload, edits)
  const field = 'space-y-1 text-sm'
  return (
    <section aria-labelledby="draft-heading" className="space-y-4 rounded-lg border border-border bg-card p-4">
      <h2 id="draft-heading" className="text-base font-semibold">
        Reply draft {dirty && <span className="text-sm font-normal text-muted-foreground">(edited)</span>}
      </h2>
      <div className={field}>
        <label htmlFor="draft-to" className="font-medium">
          To
        </label>
        <Input
          id="draft-to"
          value={edits.to}
          readOnly={readOnly}
          onChange={(e) => onChange({ ...edits, to: e.target.value })}
        />
      </div>
      <div className={field}>
        <label htmlFor="draft-cc" className="font-medium">
          Cc
        </label>
        <Input
          id="draft-cc"
          value={edits.cc}
          readOnly={readOnly}
          onChange={(e) => onChange({ ...edits, cc: e.target.value })}
        />
      </div>
      <div className={field}>
        <label htmlFor="draft-subject" className="font-medium">
          Subject
        </label>
        <Input id="draft-subject" value={payload.subject} readOnly className="bg-muted" />
      </div>
      <div className={field}>
        <label htmlFor="draft-body" className="font-medium">
          Message
        </label>
        <Textarea
          id="draft-body"
          rows={8}
          value={edits.body_text}
          readOnly={readOnly}
          onChange={(e) => onChange({ ...edits, body_text: e.target.value })}
        />
      </div>
      <section aria-labelledby="exact-heading" className="space-y-1">
        <h3 id="exact-heading" className="text-sm font-medium">
          Exact text to be sent
        </h3>
        {dirty && (
          <p className="text-xs text-muted-foreground">
            This is the stored version. Your edits become a new version when you choose Edit and approve.
          </p>
        )}
        <pre className="whitespace-pre-wrap break-words rounded-md bg-muted p-3 text-sm">{exactText(payload)}</pre>
      </section>
    </section>
  )
}
