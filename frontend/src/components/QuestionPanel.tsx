/** The assistant needs information it cannot look up: the owner answers in free text. */
import { useState } from 'react'
import type { z } from 'zod'
import { api } from '@/api/client'
import { ReviewOutcome, type ReviewItemDetailOut } from '@/api/schemas'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { useSubmit } from '@/hooks/useSubmit'
import { toast } from 'sonner'

type Item = z.infer<typeof ReviewItemDetailOut>

export function QuestionPanel({ workflowId, item }: { workflowId: string; item: Item }) {
  const { submit, pending, error } = useSubmit(workflowId)
  const [answer, setAnswer] = useState('')
  const question = typeof item.payload.question === 'string' ? item.payload.question : ''
  const interruptId = item.interrupt_id

  async function send() {
    if (!interruptId || !answer.trim()) return
    const ok = await submit((request_id) =>
      api.post(
        `/questions/${encodeURIComponent(item.id)}/answer`,
        { answer: answer.trim(), request_id, interrupt_id: interruptId },
        ReviewOutcome,
      ),
    )
    if (ok) {
      toast.success('Answer sent')
      setAnswer('')
    }
  }

  return (
    <section aria-labelledby="question-heading" className="space-y-3 rounded-lg border border-border bg-card p-4">
      <h2 id="question-heading" className="text-base font-semibold">
        The assistant needs information
      </h2>
      <p className="whitespace-pre-wrap text-sm">{question}</p>
      <form
        className="space-y-2"
        onSubmit={(e) => {
          e.preventDefault()
          void send()
        }}
      >
        <label htmlFor="answer-text" className="text-sm font-medium">
          Your answer
        </label>
        <Textarea id="answer-text" rows={4} value={answer} onChange={(e) => setAnswer(e.target.value)} />
        <Button type="submit" disabled={pending || !interruptId || !answer.trim()}>
          Send answer
        </Button>
      </form>
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
    </section>
  )
}
