/** Pure helpers for the reply draft editor: owner edits, dirtiness and the exact-text preview. */
import type { ReplyPayload } from '@/api/schemas'

export interface ReplyEdits {
  to: string
  cc: string
  body_text: string
}

export const editsOf = (p: ReplyPayload): ReplyEdits => ({
  to: p.to.join(', '),
  cc: p.cc.join(', '),
  body_text: p.body_text,
})

export const parseAddresses = (text: string): string[] =>
  text.split(/[\s,;]+/).filter((a) => a.length > 0)

export function isDirty(p: ReplyPayload, e: ReplyEdits): boolean {
  const same = (a: string[], b: string[]) => a.length === b.length && a.every((x, i) => x === b[i])
  return !same(parseAddresses(e.to), p.to) || !same(parseAddresses(e.cc), p.cc) || e.body_text !== p.body_text
}

/**
 * The owner-editable fields only. The server copies everything else (subject, threading
 * headers, evidence) from the reviewed revision and rejects any other field.
 */
export const editedPayload = (e: ReplyEdits) => ({
  to: parseAddresses(e.to),
  cc: parseAddresses(e.cc),
  body_text: e.body_text,
})

export function exactText(p: ReplyPayload): string {
  const lines = [`To: ${p.to.join(', ')}`]
  if (p.cc.length > 0) lines.push(`Cc: ${p.cc.join(', ')}`)
  lines.push(`Subject: ${p.subject}`, '', p.body_text)
  return lines.join('\n')
}
