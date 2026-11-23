/** A small coloured pill: used for classifications and for the header status strip. */
import { Badge } from '@/components/ui/badge'

export type Tone = 'neutral' | 'good' | 'warn' | 'bad'

const VARIANT = {
  neutral: 'secondary',
  good: 'default',
  warn: 'outline',
  bad: 'destructive',
} as const

export function StatusBadge({ tone = 'neutral', children }: { tone?: Tone; children: React.ReactNode }) {
  return <Badge variant={VARIANT[tone]}>{children}</Badge>
}
