/**
 * Settings: the two pause switches, today's model spend against the daily budget, the three
 * preference namespaces, and (development backends only) a fixture injector and fault toggles.
 *
 * Preferences are versioned on the server. Each save sends the revision the form was loaded
 * from; if another tab saved in between, the server answers 409 and the form reloads.
 */
import { useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { z } from 'zod'
import { ApiError, api } from '@/api/client'
import { useFixtures, useFlags, usePreferences, useUsage } from '@/api/queries'
import { FaultsOut, FlagsOut, InjectOut, PreferenceOut, faultMode } from '@/api/schemas'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { toast } from 'sonner'

const card = 'space-y-3 rounded-lg border border-border bg-card p-4'

function PauseSwitches() {
  const flags = useFlags()
  const queryClient = useQueryClient()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function set(field: 'ingestion_paused' | 'outbound_paused', value: boolean) {
    setBusy(true)
    setError(null)
    try {
      queryClient.setQueryData(['flags'], await api.patch('/flags', { [field]: value }, FlagsOut))
    } catch {
      setError('Could not change the setting. Try again.')
    } finally {
      setBusy(false)
    }
  }

  const row = 'flex items-center gap-2 text-sm'
  return (
    <section aria-labelledby="pause-heading" className={card}>
      <h2 id="pause-heading" className="text-base font-semibold">
        Pauses
      </h2>
      <label className={row}>
        <input
          type="checkbox"
          checked={flags.data?.ingestion_paused ?? false}
          disabled={!flags.data || busy}
          onChange={(e) => void set('ingestion_paused', e.target.checked)}
        />
        Pause ingestion
      </label>
      <label className={row}>
        <input
          type="checkbox"
          checked={flags.data?.outbound_paused ?? false}
          disabled={!flags.data || busy}
          onChange={(e) => void set('outbound_paused', e.target.checked)}
        />
        Pause outgoing
      </label>
      <p className="text-xs text-muted-foreground">
        Pausing outgoing stops new sends and calendar writes; anything already sent is still reconciled.
      </p>
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
    </section>
  )
}

function UsageToday() {
  const usage = useUsage()
  const usd = (n: number) => `$${n.toFixed(2)}`
  return (
    <section aria-labelledby="usage-heading" className={card}>
      <h2 id="usage-heading" className="text-base font-semibold">
        Model usage
      </h2>
      {usage.isPending && <p className="text-sm text-muted-foreground">Loading…</p>}
      {usage.isError && <p className="text-sm text-destructive">Could not load usage.</p>}
      {usage.data && (
        <>
          <p className="text-sm">
            {usd(usage.data.today.cost_usd)} of {usd(usage.data.today.budget_usd)} used today
          </p>
          <meter
            aria-label="Share of today's budget used"
            className="h-2 w-full"
            min={0}
            max={usage.data.today.budget_usd || 1}
            high={(usage.data.today.budget_usd || 1) * 0.8}
            value={usage.data.today.cost_usd}
          />
        </>
      )}
    </section>
  )
}

// ---- Preferences ----

type FieldKind = 'text' | 'optional' | 'textarea' | 'lines' | 'number' | 'time' | 'days'
interface Field {
  key: string
  label: string
  kind: FieldKind
}
interface Namespace {
  name: 'triage' | 'writing' | 'scheduling'
  title: string
  defaults: Record<string, unknown>
  fields: Field[]
}

const NAMESPACES: Namespace[] = [
  {
    name: 'triage',
    title: 'Triage',
    defaults: { vip: [], ignore_senders: [], notes: null },
    fields: [
      { key: 'vip', label: 'VIP senders (one per line)', kind: 'lines' },
      { key: 'ignore_senders', label: 'Ignored senders (one per line)', kind: 'lines' },
      { key: 'notes', label: 'Triage notes', kind: 'textarea' },
    ],
  },
  {
    name: 'writing',
    title: 'Writing',
    defaults: { tone: null, language: null, signature: null },
    fields: [
      { key: 'tone', label: 'Tone', kind: 'optional' },
      { key: 'language', label: 'Language', kind: 'optional' },
      { key: 'signature', label: 'Signature', kind: 'textarea' },
    ],
  },
  {
    name: 'scheduling',
    title: 'Scheduling',
    defaults: {
      time_zone: 'UTC',
      work_start: '09:00',
      work_end: '17:00',
      work_days: [0, 1, 2, 3, 4],
      buffer_minutes: 10,
      default_duration_minutes: 30,
    },
    fields: [
      { key: 'time_zone', label: 'Time zone (IANA, e.g. Europe/Paris)', kind: 'text' },
      { key: 'work_start', label: 'Workday starts', kind: 'time' },
      { key: 'work_end', label: 'Workday ends', kind: 'time' },
      { key: 'work_days', label: 'Working days', kind: 'days' },
      { key: 'buffer_minutes', label: 'Buffer between meetings (minutes)', kind: 'number' },
      { key: 'default_duration_minutes', label: 'Default meeting length (minutes)', kind: 'number' },
    ],
  },
]

const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'] // Python weekday(): 0 = Monday

/** Form state is text for every input; `toValue` converts it back to the API's types. */
function toForm(ns: Namespace, value: Record<string, unknown>): Record<string, unknown> {
  const merged = { ...ns.defaults, ...value }
  const form: Record<string, unknown> = { ...merged }
  for (const f of ns.fields) {
    const v = merged[f.key]
    if (f.kind === 'lines') form[f.key] = Array.isArray(v) ? v.join('\n') : ''
    else if (f.kind === 'days') form[f.key] = Array.isArray(v) ? v : []
    else form[f.key] = v === null || v === undefined ? '' : String(v)
  }
  return form
}

function toValue(ns: Namespace, form: Record<string, unknown>): Record<string, unknown> {
  const value: Record<string, unknown> = { ...form }
  for (const f of ns.fields) {
    const v = form[f.key]
    if (f.kind === 'lines')
      value[f.key] = String(v)
        .split('\n')
        .map((s) => s.trim())
        .filter(Boolean)
    else if (f.kind === 'number') value[f.key] = Number(v)
    else if (f.kind === 'optional' || f.kind === 'textarea') value[f.key] = String(v).trim() ? String(v) : null
  }
  return value
}

function prefError(e: unknown): string {
  if (e instanceof ApiError && e.status === 409)
    return 'These preferences were changed in another tab. The latest version has been loaded; apply your change again.'
  if (e instanceof ApiError && e.status === 422) {
    const errors = (e.body as { errors?: { loc?: unknown[]; msg?: string }[] } | null)?.errors ?? []
    const detail = errors.map((x) => `${String(x.loc?.[x.loc.length - 1] ?? 'value')}: ${x.msg ?? 'invalid'}`)
    return `Some values are invalid${detail.length ? `: ${detail.join('; ')}` : ''}.`
  }
  return 'Could not save. Try again.'
}

function PreferenceForm({ ns, current }: { ns: Namespace; current?: z.infer<typeof PreferenceOut> }) {
  const queryClient = useQueryClient()
  const [form, setForm] = useState(() => toForm(ns, current?.value ?? {}))
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const id = (key: string) => `pref-${ns.name}-${key}`

  async function save() {
    setSaving(true)
    setError(null)
    try {
      await api.patch(
        `/preferences/${ns.name}`,
        { value: toValue(ns, form), expected_revision: current?.revision ?? 0 },
        PreferenceOut,
      )
      toast.success(`${ns.title} preferences saved`)
      await queryClient.invalidateQueries({ queryKey: ['preferences'] })
    } catch (e) {
      setError(prefError(e))
      if (e instanceof ApiError && e.status === 409) await queryClient.invalidateQueries({ queryKey: ['preferences'] })
    } finally {
      setSaving(false)
    }
  }

  const set = (key: string, v: unknown) => setForm((f) => ({ ...f, [key]: v }))
  return (
    <form
      aria-labelledby={id('heading')}
      className={card}
      onSubmit={(e) => {
        e.preventDefault()
        void save()
      }}
    >
      <h3 id={id('heading')} className="text-sm font-semibold">
        {ns.title}{' '}
        <span className="font-normal text-muted-foreground">
          {current ? `revision ${current.revision}` : 'not set yet'}
        </span>
      </h3>
      {ns.fields.map((f) => {
        if (f.kind === 'days') {
          const days = form[f.key] as number[]
          return (
            <fieldset key={f.key} className="space-y-1 text-sm">
              <legend className="font-medium">{f.label}</legend>
              <div className="flex flex-wrap gap-3">
                {DAYS.map((d, i) => (
                  <label key={d} className="flex items-center gap-1">
                    <input
                      type="checkbox"
                      checked={days.includes(i)}
                      onChange={(e) =>
                        set(f.key, e.target.checked ? [...days, i].sort((a, b) => a - b) : days.filter((x) => x !== i))
                      }
                    />
                    {d}
                  </label>
                ))}
              </div>
            </fieldset>
          )
        }
        const common = {
          id: id(f.key),
          value: String(form[f.key] ?? ''),
          onChange: (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => set(f.key, e.target.value),
        }
        return (
          <div key={f.key} className="space-y-1 text-sm">
            <label htmlFor={id(f.key)} className="font-medium">
              {f.label}
            </label>
            {f.kind === 'textarea' || f.kind === 'lines' ? (
              <Textarea rows={3} {...common} />
            ) : (
              <Input type={f.kind === 'number' ? 'number' : f.kind === 'time' ? 'time' : 'text'} {...common} />
            )}
          </div>
        )
      })}
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
      <Button type="submit" size="sm" disabled={saving} aria-label={`Save ${ns.name} preferences`}>
        Save
      </Button>
    </form>
  )
}

function Preferences() {
  const prefs = usePreferences()
  return (
    <section aria-labelledby="prefs-heading" className="space-y-3">
      <h2 id="prefs-heading" className="text-base font-semibold">
        Preferences
      </h2>
      {prefs.isPending && <p className="text-sm text-muted-foreground">Loading…</p>}
      {prefs.isError && <p className="text-sm text-destructive">Could not load preferences.</p>}
      {prefs.data && (
        <div className="grid gap-4 md:grid-cols-2">
          {NAMESPACES.map((ns) => {
            const current = prefs.data[ns.name]
            // Keyed by revision: a newer server revision resets the form to it.
            return <PreferenceForm key={`${ns.name}:${current?.revision ?? 0}`} ns={ns} current={current} />
          })}
        </div>
      )}
    </section>
  )
}

// ---- Dev tools ----

type FaultMode = z.infer<typeof faultMode>
const FAULTS: FaultMode[] = faultMode.options

function DevTools() {
  const fixtures = useFixtures()
  const [fixture, setFixture] = useState('')
  const [faults, setFaults] = useState<z.infer<typeof FaultsOut>>({ send: 'ok', calendar_create: 'ok' })
  const [busy, setBusy] = useState(false)
  if (!fixtures.isSuccess) return null // production backends answer 404: no dev tools

  async function inject() {
    if (!fixture) return
    setBusy(true)
    try {
      const out = await api.post('/dev/inject', { fixture_id: fixture }, InjectOut)
      toast.success(`Injected ${out.fixture_id} (${out.results.length} messages)`)
    } catch {
      toast.error('Injection failed')
    } finally {
      setBusy(false)
    }
  }

  async function setFault(key: keyof typeof faults, mode: FaultMode) {
    try {
      setFaults(await api.patch('/dev/faults', { [key]: mode }, FaultsOut))
    } catch {
      toast.error('Could not change the fault')
    }
  }

  const select = 'h-9 rounded-md border border-border bg-card px-2 text-sm'
  return (
    <section aria-labelledby="dev-heading" className={card}>
      <h2 id="dev-heading" className="text-base font-semibold">
        Developer tools
      </h2>
      <div className="flex flex-wrap items-end gap-2 text-sm">
        <div className="space-y-1">
          <label htmlFor="dev-fixture" className="block font-medium">
            Fixture
          </label>
          <select id="dev-fixture" className={select} value={fixture} onChange={(e) => setFixture(e.target.value)}>
            <option value="">Choose…</option>
            {fixtures.data.map((f) => (
              <option key={f.id} value={f.id}>
                {f.id} — {f.subject} ({f.classification})
              </option>
            ))}
          </select>
        </div>
        <Button onClick={() => void inject()} disabled={!fixture || busy}>
          Inject fixture
        </Button>
      </div>
      <div className="flex flex-wrap gap-4 text-sm">
        {(
          [
            ['send', 'Send fault'],
            ['calendar_create', 'Calendar fault'],
          ] as const
        ).map(([key, label]) => (
          <div key={key} className="space-y-1">
            <label htmlFor={`dev-fault-${key}`} className="block font-medium">
              {label}
            </label>
            <select
              id={`dev-fault-${key}`}
              className={select}
              value={faults[key]}
              onChange={(e) => void setFault(key, e.target.value as FaultMode)}
            >
              {FAULTS.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
        ))}
      </div>
      <p className="text-xs text-muted-foreground">
        Faults apply to the API process; a worker running separately keeps its own until restarted.
      </p>
    </section>
  )
}

export function Settings() {
  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">Settings</h1>
      <div className="grid gap-4 md:grid-cols-2">
        <PauseSwitches />
        <UsageToday />
      </div>
      <Preferences />
      <DevTools />
    </div>
  )
}
