/**
 * A proposed calendar event, shown in the event's own time zone with its UTC offset, plus
 * whether attendees will be emailed invitations and when availability was last checked.
 */
import type { z } from 'zod'
import type { AvailabilityCheck, EventPayload } from '@/api/schemas'
import { formatReceived } from '@/lib/utils'

/** "UTC+01:00" for the zone at that instant ("UTC+00:00" for UTC itself). */
function utcOffset(iso: string, timeZone: string): string {
  const part = new Intl.DateTimeFormat('en-US', { timeZone, timeZoneName: 'longOffset' })
    .formatToParts(new Date(iso))
    .find((p) => p.type === 'timeZoneName')?.value
  const offset = part?.replace('GMT', '') || '+00:00'
  return `UTC${offset}`
}

const dateIn = (iso: string, timeZone: string) =>
  new Date(iso).toLocaleDateString(undefined, {
    timeZone,
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  })

const timeIn = (iso: string, timeZone: string) =>
  new Date(iso).toLocaleTimeString(undefined, { timeZone, hour: '2-digit', minute: '2-digit', hourCycle: 'h23' })

export function MeetingReview({
  payload,
  availability = [],
}: {
  payload: EventPayload
  availability?: z.infer<typeof AvailabilityCheck>[]
}) {
  const zone = payload.time_zone
  const startDate = dateIn(payload.start, zone)
  const endDate = dateIn(payload.end, zone)
  const checkedAt = availability.map((a) => a.retrieved_at).find((t): t is string => Boolean(t))
  const row = 'grid grid-cols-[8rem_1fr] gap-2 py-1'
  return (
    <section
      aria-labelledby="meeting-heading"
      className="space-y-3 rounded-lg border border-border bg-card p-4 text-sm"
    >
      <h2 id="meeting-heading" className="text-base font-semibold">
        Meeting to create
      </h2>
      <dl className="divide-y divide-border">
        <div className={row}>
          <dt className="text-muted-foreground">Calendar</dt>
          <dd>{payload.calendar_id}</dd>
        </div>
        <div className={row}>
          <dt className="text-muted-foreground">Title</dt>
          <dd className="font-medium">{payload.title}</dd>
        </div>
        <div className={row}>
          <dt className="text-muted-foreground">Date</dt>
          <dd>{startDate}</dd>
        </div>
        <div className={row}>
          <dt className="text-muted-foreground">Time</dt>
          <dd>
            {timeIn(payload.start, zone)} – {endDate !== startDate && `${endDate} `}
            {timeIn(payload.end, zone)}{' '}
            <span className="text-muted-foreground">
              {zone} ({utcOffset(payload.start, zone)})
            </span>
          </dd>
        </div>
        <div className={row}>
          <dt className="text-muted-foreground">Attendees</dt>
          <dd>
            <ul>
              {payload.attendees.map((a) => (
                <li key={a}>{a}</li>
              ))}
            </ul>
          </dd>
        </div>
        <div className={row}>
          <dt className="text-muted-foreground">Location</dt>
          <dd>{payload.location || 'None'}</dd>
        </div>
      </dl>
      <p className="font-medium">Invitations will be sent: {payload.send_updates === 'all' ? 'yes' : 'no'}</p>
      <p className="text-muted-foreground">
        {checkedAt ? `Availability checked ${formatReceived(checkedAt)}` : 'Availability check time not recorded'}
      </p>
    </section>
  )
}
