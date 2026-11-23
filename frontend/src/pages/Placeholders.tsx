/** Routes Task 14 fills in; for now just a heading. */
import { useParams } from 'react-router-dom'

export function WorkflowPage() {
  const { id } = useParams()
  return <h1 className="text-xl font-semibold">Workflow {id}</h1>
}

export function ActivityPage() {
  return <h1 className="text-xl font-semibold">Activity</h1>
}

export function SettingsPage() {
  return <h1 className="text-xl font-semibold">Settings</h1>
}
