/** Passphrase login. Maps the backend's 401 / 429 / 503 answers to readable messages. */
import { useQueryClient } from '@tanstack/react-query'
import { useState, type FormEvent } from 'react'
import { Navigate, useNavigate } from 'react-router-dom'
import { ApiError, api } from '@/api/client'
import { useMe } from '@/api/queries'
import { LoginOut } from '@/api/schemas'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'

function loginErrorMessage(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 401) return 'Wrong passphrase'
    if (err.status === 429) return 'Too many attempts, wait a minute'
    if (err.status === 503) return 'Server not configured — set OWNER_PASSPHRASE'
  }
  return 'Could not reach the server'
}

export function Login() {
  const [passphrase, setPassphrase] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const me = useMe()

  if (me.data) return <Navigate to="/" replace />

  async function submit(e: FormEvent) {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      await api.post('/auth/login', { passphrase }, LoginOut)
      await queryClient.invalidateQueries({ queryKey: ['me'] })
      navigate('/', { replace: true })
    } catch (err) {
      setError(loginErrorMessage(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="mx-auto mt-24 w-full max-w-sm px-4">
      <form onSubmit={submit} className="space-y-4 rounded-lg border border-border bg-card p-6">
        <h1 className="text-xl font-semibold">Email assistant</h1>
        <div className="space-y-1">
          <label htmlFor="passphrase" className="text-sm font-medium">
            Passphrase
          </label>
          <Input
            id="passphrase"
            type="password"
            autoComplete="current-password"
            value={passphrase}
            onChange={(e) => setPassphrase(e.target.value)}
            required
          />
        </div>
        {error && (
          <p role="alert" className="text-sm text-destructive">
            {error}
          </p>
        )}
        <Button type="submit" className="w-full" disabled={busy || !passphrase}>
          Sign in
        </Button>
      </form>
    </main>
  )
}
