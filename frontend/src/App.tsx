/**
 * App shell: routes, the auth gate and the 401 redirect.
 *
 * `AppRoutes` is separate from the providers so tests can mount it in a MemoryRouter with
 * their own QueryClient.
 */
import { useQueryClient } from '@tanstack/react-query'
import { useEffect } from 'react'
import { Link, Navigate, NavLink, Outlet, Route, Routes, useNavigate } from 'react-router-dom'
import { z } from 'zod'
import { ApiError, api, onUnauthorized } from '@/api/client'
import { useMe } from '@/api/queries'
import { StatusStrip } from '@/components/StatusStrip'
import { Button } from '@/components/ui/button'
import { Toaster } from '@/components/ui/sonner'
import { Login } from '@/pages/Login'
import { ActivityPage, SettingsPage, WorkflowPage } from '@/pages/Placeholders'
import { Queue } from '@/pages/Queue'

function Shell() {
  const me = useMe()
  const queryClient = useQueryClient()
  const navigate = useNavigate()

  async function logout() {
    try {
      await api.post('/auth/logout', {}, z.object({ status: z.string() }))
    } finally {
      queryClient.clear()
      navigate('/login', { replace: true })
    }
  }

  if (me.isPending) return <p className="p-6 text-sm text-muted-foreground">Loading…</p>
  if (me.isError) {
    if (me.error instanceof ApiError && me.error.status === 401) return <Navigate to="/login" replace />
    return <p role="alert" className="p-6 text-sm text-destructive">Could not reach the server.</p>
  }

  const link = ({ isActive }: { isActive: boolean }) =>
    isActive ? 'font-semibold text-foreground' : 'text-muted-foreground hover:text-foreground'
  return (
    <div className="mx-auto w-full max-w-5xl px-4 py-4">
      <header className="mb-6 space-y-3 border-b border-border pb-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <Link to="/" className="text-lg font-semibold">
            Email assistant
          </Link>
          <nav aria-label="Main" className="flex items-center gap-4 text-sm">
            <NavLink to="/" end className={link}>
              Queue
            </NavLink>
            <NavLink to="/activity" className={link}>
              Activity
            </NavLink>
            <NavLink to="/settings" className={link}>
              Settings
            </NavLink>
            <Button variant="ghost" size="sm" onClick={() => void logout()}>
              Log out
            </Button>
          </nav>
        </div>
        <StatusStrip />
      </header>
      <main>
        <Outlet />
      </main>
    </div>
  )
}

export function AppRoutes() {
  const navigate = useNavigate()
  useEffect(() => {
    onUnauthorized(() => navigate('/login', { replace: true }))
    return () => onUnauthorized(null)
  }, [navigate])

  return (
    <>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route element={<Shell />}>
          <Route index element={<Queue />} />
          <Route path="workflows/:id" element={<WorkflowPage />} />
          <Route path="activity" element={<ActivityPage />} />
          <Route path="settings" element={<SettingsPage />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      <Toaster />
    </>
  )
}
