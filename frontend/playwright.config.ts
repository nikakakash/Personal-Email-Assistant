/**
 * Playwright E2E config. The stack is started separately (compose E2E overlay, see
 * docs/setup.md); these tests only drive the browser and the API at E2E_BASE_URL.
 *
 * One worker, in order: every journey shares the one backend worker process and database.
 * The `setup` project logs in once and saves the session for the journeys.
 */
import { defineConfig, devices } from '@playwright/test'

export const BASE_URL = process.env.E2E_BASE_URL ?? 'http://127.0.0.1:8081'
export const AUTH_FILE = 'test-results/.auth/owner.json'

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 90_000,
  expect: { timeout: 30_000 },
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : 'list',
  use: {
    baseURL: BASE_URL,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [
    { name: 'setup', testMatch: /auth\.setup\.ts/ },
    {
      name: 'journeys',
      testMatch: /\.spec\.ts/,
      dependencies: ['setup'],
      use: { ...devices['Desktop Chrome'], storageState: AUTH_FILE },
    },
  ],
})
