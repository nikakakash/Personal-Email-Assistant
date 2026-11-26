/** Log in through the real login page once; the journeys reuse the saved session cookies. */
import { expect, test as setup } from '@playwright/test'
import { AUTH_FILE } from '../../playwright.config.ts'

const PASSPHRASE = process.env.E2E_PASSPHRASE ?? 'e2e-owner-passphrase'

setup('log in as the owner', async ({ page }) => {
  await page.goto('/login')
  await page.getByLabel('Passphrase').fill(PASSPHRASE)
  await page.getByRole('button', { name: /sign in/i }).click()
  await expect(page.getByRole('tab', { name: /Reply ready/ })).toBeVisible()
  await page.context().storageState({ path: AUTH_FILE })
})
