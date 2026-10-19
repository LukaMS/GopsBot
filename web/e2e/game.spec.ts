import { expect, test } from '@playwright/test'

test('a player can enter the table and complete a full match', async ({ page }) => {
  test.setTimeout(90000)
  await page.goto('/')
  const closeTutorial = page.getByRole('button', { name: 'Close tutorial' })
  if (await closeTutorial.isVisible()) await closeTutorial.click()
  await expect(page.getByRole('heading', { name: /nothing left to chance/i })).toBeVisible()
  await page.getByRole('button', { name: /Beginner Still learning/i }).click()
  await page.getByRole('button', { name: 'Play Beginner' }).click()
  try {
    await expect(page.getByText('SELF-PLAY BOT')).toBeVisible({ timeout: 50000 })
  } catch (cause) {
    const modelError = await page.getByRole('alert').textContent().catch(() => null)
    if (modelError) throw new Error(`The bot failed to load: ${modelError}`)
    throw cause
  }

  for (let round = 1; round <= 13; round += 1) {
    const card = page.getByRole('button', { name: /^Bid .* of spades/ }).first()
    await card.click()
    await page.getByRole('button', { name: /^Play / }).click()
    if (round < 13) await expect(page.getByText(new RegExp(`^${String(round).padStart(2, '0')}$`)).first()).toBeVisible({ timeout: 5000 })
    if (round === 12) {
      const latency = await page.locator('main.game-shell').getAttribute('data-last-inference-ms')
      expect(Number(latency)).toBeLessThan(100)
    }
  }
  await expect(page.getByText(/Match complete/)).toBeVisible({ timeout: 10000 })
  await expect(page.getByRole('button', { name: /Play again/i })).toBeVisible()
})
