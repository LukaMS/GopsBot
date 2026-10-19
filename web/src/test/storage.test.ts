import { clearPlayerData, loadPlayerData, markTutorialSeen, savePreference } from '../game/storage'

describe('local player data', () => {
  beforeEach(() => localStorage.clear())

  it('uses versioned defaults and persists preferences', () => {
    expect(loadPlayerData()).toMatchObject({ schemaVersion: 1, preferredDifficulty: 'casual', matches: [] })
    savePreference('expert')
    markTutorialSeen()
    expect(loadPlayerData()).toMatchObject({ preferredDifficulty: 'expert', tutorialSeen: true })
    expect(clearPlayerData().preferredDifficulty).toBe('casual')
  })

  it('recovers safely from corrupt browser data', () => {
    localStorage.setItem('gops.player-data.v1', '{broken')
    expect(loadPlayerData().matches).toEqual([])
  })
})

