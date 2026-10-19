import type { Difficulty, MatchState, MatchWinner } from './engine'

const STORAGE_KEY = 'gops.player-data.v1'
const SCHEMA_VERSION = 1

export interface MatchSummary {
  readonly id: string
  readonly playedAt: string
  readonly difficulty: Difficulty
  readonly modelVersion: string
  readonly humanScore: number
  readonly botScore: number
  readonly result: MatchWinner
}

export interface PlayerData {
  readonly schemaVersion: 1
  readonly preferredDifficulty: Difficulty
  readonly tutorialSeen: boolean
  readonly matches: readonly MatchSummary[]
}

const defaults: PlayerData = {
  schemaVersion: SCHEMA_VERSION,
  preferredDifficulty: 'casual',
  tutorialSeen: false,
  matches: [],
}

function isDifficulty(value: unknown): value is Difficulty {
  return value === 'beginner' || value === 'casual' || value === 'skilled' || value === 'expert'
}

export function loadPlayerData(): PlayerData {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null') as Partial<PlayerData> | null
    if (!parsed || parsed.schemaVersion !== SCHEMA_VERSION) return defaults
    return {
      schemaVersion: SCHEMA_VERSION,
      preferredDifficulty: isDifficulty(parsed.preferredDifficulty) ? parsed.preferredDifficulty : 'casual',
      tutorialSeen: parsed.tutorialSeen === true,
      matches: Array.isArray(parsed.matches) ? parsed.matches.slice(0, 100) : [],
    }
  } catch {
    return defaults
  }
}

export function savePlayerData(data: PlayerData): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(data))
}

export function savePreference(difficulty: Difficulty): PlayerData {
  const data = { ...loadPlayerData(), preferredDifficulty: difficulty }
  savePlayerData(data)
  return data
}

export function markTutorialSeen(): PlayerData {
  const data = { ...loadPlayerData(), tutorialSeen: true }
  savePlayerData(data)
  return data
}

export function recordMatch(state: MatchState, difficulty: Difficulty, modelVersion: string): PlayerData {
  if (state.phase !== 'complete' || state.winner === null) throw new Error('Only complete matches can be recorded.')
  const summary: MatchSummary = {
    id: `${Date.now()}-${state.seed}`,
    playedAt: new Date().toISOString(),
    difficulty,
    modelVersion,
    humanScore: state.humanScore,
    botScore: state.botScore,
    result: state.winner,
  }
  const current = loadPlayerData()
  const data = { ...current, matches: [summary, ...current.matches].slice(0, 100) }
  savePlayerData(data)
  return data
}

export function clearPlayerData(): PlayerData {
  localStorage.removeItem(STORAGE_KEY)
  return defaults
}

