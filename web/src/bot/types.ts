import type { Difficulty, Rank } from '../game/engine'

export interface ModelEntry {
  readonly difficulty: Difficulty
  readonly label: string
  readonly file: string
  readonly sha256: string
  readonly modelVersion: string
  readonly trainingRun: string
  readonly observationVersion: number
  readonly metrics: {
    readonly arenaElo: number
    readonly baselineWinRate: number
    readonly games: number
  }
}

export interface ModelManifest {
  readonly schemaVersion: 1
  readonly generatedAt: string
  readonly observationSize: 56
  readonly actionSize: 13
  readonly models: readonly ModelEntry[]
}

export type WorkerRequest =
  | { type: 'init'; requestId: number; modelUrl: string; sha256: string }
  | { type: 'choose'; requestId: number; observation: number[]; legalRanks: Rank[] }

export type WorkerResponse =
  | { type: 'ready'; requestId: number }
  | { type: 'choice'; requestId: number; rank: Rank; latencyMs: number }
  | { type: 'error'; requestId: number; message: string }
