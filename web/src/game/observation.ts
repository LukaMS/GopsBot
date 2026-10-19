import type { MatchState, Rank } from './engine'

export const OBSERVATION_VERSION = 1
export const OBSERVATION_SIZE = 56

function mask(cards: readonly Rank[]): number[] {
  const held = new Set(cards)
  return Array.from({ length: 13 }, (_, index) => (held.has((index + 1) as Rank) ? 1 : 0))
}

function oneHot(rank: Rank | null): number[] {
  return Array.from({ length: 13 }, (_, index) => (rank === index + 1 ? 1 : 0))
}

export function encodeBotObservation(state: MatchState): Float32Array {
  if (state.phase !== 'awaiting_bid' || state.currentPrize === null) {
    throw new Error('A bot observation requires an active round.')
  }
  return new Float32Array([
    ...mask(state.remainingBotBids),
    ...mask(state.remainingHumanBids),
    ...mask(state.remainingPrizes),
    ...oneHot(state.currentPrize),
    state.carriedPot / 91,
    state.botScore / 91,
    state.humanScore / 91,
    (state.round - 1) / 12,
  ])
}

