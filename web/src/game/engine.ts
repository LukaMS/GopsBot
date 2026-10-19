export const RANKS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] as const
export type Rank = (typeof RANKS)[number]
export type Difficulty = 'beginner' | 'casual' | 'skilled' | 'expert'
export type RoundWinner = 'human' | 'bot' | 'tie'
export type MatchWinner = 'human' | 'bot' | 'draw'
export type MatchPhase = 'awaiting_bid' | 'complete'

export interface RoundResult {
  readonly round: number
  readonly prize: Rank
  readonly potValue: number
  readonly humanBid: Rank
  readonly botBid: Rank
  readonly winner: RoundWinner
  readonly awarded: number
  readonly carriedAfter: number
  readonly humanScore: number
  readonly botScore: number
}

export interface MatchState {
  readonly seed: number
  readonly round: number
  readonly remainingHumanBids: readonly Rank[]
  readonly remainingBotBids: readonly Rank[]
  readonly remainingPrizes: readonly Rank[]
  readonly revealedPrizes: readonly Rank[]
  readonly currentPrize: Rank | null
  readonly carriedPot: number
  readonly humanScore: number
  readonly botScore: number
  readonly history: readonly RoundResult[]
  readonly phase: MatchPhase
  readonly winner: MatchWinner | null
}

export interface PlayerAction {
  readonly rank: Rank
}

function isRank(value: number): value is Rank {
  return Number.isInteger(value) && value >= 1 && value <= 13
}

function mulberry32(seed: number): () => number {
  let value = seed >>> 0
  return () => {
    value += 0x6d2b79f5
    let next = value
    next = Math.imul(next ^ (next >>> 15), next | 1)
    next ^= next + Math.imul(next ^ (next >>> 7), next | 61)
    return ((next ^ (next >>> 14)) >>> 0) / 4294967296
  }
}

export function shuffledPrizes(seed: number): Rank[] {
  const random = mulberry32(seed)
  const prizes = [...RANKS]
  for (let index = prizes.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(random() * (index + 1))
    ;[prizes[index], prizes[swapIndex]] = [prizes[swapIndex], prizes[index]]
  }
  return prizes
}

export function startMatch(seed: number): MatchState {
  if (!Number.isFinite(seed)) throw new Error('Match seed must be a finite number.')
  const normalizedSeed = Math.trunc(seed) >>> 0
  const prizes = shuffledPrizes(normalizedSeed)
  return {
    seed: normalizedSeed,
    round: 1,
    remainingHumanBids: [...RANKS],
    remainingBotBids: [...RANKS],
    remainingPrizes: prizes.slice(1),
    revealedPrizes: [prizes[0]],
    currentPrize: prizes[0],
    carriedPot: 0,
    humanScore: 0,
    botScore: 0,
    history: [],
    phase: 'awaiting_bid',
    winner: null,
  }
}

function removeRank(cards: readonly Rank[], bid: Rank): Rank[] {
  const index = cards.indexOf(bid)
  if (index === -1) throw new Error(`Bid ${bid} is not available.`)
  return [...cards.slice(0, index), ...cards.slice(index + 1)]
}

export function applyBidPair(state: MatchState, humanBid: number, botBid: number): MatchState {
  if (state.phase !== 'awaiting_bid' || state.currentPrize === null) {
    throw new Error('This match is already complete.')
  }
  if (!isRank(humanBid) || !isRank(botBid)) throw new Error('Bids must be ranks from 1 to 13.')

  const nextHumanBids = removeRank(state.remainingHumanBids, humanBid)
  const nextBotBids = removeRank(state.remainingBotBids, botBid)
  const potValue = state.carriedPot + state.currentPrize
  const roundWinner: RoundWinner = humanBid === botBid ? 'tie' : humanBid > botBid ? 'human' : 'bot'
  const humanScore = state.humanScore + (roundWinner === 'human' ? potValue : 0)
  const botScore = state.botScore + (roundWinner === 'bot' ? potValue : 0)
  const carriedAfter = roundWinner === 'tie' ? potValue : 0
  const result: RoundResult = {
    round: state.round,
    prize: state.currentPrize,
    potValue,
    humanBid,
    botBid,
    winner: roundWinner,
    awarded: roundWinner === 'tie' ? 0 : potValue,
    carriedAfter,
    humanScore,
    botScore,
  }
  const history = [...state.history, result]
  const isComplete = state.round === RANKS.length

  if (isComplete) {
    const winner: MatchWinner = humanScore === botScore ? 'draw' : humanScore > botScore ? 'human' : 'bot'
    return {
      ...state,
      round: RANKS.length,
      remainingHumanBids: nextHumanBids,
      remainingBotBids: nextBotBids,
      remainingPrizes: [],
      currentPrize: null,
      carriedPot: carriedAfter,
      humanScore,
      botScore,
      history,
      phase: 'complete',
      winner,
    }
  }

  const nextPrize = state.remainingPrizes[0]
  return {
    ...state,
    round: state.round + 1,
    remainingHumanBids: nextHumanBids,
    remainingBotBids: nextBotBids,
    remainingPrizes: state.remainingPrizes.slice(1),
    revealedPrizes: [...state.revealedPrizes, nextPrize],
    currentPrize: nextPrize,
    carriedPot: carriedAfter,
    humanScore,
    botScore,
    history,
  }
}

export function rankLabel(rank: Rank): string {
  if (rank === 1) return 'A'
  if (rank === 11) return 'J'
  if (rank === 12) return 'Q'
  if (rank === 13) return 'K'
  return String(rank)
}

export function createSeed(): number {
  const values = new Uint32Array(1)
  globalThis.crypto?.getRandomValues?.(values)
  return values[0] || (Date.now() >>> 0)
}

