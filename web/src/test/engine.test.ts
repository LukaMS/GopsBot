import fixture from '../../../contracts/fixtures/seed-42-match.json'
import { applyBidPair, shuffledPrizes, startMatch, type Rank } from '../game/engine'

describe('GOPS rules engine', () => {
  it('uses the cross-runtime seeded prize order', () => {
    expect(shuffledPrizes(fixture.seed)).toEqual(fixture.prizeOrder)
  })

  it('matches the shared complete-match fixture', () => {
    let state = startMatch(fixture.seed)
    for (let index = 0; index < 13; index += 1) {
      state = applyBidPair(state, fixture.humanBids[index], fixture.botBids[index])
    }
    expect({
      humanScore: state.humanScore,
      botScore: state.botScore,
      carriedPot: state.carriedPot,
      winner: state.winner,
    }).toEqual(fixture.expected)
    expect(state.phase).toBe('complete')
    expect(state.remainingHumanBids).toHaveLength(0)
    expect(state.remainingBotBids).toHaveLength(0)
  })

  it('carries every tied prize and leaves the final pot unawarded', () => {
    let state = startMatch(5)
    for (let rank = 1; rank <= 13; rank += 1) state = applyBidPair(state, rank, rank)
    expect(state.humanScore).toBe(0)
    expect(state.botScore).toBe(0)
    expect(state.carriedPot).toBe(91)
    expect(state.winner).toBe('draw')
  })

  it('rejects unavailable and invalid bids without mutating the state', () => {
    const original = startMatch(3)
    const next = applyBidPair(original, 1, 2)
    expect(original.remainingHumanBids).toContain(1)
    expect(() => applyBidPair(next, 1, 3)).toThrow('not available')
    expect(() => applyBidPair(next, 99, 3)).toThrow('ranks from 1 to 13')
  })

  it('conserves all 91 prize points across varied legal games', () => {
    for (let seed = 0; seed < 250; seed += 1) {
      let state = startMatch(seed)
      const human = shuffledPrizes(seed + 1000)
      const bot = shuffledPrizes(seed + 2000)
      for (let round = 0; round < 13; round += 1) {
        state = applyBidPair(state, human[round] as Rank, bot[round] as Rank)
      }
      expect(state.humanScore + state.botScore + state.carriedPot).toBe(91)
    }
  })
})

