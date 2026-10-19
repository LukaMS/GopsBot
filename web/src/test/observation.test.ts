import { applyBidPair, startMatch } from '../game/engine'
import { encodeBotObservation, OBSERVATION_SIZE } from '../game/observation'

describe('bot observation contract', () => {
  it('encodes only public table information in the versioned 56 features', () => {
    const state = startMatch(42)
    const observation = encodeBotObservation(state)
    expect(observation).toHaveLength(OBSERVATION_SIZE)
    expect([...observation.slice(0, 13)]).toEqual(Array(13).fill(1))
    expect([...observation.slice(13, 26)]).toEqual(Array(13).fill(1))
    expect(observation.slice(39, 52).reduce((sum, value) => sum + value, 0)).toBe(1)
    expect([...observation.slice(52)]).toEqual([0, 0, 0, 0])
  })

  it('updates remaining-card masks and normalized round values', () => {
    const next = applyBidPair(startMatch(42), 1, 13)
    const observation = encodeBotObservation(next)
    expect(observation[12]).toBe(0)
    expect(observation[13]).toBe(0)
    expect(observation[55]).toBeCloseTo(1 / 12)
  })
})

