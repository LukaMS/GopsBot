import type { Rank } from '../game/engine'
import type { WorkerRequest, WorkerResponse } from './types'

type WorkerRequestPayload =
  | { type: 'init'; modelUrl: string; sha256: string }
  | { type: 'choose'; observation: number[]; legalRanks: Rank[] }

interface PendingRequest {
  readonly resolve: (response: WorkerResponse) => void
  readonly reject: (error: Error) => void
  readonly timer: ReturnType<typeof setTimeout>
}

export class BotClient {
  private readonly worker = new Worker(new URL('./bot.worker.ts', import.meta.url), { type: 'module' })
  private readonly pending = new Map<number, PendingRequest>()
  private nextId = 1

  constructor() {
    this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const request = this.pending.get(event.data.requestId)
      if (!request) return
      clearTimeout(request.timer)
      this.pending.delete(event.data.requestId)
      if (event.data.type === 'error') request.reject(new Error(event.data.message))
      else request.resolve(event.data)
    }
    this.worker.onerror = (event) => {
      const error = new Error(event.message || 'The bot worker stopped unexpectedly.')
      for (const request of this.pending.values()) request.reject(error)
      this.pending.clear()
    }
  }

  private request(message: WorkerRequestPayload): Promise<WorkerResponse> {
    const requestId = this.nextId++
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(requestId)
        reject(new Error('The bot took too long to respond.'))
      }, 15000)
      this.pending.set(requestId, { resolve, reject, timer })
      this.worker.postMessage({ ...message, requestId } satisfies WorkerRequest)
    })
  }

  async initialize(modelUrl: string, sha256: string): Promise<void> {
    const response = await this.request({ type: 'init', modelUrl, sha256 })
    if (response.type !== 'ready') throw new Error('The bot did not finish loading.')
  }

  async choose(observation: Float32Array, legalRanks: readonly Rank[]): Promise<{ rank: Rank; latencyMs: number }> {
    const response = await this.request({ type: 'choose', observation: [...observation], legalRanks: [...legalRanks] })
    if (response.type !== 'choice') throw new Error('The bot did not return a bid.')
    return response
  }

  terminate(): void {
    this.worker.terminate()
    for (const request of this.pending.values()) request.reject(new Error('Bot client closed.'))
    this.pending.clear()
  }
}
