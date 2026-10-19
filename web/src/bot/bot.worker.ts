/// <reference lib="webworker" />
import * as ort from 'onnxruntime-web/wasm'
import type { Rank } from '../game/engine'
import type { WorkerRequest, WorkerResponse } from './types'

let session: ort.InferenceSession | null = null
ort.env.wasm.numThreads = 1
ort.env.wasm.wasmPaths = new URL(`${import.meta.env.BASE_URL}ort/`, self.location.origin).href

function post(message: WorkerResponse): void {
  self.postMessage(message)
}

function sampleLegal(logits: Float32Array, legalRanks: readonly Rank[]): Rank {
  const legalLogits = legalRanks.map((rank) => logits[rank - 1])
  const maximum = Math.max(...legalLogits)
  const weights = legalLogits.map((value) => Math.exp(value - maximum))
  const total = weights.reduce((sum, weight) => sum + weight, 0)
  let threshold = Math.random() * total
  for (let index = 0; index < legalRanks.length; index += 1) {
    threshold -= weights[index]
    if (threshold <= 0) return legalRanks[index]
  }
  return legalRanks[legalRanks.length - 1]
}

async function fetchVerifiedModel(modelUrl: string, expectedSha256: string): Promise<Uint8Array> {
  const response = await fetch(modelUrl)
  if (!response.ok) throw new Error(`The policy download failed (${response.status}).`)
  const buffer = await response.arrayBuffer()
  const digest = await crypto.subtle.digest('SHA-256', buffer)
  const actualSha256 = Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('')
  if (actualSha256 !== expectedSha256.toLowerCase()) {
    throw new Error('The policy file did not pass its integrity check.')
  }
  return new Uint8Array(buffer)
}

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const message = event.data
  try {
    if (message.type === 'init') {
      const model = await fetchVerifiedModel(message.modelUrl, message.sha256)
      session = await ort.InferenceSession.create(model, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      })
      post({ type: 'ready', requestId: message.requestId })
      return
    }
    if (!session) throw new Error('The bot model has not loaded.')
    if (message.legalRanks.length === 0) throw new Error('The bot has no legal bid.')
    const started = performance.now()
    const observation = new ort.Tensor('float32', new Float32Array(message.observation), [1, 56])
    const output = await session.run({ observation })
    const logits = output.logits?.data
    if (!(logits instanceof Float32Array) || logits.length !== 13) {
      throw new Error('The model returned an invalid action vector.')
    }
    post({
      type: 'choice',
      requestId: message.requestId,
      rank: sampleLegal(logits, message.legalRanks),
      latencyMs: performance.now() - started,
    })
  } catch (error) {
    post({ type: 'error', requestId: message.requestId, message: error instanceof Error ? error.message : 'Unknown bot error' })
  }
}
