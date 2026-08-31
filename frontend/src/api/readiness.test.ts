import { describe, expect, it, vi } from 'vitest'
import { READINESS_POLL_INTERVAL_MS, waitUntilReady } from './readiness'

const API_URL = 'https://api.example.test'

describe('waitUntilReady', () => {
  it('polls pending readiness until ready', async () => {
    vi.stubEnv('VITE_API_BASE_URL', API_URL)
    const fetchImpl = vi.fn<typeof fetch>()
      .mockResolvedValueOnce(readinessResponse(503, 'pending'))
      .mockResolvedValueOnce(readinessResponse(200, 'ready'))

    await waitUntilReady({ fetchImpl, sleep: async () => {} })

    expect(fetchImpl).toHaveBeenCalledTimes(2)
    expect(fetchImpl).toHaveBeenNthCalledWith(
      1,
      `${API_URL}/ready`,
      expect.objectContaining({ method: 'GET' }),
    )
  })

  it.each(['failed', 'unavailable'] as const)(
    'stops without retrying when readiness is %s',
    async (status) => {
      vi.stubEnv('VITE_API_BASE_URL', API_URL)
      const fetchImpl = vi.fn<typeof fetch>().mockResolvedValue(
        readinessResponse(503, status),
      )

      await expect(waitUntilReady({ fetchImpl })).rejects.toMatchObject({
        kind: 'service-unavailable',
      })
      expect(fetchImpl).toHaveBeenCalledTimes(1)
    },
  )

  it('recovers from a transient readiness network error within its budget', async () => {
    vi.stubEnv('VITE_API_BASE_URL', API_URL)
    const fetchImpl = vi.fn<typeof fetch>()
      .mockRejectedValueOnce(new TypeError('network interrupted'))
      .mockResolvedValueOnce(readinessResponse(200, 'ready'))

    await waitUntilReady({ fetchImpl, sleep: async () => {} })

    expect(fetchImpl).toHaveBeenCalledTimes(2)
  })

  it('stops after the finite readiness budget', async () => {
    vi.stubEnv('VITE_API_BASE_URL', API_URL)
    let now = 0
    const fetchImpl = vi.fn<typeof fetch>().mockImplementation(async () =>
      readinessResponse(503, 'pending'),
    )

    await expect(
      waitUntilReady({
        fetchImpl,
        timeoutMs: 10,
        pollIntervalMs: 5,
        now: () => now,
        sleep: async (milliseconds) => {
          now += milliseconds
        },
      }),
    ).rejects.toMatchObject({ kind: 'service-unavailable' })
    expect(fetchImpl).toHaveBeenCalledTimes(2)
  })

  it('does not request readiness when the signal is already aborted', async () => {
    vi.stubEnv('VITE_API_BASE_URL', API_URL)
    const controller = new AbortController()
    const fetchImpl = vi.fn<typeof fetch>()
    controller.abort()

    await expect(
      waitUntilReady({ fetchImpl, signal: controller.signal }),
    ).rejects.toMatchObject({ kind: 'aborted' })
    expect(fetchImpl).not.toHaveBeenCalled()
  })

  it('stops polling when cancelled between readiness checks', async () => {
    vi.useFakeTimers()
    vi.stubEnv('VITE_API_BASE_URL', API_URL)
    const controller = new AbortController()
    const fetchImpl = vi.fn<typeof fetch>().mockImplementation(async () =>
      readinessResponse(503, 'pending'),
    )
    const waiting = waitUntilReady({ fetchImpl, signal: controller.signal })
    const abortedExpectation = expect(waiting).rejects.toMatchObject({ kind: 'aborted' })

    await vi.advanceTimersByTimeAsync(0)
    expect(fetchImpl).toHaveBeenCalledTimes(1)

    controller.abort()
    await vi.advanceTimersByTimeAsync(READINESS_POLL_INTERVAL_MS)

    await abortedExpectation
    expect(fetchImpl).toHaveBeenCalledTimes(1)
    vi.useRealTimers()
  })

  it('handles cancellation while registering the sleep listener', async () => {
    vi.useFakeTimers()
    vi.stubEnv('VITE_API_BASE_URL', API_URL)
    const controller = new AbortController()
    const addEventListener = controller.signal.addEventListener.bind(controller.signal)
    let abortListenerRegistrations = 0
    const addEventListenerSpy = vi.spyOn(controller.signal, 'addEventListener')
      .mockImplementation((type, listener, options) => {
        addEventListener(type, listener, options)
        if (type === 'abort' && ++abortListenerRegistrations === 2) {
          controller.abort()
        }
      })
    const fetchImpl = vi.fn<typeof fetch>().mockImplementation(async () =>
      readinessResponse(503, 'pending'),
    )
    const waiting = waitUntilReady({ fetchImpl, signal: controller.signal })
    const abortedExpectation = expect(waiting).rejects.toMatchObject({ kind: 'aborted' })

    await vi.advanceTimersByTimeAsync(0)
    await abortedExpectation
    await vi.runAllTimersAsync()

    expect(fetchImpl).toHaveBeenCalledTimes(1)
    addEventListenerSpy.mockRestore()
    vi.useRealTimers()
  })
})

function readinessResponse(statusCode: number, status: string): Response {
  return new Response(JSON.stringify({ status }), {
    status: statusCode,
    headers: { 'Content-Type': 'application/json' },
  })
}
