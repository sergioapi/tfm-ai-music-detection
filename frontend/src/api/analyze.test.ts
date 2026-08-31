import { describe, expect, it, vi } from 'vitest'
import { analyzeAudio } from './analyze'

describe('analyzeAudio', () => {
  it('aborts a timed-out POST without retrying it', async () => {
    vi.useFakeTimers()
    vi.stubEnv('VITE_API_BASE_URL', 'https://api.example.test')
    const fetchImpl = vi.fn<typeof fetch>((_url, init) =>
      new Promise((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')))
      }),
    )

    const request = analyzeAudio(
      new File(['audio'], 'song.wav', { type: 'audio/wav' }),
      { fetchImpl, timeoutMs: 10 },
    )
    const timeoutExpectation = expect(request).rejects.toMatchObject({ kind: 'timeout' })
    await vi.advanceTimersByTimeAsync(10)

    await timeoutExpectation
    expect(fetchImpl).toHaveBeenCalledTimes(1)
    vi.useRealTimers()
  })

  it('does not retry a failed POST', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'https://api.example.test')
    const fetchImpl = vi.fn<typeof fetch>().mockRejectedValue(new TypeError('network'))

    await expect(
      analyzeAudio(new File(['audio'], 'song.wav', { type: 'audio/wav' }), { fetchImpl }),
    ).rejects.toMatchObject({ kind: 'network' })
    expect(fetchImpl).toHaveBeenCalledTimes(1)
  })
})
