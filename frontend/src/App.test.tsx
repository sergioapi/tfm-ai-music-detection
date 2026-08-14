import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { analyzeAudio } from './api'
import { buildAnalyzeResponse } from './test/fixtures'

vi.mock('./api', async () => {
  const actual = await vi.importActual<typeof import('./api')>('./api')

  return {
    ...actual,
    analyzeAudio: vi.fn(),
  }
})

const mockedAnalyzeAudio = vi.mocked(analyzeAudio)

describe('App', () => {
  beforeEach(() => {
    mockedAnalyzeAudio.mockReset()
  })

  it('runs the main analysis flow for a valid audio file', async () => {
    mockedAnalyzeAudio.mockResolvedValue(buildAnalyzeResponse())
    render(<App />)

    const file = new File(['audio'], 'song.wav', { type: 'audio/wav' })
    fireEvent.change(screen.getByLabelText('Archivo de audio'), {
      target: { files: [file] },
    })

    expect(screen.getByText(/Archivo seleccionado:/)).toBeInTheDocument()
    expect(screen.getByText('song.wav')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Analizar audio' }))

    expect(screen.getByRole('status')).toHaveTextContent(
      'Analizando el audio. Espera unos instantes.',
    )
    expect(mockedAnalyzeAudio).toHaveBeenCalledWith(file)

    expect(
      await screen.findByText('Posible generación con IA'),
    ).toBeInTheDocument()
    expect(screen.getByText('Análisis completado.')).toBeInTheDocument()
  })

  it('shows a user-facing error when analysis fails', async () => {
    mockedAnalyzeAudio.mockRejectedValue({
      kind: 'network',
      message: 'Could not contact the analysis service.',
    })
    render(<App />)

    fireEvent.change(screen.getByLabelText('Archivo de audio'), {
      target: {
        files: [new File(['audio'], 'song.mp3', { type: 'audio/mpeg' })],
      },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Analizar audio' }))

    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent(
        'No se ha podido contactar con el servicio de análisis.',
      )
    })
  })
})
