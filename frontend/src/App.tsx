import { useState } from 'react'
import {
  analyzeAudio,
  getApiErrorMessage,
  isApiError,
  type AnalyzeResponse,
  type ApiError,
} from './api'
import { AnalysisResult } from './components/AnalysisResult'
import { AudioAnalysisForm } from './components/AudioAnalysisForm'

type AnalysisState =
  | { status: 'idle' }
  | { status: 'selected'; file: File }
  | { status: 'analyzing'; file: File }
  | { status: 'success'; file: File; result: AnalyzeResponse }
  | { status: 'error'; file: File | null; message: string }

function App() {
  const [analysisState, setAnalysisState] = useState<AnalysisState>({
    status: 'idle',
  })

  const selectedFile =
    analysisState.status === 'idle' ? null : analysisState.file
  const isAnalyzing = analysisState.status === 'analyzing'
  const feedback = getFeedback(analysisState)

  async function handleAnalyze() {
    if (!selectedFile || isAnalyzing) {
      return
    }

    setAnalysisState({ status: 'analyzing', file: selectedFile })

    try {
      const result = await analyzeAudio(selectedFile)
      setAnalysisState({ status: 'success', file: selectedFile, result })
    } catch (error) {
      const apiError = normalizeError(error)
      setAnalysisState({
        status: 'error',
        file: selectedFile,
        message: getApiErrorMessage(apiError),
      })
    }
  }

  return (
    <main className="app-shell">
      <section className="intro" aria-labelledby="app-title">
        <h1 id="app-title">
          Analiza si una canción puede haber sido generada con IA
        </h1>
        <p>
          Sube un archivo de audio y obtén una estimación basada en sus
          características sonoras.
        </p>
        <AudioAnalysisForm
          selectedFile={selectedFile}
          isAnalyzing={isAnalyzing}
          feedback={feedback}
          onFileAccepted={(file) => setAnalysisState({ status: 'selected', file })}
          onFileRejected={() =>
            setAnalysisState({
              status: 'error',
              file: null,
              message: getApiErrorMessage({
                kind: 'validation',
                code: 'unsupported_file_type',
                message: 'Unsupported audio file type',
              }),
            })
          }
          onFileCleared={() => setAnalysisState({ status: 'idle' })}
          onAnalyze={handleAnalyze}
        />
        {analysisState.status === 'success' ? (
          <AnalysisResult result={analysisState.result} />
        ) : null}
      </section>
    </main>
  )
}

function getFeedback(
  analysisState: AnalysisState,
): { kind: 'status' | 'error'; message: string } | null {
  switch (analysisState.status) {
    case 'idle':
    case 'selected':
      return null
    case 'analyzing':
      return {
        kind: 'status',
        message: 'Analizando el audio. Espera unos instantes.',
      }
    case 'success':
      return null
    case 'error':
      return {
        kind: 'error',
        message: analysisState.message,
      }
  }
}

function normalizeError(error: unknown): ApiError {
  if (isApiError(error)) {
    return error
  }

  return {
    kind: 'unexpected-response',
    message: 'Unexpected analysis error.',
  }
}

export default App
