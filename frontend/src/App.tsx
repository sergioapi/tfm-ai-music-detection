import { useEffect, useRef, useState } from 'react'
import {
  analyzeAudio,
  getApiErrorMessage,
  isApiError,
  type AnalyzeResponse,
  type ApiError,
} from './api'
import { AnalysisResult } from './components/AnalysisResult'
import { AudioAnalysisForm } from './components/AudioAnalysisForm'
import { SiteFooter } from './components/SiteFooter'
import { SiteHeader } from './components/SiteHeader'

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
  const resultRef = useRef<HTMLDivElement | null>(null)

  const selectedFile =
    analysisState.status === 'idle' ? null : analysisState.file
  const isAnalyzing = analysisState.status === 'analyzing'
  const feedback = getFeedback(analysisState)

  useEffect(() => {
    if (analysisState.status === 'success') {
      resultRef.current?.scrollIntoView({
        behavior: 'smooth',
        block: 'start',
      })
    }
  }, [analysisState.status])

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

  function handleReset() {
    setAnalysisState({ status: 'idle' })
  }

  return (
    <div className="app-layout">
      <SiteHeader />
      <main className="app-shell">
        <section className="intro" aria-labelledby="app-title">
          <div className="analysis-card">
            <header className="analysis-header">
              <h1 id="app-title">Detector de música generada con IA</h1>
              <p>
                Sube un archivo de audio y obtén una estimación sobre su posible
                origen.
              </p>
            </header>
            {analysisState.status === 'success' ? (
              <div ref={resultRef}>
                <AnalysisResult result={analysisState.result} onReset={handleReset} />
              </div>
            ) : (
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
            )}
          </div>
        </section>
      </main>
      <SiteFooter />
    </div>
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
