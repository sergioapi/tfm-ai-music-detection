import type { AnalyzeResponse } from '../api'
import {
  getVisibleClassDescription,
  getVisibleClassLabel,
} from '../results/classLabels'

type AnalysisResultProps = {
  result: AnalyzeResponse
  onReset: () => void
}

export function AnalysisResult({ result, onReset }: AnalysisResultProps) {
  const labelClassName = `result-label ${resultLabelModifier(result.predicted_class)}`

  return (
    <section className="result-card" aria-labelledby="result-title" role="status">
      <div className="result-header">
        <h2 id="result-title" className="visually-hidden">
          Resultado
        </h2>
        <button type="button" className="secondary-action" onClick={onReset}>
          Analizar otra canción
        </button>
      </div>
      <div className="result-main">
        <p className={labelClassName}>
          {getVisibleClassLabel(result.predicted_class)}
        </p>
        <p className="result-description">
          {getVisibleClassDescription(result.predicted_class)}
        </p>
      </div>
    </section>
  )
}

function resultLabelModifier(predictedClass: string): string {
  if (predictedClass === 'ai_generated') {
    return 'result-label-ai'
  }
  if (predictedClass === 'human') {
    return 'result-label-human'
  }
  return 'result-label-unknown'
}
