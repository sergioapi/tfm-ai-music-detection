import type { AnalyzeResponse } from '../api'
import {
  getVisibleClassDescription,
  getVisibleClassLabel,
} from '../results/classLabels'

type AnalysisResultProps = {
  result: AnalyzeResponse
}

export function AnalysisResult({ result }: AnalysisResultProps) {
  const labelClassName = `result-label ${resultLabelModifier(result.predicted_class)}`

  return (
    <section className="result-card" aria-labelledby="result-title" role="status">
      <div className="result-main">
        <h2 id="result-title">Resultado</h2>
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
