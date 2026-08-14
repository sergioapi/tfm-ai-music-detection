import type { AnalyzeResponse } from '../api'
import {
  getVisibleClassDescription,
  getVisibleClassLabel,
} from '../results/classLabels'

type AnalysisResultProps = {
  result: AnalyzeResponse
}

export function AnalysisResult({ result }: AnalysisResultProps) {
  return (
    <section className="result-card" aria-labelledby="result-title" role="status">
      <div className="result-main">
        <h2 id="result-title">Resultado</h2>
        <p className="result-label">
          {getVisibleClassLabel(result.predicted_class)}
        </p>
        <p className="result-description">
          {getVisibleClassDescription(result.predicted_class)}
        </p>
      </div>
    </section>
  )
}
