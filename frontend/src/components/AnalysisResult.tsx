import type { AnalyzeResponse } from '../api'
import { getVisibleClassLabel } from '../results/classLabels'
import {
  formatAnalysisTime,
  formatCount,
  formatDuration,
  formatScore,
} from '../results/format'

const LIMITATIONS_TEXT =
  'Este resultado es una estimación orientativa y no constituye una verificación definitiva de la autenticidad del audio.'

type AnalysisResultProps = {
  result: AnalyzeResponse
}

export function AnalysisResult({ result }: AnalysisResultProps) {
  const shouldShowProbabilityNote =
    !result.model.score_is_calibrated_probability

  return (
    <section className="result-card" aria-labelledby="result-title">
      <div className="result-main">
        <p className="eyebrow">Resultado del análisis</p>
        <h2 id="result-title">Estimación</h2>
        <p className="result-label">
          {getVisibleClassLabel(result.predicted_class)}
        </p>
      </div>

      <div className="score-panel">
        <div>
          <p className="metric-label">Puntuación del análisis</p>
          <p className="score-value">{formatScore(result.ai_score)}</p>
        </div>
        {shouldShowProbabilityNote ? (
          <p className="score-note">
            Esta puntuación no representa una probabilidad.
          </p>
        ) : null}
      </div>

      <dl className="result-metrics" aria-label="Datos del análisis">
        <div>
          <dt>Duración</dt>
          <dd>{formatDuration(result.audio_duration_seconds)}</dd>
        </div>
        <div>
          <dt>Fragmentos analizados</dt>
          <dd>{formatCount(result.n_fragments)}</dd>
        </div>
        <div>
          <dt>Tiempo de análisis</dt>
          <dd>{formatAnalysisTime(result.timings.total_seconds)}</dd>
        </div>
        <div>
          <dt>Umbral de decisión</dt>
          <dd>{formatScore(result.decision_threshold)}</dd>
        </div>
      </dl>

      <div className="limitations">
        <h3>Cómo interpretar este resultado</h3>
        <p>{LIMITATIONS_TEXT}</p>
      </div>
    </section>
  )
}
