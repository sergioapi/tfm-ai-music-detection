import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { buildAnalyzeResponse } from '../test/fixtures'
import { AnalysisResult } from './AnalysisResult'

describe('AnalysisResult', () => {
  it('shows the AI-generated label for ai_generated results', () => {
    render(<AnalysisResult result={buildAnalyzeResponse()} />)

    expect(screen.getByText('Posible generación con IA')).toBeInTheDocument()
  })

  it('shows the human-origin label for human results', () => {
    render(
      <AnalysisResult
        result={buildAnalyzeResponse({
          predicted_class: 'human',
          predicted_label: 0,
        })}
      />,
    )

    expect(screen.getByText('Posible origen humano')).toBeInTheDocument()
  })

  it('shows a safe fallback for unknown result classes', () => {
    render(
      <AnalysisResult
        result={buildAnalyzeResponse({ predicted_class: 'unexpected' })}
      />,
    )

    expect(
      screen.getByText('No se ha podido interpretar el resultado'),
    ).toBeInTheDocument()
  })

  it('shows the score as a numeric analysis score, not a percentage', () => {
    render(<AnalysisResult result={buildAnalyzeResponse({ ai_score: 0.428 })} />)

    expect(screen.getByText('Puntuación del análisis')).toBeInTheDocument()
    expect(screen.getByText('0,428')).toBeInTheDocument()
    expect(screen.queryByText('42,8 %')).not.toBeInTheDocument()
  })

  it('warns when the score is not a calibrated probability', () => {
    render(
      <AnalysisResult
        result={buildAnalyzeResponse({
          model: { score_is_calibrated_probability: false },
        })}
      />,
    )

    expect(
      screen.getByText('Esta puntuación no representa una probabilidad.'),
    ).toBeInTheDocument()
  })

  it('keeps the orientative limitation visible', () => {
    render(<AnalysisResult result={buildAnalyzeResponse()} />)

    expect(
      screen.getByText(
        'Este resultado es una estimación orientativa y no constituye una verificación definitiva de la autenticidad del audio.',
      ),
    ).toBeInTheDocument()
  })
})
