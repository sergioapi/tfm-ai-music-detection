import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { buildAnalyzeResponse } from '../test/fixtures'
import { AnalysisResult } from './AnalysisResult'

describe('AnalysisResult', () => {
  it('shows the AI-generated label and explanation for ai_generated results', () => {
    render(<AnalysisResult result={buildAnalyzeResponse()} />)

    expect(screen.getByRole('status')).toBeInTheDocument()
    expect(screen.getByText('Resultado')).toBeInTheDocument()
    expect(screen.getByText('Posible generación con IA')).toHaveClass(
      'result-label-ai',
    )
    expect(
      screen.getByText(
        'El modelo ha identificado patrones acústicos más compatibles con música generada mediante IA.',
      ),
    ).toBeInTheDocument()
  })

  it('shows the human-origin label and explanation for human results', () => {
    render(
      <AnalysisResult
        result={buildAnalyzeResponse({
          predicted_class: 'human',
          predicted_label: 0,
        })}
      />,
    )

    expect(screen.getByText('Posible origen humano')).toHaveClass(
      'result-label-human',
    )
    expect(
      screen.getByText(
        'El modelo ha identificado patrones acústicos más compatibles con música de origen humano.',
      ),
    ).toBeInTheDocument()
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
})
