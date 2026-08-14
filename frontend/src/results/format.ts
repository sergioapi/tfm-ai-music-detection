const SCORE_FORMATTER = new Intl.NumberFormat('es-ES', {
  maximumFractionDigits: 3,
})

const DURATION_FORMATTER = new Intl.NumberFormat('es-ES', {
  maximumFractionDigits: 1,
})

const ANALYSIS_TIME_FORMATTER = new Intl.NumberFormat('es-ES', {
  maximumFractionDigits: 3,
})

const COUNT_FORMATTER = new Intl.NumberFormat('es-ES', {
  maximumFractionDigits: 0,
})

export function formatScore(value: number): string {
  return SCORE_FORMATTER.format(value)
}

export function formatDuration(seconds: number): string {
  return `${DURATION_FORMATTER.format(seconds)} s`
}

export function formatAnalysisTime(seconds: number): string {
  return `${ANALYSIS_TIME_FORMATTER.format(seconds)} s`
}

export function formatCount(value: number): string {
  return COUNT_FORMATTER.format(value)
}
