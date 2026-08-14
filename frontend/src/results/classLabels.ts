const CLASS_LABELS: Record<string, string> = {
  ai_generated: 'Posible generación con IA',
  human: 'Posible origen humano',
}

export function getVisibleClassLabel(predictedClass: string): string {
  return (
    CLASS_LABELS[predictedClass] ?? 'No se ha podido interpretar el resultado'
  )
}
