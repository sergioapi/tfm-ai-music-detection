import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { SiteFooter } from './SiteFooter'

describe('SiteFooter', () => {
  it('shows the application footer summary', () => {
    render(<SiteFooter />)

    expect(screen.getByText('© 2026 VeriSon')).toBeInTheDocument()
    expect(screen.getByText('WAV · MP3 · Hasta 5 min')).toBeInTheDocument()
  })
})
