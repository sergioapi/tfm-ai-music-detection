import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { SiteHeader } from './SiteHeader'

describe('SiteHeader', () => {
  it('shows the application brand with the decorative SVG logo', () => {
    const { container } = render(<SiteHeader />)

    expect(screen.getByText('VeriSon')).toBeInTheDocument()
    expect(container.querySelector('img')).toHaveAttribute('src', '/logo.svg')
    expect(container.querySelector('img')).toHaveAttribute('alt', '')
  })
})
