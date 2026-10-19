import { fireEvent, render, screen } from '@testing-library/react'
import { Tutorial } from '../components/Tutorial'

describe('tutorial', () => {
  it('walks through all three lessons', () => {
    const onClose = vi.fn()
    render(<Tutorial onClose={onClose} />)
    expect(screen.getByRole('heading', { name: 'A prize appears' })).toBeVisible()
    fireEvent.click(screen.getByRole('button', { name: /next lesson/i }))
    expect(screen.getByRole('heading', { name: 'Choose your bid' })).toBeVisible()
    fireEvent.click(screen.getByRole('button', { name: /next lesson/i }))
    expect(screen.getByRole('heading', { name: 'Read the rival' })).toBeVisible()
    fireEvent.click(screen.getByRole('button', { name: /take a seat/i }))
    expect(onClose).toHaveBeenCalledOnce()
  })
})

