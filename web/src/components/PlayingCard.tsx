import { rankLabel, type Rank } from '../game/engine'

interface PlayingCardProps {
  readonly rank: Rank
  readonly suit: 'diamond' | 'spade'
  readonly selected?: boolean
  readonly disabled?: boolean
  readonly onClick?: () => void
  readonly compact?: boolean
}

export function PlayingCard({ rank, suit, selected = false, disabled = false, onClick, compact = false }: PlayingCardProps) {
  const symbol = suit === 'diamond' ? '♦' : '♠'
  const label = `${rankLabel(rank)} of ${suit === 'diamond' ? 'diamonds' : 'spades'}`
  const content = (
    <>
      <span className="card-corner">{rankLabel(rank)}</span>
      <span className="card-suit" aria-hidden="true">{symbol}</span>
      <span className="sr-only">{label}</span>
    </>
  )
  if (onClick) {
    return (
      <button
        type="button"
        className={`playing-card ${suit} ${selected ? 'selected' : ''} ${compact ? 'compact' : ''}`}
        aria-label={`Bid ${label}${selected ? ', selected' : ''}`}
        aria-pressed={selected}
        disabled={disabled}
        onClick={onClick}
      >
        {content}
      </button>
    )
  }
  return <div className={`playing-card ${suit} ${compact ? 'compact' : ''}`}>{content}</div>
}

