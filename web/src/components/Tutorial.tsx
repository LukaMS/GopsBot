import { useState } from 'react'
import { PlayingCard } from './PlayingCard'

interface TutorialProps {
  readonly onClose: () => void
}

const steps = [
  {
    eyebrow: 'Step 1 of 3',
    title: 'A prize appears',
    body: 'Each round reveals one diamond. Its rank is its point value. A tied prize joins the next one, so the stakes can grow quickly.',
  },
  {
    eyebrow: 'Step 2 of 3',
    title: 'Choose your bid',
    body: 'You and the bot each have Ace through King. Every card can be used once, and both bids stay hidden until they are locked in.',
  },
  {
    eyebrow: 'Step 3 of 3',
    title: 'Read the rival',
    body: 'The higher bid takes the entire pot. Track which cards are gone, protect your high cards, and finish with more points after 13 rounds.',
  },
]

export function Tutorial({ onClose }: TutorialProps) {
  const [step, setStep] = useState(0)
  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className="tutorial-modal" role="dialog" aria-modal="true" aria-labelledby="tutorial-title">
        <button className="icon-button modal-close" type="button" onClick={onClose} aria-label="Close tutorial">×</button>
        <div className="tutorial-visual" aria-hidden="true">
          {step === 0 && <><PlayingCard rank={8} suit="diamond" /><span className="pot-chip">8 pts</span></>}
          {step === 1 && <><PlayingCard rank={5} suit="spade" /><div className="card-back"><span>G</span></div></>}
          {step === 2 && <div className="history-demo"><span>Played</span><strong>A · 7 · Q</strong><span>Still holding K</span></div>}
        </div>
        <p className="eyebrow">{steps[step].eyebrow}</p>
        <h2 id="tutorial-title">{steps[step].title}</h2>
        <p>{steps[step].body}</p>
        <div className="tutorial-footer">
          <div className="step-dots" aria-label={`Tutorial step ${step + 1} of 3`}>
            {steps.map((_, index) => <span key={index} className={index === step ? 'active' : ''} />)}
          </div>
          {step < steps.length - 1 ? (
            <button className="primary-button" type="button" onClick={() => setStep(step + 1)}>Next lesson <span>→</span></button>
          ) : (
            <button className="primary-button" type="button" onClick={onClose}>Take a seat <span>→</span></button>
          )}
        </div>
      </section>
    </div>
  )
}

