import { useEffect, useMemo, useRef, useState } from 'react'
import { BotClient } from './bot/client'
import type { ModelEntry, ModelManifest } from './bot/types'
import { PlayingCard } from './components/PlayingCard'
import { Tutorial } from './components/Tutorial'
import { applyBidPair, createSeed, rankLabel, startMatch, type Difficulty, type MatchState, type Rank, type RoundResult } from './game/engine'
import { encodeBotObservation, OBSERVATION_SIZE, OBSERVATION_VERSION } from './game/observation'
import { clearPlayerData, loadPlayerData, markTutorialSeen, recordMatch, savePreference, type PlayerData } from './game/storage'

type Screen = 'home' | 'game' | 'results'
type LoadState = 'idle' | 'loading' | 'ready' | 'error'

const DIFFICULTIES: ReadonlyArray<{
  id: Difficulty
  number: string
  label: string
  detail: string
  accent: string
}> = [
  { id: 'beginner', number: '01', label: 'Beginner', detail: 'Still learning when to hold back.', accent: 'Open table' },
  { id: 'casual', number: '02', label: 'Casual', detail: 'A thoughtful opponent for relaxed games.', accent: 'House favourite' },
  { id: 'skilled', number: '03', label: 'Skilled', detail: 'Tracks the deck and protects leverage.', accent: 'Sharp play' },
  { id: 'expert', number: '04', label: 'Expert', detail: 'The strongest policy from the self-play league.', accent: 'No mercy' },
]

const delay = (milliseconds: number) => new Promise((resolve) => setTimeout(resolve, milliseconds))

function difficultyLabel(difficulty: Difficulty): string {
  return DIFFICULTIES.find((item) => item.id === difficulty)?.label ?? difficulty
}

function AppMark() {
  return <span className="app-mark" aria-hidden="true"><span>G</span></span>
}

export function App() {
  const initialData = useMemo(() => loadPlayerData(), [])
  const [screen, setScreen] = useState<Screen>('home')
  const [difficulty, setDifficulty] = useState<Difficulty>(initialData.preferredDifficulty)
  const [playerData, setPlayerData] = useState<PlayerData>(initialData)
  const [showTutorial, setShowTutorial] = useState(!initialData.tutorialSeen)
  const [match, setMatch] = useState<MatchState | null>(null)
  const [selectedBid, setSelectedBid] = useState<Rank | null>(null)
  const [modelEntry, setModelEntry] = useState<ModelEntry | null>(null)
  const [loadState, setLoadState] = useState<LoadState>('idle')
  const [error, setError] = useState<string | null>(null)
  const [thinking, setThinking] = useState(false)
  const [roundReveal, setRoundReveal] = useState<RoundResult | null>(null)
  const [lastInferenceMs, setLastInferenceMs] = useState<number | null>(null)
  const botRef = useRef<BotClient | null>(null)
  const revealing = roundReveal !== null

  useEffect(() => () => botRef.current?.terminate(), [])

  const totals = useMemo(() => {
    const wins = playerData.matches.filter((item) => item.result === 'human').length
    const losses = playerData.matches.filter((item) => item.result === 'bot').length
    const draws = playerData.matches.filter((item) => item.result === 'draw').length
    return { games: playerData.matches.length, wins, losses, draws }
  }, [playerData])

  function selectDifficulty(next: Difficulty) {
    setDifficulty(next)
    setPlayerData(savePreference(next))
    setError(null)
  }

  async function startGame() {
    setError(null)
    setLoadState('loading')
    botRef.current?.terminate()
    const bot = new BotClient()
    botRef.current = bot
    try {
      const manifestUrl = `${import.meta.env.BASE_URL}models/manifest.json`
      const response = await fetch(manifestUrl)
      if (!response.ok) throw new Error('The model catalogue could not be downloaded.')
      const manifest = (await response.json()) as ModelManifest
      if (manifest.schemaVersion !== 1 || manifest.observationSize !== OBSERVATION_SIZE || manifest.actionSize !== 13) {
        throw new Error('The model catalogue uses an unsupported contract version.')
      }
      const entry = manifest.models.find((item) => item.difficulty === difficulty)
      if (!entry) throw new Error(`No ${difficultyLabel(difficulty)} policy is listed in the model catalogue.`)
      if (entry.observationVersion !== OBSERVATION_VERSION) {
        throw new Error('The selected policy uses an unsupported observation version.')
      }
      await bot.initialize(`${import.meta.env.BASE_URL}models/${entry.file}`, entry.sha256)
      setModelEntry(entry)
      setMatch(startMatch(createSeed()))
      setSelectedBid(null)
      setRoundReveal(null)
      setLastInferenceMs(null)
      setLoadState('ready')
      setScreen('game')
    } catch (caught) {
      bot.terminate()
      if (botRef.current === bot) botRef.current = null
      setLoadState('error')
      setError(caught instanceof Error ? caught.message : 'The bot could not be loaded.')
    }
  }

  async function playSelectedBid() {
    if (!match || selectedBid === null || thinking || revealing || !botRef.current) return
    const stateBeforeBid = match
    const humanBid = selectedBid
    setThinking(true)
    setError(null)
    try {
      const [choice] = await Promise.all([
        botRef.current.choose(encodeBotObservation(stateBeforeBid), stateBeforeBid.remainingBotBids),
        delay(300),
      ])
      setLastInferenceMs(choice.latencyMs)
      const next = applyBidPair(stateBeforeBid, humanBid, choice.rank)
      const result = next.history.at(-1)
      if (!result) throw new Error('The round result could not be displayed.')
      setRoundReveal(result)
      setSelectedBid(null)
      setThinking(false)
      await delay(1900)
      setRoundReveal(null)
      setMatch(next)
      if (next.phase === 'complete') {
        const data = recordMatch(next, difficulty, modelEntry?.modelVersion ?? 'unknown')
        setPlayerData(data)
        setScreen('results')
      }
    } catch (caught) {
      setThinking(false)
      setRoundReveal(null)
      setError(caught instanceof Error ? caught.message : 'The bot could not choose a bid.')
    }
  }

  function closeTutorial() {
    setShowTutorial(false)
    setPlayerData(markTutorialSeen())
  }

  function returnHome() {
    setScreen('home')
    setMatch(null)
    setSelectedBid(null)
    setRoundReveal(null)
    setError(null)
    setLoadState('idle')
  }

  function resetStats() {
    if (!window.confirm('Reset all local match history and preferences? This cannot be undone.')) return
    const data = clearPlayerData()
    setPlayerData(data)
    setDifficulty(data.preferredDifficulty)
  }

  if (screen === 'game' && match) {
    const displayPrize = match.currentPrize ?? roundReveal?.prize ?? 1
    const displayedHumanBids = roundReveal ? match.remainingHumanBids.filter((rank) => rank !== roundReveal.humanBid) : match.remainingHumanBids
    const displayedBotCount = Math.max(0, match.remainingBotBids.length - (roundReveal ? 1 : 0))
    const displayedHumanScore = roundReveal?.humanScore ?? match.humanScore
    const displayedBotScore = roundReveal?.botScore ?? match.botScore
    const displayedCarriedPot = roundReveal?.carriedAfter ?? match.carriedPot
    return (
      <main className="game-shell" data-last-inference-ms={lastInferenceMs?.toFixed(3)}>
        <header className="game-topbar">
          <button className="brand-button" type="button" onClick={returnHome} aria-label="Return to home"><AppMark /><span>GOPS</span></button>
          <div className="round-marker"><span>Round</span><strong>{match.round.toString().padStart(2, '0')}</strong><span>/ 13</span></div>
          <div className="game-actions">
            <span className={`difficulty-pill ${difficulty}`}>{difficultyLabel(difficulty)}</span>
            <button className="icon-button" type="button" onClick={() => setShowTutorial(true)} aria-label="Open rules">?</button>
          </div>
        </header>

        <section className="score-ribbon" aria-label="Score">
          <div><span>You</span><strong>{displayedHumanScore}</strong></div>
          <p>{displayedCarriedPot > 0 ? `${displayedCarriedPot} points carried` : 'The table is clear'}</p>
          <div className="bot-score"><strong>{displayedBotScore}</strong><span>Bot</span></div>
        </section>

        <div className="game-layout">
          <aside className="history-panel" aria-label="Round history">
            <div className="panel-heading"><span>Ledger</span><strong>{match.history.length}/13</strong></div>
            {match.history.length === 0 ? (
              <div className="empty-history"><span>♢</span><p>Your plays will appear here.</p></div>
            ) : (
              <ol className="history-list">
                {[...match.history].reverse().map((round) => (
                  <li key={round.round}>
                    <span>{round.round.toString().padStart(2, '0')}</span>
                    <b className={round.winner}>{round.winner === 'tie' ? 'Carry' : round.winner === 'human' ? 'Won' : 'Lost'}</b>
                    <span>{rankLabel(round.humanBid)} : {rankLabel(round.botBid)}</span>
                    <strong>{round.winner === 'tie' ? `+${round.carriedAfter}` : `${round.awarded} pts`}</strong>
                  </li>
                ))}
              </ol>
            )}
          </aside>

          <section className={`table ${revealing ? 'is-revealing' : ''}`} aria-label="Game table">
            <div className="bot-zone">
              <div className="player-label"><span className="status-dot" />SELF-PLAY BOT</div>
              <div className="bot-hand" aria-label={`${displayedBotCount} cards in the bot hand`}>
                {Array.from({ length: Math.min(7, displayedBotCount) }, (_, index) => (
                  <div className="card-back mini" key={index} style={{ '--card-index': index } as React.CSSProperties}><span>G</span></div>
                ))}
                <span className="card-count">{displayedBotCount} cards</span>
              </div>
            </div>

            <div className={`prize-zone ${revealing ? 'revealing' : ''}`}>
              <div className="prize-deck" aria-hidden="true"><span>{match.remainingPrizes.length}</span></div>
              <div className="showdown-stage">
                {roundReveal && (
                  <div className="showdown-bid human-bid">
                    <span>Your bid</span>
                    <PlayingCard rank={roundReveal.humanBid} suit="spade" />
                  </div>
                )}
                <div className={`prize-card-wrap ${revealing ? 'contested' : ''}`}>
                  <p className="eyebrow">{match.carriedPot ? `Pot worth ${match.carriedPot + displayPrize}` : 'Current prize'}</p>
                  <PlayingCard rank={displayPrize as Rank} suit="diamond" />
                  {match.carriedPot > 0 && <span className="carry-badge">+{match.carriedPot} carried</span>}
                </div>
                {roundReveal && (
                  <div className="showdown-bid bot-bid">
                    <span>Bot's bid</span>
                    <div className="bot-card-flip">
                      <div className="bot-card-inner">
                        <div className="bot-card-face bot-card-back" aria-hidden="true"><div className="card-back"><span>G</span></div></div>
                        <div className="bot-card-face bot-card-front"><PlayingCard rank={roundReveal.botBid} suit="spade" /></div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
              <div className="bid-reveal" aria-live="polite">
                {thinking && <span className="thinking"><i /><i /><i /> Bot is considering the table</span>}
                {roundReveal && (
                  <span className={`round-outcome ${roundReveal.winner}`} role="status">
                    <strong>{roundReveal.winner === 'tie' ? 'Tie — pot carries' : roundReveal.winner === 'human' ? 'You win the round' : 'Bot wins the round'}</strong>
                    <span>{roundReveal.winner === 'tie' ? `${roundReveal.carriedAfter} points move forward` : `${roundReveal.awarded} points awarded`}</span>
                  </span>
                )}
                {!thinking && !revealing && <span>Choose one card. Both bids reveal together.</span>}
              </div>
            </div>

            <div className="human-zone">
              <div className="hand-heading">
                <div><span className="status-dot human" />YOUR HAND</div>
                <span>{selectedBid ? `${rankLabel(selectedBid)} selected` : 'Select a bid'}</span>
              </div>
              <div className="human-hand">
                {displayedHumanBids.map((rank) => (
                  <PlayingCard
                    key={rank}
                    rank={rank}
                    suit="spade"
                    selected={selectedBid === rank}
                    disabled={thinking || revealing}
                    onClick={() => setSelectedBid(rank)}
                    compact
                  />
                ))}
              </div>
              <button className="bid-button" type="button" disabled={selectedBid === null || thinking || revealing} onClick={playSelectedBid}>
                {thinking ? 'Bid locked…' : selectedBid ? `Play ${rankLabel(selectedBid)}` : 'Choose a card'}
                <span aria-hidden="true">↗</span>
              </button>
              {error && <div className="inline-error" role="alert">{error} <button type="button" onClick={playSelectedBid}>Try again</button></div>}
            </div>
          </section>
        </div>
        {showTutorial && <Tutorial onClose={closeTutorial} />}
      </main>
    )
  }

  if (screen === 'results' && match) {
    const won = match.winner === 'human'
    const draw = match.winner === 'draw'
    return (
      <main className="results-page">
        <div className="results-card">
          <AppMark />
          <p className="eyebrow">Match complete · {difficultyLabel(difficulty)}</p>
          <h1>{draw ? 'An even table.' : won ? 'Well played.' : 'The bot takes it.'}</h1>
          <p className="results-copy">{draw ? 'Every awarded point balanced out.' : won ? 'You read the table better across thirteen rounds.' : 'Review the ledger, adjust your timing, and deal again.'}</p>
          <div className="final-score">
            <div><span>You</span><strong>{match.humanScore}</strong></div>
            <span>—</span>
            <div><span>Bot</span><strong>{match.botScore}</strong></div>
          </div>
          {match.carriedPot > 0 && <p className="unclaimed-note">{match.carriedPot} points remained unclaimed after the final tie.</p>}
          <div className="results-actions">
            <button className="primary-button" type="button" onClick={startGame} disabled={loadState === 'loading'}>{loadState === 'loading' ? 'Loading rival…' : 'Play again'} <span>↻</span></button>
            <button className="secondary-button" type="button" onClick={returnHome}>Change difficulty</button>
          </div>
          {error && <div className="model-error" role="alert"><strong>Couldn’t load the rematch.</strong><span>{error}</span></div>}
        </div>
      </main>
    )
  }

  return (
    <main className="home-page">
      <nav className="home-nav">
        <a className="brand" href="#top" aria-label="GOPS home"><AppMark /><span>GOPS</span></a>
        <div>
          <button type="button" onClick={() => setShowTutorial(true)}>How to play</button>
          <a href="#statistics">Your record</a>
        </div>
      </nav>

      <section className="hero" id="top">
        <div className="hero-copy">
          <p className="eyebrow"><span /> Game of Pure Strategy</p>
          <h1>Thirteen cards.<br /><em>Nothing</em> left to chance.</h1>
          <p className="hero-lede">Read the prize. Read the opponent. Outbid a rival that learned the game by playing itself.</p>
          <button className="text-button" type="button" onClick={() => setShowTutorial(true)}>Learn the game <span>↗</span></button>
        </div>
        <div className="hero-table" aria-hidden="true">
          <div className="orbit orbit-one" />
          <div className="orbit orbit-two" />
          <div className="hero-card hero-card-left"><span>A</span><b>♠</b></div>
          <div className="hero-card hero-card-main"><span>9</span><b>♦</b><small>PRIZE</small></div>
          <div className="hero-card hero-card-right"><span>K</span><b>♠</b></div>
          <span className="hero-caption">Every bid changes<br />what comes next.</span>
        </div>
      </section>

      <section className="difficulty-section" aria-labelledby="difficulty-title">
        <div className="section-heading">
          <div><p className="eyebrow">Choose your rival</p><h2 id="difficulty-title">How sharp should the table be?</h2></div>
          <p>Each level is a distinct policy captured and tested during self-play training.</p>
        </div>
        <div className="difficulty-grid">
          {DIFFICULTIES.map((item) => (
            <button
              className={`difficulty-card ${difficulty === item.id ? 'selected' : ''}`}
              type="button"
              key={item.id}
              onClick={() => selectDifficulty(item.id)}
              aria-pressed={difficulty === item.id}
            >
              <span className="difficulty-number">{item.number}</span>
              <span className={`difficulty-signal ${item.id}`}><i /><i /><i /><i /></span>
              <strong>{item.label}</strong>
              <span className="difficulty-detail">{item.detail}</span>
              <span className="difficulty-accent">{item.accent}</span>
              <span className="selection-check">✓</span>
            </button>
          ))}
        </div>
        <div className="start-row">
          <div><span className="privacy-dot" /> Runs privately in your browser</div>
          <button className="start-button" type="button" onClick={startGame} disabled={loadState === 'loading'}>
            <span>{loadState === 'loading' ? 'Loading the table' : `Play ${difficultyLabel(difficulty)}`}</span><b>{loadState === 'loading' ? '···' : '→'}</b>
          </button>
        </div>
        {error && <div className="model-error" role="alert"><strong>The bot couldn’t take its seat.</strong><span>{error}</span><button type="button" onClick={startGame}>Retry</button></div>}
      </section>

      <section className="record-section" id="statistics">
        <div>
          <p className="eyebrow">Your record · This device</p>
          <h2>{totals.games === 0 ? 'The ledger is clean.' : `${totals.games} ${totals.games === 1 ? 'match' : 'matches'} at the table.`}</h2>
          <p>{totals.games === 0 ? 'Your results stay in this browser and appear here after your first game.' : 'No account, no cloud profile, and no tracking. These results belong to this browser.'}</p>
        </div>
        <dl className="record-grid">
          <div><dt>Wins</dt><dd>{totals.wins}</dd></div>
          <div><dt>Losses</dt><dd>{totals.losses}</dd></div>
          <div><dt>Draws</dt><dd>{totals.draws}</dd></div>
          <div><dt>Win rate</dt><dd>{totals.games ? `${Math.round((totals.wins / totals.games) * 100)}%` : '—'}</dd></div>
        </dl>
        {totals.games > 0 && <button className="reset-button" type="button" onClick={resetStats}>Reset local data</button>}
      </section>

      <footer>
        <span>Built for quiet rivalries.</span>
        <span>13 cards · 91 points · 0 hidden advantages</span>
      </footer>
      {showTutorial && <Tutorial onClose={closeTutorial} />}
    </main>
  )
}
