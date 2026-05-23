import { useState } from 'react'

const LABEL_COLORS = {
  Glasses:   '#32c832',
  'Face Mask': '#ff9632',
  'Hat/Cap': '#c832c8',
  Helmet:    '#dc3232',
  Watch:     '#00e6c8',
  Hat:       '#ff7800',
  Headband:  '#b43cff',
  Tie:       '#64b450',
  Scarf:     '#00c8b4',
  Glove:     '#1e6496',
  Belt:      '#ffa500',
  Bag:       '#8232c8',
  Umbrella:  '#c8c800',
}

export default function SessionSummary({ data, onNewSession }) {
  if (!data) return null

  const {
    session_id,
    total_detections,
    avg_age,
    age_range,
    accessories = {},
    fashion_items = {},
    captured_faces = [],
  } = data

  const [feedbackAge, setFeedbackAge] = useState(avg_age ? String(avg_age) : '')
  const [feedbackStatus, setFeedbackStatus] = useState('idle') // idle | submitting | done | skipped

  const submitFeedback = async () => {
    const age = parseInt(feedbackAge, 10)
    if (!age || age < 1 || age > 120) return
    setFeedbackStatus('submitting')
    try {
      await fetch(`/api/session/${session_id}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ real_age: age }),
      })
    } catch {
      // silently fail — still mark done
    }
    setFeedbackStatus('done')
  }

  const allItems = { ...accessories, ...fashion_items }
  const hasItems = Object.keys(allItems).length > 0

  return (
    <div style={{ minHeight: '100vh', background: '#08080f', padding: '48px 20px' }}>
      <div style={{ maxWidth: '780px', margin: '0 auto' }}>

        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: '48px' }}>
          <h1 style={{ fontSize: '2.5rem', fontWeight: 800, color: '#d4a017', marginBottom: '8px' }}>
            Session Complete
          </h1>
          <p style={{ color: '#44445a', fontSize: '0.9rem' }}>
            {total_detections > 0
              ? `${total_detections} detection${total_detections > 1 ? 's' : ''} recorded`
              : 'No detections were recorded'}
          </p>
        </div>

        {total_detections > 0 ? (
          <>
            {/* Age feedback card */}
            {feedbackStatus !== 'skipped' && (
              <div style={{
                background: '#0e0e1a',
                border: '1px solid #2a2a1e',
                borderRadius: '12px',
                padding: '22px 24px',
                marginBottom: '20px',
              }}>
                {feedbackStatus === 'done' ? (
                  <div style={{ textAlign: 'center', padding: '8px 0' }}>
                    <span style={{ color: '#32c832', fontSize: '1.1rem', fontWeight: 700 }}>Thanks!</span>
                    <span style={{ color: '#44445a', fontSize: '0.9rem', marginLeft: '10px' }}>
                      Age model will improve over time.
                    </span>
                  </div>
                ) : (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
                    <div style={{ flex: 1, minWidth: '160px' }}>
                      <div style={{ color: '#d4a017', fontWeight: 700, fontSize: '0.88rem', marginBottom: '4px' }}>
                        Help improve age detection
                      </div>
                      <div style={{ color: '#44445a', fontSize: '0.8rem' }}>
                        {avg_age ? `We predicted age ${avg_age}. ` : ''}What is your actual age?
                      </div>
                    </div>
                    <input
                      type="number"
                      min="1"
                      max="120"
                      value={feedbackAge}
                      onChange={e => setFeedbackAge(e.target.value)}
                      onKeyDown={e => e.key === 'Enter' && submitFeedback()}
                      placeholder="Your age"
                      style={{
                        width: '90px',
                        padding: '9px 12px',
                        background: '#12121e',
                        border: '1px solid #2a2a2e',
                        borderRadius: '8px',
                        color: '#c0c0d8',
                        fontSize: '0.95rem',
                        outline: 'none',
                        textAlign: 'center',
                      }}
                    />
                    <button
                      onClick={submitFeedback}
                      disabled={feedbackStatus === 'submitting' || !feedbackAge}
                      style={{
                        padding: '9px 20px',
                        background: feedbackStatus === 'submitting' || !feedbackAge ? '#1a1a2e' : '#d4a017',
                        color: feedbackStatus === 'submitting' || !feedbackAge ? '#44445a' : '#080808',
                        border: 'none',
                        borderRadius: '8px',
                        fontWeight: 700,
                        fontSize: '0.88rem',
                        cursor: feedbackStatus === 'submitting' || !feedbackAge ? 'default' : 'pointer',
                      }}
                    >
                      {feedbackStatus === 'submitting' ? 'Saving…' : 'Submit'}
                    </button>
                    <button
                      onClick={() => setFeedbackStatus('skipped')}
                      style={{
                        background: 'none',
                        border: 'none',
                        color: '#33334a',
                        fontSize: '0.8rem',
                        cursor: 'pointer',
                        padding: '4px',
                      }}
                    >
                      Skip
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* Stat cards */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '14px', marginBottom: '28px' }}>
              <StatCard label="Detections" value={total_detections} />
              <StatCard label="Average Age" value={avg_age ? `${avg_age} yrs` : '—'} />
              <StatCard
                label="Age Range"
                value={age_range ? `${age_range.min}–${age_range.max}` : '—'}
              />
            </div>

            {/* Detected items */}
            {hasItems && (
              <Section title="Detected Items">
                <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                  {Object.entries(allItems).sort((a, b) => b[1] - a[1]).map(([item, count]) => {
                    const color = LABEL_COLORS[item] || '#888'
                    return (
                      <div key={item} style={{
                        padding: '8px 16px',
                        borderRadius: '8px',
                        background: `${color}15`,
                        border: `1px solid ${color}44`,
                        display: 'flex',
                        gap: '8px',
                        alignItems: 'center',
                      }}>
                        <span style={{ color: '#d4a017', fontWeight: 700 }}>{count}×</span>
                        <span style={{ color: '#c0c0d8', fontWeight: 500 }}>{item}</span>
                      </div>
                    )
                  })}
                </div>
              </Section>
            )}

            {/* Face thumbnails */}
            {captured_faces.length > 0 && (
              <Section title={`Captured Faces (${captured_faces.length})`}>
                <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
                  {captured_faces.map((face, i) => (
                    <FaceThumbnail key={i} face={face} index={i} />
                  ))}
                </div>
              </Section>
            )}
          </>
        ) : (
          <div style={{ textAlign: 'center', color: '#33334a', padding: '60px 0' }}>
            <p style={{ fontSize: '1.1rem' }}>No faces were detected during this session.</p>
            <p style={{ marginTop: '8px', fontSize: '0.9rem' }}>Try moving closer to the camera.</p>
          </div>
        )}

        {/* Session ID */}
        <div style={{ textAlign: 'center', marginTop: '32px' }}>
          <span style={{ color: '#44445a', fontSize: '0.78rem' }}>Session ID: </span>
          <span style={{
            fontFamily: 'monospace', fontSize: '0.78rem',
            color: '#6666aa', background: '#0e0e1a',
            padding: '3px 10px', borderRadius: '4px',
            border: '1px solid #1a1a2e', userSelect: 'all',
          }}>
            {session_id}
          </span>
        </div>

        {/* CTA */}
        <div style={{ textAlign: 'center', marginTop: '40px' }}>
          <button
            onClick={onNewSession}
            style={{
              padding: '15px 40px',
              background: '#d4a017',
              color: '#080808',
              border: 'none',
              borderRadius: '10px',
              fontSize: '1rem',
              fontWeight: 700,
              cursor: 'pointer',
              boxShadow: '0 0 28px #d4a01744',
            }}
          >
            Start New Session
          </button>
        </div>
      </div>
    </div>
  )
}

function StatCard({ label, value }) {
  return (
    <div style={{
      background: '#0e0e1a',
      border: '1px solid #1a1a2e',
      borderRadius: '12px',
      padding: '22px 20px',
      textAlign: 'center',
    }}>
      <div style={{ fontSize: '2rem', fontWeight: 800, color: '#d4a017', lineHeight: 1 }}>
        {value}
      </div>
      <div style={{ color: '#44445a', fontSize: '0.8rem', marginTop: '6px', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
        {label}
      </div>
    </div>
  )
}

function Section({ title, children }) {
  return (
    <div style={{
      background: '#0e0e1a',
      border: '1px solid #1a1a2e',
      borderRadius: '12px',
      padding: '22px',
      marginBottom: '16px',
    }}>
      <h3 style={{
        color: '#44445a',
        fontSize: '0.75rem',
        textTransform: 'uppercase',
        letterSpacing: '0.08em',
        marginBottom: '16px',
        fontWeight: 600,
      }}>
        {title}
      </h3>
      {children}
    </div>
  )
}

function FaceThumbnail({ face, index }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{
        width: '80px',
        height: '80px',
        borderRadius: '10px',
        overflow: 'hidden',
        border: '2px solid #1a1a2e',
        background: '#12121e',
      }}>
        {face.frame_path ? (
          <img
            src={`/api/frames/${face.frame_path}`}
            alt={`Face ${index + 1}`}
            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
            onError={e => { e.target.style.display = 'none' }}
          />
        ) : (
          <div style={{
            width: '100%', height: '100%',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            color: '#33334a', fontSize: '1.5rem',
          }}>
            ?
          </div>
        )}
      </div>
      {face.age && (
        <div style={{ color: '#d4a017', fontSize: '0.78rem', marginTop: '5px', fontWeight: 600 }}>
          Age {face.age}
        </div>
      )}
    </div>
  )
}
