import { useState } from 'react'

export default function Welcome({ onStart }) {
  const [loading, setLoading] = useState(false)

  const handleClick = async () => {
    setLoading(true)
    try {
      await onStart()
    } catch {
      setLoading(false)
    }
  }

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      minHeight: '100vh',
      padding: '32px',
      textAlign: 'center',
      background: 'radial-gradient(ellipse at 50% 40%, #12121e 0%, #08080f 70%)',
    }}>
      {/* Greeting */}
      <div style={{ marginBottom: '12px' }}>
        <h1 style={{
          fontSize: 'clamp(3rem, 10vw, 6rem)',
          fontWeight: 800,
          color: '#d4a017',
          letterSpacing: '-0.02em',
          lineHeight: 1,
        }}>
          Hello!
        </h1>
      </div>

      <h2 style={{
        fontSize: '1.25rem',
        fontWeight: 400,
        color: '#6666aa',
        marginBottom: '20px',
        letterSpacing: '0.1em',
        textTransform: 'uppercase',
        fontSize: '0.9rem',
      }}>
        Face Detection System
      </h2>

      <p style={{
        maxWidth: '420px',
        color: '#55556a',
        lineHeight: 1.7,
        fontSize: '0.95rem',
        marginBottom: '40px',
      }}>
        AI-powered detection for faces, age, glasses, accessories, and fashion items.
        Start a session to see it in action.
      </p>

      {/* Feature badges */}
      <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', justifyContent: 'center', marginBottom: '40px' }}>
        {['Face Detection', 'Age Estimation', 'Glasses', 'Accessories', 'Fashion', 'Watch'].map(feat => (
          <span key={feat} style={{
            padding: '4px 12px',
            borderRadius: '20px',
            fontSize: '0.78rem',
            background: '#12121e',
            color: '#8888aa',
            border: '1px solid #22223a',
          }}>
            {feat}
          </span>
        ))}
      </div>

      <button
        onClick={handleClick}
        disabled={loading}
        style={{
          padding: '16px 44px',
          fontSize: '1rem',
          fontWeight: 700,
          background: loading ? '#444' : '#d4a017',
          color: '#080808',
          border: 'none',
          borderRadius: '10px',
          cursor: loading ? 'not-allowed' : 'pointer',
          letterSpacing: '0.03em',
          transition: 'transform 0.15s, box-shadow 0.15s',
          boxShadow: loading ? 'none' : '0 0 32px #d4a01755',
        }}
        onMouseEnter={e => { if (!loading) e.target.style.transform = 'scale(1.04)' }}
        onMouseLeave={e => { e.target.style.transform = 'scale(1)' }}
      >
        {loading ? 'Starting…' : 'Open Camera'}
      </button>
    </div>
  )
}
