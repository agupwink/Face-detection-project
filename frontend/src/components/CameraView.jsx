import { useEffect, useRef, useState, useCallback } from 'react'

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

const FACE_GOLD = '#d4a017'

export default function CameraView({ sessionId, onEnd }) {
  const videoRef = useRef(null)
  const overlayRef = useRef(null)
  const wsRef = useRef(null)
  const captureRef = useRef(null)
  const streamRef = useRef(null)
  const offscreenRef = useRef(null)
  const waitingRef = useRef(false)
  const lastResponseRef = useRef(Date.now())
  const clearTimerRef = useRef(null)

  const [status, setStatus] = useState('Initialising camera…')
  const [frameCount, setFrameCount] = useState(0)
  const [live, setLive] = useState({ faces: [], watches: [], fashion: [] })
  const [ending, setEnding] = useState(false)
  const [error, setError] = useState(null)

  const drawDetections = useCallback((data) => {
    const canvas = overlayRef.current
    const video = videoRef.current
    if (!canvas || !video || !video.videoWidth) return

    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const sx = canvas.width / video.videoWidth
    const sy = canvas.height / video.videoHeight

    const drawBox = (x1, y1, x2, y2, color, label) => {
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy)

      ctx.font = 'bold 13px sans-serif'
      const tw = ctx.measureText(label).width + 12
      const bh = 22
      const by = y1 * sy > bh ? y1 * sy - bh : y2 * sy
      ctx.fillStyle = color
      ctx.fillRect(x1 * sx, by, tw, bh)
      ctx.fillStyle = '#000'
      ctx.fillText(label, x1 * sx + 6, by + 15)
    }

    data.faces.forEach(face => {
      const [x1, y1, x2, y2] = face.box
      const conf = (face.confidence * 100).toFixed(0)
      const label = face.age
        ? `Face ${conf}%  ·  Age ${face.age}`
        : `Face ${conf}%`
      drawBox(x1, y1, x2, y2, FACE_GOLD, label)

      face.accessories.forEach(acc => {
        const [ax1, ay1, ax2, ay2] = acc.box
        const color = LABEL_COLORS[acc.label] || '#aaa'
        drawBox(ax1, ay1, ax2, ay2, color, `${acc.label} ${(acc.confidence * 100).toFixed(0)}%`)
      })
    })

    ;[...data.watches, ...data.fashion].forEach(item => {
      const [x1, y1, x2, y2] = item.box
      const color = LABEL_COLORS[item.label] || '#aaa'
      drawBox(x1, y1, x2, y2, color, `${item.label} ${(item.confidence * 100).toFixed(0)}%`)
    })
  }, [])

  useEffect(() => {
    let alive = true

    const init = async () => {
      let stream
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 } },
        })
      } catch (e) {
        setError(`Camera access denied: ${e.message}`)
        return
      }

      if (!alive) { stream.getTracks().forEach(t => t.stop()); return }

      streamRef.current = stream
      const video = videoRef.current
      video.srcObject = stream
      await new Promise(resolve => { video.onloadedmetadata = resolve })

      const canvas = overlayRef.current
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight

      // Offscreen canvas for frame capture
      const off = document.createElement('canvas')
      off.width = 640
      off.height = 480
      offscreenRef.current = off

      // WebSocket
      const wsProto = window.location.protocol === 'https:' ? 'wss' : 'ws'
      const ws = new WebSocket(`${wsProto}://${window.location.host}/ws/${sessionId}`)
      wsRef.current = ws

      ws.onopen = () => alive && setStatus('Detecting…')
      ws.onclose = () => alive && !ending && setStatus('Disconnected')
      ws.onerror = () => alive && setStatus('Connection error')
      ws.onmessage = (ev) => {
        if (!alive) return
        waitingRef.current = false
        lastResponseRef.current = Date.now()
        const msg = JSON.parse(ev.data)
        if (msg.type === 'detection') {
          setFrameCount(msg.frame_count)
          setLive(msg)
          drawDetections(msg)
        }
      }

      // Dedicated clearing timer — if no server response in 1s, wipe canvas hard
      clearTimerRef.current = setInterval(() => {
        if (Date.now() - lastResponseRef.current > 1000) {
          const ctx = overlayRef.current?.getContext('2d')
          if (ctx) ctx.clearRect(0, 0, overlayRef.current.width, overlayRef.current.height)
          setLive({ faces: [], watches: [], fashion: [] })
          waitingRef.current = false
        }
      }, 200)

      // Send next frame only after previous response arrives (no queuing)
      const octx = off.getContext('2d')
      captureRef.current = setInterval(() => {
        if (!video || ws.readyState !== WebSocket.OPEN || waitingRef.current) return
        waitingRef.current = true
        octx.drawImage(video, 0, 0, 640, 480)
        const b64 = off.toDataURL('image/jpeg', 0.7).split(',')[1]
        ws.send(JSON.stringify({ type: 'frame', data: b64 }))
      }, 150)

      setStatus('Detecting…')
    }

    init().catch(e => setError(e.message))

    return () => {
      alive = false
      clearInterval(captureRef.current)
      clearInterval(clearTimerRef.current)
      wsRef.current?.close()
      streamRef.current?.getTracks().forEach(t => t.stop())
    }
  }, [sessionId, drawDetections])

  const handleEnd = async () => {
    setEnding(true)
    clearInterval(captureRef.current)
    wsRef.current?.close()
    streamRef.current?.getTracks().forEach(t => t.stop())
    try {
      const res = await fetch(`/api/session/${sessionId}/end`, { method: 'POST' })
      const data = await res.json()
      onEnd(data)
    } catch {
      onEnd({ session_id: sessionId, total_detections: 0 })
    }
  }

  // Aggregate live labels for the bottom bar
  const liveLabels = [
    ...live.faces.flatMap(f => f.accessories.map(a => a.label)),
    ...live.watches.map(w => w.label),
    ...live.fashion.map(f => f.label),
  ]
  const uniqueLabels = [...new Set(liveLabels)]
  const faceInfo = live.faces.length > 0
    ? `${live.faces.length} face${live.faces.length > 1 ? 's' : ''}${live.faces[0]?.age ? `  ·  Age ${live.faces[0].age}` : ''}`
    : null

  if (error) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', flexDirection: 'column', gap: '16px' }}>
        <p style={{ color: '#ff5555' }}>{error}</p>
        <button onClick={() => window.location.reload()} style={btnStyle('#d4a017', '#000')}>Retry</button>
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: '#08080f' }}>
      {/* Top bar */}
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        padding: '10px 20px', background: '#0e0e1a', borderBottom: '1px solid #1a1a2e',
        flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{ color: FACE_GOLD, fontWeight: 700, fontSize: '0.95rem' }}>Live Detection</span>
          <span style={{
            width: '8px', height: '8px', borderRadius: '50%',
            background: status === 'Detecting…' ? '#32c832' : '#666',
            display: 'inline-block',
          }} />
          <span style={{ color: '#44445a', fontSize: '0.82rem' }}>{status}</span>
          <span style={{
            fontFamily: 'monospace', fontSize: '0.75rem',
            color: '#33334a', background: '#12121e',
            padding: '2px 8px', borderRadius: '4px',
            border: '1px solid #1a1a2e', userSelect: 'all',
          }}>
            {sessionId}
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <span style={{ color: '#33334a', fontSize: '0.8rem' }}>frames: {frameCount}</span>
          <button
            onClick={handleEnd}
            disabled={ending}
            style={btnStyle(ending ? '#333' : '#c03030', '#fff')}
          >
            {ending ? 'Ending…' : 'End Session'}
          </button>
        </div>
      </div>

      {/* Video + overlay */}
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden', position: 'relative' }}>
        <div style={{ position: 'relative', lineHeight: 0 }}>
          <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            style={{ display: 'block', maxWidth: '100%', maxHeight: 'calc(100vh - 100px)' }}
          />
          <canvas
            ref={overlayRef}
            style={{
              position: 'absolute', top: 0, left: 0,
              width: '100%', height: '100%',
              pointerEvents: 'none',
            }}
          />
        </div>
      </div>

      {/* Bottom detection badges */}
      <div style={{
        minHeight: '42px', padding: '8px 16px',
        background: '#0e0e1a', borderTop: '1px solid #1a1a2e',
        display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center',
        flexShrink: 0,
      }}>
        {faceInfo && (
          <Tag label={faceInfo} color={FACE_GOLD} />
        )}
        {uniqueLabels.map(label => (
          <Tag key={label} label={label} color={LABEL_COLORS[label] || '#888'} />
        ))}
        {!faceInfo && uniqueLabels.length === 0 && (
          <span style={{ color: '#33334a', fontSize: '0.8rem' }}>No detections yet…</span>
        )}
      </div>
    </div>
  )
}

function Tag({ label, color }) {
  return (
    <span style={{
      padding: '3px 10px',
      borderRadius: '12px',
      fontSize: '0.78rem',
      fontWeight: 600,
      background: `${color}22`,
      color,
      border: `1px solid ${color}55`,
    }}>
      {label}
    </span>
  )
}

function btnStyle(bg, fg) {
  return {
    padding: '7px 18px',
    background: bg,
    color: fg,
    border: 'none',
    borderRadius: '7px',
    cursor: 'pointer',
    fontWeight: 600,
    fontSize: '0.88rem',
  }
}
