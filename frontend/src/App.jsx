import { useState } from 'react'
import Welcome from './components/Welcome'
import CameraView from './components/CameraView'
import SessionSummary from './components/SessionSummary'

export default function App() {
  const [screen, setScreen] = useState('welcome')
  const [sessionId, setSessionId] = useState(null)
  const [summaryData, setSummaryData] = useState(null)

  const handleStart = async () => {
    const res = await fetch('/api/session/start', { method: 'POST' })
    const { session_id } = await res.json()
    setSessionId(session_id)
    setScreen('camera')
  }

  const handleEnd = (data) => {
    setSummaryData(data)
    setScreen('summary')
  }

  const handleNewSession = () => {
    setSessionId(null)
    setSummaryData(null)
    setScreen('welcome')
  }

  return (
    <>
      {screen === 'welcome' && <Welcome onStart={handleStart} />}
      {screen === 'camera' && <CameraView sessionId={sessionId} onEnd={handleEnd} />}
      {screen === 'summary' && <SessionSummary data={summaryData} onNewSession={handleNewSession} />}
    </>
  )
}
