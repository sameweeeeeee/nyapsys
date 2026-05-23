'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { fetchTrainingLogs, TrainingLogs } from '../lib/api'

interface Props {
  onBack: () => void
}

export function SettingsView({ onBack }: Props) {
  const [logs, setLogs] = useState<TrainingLogs | null>(null)
  const [autoScroll, setAutoScroll] = useState(true)
  const [polling, setPolling] = useState(true)
  const preRef = useRef<HTMLPreElement>(null)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const fetchLogs = useCallback(async () => {
    const result = await fetchTrainingLogs(500)
    setLogs(result)
  }, [])

  useEffect(() => {
    fetchLogs()
    if (polling) {
      intervalRef.current = setInterval(fetchLogs, 5000)
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [fetchLogs, polling])

  useEffect(() => {
    if (autoScroll && preRef.current) {
      preRef.current.scrollTop = preRef.current.scrollHeight
    }
  }, [logs, autoScroll])

  useEffect(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      if (polling) {
        intervalRef.current = setInterval(fetchLogs, 5000)
      }
    }
  }, [polling, fetchLogs])

  return (
    <div className="settings-page">
      <header className="topbar">
        <button className="topbar-hamburger" onClick={onBack}>←</button>
        <h1 className="settings-title">Settings</h1>
      </header>

      <main className="settings-content">
        <section className="settings-section">
          <div className="settings-section-header">
            <h2 className="settings-section-title">Training Logs</h2>
            <div className="settings-controls">
              <label className="settings-toggle">
                <input type="checkbox" checked={polling} onChange={e => setPolling(e.target.checked)} />
                <span>Auto-refresh (5s)</span>
              </label>
              <label className="settings-toggle">
                <input type="checkbox" checked={autoScroll} onChange={e => setAutoScroll(e.target.checked)} />
                <span>Auto-scroll</span>
              </label>
              <button className="settings-refresh-btn" onClick={fetchLogs}>Refresh</button>
            </div>
          </div>
          {logs && (
            <div className="settings-status">
              <span className={`status-indicator ${logs.connected ? 'connected' : 'disconnected'}`} />
              {logs.connected ? `Connected — ${logs.line_count} lines` : 'Disconnected'}
              {logs.error && <span className="settings-error"> — {logs.error}</span>}
            </div>
          )}
        </section>

        <section className="settings-section">
          <pre ref={preRef} className="log-viewer">
            {logs?.log ? logs.log : 'Waiting for log data...'}
          </pre>
        </section>
      </main>
    </div>
  )
}