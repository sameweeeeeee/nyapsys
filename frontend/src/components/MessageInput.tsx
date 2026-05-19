import { useState, useRef, useCallback, ChangeEvent } from 'react'
import { FilePreview } from './FilePreview'

interface MessageInputProps {
  onSend: (message: string) => void
  onAttach: (file: File) => void
  onRemoveFile: () => void
  attachedFile: File | null
  disabled?: boolean
}

export function MessageInput({ onSend, onAttach, onRemoveFile, attachedFile, disabled }: MessageInputProps) {
  const [input, setInput] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() && !attachedFile) return
    onSend(input)
    setInput('')
  }, [input, attachedFile, onSend])

  const handleFileSelect = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) { onAttach(file); if (fileInputRef.current) fileInputRef.current.value = '' }
  }, [onAttach])

  return (
    <form className="message-input-form" onSubmit={handleSubmit}>
      {attachedFile && <FilePreview file={attachedFile} onRemove={onRemoveFile} />}
      <div className="input-container">
        <input type="file" ref={fileInputRef} onChange={handleFileSelect} accept="image/*,.pdf,.docx,.txt,.md,.csv,.json" style={{ display: 'none' }} />
        <button type="button" className="attach-btn" onClick={() => fileInputRef.current?.click()} disabled={disabled}>📎</button>
        <input type="text" value={input} onChange={(e) => setInput(e.target.value)} placeholder="Type a message..." disabled={disabled} className="message-input" />
        <button type="submit" className="send-btn" disabled={disabled || (!input.trim() && !attachedFile)}>➤</button>
      </div>
    </form>
  )
}