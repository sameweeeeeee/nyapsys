import { useRef, useCallback, ChangeEvent } from 'react'
import { FilePreview } from './FilePreview'

interface MessageInputProps {
  onSend: (message: string) => void
  onAttach: (file: File) => void
  onRemoveFile: () => void
  attachedFile: File | null
  disabled?: boolean
  value: string
  onChange: (value: string) => void
}

export function MessageInput({ onSend, onAttach, onRemoveFile, attachedFile, disabled, value, onChange }: MessageInputProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault()
    if (!value.trim() && !attachedFile) return
    onSend(value)
    onChange('')
  }, [value, attachedFile, onSend, onChange])

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }, [handleSubmit])

  const handleFileSelect = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) { onAttach(file); if (fileInputRef.current) fileInputRef.current.value = '' }
  }, [onAttach])

  const handleInput = useCallback(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 160) + 'px'
    }
  }, [])

  return (
    <form className="message-input-form" onSubmit={handleSubmit}>
      {attachedFile && <FilePreview file={attachedFile} onRemove={onRemoveFile} />}
      <div className="input-container">
        <input type="file" ref={fileInputRef} onChange={handleFileSelect} accept="image/*,.pdf,.docx,.txt,.md,.csv,.json" style={{ display: 'none' }} />
        <button type="button" className="attach-btn" onClick={() => fileInputRef.current?.click()} disabled={disabled}>📎</button>
        <textarea ref={textareaRef} value={value} onChange={(e) => onChange(e.target.value)} onKeyDown={handleKeyDown} onInput={handleInput} placeholder="Message Nyapsys…" disabled={disabled} className="message-input" rows={1} />
        <button type="submit" className="send-btn" disabled={disabled || (!value.trim() && !attachedFile)}>→</button>
      </div>
      <p className="input-hint">Nyapsys runs entirely on your Mac — your data never leaves.</p>
    </form>
  )
}
