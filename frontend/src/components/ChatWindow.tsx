'use client'

import { useChat } from '../hooks/useChat'
import { MessageBubble } from '../components/MessageBubble'
import { MessageInput } from '../components/MessageInput'

export function ChatWindow() {
  const { messages, conversations, conversationId, isLoading, error, attachedFile, sidebarOpen, setSidebarOpen, inputText, setInputText, sendMessage, selectConversation, startNewConversation, removeConversation, attachFile, removeFile, messagesEndRef } = useChat()
  const isEmpty = messages.length === 0

  const starterPills = [
    { label: 'Summarise a document', text: 'I have a document I need summarised. Can you help?' },
    { label: 'Explain something complex', text: 'Can you explain quantum computing in simple terms?' },
    { label: 'Analyse an image', text: 'I have an image I need you to analyse.' },
    { label: 'Write a draft', text: 'Help me write a professional email to my team.' },
  ]

  return (
    <>
      <div className={`sidebar-overlay ${sidebarOpen ? 'open' : ''}`} onClick={() => setSidebarOpen(false)} />
      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <img src="/logo.svg" alt="Nyapsys" className="sidebar-logo-img" />
          <button className="sidebar-close" onClick={() => setSidebarOpen(false)}>×</button>
        </div>
        <button className="sidebar-new-btn" onClick={startNewConversation}>New conversation</button>
        <div className="sidebar-section-label">Recent</div>
        <div className="sidebar-conversations">
          {conversations.map(conv => (
            <div key={conv.id} className={`sidebar-item ${conv.id === conversationId ? 'active' : ''}`} onClick={() => selectConversation(conv.id)}>
              <div className="sidebar-item-title">{conv.title || 'Untitled'}</div>
              <div className="sidebar-item-meta">{conv.message_count} messages</div>
            </div>
          ))}
        </div>
        <div className="sidebar-footer">
          <button className="sidebar-model-btn">Nyapsys 2B MoE</button>
        </div>
      </aside>

      <header className="topbar">
        <button className="topbar-hamburger" onClick={() => setSidebarOpen(true)}>☰</button>
        <img src="/logo.svg" alt="Nyapsys" className="topbar-logo-img" />
        <div className="status-pill"><span className="status-dot" /> Running locally</div>
      </header>

      <main className="messages-area">
        <div className="messages-container">
          {isEmpty ? (
            <div className="empty-state">
              <img src="/logo.svg" alt="Nyapsys" className="empty-logo" />
              <h2 className="empty-title">What can I help with?</h2>
              <p className="empty-subtitle">Ask anything, upload a file, or share an image. Nyapsys runs entirely on your Mac.</p>
              <div className="empty-pills">
                {starterPills.map(pill => (
                  <button key={pill.label} className="empty-pill" onClick={() => setInputText(pill.text)}>{pill.label}</button>
                ))}
              </div>
            </div>
          ) : (
            messages.map(msg => (
              <div key={msg.id} className={`message message-${msg.role}`}>
                {msg.role === 'assistant' && msg.isStreaming && !msg.content ? (
                  <div className="message-assistant">
                    <div className="message-avatar"><span className="avatar-n">N</span></div>
                    <div className="message-bubble typing-indicator"><span className="typing-dot" /><span className="typing-dot" /><span className="typing-dot" /></div>
                  </div>
                ) : (
                  <MessageBubble role={msg.role} content={msg.content} isStreaming={msg.isStreaming} />
                )}
              </div>
            ))
          )}
          {error && <div className="error-message">{error}</div>}
          <div ref={messagesEndRef} />
        </div>
      </main>

      <MessageInput onSend={sendMessage} onAttach={attachFile} onRemoveFile={removeFile} attachedFile={attachedFile} disabled={isLoading} value={inputText} onChange={setInputText} />
    </>
  )
}
