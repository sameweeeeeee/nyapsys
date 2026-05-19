'use client'

import { useChat } from '../hooks/useChat'
import { ChatWindow } from '../components/ChatWindow'
import { MessageInput } from '../components/MessageInput'

export default function Home() {
  const {
    messages,
    conversations,
    conversationId,
    isLoading,
    error,
    attachedFile,
    sendMessage,
    selectConversation,
    startNewConversation,
    removeConversation,
    attachFile,
    removeFile,
    messagesEndRef,
  } = useChat()

  return (
    <div className="app-container">
      <aside className="sidebar">
        <div className="sidebar-header">
          <h1>Nyapsys</h1>
          <button className="new-chat-btn" onClick={startNewConversation}>
            + New
          </button>
        </div>
        <div className="conversations-list">
          {conversations.map((conv) => (
            <div
              key={conv.id}
              className={`conversation-item ${conv.id === conversationId ? 'active' : ''}`}
              onClick={() => selectConversation(conv.id)}
            >
              <span className="conversation-title">
                {conv.title || 'New conversation'}
              </span>
              <button
                className="delete-conv-btn"
                onClick={(e) => {
                  e.stopPropagation()
                  removeConversation(conv.id)
                }}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      </aside>

      <main className="main-chat">
        <header className="chat-header">
          <h2>Chat</h2>
        </header>

        {error && (
          <div className="error-banner">
            {error}
          </div>
        )}

        <ChatWindow messages={messages} messagesEndRef={messagesEndRef} />

        <MessageInput
          onSend={sendMessage}
          onAttach={attachFile}
          onRemoveFile={removeFile}
          attachedFile={attachedFile}
          disabled={isLoading}
        />
      </main>
    </div>
  )
}