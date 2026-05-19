import { MessageBubble } from './MessageBubble'

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  isStreaming?: boolean
}

interface ChatWindowProps {
  messages: ChatMessage[]
  messagesEndRef: React.RefObject<HTMLDivElement>
}

export function ChatWindow({ messages, messagesEndRef }: ChatWindowProps) {
  return (
    <div className="chat-window">
      {messages.length === 0 ? (
        <div className="chat-empty">
          <h2>Welcome to Nyapsys</h2>
          <p>Ask me anything, upload files, or share images.</p>
        </div>
      ) : (
        messages.map((msg) => (
          <MessageBubble
            key={msg.id}
            role={msg.role}
            content={msg.content}
            isStreaming={msg.isStreaming}
          />
        ))
      )}
      <div ref={messagesEndRef} />
    </div>
  )
}