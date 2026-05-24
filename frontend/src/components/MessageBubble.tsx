'use client'

import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneLight } from 'react-syntax-highlighter/dist/cjs/styles/prism'

interface MessageBubbleProps {
  role: 'user' | 'assistant'
  content: string
  isStreaming?: boolean
}

export function MessageBubble({ role, content, isStreaming }: MessageBubbleProps) {
  const isUser = role === 'user'
  return (
    <>
      <div className="message-avatar">{isUser ? '👤' : <span className="avatar-n">N</span>}</div>
      <div className="message-bubble">
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={{
          code({ className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || '')
            const isInline = !match
            return isInline ? <code className="inline-code" {...props}>{children}</code> : <SyntaxHighlighter style={oneLight} language={match[1]} PreTag="div">{String(children).replace(/\n$/, '')}</SyntaxHighlighter>
          },
        }}>{content}</ReactMarkdown>
        {isStreaming && <span className="streaming-cursor">▊</span>}
      </div>
    </>
  )
}