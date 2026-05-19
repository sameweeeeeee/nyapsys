import { useState, useCallback, useRef, useEffect } from 'react'
import { streamChat, fetchConversations, fetchMessages, deleteConversation, Message, Conversation } from '../lib/api'

function generateId(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = Math.random() * 16 | 0
    const v = c === 'x' ? r : (r & 0x3 | 0x8)
    return v.toString(16)
  })
}

interface ChatMessage extends Message {
  isStreaming?: boolean
}

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [conversationId, setConversationId] = useState<string>(generateId())
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [attachedFile, setAttachedFile] = useState<File | null>(null)

  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  const loadConversations = useCallback(async () => {
    try {
      const convs = await fetchConversations()
      setConversations(convs)
    } catch (err) {
      console.error('Failed to load conversations:', err)
    }
  }, [])

  const loadMessages = useCallback(async (convId: string) => {
    try {
      const msgs = await fetchMessages(convId)
      setMessages(msgs.map(m => ({ ...m, isStreaming: false })))
    } catch (err) {
      console.error('Failed to load messages:', err)
    }
  }, [])

  useEffect(() => {
    loadConversations()
  }, [loadConversations])

  const selectConversation = useCallback((convId: string) => {
    setConversationId(convId)
    setMessages([])
    setError(null)
    loadMessages(convId)
  }, [loadMessages])

  const startNewConversation = useCallback(() => {
    const newId = generateId()
    setConversationId(newId)
    setMessages([])
    setError(null)
    setAttachedFile(null)
  }, [])

  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim() && !attachedFile) return

    setIsLoading(true)
    setError(null)

    const userMessage: ChatMessage = {
      id: generateId(),
      role: 'user',
      content: content || (attachedFile ? `File: ${attachedFile.name}` : ''),
      created_at: new Date().toISOString(),
      isStreaming: false,
    }

    const assistantMessage: ChatMessage = {
      id: generateId(),
      role: 'assistant',
      content: '',
      created_at: new Date().toISOString(),
      isStreaming: true,
    }

    setMessages(prev => [...prev, userMessage, assistantMessage])

    try {
      const stream = streamChat(content, conversationId, attachedFile || undefined)
      let fullResponse = ''

      for await (const token of stream) {
        fullResponse += token
        setMessages(prev =>
          prev.map((m, i) =>
            m.id === assistantMessage.id
              ? { ...m, content: fullResponse }
              : m
          )
        )
      }

      setMessages(prev =>
        prev.map(m =>
          m.id === assistantMessage.id
            ? { ...m, isStreaming: false }
            : m
        )
      )

      setAttachedFile(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send message')
      setMessages(prev =>
        prev.map(m =>
          m.id === assistantMessage.id
            ? { ...m, content: 'Error: Failed to get response', isStreaming: false }
            : m
        )
      )
    } finally {
      setIsLoading(false)
    }
  }, [conversationId, attachedFile])

  const removeConversation = useCallback(async (convId: string) => {
    try {
      await deleteConversation(convId)
      setConversations(prev => prev.filter(c => c.id !== convId))
      if (convId === conversationId) {
        startNewConversation()
      }
    } catch (err) {
      console.error('Failed to delete conversation:', err)
    }
  }, [conversationId, startNewConversation])

  const attachFile = useCallback((file: File) => {
    setAttachedFile(file)
  }, [])

  const removeFile = useCallback(() => {
    setAttachedFile(null)
  }, [])

  return {
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
  }
}