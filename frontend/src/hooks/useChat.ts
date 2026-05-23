import { useState, useCallback, useRef, useEffect } from 'react'
import { streamChat, getConversations, deleteConversation, getMessages, Message, Conversation } from '../lib/api'
import { v4 as uuidv4 } from 'uuid'

interface ChatMessage extends Message {
  isStreaming?: boolean
}

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [conversationId, setConversationId] = useState<string>(uuidv4())
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [attachedFile, setAttachedFile] = useState<File | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [inputText, setInputText] = useState('')
  const [view, setView] = useState<'chat' | 'settings'>('chat')
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => { scrollToBottom() }, [messages, scrollToBottom])

  const loadConversations = useCallback(async () => {
    try {
      const convs = await getConversations()
      setConversations(convs)
    } catch (err) { console.error('Failed to load conversations:', err) }
  }, [])

  useEffect(() => { loadConversations() }, [loadConversations])

  const selectConversation = useCallback(async (convId: string) => {
    setConversationId(convId)
    setError(null)
    setSidebarOpen(false)
    try {
      const msgs = await getMessages(convId)
      setMessages(msgs.map(m => ({ ...m, isStreaming: false })))
    } catch (err) { console.error('Failed to load messages:', err) }
  }, [])

  const startNewConversation = useCallback(() => {
    const newId = uuidv4()
    setConversationId(newId)
    setMessages([])
    setError(null)
    setAttachedFile(null)
    setInputText('')
    setSidebarOpen(false)
  }, [])

  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim() && !attachedFile) return
    setIsLoading(true)
    setError(null)

    const userMessage: ChatMessage = { id: uuidv4(), role: 'user', content: content || `File: ${attachedFile?.name}`, created_at: new Date().toISOString(), isStreaming: false }
    const assistantMessage: ChatMessage = { id: uuidv4(), role: 'assistant', content: '', created_at: new Date().toISOString(), isStreaming: true }

    setMessages(prev => [...prev, userMessage, assistantMessage])

    try {
      const stream = streamChat(content, conversationId, attachedFile || undefined)
      let fullResponse = ''
      for await (const token of stream) {
        fullResponse += token
        setMessages(prev => prev.map((m, i) => m.id === assistantMessage.id ? { ...m, content: fullResponse } : m))
      }
      setMessages(prev => prev.map(m => m.id === assistantMessage.id ? { ...m, isStreaming: false } : m))
      setAttachedFile(null)
      loadConversations()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send message')
      setMessages(prev => prev.map(m => m.id === assistantMessage.id ? { ...m, content: 'Error: Failed to get response', isStreaming: false } : m))
    } finally {
      setIsLoading(false)
    }
  }, [conversationId, attachedFile, loadConversations])

  const removeConversation = useCallback(async (convId: string) => {
    try {
      await deleteConversation(convId)
      setConversations(prev => prev.filter(c => c.id !== convId))
      if (convId === conversationId) startNewConversation()
    } catch (err) { console.error('Failed to delete:', err) }
  }, [conversationId, startNewConversation])

  const attachFile = useCallback((file: File) => { setAttachedFile(file) }, [])
  const removeFile = useCallback(() => { setAttachedFile(null) }, [])

  return { messages, conversations, conversationId, isLoading, error, attachedFile, sidebarOpen, setSidebarOpen, inputText, setInputText, sendMessage, selectConversation, startNewConversation, removeConversation, attachFile, removeFile, messagesEndRef, view, setView }
}
