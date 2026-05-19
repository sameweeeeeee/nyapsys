const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
const API_KEY = process.env.NEXT_PUBLIC_API_KEY || ''

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  has_file?: boolean
  has_image?: boolean
  created_at: string
}

export interface Conversation {
  id: string
  title: string | null
  message_count: number
  updated_at: string
}

export async function* streamChat(
  message: string,
  conversationId: string,
  file?: File
): AsyncGenerator<string> {
  const formData = new FormData()
  formData.append('message', message)
  formData.append('conversation_id', conversationId)
  if (file) formData.append('file', file)

  const response = await fetch(`${API_URL}/chat`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${API_KEY}`,
    },
    body: formData,
  })

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`)
  }

  const reader = response.body!.getReader()
  const decoder = new TextDecoder()

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    const chunk = decoder.decode(value)
    const lines = chunk.split('\n')

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = line.slice(6)
        if (data === '[DONE]') return
        if (data) yield data
      }
    }
  }
}

export async function fetchConversations(): Promise<Conversation[]> {
  const response = await fetch(`${API_URL}/conversations`, {
    headers: { 'Authorization': `Bearer ${API_KEY}` },
  })
  if (!response.ok) throw new Error('Failed to fetch conversations')
  return response.json()
}

export async function fetchMessages(conversationId: string): Promise<Message[]> {
  const response = await fetch(`${API_URL}/conversations/${conversationId}/messages`, {
    headers: { 'Authorization': `Bearer ${API_KEY}` },
  })
  if (!response.ok) throw new Error('Failed to fetch messages')
  return response.json()
}

export async function deleteConversation(conversationId: string): Promise<void> {
  const response = await fetch(`${API_URL}/conversations/${conversationId}`, {
    method: 'DELETE',
    headers: { 'Authorization': `Bearer ${API_KEY}` },
  })
  if (!response.ok) throw new Error('Failed to delete conversation')
}