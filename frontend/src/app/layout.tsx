import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Nyapsys - AI Assistant',
  description: 'Self-hosted AI agent with file and image understanding',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}