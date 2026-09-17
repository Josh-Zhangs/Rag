import { useState, useEffect, useCallback } from 'react'
import TopHeader from './components/TopHeader'
import SidebarMenu from './components/SidebarMenu'
import AIChatPage from './pages/AIChatPage'
import FileManagePage from './pages/FileManagePage'
import SettingsPage from './pages/SettingsPage'
import type { ChatMessage, PageKey, ConversationTurn, Turn } from './types'
import { sendChatStream } from './api/client'
import type { EvidenceItem } from './types'
import {
    createSession, getSession, appendTurn,
    getActiveSessionId, setActiveSessionId,
} from './lib/sessionStore'
import './styles/styles.css'

const initialMessages: ChatMessage[] = [
  {
    role: 'ai',
    content:
      'こんにちは。社内ナレッジ検索システムへようこそ。\n左のメニューから操作を選択し、文書をアップロードしてから質問してください。'
  }
]

function App() {
  const [activePage, setActivePage] = useState<PageKey>('aiChat')
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages)
  const [isLoading, setIsLoading] = useState(false)
  const [statsRefreshTrigger, setStatsRefreshTrigger] = useState(0)
  const [sessionId, setSessionId] = useState<string | null>(null)

  const handlePageChange = (page: PageKey) => {
    setActivePage(page)
  }

  // ── 页面加载时恢复会话 ────────────────────────────────────
  useEffect(() => {
    const restoreSession = async () => {
      const activeId = getActiveSessionId()
      if (activeId) {
        const session = await getSession(activeId)
        if (session && session.turns.length > 0) {
          // 从 IndexedDB 恢复消息列表
          const restoredMessages: ChatMessage[] = [initialMessages[0]]
          for (const turn of session.turns) {
            restoredMessages.push({ role: 'user', content: turn.question })
            restoredMessages.push({
              role: 'ai',
              content: turn.answer,
              evidence: turn.evidence,
              topicChanged: turn.topicChanged,
              resolvedQuestion: turn.resolvedQuestion,
            })
          }
          setMessages(restoredMessages)
          setSessionId(activeId)
          return
        }
      }
      // 没有活跃会话或会话为空，创建新的
      const newId = await createSession()
      setActiveSessionId(newId)
      setSessionId(newId)
    }
    restoreSession()
  }, [])

  // ── 新規会話：创建新会话，清空消息 ────────────────────────
  const handleNewConversation = useCallback(async () => {
    const newId = await createSession()
    setActiveSessionId(newId)
    setSessionId(newId)
    setMessages(initialMessages)
    setQuestion('')
  }, [])

  const handleSend = async () => {
    // 空输入或加载中时不允许发送
    if (!question.trim() || isLoading) {
      return
    }

    // 先保存用户问题再清空输入框，避免闭包问题
    const userQuestion = question
    setIsLoading(true)

    // ── 从当前消息构建对话历史（在添加当前轮之前） ──────────
    // messages[0] 是初始欢迎消息，跳过；剩余的按 user+ai 配对
    const history: ConversationTurn[] = []
    const chatMessages = messages.slice(1) // 去掉初始欢迎消息
    for (let i = 0; i < chatMessages.length; i += 2) {
      if (i + 1 < chatMessages.length) {
        const q = chatMessages[i]
        const a = chatMessages[i + 1]
        // 只收集有内容的完整轮次（排除空的 AI 占位消息）
        if (q.role === 'user' && a.role === 'ai' && a.content) {
          history.push({ question: q.content, answer: a.content })
        }
      }
    }
    // 最多 10 轮（后端预处理和回答 LLM 各自取子集）
    const recentHistory = history.slice(-10)

    // 添加用户消息到聊天历史
    setMessages((prev) => [...prev, { role: 'user', content: userQuestion }])
    setQuestion('')

    // 添加空的 AI 消息占位，后续流式更新其 content
    setMessages((prev) => [...prev, { role: 'ai', content: '' }])

    // 累积的完整回答内容
    let fullContent = ''
    // 检索证据（流式结束后从最后一个事件获取）
    let evidence: EvidenceItem[] = []
    // 预处理结果（用于保存到 IndexedDB）
    let resolvedQuestion = ''
    let topicChanged = false

    try {
      // 流式调用：for await 消费 SSE 事件流
      // 事件顺序：topic_changed? → resolved_question? → token × N → evidence
      for await (const chunk of sendChatStream(userQuestion, recentHistory)) {
        if (chunk.topic_changed) {
          // 话题切换：软信号，在 AI 消息上标记（不清空历史）
          topicChanged = true
          setMessages((prev) => {
            const updated = [...prev]
            updated[updated.length - 1] = {
              ...updated[updated.length - 1],
              topicChanged: true,
            }
            return updated
          })
        } else if (chunk.resolved_question) {
          // 标准化问题：存下来用于 IndexedDB
          resolvedQuestion = chunk.resolved_question
          setMessages((prev) => {
            const updated = [...prev]
            updated[updated.length - 1] = {
              ...updated[updated.length - 1],
              resolvedQuestion: chunk.resolved_question,
            }
            return updated
          })
        } else if (chunk.token) {
          // token 事件：逐字追加到回答内容
          fullContent += chunk.token
          // 更新最后一条 AI 消息的 content（浅拷贝数组，替换最后一项）
          setMessages((prev) => {
            const updated = [...prev]
            updated[updated.length - 1] = {
              ...updated[updated.length - 1],
              content: fullContent,
            }
            return updated
          })
        } else if (chunk.evidence) {
          // evidence 事件：流式结束后推送检索证据
          evidence = chunk.evidence
        } else if (chunk.error) {
          // error 事件：LLM 调用失败，显示错误消息
          fullContent = chunk.error
          break
        }
      }
    } catch (error) {
      // 网络异常或流中断：显示错误提示
      console.error('Chat stream error:', error)
      fullContent = fullContent || '申し訳ありません、回答を取得できませんでした。'
    } finally {
      // 无论成功或失败，最终更新 AI 消息（含 evidence）
      setMessages((prev) => {
        const updated = [...prev]
        updated[updated.length - 1] = {
          ...updated[updated.length - 1],
          content: fullContent || '回答を生成できませんでした',
          evidence,
        }
        return updated
      })

      // ── 保存本轮对话到 IndexedDB ──────────────────────────
      if (sessionId && fullContent) {
        const turn: Turn = {
          question: userQuestion,
          answer: fullContent,
          evidence: evidence.length > 0 ? evidence : undefined,
          resolvedQuestion: resolvedQuestion || undefined,
          topicChanged: topicChanged || undefined,
          timestamp: Date.now(),
        }
        try {
          await appendTurn(sessionId, turn)
        } catch (e) {
          console.error('Failed to save turn to IndexedDB:', e)
        }
      }

      setIsLoading(false)
    }
  }

  const renderPage = () => {
    switch (activePage) {
      case 'fileUpload':
        return (
          <FileManagePage onUploadComplete={() => {
            setStatsRefreshTrigger(prev => prev + 1)
          }} />
        )
      case 'settings':
        return <SettingsPage />
      default:
        return (
          <AIChatPage
            messages={messages}
            question={question}
            onQuestionChange={setQuestion}
            onSend={handleSend}
            isLoading={isLoading}
            onNewConversation={handleNewConversation}
          />
        )
    }
  }

  return (
    <div>
      <TopHeader />
      <div className="layout">
        <aside className="sidebar">
          <SidebarMenu activePage={activePage} onChange={handlePageChange} refreshTrigger={statsRefreshTrigger} />
        </aside>

        <main className="main">{renderPage()}</main>
      </div>
    </div>
  )
}

export default App