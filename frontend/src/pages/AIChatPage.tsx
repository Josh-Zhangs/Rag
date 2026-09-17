// 前端类型定义
import { useState, useEffect, useRef } from 'react'
import type { ChatMessage } from '../types'
// antd 图标：统一使用矢量图标替代 emoji，保持跨平台一致的视觉品质
import {
  UserOutlined,
  RobotOutlined,
  CaretDownOutlined,
  CaretRightOutlined,
  ProfileOutlined,
  LoadingOutlined,
  SendOutlined,
  SafetyOutlined,
  SmileTwoTone,
  PlusOutlined,
} from '@ant-design/icons'
// 共享预览组件（PDF / 图片 / 文本 / Office 文档）
import DocumentPreviewModal from '../components/DocumentPreviewModal'
// 页面样式
import '../styles/pages/AIChatPage.css'

// 等待中轮播文案：对应实际处理流程的三个阶段
const WAITING_MESSAGES = [
    '内資料から関連箇所を検索しています…',
    '資料の関連度を精査しています…',
    '回答を作成中…',
]
// 轮播间隔（毫秒）
const WAITING_INTERVAL = 2500

// 组件 Props 定义
type AIChatPageProps = {
    messages: ChatMessage[]
    question: string
    onQuestionChange: (value: string) => void
    onSend: () => void
    isLoading?: boolean
    // 新規会話回调：创建新会话，清空当前聊天界面
    onNewConversation?: () => void
}

// 计算相似度百分比：rerank_score 越高 → 相似度越高
function getSimilarityPercent(score: number): number {
    return Math.round(score * 100)
}

export default function AIChatPage({ messages, question, onQuestionChange, onSend, isLoading, onNewConversation }: AIChatPageProps) {
    // 出典展开状态：使用 Set 记录哪条消息的出典卡片已展开
    const [expandedEvidence, setExpandedEvidence] = useState<Set<number>>(new Set())
    // 等待中轮播文案索引：循环 0→1→2→0→...
    const [waitingIndex, setWaitingIndex] = useState(0)

    // 预览 Modal 状态（从出典卡片的 Open 按钮打开）
    const [previewVisible, setPreviewVisible] = useState(false)
    const [previewFilename, setPreviewFilename] = useState('')

    // 等待中轮播定时器：每 WAITING_INTERVAL ms 切换到下一句文案
    useEffect(() => {
        if (!isLoading) {
            // 加载结束，重置索引
            setWaitingIndex(0)
            return
        }
        const interval = setInterval(() => {
            setWaitingIndex((prev) => (prev + 1) % WAITING_MESSAGES.length)
        }, WAITING_INTERVAL)
        return () => clearInterval(interval)
    }, [isLoading])

    // 聊天区 DOM 引用，用于自动滚动到底部
    const chatHistoryRef = useRef<HTMLDivElement>(null)
    const prevIsLoadingRef = useRef(isLoading)

    useEffect(() => {
        // 流式输出中：即时跟随；AI 回答刚完成：平滑滚到底
        if (isLoading || (prevIsLoadingRef.current && !isLoading)) {
            chatHistoryRef.current?.scrollTo({
                top: chatHistoryRef.current.scrollHeight,
                behavior: isLoading ? 'auto' : 'smooth',
            })
        }
        prevIsLoadingRef.current = isLoading
    }, [messages, isLoading])

    // 切换出典卡片展开/折叠
    const toggleEvidence = (index: number) => {
        setExpandedEvidence((prev) => {
            const next = new Set(prev)
            if (next.has(index)) {
                next.delete(index)
            } else {
                next.add(index)
            }
            return next
        })
    }
    // 出典卡片的 Open 按钮：打开预览 Modal
    const handleOpenPreview = (sourceFile: string) => {
        setPreviewFilename(sourceFile)
        setPreviewVisible(true)
    }

    // 键盘事件：Enter 发送，Shift+Enter 换行
    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            onSend()
        }
    }

    return (
        <div className="search-page">
            <div className="chat-history" ref={chatHistoryRef}>
                {messages.map((item, index) => {
                    // 判断当前消息是否为等待中的 AI 消息（内容为空 + 正在加载 + 是最后一条）
                    const isLastMessage = index === messages.length - 1
                    const isWaitingForAI = isLastMessage && item.role === 'ai' && !item.content && isLoading

                    return (
                    <div key={index} className="message-wrapper">
                        {/* ── 话题切换分隔线（软信号） ──────────────────── */}
                        {item.topicChanged && (
                            <div className="topic-divider">
                                <span className="topic-divider-line" />
                                <span className="topic-divider-label">
                                    トピックが変更されました
                                </span>
                                <span className="topic-divider-line" />
                            </div>
                        )}
                    <div className={`message ${item.role}`}>
                        {/* 头像 */}
                        <div className="avatar">
                            {item.role === 'user'
                                ? <UserOutlined style={{ fontSize: 18 }} />
                                : <SmileTwoTone style={{ fontSize: 18 }} />
                            }
                        </div>
                        <div className="bubble">
                            {/* 消息正文 */}
                            <div className="answer-content">
                                {isWaitingForAI ? (
                                    <span className="waiting-message">
                                        {WAITING_MESSAGES[waitingIndex]}
                                    </span>
                                ) : (
                                    item.content
                                )}
                            </div>

                            {/* 出典卡片：仅在 AI 消息且有证据时展示，默认折叠 */}
                            {item.role === 'ai' && item.evidence && item.evidence.length > 0 && (
                                <div className="evidence-cards">
                                    <div
                                        className="evidence-toggle"
                                        onClick={() => toggleEvidence(index)}
                                    >
                                        <span className="evidence-toggle-icon">
                                            {expandedEvidence.has(index)
                                                ? <CaretDownOutlined style={{ fontSize: 10 }} />
                                                : <CaretRightOutlined style={{ fontSize: 10 }} />
                                            }
                                        </span>
                                        <span className="evidence-toggle-label">
                                            <ProfileOutlined style={{ fontSize: 14, marginRight: 4, verticalAlign: 'middle' }} />
                                            出典
                                        </span>
                                    </div>
                                    {expandedEvidence.has(index) && (
                                        <div className="evidence-body">
                                            {item.evidence.map((ev, i) => {
                                                const similarity = getSimilarityPercent(ev.rerank_score)
                                                return (
                                                    <div key={i} className="evidence-card">
                                                        <div className="evidence-card-header">
                                                            <span className="evidence-source">
                                                                {ev.source_file}
                                                            </span>
                                                            {ev.sheet_name ? (
                                                                <span className="evidence-page">
                                                                    Sheet: {ev.sheet_name}
                                                                </span>
                                                            ) : ev.page_number > 0 && (
                                                                <span className="evidence-page">
                                                                    Page {ev.page_number}
                                                                </span>
                                                            )}
                                                            <span className="evidence-similarity">
                                                                {similarity}% Match
                                                            </span>
                                                            {/* 打开文档按钮 — 与 Page/Similarity 同风格的 pill 型，放在右端 */}
                                                            <span
                                                                className="evidence-open-btn"
                                                                onClick={(e) => {
                                                                    e.stopPropagation()
                                                                    handleOpenPreview(ev.source_file)
                                                                }}
                                                            >
                                                                文書を開く
                                                            </span>
                                                        </div>
                                                        <div className="evidence-card-text">
                                                            {ev.text}
                                                        </div>
                                                    </div>
                                                )
                                            })}
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* ── 標準化質問表示（多轮对话 STM 预处理结果） ────────── */}
                            {item.role === 'ai' && item.resolvedQuestion && (
                                <div className="resolved-question-note">
                                    解釈: {item.resolvedQuestion}
                                </div>
                            )}
                        </div>
                    </div>
                    </div>
                )})}
            </div>

            {/* 输入区域 */}
            <div className="input-area">
                <textarea
                    value={question}
                    onChange={(e) => onQuestionChange(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="質問を入力してください… (例: インボイス制度の運用ルール)"
                    disabled={isLoading}
                />
                <button type="button" className="send-btn" onClick={onSend} disabled={isLoading}>
                    <span>送信</span>
                    {isLoading
                        ? <LoadingOutlined spin style={{ fontSize: 16 }} />
                        : <SendOutlined style={{ fontSize: 16 }} />
                    }
                </button>
            </div>

            {/* 输入提示 */}
            <div className="input-hint">
                <span className="hint-left">
                    <SafetyOutlined style={{ fontSize: 13, verticalAlign: 'middle', marginRight: 2 }} />
                    社内文書のみを根拠に回答を生成します
                    {onNewConversation && (
                        <button
                            type="button"
                            className="new-conversation-btn"
                            onClick={onNewConversation}
                            disabled={isLoading}
                        >
                            <PlusOutlined style={{ fontSize: 12 }} />
                            新規会話
                        </button>
                    )}
                </span>
                <span>Shift+Enter で改行</span>
            </div>

            {/* 预览 Modal：从出典卡片的 Open 按钮打开 */}
            <DocumentPreviewModal
                filename={previewFilename}
                visible={previewVisible}
                onClose={() => setPreviewVisible(false)}
            />
        </div>
    )
}