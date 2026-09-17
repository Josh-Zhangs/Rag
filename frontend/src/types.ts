// 页面路由标识
export type PageKey = 'aiChat' | 'fileUpload' | 'settings'

// 检索证据项：后端返回的原文片段
export interface EvidenceItem {
    // 来源文件名（如 000074366.pdf）
    source_file: string
    // PDF 页码（从 1 开始，-1 表示未知）
    page_number: number
    // 原始文档片段文本
    text: string
    // Cross-Encoder 重排序分数（0~1，越高越相关）
    rerank_score: number
    // XLSX 的 Sheet 名称（仅 Excel 文件有值，PDF/DOCX 等为 undefined）
    sheet_name?: string
}

// 聊天消息：每条消息可以是用户提问或 AI 回答
export interface ChatMessage {
    // 角色：用户或 AI
    role: 'user' | 'ai'
    // 消息正文（AI 回答时是 LLM 生成的内容，用户提问时是原始问题）
    content: string
    // AI 回答时附带的检索证据（用户消息无此字段）
    evidence?: EvidenceItem[]
    // 话题分隔线标记：当 LLM 检测到话题切换时，在该消息前显示分隔线
    topicChanged?: boolean
    // AI 理解后的标准化问题（预处理结果，可选展示）
    resolvedQuestion?: string
}

// ── 多轮对话 STM（短期记忆）类型 ──────────────────────────

// 多轮对话：一轮对话（用于 API 请求，发给后端）
export interface ConversationTurn {
    // 用户的问题
    question: string
    // AI 的回答
    answer: string
}

// IndexedDB 中存储的会话（持久化，刷新页面后恢复）
export interface Session {
    // 唯一标识（时间戳 + 随机后缀）
    id: string
    // 会话标题（第一轮问题截断 20 字，自动生成）
    title: string
    // 创建时间戳（毫秒）
    createdAt: number
    // 最后更新时间戳（毫秒）
    updatedAt: number
    // 对话轮次数组
    turns: Turn[]
}

// 会话中的一轮（IndexedDB 存储，含完整证据）
export interface Turn {
    // 用户原始问题
    question: string
    // AI 回答全文
    answer: string
    // 检索证据（可选，刷新后可回显出典卡片）
    evidence?: EvidenceItem[]
    // 预处理后的标准化问题（调试/透明性用）
    resolvedQuestion?: string
    // 话题是否切换（软信号，前端显示分隔线）
    topicChanged?: boolean
    // 时间戳（毫秒）
    timestamp: number
}

// 文档列表项：后端 GET /api/documents 返回的单条记录
export interface DocumentItem {
    source_file: string
    doc_type: string
    file_size: number
    created_at: string
}