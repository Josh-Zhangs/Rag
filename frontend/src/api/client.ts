// HTTP 请求库
import axios from 'axios'
// 前端类型定义
import type { EvidenceItem, ConversationTurn } from '../types'

// 后端聊天接口返回的响应结构
export interface ChatResponse {
    // LLM 生成的回答文本
    answer: string
    // 检索到的原文片段列表
    evidence: EvidenceItem[]
}

// 向量入库结果
export interface VectorResult {
  status: 'success' | 'error'
  error?: string
  error_type?: string
  inserted_count?: number
  chunks_count?: number
  table_name?: string
  message?: string
}

// 单个文件上传结果
export interface UploadFileResult {
  filename: string
  clean_markdown: string
  pdf_path: string
  vector_result: VectorResult | null
}

// 文件上传接口返回的整体响应
export interface UploadResponse {
    code: number
    msg: string
    data: UploadFileResult[]
}

// 后端 API 基础地址（开发模式由 Vite 代理转发，打包后同源直连）
const API_BASE = '/api'

// 创建 Axios 实例，配置基础 URL 和默认请求头
const axiosInstance = axios.create({
    baseURL: API_BASE,
    headers: {
        'Content-Type': 'application/json',
    },
})

// 流式 SSE 事件的数据结构
export interface StreamChunk {
    token?: string
    evidence?: EvidenceItem[]
    error?: string
    // 新增：预处理结果（多轮对话 STM）
    // resolved_question: 指代消解后的标准化问题（可选展示给用户）
    resolved_question?: string
    // topic_changed: 话题是否切换
    topic_changed?: boolean
}

// 流式发送聊天问题，返回 AsyncGenerator 逐 token 推送
// 使用 fetch + ReadableStream 消费 SSE 事件流
// history: 多轮对话历史（默认空数组，首轮无历史）
export async function* sendChatStream(question: string, history: ConversationTurn[] = []): AsyncGenerator<StreamChunk> {
    // 发起 POST 请求到流式端点
    const response = await fetch(`${API_BASE}/ai-chat/stream`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question, history }),
    })

    // 检查 HTTP 状态码
    if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`)
    }

    // 获取可读流 reader
    const reader = response.body?.getReader()
    if (!reader) {
        throw new Error('Response body is not readable')
    }

    // 文本解码器：将字节流解码为 UTF-8 字符串
    const decoder = new TextDecoder('utf-8')
    // 缓冲区：SSE 事件可能跨 chunk 边界，按 \n\n 分隔事件
    let buffer = ''

    try {
        while (true) {
            // 读取下一个 chunk
            const { done, value } = await reader.read()
            // 流结束
            if (done) {
                break
            }
            // 解码字节并追加到缓冲区
            buffer += decoder.decode(value, { stream: true })

            // 按 \n\n 分割 SSE 事件（SSE 协议以双换行分隔事件）
            const events = buffer.split('\n\n')
            // 最后一段可能不完整，保留在缓冲区
            buffer = events.pop() || ''

            for (const eventBlock of events) {
                // 处理事件块内的每一行
                for (const line of eventBlock.split('\n')) {
                    // 去除行首尾空白和 \r
                    const trimmedLine = line.trim()
                    // 只处理以 "data:" 开头的行
                    if (!trimmedLine.startsWith('data:')) {
                        continue
                    }
                    // 提取 JSON 字符串（去掉 "data:" 前缀）
                    const dataStr = trimmedLine.slice(5).trim()
                    if (!dataStr) {
                        continue
                    }
                    try {
                        // 解析 JSON 并 yield
                        const chunk: StreamChunk = JSON.parse(dataStr)
                        yield chunk
                    } catch {
                        // JSON 解析失败，跳过该行
                        console.warn('SSE parse error:', dataStr)
                    }
                }
            }
        }
    } finally {
        // 确保 reader 被释放，避免内存泄漏
        reader.releaseLock()
    }
}

// 上传 PDF 文件，返回处理结果
export async function uploadDocuments(files: File[]): Promise<UploadResponse> {
    const formData = new FormData()
    files.forEach((file) => {
        formData.append('files', file)
    })
    const response = await axiosInstance.post('/file-upload', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    })
    return response.data
}

// 文件存在性检查响应结构
export interface CheckFileExistsResponse {
    code: number
    msg: string
    data: {
        filename: string
        exists: boolean
    }
}

// 检查文件是否已存在于向量数据库
export async function checkFileExists(filename: string): Promise<CheckFileExistsResponse> {
    const response = await axiosInstance.get('/file-upload/check-exists', {
        params: { filename },
    })
    return response.data
}

// 文档统计响应结构
export interface DocumentStatsResponse {
    code: number
    msg: string
    data: {
        stats: Record<string, number>
        last_updated: string
    }
}

// 获取文档统计信息（按类型统计文档数量及最后更新时间）
export async function fetchDocumentStats(): Promise<DocumentStatsResponse> {
    const response = await axiosInstance.get('/document-stats')
    return response.data
}

// 文档列表响应结构
export interface DocumentListResponse {
    code: number
    msg: string
    data: {
        source_file: string
        doc_type: string
        file_size: number
        created_at: string
    }[]
}

// 获取已入库文档列表
export async function fetchDocuments(): Promise<DocumentListResponse> {
    const response = await axiosInstance.get('/documents')
    return response.data
}

// 删除文档响应结构
export interface DeleteDocumentResponse {
    code: number
    msg: string
    data: {
        filename: string
        chunks_deleted: number
        file_deleted: boolean
    }
}

// 删除指定文档（向量库 chunk + 物理文件）
export async function deleteDocument(filename: string): Promise<DeleteDocumentResponse> {
    const response = await axiosInstance.delete(
        `/file-upload/${encodeURIComponent(filename)}`
    )
    return response.data
}

// ── 文档更新 ──────────────────────────────────────────────

// 更新文档响应结构
export interface UpdateDocumentResponse {
    code: number
    msg: string
    data: {
        old_filename: string
        new_filename: string
        extension_changed: boolean
        chunks_count: number
    }
}

// 更新指定文件（文件替换+矢量数据再取入）
// 后端用parse-first，commit-last进行原子更新。
export async function updateDocument(
    oldFilename: string,
    file: File,
): Promise<UpdateDocumentResponse> {
    const formData = new FormData()
    formData.append('file', file)
    const response = await axiosInstance.put(
        '/file-upload/update',
        formData,
        {
            params: { old_filename: oldFilename },
            headers: { 'Content-Type': 'multipart/form-data' },
        },
    )
    return response.data
}

// ── 预览相关 ──────────────────────────────────────────────

// 预览内容响应（从向量库组装 Markdown 文本）
export interface PreviewContentResponse {
    code: number
    msg: string
    data: {
        filename: string
        content: string
    }
}

// 返回原始文件的预览 URL，PDF/图片用 iframe/img 内联渲染
export function getPreviewFileUrl(filename: string): string {
    return `${API_BASE}/file-upload/preview-file/${encodeURIComponent(filename)}`
}

// 获取 Office 文档的 Markdown 预览文本（从向量库 chunks 组装）
export async function fetchPreviewContent(filename: string): Promise<PreviewContentResponse> {
    const response = await axiosInstance.get(
        `/file-upload/preview-content/${encodeURIComponent(filename)}`
    )
    return response.data
}

// 返回单文件下载的完整 URL，前端通过 <a> 标签点击触发浏览器原生下载
export function getDownloadUrl(filename: string): string {
    return `${API_BASE}/file-upload/download/${encodeURIComponent(filename)}`
}

// 批量下载：请求后端打包为 ZIP 并触发浏览器下载
export async function downloadBatch(filenames: string[]): Promise<void> {
    const response = await axiosInstance.post(
        '/file-upload/download-batch',
        { filenames },
        { responseType: 'blob' },
    )

    // 从 Content-Disposition 头提取文件名，fallback 为带时间戳的默认名
    const disposition = (response.headers as Record<string, string>)['content-disposition'] || ''
    const match = disposition.match(/filename="?(.+?)"?$/i)
    const zipName = match ? match[1] : `machiari_documents.zip`

    // 创建 Blob URL 并触发下载
    const url = window.URL.createObjectURL(response.data)
    const a = document.createElement('a')
    a.href = url
    a.download = zipName
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    window.URL.revokeObjectURL(url)
}