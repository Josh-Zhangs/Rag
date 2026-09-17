import { useState, useEffect } from 'react'
import { Modal, Spin, Alert, Button, Space, Empty } from 'antd'
import { DownloadOutlined, FilePdfOutlined, FileImageOutlined, FileTextOutlined, FileWordOutlined, FileExcelOutlined } from '@ant-design/icons'
import { getPreviewFileUrl, fetchPreviewContent, getDownloadUrl } from '../api/client'

// ── 类型定义 ──────────────────────────────────────────────

/** 预览渲染模式：决定 Modal 内用什么组件展示内容 */
type PreviewMode = 'iframe' | 'image' | 'text' | 'office'

// ── 文件类型判断工具函数 ───────────────────────────────────

/** PDF 扩展名集合 */
const PDF_EXTS = new Set(['pdf'])

/** 图片扩展名集合 */
const IMAGE_EXTS = new Set(['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'])

/** 文本扩展名集合（浏览器可直接渲染为文本的格式） */
const TEXT_EXTS = new Set(['txt', 'csv', 'json', 'html', 'htm', 'md', 'xml'])

/** 根据扩展名返回对应的文件类型图标 */
function getFileIcon(ext: string) {
    if (PDF_EXTS.has(ext)) return <FilePdfOutlined style={{ color: '#ef4444' }} />
    if (IMAGE_EXTS.has(ext)) return <FileImageOutlined style={{ color: '#8b5cf6' }} />
    if (TEXT_EXTS.has(ext)) return <FileTextOutlined style={{ color: '#64748b' }} />
    if (['docx', 'doc'].includes(ext)) return <FileWordOutlined style={{ color: '#2563eb' }} />
    if (['xlsx', 'xlsm', 'xls'].includes(ext)) return <FileExcelOutlined style={{ color: '#059669' }} />
    return <FileTextOutlined style={{ color: '#64748b' }} />
}

// ── Props ──────────────────────────────────────────────────

interface DocumentPreviewModalProps {
    /** 文件名（包含扩展名的完整文件名） */
    filename: string
    /** Modal 的显示状态 */
    visible: boolean
    /** 关闭 Modal 的回调 */
    onClose: () => void
}

// ── 组件 ──────────────────────────────────────────────────

export default function DocumentPreviewModal({ filename, visible, onClose }: DocumentPreviewModalProps) {
    const [previewMode, setPreviewMode] = useState<PreviewMode>('text')
    const [content, setContent] = useState('')
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState('')

    // 每次 Modal 显示时加载预览内容
    useEffect(() => {
        if (!visible || !filename) return
        loadPreview()
    }, [filename, visible])

    /** 获取文件的扩展名 */
    function getExt(name: string): string {
        return name.split('.').pop()?.toLowerCase() || ''
    }

    /** 根据扩展名选择合适的预览模式并加载内容 */
    async function loadPreview() {
        setLoading(true)
        setError('')
        setContent('')

        const ext = getExt(filename)

        try {
            if (PDF_EXTS.has(ext)) {
                // PDF：使用 iframe 嵌入浏览器原生 PDF 查看器
                setPreviewMode('iframe')
            } else if (IMAGE_EXTS.has(ext)) {
                // 图片：用 <img> 标签直接显示
                setPreviewMode('image')
            } else if (TEXT_EXTS.has(ext)) {
                // 文本文件：用 fetch 获取原始文本并在 <pre> 中显示
                setPreviewMode('text')
                const url = getPreviewFileUrl(filename)
                const res = await fetch(url)
                if (!res.ok) throw new Error(`HTTP ${res.status}`)
                setContent(await res.text())
            } else {
                // Office 文档（docx/xlsx 等）：从向量库重新组装 Markdown 文本
                setPreviewMode('office')
                const res = await fetchPreviewContent(filename)
                setContent(res.data.content)
            }
        } catch (err: any) {
            setError(err?.message || 'プレビューの読み込みに失敗しました')
        } finally {
            setLoading(false)
        }
    }

    /** 下载处理：点击 <a> 标签触发浏览器原生下载 */
    function handleDownload() {
        const url = getDownloadUrl(filename)
        const a = document.createElement('a')
        a.href = url
        a.download = filename
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
    }

    // ── 渲染 ────────────────────────────────────────────

    const ext = getExt(filename)

    return (
        <Modal
            open={visible}
            onCancel={onClose}
            footer={null}
            width={900}
            title={
                <Space>
                    {getFileIcon(ext)}
                    <span>{filename}</span>
                </Space>
            }
            destroyOnHidden
            styles={{
                body: { padding: loading ? '48px 24px' : '16px 24px', minHeight: 300 },
            }}
        >
            <Spin spinning={loading} tip="プレビューを読み込み中…">
                {/* 错误显示 */}
                {error && (
                    <Empty
                        description={error}
                        style={{ padding: '40px 0' }}
                    />
                )}

                {/* PDF 预览：在 iframe 中嵌入浏览器原生 PDF 查看器 */}
                {!error && previewMode === 'iframe' && (
                    <iframe
                        src={getPreviewFileUrl(filename)}
                        style={{
                            width: '100%',
                            height: '70vh',
                            border: '1px solid #e2e8f0',
                            borderRadius: 8,
                        }}
                        title={filename}
                    />
                )}

                {/* 图片预览：用 <img> 标签显示 */}
                {!error && previewMode === 'image' && (
                    <div style={{ textAlign: 'center' }}>
                        <img
                            src={getPreviewFileUrl(filename)}
                            alt={filename}
                            style={{
                                maxWidth: '100%',
                                maxHeight: '70vh',
                                objectFit: 'contain',
                                borderRadius: 8,
                            }}
                        />
                    </div>
                )}

                {/* 文本 / Office 文档预览 */}
                {!error && (previewMode === 'text' || previewMode === 'office') && (
                    <div>
                        {/* Office 文档专用：警告横幅（黄色醒目） */}
                        {previewMode === 'office' && (
                            <Alert
                                type="warning"
                                showIcon
                                message={
                                    <span style={{ fontWeight: 700, fontSize: 15 }}>
                                        マークダウン形式で表示中です
                                    </span>
                                }
                                description={
                                    <span style={{ fontSize: 13 }}>
                                        元の書式でご覧になる場合はダウンロードをおすすめします。
                                    </span>
                                }
                                action={
                                    <Button
                                        type="primary"
                                        size="small"
                                        icon={<DownloadOutlined />}
                                        onClick={handleDownload}
                                    >
                                        ダウンロード
                                    </Button>
                                }
                                style={{
                                    marginBottom: 16,
                                    border: '2px solid #fbbf24',
                                }}
                            />
                        )}
                        <pre
                            style={{
                                maxHeight: '65vh',
                                overflow: 'auto',
                                whiteSpace: 'pre-wrap',
                                wordBreak: 'break-word',
                                fontFamily: "'Segoe UI', 'Noto Sans JP', Meiryo, sans-serif",
                                fontSize: 14,
                                lineHeight: 1.8,
                                color: '#1e293b',
                                background: '#f8fafc',
                                border: '1px solid #e2e8f0',
                                borderRadius: 8,
                                padding: '16px 20px',
                                margin: 0,
                            }}
                        >
                            {content || '(内容が空です)'}
                        </pre>
                    </div>
                )}
            </Spin>
        </Modal>
    )
}
