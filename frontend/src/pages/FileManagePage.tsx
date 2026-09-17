import { useState, useEffect, useMemo, useRef, useCallback } from 'react'
import type { ReactNode } from 'react'
import {
  Button,
  Card,
  Space,
  Typography,
  Upload,
  Tabs,
  Table,
  Tag,
  Input,
  Segmented,
  Popconfirm,
  Tooltip,
  message,
  Modal,
  Alert,
} from 'antd'
import {
  InboxOutlined,
  UploadOutlined,
  SearchOutlined,
  EyeOutlined,
  DownloadOutlined,
  SyncOutlined,
  DeleteOutlined,
  CloseOutlined,
  ReloadOutlined,
  ClearOutlined,
  FilePdfOutlined,
  FileWordOutlined,
  FileExcelOutlined,
  FileTextOutlined,
  FileImageOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  WarningOutlined,
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import { uploadDocuments, checkFileExists, fetchDocuments, deleteDocument, getDownloadUrl, downloadBatch, updateDocument, type UploadResponse } from '../api/client'
import DocumentPreviewModal from '../components/DocumentPreviewModal'
import '../styles/pages/FileManagePage.css'

// 错误类型 → 日文提示映射
const errorTypeMessages: Record<string, string> = {
  libreoffice_not_found: '旧形式ファイル(.doc/.xls)を処理するため、LibreOfficeが見つかりません。Cドライブ標準パスへインストール後、再度アップロードしてください。',
  file_conversion_failed: 'ファイルの変換に失敗しました。',
  unknown_error: 'ファイルの処理中にエラーが発生しました。',
}

// 从后端响应中提取错误信息
function extractErrorMessage(response: UploadResponse, index: number): string | null {
  const fileResult = response.data[index]
  if (!fileResult) return null

  const vectorResult = fileResult.vector_result
  // vector_result 为 null 表示文件未进行向量入库（auto_index=false 或内容为空）
  // 这种情况不算错误，只是未入库
  if (!vectorResult) return null

  if (vectorResult.status === 'error') {
    // 优先使用 error_type 映射的日文提示
    if (vectorResult.error_type && errorTypeMessages[vectorResult.error_type]) {
      return errorTypeMessages[vectorResult.error_type]
    }
    // 其次使用后端返回的具体错误信息
    if (vectorResult.error) {
      return vectorResult.error
    }
    return 'ファイルの処理に失敗しました。'
  }

  return null
}

type FileManagePageProps = {
  onUploadComplete?: () => void
}

// 待上传文件状态
interface PendingFile {
  uid: string
  name: string
  size: number
  file: File
  status: 'pending' | 'uploading' | 'error'
  errorMsg?: string
}

// 文件记录类型
interface FileRecord {
  key: string
  name: string
  type: string
  size: string
  uploadDate: string
}

// 文件类型配置
const fileTypeConfig: Record<string, { icon: ReactNode; label: string }> = {
  pdf: { icon: <FilePdfOutlined style={{ color: '#ef4444' }} />, label: 'PDF' },
  docx: { icon: <FileWordOutlined style={{ color: '#3b82f6' }} />, label: 'Word' },
  xlsx: { icon: <FileExcelOutlined style={{ color: '#22c55e' }} />, label: 'Excel' },
  txt: { icon: <FileTextOutlined style={{ color: '#64748b' }} />, label: 'テキスト' },
  img: { icon: <FileImageOutlined style={{ color: '#8b5cf6' }} />, label: '画像' },
}

// 筛选选项
const filterOptions = ['すべて', 'PDF', 'Word', 'Excel', 'テキスト', '画像']

// 待上传文件状态配置
const pendingStatusConfig: Record<string, { color: string; label: string }> = {
  pending: { color: 'default', label: '待アップロード' },
  uploading: { color: 'processing', label: '処理中' },
  error: { color: 'error', label: 'エラー' },
}

// 文件后缀 → 图标映射
function getFileIcon(fileName: string): ReactNode {
  const ext = fileName.split('.').pop()?.toLowerCase()
  switch (ext) {
    case 'pdf':
      return <FilePdfOutlined style={{ color: '#ef4444', fontSize: 16 }} />
    case 'docx':
    case 'doc':
      return <FileWordOutlined style={{ color: '#3b82f6', fontSize: 16 }} />
    case 'xlsx':
    case 'xls':
      return <FileExcelOutlined style={{ color: '#22c55e', fontSize: 16 }} />
    case 'txt':
      return <FileTextOutlined style={{ color: '#64748b', fontSize: 16 }} />
    case 'png':
    case 'jpg':
    case 'jpeg':
    case 'gif':
    case 'bmp':
    case 'webp':
      return <FileImageOutlined style={{ color: '#8b5cf6', fontSize: 16 }} />
    default:
      return <FileTextOutlined style={{ color: '#94a3b8', fontSize: 16 }} />
  }
}

// 字节格式化
function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
}

export default function FileManagePage({ onUploadComplete }: FileManagePageProps) {
  const [activeTab, setActiveTab] = useState('upload')
  const [searchText, setSearchText] = useState('')
  const [typeFilter, setTypeFilter] = useState('すべて')
  const [selectedRowKeys, setSelectedRowKeys] = useState<React.Key[]>([])

  const [fileRecords, setFileRecords] = useState<FileRecord[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [fetchError, setFetchError] = useState<string | null>(null)

  // 上传相关状态
  const [pendingFiles, setPendingFiles] = useState<PendingFile[]>([])
  const [isUploading, setIsUploading] = useState(false)
  // 上传进度
  const [uploadProgress, setUploadProgress] = useState({ total: 0, processed: 0 })
  // 上传过程中被用户删除的文件 uid 集合
  const removedDuringUploadRef = useRef<Set<string>>(new Set())
  // 批量添加文件的防抖：收集 beforeUpload 的逐文件调用，一次性批量添加
  const pendingBatchRef = useRef<File[]>([])

  // 预览 Modal 状态
  const [previewVisible, setPreviewVisible] = useState(false)
  const [previewFilename, setPreviewFilename] = useState('')

  // 更新 Modal 状态
  const [updateModalVisible, setUpdateModalVisible] = useState(false)
  const [updatingRecord, setUpdatingRecord] = useState<FileRecord | null>(null)
  const [updateLoading, setUpdateLoading] = useState(false)
  // 选中的新文件（由 Upload 的 onChange 设置）
  const [selectedNewFile, setSelectedNewFile] = useState<File | null>(null)
  // 更新处理的错误信息
  const [updateError, setUpdateError] = useState<string | null>(null)
  const batchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // 从后端获取文档列表并映射为页面展示格式
  const fetchDocumentList = useCallback(async () => {
    setIsLoading(true)
    setFetchError(null)
    try {
      const res = await fetchDocuments()
      if (res.code === 0 && Array.isArray(res.data)) {
        const records: FileRecord[] = res.data.map((doc) => ({
          key: doc.source_file,
          name: doc.source_file,
          type: doc.doc_type || '',
          size: formatBytes(doc.file_size),
          uploadDate: (doc.created_at || '').replace(/-/g, '/').slice(0, 16),
        }))
        setFileRecords(records)
      } else {
        setFetchError(res.msg || 'データの取得に失敗しました')
      }
    } catch {
      setFetchError('サーバーとの通信に失敗しました')
    } finally {
      setIsLoading(false)
    }
  }, [])

  // 初次加载时获取文档列表
  useEffect(() => {
    fetchDocumentList()
  }, [fetchDocumentList])

  // 筛选后的数据
  const filteredRecords = useMemo(() => {
    return fileRecords.filter((record) => {
      const matchType = typeFilter === 'すべて' || fileTypeConfig[record.type]?.label === typeFilter
      const matchSearch = !searchText.trim() || record.name.toLowerCase().includes(searchText.trim().toLowerCase())
      return matchType && matchSearch
    })
  }, [fileRecords, typeFilter, searchText])

  const hasSelected = selectedRowKeys.length > 0

  // 批量删除
  const handleBatchDelete = async () => {
    const keys = selectedRowKeys.map(String)
    let successCount = 0
    let failCount = 0

    for (const key of keys) {
      try {
        await deleteDocument(key)
        successCount++
      } catch {
        failCount++
      }
    }

    if (successCount > 0) {
      message.success(`${successCount} 件のファイルを削除しました。`)
      onUploadComplete?.()
      fetchDocumentList()
    }
    if (failCount > 0) {
      message.error(`${failCount} 件のファイルの削除に失敗しました。`)
    }
    setSelectedRowKeys([])
  }

  // 单文件预览：打开 DocumentPreviewModal 共享组件
  const handlePreview = (filename: string) => {
    setPreviewFilename(filename)
    setPreviewVisible(true)
  }

  // 单文件下载：通过 <a> 标签点击触发浏览器原生下载（流式写入磁盘，不占内存）
  const handleSingleDownload = (filename: string) => {
    const url = getDownloadUrl(filename)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  // 批量下载：请求后端打包 ZIP 后触发浏览器下载
  const handleBatchDownload = async () => {
    const keys = selectedRowKeys.map(String)
    if (keys.length === 0) return
    try {
      await downloadBatch(keys)
      message.success(`${keys.length} 件のファイルをダウンロードしました。`)
      setSelectedRowKeys([])
    } catch {
      message.error('一括ダウンロードに失敗しました。')
    }
  }

  // 删除文件（真实删除：调用 DELETE API）
  const handleDelete = async (key: string) => {
    try {
      await deleteDocument(key)
      message.success('ファイルを削除しました。')
      onUploadComplete?.()
      fetchDocumentList()
    } catch {
      message.error('ファイルの削除に失敗しました。')
    }
  }

  // 打开更新 Modal：记录被更新的文件信息
  const handleOpenUpdate = (record: FileRecord) => {
    setUpdatingRecord(record)
    setSelectedNewFile(null)
    setUpdateError(null)
    setUpdateModalVisible(true)
  }

  // 选择新文件时的校验
  const handleNewFileSelected = (file: File) => {
    setSelectedNewFile(file)
    setUpdateError(null)
  }

  // 校验逻辑：文件名（不含扩展名）是否一致 + 扩展名是否一致
  const getUpdateValidation = (): { nameOk: boolean; extSame: boolean } | null => {
    if (!updatingRecord || !selectedNewFile) return null
    // 比较去掉扩展名后的基础名（不区分大小写）
    const oldBase = updatingRecord.name.replace(/\.[^.]+$/, '').toLowerCase()
    const newBase = selectedNewFile.name.replace(/\.[^.]+$/, '').toLowerCase()
    const oldExt = (updatingRecord.name.split('.').pop() || '').toLowerCase()
    const newExt = (selectedNewFile.name.split('.').pop() || '').toLowerCase()
    return { nameOk: oldBase === newBase, extSame: oldExt === newExt }
  }

  // 执行更新
  const handleUpdateSubmit = async () => {
    if (!updatingRecord || !selectedNewFile) return

    // 再次确认校验
    const validation = getUpdateValidation()
    if (!validation) return
    if (!validation.nameOk) {
      setUpdateError(
        'ファイル名（拡張子を除く）が一致しません。' +
        '更新対象と同じファイルを選択してください。'
      )
      return
    }
    if (!validation.extSame) {
      // 扩展名不同时：二次确认
      const confirmed = await new Promise<boolean>((resolve) => {
        Modal.confirm({
          title: 'ファイル形式が異なります',
          icon: <ExclamationCircleOutlined />,
          content: (
            <div>
              <p>選択されたファイルの形式（.{selectedNewFile.name.split('.').pop()}）が
              元ファイルの形式（.{updatingRecord.name.split('.').pop()}）と異なります。</p>
              <p style={{ color: '#d97706', fontWeight: 500 }}>
                旧ファイルは削除され、新規エントリーとして取り込まれます。続行しますか？
              </p>
            </div>
          ),
          okText: '続行',
          cancelText: 'キャンセル',
          okButtonProps: { danger: true },
          onOk: () => resolve(true),
          onCancel: () => resolve(false),
        })
      })
      if (!confirmed) return
    }

    setUpdateLoading(true)
    setUpdateError(null)

    try {
      const res = await updateDocument(updatingRecord.name, selectedNewFile)
      if (res.code === 0) {
        message.success(
          `「${updatingRecord.name}」を更新しました。`
        )
        setUpdateModalVisible(false)
        onUploadComplete?.()
        fetchDocumentList()
      } else {
        setUpdateError(res.msg || '更新に失敗しました。')
      }
    } catch (err: unknown) {
      const detail =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setUpdateError(detail || '更新処理中にエラーが発生しました。')
    } finally {
      setUpdateLoading(false)
    }
  }

  // 关闭更新 Modal（重置状态）
  const handleCloseUpdateModal = () => {
    setUpdateModalVisible(false)
    setUpdatingRecord(null)
    setSelectedNewFile(null)
    setUpdateError(null)
  }

  // ======================== 上传相关逻辑 ========================

  const MAX_FILE_SIZE = 50 * 1024 * 1024
  const ALLOWED_EXTENSIONS = new Set([
    'pdf',
    'docx', 'doc',
    'xlsx', 'xls',
    'txt',
    'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp',
  ])

// 添加待上传文件
  const handleAddFiles = (files: File[]) => {
    if (hasUploadingFile) return
    setPendingFiles((prev) => {
      const newFiles: PendingFile[] = []
      let skippedCount = 0
      let oversizedCount = 0
      let invalidTypeCount = 0

      for (const file of files) {
        const ext = file.name.split('.').pop()?.toLowerCase()
        if (!ext || !ALLOWED_EXTENSIONS.has(ext)) {
          invalidTypeCount++
          continue
        }
        if (file.size > MAX_FILE_SIZE) {
          oversizedCount++
          continue
        }
        const exists = prev.some(
          (pf) => pf.name === file.name && pf.size === file.size
        )
        if (exists) {
          skippedCount++
        } else {
          newFiles.push({
            uid: crypto.randomUUID(),
            name: file.name,
            size: file.size,
            file,
            status: 'pending' as const,
          })
        }
      }

      if (newFiles.length > 0) {
        message.success(`${newFiles.length}件のファイルを追加しました。`)
      }
      if (skippedCount > 0) {
        message.warning(`${skippedCount}件のファイルは既に追加済みです。`)
      }
      if (invalidTypeCount > 0) {
        message.warning(`${invalidTypeCount}件のファイルは非対応形式のためスキップされました。`)
      }
      if (oversizedCount > 0) {
        message.warning(`${oversizedCount}件のファイルが50MBを超えているためスキップされました。`)
      }

      return [...prev, ...newFiles]
    })
  }

  // 删除单个待上传文件
  const handleRemovePending = (uid: string) => {
    if (isUploading) {
      removedDuringUploadRef.current.add(uid)
      setUploadProgress((prev) => ({ ...prev, total: prev.total - 1 }))
    }
    setPendingFiles((prev) => prev.filter((f) => f.uid !== uid))
  }

  // 清空全部待上传文件
  const handleClearAll = () => {
    if (hasUploadingFile) return
    setPendingFiles([])
    message.success('すべてのファイルをクリアしました。')
  }

  // 计算待上传文件数量
  const pendingCount = pendingFiles.filter(
    (f) => f.status === 'pending' || f.status === 'error'
  ).length

  // 是否有文件正在上传中（包括批量上传和单独重试）
  const hasUploadingFile = pendingFiles.some((f) => f.status === 'uploading')

  // 上传按钮文案
  const getUploadButtonText = () => {
    if (isUploading) return '処理中...'
    return 'アップロードを開始'
  }

  // 逐文件上传
  const handleStartUpload = async () => {
    const toUpload = pendingFiles.filter(
      (f) => f.status === 'pending' || f.status === 'error'
    )
    if (toUpload.length === 0) {
      message.warning('アップロードするファイルがありません。')
      return
    }

    setIsUploading(true)
    setUploadProgress({ total: toUpload.length, processed: 0 })
    removedDuringUploadRef.current = new Set()

    let successCount = 0
    let failCount = 0

    for (const pf of toUpload) {
      // 跳过已被用户删除的文件
      if (removedDuringUploadRef.current.has(pf.uid)) {
        continue
      }
      // 更新为上传中状态
      setPendingFiles((prev) =>
        prev.map((f) =>
          f.uid === pf.uid ? { ...f, status: 'uploading' as const } : f
        )
      )

      try {
        // 检查文件是否已存在于向量数据库
        const checkRes = await checkFileExists(pf.name)
        if (checkRes.data.exists) {
          // 文件已存在，弹出确认对话框
          const confirmed = await new Promise<boolean>((resolve) => {
            Modal.confirm({
              title: 'ファイルが既に存在します',
              content: `「${pf.name}」は既にデータベースに存在します。上書きしますか？`,
              okText: '上書き',
              cancelText: 'キャンセル',
              onOk: () => resolve(true),
              onCancel: () => resolve(false),
            })
          })
          if (!confirmed) {
            // 用户取消，移除该文件
            setPendingFiles((prev) => prev.filter((f) => f.uid !== pf.uid))
            setUploadProgress((prev) => ({ ...prev, processed: prev.processed + 1 }))
            continue
          }
        }

        const response = await uploadDocuments([pf.file])
        
        // 检查后端返回的每个文件的处理结果
        if (response.code === 0 && response.data && response.data.length > 0) {
          const errorMsg = extractErrorMessage(response, 0)
          
          if (errorMsg) {
            // 业务错误：显示具体错误信息，保留文件在列表中
            setPendingFiles((prev) =>
              prev.map((f) =>
                f.uid === pf.uid
                  ? { ...f, status: 'error' as const, errorMsg }
                  : f
              )
            )
            failCount++
          } else {
            // 成功：移除文件
            setPendingFiles((prev) => prev.filter((f) => f.uid !== pf.uid))
            successCount++
          }
        } else {
          // 响应格式异常
          setPendingFiles((prev) =>
            prev.map((f) =>
              f.uid === pf.uid
                ? { ...f, status: 'error' as const, errorMsg: 'サーバーからの応答が異常です。' }
                : f
            )
          )
          failCount++
        }
      } catch {
        // 网络异常或其他异常
        setPendingFiles((prev) =>
          prev.map((f) =>
            f.uid === pf.uid
              ? { ...f, status: 'error' as const, errorMsg: 'アップロードに失敗しました。ネットワーク接続を確認してください。' }
              : f
          )
        )
        failCount++
      }

      setUploadProgress((prev) => ({ ...prev, processed: prev.processed + 1 }))
    }

    setIsUploading(false)

    if (successCount > 0) {
      message.success(`${successCount}件のファイルをアップロードしました。`)
      onUploadComplete?.()
      fetchDocumentList()
    }
    if (failCount > 0) {
      message.error(`${failCount}件のファイルのアップロードに失敗しました。`)
    }
  }

  // 重试单个文件
  const handleRetry = async (pf: PendingFile) => {
    if (hasUploadingFile) return
    setPendingFiles((prev) =>
      prev.map((f) =>
        f.uid === pf.uid ? { ...f, status: 'uploading' as const } : f
      )
    )

    try {
      // 检查文件是否已存在于向量数据库
      const checkRes = await checkFileExists(pf.name)
      if (checkRes.data.exists) {
        const confirmed = await new Promise<boolean>((resolve) => {
          Modal.confirm({
            title: 'ファイルが既に存在します',
            content: `「${pf.name}」は既にデータベースに存在します。上書きしますか？`,
            okText: '上書き',
            cancelText: 'キャンセル',
            onOk: () => resolve(true),
            onCancel: () => resolve(false),
          })
        })
        if (!confirmed) {
          setPendingFiles((prev) => prev.filter((f) => f.uid !== pf.uid))
          return
        }
      }

      const response = await uploadDocuments([pf.file])
      
      // 检查后端返回的每个文件的处理结果
      if (response.code === 0 && response.data && response.data.length > 0) {
        const errorMsg = extractErrorMessage(response, 0)
        
        if (errorMsg) {
          // 业务错误：显示具体错误信息
          setPendingFiles((prev) =>
            prev.map((f) =>
              f.uid === pf.uid
                ? { ...f, status: 'error' as const, errorMsg }
                : f
            )
          )
          message.error(`「${pf.name}」の処理に失敗しました。`)
        } else {
          // 成功：移除文件
          setPendingFiles((prev) => prev.filter((f) => f.uid !== pf.uid))
          message.success(`「${pf.name}」をアップロードしました。`)
          onUploadComplete?.()
          fetchDocumentList()
        }
      } else {
        // 响应格式异常
        setPendingFiles((prev) =>
          prev.map((f) =>
            f.uid === pf.uid
              ? { ...f, status: 'error' as const, errorMsg: 'サーバーからの応答が異常です。' }
              : f
          )
        )
        message.error(`「${pf.name}」の処理に失敗しました。`)
      }
    } catch {
      // 网络异常或其他异常
      setPendingFiles((prev) =>
        prev.map((f) =>
          f.uid === pf.uid
            ? { ...f, status: 'error' as const, errorMsg: 'アップロードに失敗しました。ネットワーク接続を確認してください。' }
            : f
        )
      )
      message.error(`「${pf.name}」のアップロードに失敗しました。`)
    }
  }

  // ======================== 表格列定义 ========================

  const columns: ColumnsType<FileRecord> = [
    {
      title: 'ファイル名',
      dataIndex: 'name',
      key: 'name',
      width: '35%',
      render: (name: string) => (
        <Tooltip title={name} styles={{ root: { maxWidth: 'none' } }}>
          <span style={{ fontWeight: 500, color: '#1e293b' }}>{name}</span>
        </Tooltip>
      ),
      ellipsis: true,
    },
    {
      title: '種別',
      dataIndex: 'type',
      key: 'type',
      width: '12%',
      render: (type: string) => {
        const config = fileTypeConfig[type]
        return config ? (
          <Space size={6}>
            {config.icon}
            <span style={{ fontSize: 12, color: '#64748b' }}>{config.label}</span>
          </Space>
        ) : (
          '-'
        )
      },
    },
    {
      title: 'サイズ',
      dataIndex: 'size',
      key: 'size',
      width: '12%',
      render: (size: string) => <span style={{ color: '#64748b' }}>{size}</span>,
    },
    {
      title: 'アップロード日',
      dataIndex: 'uploadDate',
      key: 'uploadDate',
      width: '18%',
      render: (date: string) => <span style={{ color: '#64748b' }}>{date}</span>,
    },
    {
      title: '操作',
      key: 'action',
      width: '20%',
      align: 'left' as const,
      render: (_: unknown, record: FileRecord) => {
        return (
          <Space size={4}>
            <Tooltip title="プレビュー">
              <Button
                type="default"
                size="small"
                icon={<EyeOutlined />}
                onClick={() => handlePreview(record.name)}
              />
            </Tooltip>
            <Tooltip title="ダウンロード">
              <Button
                type="default"
                size="small"
                icon={<DownloadOutlined />}
                onClick={() => handleSingleDownload(record.name)}
              />
            </Tooltip>
            <Tooltip title="更新">
              <Button
                type="default"
                size="small"
                icon={<SyncOutlined />}
                onClick={() => handleOpenUpdate(record)}
              />
            </Tooltip>
            <Popconfirm
              title="ファイルを削除"
              description={`「${record.name}」を削除しますか？削除すると検索結果にも表示されなくなります。`}
              onConfirm={() => handleDelete(record.key)}
              okText="削除"
              cancelText="キャンセル"
              okButtonProps={{ danger: true }}
            >
              <Tooltip title="削除">
                <Button
                  danger
                  size="small"
                  icon={<DeleteOutlined />}
                />
              </Tooltip>
            </Popconfirm>
          </Space>
        )
      },
    },
  ]

  // ======================== 上传组件配置 ========================

  const uploadProps = {
    multiple: true,
    accept: '.pdf,.docx,.xlsx,.txt,.jpg,.jpeg,.png,.gif,.bmp,.webp',
    showUploadList: false,
    disabled: hasUploadingFile,
    beforeUpload: (file: File) => {
      // 收集到批次数组，防抖后批量添加（避免逐文件弹多次消息）
      pendingBatchRef.current.push(file)
      if (batchTimerRef.current !== null) {
        clearTimeout(batchTimerRef.current)
      }
      batchTimerRef.current = setTimeout(() => {
        const batch = pendingBatchRef.current.splice(0)
        if (batch.length > 0) {
          handleAddFiles(batch)
        }
        batchTimerRef.current = null
      }, 60)
      return false
    },
  }

  // ======================== 新上传 Tab ========================

  const uploadTab = (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      {/* 拖拽上传区 */}
      <Upload.Dragger {...uploadProps} style={{ padding: 30 }} className="upload-dragger-area">
        <p className="ant-upload-drag-icon">
          <InboxOutlined />
        </p>
        <p className="ant-upload-text">ここにファイルをドラッグ＆ドロップ</p>
        <p className="ant-upload-hint">
          またはクリックしてファイルを選択してください。
        </p>
        <div className="upload-formats">
          <span>対応ファイル形式：</span>
          PDF / Word(DOCX/DOC) / Excel(XLSX/XLS/XLSM) / テキスト(TXT) / 画像(PNG/JPG/GIF)
          <br />
          ※ 1ファイル最大50MBまで。暗号化・破損ファイルは取込不可
          <br />
          ※ .doc / .xls旧形式ファイルを利用するにはLibreOfficeの事前インストールが必要です。
          <br />
          ※ マクロ付きExcelファイル（.xlsm）は保存済みの数値のみ読み取り、マクロ実行による再計算には対応しておりません。
          <br />
        </div>
      </Upload.Dragger>

      {/* 上传按钮 */}
      <Button
        type="primary"
        icon={<UploadOutlined />}
        size="large"
        onClick={handleStartUpload}
        loading={isUploading}
        disabled={pendingCount === 0 || hasUploadingFile}
        block
      >
        {getUploadButtonText()}
      </Button>

      {/* 待上传文件列表 */}
      <div className="pending-files">
        <div className="pending-header">
          <span className="pending-title">
            アップロード予定ファイル
            {pendingFiles.length > 0 && (
              <span className="pending-count">{pendingFiles.length}件</span>
            )}
          </span>
          {pendingFiles.length > 0 && !hasUploadingFile && (
            <Button
              type="link"
              size="small"
              danger
              icon={<ClearOutlined />}
              onClick={handleClearAll}
              className="clear-all-btn"
            >
              すべてクリア
            </Button>
          )}
        </div>

        {pendingFiles.length === 0 ? (
          <div className="empty-files">まだファイルがありません。</div>
        ) : (
          <div className="file-list">
            {pendingFiles.map((pf) => {
              const statusCfg = pendingStatusConfig[pf.status]
              const isUploadingThis = pf.status === 'uploading'

              return (
                <div key={pf.uid} className="file-item">
                  <div className="file-info">
                    <span className="file-icon">{getFileIcon(pf.name)}</span>
                    <span className="file-name" title={pf.name}>
                      {pf.name}
                    </span>
                    <span className="file-size">{formatBytes(pf.size)}</span>
                  </div>
                  <div className="file-right">
                    <Tag color={statusCfg.color}>{statusCfg.label}</Tag>
                    {/* 错误文件：显示重试按钮（上传中隐藏，防止并发） */}
                    {pf.status === 'error' && !hasUploadingFile && (
                      <Tooltip title="再試行">
                        <Button
                          type="text"
                          size="small"
                          icon={<ReloadOutlined />}
                          onClick={() => handleRetry(pf)}
                          className="file-action-btn"
                        />
                      </Tooltip>
                    )}
                    {/* 非上传中：显示删除按钮 */}
                    {!isUploadingThis && (
                      <Tooltip title="削除">
                        <Button
                          type="text"
                          size="small"
                          icon={<CloseOutlined />}
                          onClick={() => handleRemovePending(pf.uid)}
                          className="file-action-btn file-remove-btn"
                        />
                      </Tooltip>
                    )}
                  </div>
                  {/* 错误信息展示 */}
                  {pf.errorMsg && (
                    <div className="file-error-message">
                      {pf.errorMsg}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        )}

        {/* 进度条 */}
        {isUploading && uploadProgress.total > 0 && (
          <div className="progress-footer">
            <span>
              {uploadProgress.processed} / {uploadProgress.total} 件 処理中...
            </span>
          </div>
        )}
      </div>
    </Space>
  )

  // ======================== 已注册文档列表 Tab ========================

  const listTab = (
    <Space direction="vertical" size="middle" style={{ width: '100%' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
        <Segmented
          options={filterOptions}
          value={typeFilter}
          onChange={(value) => {
            setTypeFilter(value as string)
            setSelectedRowKeys([])
          }}
        />
        <Input
          placeholder="ファイル名で検索..."
          prefix={<SearchOutlined style={{ color: '#94a3b8' }} />}
          allowClear
          style={{ width: 280 }}
          value={searchText}
          onChange={(e) => setSearchText(e.target.value)}
        />
      </div>
      {hasSelected && (
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '8px 16px',
          background: '#e6f4ff',
          border: '1px solid #91caff',
          borderRadius: 6,
        }}>
          <Typography.Text strong style={{ color: '#1677ff' }}>
            {selectedRowKeys.length} 件選択中
          </Typography.Text>
          <Space>
            <Button icon={<DownloadOutlined />} onClick={handleBatchDownload}>
              一括ダウンロード
            </Button>
            <Popconfirm
              title={`${selectedRowKeys.length} 件のファイルを削除しますか？`}
              onConfirm={handleBatchDelete}
              okText="削除"
              cancelText="キャンセル"
              okButtonProps={{ danger: true }}
            >
              <Button danger icon={<DeleteOutlined />}>一括削除</Button>
            </Popconfirm>
          </Space>
        </div>
      )}
      {fetchError && (
        <Alert
          message="データ取得エラー"
          description={fetchError}
          type="error"
          showIcon
          closable
          onClose={() => setFetchError(null)}
          action={
            <Button size="small" onClick={fetchDocumentList} type="primary" ghost>
              再試行
            </Button>
          }
        />
      )}
      <Table
        columns={columns}
        dataSource={filteredRecords}
        loading={isLoading}
        rowSelection={{
          selectedRowKeys,
          onChange: (keys) => setSelectedRowKeys(keys),
        }}
        pagination={{
          total: filteredRecords.length,
          showTotal: (total) => `全 ${total} 件`,
          showSizeChanger: true,
          showQuickJumper: true,
          pageSizeOptions: ['10', '20', '50'],
          defaultPageSize: 10,
        }}
        locale={{ emptyText: '該当するファイルがありません。' }}
        size="middle"
        bordered
      />
    </Space>
  )

  const tabItems = [
    { key: 'upload', label: '新規アップロード', children: uploadTab },
    { key: 'list', label: '登録済文書一覧', children: listTab },
  ]

  return (
    <Card
      title="電子文書管理"
      className="page-card manage-page"
    >
      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        items={tabItems}
        size="large"
      />

      {/* 预览 Modal：从文件列表的 EyeOutlined 按钮打开 */}
      <DocumentPreviewModal
        filename={previewFilename}
        visible={previewVisible}
        onClose={() => setPreviewVisible(false)}
      />

      {/* ======================== 更新 Modal ======================== */}
      <Modal
        title="文書の更新"
        open={updateModalVisible}
        onCancel={handleCloseUpdateModal}
        width={560}
        footer={
          <Space>
            <Button onClick={handleCloseUpdateModal} disabled={updateLoading}>
              キャンセル
            </Button>
            <Button
              type="primary"
              loading={updateLoading}
              disabled={!selectedNewFile || (() => {
                const v = getUpdateValidation()
                return v === null || !v.nameOk
              })()}
              onClick={handleUpdateSubmit}
            >
              更新を実行
            </Button>
          </Space>
        }
        destroyOnHidden
      >
        {updatingRecord && (
          <Space direction="vertical" size="middle" style={{ width: '100%' }}>
            {/* ── 注意事项（固定显示） ── */}
            <Alert
              message="更新時の注意事項"
              description={
                <ul style={{ margin: 0, paddingLeft: 18 }}>
                  <li>更新対象と同名・同形式のファイルを選択してください。</li>
                  <li>形式が異なる場合、旧ファイルは削除され新規エントリーとして取り込まれます。</li>
                  <li>更新を実行すると、既存のベクトルデータは削除され、新しい内容で再取り込みされます。</li>
                  <li>処理中にエラーが発生した場合、元のデータは保持されます。</li>
                </ul>
              }
              type="info"
              showIcon
            />

            {/* ── 旧文件信息 ── */}
            <div className="update-old-file-info">
              <div className="update-info-label">更新対象ファイル</div>
              <div className="update-info-row">
                <span className="update-info-icon">{getFileIcon(updatingRecord.name)}</span>
                <span className="update-info-name">{updatingRecord.name}</span>
                <Tag>{fileTypeConfig[updatingRecord.type]?.label || updatingRecord.type}</Tag>
                <span className="update-info-meta">{updatingRecord.size}</span>
                <span className="update-info-meta">{updatingRecord.uploadDate}</span>
              </div>
            </div>

            {/* ── 新文件选择 ── */}
            <div className="update-new-file-area">
              <div className="update-info-label">新しいファイルを選択</div>
              <Upload
                maxCount={1}
                accept={Object.keys(ALLOWED_EXTENSIONS).length > 0
                  ? '.' + Array.from(new Set([
                      'pdf','docx','doc','xlsx','xls','xlsm',
                      'txt','csv','json','html','htm','md','xml',
                      'jpg','jpeg','png','gif','bmp','webp',
                    ])).join(',.')
                  : undefined
                }
                showUploadList={false}
                beforeUpload={(file) => {
                  // 大小检查
                  if (file.size > MAX_FILE_SIZE) {
                    message.warning(`ファイルサイズが50MBを超えています: ${file.name}`)
                    return Upload.LIST_IGNORE
                  }
                  // 扩展名检查
                  const ext = file.name.split('.').pop()?.toLowerCase()
                  if (!ext || !ALLOWED_EXTENSIONS.has(ext)) {
                    message.warning(`非対応のファイル形式です: ${file.name}`)
                    return Upload.LIST_IGNORE
                  }
                  handleNewFileSelected(file)
                  return false  // 不自动上传
                }}
                fileList={
                  selectedNewFile
                    ? [{
                        uid: '-1',
                        name: selectedNewFile.name,
                        status: 'done' as const,
                      }]
                    : []
                }
                onRemove={() => {
                  setSelectedNewFile(null)
                  setUpdateError(null)
                }}
              >
                <Button icon={<UploadOutlined />}>
                  {selectedNewFile ? 'ファイルを変更' : 'ファイルを選択'}
                </Button>
              </Upload>
            </div>

            {/* ── 校验结果 ── */}
            {selectedNewFile && (() => {
              const validation = getUpdateValidation()
              if (!validation) return null

              return (
                <Space direction="vertical" size={8} style={{ width: '100%' }}>
                  {/* 文件名检查 */}
                  {validation.nameOk ? (
                    <div className="update-validation-ok">
                      <CheckCircleOutlined /> ファイル名が一致しています
                    </div>
                  ) : (
                    <div className="update-validation-ng">
                      <WarningOutlined />
                      {' '}ファイル名が一致しません。
                      更新対象「{updatingRecord.name.replace(/\.[^.]+$/, '')}」と同じ名前のファイルを選択してください。
                    </div>
                  )}

                  {/* 扩展名检查 */}
                  {validation.nameOk && !validation.extSame && (
                    <Alert
                      message="ファイル形式が異なります"
                      description={
                        '選択されたファイルの形式が元ファイルと異なります。' +
                        '更新を実行すると、旧ファイルは削除され新規エントリーとして取り込まれます。'
                      }
                      type="warning"
                      showIcon
                    />
                  )}
                </Space>
              )
            })()}

            {/* ── 错误显示 ── */}
            {updateError && (
              <Alert
                message="更新エラー"
                description={updateError}
                type="error"
                showIcon
                closable
                onClose={() => setUpdateError(null)}
              />
            )}
          </Space>
        )}
      </Modal>
    </Card>
  )
}