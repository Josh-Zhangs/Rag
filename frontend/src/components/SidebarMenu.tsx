import { useState, useEffect, useRef } from 'react'
import type { PageKey } from '../types'
import { fetchDocumentStats } from '../api/client'
import {
  MessageOutlined,
  FileTextOutlined,
  SettingOutlined,
  FilePdfOutlined,
  FileWordOutlined,
  FileExcelOutlined,
  FileImageOutlined,
  ClockCircleOutlined,
} from '@ant-design/icons'

type SidebarMenuProps = {
  activePage: PageKey
  onChange: (page: PageKey) => void
  refreshTrigger?: number
}

const menuItems: Array<{ key: PageKey; icon: React.ReactNode; label: string }> = [
  { key: 'aiChat', icon: <MessageOutlined />, label: 'AIアシスタント' },
  { key: 'fileUpload', icon: <FileTextOutlined />, label: '文書管理' },
  { key: 'settings', icon: <SettingOutlined />, label: '設定' }
]

const docTypeConfig: Array<{ key: string; icon: React.ReactNode; label: string }> = [
  { key: 'pdf', icon: <FilePdfOutlined style={{ color: '#ef4444' }}/>, label: 'PDF' },
  { key: 'docx', icon: <FileWordOutlined style={{ color: '#3b82f6' }}/>, label: 'Word' },
  { key: 'xlsx', icon: <FileExcelOutlined style={{ color: '#22c55e' }}/>, label: 'Excel' },
  { key: 'txt', icon: <FileTextOutlined style={{ color: '#64748b' }}/>, label: 'テキスト' },
  { key: 'img', icon: <FileImageOutlined style={{ color: '#8b5cf6' }}/>, label: '画像' },
]

// 将后端返回的时间戳拆分为日期和时分秒两部分
function splitTimestamp(ts: string): { date: string; time: string } {
  const parts = ts.split(' ')
  return {
    date: parts[0] || '',
    time: parts.slice(1).join(' ') || '',
  }
}

export default function SidebarMenu({ activePage, onChange, refreshTrigger }: SidebarMenuProps) {
  // 文档统计数据：按类型计数
  const [stats, setStats] = useState<Record<string, number>>({})
  // 最后文档更新日期
  const [lastUpdated, setLastUpdated] = useState('')

  // 防止组件卸载后仍然 setState
  const mountedRef = useRef(true)
  // 重试定时器引用，用于在 refreshTrigger 变化时取消旧的重试
  const retryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // 组件挂载 / refreshTrigger 变化时拉取文档统计
  // 失败后指数退避重试，不设上限，确保后端启动后必然拿到数据
  useEffect(() => {
    mountedRef.current = true

    // 取消上一次还在等待的重试定时器
    if (retryTimerRef.current) {
      clearTimeout(retryTimerRef.current)
      retryTimerRef.current = null
    }

    let attempt = 0

    const tryFetch = () => {
      fetchDocumentStats()
        .then((res) => {
          if (!mountedRef.current) return
          if (res.code === 0) {
            setStats(res.data.stats)
            setLastUpdated(res.data.last_updated)
          }
        })
        .catch((err) => {
          if (!mountedRef.current) return
          attempt++
          // 指数退避：1s → 2s → 4s → 8s → ... → 上限 30s
          const delay = Math.min(1000 * Math.pow(2, attempt - 1), 30_000)
          console.warn(
            `Failed to fetch document stats (attempt ${attempt}), retrying in ${delay / 1000}s...`,
            err
          )
          retryTimerRef.current = setTimeout(tryFetch, delay)
        })
    }

    tryFetch()

    return () => {
      mountedRef.current = false
      if (retryTimerRef.current) {
        clearTimeout(retryTimerRef.current)
      }
    }
  }, [refreshTrigger])

  const { date, time } = splitTimestamp(lastUpdated)

  return (
    <>
      <nav>
        <div className="menu-title">メインメニュー</div>
        {menuItems.map((item) => (
          <div
            key={item.key}
            className={`menu ${activePage === item.key ? 'active' : ''}`}
            onClick={() => onChange(item.key)}
          >
            <span className="menu-icon">{item.icon}</span>
            {item.label}
          </div>
        ))}
        <hr className="menu-divider" />
      </nav>
      <div className="version-box">
        <strong>Version 1.0</strong>
        <br />
        標準版
        {docTypeConfig.map((dt) => (
          <div className="stat" key={dt.key}>
            <span>{dt.icon} {dt.label}</span>
            <span>{stats[dt.key] ?? '0'}</span>
          </div>
        ))}
        <div className="stat">
          <span><ClockCircleOutlined /> 最終文書更新</span>
          <span>
            {lastUpdated ? (
              <>
                {date}
                <br />
                {time}
              </>
            ) : (
              '-'
            )}
          </span>
        </div>
      </div>
    </>
  )
}