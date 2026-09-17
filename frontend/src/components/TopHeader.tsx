import { FileSearchOutlined } from '@ant-design/icons'

export default function TopHeader() {
  return (
    <div className="topbar">
      <div className="logo">
        <span className="logo-icon"><FileSearchOutlined /></span>
        社内AI検索システム
      </div>
    </div>
  )
}