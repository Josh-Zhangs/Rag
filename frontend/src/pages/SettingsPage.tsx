import { Card } from 'antd'
import '../styles/pages/SettingsPage.css'

export default function SettingsPage() {
  return (
    <Card title="システム設定" className="page-card settings-page">
      <div className="setting-section">
        <div className="setting-item" style={{ justifyContent: 'center', padding: '32px 20px' }}>
          <span style={{ color: '#94a3b8', fontSize: 14 }}>機能開発中</span>
        </div>
      </div>
    </Card>
  )
}