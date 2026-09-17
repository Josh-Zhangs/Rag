import React from 'react'
import ReactDOM from 'react-dom/client'
import { ConfigProvider } from 'antd'
import jaJP from 'antd/locale/ja_JP'
import App from './App'
import 'antd/dist/reset.css'
import './styles/styles.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ConfigProvider locale={jaJP}>
      <App />
    </ConfigProvider>
  </React.StrictMode>
)