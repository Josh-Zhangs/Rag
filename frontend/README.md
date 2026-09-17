# Machiari RAG フロントエンド

Windows 単機オフライン AI 文書検索システムのフロントエンドです。

## 技術スタック

| 分類 | 技術 |
|------|------|
| フレームワーク | React 18 + TypeScript |
| ビルドツール | Vite 5 |
| UI ライブラリ | Ant Design 5 + @ant-design/icons |
| HTTP クライアント | Axios |
| ローカルストレージ | IndexedDB（idb） |

## ディレクトリ構成

```
src/
├── api/
│   └── client.ts          # Axios インスタンス / API 接続設定
├── components/
│   ├── SidebarMenu.tsx     # 左サイドバー（ナビゲーション）
│   ├── TopHeader.tsx       # 上部ヘッダー
│   └── DocumentPreviewModal.tsx  # 文書プレビューモーダル
├── lib/
│   └── sessionStore.ts     # 会話セッション永続化（IndexedDB）
├── pages/
│   ├── AIChatPage.tsx      # ナレッジ検索（AI 対話）
│   ├── FileManagePage.tsx  # 文書アップロード・管理
│   └── SettingsPage.tsx    # システム設定
├── styles/
│   ├── styles.css          # グローバルスタイル
│   └── pages/              # ページ別スタイル
├── App.tsx                 # ルートコンポーネント
├── main.tsx                # エントリポイント
└── types.ts                # 型定義
```

## 画面構成

| ページ | 機能 |
|--------|------|
| ナレッジ検索 | AI 対話 UI、検索証拠表示、指代消解結果の可視化 |
| 文書管理 | 電子文書（PDF/DOCX/XLSX）・紙文書（OCR）のアップロード、一覧・削除 |
| システム設定 | モデル選択、バックエンド設定、データベース管理 |

## 開発の始め方

### 前提条件

- Node.js 18 以上
- バックエンド（FastAPI）が `localhost:8000` で起動中

### セットアップ

```bash
cd frontend
npm install
```

### 起動

```bash
npm run dev
```

ブラウザで `http://localhost:5173` を開きます。

### 本番ビルド

```bash
npm run build
```

ビルド成果物は `dist/` に出力されます。