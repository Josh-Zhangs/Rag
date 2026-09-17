# ユーザー導入手順
## 前提条件
- Windows 10/11 64bit
- メモリ：32GB 以上推奨（LLM モデルが大きいため）
- ディスク空き容量：20GB 以上

## Step 1：Ollama のインストール
1. [https://ollama.com/download/windows](https://ollama.com/download/windows) から `OllamaSetup.exe` をダウンロード
2. ダウンロードしたインストーラーをダブルクリックして実行
3. 画面の指示に従ってインストール（すべてデフォルトで OK）
4. インストール完了後、タスクバー右側に Ollama アイコン（羊のマーク🐑）が表示されることを確認

## Step 2：モデルのダウンロード
PowerShell または コマンドプロンプト を開き、以下の 3 つのコマンドを順番に実行する
```bash
# ① LLM（対話用メイン）, 約 12GB
ollama pull fukeidaisuki/gpt-oss-swallow-v0.1:20b-rl

# ② 会話判定用モデル, 約 4.4GB
ollama pull yuiseki/rakuten-ai:7b-instruct

# ③ Embedding モデル（ベクトル検索用）, 約 400MB
ollama pull jeffh/intfloat-multilingual-e5-small:q8_0
```

ダウンロード完了の確認：
```bash
ollama list
```

下記3つが表示されればインストール成功：
```
yuiseki/rakuten-ai:7b-instruct        3cb73ce65eea  4.5GB
jeffh/intfloat-multilingual-e5-small:q8_0        ca64a5d489ae  1.3GB
fukeidaisuki/gpt-oss-swallow-v0.1:20b-rl        8061246c0e1a  12GB
```

## Step 3：LibreOffice のインストール（必要に応じて）
> 古い Office 形式（`.doc` / `.xls`）をアップロードする場合のみ必要です。

1. `LibreOffice_25.8.7_Win_x86-64.msi`をダブルクリックして実行
2. 画面の指示に従ってインストール（すべてデフォルトで OK）
3. インストール先は `C:\Program Files\LibreOffice\`（自動検出されるので変更不要）

## Step 4：アプリの起動
1. Ollama が起動していることを確認（タスクバー右側に🐑アイコン）
2. `machiariRAG.exe` をダブルクリック
3. コマンドプロンプトの黒い画面が表示され、数秒後にブラウザが自動で開く
4. 画面が表示されたら使用開始

## 終了方法
1. ブラウザを閉じる
2. コマンドプロンプトの黒い画面で `Ctrl + C` を押す（またはそのままウィンドウを閉じる）
