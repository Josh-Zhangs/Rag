	ユーザー導入手順																																									
																																										
		前提条件																																								
			Windows 10/11 64bit																																							
			メモリ 32GB 以上推奨（LLM モデルが大きいため）																																							
			ディスク空き容量 20GB 以上																																							
																																										
		Step 1：Ollama のインストール																																								
																																										
			https://ollama.com/download/windows													 から OllamaSetup.exe をダウンロード																										
																																										
			ダウンロードしたインストーラーをダブルクリックして実行																																							
																																										
			画面の指示に従ってインストール（すべてデフォルトで OK）																																							
																																										
			インストール完了後、タスクバー右側に Ollama アイコン（羊のマーク🐑）が表示されることを確認																																							
																																										
		Step 2：モデルのダウンロード																																								
																																										
			PowerShell または コマンドプロンプト を開き、以下の 3 つのコマンドを順番に実行する																																							
																																										
				# ① LLM（対話用メイン）, 約 12GB																																						
				ollama pull fuukeidaisuki/gpt-oss-swallow-v0.1:20b-rl																																						
																																										
				# ② 会話判定用モデル, 約 4.4GB																																						
				ollama pull yuiseki/rakuten-ai:7b-instruct																																						
																																										
				# ③ Embedding モデル（ベクトル検索用）, 約 400MB																																						
				ollama pull jeffh/intfloat-multilingual-e5-small:q8_0																																						
																																										
			ダウンロード完了の確認：																																							
				ollama list																																						
																																										
			以下の 3 つが表示されれば OK：																																							
				yuiseki/rakuten-ai:7b-instruct        3cb73ce65eea  4.5GB
				jeffh/intfloat-multilingual-e5-small:q8_0        ca64a5d489ae  1.3GB
				fukeidaisuki/gpt-oss-swallow-v0.1:20b-rl        8061246c0e1a  12GB
																																																																																																																																																													
		Step 3：LibreOffice のインストール（必要に応じて）																																								
																																										
			古い Office 形式（.doc / .xls）をアップロードする場合のみ必要です。																																							
																																										
			LibreOffice_25.8.7_Win_x86-64.msiをダブルクリックして実行																																							
																																										
			画面の指示に従ってインストール（すべてデフォルトで OK）																																							
																																										
			インストール先は C:\Program Files\LibreOffice\（自動検出されるので変更不要）																																							
																																										
		Step 4：アプリの起動																																								
																																										
			Ollama が起動していることを確認（タスクバー右側に🐑アイコン）																																							
																																										
			machiariRAG.exe をダブルクリック																																							
																																										
			コマンドプロンプトの黒い画面が表示され、数秒後にブラウザが自動で開く																																							
																																										
			画面が表示されたら使用開始																																							
																																										
		終了方法																																								
			ブラウザを閉じる																																							
			コマンドプロンプトの黒い画面で Ctrl + C を押す（またはそのまま閉じる）																																							
<img width="1072" height="1251" alt="image" src="https://github.com/user-attachments/assets/18a716b7-39da-451a-9d96-6ec9ad354f37" />
