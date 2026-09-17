@echo off
cd /d d:\ZhangShuai\1.project\machiariRAG\backend
call .\.venv\Scripts\activate.bat
set OLLAMA_KEEP_ALIVE=-1
python -m uvicorn main:app --reload