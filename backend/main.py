import logging
import os
import sys
import warnings

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

warnings.filterwarnings(
    'ignore',
    message='Field.*has conflict with protected namespace.*model_',
)

from routers import ai_chat_router
from config.settings import settings
from routers import file_upload_router
from routers import document_stats_router
from routers import document_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

app = FastAPI(title='Machiari RAG Backend', debug=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

app.include_router(ai_chat_router.router, prefix=settings.api_prefix)
app.include_router(file_upload_router.router, prefix=settings.api_prefix)
app.include_router(document_stats_router.router, prefix=settings.api_prefix)
app.include_router(document_router.router, prefix=settings.api_prefix)

if not getattr(sys, 'frozen', False):
    @app.get('/')
    async def root():
        return {'message': 'Machiari RAG backend is running'}

@app.get('/api/status')
async def status():
    return {
        'backend': 'FastAPI',
        'llm': 'ollama (default) / llama.cpp fallback',
    }

if __name__ == '__main__':
    import threading
    import uvicorn

    frozen = getattr(sys, 'frozen', False)

    if frozen:
        # 托管前端静态文件
        dist_dir = os.path.join(sys._MEIPASS, 'frontend_dist')
        if os.path.isdir(dist_dir):
            app.mount('/', StaticFiles(directory=dist_dir, html=True), name='frontend')

        # 自动打开浏览器
        def open_browser():
            import time
            import webbrowser
            time.sleep(1.5)
            webbrowser.open('http://localhost:8000')

        threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run(app, host='0.0.0.0', port=8000)