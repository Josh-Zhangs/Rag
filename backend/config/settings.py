import os
import sys
from pydantic_settings import BaseSettings

# 用户数据根目录：%LOCALAPPDATA%\machiariRAG\
# Windows 上对应 C:\Users\{用户名}\AppData\Local\machiariRAG\
_BASE_DATA_DIR = os.path.join(
    os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
    "machiariRAG"
)


class Settings(BaseSettings):
    api_prefix: str = '/api'
    ollama_base_url: str = 'http://localhost:11434'
    ollama_model: str = 'fuukeidaisuki/gpt-oss-swallow-v0.1:20b-rl'
    # 备选模型：
    # ollama_model: str = 'fuukeidaisuki/gpt-oss-swallow-v0.1:20b-rl'
    # ollama_model: str = 'cyberagent/calm2-7b-chat'

    # 查询预处理专用模型（指代消解 + 话题检测）
    ollama_preprocess_model: str = 'yuiseki/rakuten-ai:7b-instruct'

    # LanceDB 向量数据库存储路径
    lancedb_uri: str = os.path.join(_BASE_DATA_DIR, "lancedb")
    lancedb_table_name: str = 'documents'

    # 原始上传文件的永久存储路径（用于溯源和下载）
    file_storage_dir: str = os.path.join(_BASE_DATA_DIR, "data", "files")

    # Embedding 模型配置（通过 Ollama 本地调用，无需联网）
    embedding_model_name: str = 'jeffh/intfloat-multilingual-e5-small:q8_0'

    # ONNX Reranker 模型路径（用于混合检索后重排序，替代 Cross-Encoder）
    # 使用 ONNX Runtime 加载，无需 PyTorch，绕过 Windows SAC 的 shm.dll 拦截
    reranker_model_name: str = os.path.join(
        sys._MEIPASS if getattr(sys, 'frozen', False)
        else os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "bge-reranker-v2-m3-onnx"
    )

    # 混合检索配置
    # 粗排阶段：Dense 向量检索和 BM25 关键词检索各自取 top N 条候选
    coarse_top_k: int = 50
    # 精排阶段：Reranker 重排序后取 top N 条送入 LLM
    default_search_top_k: int = 8

    # 分片配置
    # chunk_size: PDF/DOCX → 800 | XLSX → 600 | TXT/Image → 1000
    chunk_size: int = 800
    chunk_overlap: int = 100

    class Config:
        env_file = '.env'
        protected_namespaces = ()


settings = Settings()