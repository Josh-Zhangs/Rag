import os
import json
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any, TYPE_CHECKING
import pyarrow as pa
import lancedb
from config.settings import settings

# TYPE_CHECKING 块仅在静态类型检查时导入，运行时不会执行
if TYPE_CHECKING:
    from services.embedding_service import EmbeddingService

# 模块级日志记录器
logger = logging.getLogger(__name__)

# 从文件名提取文档类型
def _extract_doc_type(source_file: str) -> str:
    if not source_file:
        return ""
    ext = source_file.rsplit(".", 1)[-1].lower() if "." in source_file else ""
    if ext == "pdf":
        return "pdf"
    elif ext in ("docx", "doc"):
        return "docx"
    elif ext in ("xlsx", "xlsm", "xls"):
        return "xlsx"
    elif ext == "txt":
        return "txt"
    elif ext in ("png", "jpg", "jpeg", "bmp", "tiff", "tif", "gif", "webp"):
        return "img"
    return ""


class VectorDBService:
    """向量数据库服务：封装 LanceDB 的 CRUD 操作及向量相似度搜索

    职责：
    - 表管理：init_table
    - 数据写入：insert
    - 数据删除：delete_by_source
    - 存在检查：check_exists_by_source
    - 向量搜索：search（余弦相似度 + 阈值过滤）
    - 全量导出：get_all_chunks（供 BM25 索引构建）
    - 文档统计：get_document_stats
    - 文档列表：get_documents
    """

    def __init__(self, embedding_service: "EmbeddingService" = None):
        # LanceDB 数据库连接实例（延迟初始化，首次调用 _get_db 时连接）
        self._db: Optional[lancedb.DBConnection] = None
        # 嵌入服务引用（延迟导入，避免循环依赖）
        self._embedding_service: "EmbeddingService" = embedding_service

    # ================================================================
    # 内部工具方法
    # ================================================================

    # 获取嵌入服务实例
    def _get_embedding_service(self) -> "EmbeddingService":
        if self._embedding_service is None:
            from services.embedding_service import embedding_service
            self._embedding_service = embedding_service
        return self._embedding_service

    # 获取数据库连接
    def _get_db(self) -> lancedb.DBConnection:
        if self._db is None:
            self._init_db()
        return self._db

    # 初始化 LanceDB 连接
    def _init_db(self):
        # 确保数据目录存在（exist_ok=True 表示目录已存在时不报错）
        os.makedirs(settings.lancedb_uri, exist_ok=True)
        # 建立 LanceDB 连接（本地文件模式，数据存储在 lancedb_uri 目录）
        self._db = lancedb.connect(settings.lancedb_uri)
        logger.info(f"LanceDB 已连接: {settings.lancedb_uri}")

    # 构建 LanceDB 表模式
    def _build_schema(self) -> pa.Schema:

        # 从嵌入服务获取向量维度（首次调用时会自动探测 Ollama）
        dim = self._get_embedding_service().vector_dim
        # 定义表字段：
        # - id: 主键，UUID 字符串
        # - vector: 嵌入向量，固定长度 float32 数组
        # - text: 原始文本内容
        # - source_file: 来源文件名（用于按文件过滤/删除）
        # - doc_type: 文档类型（pdf/docx/xlsx/txt），用于按类型分组统计
        # - page_number: 页码（从 PDF 提取）
        # - chunk_index: 分片序号（同一文件内的顺序）
        # - metadata: JSON 字符串，存储额外元数据
        # - created_at / updated_at: 时间戳
        # - file_hash: 文件内容 SHA256 哈希，同文件所有 chunk 共享，用于内容去重
        # - file_size: 原始文件字节大小，同文件所有 chunk 共享，用于列表展示
        return pa.schema([
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("text", pa.string()),
            pa.field("source_file", pa.string()),
            pa.field("doc_type", pa.string()),
            pa.field("page_number", pa.int32()),
            pa.field("chunk_index", pa.int32()),
            pa.field("metadata", pa.string()),
            pa.field("created_at", pa.string()),
            pa.field("updated_at", pa.string()),
            pa.field("file_hash", pa.string()),
            pa.field("file_size", pa.int64()),
        ])

    # 获取当前时间的格式化字符串
    def _now_str(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ================================================================
    # 初始化 ── 创建表
    # ================================================================
    def init_table(
        self,
        table_name: Optional[str] = None,
        mode: str = "create",
    ) -> Dict[str, Any]:

        # 获取数据库连接
        db = self._get_db()
        # 未指定表名则使用配置中的默认值
        table_name = table_name or settings.lancedb_table_name
        # 构建表结构（此时会触发向量维度探测）
        schema = self._build_schema()

        try:
            # 检查表是否已存在
            if table_name in db.table_names():
                if mode == "overwrite":
                    # 覆盖模式：先删除旧表
                    db.drop_table(table_name)
                    logger.info(f"表 '{table_name}' 已删除（覆盖模式）")
                elif mode == "create":
                    # 创建模式：表已存在，返回提示信息
                    return {
                        "status": "exists",
                        "table_name": table_name,
                        "message": f"Table '{table_name}' already exists, use mode='overwrite' to recreate",
                    }

            # 创建新表（LanceDB 的 mode='create' 表示新建）
            tbl = db.create_table(
                name=table_name,
                schema=schema,
                mode="create",
            )
            logger.info(f"表 '{table_name}' 创建成功")
            return {
                "status": "created",
                "table_name": table_name,
                "schema": {f.name: str(f.type) for f in schema},
                "message": f"Table '{table_name}' created successfully",
            }
        except Exception as e:
            logger.error(f"初始化表 '{table_name}' 失败: {e}")
            raise

    # ================================================================
    # 插入数据
    # ================================================================
    def insert(
        self,
        texts: List[str],
        source_file: str = "",
        doc_type: str = "",
        metadatas: Optional[List[Dict[str, Any]]] = None,
        page_numbers: Optional[List[int]] = None,
        table_name: Optional[str] = None,
        file_hash: str = "",
        file_size: int = 0,
    ) -> Dict[str, Any]:

        table_name = table_name or settings.lancedb_table_name
        db = self._get_db()

        if table_name not in db.table_names():
            self.init_table(table_name)

        tbl = db.open_table(table_name)

        if source_file:
            try:
                escaped_source = source_file.replace("'", "''")
                before_count = tbl.count_rows()
                tbl.delete(f"source_file = '{escaped_source}'")
                after_count = tbl.count_rows()
                deleted = before_count - after_count
                if deleted > 0:
                    logger.info(
                        f"已删除源文件 '{source_file}' 的 {deleted} 个旧分片，准备重新插入"
                    )
            except Exception as e:
                logger.warning(f"删除源文件 '{source_file}' 的旧分片失败: {e}")
        vectors = self._get_embedding_service().encode_passages(texts)
        now = self._now_str()

        rows = []
        for i, (text, vec) in enumerate(zip(texts, vectors)):
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            page_num = page_numbers[i] if page_numbers and i < len(page_numbers) else -1
            row = {
                "id": str(uuid.uuid4()),
                "vector": vec,
                "text": text,
                "source_file": source_file,
                "doc_type": doc_type,
                "page_number": page_num,
                "chunk_index": i,
                "metadata": json.dumps(meta, ensure_ascii=False),
                "created_at": now,
                "updated_at": now,
                "file_hash": file_hash,
                "file_size": file_size,
            }
            rows.append(row)

        tbl.add(rows)
        logger.info(f"已向表 '{table_name}' 插入 {len(rows)} 个分片")
        return {
            "status": "success",
            "inserted_count": len(rows),
            "table_name": table_name,
            "message": f"Inserted {len(rows)} chunks",
        }

    # ================================================================
    # 查询 ── 向量相似度搜索
    # ================================================================
    def search(
        self,
        query_text: str,
        top_k: int = 5,
        table_name: Optional[str] = None,
        where_clause: Optional[str] = None,
        metric: str = "cosine",
        distance_threshold: float = 0.5,
    ) -> Dict[str, Any]:

        # 未指定表名则使用配置中的默认值
        table_name = table_name or settings.lancedb_table_name
        # 获取数据库连接
        db = self._get_db()

        # 检查表是否存在
        if table_name not in db.table_names():
            raise ValueError(
                f"テーブル '{table_name}' が存在しません。"
                f"先に init_table() を呼び出してください。"
            )

        # 打开目标表
        tbl = db.open_table(table_name)
        # 将查询文本转为向量（使用 query: 前缀，与入库时的 passage: 前缀区分）
        query_vector = self._get_embedding_service().encode_query(query_text)

        # 构建向量搜索查询：指定向量、度量方式、返回数量
        query_builder = (
            tbl.search(query_vector)
            .metric(metric)
            .limit(top_k)
            .select(["id", "text", "source_file", "page_number", "chunk_index", "metadata", "created_at"])
        )

        # 如果有过滤条件，追加 WHERE 子句（如按来源文件过滤）
        if where_clause:
            query_builder = query_builder.where(where_clause)

        # 执行搜索，返回 Pandas DataFrame
        results = query_builder.to_pandas()

        # 将 DataFrame 转为字典列表，便于 JSON 序列化
        items = []
        filtered_count = 0  # 被阈值过滤掉的结果数
        for _, row in results.iterrows():
            # 获取 LanceDB 返回的相似度距离（余弦距离，越小越相似）
            distance = float(row.get("_distance", 0))
            # 相似度阈值过滤：距离大于阈值的视为无关，丢弃
            if distance > distance_threshold:
                filtered_count += 1
                continue
            # 安全解析元数据 JSON（可能损坏或为空）
            try:
                meta = json.loads(row.get("metadata", "{}"))
            except (json.JSONDecodeError, TypeError):
                meta = {}

            items.append({
                "id": row.get("id", ""),
                "text": row.get("text", ""),
                "source_file": row.get("source_file", ""),
                "page_number": row.get("page_number", -1),
                "chunk_index": row.get("chunk_index", -1),
                "metadata": meta,
                "distance": distance,
                "created_at": row.get("created_at", ""),
            })

        logger.info(
            f"搜索 '{query_text[:50]}...' 返回 {len(items)} 条结果"
            f"（阈值={distance_threshold} 过滤了 {filtered_count} 条）"
        )
        return {
            "status": "success",
            "query": query_text,
            "top_k": top_k,
            "metric": metric,
            "results_count": len(items),
            "results": items,
        }

    # ================================================================
    # 全量数据 ── 获取表中所有文档片段（供 BM25 索引构建）
    # ================================================================
    def get_all_chunks(
        self,
        table_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        # 未指定表名则使用配置中的默认值
        table_name = table_name or settings.lancedb_table_name
        db = self._get_db()

        if table_name not in db.table_names():
            logger.warning(f"表 '{table_name}' 不存在")
            return []

        tbl = db.open_table(table_name)
        try:
            arrow_table = tbl.to_arrow()
            needed_columns = ["id", "text", "source_file", "page_number", "chunk_index", "metadata"]
            arrow_table = arrow_table.select(needed_columns)
            df = arrow_table.to_pandas()
        except Exception as e:
            logger.error(f"读取所有分片失败: {e}")
            return []

        chunks = []
        for _, row in df.iterrows():
            _meta_str = row.get("metadata", "{}")
            try:
                _meta = json.loads(_meta_str)
            except Exception:
                _meta = {}
            chunks.append({
                "id": row.get("id", ""),
                "text": row.get("text", ""),
                "source_file": row.get("source_file", ""),
                "page_number": int(row.get("page_number", -1)),
                "chunk_index": int(row.get("chunk_index", -1)),
                "metadata": _meta,
            })

        logger.info(f"已获取 {len(chunks)} 个分片，用于 BM25 索引构建")
        return chunks
    
    # ================================================================
    # 检查指定 source_file 是否已存在于向量数据库中
    # ================================================================
    def check_exists_by_source(
        self,
        source_file: str,
        table_name: Optional[str] = None,
    ) -> bool:
        table_name = table_name or settings.lancedb_table_name
        db = self._get_db()

        if table_name not in db.table_names():
            return False

        tbl = db.open_table(table_name)
        try:
            escaped_source = source_file.replace("'", "''")
            result = (
                tbl.search()
                .where(f"source_file = '{escaped_source}'")
                .select(["source_file"])
                .limit(1)
                .to_pandas()
            )
            return len(result) > 0
        except Exception as e:
            logger.warning(f"检查源文件 '{source_file}' 是否存在失败: {e}")
            return False

    # ================================================================
    # 删除指定 source_file 的所有记录
    # ================================================================
    def delete_by_source(
        self,
        source_file: str,
        table_name: Optional[str] = None,
    ) -> Dict[str, Any]:

        # 未指定表名则使用配置中的默认值
        table_name = table_name or settings.lancedb_table_name
        db = self._get_db()

        # 检查表是否存在
        if table_name not in db.table_names():
            raise ValueError(f"テーブル '{table_name}' が存在しません。")

        # 打开目标表
        tbl = db.open_table(table_name)
        # 转义文件名中的单引号，防止 SQL 注入
        escaped_source = source_file.replace("'", "''")
        # 构建 WHERE 条件：按来源文件过滤
        where_clause = f"source_file = '{escaped_source}'"

        # 记录删除前后的行数
        before_count = tbl.count_rows()
        tbl.delete(where_clause)
        after_count = tbl.count_rows()
        deleted = before_count - after_count

        logger.info(f"已删除源文件 '{source_file}' 的 {deleted} 条记录")
        return {
            "status": "success",
            "deleted_count": deleted,
            "source_file": source_file,
            "table_name": table_name,
            "message": f"Deleted {deleted} rows for source '{source_file}'",
        }

    # ================================================================
    # 文档统计 ── 按类型统计文档数量和最后更新时间
    # ================================================================
    def get_document_stats(
        self,
        table_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        table_name = table_name or settings.lancedb_table_name
        db = self._get_db()

        if table_name not in db.table_names():
            return {
                "status": "success",
                "stats": {},
                "last_updated": "",
            }

        tbl = db.open_table(table_name)

        try:
            # to_arrow() 加载全部列，再用 PyArrow select() 只保留需要的列
            arrow_table = tbl.to_arrow()
            arrow_table = arrow_table.select(["source_file", "doc_type", "updated_at"])
            df = arrow_table.to_pandas()
        except Exception as e:
            logger.error(f"读取表数据用于统计失败: {e}")
            return {
                "status": "error",
                "stats": {},
                "last_updated": "",
                "error": str(e),
            }

        if df.empty:
            return {
                "status": "success",
                "stats": {},
                "last_updated": "",
            }

        # 按 doc_type 分组，统计每组内不重复的 source_file 数量
        stats = {}
        if "source_file" in df.columns:
            # 如果表中没有 doc_type 列（旧数据），则从 source_file 扩展名推导
            if "doc_type" not in df.columns:
                df["doc_type"] = df["source_file"].apply(
                    lambda f: _extract_doc_type(str(f))
                )

            for doc_type, group in df.groupby("doc_type"):
                # 跳过空字符串的分组（旧数据可能没有 doc_type）
                if not doc_type:
                    continue
                # nunique() 计算不重复的 source_file 数量
                stats[doc_type] = int(group["source_file"].nunique())

        # 取所有记录中最大的 updated_at 作为最终文档更新日期
        last_updated = ""
        if "updated_at" in df.columns and not df["updated_at"].isna().all():
            max_ts = df["updated_at"].max()
            # 格式化：日期部分用 / 分隔，保留时分秒
            # 输入格式 "2026-07-23 14:30:00" → 输出 "2026/07/23 14:30:00"
            ts_str = str(max_ts)
            last_updated = ts_str[:10].replace("-", "/") + ts_str[10:]

        logger.info(
            f"文档统计: {stats}, 最后更新: {last_updated}"
        )
        return {
            "status": "success",
            "stats": stats,
            "last_updated": last_updated,
        }

    # ================================================================
    # 文档列表 ── 获取所有已入库文档（去重聚合）
    # ================================================================
    def get_documents(
        self,
        table_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        table_name = table_name or settings.lancedb_table_name
        db = self._get_db()

        if table_name not in db.table_names():
            logger.info(f"表 '{table_name}' 不存在，返回空文档列表")
            return {
                "status": "success",
                "documents": [],
            }

        tbl = db.open_table(table_name)

        try:
            # to_arrow() 加载全部列，再用 PyArrow select() 只保留需要的列
            arrow_table = tbl.to_arrow()
            arrow_table = arrow_table.select(
                ["source_file", "doc_type", "file_size", "created_at"]
            )
            df = arrow_table.to_pandas()
        except Exception as e:
            logger.error(f"读取文档列表失败: {e}")
            return {
                "status": "error",
                "documents": [],
                "error": str(e),
            }

        if df.empty:
            return {
                "status": "success",
                "documents": [],
            }

        # 按 source_file 分组，取每个文档的第一条记录
        # （同一文档的所有 chunk 共享相同的 doc_type / file_size）
        if "source_file" not in df.columns:
            return {
                "status": "success",
                "documents": [],
            }

        # 如果表中没有 doc_type 列（旧数据），从 source_file 扩展名推导
        if "doc_type" not in df.columns:
            df["doc_type"] = df["source_file"].apply(
                lambda f: _extract_doc_type(str(f))
            )

        # 按 source_file 分组，取第一条记录（first() 保留所有列）
        grouped = df.groupby("source_file", as_index=False).first()

        documents = []
        for _, row in grouped.iterrows():
            doc = {
                "source_file": str(row.get("source_file", "")),
                "doc_type": str(row.get("doc_type", "")),
                "file_size": int(row.get("file_size", 0)),
                "created_at": str(row.get("created_at", "")),
            }
            documents.append(doc)

        # 按创建时间倒序排列（最新的在前）
        documents.sort(key=lambda d: d["created_at"], reverse=True)

        logger.info(f"已获取 {len(documents)} 个不重复文档")
        return {
            "status": "success",
            "documents": documents,
        }

# 全局单例：供其他模块（router 等）
vector_db_service = VectorDBService()