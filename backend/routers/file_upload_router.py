
from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
import os
import shutil
from typing import List, Dict, Any
import hashlib
import logging
import zipfile
import io
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# 导入文件处理服务（解析 + 索引入库 + 文件保存 + 删除）
from services.file_processing_service import (
    parse_file_to_markdown, index_document, FileProcessingError,
    save_file_permanent, delete_physical_file, delete_document as doc_delete,
)
# 导入向量数据库服务
from services.vector_db_service import vector_db_service
# 导入混合检索器
from services.hybrid_retriever_service import hybrid_retriever
# 导入配置设置
from config.settings import settings

router = APIRouter(prefix="/file-upload", tags=["file-upload"])

# ===============================
# 文件存储目录（统一由 settings.file_storage_dir 管理
# 指向 %LOCALAPPDATA%\machiariRAG\data\files）
# ===============================
# 确保目录存在
os.makedirs(settings.file_storage_dir, exist_ok=True)

# ===============================
# 文件上传安全限制
# ===============================
# 最大文件大小：50MB（与前端 MAX_FILE_SIZE 保持一致）
MAX_FILE_SIZE = 50 * 1024 * 1024
# 允许的文件扩展名白名单
ALLOWED_EXTENSIONS: set = {
    ".pdf", ".docx", ".doc", ".xlsx", ".xlsm", ".xls",
    ".txt", ".csv", ".json", ".html", ".htm", ".md", ".xml",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp",
}

# 安全构造存储目录内的文件路径，防止路径遍历攻击
def _safe_storage_path(filename: str) -> str:
    file_path = os.path.join(settings.file_storage_dir, filename)
    real_path = os.path.realpath(file_path)
    real_storage = os.path.realpath(settings.file_storage_dir)
    if not real_path.startswith(real_storage + os.sep):
        raise HTTPException(status_code=400, detail="不正なファイル名です")
    return real_path

# 清理临时文件和目录（上传/更新共用）
def _cleanup_temp_files(tmp_path: str, tmp_dir: str) -> None:
    if tmp_path and os.path.exists(tmp_path):
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    if tmp_dir and os.path.exists(tmp_dir):
        try:
            shutil.rmtree(tmp_dir)
        except OSError:
            pass

# ===============================
# API Route：上传并分片入库
# ===============================
@router.post("")
async def upload_documents(
    # 参数1：接收多个上传文件，来自表单的 files 字段
    files: List[UploadFile] = File(...),
    # 参数2：是否自动分片入库（查询参数，默认 True）
    auto_index: bool = Query(True, description="是否自动将文件内容分片存入向量数据库"),
):
    # 初始化结果列表，用于存储每个文件的处理结果
    result_list: List[Dict[str, Any]] = []

    # 遍历每个上传的文件
    for upload_file in files:
        # 获取文件名，若为空则默认为 "unknown"
        filename = upload_file.filename or "unknown"
        # 异步读取文件二进制内容
        file_bytes = await upload_file.read()
        # 计算文件 SHA256 指纹和字节大小，用于内容去重和列表展示
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        file_size = len(file_bytes)
        # 获取文件扩展名（小写，如 .pdf）
        ext = os.path.splitext(filename)[1].lower()
        # 初始化清理后的 Markdown 内容
        clean_md = ""

        # 初始化变量：临时文件路径、PDF永久路径、向量入库结果
        tmp_path = None
        tmp_dir = None
        pdf_perm_path = ""
        vector_result = None

        try:
            # 安全校验：文件大小 + 扩展名白名单
            if file_size > MAX_FILE_SIZE:
                raise ValueError(
                    f"ファイルサイズが上限を超えています"
                    f"（{file_size / (1024*1024):.1f}MB > {MAX_FILE_SIZE / (1024*1024):.0f}MB）"
                )
            if ext not in ALLOWED_EXTENSIONS:
                raise ValueError(
                    f"サポートされていないファイル形式です: {ext}"
                )

            # 保存永久副本用于溯源
            pdf_perm_path = save_file_permanent(filename, file_bytes)

            # 调用共通服务：解析文件为 Markdown
            result = parse_file_to_markdown(file_bytes, filename, ext)
            tmp_path = result.tmp_path
            tmp_dir = result.tmp_dir
            clean_md = result.clean_md
            doc_type = result.doc_type

            # 如果开启自动索引且 Markdown 内容非空
            if auto_index and clean_md:
                # 调用共通服务：分片 + 入库 + 刷新 BM25
                chunks_count = index_document(
                    clean_md, filename, doc_type,
                    file_hash, file_size, pdf_perm_path,
                )
                vector_result = {
                    "status": "success",
                    "inserted_count": chunks_count,
                }
            else:
                # 未开启自动索引 或 内容为空 → 统一返回 skipped 状态（避免前端收到 None）
                vector_result = {"status": "skipped"}

        except FileProcessingError as e:
            # 文件处理阶段异常（LibreOffice 未安装、格式转换失败等）
            clean_md = ""
            logger.error(f"文件处理异常: {filename} - {e}")
            vector_result = {
                "status": "error",
                "error": str(e),
                "error_type": e.error_type,
            }
        except Exception as e:
            # 其他未预期的异常
            clean_md = ""
            logger.error(f"文件处理异常: {filename}", exc_info=True)
            vector_result = {
                "status": "error",
                "error": f"ファイルの処理に失敗しました: {str(e)}",
                "error_type": "unknown_error",
            }

        finally:
            # 无论成功或失败，都执行清理操作
            _cleanup_temp_files(tmp_path, tmp_dir)

        # 将当前文件的处理结果添加到结果列表
        result_list.append({
            "filename": filename,           # 原始文件名
            "clean_markdown": clean_md,     # 清理后的 Markdown 内容
            "pdf_path": pdf_perm_path,      # PDF 永久存储路径（用于溯源）
            "vector_result": vector_result, # 向量数据库入库结果
        })

    # 返回最终结果：成功状态码、消息、处理结果列表
    return {"code": 0, "msg": "success", "data": result_list}

# ===============================
# API Route：检查文件是否已存在于向量数据库
# ===============================
@router.get("/check-exists")
async def check_file_exists(
    filename: str = Query(..., description="要检查的文件名"),
):
    exists = vector_db_service.check_exists_by_source(
        source_file=filename,
        table_name=settings.lancedb_table_name,
    )
    return {
        "code": 0,
        "msg": "success",
        "data": {
            "filename": filename,
            "exists": exists,
        },
    }


# ===============================
# API Route：删除文档（向量库记录 + 物理文件 + 刷新 BM25 索引）
# ===============================
@router.delete("/{filename:path}")
async def delete_document(
    filename: str,
):
    file_path = _safe_storage_path(filename)
    result = doc_delete(filename, file_path)
    return {
        "code": 0,
        "msg": "success",
        "data": {
            "filename": filename,
            "chunks_deleted": result["deleted_count"],
            "file_deleted": result["file_deleted"],
        },
    }


# ===============================
# API Route：单文件下载
# ===============================
@router.get("/download/{filename:path}")
async def download_file(filename: str):
    file_path = _safe_storage_path(filename)
    if not os.path.isfile(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"ファイルが見つかりません: {filename}",
        )

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream",
    )


# ===============================
# API Route：批量下载（ZIP 打包）
# ===============================
class BatchDownloadRequest(BaseModel):
    filenames: list[str]

@router.post("/download-batch")
async def download_batch(body: BatchDownloadRequest):
    filenames = body.filenames
    if not filenames:
        raise HTTPException(status_code=400, detail="ダウンロードするファイルが指定されていません")

    zip_buffer = io.BytesIO()
    missing: list[str] = []

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in filenames:
            try:
                file_path = _safe_storage_path(name)
            except HTTPException:
                missing.append(name)
                continue
            if os.path.isfile(file_path):
                zf.write(file_path, arcname=name)
            else:
                missing.append(name)

    if missing:
        logger.warning(f"Batch download: {len(missing)} files not found - {missing}")

    if zip_buffer.tell() == 0:
        raise HTTPException(
            status_code=404,
            detail="指定されたファイルは1件も見つかりませんでした",
        )

    zip_buffer.seek(0)

    # 生成带时间戳的 ZIP 文件名
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"machiari_documents_{ts}.zip"

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{zip_name}"',
        },
    )


# ===============================
# API Route：文档更新（PUT /update）
# ===============================
class UpdateDocumentResponse(BaseModel):
    code: int
    msg: str
    data: dict  # {old_filename, new_filename, extension_changed, chunks_count}

@router.put("/update")
async def update_document(
    # 被替换的原文件名
    old_filename: str = Query(..., description="要更新的文件名"),
    # 新文件
    file: UploadFile = File(..., description="新文件"),
    # 是否自动分片入库
    auto_index: bool = Query(True, description="是否自动入库到向量数据库"),
):
    # ── 基本信息的获取 ──
    new_filename = file.filename or "unknown"
    new_bytes = await file.read()
    new_ext = os.path.splitext(new_filename)[1].lower()
    old_ext = os.path.splitext(old_filename)[1].lower()
    new_size = len(new_bytes)

    # ── 安全校验：文件大小 + 扩展名白名单 ──
    if new_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"ファイルサイズが上限を超えています"
                   f"（{new_size / (1024*1024):.1f}MB > {MAX_FILE_SIZE / (1024*1024):.0f}MB）",
        )
    if new_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"サポートされていないファイル形式です: {new_ext}",
        )

    # ── 旧文件存在确认 ──
    old_path = _safe_storage_path(old_filename)
    if not os.path.isfile(old_path):
        raise HTTPException(
            status_code=404,
            detail=f"更新対象のファイルが見つかりません: {old_filename}",
        )

    # ── 扩展名一致性判定 ──
    extension_changed = (old_ext != new_ext)

    # ── 准备：临时文件路径变量（finally 中必定清理） ──
    tmp_path = None
    tmp_dir = None
    clean_md = ""
    doc_type = ""

    try:
        # ========== 阶段①：调用共通服务解析新文件 ==========
        result = parse_file_to_markdown(new_bytes, new_filename, new_ext)
        tmp_path = result.tmp_path
        tmp_dir = result.tmp_dir
        clean_md = result.clean_md
        doc_type = result.doc_type

        # ── 解析结果的验证 ──
        if not clean_md:
            raise ValueError("ファイルの解析結果が空です。破損ファイルの可能性があります。")

        # ========== 阶段②：解析成功 → 提交 ==========
        # 到这一步说明新文件内容有效。更新物理文件和向量数据库。

        # ── ① 保存新物理文件（同名时覆盖） ──
        #     auto_index=False 时也保存文件，仅跳过向量入库
        save_file_permanent(new_filename, new_bytes)
        logger.info(f"Update: new file saved '{new_filename}'")

        chunks_count = 0
        if auto_index:
            # ── 分片 + 入库 + 刷新 BM25（共通服务） ──
            pdf_perm_path = os.path.join(settings.file_storage_dir, new_filename)
            new_hash = hashlib.sha256(new_bytes).hexdigest()
            chunks_count = index_document(
                clean_md, new_filename, doc_type,
                new_hash, new_size, pdf_perm_path,
            )

            # ── ③ 扩展名变更时：删除旧数据（新数据确定后执行） ──
            if extension_changed:
                # 删除旧向量数据
                vector_db_service.delete_by_source(
                    source_file=old_filename,
                    table_name=settings.lancedb_table_name,
                )
                # 删除旧物理文件（回收站）
                delete_physical_file(old_path)
                logger.info(
                    f"Update (ext changed): cleaned up old file & chunks for '{old_filename}'"
                )

        logger.info(
            f"Document updated: '{old_filename}' → '{new_filename}'"
            f" (ext_changed={extension_changed}, chunks={chunks_count})"
        )

        return UpdateDocumentResponse(
            code=0,
            msg="success",
            data={
                "old_filename": old_filename,
                "new_filename": new_filename,
                "extension_changed": extension_changed,
                "chunks_count": chunks_count,
            },
        )

    except FileProcessingError as e:
        # 文件处理阶段异常（LibreOffice 未安装、格式转换失败等）
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        # HTTPException 直接重新抛出（FastAPI 会返回适当的错误响应）
        raise
    except Exception as e:
        # 解析/embed 期间的错误：原数据无损，finally 中仅删除临时文件
        logger.error(f"Update failed for '{old_filename}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"ファイルの更新に失敗しました: {str(e)}",
        )

    finally:
        # ── 清理：删除所有临时文件 ──
        _cleanup_temp_files(tmp_path, tmp_dir)

# ===============================
# 预览相关：MIME 类型映射
# ===============================
def _get_mime_type(ext: str) -> str:
    mapping = {
        ".pdf":  "application/pdf",
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif":  "image/gif",
        ".bmp":  "image/bmp",
        ".webp": "image/webp",
        ".txt":  "text/plain; charset=utf-8",
        ".csv":  "text/plain; charset=utf-8",
        ".json": "text/plain; charset=utf-8",
        ".html": "text/plain; charset=utf-8",
        ".htm":  "text/plain; charset=utf-8",
        ".md":   "text/plain; charset=utf-8",
        ".xml":  "text/plain; charset=utf-8",
    }
    return mapping.get(ext, "application/octet-stream")


# ===============================
# API Route：文件预览（原始文件）
# ===============================
@router.get("/preview-file/{filename:path}")
async def preview_file(filename: str):
    file_path = _safe_storage_path(filename)

    if not os.path.isfile(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"ファイルが見つかりません: {filename}",
        )

    ext = os.path.splitext(filename)[1].lower()
    media_type = _get_mime_type(ext)

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type=media_type,
        # content_disposition_type="inline": 告诉浏览器内联渲染（而非下载）
        # 配合前端 iframe/img 标签使用，PDF 在浏览器内置查看器中预览
        content_disposition_type="inline",
    )


# ===============================
# API Route：文件预览（Markdown 文本）
# ===============================
class PreviewContentResponse(BaseModel):
    code: int = 0
    msg: str = "success"
    data: dict  # {filename, content}

@router.get("/preview-content/{filename:path}")
async def preview_content(filename: str) -> PreviewContentResponse:
    # 获取全量 chunk，按 source_file 过滤
    all_chunks = vector_db_service.get_all_chunks()
    chunks = [c for c in all_chunks if c.get("source_file") == filename]

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail=f"文書のプレビュー内容が見つかりません: {filename}",
        )

    # 按 chunk_index 升序排列，拼接文本
    chunks.sort(key=lambda c: c.get("chunk_index", 0))
    content = "\n\n".join(c.get("text", "") for c in chunks)

    logger.info(f"Preview content assembled: {filename} ({len(chunks)} chunks, {len(content)} chars)")

    return PreviewContentResponse(
        code=0,
        msg="success",
        data={
            "filename": filename,
            "content": content,
        },
    )