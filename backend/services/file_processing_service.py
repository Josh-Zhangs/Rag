import os
import tempfile
import shutil
import logging
import send2trash
from dataclasses import dataclass
from typing import Optional

from services.file_upload_service import (
    parse_pdf, parse_docx, parse_xlsx, parse_txt, parse_image,
    clean_markdown, convert_doc_to_docx, convert_xls_to_xlsx,
    is_libreoffice_available,
)
from services.chunking_service import chunking_service
from services.vector_db_service import vector_db_service
from services.hybrid_retriever_service import hybrid_retriever
from config.settings import settings

logger = logging.getLogger(__name__)

# 文件处理异常类
class FileProcessingError(Exception):
    def __init__(self, message: str, error_type: str = "unknown_error"):
        super().__init__(message)
        self.error_type = error_type

# 文件解析结果数据类
@dataclass
class ParseResult:
    clean_md: str
    doc_type: str
    tmp_path: str
    tmp_dir: Optional[str] = None


# LibreOffice 未安装时的错误消息（上传/更新共用）
LIBREOFFICE_NOT_FOUND_MSG = (
    "旧形式ファイル(.doc/.xls)を処理するため、"
    "LibreOfficeが見つかりません。"
    "ソフトをインストールした上で再度アップロードしてください。"
)


# 创建临时文件并写入内容（共通：PDF/DOCX/XLSX/Image/TXT 复用）
def _create_temp_file(file_bytes: bytes, suffix: str) -> str:
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(tmp_fd)
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)
    return tmp_path


# LibreOffice 旧格式转换（.doc→.docx / .xls→.xlsx 共用）
# 返回 (tmp_path, tmp_dir)：调用方需在 finally 中清理 tmp_dir
def _convert_old_office_format(
    file_bytes: bytes,
    old_ext: str,
    new_ext: str,
    convert_func,
) -> tuple:
    tmp_dir = tempfile.mkdtemp()
    try:
        tmp_path = _create_temp_file(file_bytes, old_ext)
        converted_path = convert_func(tmp_path, tmp_dir)
        tmp_fd2, tmp_path2 = tempfile.mkstemp(suffix=new_ext)
        os.close(tmp_fd2)
        try:
            shutil.copy2(converted_path, tmp_path2)
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            tmp_path = tmp_path2
        except Exception:
            if os.path.exists(tmp_path2):
                os.unlink(tmp_path2)
            raise
        return tmp_path, tmp_dir
    except Exception:
        # 转换失败时清理 tmp_dir，防止泄漏
        if os.path.exists(tmp_dir):
            try:
                shutil.rmtree(tmp_dir)
            except OSError:
                pass
        raise

# 文件解析服务：多格式 → Markdown
def parse_file_to_markdown(
    file_bytes: bytes,
    filename: str,
    ext: str,
) -> ParseResult:

    # ========== PDF 文件处理 ==========
    if ext == ".pdf":
        tmp_path = _create_temp_file(file_bytes, ".pdf")
        raw_md = parse_pdf(tmp_path, filename)
        clean_md = clean_markdown(raw_md)
        return ParseResult(clean_md=clean_md, doc_type="pdf", tmp_path=tmp_path)

    # ========== DOCX/DOC 文件处理 ==========
    elif ext in (".docx", ".doc"):
        if ext == ".doc":
            if not is_libreoffice_available():
                raise FileProcessingError(
                    LIBREOFFICE_NOT_FOUND_MSG, "libreoffice_not_found"
                )
            try:
                tmp_path, tmp_dir = _convert_old_office_format(
                    file_bytes, ".doc", ".docx", convert_doc_to_docx,
                )
            except Exception as e:
                raise FileProcessingError(
                    f".doc ファイルの変換に失敗しました: {str(e)}",
                    "file_conversion_failed",
                ) from e
        else:
            tmp_path = _create_temp_file(file_bytes, ".docx")
            tmp_dir = None
        raw_md = parse_docx(tmp_path, filename)
        clean_md = clean_markdown(raw_md)
        return ParseResult(clean_md=clean_md, doc_type="docx", tmp_path=tmp_path, tmp_dir=tmp_dir)

    # ========== XLSX/XLSM/XLS 文件处理 ==========
    elif ext in (".xlsx", ".xlsm", ".xls"):
        if ext == ".xls":
            if not is_libreoffice_available():
                raise FileProcessingError(
                    LIBREOFFICE_NOT_FOUND_MSG, "libreoffice_not_found"
                )
            try:
                tmp_path, tmp_dir = _convert_old_office_format(
                    file_bytes, ".xls", ".xlsx", convert_xls_to_xlsx,
                )
            except Exception as e:
                raise FileProcessingError(
                    f".xls ファイルの変換に失敗しました: {str(e)}",
                    "file_conversion_failed",
                ) from e
        else:
            tmp_path = _create_temp_file(file_bytes, ext)
            tmp_dir = None
        raw_md = parse_xlsx(tmp_path, original_name=filename)
        clean_md = clean_markdown(raw_md)
        return ParseResult(clean_md=clean_md, doc_type="xlsx", tmp_path=tmp_path, tmp_dir=tmp_dir)

    # ========== 图像文件处理 ==========
    elif ext in (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"):
        tmp_path = _create_temp_file(file_bytes, ext)
        raw_md = parse_image(tmp_path, filename)
        clean_md = clean_markdown(raw_md)
        return ParseResult(clean_md=clean_md, doc_type="img", tmp_path=tmp_path)

    # ========== 文本类文件处理（TXT/CSV/JSON/HTML/MD/XML） ==========
    elif ext in (".txt", ".csv", ".json", ".html", ".htm", ".md", ".xml"):
        tmp_path = _create_temp_file(file_bytes, ext)
        raw_md = parse_txt(tmp_path, filename)
        clean_md = clean_markdown(raw_md)
        return ParseResult(clean_md=clean_md, doc_type="txt", tmp_path=tmp_path)

    else:
        raise FileProcessingError(
            f"サポートされていないファイル形式です: {ext}",
            "unsupported_format",
        )

# 文档索引服务：分片 + 向量入库 + 刷新 BM25 索引
def index_document(
    clean_md: str,
    filename: str,
    doc_type: str,
    file_hash: str,
    file_size: int,
    pdf_perm_path: str = "",
) -> int:
    # 调用分片服务：将 Markdown 切分为小块
    chunks, page_numbers, metadatas = chunking_service.chunk_markdown(
        clean_md,
        source_file=filename,
        pdf_path=pdf_perm_path,
    )
    # 调用向量数据库服务：插入分片数据
    vector_result = vector_db_service.insert(
        texts=chunks,
        source_file=filename,
        doc_type=doc_type,
        page_numbers=page_numbers,
        metadatas=metadatas,
        table_name=settings.lancedb_table_name,
        file_hash=file_hash,
        file_size=file_size,
    )
    if vector_result.get("status") != "success":
        raise RuntimeError(
            vector_result.get("message", "ベクトルDBへの挿入に失敗しました")
        )
    # 插入新文档后重建 BM25 索引，确保关键词检索包含新文档
    hybrid_retriever.refresh_bm25_index()
    return vector_result.get("inserted_count", len(chunks))


# 保存文件到永久存储目录
def save_file_permanent(filename: str, file_bytes: bytes) -> str:
    """将原始文件保存到永久存储目录，返回存储路径"""
    out_path = os.path.join(settings.file_storage_dir, filename)
    with open(out_path, "wb") as f:
        f.write(file_bytes)
    return out_path


# 删除物理文件（优先：回收站，不可用时：永久删除）
def delete_physical_file(file_path: str) -> None:
    if not os.path.exists(file_path):
        return
    try:
        send2trash.send2trash(file_path)
        logger.info(f"Moved to Recycle Bin: {file_path}")
    except send2trash.TrashPermissionError:
        try:
            os.remove(file_path)
            logger.warning(f"Recycle Bin unavailable, permanently deleted: {file_path}")
        except OSError as e:
            logger.error(f"Physical file deletion failed: {file_path} - {e}")
    except OSError as e:
        logger.error(f"Physical file deletion failed: {file_path} - {e}")

# 删除文档服务
def delete_document(filename: str, file_path: str) -> dict:

    # 1. 从向量数据库删除该文件的所有 chunk
    try:
        db_result = vector_db_service.delete_by_source(
            source_file=filename,
            table_name=settings.lancedb_table_name,
        )
    except Exception as e:
        logger.error(f"Vector DB deletion failed: {filename} - {e}")
        db_result = {
            "status": "error",
            "deleted_count": 0,
            "error": str(e),
        }

    # 2. 删除物理文件（回收站优先 → 永久删除回退）
    file_deleted = False
    if os.path.exists(file_path):
        delete_physical_file(file_path)
        file_deleted = not os.path.exists(file_path)
    else:
        logger.warning(f"Physical file not found, skipping deletion: {file_path}")

    # 3. 刷新 BM25 全文索引，确保检索结果与向量库一致
    try:
        hybrid_retriever.refresh_bm25_index()
        logger.info(f"BM25 index refreshed (after deleting {filename})")
    except Exception as e:
        logger.error(f"BM25 index refresh failed: {e}")

    return {
        "deleted_count": db_result.get("deleted_count", 0),
        "file_deleted": file_deleted,
    }