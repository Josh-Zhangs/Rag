import re
import logging
import hashlib
from typing import List, Dict, Any, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import settings

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


class ChunkingService:

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    # 根据文件格式返回合适的分片大小
    @staticmethod
    def _get_format_chunk_size(source_file: str) -> int:
        ext = source_file.lower().rsplit('.', 1)[-1] if '.' in source_file else ''
        if ext in ('txt',):
            return 1000
        if ext in ('png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif', 'gif', 'webp'):
            return 1000
        if ext in ('xlsx', 'xls'):
            return 600
        # PDF/DOCX等
        return 800

    # 将 Markdown 文本按页/Sheet 分割并分片
    def chunk_markdown(
        self,
        markdown_text: str,
        source_file: str,
        pdf_path: str = "",
    ) -> Tuple[List[str], List[int], List[Dict[str, Any]]]:

        # 根据文件格式设定当前分片大小
        format_chunk_size = self._get_format_chunk_size(source_file)
        # 先按 ## Page N 标题分割出各页内容
        pages = self._split_by_page(markdown_text)

        # 如果没有页码标记，尝试按 ## Sheet: 标题分割（Excel 场景）
        if not pages:
            sheets = self._split_by_sheet(markdown_text)
            if sheets:
                return self._chunk_by_sheet(sheets, source_file, pdf_path, format_chunk_size)

        # 如果仍然没有检测到任何标记，走降级分片（整篇文本直接切分）
        if not pages:
            return self._chunk_flat(markdown_text, source_file, -1, pdf_path, format_chunk_size)

        # 存储所有分片的文本内容
        all_texts: List[str] = []
        # 存储每个分片对应的页码
        all_page_numbers: List[int] = []
        # 存储每个分片的元数据（来源文件、页码、序号等）
        all_metadatas: List[Dict[str, Any]] = []
        # 全局分片序号计数器，跨页累加保证序号唯一
        global_chunk_idx = 0

        # 逐页进行语义分片
        for page_num, page_content in pages:
            # 对当前页内容进行分片，返回该页的文本和元数据
            chunks, metas = self._split_page_content(
                page_content, page_num, source_file, pdf_path, global_chunk_idx, format_chunk_size
            )
            # 收集当前页的分片文本
            all_texts.extend(chunks)
            # 当前页的所有分片共享同一页码
            all_page_numbers.extend([page_num] * len(chunks))
            # 收集当前页的分片元数据
            all_metadatas.extend(metas)
            # 累加全局分片序号，为下一页提供起始序号
            global_chunk_idx += len(chunks)

        # 记录分片结果日志
        logger.info(
            f"Chunked '{source_file}': {len(pages)} pages -> {len(all_texts)} chunks"
        )
        return all_texts, all_page_numbers, all_metadatas

    # ================================================================
    # 按 Page 分割
    # ================================================================
    # 匹配 ## Page 数字 格式的页面标题
    _PAGE_HEADER_PATTERN = re.compile(r"^## Page (\d+)\s*$", re.MULTILINE)
    # 匹配 ## Sheet: 名称 格式的 Sheet 标题
    _SHEET_HEADER_PATTERN = re.compile(r"^## Sheet: (.+)$", re.MULTILINE)
    def _split_by_page(self, text: str) -> List[Tuple[int, str]]:
        # 查找所有页面标题的匹配位置
        matches = list(self._PAGE_HEADER_PATTERN.finditer(text))

        # 没有找到任何页面标题，返回空列表通知调用方走降级路径
        if not matches:
            return []

        # 存储各页的 (页码, 内容) 元组
        pages: List[Tuple[int, str]] = []
        for i, m in enumerate(matches):
            # 提取页码数字（group(1) 是正则中 (\d+) 捕获的纯数字）
            page_num = int(m.group(1))
            # 当前页内容起始位置：标题行之后
            start = m.end()
            # 当前页内容结束位置：下一个标题之前，或文本末尾
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            # 截取当前页的文本内容并去除首尾空白
            content = text[start:end].strip()

            # 清理页面中的噪声内容（PDF 转换器产生的标题、版权声明等）
            content = self._clean_page_content(content)

            # 清理后仍有内容才保留，完全空白的页面直接丢弃
            if content:
                pages.append((page_num, content))

        return pages

    # 按 Sheet 分割
    def _split_by_sheet(self, text: str) -> List[Tuple[str, str]]:
        # 查找所有 Sheet 标题的匹配位置
        matches = list(self._SHEET_HEADER_PATTERN.finditer(text))

        # 没有找到任何 Sheet 标题，返回空列表通知调用方走降级路径
        if not matches:
            return []

        # 存储各 Sheet 的 (Sheet名, 内容) 元组
        sheets: List[Tuple[str, str]] = []
        for i, m in enumerate(matches):
            # 提取 Sheet 名称（group(1) 是正则中 (.+) 捕获的文本）
            sheet_name = m.group(1)
            # 当前 Sheet 内容起始位置：标题行之后
            start = m.end()
            # 当前 Sheet 内容结束位置：下一个标题之前，或文本末尾
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            # 截取当前 Sheet 的文本内容并去除首尾空白
            content = text[start:end].strip()

            # 清理 Sheet 中的噪声内容（复用 _clean_page_content 的清洗逻辑）
            content = self._clean_page_content(content)

            # 清理后仍有内容才保留，完全空白的 Sheet 直接丢弃
            if content:
                sheets.append((sheet_name, content))

        return sheets

    # 清理页面中的噪声内容
    def _clean_page_content(self, content: str) -> str:
        # 只删除 PDF 转换器产生的特定噪声标题（如 "### Text Content"、"### 表"）
        # 使用白名单模式仅匹配已知噪声，避免误删正文中的合法 ### 标题
        content = re.sub(
            r"^### (?:Text Content|表|図|目次)\s*\n", "", content, flags=re.MULTILINE
        )
        # 删除版权声明行（以 © 开头的整行，可能带前导空白）
        content = re.sub(r"^\s*©.*\n?", "", content, flags=re.MULTILINE)
        # 注意：这里不删除 - ** 键值对行，因为 PDF 转换器会将正文内容
        # 也输出为 - **key**: value 格式，删除会导致整页正文丢失
        # 合并多余空行（3 个及以上连续空行 → 2 个空行），保持排版整洁
        content = re.sub(r"\n{3,}", "\n\n", content)
        # 去除最终的首尾空白
        return content.strip()

    # ================================================================
    # 页内语义分片
    # ================================================================
    def _get_splitter(self, format_chunk_size: int = 0) -> RecursiveCharacterTextSplitter:
        # 使用格式感知的分片大小，未设定时回退到默认值
        chunk_size = format_chunk_size or self.chunk_size
        return RecursiveCharacterTextSplitter(
            # 每个分片的目标字符数（按文件格式自动调整）
            chunk_size=chunk_size,
            # 相邻分片之间的重叠字符数
            chunk_overlap=self.chunk_overlap,
            # 分隔符优先级：段落 > 换行 > 句号 > 顿号 > 空格 > 字符
            separators=["\n\n", "\n", "。", "、", " ", ""],
            # 使用 Python 内置 len 计算文本长度（字符数）
            length_function=len,
            # 分隔符为普通字符串，非正则表达式
            is_separator_regex=False,
        )

    # 对单页内容进行语义分片
    def _split_page_content(
        self,
        content: str,
        page_num: int,
        source_file: str,
        pdf_path: str,
        start_chunk_idx: int,
        format_chunk_size: int = 0,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        # 获取格式感知的文本分割器（每次新建以保证按文件格式调整 chunk_size）
        splitter = self._get_splitter(format_chunk_size)
        # 执行文本分割，得到原始分片列表
        splits = splitter.split_text(content)

        # 收集有效分片的文本和元数据
        texts = []
        metadatas = []
        # 独立计数器，避免因跳过空分片导致 chunk_index 跳号
        actual_idx = 0
        for split_text in splits:
            # 去除首尾空白
            clean = split_text.strip()
            # 跳过空白分片（全部是空格/换行的分片无意义）
            if not clean:
                continue
            # 保留有效文本
            texts.append(clean)
            # 构建分片元数据，用于后续溯源和检索
            metadatas.append({
                # 所属源文件路径
                "source_file": source_file,
                # 所属页码
                "page_number": page_num,
                # 全局分片序号（连续，无跳号）
                "chunk_index": start_chunk_idx + actual_idx,
                # 原始 PDF 路径
                "pdf_path": pdf_path,
                # 分片类型标识
                "chunk_type": "text",
            })
            # 有效分片计数 +1
            actual_idx += 1

        return texts, metadatas

    # ================================================================
    # 无页码降级分片
    # ================================================================
    def _chunk_flat(
        self,
        text: str,
        source_file: str,
        page_num: int,
        pdf_path: str,
        format_chunk_size: int = 0,
    ) -> Tuple[List[str], List[int], List[Dict[str, Any]]]:
        # 直接对整篇文本进行分片（不按页分割），page_num 统一设为 -1
        chunks, metas = self._split_page_content(
            text, page_num, source_file, pdf_path, 0, format_chunk_size
        )
        # 所有分片使用同一页码（-1 表示未知页码）
        page_numbers = [page_num] * len(chunks)
        return chunks, page_numbers, metas

    # 按 Sheet 分片
    def _chunk_by_sheet(
        self,
        sheets: List[Tuple[str, str]],
        source_file: str,
        pdf_path: str,
        format_chunk_size: int = 0,
    ) -> Tuple[List[str], List[int], List[Dict[str, Any]]]:
        all_texts: List[str] = []
        all_page_numbers: List[int] = []
        all_metadatas: List[Dict[str, Any]] = []
        global_chunk_idx = 0

        for sheet_name, sheet_content in sheets:
            # Sheet 名哈希为整数页码，确保向量数据库兼容（page_number 字段要求 int）
            page_num = int(hashlib.md5(sheet_name.encode()).hexdigest(), 16) % 100000

            # 对当前 Sheet 内容进行语义分片
            chunks, metas = self._split_page_content(
                sheet_content, page_num, source_file, pdf_path, global_chunk_idx, format_chunk_size
            )

            # 为每个分片的元数据追加 Sheet 名称信息
            for meta in metas:
                meta["sheet_name"] = sheet_name

            all_texts.extend(chunks)
            all_page_numbers.extend([page_num] * len(chunks))
            all_metadatas.extend(metas)
            global_chunk_idx += len(chunks)

        logger.info(
            f"Chunked '{source_file}': {len(sheets)} sheets -> {len(all_texts)} chunks"
        )
        return all_texts, all_page_numbers, all_metadatas

# 创建全局单例，供其他模块直接导入使用
chunking_service = ChunkingService()