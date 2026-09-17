import logging
from typing import List, Dict, Any, Optional
import numpy as np
# BM25 关键词检索：对日文文本使用形态素解析 + 字符 2-gram 混合分词
from rank_bm25 import BM25Okapi
# Cross-Encoder 重排序：使用 ONNX Runtime 加载 BGE Reranker v2 M3 模型
from services.onnx_reranker import OnnxReranker
from config.settings import settings

logger = logging.getLogger(__name__)

# 混合检索默认返回条数（精排后送入 LLM 的最终结果数）
DEFAULT_SEARCH_TOP_K = settings.default_search_top_k
# 粗排阶段每路检索取 top N 条候选（广撒网，避免截断跨页相关内容）
COARSE_TOP_K = settings.coarse_top_k
# 分数跌断策略参数
SCORE_DROPOFF_VARIANCE_THRESHOLD = 0.001
# 动态最低分比例：当前分数低于 (最高分 × 此比例) 时，视为不相关，直接截断
SCORE_DROPOFF_FLOOR_RATIO = 0.25
# 相对跌断阈值（比值法）：当前分数低于前一个分数的 drop_ratio 倍时，视为相关性断崖
SCORE_DROPOFF_RATIO = 0.5
# 绝对跌断阈值（差值法）：相邻分数差超过此值时视为断崖
SCORE_DROPOFF_ABS = 0.05

class JapaneseTokenizer:
    # 日文混合分词器：形态素解析（Janome） + 字符 2-gram

    def __init__(self):
        self._has_janome: bool = False
        self._tokenizer: Any = None
        self._initialized: bool = False

    def _ensure_janome(self):
        if self._initialized:
            return
        try:
            from janome.tokenizer import Tokenizer
            self._tokenizer = Tokenizer()
            self._has_janome = True
        except ImportError:
            pass
        self._initialized = True

    def tokenize(self, text: str) -> List[str]:
        # 混合分词策略：形态素解析（主） + 字符 2-gram（兜底未登录词）
        #
        # 第1层【形态素解析】：使用 Janome 将日文切分为有意义的词语
        #   - 提取每个形态素的「原形」作为 token，解决活用形匹配问题
        #   - 同时保留「表层形」，保留复合词中的完整表达
        #   - 例：「走った」→ 原形「走る」+ 表层形「走っ」「た」
        #
        # 第2层【字符 2-gram】：对 CJK 连续字符生成 2-gram
        #   - 兜底未登录词（専門用語、略語、社内用語）
        #   - 与形态素 token 混合，BM25 自动加权高频匹配
        #
        self._ensure_janome()
        tokens: List[str] = []

        if self._has_janome:
            for token in self._tokenizer.tokenize(text):
                surface = token.surface
                base = token.base_form
                pos = token.part_of_speech.split(",")[0]

                # 过滤助词、助动词、符号（对 BM25 检索几乎无贡献）
                if pos in ("助詞", "助動詞", "記号"):
                    continue

                if surface.strip():
                    tokens.append(surface)

                # 原形 ≠ 表层形 → 发生了活用，同时加入原形
                if base != surface and base.strip():
                    tokens.append(base)

        # 字符 2-gram（兜底未登录词，同时也是 Janome 不可用时的 fallback）
        cjk_buf: str = ""
        for ch in text:
            if "\u4e00" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff" or "\uff00" <= ch <= "\uffef":
                cjk_buf += ch
            else:
                if cjk_buf:
                    for i in range(len(cjk_buf) - 1):
                        tokens.append(cjk_buf[i : i + 2])
                    tokens.extend(list(cjk_buf))
                    cjk_buf = ""
                if ch.strip():
                    tokens.append(ch.lower())

        if cjk_buf:
            for i in range(len(cjk_buf) - 1):
                tokens.append(cjk_buf[i : i + 2])
            tokens.extend(list(cjk_buf))

        return [t for t in tokens if t.strip()]


# 模块级单例分词器，供 build_bm25_index / bm25_search 共用
_japanese_tokenizer = JapaneseTokenizer()


def japanese_tokenizer(text: str) -> List[str]:
    # 兼容旧接口：委托给 JapaneseTokenizer 单例
    return _japanese_tokenizer.tokenize(text)


class HybridRetrieverService:

    def __init__(self):
        # 向量数据库服务：懒加载，避免循环依赖
        self._vector_db_service: Any = None
        # BM25 索引实例，首次调用 _ensure_bm25_index 时构建
        self._bm25: Optional[BM25Okapi] = None
        # BM25 索引对应的文档元数据（id, text, source_file, page_number 等）
        self._bm25_chunks: List[Dict[str, Any]] = []
        # ONNX 重排序模型，首次调用时加载（替代 CrossEncoder，无 PyTorch/shm.dll 依赖）
        self._reranker: Optional[OnnxReranker] = None
        # 标记 BM25 索引是否需要重建（文档增删后置为 True）
        self._bm25_dirty: bool = True

    def _get_vector_db_service(self):
        if self._vector_db_service is None:
            from services.vector_db_service import vector_db_service
            self._vector_db_service = vector_db_service
        return self._vector_db_service

    def _get_reranker(self) -> OnnxReranker:
        # 获取 ONNX 重排序模型（懒加载，首次调用时初始化）
        if self._reranker is None:
            model_name = settings.reranker_model_name
            logger.info(f"正在加载 ONNX Reranker 模型: {model_name}")
            self._reranker = OnnxReranker(model_name)
            logger.info("ONNX Reranker 模型加载成功")
        return self._reranker

    def _ensure_bm25_index(self):
        # 确保 BM25 索引已构建且未过期
        # 如果索引已存在且未标记为脏，直接返回
        if self._bm25 is not None and not self._bm25_dirty:
            return
        # 从向量数据库拉取全量文档片段，构建 BM25 索引
        try:
            all_chunks = self._get_vector_db_service().get_all_chunks()
            if all_chunks:
                self.build_bm25_index(all_chunks)
            else:
                # 向量库为空，标记索引已尝试构建，避免无限重试
                self._bm25_dirty = False
        except Exception as e:
            logger.error(f"BM25 索引构建失败: {e}")
            # 标记已尝试构建，避免后续请求无限重试
            self._bm25_dirty = False

    def refresh_bm25_index(self):
        # 重建 BM25 索引（文档增删后调用）
        self.mark_dirty()
        self._ensure_bm25_index()

    def search(self, query: str) -> List[Dict[str, Any]]:
        # 步骤 1：Dense 向量检索（粗排，取较多候选）
        dense_results: List[Dict[str, Any]] = []
        try:
            dense_results = self._get_vector_db_service().search(
                query, top_k=COARSE_TOP_K, metric="cosine"
            ).get("results", [])
        except Exception as e:
            logger.error(f"向量检索失败: {e}")

        # 步骤 2：混合检索（Dense + BM25 → 去重 → Rerank）
        return self._hybrid_search(
            query=query,
            dense_results=dense_results,
        )

    def build_bm25_index(self, chunks: List[Dict[str, Any]]):
        # 从所有文档片段构建 BM25 索引
        # chunks: LanceDB 全表数据，每条包含 id, text, source_file, page_number 等字段
        if not chunks:
            logger.warning("没有可用于构建 BM25 索引的文档片段")
            return

        # 提取所有文本，使用日文分词器
        texts = [chunk.get("text", "") for chunk in chunks]
        tokenized_texts = [japanese_tokenizer(t) for t in texts]

        # 构建 BM25 索引
        self._bm25 = BM25Okapi(tokenized_texts)
        # 保存元数据，BM25 返回的索引号对应此列表
        self._bm25_chunks = chunks
        self._bm25_dirty = False
        logger.info(f"BM25 索引构建完成，共 {len(chunks)} 条文档")

    def bm25_search(self, query: str, top_k: int = COARSE_TOP_K) -> List[Dict[str, Any]]:
        # BM25 关键词检索
        # 返回 top_k 条结果，每条包含 BM25 分数
        self._ensure_bm25_index()
        if self._bm25 is None:
            logger.warning("BM25 索引尚未就绪")
            return []

        # 对查询文本使用相同的分词器
        tokenized_query = japanese_tokenizer(query)
        # 获取 BM25 分数和索引
        scores = self._bm25.get_scores(tokenized_query)
        # 取分数最高的 top_k 个索引
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        # 按 ID 去重：防止同一 chunk 被重复入库后返回多条相同结果
        seen_ids = set()
        for idx in top_indices:
            if scores[idx] <= 0:
                # BM25 分数为 0 表示完全不匹配，跳过
                continue
            chunk = self._bm25_chunks[idx]
            chunk_id = chunk.get("id", "")
            # 跳过已出现过的 chunk ID
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            results.append({
                "id": chunk.get("id", ""),
                "text": chunk.get("text", ""),
                "source_file": chunk.get("source_file", ""),
                "page_number": chunk.get("page_number", -1),
                "chunk_index": chunk.get("chunk_index", -1),
                "metadata": chunk.get("metadata", {}),
                "bm25_score": float(scores[idx]),
                "retrieval_type": "bm25",
            })
        return results

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        # 使用 ONNX Reranker 对候选集重排序（替代 Cross-Encoder）
        if not candidates:
            return []

        reranker = self._get_reranker()

        # 构造 (query, text) 对，作为 Reranker 的输入
        pairs = [(query, c["text"]) for c in candidates]

        # ONNX Reranker 推理（CPU 推理，batch_size 控制内存占用）
        scores = reranker.predict(
            pairs,
            show_progress_bar=False,
            batch_size=4,
        )

        # 将分数附加到候选结果的副本中，不修改原数据
        scored = []
        for i, candidate in enumerate(candidates):
            item = dict(candidate)
            item["rerank_score"] = float(scores[i])
            scored.append(item)

        # 按 rerank_score 降序排列
        ranked = sorted(scored, key=lambda x: x.get("rerank_score", 0), reverse=True)
        # 分数跌断策略：自动检测相关性断崖，截断低相关结果
        return self._apply_score_dropoff(ranked)

    def _apply_score_dropoff(
        self,
        ranked: List[Dict[str, Any]],
        drop_ratio: float = SCORE_DROPOFF_RATIO,
        drop_abs: float = SCORE_DROPOFF_ABS,
        floor_ratio: float = SCORE_DROPOFF_FLOOR_RATIO,
    ) -> List[Dict[str, Any]]:
        if not ranked:
            return []

        top_score = ranked[0].get("rerank_score", 0)
        last_score = ranked[-1].get("rerank_score", 0)
        score_range = top_score - last_score

        # 第0层：方差检测 —— reranker 无法区分文档时，降级到原始分数
        if score_range < SCORE_DROPOFF_VARIANCE_THRESHOLD:
            logger.info(
                f"分数跌断（方差检测）：reranker 无法区分文档相关性，"
                f"分数跨度={score_range:.6f} < {SCORE_DROPOFF_VARIANCE_THRESHOLD}，"
                f"降级回退到原始粗排分数"
            )
            return self._fallback_to_raw_scores(ranked)

        # 动态最低分 = 最高分 × 比例，自适应不同分数段
        dynamic_floor = top_score * floor_ratio
        cutoff = len(ranked)

        for i in range(1, len(ranked)):
            prev_score = ranked[i - 1].get("rerank_score", 0)
            curr_score = ranked[i].get("rerank_score", 0)

            # 第1层：动态最低分 —— 当前 chunk 分数低于最高分的 floor_ratio 倍
            if curr_score < dynamic_floor:
                cutoff = i
                logger.info(
                    f"分数跌断（动态最低分）：在第 {i} 位截断，"
                    f"当前分={curr_score:.4f} < 阈值={dynamic_floor:.4f} "
                    f"（最高分={top_score:.4f} × {floor_ratio}）"
                )
                break

            # 第2层：相对跌断（比值法）—— 相邻分数骤降超过一半
            if prev_score > 0 and curr_score / prev_score < drop_ratio:
                cutoff = i
                logger.info(
                    f"分数跌断（比值法）：在第 {i} 位截断，"
                    f"前={prev_score:.4f} → 当前={curr_score:.4f}，"
                    f"比值={curr_score / prev_score:.4f} < {drop_ratio}"
                )
                break

            # 第3层：绝对跌断（差值法）—— 相邻分数差超过阈值
            if prev_score - curr_score > drop_abs:
                cutoff = i
                logger.info(
                    f"分数跌断（差值法）：在第 {i} 位截断，"
                    f"前={prev_score:.4f} → 当前={curr_score:.4f}，"
                    f"差值={prev_score - curr_score:.4f} > {drop_abs}"
                )
                break

        # 第4层：硬上限兜底
        result = ranked[:min(cutoff, DEFAULT_SEARCH_TOP_K)]
        logger.info(
            f"分数跌断结果：{len(ranked)} 条候选 → "
            f"保留 {len(result)} 条（截断位={cutoff}，最高分={top_score:.4f}，"
            f"阈值={dynamic_floor:.4f}，上限={DEFAULT_SEARCH_TOP_K}）"
        )
        return result

    # 降级回退：当 reranker 无法区分文档时，使用原始检索信号重新排序
    def _fallback_to_raw_scores(
        self,
        ranked: List[Dict[str, Any]],
        top_k: int = DEFAULT_SEARCH_TOP_K,
    ) -> List[Dict[str, Any]]:

        bm25_vals = []
        dense_vals = []
        for r in ranked:
            bm25 = r.get("bm25_score")
            if bm25 is not None:
                bm25_vals.append(bm25)

            dist = r.get("distance")
            if dist is not None:
                dense_vals.append(1.0 - dist)

        # BM25 归一化参数
        bm25_min = min(bm25_vals) if bm25_vals else 0.0
        bm25_max = max(bm25_vals) if bm25_vals else 1.0
        bm25_range = (bm25_max - bm25_min) if bm25_max > bm25_min else 1.0

        # Dense 相似度归一化参数
        dense_min = min(dense_vals) if dense_vals else 0.0
        dense_max = max(dense_vals) if dense_vals else 1.0
        dense_range = (dense_max - dense_min) if dense_max > dense_min else 1.0

        for i, r in enumerate(ranked):
            combined = 0.0
            count = 0

            bm25 = r.get("bm25_score")
            if bm25 is not None:
                combined += (bm25 - bm25_min) / bm25_range
                count += 1

            dist = r.get("distance")
            if dist is not None:
                sim = 1.0 - dist
                combined += (sim - dense_min) / dense_range
                count += 1

            new_r = dict(r)
            new_r["fallback_score"] = combined / count if count > 0 else 0.0
            ranked[i] = new_r

        ranked.sort(key=lambda x: x.get("fallback_score", 0), reverse=True)
        result = ranked[:top_k]

        logger.info(
            f"降级回退到原始粗排分数：{len(ranked)} 条候选 → "
            f"保留 {len(result)} 条，"
            f"BM25范围=[{bm25_min:.2f}, {bm25_max:.2f}]，"
            f"向量范围=[{dense_min:.4f}, {dense_max:.4f}]"
        )
        return result

    def _hybrid_search(
        self,
        query: str,
        dense_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        # 混合检索主流程：Dense + BM25 → 去重 → Rerank (ONNX) → 分数跌断截断

        # 步骤 1：BM25 检索（粗排，取较多候选）
        bm25_results = self.bm25_search(query)

        # 步骤 2：合并去重（按 id 去重，同一条可能在两种检索中都出现）
        seen_ids = set()
        all_candidates = []

        # 先加入向量检索结果（Dense 优先标记）
        for r in dense_results:
            if r.get("id") not in seen_ids:
                r_copy = r.copy()
                r_copy["retrieval_type"] = "dense"
                all_candidates.append(r_copy)
                seen_ids.add(r.get("id"))

        # 再加入 BM25 结果
        for r in bm25_results:
            if r.get("id") not in seen_ids:
                all_candidates.append(r.copy())
                seen_ids.add(r.get("id"))

        logger.info(
            f"混合检索：向量={len(dense_results)} 条，"
            f"BM25={len(bm25_results)} 条，合并去重={len(all_candidates)} 条"
        )

        # 步骤 3：如果候选集 <= 1，不需要重排序，补全字段以保持返回结果一致性
        if len(all_candidates) <= 1:
            for c in all_candidates:
                c["rerank_score"] = 1.0
            return all_candidates

        # 步骤 4：Cross-Encoder 重排序 + 分数跌断截断
        return self.rerank(query, all_candidates)

    def mark_dirty(self):
        # 标记 BM25 索引为脏，下次检索前需要重建
        self._bm25_dirty = True
        logger.info("BM25 索引已标记为脏，下次检索前将重建")


# 模块级单例：供 ai_chat_service 和 file_upload_router 直接使用
hybrid_retriever = HybridRetrieverService()