import logging
from typing import List, Dict, Any, Generator, Optional
import json
from services.llm_service import LLMService
from services.hybrid_retriever_service import hybrid_retriever
from services.query_preprocessor import query_preprocessor

# 模块级日志记录器
logger = logging.getLogger(__name__)

class AIChatService:

    STOP_SEQUENCES = ["\n\n\n", "質問:", "参考文書:"]
    MAX_CONTEXT_CHARS = 6000

    def __init__(self):
        # LLM 服务实例：负责生成自然语言回答
        self._llm_service = LLMService()
        # 混合检索器：向量检索 + BM25 + Cross-Encoder 重排序
        self._hybrid_retriever = hybrid_retriever

    @staticmethod
    def _normalize_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
        meta = chunk.get("metadata", {})
        return {
            "source_file": chunk.get("source_file", "不明"),
            "page_number": chunk.get("page_number", -1),
            "sheet_name": meta.get("sheet_name", ""),
            "text": chunk.get("text", ""),
            "rerank_score": chunk.get("rerank_score", 0),
        }

    def process_question_stream(
        self, question: str, history: Optional[List[Dict[str, str]]] = None
    ) -> Generator[str, None, None]:

        # 输入校验：空问题直接返回错误 SSE 事件
        if not question or not question.strip():
            logger.warning("收到空问题，跳过处理（流式）")
            yield self._sse_event({
                "error": "質問が空です。質問を入力してください。"
            })
            return

        # ================================================================
        # 步骤 1：LLM 预处理（指代消解 + 话题检测）
        # ================================================================
        search_question = question  # 默认用原始问题检索
        if history:
            try:
                preprocessed = query_preprocessor.process(question, history)
                search_question = preprocessed.get("resolved_question", question)

                # 话题切换：软信号，前端显示分隔线（不清空历史）
                if preprocessed.get("topic_changed", False):
                    yield self._sse_event({"topic_changed": True})

                # 发送标准化问题给前端（透明性，可选展示「AI 理解した質問」）
                yield self._sse_event({"resolved_question": search_question})

                logger.info(
                    f"预处理: {question} → {search_question}, "
                    f"topic_changed={preprocessed.get('topic_changed', False)}"
                )
            except Exception as e:
                logger.error(f"预处理失败，使用原始问题: {e}")
                # 回退：用原始问题检索，不阻塞流程

        # ================================================================
        # 步骤 2：检索（用标准化问题，不用原始问题）
        # ================================================================
        try:
            reranked = self._hybrid_retriever.search(search_question)
        except Exception as e:
            logger.error(f"检索失败（流式）: {e}")
            yield self._sse_event({
                "error": "申し訳ありません。文書検索中にエラーが発生しました。"
            })
            return

        # ================================================================
        # 步骤 3：构建 Prompt（含最近 5 轮历史 + 检索 chunk）
        # ================================================================
        messages = self._build_prompt(question, reranked, history)

        # ================================================================
        # 步骤 4：流式调用 LLM 生成回答
        # ================================================================
        try:
            for token in self._llm_service.chat_stream(
                messages,
                temperature=0.1,
                max_tokens=2048,
                num_ctx=8192,
                stop=self.STOP_SEQUENCES,
            ):
                # 每个 token 包装为 SSE 事件: data: {"token":"..."}\n\n
                yield self._sse_event({"token": token})
        except Exception as e:
            # LLM 流式调用失败：发送错误事件
            logger.error(f"LLM 流式调用失败: {e}")
            yield self._sse_event({
                "error": "申し訳ありません。LLM サービスに接続できません。Ollama が起動しているか確認してください。"
            })
            return

        # ================================================================
        # 步骤 5：推送证据（answer 生成完毕后，发送 evidence）
        # ================================================================
        final_evidence = self._extract_evidence(reranked)
        yield self._sse_event({"evidence": final_evidence})

    def _build_prompt(
        self, question: str, reranked: List[Dict[str, Any]],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:

        # ================================================================
        # 多轮对话：构建历史上下文区域
        # ================================================================
        history_section = ""
        if history:
            recent = history[-5:]
            recent = [
                t for t in recent
                if t.get("question") or t.get("answer")
            ]
            history_lines = []
            for i, turn in enumerate(recent, 1):
                answer_preview = turn.get("answer", "")[:200]
                history_lines.append(
                    f"[{i}] ユーザー: {turn.get('question', '')}\n"
                    f"[{i}] 回答: {answer_preview}"
                )
            history_text = "\n".join(history_lines)
            history_section = (
                f"=== 対話履歴（直近{len(recent)}ターン） ===\n"
                f"{history_text}\n"
                f"=== 履歴終了 ===\n\n"
            )    
        if reranked:
            # 有检索结果：将文档片段作为上下文喂给 LLM
            context_parts = []
            total_chars = 0
            for i, chunk in enumerate(reranked):
                c = self._normalize_chunk(chunk)
                if c["sheet_name"]:
                    # XLSX 文件：显示 Sheet 名称而非无意义的哈希页码
                    source_info = f"{c['source_file']} (Sheet: {c['sheet_name']})"
                else:
                    # PDF/DOCX/Image 等：显示 Page 页码
                    page = c["page_number"]
                    source_info = f"{c['source_file']}" + (f" (Page {page})" if page > 0 else "")
                text = c["text"][:1000]
                if total_chars + len(text) > self.MAX_CONTEXT_CHARS:
                    break
                context_parts.append(f"[文書{i + 1}] ({source_info})\n{text}")
                total_chars += len(text)
            context = "\n\n".join(context_parts)

            system_message = (
                "あなたは社内文書検索アシスタントです。以下のルールを厳守してください：\n"
                "1. 回答には「参考文書」に書かれている内容だけを使ってください。\n"
                "2. あなたの事前知識や一般常識は一切使わないでください。\n"
                "3. 参考文書にない情報を推測・追加・補完しないでください。\n"
                "4. 参考文書から該当情報が見つからない場合は「文書内に該当情報がありません。」とだけ回答してください。\n"
                "5. 回答は日本語で、簡潔な自然文でまとめてください。箇条書きや見出しは使わないでください。"
            )
            user_message = (
                f"{history_section}"
                "===== 参考文書エリア =====\n"
                f"{context}\n"
                "===== 参考文書エリア 終了 =====\n\n"
                f"質問: {question}\n\n"
                "上の参考文書の内容に基づいて回答してください。\n"
                "回答:"
            )
        else:
            # 无检索结果：告知用户当前知识库中未找到相关文档
            system_message = (
                "あなたは社内文書検索アシスタントです。"
                "現在、参考文書が見つかっていないため、"
                "「文書内に該当情報がありません。」と回答してください。"
            )
            user_message = (
                f"{history_section}"
                f"質問: {question}\n\n"
                f"回答:"
            )

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    def _extract_evidence(
        self, reranked: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        if not reranked:
            return []
        result = []
        for r in reranked:
            c = self._normalize_chunk(r)
            ev = {
                "source_file": c["source_file"],
                "page_number": c["page_number"],
                "text": c["text"],
                "rerank_score": c["rerank_score"],
            }
            if c["sheet_name"]:
                ev["sheet_name"] = c["sheet_name"]
            result.append(ev)
        return result
    
    @staticmethod
    def _sse_event(data: dict) -> str:
        # 将 dict 转换为标准 SSE 格式: data: {json}\n\n
        # SSE 协议要求每条消息以 "data: " 开头，以 "\n\n" 结尾
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

# 全局单例实例：供路由层直接调用 process_question_stream
ai_chat_service = AIChatService()