# Pydantic 数据校验和序列化
from pydantic import BaseModel
# 类型注解
from typing import List


class EvidenceItem(BaseModel):
    """单条检索证据：文档原文片段及其元数据"""
    # 来源文件名（如 000074366.pdf）
    source_file: str
    # PDF 页码（从 1 开始，-1 表示未知）
    page_number: int
    # 原始文档片段文本
    text: str
    # Cross-Encoder 重排序分数（0~1，越高越相关）
    rerank_score: float


class ConversationTurn(BaseModel):
    """多轮对话中的一轮：用户问题 + AI 回答

    前端管理会话历史，每次请求携带最近 N 轮的 Q&A 对。
    后端无状态，不存储会话历史。
    """
    # 用户的问题
    question: str
    # AI 的回答
    answer: str


class AIChatRequest(BaseModel):
    """前端发送的 AI 对话请求"""
    # 用户输入的问题
    question: str
    # 使用的模型来源（默认 Ollama）
    source: str = 'ollama'
    # 多轮对话历史（前端管理，最多 10 轮），首轮时为空列表
    history: List[ConversationTurn] = []


class AIChatResponse(BaseModel):
    """后端返回的 AI 对话响应"""
    # LLM 生成的回答文本
    answer: str
    # 检索到的原文片段列表（含文件名、页码、重排序分数）
    evidence: List[EvidenceItem]