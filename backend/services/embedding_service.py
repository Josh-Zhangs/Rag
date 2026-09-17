import math
import time
import logging
import numpy as np
import requests
from typing import List, Optional
from requests.exceptions import RequestException, Timeout, ConnectionError
from config.settings import settings
logger = logging.getLogger(__name__)


class EmbeddingService:

    def __init__(self):
        # HTTP 长连接会话，延迟初始化，首次调用时创建
        self._session: Optional[requests.Session] = None
        # Ollama 批量嵌入接口地址（/api/embed 支持列表输入）
        self._embed_url: str = f"{settings.ollama_base_url}/api/embed"
        # 嵌入模型名称
        self._model: str = settings.embedding_model_name
        # 向量维度，延迟探测，首次调用 encode 时通过接口获取
        self._vector_dim: Optional[int] = None

    # ================================================================
    # 公开接口
    # ================================================================

    # 获取向量维度
    @property
    def vector_dim(self) -> int:
        if self._vector_dim is None:
            self._probe_dim()
        return self._vector_dim

    # 批量向量化文档段落
    def encode_passages(self, texts: List[str]) -> List[List[float]]:
        return self._encode_batch(texts, prefix="passage: ")

    # 向量化查询
    def encode_query(self, query: str) -> List[float]:
        # 调用批量方法，传入单元素列表
        results = self._encode_batch([query], prefix="query: ")
        # 防御：如果 embedding 全部失败，抛出明确错误而非 IndexError
        if not results:
            raise RuntimeError(
                "クエリのベクトル化に失敗しました：埋め込みベクトルを取得できません。Ollama サービスの状態を確認してください"
            )
        # 取第一个（也是唯一一个）结果
        return results[0]

    # ================================================================
    # 内部实现
    # ================================================================

    # 获取 HTTP 会话
    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
        return self._session

    # 探测向量维度
    def _probe_dim(self):
        # 发送一条短文本，获取嵌入向量
        results = self._encode_batch(["dim probe"], prefix="passage: ")
        # 防御：如果 embedding 全部失败，抛出明确错误而非 IndexError
        if not results:
            raise RuntimeError(
                "ベクトル次元の検出に失敗しました：埋め込みベクトルを取得できません。Ollama サービスの状態を確認してください"
            )
        # 从第一条向量的长度获取维度数
        self._vector_dim = len(results[0])
        logger.info(f"Embedding model ready via Ollama, vector dim: {self._vector_dim}")

    # 批量向量化核心方法
    def _encode_batch(
        self,
        texts: List[str],
        prefix: str = "",
    ) -> List[List[float]]:

        # 单批最大分片数，平衡速度与 Ollama 负载
        batch_size = 20
        # 单次请求超时秒数（批量请求比单条慢，给 60s）
        req_timeout = 60
        # 最大重试次数（不含首次，即最多尝试 3 次）
        max_retry = 2
        # 获取复用 HTTP 会话
        session = self._get_session()

        # 计算总批次数（向上取整，确保最后一批不足 batch_size 也能处理）
        total_batches = math.ceil(len(texts) / batch_size)
        # 存储所有批次的向量结果
        all_vectors: List[List[float]] = []

        # 逐批处理
        for batch_idx in range(total_batches):
            # 切片获取当前批次文本
            start = batch_idx * batch_size
            end = start + batch_size
            batch_texts = texts[start:end]

            # 构建带前缀的输入列表（E5 模型要求区分 passage 和 query）
            batch_inputs = [f"{prefix}{t}" for t in batch_texts]

            # 重试计数器
            retry = 0
            # 当前批次是否成功
            success = False
            # 当前批次的有效向量列表
            batch_vecs: List[List[float]] = []

            # 指数退避重试循环
            while retry <= max_retry and not success:
                try:
                    # 调用 Ollama /api/embed 批量接口
                    # input 字段传列表，返回 embeddings 字段（复数）
                    resp = session.post(
                        url=self._embed_url,
                        json={"model": self._model, "input": batch_inputs},
                        timeout=req_timeout,
                    )
                    # 检查 HTTP 状态码，4xx/5xx 抛出异常
                    resp.raise_for_status()
                    # 解析返回的嵌入向量列表，顺序与 input 完全对应
                    raw_vecs = resp.json()["embeddings"]

                    # 逐条向量进行 L2 归一化
                    for j, vec in enumerate(raw_vecs):
                        # 转为 numpy 数组，使用 float32 节省内存
                        vec_np = np.array(vec, dtype=np.float32)
                        # 计算 L2 范数（欧几里得长度）
                        l2 = np.linalg.norm(vec_np)
                        # 零向量通常意味着输入文本无意义，保留原向量并警告
                        if l2 < 1e-9:
                            logger.warning(
                                f"零向量分片，保留原向量: "
                                f"'{batch_texts[j][:80]}...'"
                            )
                            # 零向量直接保留原始值，不做归一化
                            batch_vecs.append(vec_np.tolist())
                        else:
                            # L2 归一化后转为 Python list 存入结果
                            batch_vecs.append((vec_np / l2).tolist())

                    # 标记当前批次成功
                    success = True
                    logger.debug(
                        f"第{batch_idx + 1}/{total_batches}批嵌入完成，"
                        f"向量数: {len(batch_vecs)}"
                    )

                # 场景1：Ollama 服务未启动或端口无法连接
                except ConnectionError:
                    retry += 1
                    if retry > max_retry:
                        # 重试耗尽，ConnectionError 是致命错误，直接向上抛出
                        logger.error(
                            "Ollama 服务未启动或端口无法连接，"
                            "请先打开 Ollama 客户端"
                        )
                        raise ConnectionError(
                            "Ollama サービスが起動していません。埋め込みベクトルの生成に失敗しました"
                        )
                    # 指数退避：第1次等2s，第2次等4s
                    wait = retry * 2
                    logger.warning(f"连接 Ollama 失败，{wait}s 后第{retry}次重试")
                    time.sleep(wait)

                # 场景2：请求超时（Ollama 处理过慢或网络卡顿）
                except Timeout:
                    retry += 1
                    if retry > max_retry:
                        # 超时重试耗尽，用零向量占位，保证输入输出数量对齐
                        logger.error(
                            f"第{batch_idx + 1}批请求超时达到最大重试次数，"
                            f"本批 {len(batch_texts)} 条以零向量占位"
                        )
                        batch_vecs = self._make_fallback_vectors(len(batch_texts))
                        success = True
                        break
                    # 超时后等待更长时间，给 Ollama 恢复机会
                    wait = retry * 2
                    logger.warning(f"请求超时，{wait}s 后重试")
                    time.sleep(wait)

                # 场景3：HTTP 4xx/5xx、接口返回异常
                except RequestException as e:
                    retry += 1
                    if retry > max_retry:
                        # HTTP 异常重试耗尽，用零向量占位，保证输入输出数量对齐
                        logger.error(
                            f"批次嵌入接口异常，"
                            f"本批 {len(batch_texts)} 条以零向量占位: {e}"
                        )
                        batch_vecs = self._make_fallback_vectors(len(batch_texts))
                        success = True
                        break
                    logger.warning(f"接口请求异常，重试{retry}次: {e}")
                    time.sleep(retry * 2)

                # 场景4：JSON 解析失败、返回格式异常（不重试，数据错误重试无意义）
                except (KeyError, ValueError) as e:
                    logger.error(
                        f"Ollama 返回数据格式异常，"
                        f"本批 {len(batch_texts)} 条以零向量占位: {e}"
                    )
                    batch_vecs = self._make_fallback_vectors(len(batch_texts))
                    success = True
                    break

                # 场景5：通用兜底异常，打印完整堆栈便于排查
                except Exception as e:
                    logger.error(f"嵌入处理未知异常: {e}", exc_info=True)
                    batch_vecs = self._make_fallback_vectors(len(batch_texts))
                    success = True
                    break

            # 将当前批次的有效向量合并到总结果
            all_vectors.extend(batch_vecs)

        # 记录最终处理结果（包含原始数与有效数对比，便于发现批次丢失）
        logger.info(
            f"分片嵌入完成，原始{len(texts)}条，"
            f"有效向量{len(all_vectors)}条"
        )
        return all_vectors

    def _make_fallback_vectors(self, count: int) -> List[List[float]]:
        """生成零向量占位，保证输入输出数量一致"""
        dim = self._vector_dim if self._vector_dim else 384
        return [[0.0] * dim for _ in range(count)]


# 全局单例：供其他模块直接导入使用
embedding_service = EmbeddingService()