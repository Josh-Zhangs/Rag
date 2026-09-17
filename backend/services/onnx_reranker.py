import logging
import os
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class OnnxReranker:

    # ONNX 模型文件名（固定名称，由模型导出时确定）
    ONNX_MODEL_FILE = "model.onnx"
    # Tokenizer 分词器文件名（HuggingFace tokenizers 库的 JSON 格式）
    TOKENIZER_FILE = "tokenizer.json"

    def __init__(self, model_path: str):

        # 延迟导入 onnxruntime，避免在不需要 reranker 的场景下强制加载依赖
        import onnxruntime as ort

        # 拼接模型文件路径和分词器文件路径
        onnx_model = os.path.join(model_path, self.ONNX_MODEL_FILE)
        tokenizer_file = os.path.join(model_path, self.TOKENIZER_FILE)

        # 校验 ONNX 模型文件是否存在
        if not os.path.exists(onnx_model):
            raise FileNotFoundError(
                f"ONNX モデルファイルが見つかりません: {onnx_model}\n"
                f"BGE Reranker v2 M3 の ONNX モデルをこのパスに配置してください"
            )

        # 校验分词器文件是否存在
        if not os.path.exists(tokenizer_file):
            raise FileNotFoundError(f"Tokenizer ファイルが見つかりません: {tokenizer_file}")

        # 创建 ONNX Runtime 推理会话
        # 使用 CPUExecutionProvider 确保在无 GPU 的机器上也能运行，打包部署兼容性最好
        self._session = ort.InferenceSession(
            onnx_model,
            providers=["CPUExecutionProvider"],
        )

        # 加载 HuggingFace tokenizers 分词器
        from tokenizers import Tokenizer
        self._tokenizer = Tokenizer.from_file(tokenizer_file)

        # 启用自动 padding，使同一 batch 内的序列自动对齐到该 batch 中最长的序列长度
        self._tokenizer.enable_padding()

        # 缓存 ONNX 模型的输入/输出张量名称，避免每次推理时重复查询
        # 输入名称通常为 ['input_ids', 'attention_mask']
        # 输出名称通常为 ['logits']（形状为 [batch_size, 1]）
        self._input_names: List[str] = [inp.name for inp in self._session.get_inputs()]
        self._output_names: List[str] = [out.name for out in self._session.get_outputs()]

        logger.info(
            f"OnnxReranker 加载完成: 模型路径={onnx_model}, "
            f"输入张量={self._input_names}, 输出张量={self._output_names}"
        )

    def predict(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 4,
        show_progress_bar: bool = False,
    ) -> List[float]:

        # 参数校验：batch_size 必须为正整数
        if batch_size <= 0:
            raise ValueError(f"batch_size は正の整数である必要があります。現在の値: {batch_size}")

        # 空输入直接返回空列表，避免后续处理异常
        if not pairs:
            return []

        all_scores: List[float] = []

        # 按 batch_size 分批处理，避免一次性加载过多数据导致内存溢出
        for batch_start in range(0, len(pairs), batch_size):
            batch = pairs[batch_start:batch_start + batch_size]

            # 分词编码：将 query 和 document 拼接为 [CLS] query [SEP] document [SEP] 格式
            # encode_batch 接受 [(text, text_pair), ...] 格式，自动完成 pair 编码
            # 编码结果包含 input_ids、attention_mask、token_type_ids 等字段
            encodings = self._tokenizer.encode_batch(batch)

            # 从编码结果中提取所有可能的输入张量，转换为 numpy 数组
            # dtype=np.int64 是 ONNX 模型要求的输入数据类型
            encoding_tensors = {
                "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
                "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
            }

            # 根据模型实际需要的输入名称，动态构建 feed_dict
            # 这样无论模型是否需要 token_type_ids 都能正确适配
            feed_dict = {name: encoding_tensors[name] for name in self._input_names}

            # 执行 ONNX Runtime 推理
            outputs = self._session.run(self._output_names, feed_dict)

            # 模型输出 logits 的形状为 (batch_size, 1)，使用 flatten() 展开为一维列表
            batch_scores = outputs[0].flatten().tolist()
            all_scores.extend(batch_scores)

        # Sigmoid 归一化：将原始 logits 映射到 [0, 1] 区间
        # 公式: score = 1 / (1 + e^(-logit))
        # 这样处理后的分数与 sentence-transformers CrossEncoder 的输出范围保持一致
        all_scores = [1.0 / (1.0 + np.exp(-s)) for s in all_scores]

        return all_scores