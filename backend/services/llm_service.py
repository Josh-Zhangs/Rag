import requests
import logging
from typing import List, Dict, Optional, Generator
import time
import json
from config.settings import settings

# 模块级日志记录器
logger = logging.getLogger(__name__)


class LLMService:

    def __init__(self, model: Optional[str] = None):
        # Ollama 服务地址（如 http://localhost:11434）
        self.base_url = settings.ollama_base_url
        # 模型名称：优先使用传入的 model 参数，未指定时回退到全局默认模型
        self.model = model if model else settings.ollama_model
        # OpenAI 兼容的 Chat Completions 端点
        self._chat_url = f"{self.base_url}/v1/chat/completions"
        # HTTP 请求超时时间（秒）
        self._timeout = 300

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048,
        num_ctx: int = 4096,
        # 停止序列：模型生成到这些字符串时停止输出
        stop: Optional[List[str]] = None,
    ) -> Optional[str]:
        # 最大重试次数（Ollama 冷启动偶尔返回空内容，重试一次通常能解决）
        max_retry = 1
        retry = 0

        while retry <= max_retry:
            try:
                # 构建 OpenAI 兼容的请求体
                request_body = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False,
                    "keep_alive": "20m",
                    # rakuten-ai 用轻量重复惩罚（1.1），低于 calm2 的 1.15
                    # 防止低 temperature 下 token 循环，同时不伤日语自然表达
                    "repeat_penalty": 1.1,
                    # 上下文窗口：显式设置
                    "num_ctx": num_ctx,
                }
                # stop 仅在有值时加入请求体（None 时省略）
                if stop:
                    request_body["stop"] = stop

                response = requests.post(
                    url=self._chat_url,
                    json=request_body,
                    timeout=self._timeout,
                )
                # 检查 HTTP 状态码，非 2xx 抛出异常
                response.raise_for_status()
                # 解析 OpenAI 格式响应：choices[0].message.content
                data = response.json()
                content = data["choices"][0]["message"]["content"]

                # 如果返回空内容且还有重试次数，等待后重试
                if not content and retry < max_retry:
                    retry += 1
                    logger.warning("LLM 返回空内容，1秒后重试...")
                    time.sleep(1)
                    continue

                logger.info(f"LLM response generated, length: {len(content)} chars")
                return content

            except requests.exceptions.ConnectionError:
                # Ollama 服务未启动或端口无法连接
                logger.error("Ollama 服务未启动，无法连接 LLM")
                return None
            except requests.exceptions.Timeout:
                # 模型推理超时
                logger.error(f"LLM 请求超时（{self._timeout}s）")
                return None
            except (KeyError, IndexError, TypeError) as e:
                # 响应格式异常（Ollama 版本不兼容或返回了非预期格式）
                logger.error(f"LLM 响应格式异常: {e}")
                return None
            except Exception as e:
                # 兜底：未知异常
                logger.error(f"LLM 调用未知异常: {e}")
                return None

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        # 上下文窗口大小（token 数），默认 4096
        # 多轮对话 + chunk 时建议设为 8192
        num_ctx: int = 4096,
        # 停止序列：同 chat() 方法
        stop: Optional[List[str]] = None,
    ) -> Generator[str, None, None]:
        # 流式调用 LLM，逐 token 推送，不阻塞等待完整响应
        # 使用 Ollama 的 OpenAI 兼容端点 + stream=True
        try:
            # 构建请求体
            request_body = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
                "keep_alive": "20m",
                "repeat_penalty": 1.1,
                "num_ctx": num_ctx,
            }
            if stop:
                request_body["stop"] = stop

            # 发起流式 HTTP 请求，stream=True 让 response 不会一次性加载全部内容
            response = requests.post(
                url=self._chat_url,
                json=request_body,
                timeout=self._timeout,
                stream=True,  # HTTP 层面启用流式读取
            )
            # 检查 HTTP 状态码
            response.raise_for_status()

            # SSE 缓冲区：跨 chunk 拼接不完整的行
            sse_buffer = ""
            token_count = 0

            # 逐块读取原始字节，手动解码和解析 SSE 事件
            # 不使用 iter_lines(decode_unicode=True) 以避免编码问题
            for chunk in response.iter_content(chunk_size=1024, decode_unicode=False):
                if not chunk:
                    continue
                # 字节解码为 UTF-8 字符串，追加到缓冲区
                sse_buffer += chunk.decode("utf-8")

                # 按 \n\n 分割 SSE 事件（SSE 协议以双换行分隔事件）
                while "\n\n" in sse_buffer:
                    event_block, sse_buffer = sse_buffer.split("\n\n", 1)
                    # 处理事件块内的每一行
                    for line in event_block.split("\n"):
                        # 去除行首尾空白和 \r
                        line = line.strip()
                        # 只处理以 "data:" 开头的行
                        if not line.startswith("data:"):
                            continue
                        # 去掉 "data:" 前缀，去除空白
                        data_str = line[5:].strip()
                        # 结束标记：[DONE] 表示流结束
                        if data_str == "[DONE]":
                            break
                        try:
                            # 解析 JSON
                            data = json.loads(data_str)
                            # 提取 delta.content（增量内容），结构: choices[0].delta.content
                            choices = data.get("choices", [])
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            # 如果本次有内容，yield 出去
                            if content:
                                token_count += 1
                                yield content
                        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                            # 某行 JSON 解析失败，跳过继续处理下一行
                            continue

            logger.info(f"LLM 流式响应完成，共 {token_count} 个 token")

        except requests.exceptions.ConnectionError:
            # Ollama 服务未启动
            logger.error("Ollama 服务未启动，无法连接 LLM（流式）")
            raise
        except requests.exceptions.Timeout:
            # 流式请求超时（注意：流式请求的超时是整体超时，不是单次读取超时）
            logger.error(f"LLM 流式请求超时（{self._timeout}s）")
            raise
        except Exception as e:
            # 兜底：未知异常
            logger.error(f"LLM 流式调用异常: {e}")
            raise