"""
LLM 生成模块
使用 OpenAI 兼容接口调用大模型，通过严格的 System Prompt 进行"第二段式防幻觉"约束。
支持 DeepSeek / Qwen / 智谱 / 硅基流动 / 任意 OpenAI 兼容服务。

内置：
  - 指数退避重试（同模型内重试）
  - 自动模型降级（主模型限流时切换备选模型）
"""

from typing import List, Dict
import logging
import time

from openai import OpenAI

from . import config

logger = logging.getLogger(__name__)

# 全局 LLM 客户端（惰性初始化）
_llm_client = None

# ============================================================
# System Prompt（第二道防幻觉防线）
# ============================================================
SYSTEM_PROMPT = """你是一个严谨的文档问答助手。请严格根据用户提供的【参考证据】回答问题。

重要规则：
1. 只能基于【参考证据】中给出的事实回答，绝对不能利用你自身的外部知识编造任何内容！
2. 如果【参考证据】内的信息不足以回答问题的核心，你必须直接且精准地回复："文档中没有提供相关信息"，绝对严禁强行解答或猜测。
3. 在你的回答中，必须在每个核心事实后面以 [引用: 文件名, 第X页] 的格式清晰标注出处。
4. 回答要简洁、准确，使用中文。不要添加与证据无关的背景介绍或扩展说明。"""

UNANSWERABLE_RESPONSE = "文档中没有提供相关信息"

# 重试配置
MAX_RETRIES = 3           # 每个模型最多重试次数
RETRY_BASE_DELAY = 2.0    # 基础等待秒数


def _get_client() -> OpenAI:
    """惰性初始化 LLM 客户端。"""
    global _llm_client
    if _llm_client is None:
        logger.info(f"初始化 LLM 客户端: {config.LLM_BASE_URL}")
        _llm_client = OpenAI(
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
            max_retries=0,
            timeout=60.0,
        )
    return _llm_client


def _try_generate(
    model: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """
    用指定模型尝试生成，内置指数退避重试。
    若所有重试都失败，抛出最后一次的异常。
    """
    client = _get_client()

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

        except Exception as e:
            error_str = str(e)
            is_retryable = any(
                code in error_str
                for code in ["429", "503", "500", "timed out", "timeout",
                             "Too Many Requests", "rate limiting", "busy"]
            )

            if is_retryable and attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    f"[{model}] 请求被限流/繁忙 "
                    f"(attempt {attempt + 1}/{MAX_RETRIES})，{delay:.0f}秒后重试..."
                )
                time.sleep(delay)
            else:
                raise

    raise RuntimeError(f"[{model}] 重试耗尽")


def generate(
    query: str,
    contexts: List[Dict],
    temperature: float = config.LLM_TEMPERATURE,
    max_tokens: int = config.LLM_MAX_TOKENS,
) -> str:
    """
    基于检索到的上下文生成答案。
    自动降级：主模型 → 备选模型 1 → 备选模型 2 → ...

    Args:
        query: 用户问题
        contexts: 重排序后的 Top-K 文档
        temperature: LLM 采样温度
        max_tokens: 最大生成 token 数

    Returns:
        str: 生成的答案文本
    """
    if not contexts:
        return UNANSWERABLE_RESPONSE

    # 拼接参考证据
    context_parts = []
    for i, doc in enumerate(contexts, 1):
        source = doc.get("metadata", {}).get("source", "未知文件")
        page = doc.get("metadata", {}).get("page", "未知页")
        content = doc.get("content", "")
        context_parts.append(f"[证据{i}] 来源: {source}, 第{page}页\n{content}")

    context_str = "\n\n---\n\n".join(context_parts)
    user_prompt = f"【参考证据】：\n{context_str}\n\n【用户问题】：{query}"

    # 模型优先级列表
    all_models = [config.LLM_MODEL] + list(config.LLM_FALLBACK_MODELS)

    last_error = None
    for model in all_models:
        try:
            answer = _try_generate(model, user_prompt, temperature, max_tokens)
            logger.info(f"[{model}] 生成完成: {len(answer)} 字符")
            return answer
        except Exception as e:
            last_error = e
            error_str = str(e)
            # 不可重试的错误（如 401 认证失败）直接退出，不换模型
            if "401" in error_str or "403" in error_str and "disabled" in error_str.lower():
                logger.error(f"[{model}] 认证/授权失败，不切换模型: {e}")
                break
            logger.warning(f"[{model}] 不可用，尝试下一个模型...")

    # 所有模型都失败
    logger.error(f"所有 LLM 模型均调用失败: {last_error}")

    fallback = "LLM 调用失败，以下是检索到的相关证据：\n\n"
    for i, doc in enumerate(contexts, 1):
        src = doc.get("metadata", {}).get("source", "")
        pg = doc.get("metadata", {}).get("page", "")
        content = doc.get("content", "")[:300]
        fallback += f"[证据{i}] {src} 第{pg}页:\n{content}\n\n"
    return fallback
