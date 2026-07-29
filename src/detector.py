"""
不可回答问题检测模块
"两段式防幻觉"的第一道防线：通过 Reranker 分数阈值硬编码截断。

设计动机：
  当检索到的所有文档与问题的相关性都很低时（Reranker 最高分 < 阈值），
  说明 PDF 中没有相关信息。此时直接返回"文档中没有提供相关信息"，
  避免将不相关的文本喂给 LLM 导致其强行编造（幻觉）。

分数说明：
  BGE-Reranker normalize=True 后的分数通常在 0~1 之间，
  相关性越高分数越大。阈值需根据实际数据调试。
"""

import logging

from . import config

logger = logging.getLogger(__name__)


def is_unanswerable(
    max_rerank_score: float,
    threshold: float = config.UNANSWERABLE_SCORE_THRESHOLD,
) -> bool:
    """
    判断问题是否无法从文档中回答。

    Args:
        max_rerank_score: Reranker 给出的最高相关性分数
        threshold: 判定阈值，低于此值视为 unanswerable

    Returns:
        True: 文档中没有相关证据，应拒绝回答
        False: 找到了相关证据，可以继续生成
    """
    if max_rerank_score < threshold:
        logger.info(
            f"判定为 unanswerable: max_score={max_rerank_score:.4f} < threshold={threshold}"
        )
        return True

    logger.info(
        f"有相关证据: max_score={max_rerank_score:.4f} >= threshold={threshold}"
    )
    return False
