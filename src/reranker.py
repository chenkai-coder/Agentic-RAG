"""
重排序模块
使用 transformers 原生 API 加载 BGE-Reranker（CrossEncoder）。
避免 torchvision 依赖。

注意：首次使用需要下载模型文件（~1GB），请确保网络通畅。
"""

from typing import List, Dict, Tuple
import logging

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import config

logger = logging.getLogger(__name__)

# 全局模型实例（惰性加载）
_reranker_tokenizer = None
_reranker_model = None
_device = None


def _get_device():
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def _get_model():
    """惰性加载 Reranker 模型。"""
    global _reranker_model, _reranker_tokenizer
    if _reranker_model is None:
        logger.info(f"正在从本地加载重排序模型: {config.RERANKER_MODEL_PATH} ...")
        _reranker_tokenizer = AutoTokenizer.from_pretrained(
            config.RERANKER_MODEL_PATH
        )
        _reranker_model = AutoModelForSequenceClassification.from_pretrained(
            config.RERANKER_MODEL_PATH
        )
        _reranker_model.to(_get_device())
        _reranker_model.eval()
        logger.info(f"重排序模型加载完成，设备: {_get_device()}")
    return _reranker_tokenizer, _reranker_model


def rerank(
    query: str,
    candidates: List[Dict],
    top_k: int = config.FINAL_TOP_K,
) -> Tuple[List[Dict], float]:
    """
    对候选文档进行重排序，返回 Top-K 结果及最高分数。

    Args:
        query: 用户问题
        candidates: 混合检索返回的候选文档列表
        top_k: 精排后保留的数量

    Returns:
        (sorted_docs, max_score)
    """
    if not candidates:
        logger.warning("候选文档为空，跳过重排序")
        return [], -100.0

    tokenizer, model = _get_model()
    device = _get_device()

    # 构造 (query, doc) 对
    pairs = [(query, doc["content"]) for doc in candidates]

    # 编码所有 pairs
    encoded = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    # 推理打分
    with torch.no_grad():
        outputs = model(**encoded)
        scores = outputs.logits.squeeze(-1).cpu().tolist()

    # 处理单结果
    if isinstance(scores, float):
        scores = [scores]

    max_score = max(scores) if scores else -100.0

    # 按分数降序排列
    sorted_pairs = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True,
    )

    top_docs = []
    for score, doc in sorted_pairs[:top_k]:
        doc = doc.copy()
        doc["rerank_score"] = float(score)
        top_docs.append(doc)

    logger.info(
        f"重排序完成: {len(candidates)} → {len(top_docs)} 条，"
        f"最高分 {max_score:.4f}"
    )

    return top_docs, max_score
