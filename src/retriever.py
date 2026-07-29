"""
混合检索模块
融合 Dense（向量语义）和 Sparse（BM25 关键词）两路检索结果，
使用 RRF (Reciprocal Rank Fusion) 算法合并排序。

支持：
  - HyDE：LLM 生成假设性文档，弥合 query↔document 语义鸿沟
  - LLM Query 分解：复杂问题拆分为多个子查询并行检索
  - 源文档感知：问题提到"产品手册"/"杂志"时，优先召回对应源
"""

from typing import List, Dict, Optional
import logging

from . import config
from . import indexer

logger = logging.getLogger(__name__)


# ============================================================
# 源文档感知
# ============================================================
def _detect_source_hint(query: str) -> str:
    """
    检测问题中是否暗示了来源文档。

    Returns:
        "产品手册.pdf" / "杂志.pdf" / "" (不限制)
    """
    query_lower = query.lower()
    if "产品手册" in query_lower or "手册" in query_lower:
        return "产品手册.pdf"
    if "杂志" in query_lower or "中国无线电" in query_lower:
        return "杂志.pdf"
    return ""


SOURCE_BOOST_FACTOR = 3.0   # 源文档匹配时，RRF 分数倍率


def _call_llm_with_fallback(messages, temp=0.2, max_tok=200):
    """带重试+模型降级的 LLM 调用。"""
    import time as t
    from openai import OpenAI
    all_models = [config.LLM_MODEL] + list(config.LLM_FALLBACK_MODELS)
    for model in all_models:
        client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL, timeout=15.0)
        for attempt in range(2):
            try:
                return client.chat.completions.create(
                    model=model, messages=messages, temperature=temp, max_tokens=max_tok,
                )
            except Exception as e:
                if attempt < 1 and any(x in str(e) for x in ["429", "503", "busy"]):
                    t.sleep(2 * (2 ** attempt))
                else:
                    break
    raise RuntimeError("所有模型不可用")


def _extract_keywords(query: str) -> List[str]:
    """
    从 verbose 查询中提取关键词用于 BM25 精确匹配。
    按句号/逗号/要求/需求拆分，取长度适中的片段。
    """
    import re
    keywords = []

    # 按标点拆分
    parts = re.split(r'[，。；：；、\n]', query)
    for part in parts:
        part = part.strip()
        # 保留 4-20 字的片段，这些通常是关键需求描述
        if 4 <= len(part) <= 30:
            keywords.append(part)

    # 提取 "XX性/XX能力/XX功能" 等特征词
    features = re.findall(r'[一-鿿]{2,6}(?:能力|功能|时间|精度|温度|环境)', query)
    keywords.extend(features)

    return keywords[:5]  # 最多 5 个


# ============================================================
# RRF 融合
# ============================================================
def rrf_fusion(
    dense_results: List[Dict],
    bm25_results: List[Dict],
    source_hint: str = "",
    k: int = config.RRF_K,
) -> List[Dict]:
    """
    RRF 融合两路检索结果，支持源文档优先级提升。

    公式：score_rrf(d) = sum_{r in ranks} 1 / (k + rank_r(d))
    """
    rrf_scores = {}
    doc_map = {}

    # Dense 路
    for rank, doc in enumerate(dense_results, start=1):
        chunk_id = doc["metadata"]["chunk_id"]
        boost = SOURCE_BOOST_FACTOR if (
            source_hint and doc["metadata"].get("source") == source_hint
        ) else 1.0
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + boost / (k + rank)
        if chunk_id not in doc_map:
            doc_map[chunk_id] = doc

    # BM25 路
    for rank, doc in enumerate(bm25_results, start=1):
        chunk_id = doc["metadata"]["chunk_id"]
        boost = SOURCE_BOOST_FACTOR if (
            source_hint and doc["metadata"].get("source") == source_hint
        ) else 1.0
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + boost / (k + rank)
        if chunk_id not in doc_map:
            if not doc.get("content"):
                content = indexer.get_document_by_chunk_id(chunk_id)
                doc["content"] = content or ""
            doc_map[chunk_id] = doc

    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

    merged = []
    for chunk_id in sorted_ids:
        doc = doc_map[chunk_id].copy()
        doc["rrf_score"] = rrf_scores[chunk_id]
        merged.append(doc)

    return merged


# ============================================================
# HyDE (Hypothetical Document Embeddings)
# 让 LLM 生成假设性文档，用假设文档的 embedding 去检索
# 解决了 verbose query 与稀疏 spec 文本之间的语义鸿沟
# ============================================================
HYDE_PROMPT = """请针对以下问题，生成一个简洁、精确、不包含废话的一句话事实答案（50字以内）。只输出答案本身。

问题：{query}

答案："""


def _hyde_expand(query: str) -> Optional[str]:
    """用 LLM 生成假设性文档。失败返回 None。"""
    try:
        response = _call_llm_with_fallback(
            [{"role": "user", "content": HYDE_PROMPT.format(query=query)}],
            temp=0.2, max_tok=200,
        )
        hyde_text = response.choices[0].message.content.strip()
        if hyde_text and len(hyde_text) >= 10:
            logger.info(f"HyDE 生成: \"{hyde_text[:80]}...\"")
            return hyde_text
    except Exception as e:
        logger.warning(f"HyDE 失败: {e}")
    return None


# ============================================================
# LLM Query 分解
# ============================================================
DECOMPOSE_PROMPT = """将以下用户问题拆分为 2-4 个简短的关键词查询，用于搜索引擎检索。
每个查询 5-15 字，提取核心概念和关键需求。只输出查询，每行一个，不要编号。

用户问题：{query}

拆分的查询："""


def _decompose_query(query: str) -> List[str]:
    """
    将长问题拆解为多个子查询。
    优先用 LLM 分解，失败时降级为启发式拆分。
    """
    # ---- LLM 分解（带重试+降级） ----
    try:
        response = _call_llm_with_fallback(
            [{"role": "user", "content": DECOMPOSE_PROMPT.format(query=query)}],
            temp=0.0, max_tok=150,
        )
        raw = response.choices[0].message.content.strip()
        lines = [line.strip().lstrip('-•·1234567890.、) ') for line in raw.split('\n')]
        sub_queries = [l for l in lines if 3 <= len(l) <= 40]
        if len(sub_queries) >= 2:
            logger.info(f"LLM Query分解: {sub_queries}")
            return [query] + sub_queries[:3]
    except Exception as e:
        logger.warning(f"LLM Query分解失败，使用启发式: {e}")

    # ---- 启发式降级 ----
    import re
    sub_queries = [query]
    parts = re.split(r'[。；\n]|(?:\d+[\.\、\s])', query)
    for part in parts:
        part = part.strip()
        if 6 <= len(part) <= 40:
            sub_queries.append(part)
    if len(query) > 80:
        mid = len(query) // 2
        sub_queries.append(query[:mid])
    return sub_queries[:4]  # 最多 4 个


# ============================================================
# 混合检索主入口
# ============================================================
def hybrid_search(
    query: str,
    dense_top_k: int = config.DENSE_TOP_K,
    bm25_top_k: int = config.BM25_TOP_K,
) -> List[Dict]:
    """
    混合检索主入口：Dense + BM25 → RRF 融合。

    新增优化：
      1. 源文档感知：问题提到"产品手册"/"杂志"时 boost 对应源
      2. Query 分解：长问题拆分多个子查询，扩展召回覆盖
    """
    logger.info(f"混合检索: \"{query[:80]}...\"")

    source_hint = _detect_source_hint(query)
    if source_hint:
        logger.info(f"  检测到源文档偏好: {source_hint}")

    # ---- HyDE：生成假设性文档用于 Dense 检索 ----
    hyde_text = _hyde_expand(query)

    # ---- Query 分解 ----
    sub_queries = _decompose_query(query)
    if len(sub_queries) > 1:
        logger.info(f"  Query 分解为 {len(sub_queries)} 个子查询")

    # 对每个子查询做 Dense 检索，合并去重
    all_dense = {}   # chunk_id → doc
    all_bm25 = {}

    for sq in sub_queries:
        # Dense（全局）
        for doc in indexer.search_dense(sq, top_k=dense_top_k):
            cid = doc["metadata"]["chunk_id"]
            if cid not in all_dense:
                all_dense[cid] = doc

        # HyDE Dense：用假设性文档检索（语义更接近产品手册风格）
        if hyde_text:
            for doc in indexer.search_dense(hyde_text, top_k=dense_top_k):
                cid = doc["metadata"]["chunk_id"]
                if cid not in all_dense:
                    all_dense[cid] = doc
            cid = doc["metadata"]["chunk_id"]
            if cid not in all_dense:
                all_dense[cid] = doc

        # Dense（源过滤）：当检测到源偏好时，在目标源内做增强检索
        if source_hint:
            for doc in indexer.search_dense(sq, top_k=dense_top_k,
                                            source_filter=source_hint):
                cid = doc["metadata"]["chunk_id"]
                if cid not in all_dense:
                    all_dense[cid] = doc

        # BM25
        for doc in indexer.search_bm25(sq, top_k=bm25_top_k):
            cid = doc["metadata"]["chunk_id"]
            if cid not in all_bm25:
                all_bm25[cid] = doc

    # 源偏好时，对提取的关键词做额外 BM25 精确匹配
    if source_hint:
        keywords = _extract_keywords(query)
        for kw in keywords:
            for doc in indexer.search_bm25(kw, top_k=5):
                cid = doc["metadata"]["chunk_id"]
                if doc["metadata"].get("source") == source_hint and cid not in all_bm25:
                    all_bm25[cid] = doc

    logger.info(f"  Dense 召回: {len(all_dense)} 条 (去重后)")
    logger.info(f"  BM25 召回: {len(all_bm25)} 条 (去重后)")

    # 重新按原始分数排序（保持检索器返回的顺序近似）
    dense_sorted = sorted(
        all_dense.values(),
        key=lambda d: d.get("dense_score", 0),
        reverse=True,
    )
    bm25_sorted = sorted(
        all_bm25.values(),
        key=lambda d: d.get("bm25_score", 0),
        reverse=True,
    )

    # RRF 融合（带源文档 boost）
    merged = rrf_fusion(dense_sorted, bm25_sorted, source_hint=source_hint)
    logger.info(f"  RRF 融合后: {len(merged)} 条候选")

    # 源文档平衡：当检测到源偏好时，确保至少一半候选来自目标源
    if source_hint:
        from_source = [d for d in merged if d["metadata"].get("source") == source_hint]
        from_other = [d for d in merged if d["metadata"].get("source") != source_hint]
        # 只保留前 30 条混合排序，但强制至少 10 条来自目标源
        balanced = from_source[:15] + from_other[:15]
        # 按 RRF 分数重排
        balanced.sort(key=lambda d: d.get("rrf_score", 0), reverse=True)
        merged = balanced
        logger.info(f"  源平衡后: {len(from_source)} 条来自 {source_hint}, "
                    f"{len(from_other)} 条来自其他源")

    return merged
