"""
全流程编排模块
将 PDF 解析、分块、索引构建、检索、重排序、生成串联为完整 RAG 流程。

使用方式：
    pipeline = RAGPipeline()
    pipeline.build()              # 首次运行：解析 PDF + 构建索引
    result = pipeline.ask("问题")  # 问答
"""

import logging
from typing import Dict, List, Optional

from . import config
from .parser import parse_all_pdfs
from .chunker import chunk_pages
from .indexer import build_index, load_index
from .retriever import hybrid_search
from .detector import is_unanswerable

# Reranker 惰性导入（模型路径可能不存在于新机器）
_rerank_fn = None


def _get_rerank():
    global _rerank_fn
    if _rerank_fn is None:
        try:
            from .reranker import rerank as _r
            _rerank_fn = _r
        except Exception:
            _rerank_fn = False
    return _rerank_fn if _rerank_fn is not False else None
from .generator import generate, UNANSWERABLE_RESPONSE

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    RAG 全流程编排器。

    使用单例模式确保模型只加载一次：
        pipeline = RAGPipeline()
        pipeline.build()
        answer = pipeline.ask("你的问题")
    """

    def __init__(self):
        self._built = False

    # ============================================================
    # 索引构建
    # ============================================================
    def build(self, force_rebuild: bool = False) -> None:
        """
        构建完整索引：解析 PDF → 分块 → 嵌入 → 存入 ChromaDB + BM25。

        Args:
            force_rebuild: 是否强制重建（清空已有索引）
        """
        logger.info("=" * 60)
        logger.info("开始构建 RAG 索引...")
        logger.info("=" * 60)

        # Step 1: 解析 PDF
        pages = parse_all_pdfs(config.PDF_FILES)

        # Step 2: 文本分块 + 元数据富化
        chunks = chunk_pages(pages)

        # Step 3: 嵌入 + 构建双路索引
        build_index(chunks, force_rebuild=force_rebuild)

        # Step 4: 验证加载
        load_index()

        self._built = True
        logger.info("=" * 60)
        logger.info("RAG 索引构建完成！现在可以使用 pipeline.ask() 提问。")
        logger.info("=" * 60)

    # ============================================================
    # 问答
    # ============================================================
    def ask(
        self,
        question: str,
        return_details: bool = False,
    ) -> Dict:
        """
        执行一次完整的 RAG 问答。

        Args:
            question: 用户问题
            return_details: 是否返回检索详情（用于调试和评估）

        Returns:
            Dict: {
                "question": str,
                "answer": str,
                "is_unanswerable": bool,
                "contexts": List[Dict],         # 最终使用的证据
                "max_rerank_score": float,
                "candidate_count": int,
            }
        """
        # 确保索引已加载
        if not load_index():
            return {
                "question": question,
                "answer": "索引未构建，请先执行 pipeline.build()",
                "is_unanswerable": True,
                "contexts": [],
                "max_rerank_score": -100,
                "candidate_count": 0,
            }

        # Step 1: 混合检索
        from .retriever import _detect_source_hint
        source_hint = _detect_source_hint(question)
        candidates = hybrid_search(question)

        # Step 2: 重排序 + 取最高分
        _r = _get_rerank()
        top_docs, max_score = _r(question, candidates) if _r else (candidates[:config.FINAL_TOP_K], 0)

        # Step 2.5: 源文档覆盖保障 —— 当问题指定了来源时，
        # 确保最终上下文中至少包含 2 条目标源的内容
        if source_hint and len(top_docs) >= 2:
            from_source = [d for d in top_docs if d["metadata"].get("source") == source_hint]
            from_other = [d for d in top_docs if d["metadata"].get("source") != source_hint]
            # 如果目标源内容不足 2 条 OR 有重复页面，用 RRF 结果补充
            seen_pages = set(d["metadata"]["page"] for d in from_source)
            if len(from_source) < 2 or len(seen_pages) < len(from_source):
                extra_source = [
                    d for d in candidates
                    if d["metadata"].get("source") == source_hint
                    and d["metadata"]["page"] not in seen_pages
                    and d not in top_docs
                ]
                extra_source.sort(key=lambda d: d.get("rrf_score", 0), reverse=True)
                # 补充不同页面的内容
                for d in extra_source:
                    if d["metadata"]["page"] not in seen_pages:
                        from_source.append(d)
                        seen_pages.add(d["metadata"]["page"])
                    if len(from_source) >= 3:  # 最多取 3 条目标源
                        break
                # 重组 top_docs
                top_docs = from_source[:3] + from_other[:max(1, config.FINAL_TOP_K - 3)]
                logger.info(
                    f"源覆盖补充: 目标源 {len(from_source[:2])} 条 + "
                    f"其他源 {len(from_other[:config.FINAL_TOP_K - 2])} 条"
                )

        # Step 3: 防幻觉第一道防线 — 分数阈值截断
        if is_unanswerable(max_score):
            return {
                "question": question,
                "answer": UNANSWERABLE_RESPONSE,
                "is_unanswerable": True,
                "contexts": [],
                "max_rerank_score": max_score,
                "candidate_count": len(candidates),
            }

        # Step 4: LLM 生成（防幻觉第二道防线在 Prompt 中）
        answer = generate(question, top_docs)

        return {
            "question": question,
            "answer": answer,
            "is_unanswerable": False,
            "contexts": top_docs,
            "max_rerank_score": max_score,
            "candidate_count": len(candidates),
        }

    # ============================================================
    # 检索（不调用 LLM，用于调试）
    # ============================================================
    def retrieve(self, question: str, top_k: int = config.FINAL_TOP_K) -> List[Dict]:
        """
        仅执行检索+重排序，不调用 LLM。用于调试和评估检索质量。
        """
        if not load_index():
            return []

        candidates = hybrid_search(question)
        top_docs, _ = rerank(question, candidates)
        return top_docs[:top_k]


# 全局单例
_pipeline_instance: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    """获取全局 RAGPipeline 单例。"""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()
    return _pipeline_instance
