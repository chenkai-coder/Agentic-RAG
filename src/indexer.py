"""
索引构建模块
构建双路索引：
  1. ChromaDB 向量索引（Dense 语义检索）
  2. BM25 关键词索引（精确匹配产品型号、地名等）
支持首次构建后持久化，后续直接加载。
"""

import os
import pickle
import logging
from typing import List, Dict, Optional
import numpy as np

import chromadb
from rank_bm25 import BM25Okapi
import jieba

from . import config
from .embedder import encode_documents, encode_query

logger = logging.getLogger(__name__)


# ============================================================
# 全局客户端（惰性初始化）
# ============================================================
_chroma_client = None
_collection = None
_bm25_index = None
_bm25_docs_meta = None   # BM25 中文档对应的 metadata 列表


def _get_chroma_client() -> chromadb.PersistentClient:
    """获取 ChromaDB 持久化客户端。"""
    global _chroma_client
    if _chroma_client is None:
        os.makedirs(config.CHROMA_PERSIST_PATH, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_PATH)
    return _chroma_client


def _get_collection() -> chromadb.Collection:
    """获取或创建 ChromaDB collection。"""
    global _collection
    if _collection is None:
        client = _get_chroma_client()
        _collection = client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ============================================================
# 中文分词（BM25 用）
# ============================================================
def _tokenize(text: str) -> List[str]:
    """对中文文本进行分词，用于 BM25 索引。"""
    # 移除元数据前缀，只对正文分词
    # 元数据前缀格式: 【文件来源：xxx | 第 N 页】\n
    if text.startswith("【文件来源："):
        text = text.split("\n", 1)[-1] if "\n" in text else text
    return list(jieba.cut(text))


# ============================================================
# 索引构建
# ============================================================
def build_index(chunks: List[Dict], force_rebuild: bool = False) -> None:
    """
    构建完整的双路索引（ChromaDB + BM25）。

    执行流程：
    1. 提取所有 chunk 的纯内容用于 BM25 分词
    2. 用 BGE-M3 将所有 chunk 编码为稠密向量
    3. 写入 ChromaDB（自动持久化到磁盘）
    4. 构建 BM25 索引并保存为 pickle 文件

    Args:
        chunks: chunker.chunk_pages() 的输出
        force_rebuild: 是否强制重建（清空已有数据）
    """
    # --- 前置检查 ---
    if not chunks:
        logger.warning("chunks 为空，跳过索引构建")
        return

    collection = _get_collection()
    existing_count = collection.count()

    if existing_count > 0 and not force_rebuild:
        logger.info(
            f"ChromaDB 中已有 {existing_count} 条记录，跳过索引构建。"
            f"如需重建请设置 force_rebuild=True 或删除 {config.CHROMA_PERSIST_PATH}"
        )
        return

    if force_rebuild and existing_count > 0:
        logger.info(f"强制重建：正在清空已有的 {existing_count} 条记录...")
        # ChromaDB 没有 delete_collection 的单条清空，重建 collection
        client = _get_chroma_client()
        client.delete_collection(config.CHROMA_COLLECTION_NAME)
        global _collection
        _collection = None
        collection = _get_collection()

    # --- Step 1: 准备数据 ---
    documents = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [c["metadata"]["chunk_id"] for c in chunks]

    logger.info(f"开始构建索引，共 {len(chunks)} 个文本块...")

    # --- Step 2: 编码向量 ---
    embeddings = encode_documents(documents)

    # --- Step 3: 写入 ChromaDB ---
    logger.info("正在写入 ChromaDB...")
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        end = min(i + batch_size, len(documents))
        collection.add(
            documents=documents[i:end],
            embeddings=embeddings[i:end].tolist(),
            metadatas=metadatas[i:end],
            ids=ids[i:end],
        )
        if (i // batch_size) % 10 == 0:
            logger.info(f"ChromaDB 写入进度: {end}/{len(documents)}")

    logger.info(f"ChromaDB 索引构建完成，共 {collection.count()} 条记录")

    # --- Step 4: 构建 BM25 索引 ---
    logger.info("正在构建 BM25 关键词索引...")
    global _bm25_index, _bm25_docs_meta

    tokenized_docs = [_tokenize(doc) for doc in documents]
    _bm25_index = BM25Okapi(tokenized_docs)
    _bm25_docs_meta = [c["metadata"] for c in chunks]

    # 保存到磁盘
    bm25_path = os.path.join(config.CHROMA_PERSIST_PATH, "bm25_index.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump({
            "tokenized_docs": tokenized_docs,
            "metadatas": _bm25_docs_meta,
        }, f)
    logger.info(f"BM25 索引已保存至: {bm25_path}")

    logger.info("索引构建全部完成！")


# ============================================================
# 索引加载
# ============================================================
def load_index() -> bool:
    """
    从磁盘加载已有索引。

    返回 True 表示索引加载成功，False 表示索引不存在需要先 build。
    """
    global _bm25_index, _bm25_docs_meta

    # 检查 ChromaDB 是否有数据
    collection = _get_collection()
    count = collection.count()

    if count == 0:
        logger.warning("ChromaDB 为空，请先执行 build")
        return False

    # 加载 BM25
    bm25_path = os.path.join(config.CHROMA_PERSIST_PATH, "bm25_index.pkl")
    if os.path.exists(bm25_path):
        logger.info(f"正在加载 BM25 索引: {bm25_path}")
        with open(bm25_path, "rb") as f:
            data = pickle.load(f)
        _bm25_index = BM25Okapi(data["tokenized_docs"])
        _bm25_docs_meta = data["metadatas"]
        logger.info(f"BM25 索引加载完成，共 {len(data['tokenized_docs'])} 篇文档")
    else:
        logger.warning("BM25 索引文件不存在，混合检索将仅使用 Dense 向量检索")

    logger.info(f"索引加载完成: ChromaDB {count} 条, BM25 {'可用' if _bm25_index else '不可用'}")
    return True


# ============================================================
# 检索接口
# ============================================================
def search_dense(query: str, top_k: int = config.DENSE_TOP_K,
                 source_filter: str = "") -> List[Dict]:
    """
    纯向量语义检索。

    Args:
        query: 查询文本
        top_k: 召回数量
        source_filter: 只检索指定来源（如 "产品手册.pdf"），空字符串表示不限制
    """
    collection = _get_collection()

    query_kwargs = {
        "query_embeddings": [encode_query(query).tolist()],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if source_filter:
        query_kwargs["where"] = {"source": source_filter}

    results = collection.query(**query_kwargs)

    docs = []
    for i in range(len(results["documents"][0])):
        docs.append({
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "dense_score": 1.0 - results["distances"][0][i],
        })
    return docs


def search_bm25(query: str, top_k: int = config.BM25_TOP_K) -> List[Dict]:
    """
    纯 BM25 关键词检索。
    """
    global _bm25_index, _bm25_docs_meta

    if _bm25_index is None:
        logger.info("BM25 索引未加载，尝试自动加载...")
        load_index()

    if _bm25_index is None:
        logger.warning("BM25 索引加载失败，返回空结果")
        return []

    tokenized_query = _tokenize(query)
    scores = _bm25_index.get_scores(tokenized_query)

    # 取 Top-K
    top_indices = np.argsort(scores)[::-1][:top_k]

    docs = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        docs.append({
            "content": "",   # BM25 不存原文，从 metadata 可回溯
            "metadata": _bm25_docs_meta[idx],
            "bm25_score": float(scores[idx]),
        })
    return docs


def get_document_by_chunk_id(chunk_id: str) -> Optional[str]:
    """
    通过 chunk_id 获取完整文档内容（用于 BM25 命中后补全）。
    """
    collection = _get_collection()
    result = collection.get(ids=[chunk_id], include=["documents"])
    if result and result["documents"]:
        return result["documents"][0]
    return None
