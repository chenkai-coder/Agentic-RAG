"""
嵌入模型封装模块
支持双后端：
  - API: 硅基流动免费 BAAI/bge-m3 (1024维, 多语言, 无需下载)
  - Local: 本地 transformers 模型（需要先下载到本地）
"""

from typing import List
import logging
import numpy as np
import time

from . import config

logger = logging.getLogger(__name__)

# ============================================================
# API 后端（默认）：硅基流动 / 任意 OpenAI 兼容 Embedding API
# ============================================================
_api_embedding_dim = None


def _encode_via_api(texts: List[str], is_query: bool = False) -> np.ndarray:
    """通过 API 编码文本。"""
    global _api_embedding_dim

    from openai import OpenAI
    client = OpenAI(
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
        timeout=30.0,
    )

    all_embeddings = []
    batch_size = 16

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for attempt in range(3):
            try:
                resp = client.embeddings.create(
                    model=config.EMBEDDING_API_MODEL,
                    input=batch,
                )
                batch_vecs = [d.embedding for d in resp.data]
                all_embeddings.extend(batch_vecs)
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise e

        if (i // batch_size) % 20 == 0 and len(texts) > batch_size:
            logger.info(f"API 编码进度: {min(i + batch_size, len(texts))}/{len(texts)}")

    result = np.array(all_embeddings, dtype=np.float32)
    if _api_embedding_dim is None:
        _api_embedding_dim = result.shape[1]
        logger.info(f"API Embedding 维度: {_api_embedding_dim}")
    return result


# ============================================================
# Local 后端：本地 transformers 模型
# ============================================================
_local_tokenizer = None
_local_model = None
_local_device = None


def _get_local_device():
    global _local_device
    if _local_device is None:
        import torch
        _local_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _local_device


def _get_local_model():
    global _local_tokenizer, _local_model
    if _local_model is None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        logger.info(f"正在从本地加载嵌入模型: {config.EMBEDDING_LOCAL_PATH} ...")
        _local_tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_LOCAL_PATH)
        _local_model = AutoModel.from_pretrained(config.EMBEDDING_LOCAL_PATH)
        _local_model.to(_get_local_device())
        _local_model.eval()
        logger.info(f"本地嵌入模型加载完成，设备: {_get_local_device()}")
    return _local_tokenizer, _local_model


def _mean_pooling(model_output, attention_mask):
    import torch
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def _encode_via_local(texts: List[str]) -> np.ndarray:
    import torch
    tokenizer, model = _get_local_model()
    device = _get_local_device()
    normalized = []

    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            max_length=config.EMBEDDING_MAX_LENGTH, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            output = model(**encoded)
            emb = _mean_pooling(output, encoded["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        normalized.append(emb.cpu().numpy())

    return np.concatenate(normalized, axis=0)


# ============================================================
# 统一接口
# ============================================================
def encode_query(query: str) -> np.ndarray:
    """编码单个查询文本。"""
    if config.EMBEDDING_BACKEND == "api":
        # BGE-M3 query 需要 instruction prefix
        query_with_prefix = f"为这个句子生成表示以用于检索相关文章：{query}"
        return _encode_via_api([query_with_prefix], is_query=True)[0]
    else:
        return _encode_via_local([f"为这个句子生成表示以用于检索相关文章：{query}"])[0]


def encode_documents(documents: List[str], batch_size: int = 32) -> np.ndarray:
    """批量编码文档文本。"""
    if config.EMBEDDING_BACKEND == "api":
        return _encode_via_api(documents)
    else:
        logger.info(f"开始本地编码 {len(documents)} 个文档块...")
        return _encode_via_local(documents)
