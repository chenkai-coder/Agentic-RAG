"""
全局配置模块
所有路径、模型名、超参数集中管理，支持环境变量覆盖。
"""

import os
from pathlib import Path

# ============================================================
# 项目根目录
# ============================================================
ROOT_DIR = Path(__file__).resolve().parent.parent

# ============================================================
# PDF 文件路径
# ============================================================
PDF_DIR = ROOT_DIR
PDF_FILES = {
    "产品手册.pdf": str(PDF_DIR / "产品手册.pdf"),
    "杂志.pdf": str(PDF_DIR / "杂志.pdf"),
}

# ============================================================
# 数据处理参数
# ============================================================
CHUNK_SIZE = 400       # 每个文本块的字符数
CHUNK_OVERLAP = 50      # 相邻块之间的重叠字符数

# ============================================================
# Embedding 配置
# backend: "api"=硅基流动免费BGE-M3 | "local"=本地模型
# ============================================================
EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "api")
EMBEDDING_API_MODEL = os.environ.get("EMBEDDING_API_MODEL", "BAAI/bge-m3")
EMBEDDING_LOCAL_PATH = "E:/UltraRAG/cache/modelscope/BAAI/bge-base-zh-v1___5"
EMBEDDING_USE_FP16 = True
EMBEDDING_MAX_LENGTH = 512

# ============================================================
# 重排序模型配置（通过 ModelScope 下载到本地）
# BAAI/bge-reranker-base: ~1GB, 性价比高
# ============================================================
RERANKER_MODEL_PATH = "E:/UltraRAG/cache/modelscope/BAAI/bge-reranker-base"
RERANKER_USE_FP16 = True

# ============================================================
# ChromaDB 配置
# ============================================================
CHROMA_PERSIST_PATH = str(ROOT_DIR / "my_rag_db")
CHROMA_COLLECTION_NAME = "pdf_knowledge_base"

# ============================================================
# 检索参数
# ============================================================
DENSE_TOP_K = 15          # 向量检索粗召回数量
BM25_TOP_K = 15           # BM25 粗召回数量
FINAL_TOP_K = 4           # Reranker 精排后保留数量
RRF_K = 60                # RRF 融合常数

# ============================================================
# 防幻觉阈值
# CrossEncoder 原始 logits 分数，越高越相关。
# 如果最高分低于此阈值，直接判定为 unanswerable。
# 实测: 相关 ~4-8分, 真正无关 ~2-3分, 阈值取 3.5 可较好分离
# 注：阈值宁可偏低（让LLM二次判断），也不要偏高（误杀可回答问题）
# ============================================================
UNANSWERABLE_SCORE_THRESHOLD = 3.5

# ============================================================
# LLM API 配置（OpenAI 兼容接口）
# 支持 DeepSeek / Qwen / 智谱 / 任意兼容服务
# 多模型自动降级：主模型限流时自动切换下一个
# ============================================================
LLM_API_KEY = os.environ.get("LLM_API_KEY", "your-siliconflow-api-key")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.siliconflow.cn/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-ai/DeepSeek-V3.2")
LLM_TEMPERATURE = 0.1      # 低温度降低幻觉
LLM_MAX_TOKENS = 2048

# 自动降级模型列表：主模型限流时按能力强→弱依次尝试
LLM_FALLBACK_MODELS = [
    "Pro/deepseek-ai/DeepSeek-V3.2",      # DeepSeek V3.2 付费版
    "Qwen/Qwen3.5-397B-A17B",             # Qwen3.5 397B MoE
    "Pro/zai-org/GLM-5",                  # GLM-5
    "Qwen/Qwen3-32B",                     # Qwen3 32B
]

# ============================================================
# 评估配置
# ============================================================
QA_PAIRS_PATH = str(ROOT_DIR / "qa_pairs.jsonl")
