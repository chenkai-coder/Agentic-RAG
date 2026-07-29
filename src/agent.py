"""
Agentic RAG 模块
基于 ReAct (Reasoning + Acting) 模式的多步推理检索。
Agent 可以自主搜索、按页获取、多轮迭代，最终合成带引用的答案。

特性：
  - 多轮推理: 分析问题 → 多路搜索 → 评估结果 → 补充搜索 → 综合答案
  - 上下文压缩: 多轮搜索结果去重 + 按相关性截断，避免噪声累积
"""

import json
import logging
from typing import List, Dict, Optional
from openai import OpenAI

from . import config
from .indexer import (
    load_index, search_dense, search_bm25, get_document_by_chunk_id,
)
from .retriever import rrf_fusion

logger = logging.getLogger("agent")

# ============================================================
# Agent System Prompt
# ============================================================
AGENT_SYSTEM_PROMPT = """你是一个能够自主搜索文档库的智能问答助手。你可以使用以下工具：

工具1: search(query, source_hint)
  - 在文档库中搜索相关内容。query 用简洁关键词（如"欺骗式干扰检测 室外"），而非长句
  - source_hint 可选 "产品手册.pdf" 或 "杂志.pdf"，限定来源
  - 返回相关的文档片段及其页码

工具2: get_page(source, page)
  - 读取指定文档的指定页面完整内容

搜索策略：
1. 列举/汇总/总结类问题（如"记录了哪些案例"、"有哪些类型"、"列出所有"）：
   a. 先脑暴：文档中可能有哪些相关章节？列出所有可能涉及的栏目名、专题名、关键词
   b. 搜索时用具体词组合：核心主题+子类型（如"机场 干扰"、"高铁 GSM-R"、"黑广播 查处"、"高考 保障"、"演练"、"专题"）+ 具体地名（如"海口"、"烟台"、"黄冈"、"凉山"、"陕西"）。每个组合搜1次，覆盖尽量全
   c. 第一轮至少8次搜索，确保穷尽。8次不止——如果还有没覆盖的栏目/子类型，继续加搜
   d. 每轮结束前检查：我是否搜了所有能想到的子类型？是否搜了所有可能的栏目名称？缺了就补搜
   e. 要涵盖所有的地点，比如包括四川、海口、陕西、北京、上海等
   f. 要涵盖所有的干扰类型，比如包括欺骗式干扰、干扰检测、干扰排查、GSM-R、黑广播等
2. 事实/对比/推荐类问题：3-5次搜索覆盖核心维度
3. 搜索结果中排名靠前的页面，必须用 get_page 读取完整内容

答案规则：
- 只基于文档原文，不编造
- 事实类：一句话直接回答
- 列举类：禁止表格，按栏目/类型分节（如"一、干扰排查栏目"、"二、高考保障专题"、"三、其他典型案例"），每节下列出该栏目全部案例。每案例独立成段含地点+现象+排查+结果+引用。至少15个。开头："根据《中国无线电》杂志，本期记录了多个具有代表性的无线电干扰排查案例，包括："
- 对比类：先说A（功能→参数→场景），"相比之下"再说B。只对比不推荐
- 推荐类：只用文档中的产品大类名（如用"欺骗式/压制式干扰检测设备"而不用"车载型检测终端"），只推一个。理由含具体参数值（温度范围、检测时间、灵敏度等）
- 无信息只回"文档中没有提供相关信息"
- 引用：[文件名, 第X页]"""


# ============================================================
# Agent 工具实现
# ============================================================
def _search(query: str, source_hint: str = "") -> List[Dict]:
    """搜索文档库，Dense+BM25+RRF，返回 Top-8 结果（去重）。"""
    load_index()

    # Dense + BM25 混合检索（大召回量）
    dense = search_dense(query, top_k=40, source_filter=source_hint if source_hint else "")

    # 短 HyDE：用假设性简短答案增强短文本匹配（仅枚举类查询）
    ENUM_KW = ["案例", "哪些", "列表", "列出", "记录", "典型", "总结"]
    if any(kw in query for kw in ENUM_KW):
        try:
            from .retriever import _hyde_expand
            hyde_text = _hyde_expand(query)
            if hyde_text:
                for doc in search_dense(hyde_text, top_k=20, source_filter=source_hint if source_hint else ""):
                    dense.append(doc)
        except Exception:
            pass

    bm25 = search_bm25(query, top_k=40)

    # BM25 精确匹配对短文本更公平——多做一次纯 BM25 查询作为补召
    bm25_extra = search_bm25(query, top_k=20)
    # 合并两批 BM25 结果（去重）
    bm25_combined = bm25.copy()
    seen_bm25 = {d["metadata"]["chunk_id"] for d in bm25}
    for d in bm25_extra:
        if d["metadata"]["chunk_id"] not in seen_bm25:
            bm25_combined.append(d)

    merged = rrf_fusion(dense, bm25_combined, source_hint=source_hint)

    # 去重页面，BM25 按搜索查询分组取 top-1，保证多样性
    from collections import defaultdict
    seen_pages = set()
    unique = []
    bm25_by_query = defaultdict(list)
    for doc in bm25_combined:
        bm25_by_query[doc.get("_query", "")].append(doc)
    for sq, docs in bm25_by_query.items():
        docs.sort(key=lambda d: d.get("bm25_score", 0), reverse=True)
        for doc in docs:
            key = (doc["metadata"]["source"], doc["metadata"]["page"])
            if key not in seen_pages:
                seen_pages.add(key)
                if not doc.get("content"):
                    doc["content"] = get_document_by_chunk_id(doc["metadata"]["chunk_id"]) or ""
                unique.append(doc)
                break
    # RRF 结果补充——补全 content
    for doc in merged:
        key = (doc["metadata"]["source"], doc["metadata"]["page"])
        if key not in seen_pages:
            seen_pages.add(key)
            if not doc.get("content"):
                cid = doc["metadata"]["chunk_id"]
                doc["content"] = get_document_by_chunk_id(cid) or ""
            unique.append(doc)
    for doc in unique:
        doc["_query"] = query
    return unique[:12]


def _get_page(source: str, page: int) -> Optional[str]:
    """获取指定文档指定页的完整内容（从 ChromaDB 直接按 metadata 查）。"""
    load_index()
    from .indexer import _get_collection
    collection = _get_collection()
    try:
        result = collection.get(
            where={"$and": [{"source": source}, {"page": page}]},
            include=["documents"],
        )
        if result and result["documents"]:
            return "\n---\n".join(result["documents"])
    except Exception as e:
        logger.warning(f"get_page 失败: {e}")
    return None


# ============================================================
# Agent 工具调用解析
# 上下文压缩：去重 + BM25排序 + IDF Token级剪枝（类LLMLingua）
# ============================================================
def _compute_idf(tokenized_docs: List[List[str]]) -> Dict[str, float]:
    """计算所有文档的 IDF（逆文档频率）。"""
    import math
    N = len(tokenized_docs)
    df = {}
    for tokens in tokenized_docs:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1
    return {t: math.log((N + 1) / (df[t] + 1)) + 1.0 for t in df}


def _compress_content_idf(
    content: str,
    idf: Dict[str, float],
    keep_ratio: float = 0.5,
) -> str:
    """
    Token级压缩：保留高IDF token，剔除低信息量冗余词。
    类似 LLMLingua 但用 IDF 替代困惑度，无需额外模型。
    """
    tokens = _tokenize_for_bm25(content)
    if len(tokens) < 20:
        return content  # 太短不压缩

    # 计算每个 token 的信息分数（IDF × 是否在元数据中）
    scores = []
    for i, t in enumerate(tokens):
        s = idf.get(t, 1.0)
        # 元数据标记永远保留
        if t in ("【文件来源", "产品手册", "杂志", "第", "页", "引用", "P"):
            s = 999.0
        scores.append((i, t, s))

    # 保留 top keep_ratio 分数的 token
    threshold_idx = max(1, int(len(tokens) * keep_ratio))
    top_tokens = sorted(scores, key=lambda x: x[2], reverse=True)[:threshold_idx]
    # 按原位置排序
    top_tokens.sort(key=lambda x: x[0])
    return "".join(t[1] for t in top_tokens)


def _compress_context(
    collected_info: List[Dict],
    question: str,
    max_chunks: int = 15,
    max_chars_per_chunk: int = 400,
    token_keep_ratio: float = 0.5,
) -> List[Dict]:
    """
    多级压缩：
      1. 页面去重（保留最高分）
      2. BM25 对问题打分排序
      3. IDF Token级剪枝（类LLMLingua，去冗余词）
      4. 截断 top-N
    """
    if len(collected_info) <= max_chunks:
        return collected_info

    # 1. 按页面去重
    seen = {}
    for doc in collected_info:
        key = (doc["metadata"]["source"], doc["metadata"]["page"])
        score = doc.get("rerank_score", doc.get("dense_score", doc.get("rrf_score", 0)))
        if key not in seen or score > seen[key][0]:
            seen[key] = (score, doc)
    unique_docs = [doc for _, doc in seen.values()]

    # 2. BM25 排序
    try:
        tokenized_q = _tokenize_for_bm25(question)
        tokenized_docs = [_tokenize_for_bm25(d.get("content", "")) for d in unique_docs]
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi(tokenized_docs)
        scores = bm25.get_scores(tokenized_q)
        ranked = sorted(zip(scores, unique_docs), key=lambda x: x[0], reverse=True)
        unique_docs = [doc for _, doc in ranked]
    except Exception:
        tokenized_docs = [_tokenize_for_bm25(d.get("content", "")) for d in unique_docs]

    # 3. IDF Token级剪枝（类LLMLingua）
    idf = _compute_idf(tokenized_docs)
    logger.info(f"IDF词汇表: {len(idf)} tokens, 压缩比={token_keep_ratio}")

    # 4. 截断 + Token剪枝
    result = []
    for doc in unique_docs[:max_chunks]:
        doc = doc.copy()
        content = doc.get("content", "")
        # 字符级截断
        prefix = ""
        body = content
        if len(content) > max_chars_per_chunk:
            if "【文件来源" in content and "\n" in content:
                prefix_end = content.index("\n") + 1
                prefix = content[:prefix_end]
                body = content[prefix_end:][:max_chars_per_chunk]
            else:
                body = content[:max_chars_per_chunk]
        # (Token级剪枝已禁用——会破坏"欺骗式/压制式干扰检测设备"等技术术语)
        doc["content"] = prefix + body if prefix else body
        result.append(doc)

    total_chars_before = sum(len(d.get("content", "")) for d in collected_info)
    total_chars_after = sum(len(d["content"]) for d in result)
    logger.info(
        f"上下文压缩: {len(collected_info)}→{len(unique_docs)}(去重)→{len(result)}(截断), "
        f"{total_chars_before}→{total_chars_after}字符"
    )
    return result


def _tokenize_for_bm25(text: str) -> List[str]:
    """简单中文分词（兼容无 jieba 环境）。"""
    try:
        import jieba
        return list(jieba.cut(text))
    except ImportError:
        import re
        return re.findall(r'[一-鿿]+|[a-zA-Z0-9]+', text)


def _parse_tool_calls(text: str) -> List[Dict]:
    """解析 LLM 返回文本中的工具调用。"""
    import re
    calls = []

    # 匹配 search("query", "source_hint") 或 search("query")
    for m in re.finditer(
        r'search\s*\(\s*"([^"]+)"(?:\s*,\s*"([^"]*)")?\s*\)',
        text, re.IGNORECASE,
    ):
        calls.append({
            "tool": "search",
            "query": m.group(1),
            "source_hint": m.group(2) or "",
        })

    # 匹配 get_page("source", page)
    for m in re.finditer(
        r'get_page\s*\(\s*"([^"]+)"\s*,\s*(\d+)\s*\)',
        text, re.IGNORECASE,
    ):
        calls.append({
            "tool": "get_page",
            "source": m.group(1),
            "page": int(m.group(2)),
        })

    return calls


def _format_search_results(results: List[Dict]) -> str:
    """格式化搜索结果给 LLM。"""
    if not results:
        return "（未找到相关内容）"

    parts = []
    for i, doc in enumerate(results, 1):
        meta = doc["metadata"]
        content = doc.get("content", "")
        # 去掉元数据前缀
        if "【文件来源" in content:
            content = content.split("\n", 1)[-1] if "\n" in content else content
            # 也去掉产品名标签
            if content.startswith("【相关产品"):
                content = content.split("\n", 1)[-1] if "\n" in content else content
        parts.append(
            f"[结果{i}] 来源: {meta['source']}, 第{meta['page']}页\n{content[:500]}"
        )
    return "\n\n---\n\n".join(parts)


# ============================================================
# Agent 主循环
# ============================================================
def agentic_ask(question: str, max_rounds: int = 2) -> Dict:
    """
    Agentic RAG 主入口。
    多轮推理：搜索 → 评估 → 补充 → 综合

    Args:
        question: 用户问题
        max_rounds: 最大推理轮次

    Returns:
        {"answer": str, "sources_used": list, "search_history": list}
    """
    # 带重试和模型降级的 LLM 调用
    import time as time_mod
    all_models = [config.LLM_MODEL] + list(config.LLM_FALLBACK_MODELS)

    def _call_llm(msgs, temp=0.1, max_tok=1500):
        last_err = None
        for model in all_models:
            client = OpenAI(
                api_key=config.LLM_API_KEY,
                base_url=config.LLM_BASE_URL,
                timeout=60.0,
            )
            for attempt in range(3):
                try:
                    return client.chat.completions.create(
                        model=model, messages=msgs,
                        temperature=temp, max_tokens=max_tok,
                    )
                except Exception as e:
                    last_err = e
                    if "429" in str(e) or "503" in str(e) or "busy" in str(e).lower():
                        delay = 2 * (2 ** attempt)
                        logger.warning(f"[{model}] 限流，{delay}s后重试...")
                        time_mod.sleep(delay)
                    else:
                        break  # 不可重试，换下一个模型
            logger.warning(f"[{model}] 不可用，切换...")
        raise last_err or RuntimeError("所有模型不可用")

    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"问题：{question}\n\n"
            f"请先搜索文档库获取相关信息。如果搜索结果中有相关页面，"
            f"用 get_page 读取完整内容。禁止跳过搜索。"
            f"格式：search(\"关键词\", \"文档名\") 或 get_page(\"文档名\", 页码)"
        )},
    ]

    collected_info = []  # 收集的所有搜索结果
    search_history = []

    for round_num in range(max_rounds):
        logger.info(f"=== Agent 第 {round_num + 1} 轮 ===")

        # 调用 LLM（带重试+降级）
        response = _call_llm(messages)
        reply = response.choices[0].message.content
        logger.info(f"LLM 回复 ({len(reply)} chars): {reply[:200]}...")

        # 解析工具调用
        tool_calls = _parse_tool_calls(reply)
        logger.info(f"解析到 {len(tool_calls)} 个工具调用")

        if not tool_calls:
            # 没有工具调用 = LLM 认为信息足够了，返回最终答案
            return {
                "answer": reply,
                "sources_used": collected_info,
                "search_history": search_history,
                "rounds": round_num + 1,
            }

        # 执行工具调用
        tool_results = ""
        for tc in tool_calls:
            if tc["tool"] == "search":
                results = _search(tc["query"], tc.get("source_hint", ""))
                formatted = _format_search_results(results)
                tool_results += f"\nsearch(\"{tc['query']}\", \"{tc.get('source_hint', '')}\") 的结果：\n{formatted}\n"
                collected_info.extend(results)
                search_history.append({
                    "round": round_num + 1,
                    "tool": "search",
                    "query": tc["query"],
                    "source_hint": tc.get("source_hint", ""),
                    "result_count": len(results),
                })
            elif tc["tool"] == "get_page":
                content = _get_page(tc["source"], tc["page"])
                if content:
                    tool_results += f"\nget_page(\"{tc['source']}\", {tc['page']}) 的内容：\n{content[:1000]}\n"
                else:
                    tool_results += f"\nget_page(\"{tc['source']}\", {tc['page']}) ：页面不存在\n"
                search_history.append({
                    "round": round_num + 1,
                    "tool": "get_page",
                    "source": tc["source"],
                    "page": tc["page"],
                })

        messages.append({"role": "assistant", "content": reply})

        # 首轮搜索后：每个搜索查询取 top-1 页面自动读取（保证多样性）
        if round_num == 0:
            auto_pages = set()
            # 按搜索查询分组
            query_groups = {}
            for doc in collected_info:
                sq = doc.get("_query", "")  # 标记来源查询
                if sq not in query_groups:
                    query_groups[sq] = []
                query_groups[sq].append(doc)
            # 每组取 top-2 最高分（不同页面）
            for sq, docs in query_groups.items():
                docs.sort(key=lambda d: d.get("bm25_score", d.get("dense_score", 0)), reverse=True)
                count = 0
                for doc in docs:
                    key = (doc["metadata"]["source"], doc["metadata"]["page"])
                    if key not in auto_pages:
                        auto_pages.add(key)
                        page_text = _get_page(key[0], key[1])
                        if page_text:
                            tool_results += f"\n[自动读取] {key[0]} 第{key[1]}页:\n{page_text[:500]}\n"
                        count += 1
                        if count >= 3:
                            break

        # 压缩本轮新收集的结果
        compressed = _compress_context(
            collected_info, question, max_chunks=20, max_chars_per_chunk=600,
        )
        compressed_text = _format_search_results(compressed[:15])

        messages.append({
            "role": "user",
            "content": f"以下是工具执行结果（已压缩去重）：\n{compressed_text}\n\n请根据以上结果继续搜索或给出最终答案。",
        })

    # 达到最大轮次，强制生成答案
    messages.append({
        "role": "user",
        "content": (
            "请给出最终答案。\n"
            "1. 推荐类：用文档中产品全称，只推一个\n"
            "2. 对比类：先说A再说B，只对比不推荐\n"
            "3. 列举类：确保覆盖所有案例\n"
            "4. 无信息只回'文档中没有提供相关信息'"
        ),
    })
    final_response = _call_llm(messages)

    return {
        "answer": final_response.choices[0].message.content,
        "sources_used": collected_info,
        "search_history": search_history,
        "rounds": max_rounds + 1,
    }
