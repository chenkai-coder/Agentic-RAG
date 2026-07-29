"""
评估模块 — 支持 Fast RAG 和 Agentic RAG 双模式

指标：
  1. 页码召回率：gold_chunks 标注的页码是否被检索到
  2. 内容准确性：答案中是否包含关键实体词
  3. Unanswerable 分类准确率
"""

import json
import os
import logging
import time
from datetime import datetime
from typing import List, Dict

from src import config

logger = logging.getLogger("eval")

# 内容评估：每道题的关键实体词
CONTENT_CHECKS = {
    "q_01": {"keywords": ["公安", "综合执法", "广电"], "min_len": 20},
    "q_02": {"keywords": ["海口", "美兰", "烟台", "黄冈", "GSM", "凉山", "陕西", "高考"], "min_len": 200},
    "q_03": {"keywords": ["检测", "授时", "告警", "应用场景"], "min_len": 150},
    "q_04": {"keywords": ["欺骗式", "干扰检测", "定位", "室外", "检测时间", "测向"], "min_len": 100},
    "q_05": {"keywords": ["没有提供相关信息"], "min_len": 5},
}


def _check_content(q_id: str, answer: str) -> dict:
    cfg = CONTENT_CHECKS.get(q_id, {})
    if not cfg:
        return {"score": 1.0, "hits": [], "misses": []}
    hits = [kw for kw in cfg["keywords"] if kw in answer]
    misses = [kw for kw in cfg["keywords"] if kw not in answer]
    return {
        "score": len(hits) / max(len(cfg["keywords"]), 1),
        "hits": hits,
        "misses": misses,
        "len_ok": len(answer) >= cfg["min_len"],
    }


def load_qa_pairs(path: str = None) -> List[Dict]:
    path = path or config.QA_PAIRS_PATH
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def _parse_pages(gold_chunks: List[Dict]) -> set:
    pages = set()
    for c in gold_chunks:
        p = str(c.get("page", ""))
        for part in p.replace("，", ",").split(","):
            part = part.strip()
            if "-" in part:
                try:
                    s, e = part.split("-")
                    for pg in range(int(s), int(e) + 1):
                        pages.add(pg)
                except ValueError:
                    pass
            else:
                try:
                    pages.add(int(part))
                except ValueError:
                    pass
    return pages


def evaluate_fast(qa_pairs: List[Dict]) -> Dict:
    """Fast RAG 评估：pipeline 单次检索（HyDE+Reranker+Hybrid Search）。"""
    from src.pipeline import RAGPipeline
    p = RAGPipeline()
    ans = [q for q in qa_pairs if q["query_type"] == "answerable"]
    unans = [q for q in qa_pairs if q["query_type"] == "unanswerable"]

    r = {"mode": "fast", "total": len(qa_pairs),
         "page_hits": 0, "page_total": 0,
         "unans_correct": 0, "unans_total": len(unans), "details": []}

    for qa in ans:
        gold = _parse_pages(qa.get("gold_chunks", []))
        docs = p.retrieve(qa["question"], top_k=config.FINAL_TOP_K)
        found = {d["metadata"]["page"] for d in docs}
        pg_hits = gold & found
        full = p.ask(qa["question"])
        ct = _check_content(qa["q_id"], full["answer"])
        r["page_hits"] += len(pg_hits)
        r["page_total"] += len(gold)
        r["details"].append({
            "q_id": qa["q_id"], "type": "answerable",
            "gold_pages": sorted(gold), "found_pages": sorted(found),
            "page_hits": sorted(pg_hits),
            "page_recall": len(pg_hits) / max(len(gold), 1),
            "content_score": ct["score"],
            "content_hits": ct["hits"], "content_misses": ct["misses"],
            "answer": full["answer"],
        })

    for qa in unans:
        full = p.ask(qa["question"])
        ok = "没有提供相关信息" in full["answer"]
        if ok:
            r["unans_correct"] += 1
        r["details"].append({
            "q_id": qa["q_id"], "type": "unanswerable", "correct": ok,
            "answer": full["answer"],
        })

    r["page_recall"] = r["page_hits"] / max(r["page_total"], 1)
    r["unans_acc"] = r["unans_correct"] / max(r["unans_total"], 1)
    r["content_avg"] = sum(d.get("content_score", 1) for d in r["details"] if d["type"] == "answerable") / max(len(ans), 1)
    return r


def evaluate_agentic(qa_pairs: List[Dict]) -> Dict:
    from src.agent import agentic_ask
    ans = [q for q in qa_pairs if q["query_type"] == "answerable"]
    unans = [q for q in qa_pairs if q["query_type"] == "unanswerable"]

    r = {"mode": "agentic", "total": len(qa_pairs),
         "page_hits": 0, "page_total": 0,
         "unans_correct": 0, "unans_total": len(unans), "details": []}

    for qa in ans:
        gold = _parse_pages(qa.get("gold_chunks", []))
        result = agentic_ask(qa["question"], max_rounds=2)
        found = {d["metadata"]["page"] for d in result.get("sources_used", [])}
        pg_hits = gold & found
        ct = _check_content(qa["q_id"], result["answer"])
        r["page_hits"] += len(pg_hits)
        r["page_total"] += len(gold)
        r["details"].append({
            "q_id": qa["q_id"], "type": "answerable",
            "gold_pages": sorted(gold), "found_pages": sorted(found),
            "page_hits": sorted(pg_hits),
            "page_recall": len(pg_hits) / max(len(gold), 1),
            "content_score": ct["score"],
            "content_hits": ct["hits"], "content_misses": ct["misses"],
            "rounds": result["rounds"],
            "searches": len([h for h in result["search_history"] if h["tool"] == "search"]),
            "answer": result["answer"],
        })

    for qa in unans:
        result = agentic_ask(qa["question"], max_rounds=2)
        ok = "没有提供相关信息" in result["answer"]
        if ok:
            r["unans_correct"] += 1
        r["details"].append({
            "q_id": qa["q_id"], "type": "unanswerable", "correct": ok,
            "answer": result["answer"],
        })

    r["page_recall"] = r["page_hits"] / max(r["page_total"], 1)
    r["unans_acc"] = r["unans_correct"] / max(r["unans_total"], 1)
    r["content_avg"] = sum(d.get("content_score", 1) for d in r["details"] if d["type"] == "answerable") / max(len(ans), 1)
    return r


def _count_cases(answer: str) -> int:
    """统计列举类答案中的案例数。"""
    import re
    return len(re.findall(r'\d+\.\s', answer))


def _count_citations(answer: str) -> int:
    """统计引用标注数。"""
    import re
    return len(re.findall(r'\[.*?第\s*\d+', answer))


def run_evaluation(args=None):
    fast = args.fast if args else False
    name = "Fast RAG (1轮)" if fast else "Agentic RAG (2轮)"
    print(f"\n{'='*60}\n  RAG 评估 — {name}\n{'='*60}")

    qa = load_qa_pairs()
    t0 = time.time()
    results = evaluate_fast(qa) if fast else evaluate_agentic(qa)
    t = time.time() - t0

    # 完整性指标
    for d in results["details"]:
        d["cases"] = _count_cases(d.get("answer", ""))
        d["citations"] = _count_citations(d.get("answer", ""))
        d["answer_len"] = len(d.get("answer", ""))

    print(f"\n⏱ {t:.0f}s  |  页码: {results['page_recall']:.0%}  |  内容: {results.get('content_avg',0):.0%}  |  Unans: {results.get('unans_acc',0):.0%}")
    print(f"\n{'─'*60}")

    for d in results["details"]:
        if d["type"] == "answerable":
            pg = d.get("page_recall", 0)
            ct = d.get("content_score", 0)
            print(f"\n{'='*60}")
            print(f"[{d['q_id']}] 页面 {pg:.0%} | 内容 {ct:.0%} | {d['answer_len']}字 | {d['cases']}案例 | {d['citations']}引用")
            print(f"  Gold页: {d['gold_pages']} | 命中: {d['page_hits']}")
            if d.get("content_misses"):
                print(f"  缺失词: {d['content_misses']}")
            print(f"  答案:\n{d['answer']}")
        else:
            ok = "✅" if d["correct"] else "❌"
            print(f"\n{'='*60}")
            print(f"{ok} [{d['q_id']}] unanswerable | 正确={d['correct']}")
            print(f"  答案 ({d['answer_len']}字):\n{d['answer']}")

    print(f"\n{'='*60}\n  评估完成\n{'='*60}\n")

    # 自动保存报告
    _save_reports(results, name)


def run_full_evaluation():
    """运行双模式评估并生成对比报告。"""
    qa = load_qa_pairs()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"\n{'='*60}")
    print(f"  双模式评估对比")
    print(f"{'='*60}")

    results = {}
    for fn, label in [(evaluate_fast, "Fast RAG (pipeline)"), (evaluate_agentic, "Agentic RAG (2轮)")]:
        print(f"\n--- {label} ---")
        t0 = time.time()
        r = fn(qa)
        t = time.time() - t0
        for d in r["details"]:
            d["cases"] = _count_cases(d.get("answer", ""))
            d["citations"] = _count_citations(d.get("answer", ""))
            d["answer_len"] = len(d.get("answer", ""))
        results[label] = r
        print(f"  页码: {r['page_recall']:.0%} | 内容: {r.get('content_avg',0):.0%} | Unans: {r.get('unans_acc',0):.0%} | {t:.0f}s")

        # 保存单个报告
        mode_key = "fast" if "Fast" in label else "agentic"
        _save_reports(r, label, ts, mode_key)

    # 生成对比报告
    a = results["Agentic RAG (2轮)"]
    f_r = results["Fast RAG (pipeline)"]
    _save_comparison(a, f_r, ts)

    print(f"\n  报告已保存至 eval_results/{ts}_*.json|.md")
    print(f"{'='*60}\n")


def _save_reports(r: dict, name: str, ts: str = None, mode_key: str = None):
    """保存 JSON + MD 报告。"""
    if ts is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    if mode_key is None:
        mode_key = "fast" if "Fast" in name else "agentic"
    os.makedirs("eval_results", exist_ok=True)

    # JSON
    with open(f"eval_results/{ts}_{mode_key}.json", "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2, default=str)

    # MD
    with open(f"eval_results/{ts}_{mode_key}.md", "w", encoding="utf-8") as f:
        f.write(f"# RAG 评估 — {name}\n\n")
        f.write(f"**时间**: {ts}  \n")
        f.write(f"**页码召回**: {r['page_recall']:.0%} ({r['page_hits']}/{r['page_total']})  \n")
        f.write(f"**内容准确性**: {r.get('content_avg',0):.0%}  \n")
        f.write(f"**Unanswerable**: {r.get('unans_acc',0):.0%} ({r['unans_correct']}/{r['unans_total']})  \n\n---\n\n")
        for d in r["details"]:
            if d["type"] == "answerable":
                f.write(f"## {d['q_id']} | 页面{d['page_recall']:.0%} | 内容{d['content_score']:.0%}\n")
                if d.get("content_misses"):
                    f.write(f"缺失: {', '.join(d['content_misses'])}  \n")
                f.write(f"\n{d['answer']}\n\n---\n\n")
            else:
                ok = "✅" if d["correct"] else "❌"
                f.write(f"## {d['q_id']} unanswerable {ok}\n\n{d['answer']}\n\n---\n\n")


def _save_comparison(a: dict, f_r: dict, ts: str):
    """保存双模式对比报告。"""
    with open(f"eval_results/{ts}_comparison.md", "w", encoding="utf-8") as fp:
        fp.write(f"# 双模式评估对比报告\n\n")
        fp.write(f"**时间**: {ts}  \n")
        fp.write(f"**Fast RAG**: pipeline (HyDE+Reranker+Hybrid Search)  \n")
        fp.write(f"**Agentic RAG**: 多轮推理 (ReAct+BM25直通+上下文压缩)  \n\n")
        fp.write(f"## 总览\n\n")
        fp.write(f"| 指标 | Fast (pipeline) | Agentic (2轮) | 提升 |\n|------|:---:|:---:|:---:|\n")
        fp.write(f"| 页码召回率 | {f_r['page_recall']:.0%} | **{a['page_recall']:.0%}** | — |\n")
        fp.write(f"| 内容准确性 | {f_r.get('content_avg',0):.0%} | **{a.get('content_avg',0):.0%}** | — |\n")
        fp.write(f"| Unanswerable | {f_r.get('unans_acc',0):.0%} | **{a.get('unans_acc',0):.0%}** | — |\n\n")
        fp.write(f"## 逐题对比\n\n")
        fp.write(f"| 问题 | Fast页面 | Fast内容 | A页面 | A内容 | Fast字 | A字 |\n")
        fp.write(f"|------|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for i in range(5):
            aa = a["details"][i]; ff = f_r["details"][i]
            fp.write(f"| {aa['q_id']} | {ff['page_recall']:.0%} | {ff.get('content_score',0):.0%} | {aa['page_recall']:.0%} | {aa.get('content_score',0):.0%} | {ff.get('answer_len',0)} | {aa.get('answer_len',0)} |\n")
        fp.write(f"\n## 逐题答案\n\n")
        for i in range(5):
            aa = a["details"][i]; ff = f_r["details"][i]
            fp.write(f"### {aa['q_id']}\n\n**Fast**:\n\n{ff['answer']}\n\n**Agentic**:\n\n{aa['answer']}\n\n---\n\n")
