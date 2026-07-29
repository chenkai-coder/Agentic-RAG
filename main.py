"""
CLI 入口模块
用法:
    python main.py build              # 构建索引
    python main.py ask "你的问题"      # 单次问答
    python main.py eval               # 评估
    python main.py interactive        # 交互式问答
     echo 问题 | python main.py ask   # 管道输入
"""

import sys
import argparse
import locale
import logging

from src.pipeline import get_pipeline
from src import config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ============================================================
# stdin 管道中文解码
# PowerShell 管道中文乱码的根本原因：
#   PowerShell 的 $OutputEncoding 默认不是 UTF-8，
#   导致中文在进入管道前就被转成了 ?。
#   修复方法（在 PowerShell 中执行一次即可）：
#     $OutputEncoding = [Text.UTF8Encoding]::new()
#     [Console]::OutputEncoding = [Text.Encoding]::UTF8
# 本函数尝试从原始字节流中恢复中文，若失败则给出明确提示。
# ============================================================
def _read_stdin() -> str:
    """
    从 stdin 读取管道输入，自动检测编码。
    尝试顺序: UTF-8 → GBK → GB2312 → GB18030 → 系统 locale 编码
    """
    raw = sys.stdin.buffer.read()

    if not raw:
        return ""

    # ---- 优先 UTF-8：检查是否符合 UTF-8 编码规律 ----
    # 中文在 UTF-8 中占 3 字节，首字节 0xE4-0xE9，后续字节 0x80-0xBF
    # 如果原始字节符合这个模式，就是 UTF-8，直接使用
    try:
        utf8_text = raw.decode("utf-8-sig")  # utf-8-sig 自动去除 BOM
        utf8_chinese = sum(1 for c in utf8_text if "一" <= c <= "鿿")
        if utf8_chinese >= 3:  # 至少 3 个中文字
            return utf8_text.strip()
    except (UnicodeDecodeError, LookupError):
        pass

    # ---- UTF-8 失败，尝试中文 ANSI 编码 ----
    candidates = ["gbk", "gb2312", "gb18030"]
    sys_enc = locale.getpreferredencoding()
    for enc in [sys_enc, "cp936"]:
        if enc.lower() not in candidates:
            candidates.append(enc)

    for enc in candidates:
        try:
            text = raw.decode(enc)
            chinese = sum(1 for c in text if "一" <= c <= "鿿")
            if chinese >= 3:
                return text.strip()
        except (UnicodeDecodeError, LookupError):
            continue

    # ---- 兜底：返回 UTF-8 解码结果（即使是乱码） ----
    return raw.decode("utf-8-sig", errors="replace").strip()

    # 所有编码都失败了——很可能是 PowerShell 在管道源头就已经丢了中文
    # 此时 raw bytes 里全是 0x3F (? 的 ASCII 码)
    if best_text:
        question_count = best_text.count("?")
        total = max(len(best_text), 1)
        if question_count / total > 0.3:
            print(
                "\n⚠️  检测到输入中文已丢失（大量 ? 字符）。\n"
                "   这是 PowerShell 的 $OutputEncoding 设置问题。\n"
                "   请在 PowerShell 中先执行以下命令，再重试管道：\n"
                "\n"
                "     $OutputEncoding = [Text.UTF8Encoding]::new()\n"
                "     [Console]::OutputEncoding = [Text.Encoding]::UTF8\n"
                "\n"
                "   或者直接用交互模式（无编码问题）：\n"
                "\n"
                "     python main.py interactive\n",
                file=sys.stderr,
            )

    return best_text.strip() if best_text else ""


def cmd_build(args):
    """构建索引命令"""
    pipeline = get_pipeline()
    pipeline.build(force_rebuild=args.force)


def _print_ask_result(result, verbose):
    """打印单次 RAG 结果"""
    print()
    print("=" * 70)
    print(f"问题: {result['question']}")
    print("-" * 70)
    if result["is_unanswerable"]:
        print(f"⚠️  {result['answer']}")
        if verbose:
            print(f"  (最高相关性分数: {result['max_rerank_score']:.4f})")
    else:
        print(f"回答:\n{result['answer']}")
        if verbose:
            print("-" * 70)
            print(f"检索统计: {result['candidate_count']} 候选 → "
                  f"{len(result['contexts'])} 条证据, "
                  f"最高分={result['max_rerank_score']:.4f}")
            print()
            for i, ctx in enumerate(result["contexts"], 1):
                src = ctx.get("metadata", {}).get("source", "?")
                pg = ctx.get("metadata", {}).get("page", "?")
                score = ctx.get("rerank_score", "?")
                print(f"  [{i}] {src} 第{pg}页 (score={score:.4f})")
    print("=" * 70)
    print()


def cmd_ask(args):
    """单次问答命令（默认使用 Agentic RAG）"""
    question = args.question

    # 如果没给问题参数，从 stdin 读取
    if not question:
        if not sys.stdin.isatty():
            question = _read_stdin()
        else:
            print("请输入问题（输入后按 Ctrl+Z 再回车结束）：")
            lines = []
            try:
                while True:
                    line = input()
                    lines.append(line)
            except EOFError:
                pass
            question = " ".join(lines).strip()

    if not question:
        print("错误：未提供问题。"
              "用法: python main.py ask '你的问题'  或  echo 问题 | python main.py ask")
        return

    if args.fast:
        # Fast: pipeline 单次 RAG（HyDE + Reranker + Detector）
        pipeline = get_pipeline()
        result = pipeline.ask(question, return_details=args.verbose)
        _print_ask_result(result, args.verbose)
    else:
        # 默认：Agentic RAG
        from src.agent import agentic_ask
        print()
        print("=" * 70)
        print(f"问题: {question}")
        print("=" * 70)
        result = agentic_ask(question, max_rounds=args.rounds)
        print()
        print(f"📝 回答 (经过 {result['rounds']} 轮推理):")
        print(result["answer"])
        print()
        print("搜索历史:")
        for h in result["search_history"]:
            if h["tool"] == "search":
                print(f"  🔍 search(\"{h['query']}\", \"{h.get('source_hint', '')}\") → {h['result_count']} 条")
            else:
                print(f"  📄 get_page(\"{h['source']}\", {h['page']})")
        print()


def cmd_eval(args):
    """评估命令"""
    if args.compare:
        from eval.evaluate import run_full_evaluation
        run_full_evaluation()
    else:
        from eval.evaluate import run_evaluation
        run_evaluation(args)


def cmd_agent(args):
    """Agentic RAG 问答命令"""
    question = args.question
    if not question:
        if not sys.stdin.isatty():
            question = _read_stdin()
        else:
            print("请输入问题：")
            question = input().strip()

    if not question:
        print("错误：未提供问题")
        return

    from src.agent import agentic_ask
    print()
    print("=" * 70)
    print(f"问题: {question}")
    print("=" * 70)
    print()

    result = agentic_ask(question, max_rounds=args.rounds)

    print(f"📝 回答 (经过 {result['rounds']} 轮推理):")
    print(result["answer"])
    print()
    print(f"搜索历史:")
    for h in result["search_history"]:
        if h["tool"] == "search":
            print(f"  🔍 search(\"{h['query']}\", \"{h.get('source_hint', '')}\") → {h['result_count']} 条")
        else:
            print(f"  📄 get_page(\"{h['source']}\", {h['page']})")
    print()


def cmd_interactive(args):
    """交互式问答模式"""
    pipeline = get_pipeline()

    # 确保索引已加载
    from src.indexer import load_index
    if not load_index():
        print("索引未构建，正在自动构建...")
        pipeline.build()

    print()
    print("=" * 70)
    print("  长文档 RAG 问答系统 — 交互模式")
    print("  输入问题开始问答，输入 quit / exit / q 退出")
    print("=" * 70)
    print()

    while True:
        try:
            question = input("🤔 你的问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        if args.fast:
            # 快速模式：单次 RAG
            result = pipeline.ask(question)
            print()
            if result["is_unanswerable"]:
                print(f"⚠️  {result['answer']}")
            else:
                print(f"📝 回答:\n{result['answer']}")
            print()
        else:
            # 默认：Agentic RAG
            from src.agent import agentic_ask
            result = agentic_ask(question)
            print()
            print(f"📝 回答 (经过 {result['rounds']} 轮推理):")
            print(result["answer"])
            print()


def main():
    parser = argparse.ArgumentParser(
        description="长文档 RAG 问答系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py build                    构建索引
  python main.py build --force            强制重建索引
  python main.py ask "问题"                单次问答
  python main.py agent "复杂问题"           Agentic RAG（多步推理）
  python main.py ask "问题" --verbose      显示检索详情
  echo 问题 | python main.py ask           管道输入
  python main.py eval                     运行评估
  python main.py interactive              交互模式
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # build 命令
    build_parser = subparsers.add_parser("build", help="构建索引")
    build_parser.add_argument(
        "--force", action="store_true",
        help="强制重建索引（清空已有数据）"
    )

    # ask 命令（默认 Agentic RAG）
    ask_parser = subparsers.add_parser(
        "ask", help="Agentic RAG 问答（默认）",
        epilog="示例: python main.py ask '问题'  |  --fast 快速模式"
    )
    ask_parser.add_argument(
        "question", type=str, nargs="?", default=None,
        help="用户问题（从 stdin 管道输入时可省略）"
    )
    ask_parser.add_argument(
        "--fast", action="store_true",
        help="快速模式：1轮Agentic检索（默认2轮）"
    )
    ask_parser.add_argument(
        "--rounds", type=int, default=2,
        help="Agentic RAG 最大推理轮次（默认2）"
    )
    ask_parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="显示检索详情（仅 fast 模式）"
    )

    # eval 命令
    eval_parser = subparsers.add_parser("eval", help="运行评估")
    eval_parser.add_argument("--fast", action="store_true",
                             help="Fast RAG 模式（默认 Agentic）")
    eval_parser.add_argument("--compare", action="store_true",
                             help="双模式对比评估，自动保存 JSON+MD 报告")

    # agent 命令
    agent_parser = subparsers.add_parser("agent", help="Agentic RAG 多步推理问答")
    agent_parser.add_argument("question", type=str, nargs="?", default=None,
                              help="用户问题")
    agent_parser.add_argument("--rounds", type=int, default=3,
                              help="最大推理轮次（默认3）")

    # interactive 命令
    int_parser = subparsers.add_parser("interactive", help="交互式问答")
    int_parser.add_argument("--fast", action="store_true",
                            help="快速模式：1轮Agentic（默认2轮）")

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "ask":
        cmd_ask(args)
    elif args.command == "agent":
        cmd_agent(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "interactive":
        cmd_interactive(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
