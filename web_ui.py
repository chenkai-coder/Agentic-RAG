"""
Web UI — 双模式 RAG 对话界面
Fast: pipeline 单次检索(HyDE+Reranker) | Agentic: 多轮推理
"""

import gradio as gr
import json
import os
from datetime import datetime
from src.agent import agentic_ask

SAVE_DIR = "conversations"
os.makedirs(SAVE_DIR, exist_ok=True)


def ask_question(question: str, mode: str, history: list):
    if not question.strip():
        return "", history
    try:
        if "Agentic" in mode:
            r = agentic_ask(question, max_rounds=2)
            display = f"**🤖 Agentic RAG**（{r['rounds']}轮）:\n\n{r['answer']}"
        else:
            from src.pipeline import RAGPipeline
            r = RAGPipeline().ask(question)
            prefix = "**⚡ Fast RAG (pipeline)**:\n\n"
            display = prefix + (f"⚠️ {r['answer']}" if r["is_unanswerable"] else r["answer"])
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": display})
    except Exception as e:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": f"**❌ 错误**: {str(e)[:300]}"})
    return "", history


def save_conversation(history: list, mode: str):
    if not history:
        return "无对话内容可保存"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fn = os.path.join(SAVE_DIR, f"conversation_{ts}.jsonl")
    with open(fn, "w", encoding="utf-8") as f:
        for i in range(0, len(history), 2):
            q = history[i]["content"] if i < len(history) else ""
            raw = history[i+1]["content"] if i+1 < len(history) else ""
            clean = raw
            for p in ["**🤖 Agentic RAG**", "**⚡ Fast RAG (pipeline)**:\n\n"]:
                if p in clean:
                    clean = clean.split(p, 1)[-1].strip()
                    if clean.startswith("（"):
                        clean = clean.split(":\n\n", 1)[-1] if ":\n\n" in clean else clean
                    break
            f.write(json.dumps({
                "q_id": f"conv_{ts}_{i//2+1:02d}",
                "query_type": "answerable" if "没有提供相关信息" not in clean else "unanswerable",
                "question": q,
                "answer": clean.strip(),
                "mode": mode,
            }, ensure_ascii=False) + "\n")
    return f"✅ 已保存到 {fn}"


def clear_history():
    return []


with gr.Blocks(title="RAG 问答系统") as app:
    gr.Markdown("# 📚 长文档 RAG 问答系统\n**Agentic RAG** (多轮推理) + **Fast RAG** (pipeline单次检索)")
    with gr.Row():
        with gr.Column(scale=1):
            mode = gr.Radio(["Agentic RAG（多轮推理）", "Fast RAG（pipeline）"], value="Agentic RAG（多轮推理）", label="模式")
            save_btn = gr.Button("💾 保存对话", variant="secondary")
            save_msg = gr.Textbox(interactive=False, show_label=False)
            clear_btn = gr.Button("🗑️ 清空", variant="secondary")
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(label="对话", height=550)
            question = gr.Textbox(placeholder="输入问题，按 Enter 发送...", show_label=False)
    question.submit(ask_question, [question, mode, chatbot], [question, chatbot])
    save_btn.click(save_conversation, [chatbot, mode], [save_msg])
    clear_btn.click(clear_history, outputs=[chatbot])

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
