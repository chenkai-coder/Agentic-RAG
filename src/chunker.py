"""
文本分块模块
自实现递归字符分块器，按 \n\n > \n > 。> ； > ， > 空格 优先级切分，
并在每个块中嵌入来源和页码元数据（Metadata-Enriched Chunking）。
"""

from typing import List, Dict
import re
import logging

from . import config

logger = logging.getLogger(__name__)


def _recursive_split(
    text: str,
    separators: List[str],
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """
    递归字符分割：优先用前面的分隔符切分，若某段仍超长则用下一级分隔符继续切。

    Args:
        text: 待切分文本
        separators: 分隔符列表（优先级从高到低）
        chunk_size: 目标块大小
        chunk_overlap: 重叠大小

    Returns:
        List[str]: 切分后的文本块
    """
    if not separators:
        # 无分隔符了，直接按长度硬切
        chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    sep = separators[0]
    remaining_seps = separators[1:]

    # 用当前分隔符切分
    if sep:
        parts = re.split(f"({re.escape(sep)})", text)
        # 合并分隔符到前面的片段
        merged = []
        buf = ""
        for part in parts:
            buf += part
            if part == sep or not part:
                continue
            merged.append(buf)
            buf = ""
        if buf:
            merged.append(buf)
    else:
        # 空字符串分隔符 = 逐字符切分
        merged = list(text)

    # 合并短片段、切分长片段
    chunks = []
    current = ""

    for part in merged:
        if len(current) + len(part) <= chunk_size:
            current += part
        else:
            # 当前累积的片段可以作为一个块
            if current.strip():
                chunks.append(current)

            # 如果 part 本身超长，递归切分
            if len(part) > chunk_size:
                sub_chunks = _recursive_split(
                    part, remaining_seps, chunk_size, chunk_overlap
                )
                # 与前一个块做 overlap
                if chunks and chunk_overlap > 0:
                    prev = chunks[-1]
                    if len(prev) > chunk_overlap:
                        overlap_text = prev[-chunk_overlap:]
                        if sub_chunks:
                            sub_chunks[0] = overlap_text + sub_chunks[0]
                chunks.extend(sub_chunks)
                current = ""
            else:
                current = part

    if current.strip():
        chunks.append(current)

    return chunks


def chunk_pages(
    pages: List[Dict],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> List[Dict]:
    """
    将逐页文本切分为带元数据的文本块。

    每个 chunk 格式：
    {
        "content": "【文件来源：产品手册.pdf | 第 5 页】\n实际文本...",
        "metadata": {
            "source": "产品手册.pdf",
            "page": 5,
            "chunk_id": "产品手册.pdf_p5_0"
        }
    }

    Args:
        pages: parser.parse_all_pdfs() 的输出
        chunk_size: 每个文本块的最大字符数
        chunk_overlap: 相邻块的重叠字符数

    Returns:
        List[Dict]: 富化元数据的文本块列表
    """
    # 分割优先级：段落 > 换行 > 句号 > 分号 > 逗号 > 空格 > 逐字
    separators = ["\n\n", "\n", "。", "；", "，", " ", ""]

    all_chunks = []
    total_chars = 0
    prev_product_name = ""  # 追踪上一个产品名（用于 spec 页的上下文补充）

    for page in pages:
        source = page["source"]
        page_num = page["page_num"]
        text = page["text"]

        # 产品手册信息稀疏（5700字/49页），用更大chunk保持产品上下文完整
        is_product_manual = "产品手册" in source
        effective_chunk_size = chunk_size * 2 if is_product_manual else chunk_size
        effective_overlap = chunk_overlap * 2 if is_product_manual else chunk_overlap

        page_chunks = _recursive_split(text, separators, effective_chunk_size, effective_overlap)

        # 提取页面摘要（首句或标题），用于短 chunk 上下文富化
        page_summary = ""
        first_line = text.strip().split("\n")[0] if text.strip() else ""
        if first_line and len(first_line) >= 4:
            page_summary = first_line[:60]

        for idx, chunk_text in enumerate(page_chunks):
            # 富化内容：来源 + 页码
            context_prefix = f"【文件来源：{source} | 第 {page_num} 页】"
            # 短 chunk（<80字正文）自动补充页面上下文（普适规则，不限文档类型）
            if len(chunk_text.strip()) < 80 and page_summary:
                context_prefix += f"\n【本页主题：{page_summary}】"

            enriched_content = f"{context_prefix}\n{chunk_text}"

            all_chunks.append({
                "content": enriched_content,
                "metadata": {
                    "source": source,
                    "page": page_num,
                    "chunk_id": f"{source}_p{page_num}_{idx}",
                }
            })
            total_chars += len(enriched_content)

    logger.info(f"分块完成: {len(pages)} 页 → {len(all_chunks)} 个文本块，"
                f"平均每块 {total_chars // max(len(all_chunks), 1)} 字符")

    return all_chunks
