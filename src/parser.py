"""
PDF 解析模块
使用 PyMuPDF (fitz) 逐页提取文本，保留页码和来源信息。
"""

import fitz  # PyMuPDF
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def parse_pdf(pdf_path: str, source_name: str) -> List[Dict]:
    """
    解析单个 PDF 文件，返回每页的文本和元数据。

    Args:
        pdf_path: PDF 文件的绝对路径
        source_name: 来源标识（如 "产品手册.pdf"）

    Returns:
        List[Dict]: 每页的解析结果
            - page_num: 页码（从 1 开始）
            - text: 该页的纯文本内容
            - source: 来源文件名
    """
    logger.info(f"正在解析 PDF: {source_name} ({pdf_path})")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"无法打开 PDF 文件 {pdf_path}: {e}")
        return []

    pages = []
    total_pages = len(doc)

    for page_idx in range(total_pages):
        page = doc[page_idx]
        text = page.get_text()

        # 跳过空白页
        if not text or not text.strip():
            continue

        pages.append({
            "page_num": page_idx + 1,   # 页码从 1 开始
            "text": text.strip(),
            "source": source_name,
        })

    doc.close()
    logger.info(f"解析完成: {source_name}，共 {total_pages} 页，"
                f"有效页 {len(pages)} 页，总字符数 {sum(len(p['text']) for p in pages)}")

    return pages


def parse_all_pdfs(pdf_files: Dict[str, str]) -> List[Dict]:
    """
    批量解析所有 PDF 文件。

    Args:
        pdf_files: {文件名: 文件路径} 的映射字典

    Returns:
        List[Dict]: 所有 PDF 的逐页解析结果
    """
    all_pages = []
    for source_name, pdf_path in pdf_files.items():
        pages = parse_pdf(pdf_path, source_name)
        all_pages.extend(pages)

    logger.info(f"全部 PDF 解析完成，共 {len(all_pages)} 页有效内容")
    return all_pages
