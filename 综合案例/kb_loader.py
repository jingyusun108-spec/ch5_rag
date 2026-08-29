"""知识库加载模块（综合案例共享）

批量加载 knowledge_base 目录下所有 PDF / TXT / DOCX 文档，
统一分割 + 幂等去重，供 rag_agent.py 与 rag_graph.py 复用。

覆盖教材 5.3 节的三种加载器：PyPDFLoader / TextLoader / Docx2txtLoader。
"""
import os
import hashlib
import tiktoken
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 知识库根目录（相对本文件定位到 ../knowledge_base）
KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "knowledge_base")


def tiktoken_len(text: str) -> int:
    """按 token 分词计数"""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(text))


def load_documents():
    """批量加载知识库所有 pdf / txt / docx 文档，返回 List[Document]。

    使用 DirectoryLoader 递归扫描 knowledge_base 目录，
    按扩展名分别用对应加载器加载后合并。
    """
    docs = []
    # PDF
    docs += DirectoryLoader(
        path=KB_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader, recursive=True,
    ).load()
    # TXT（指定 UTF-8 编码，避免中文乱码）
    docs += DirectoryLoader(
        path=KB_DIR, glob="**/*.txt", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}, recursive=True,
    ).load()
    # DOCX
    docs += DirectoryLoader(
        path=KB_DIR, glob="**/*.docx", loader_cls=Docx2txtLoader, recursive=True,
    ).load()
    return docs


def split_documents(documents):
    """递归字符分割：chunk 150 token、重叠 20，中文分隔符。

    说明：知识库中「结构化条目」（pest_manual_structured.txt）每条病虫害
    独立成行（约 90~110 token），故 chunk_size 取 150，确保每条目独立成 chunk，
    避免多条合并导致向量语义被稀释。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=20,
        separators=["\n\n", "\n", "。", "，", "；", "、", " "],
        length_function=tiktoken_len,
        add_start_index=True,
        strip_whitespace=True,
    )
    return splitter.split_documents(documents)


def build_ids(documents):
    """MD5 幂等去重：以正文内容生成唯一 ID"""
    return [
        hashlib.md5(d.page_content.strip().encode("utf-8")).hexdigest()
        for d in documents
    ]
