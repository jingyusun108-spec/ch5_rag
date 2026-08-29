"""例 5.3-3 加载 PDF 文档（PyPDFLoader）

验证 PyPDFLoader：每页返回一个 Document，元数据含 page/source；
并结合 load_and_split 一站式「加载 + 分割」。
依赖：pip install pypdf
运行：python ex_5_3_3_pdf_loader.py
"""
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

KB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "knowledge_base")


if __name__ == "__main__":
    file_path = os.path.join(KB, "rag_intro.pdf")

    # 初始化 PDF 加载器
    loader = PyPDFLoader(file_path=file_path)

    # 加载 PDF：每页返回一个 Document，元数据含 page（页码）、source（路径）
    docs = loader.load()
    print(f"PDF 总页数：{len(docs)}")
    print(f"第 1 页内容：\n{docs[0].page_content[:300]}...\n")
    print(f"第 1 页元数据：{docs[0].metadata}")  # {'source': '...', 'page': 0}

    # 可选：加载后直接分割（结合文本分割器）
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = loader.load_and_split(text_splitter=text_splitter)
    print(f"\n加载并分割后片段数：{len(split_docs)}")
