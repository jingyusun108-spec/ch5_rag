"""例 5.3-5 批量加载文件（DirectoryLoader）

验证 DirectoryLoader：按 glob 模式批量加载指定目录下不同格式文件。
依赖：pip install pypdf docx2txt
运行：python ex_5_3_5_directory_loader.py
"""
import os
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)

KB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "knowledge_base")


if __name__ == "__main__":
    # 加载 PDF（递归）
    pdf_loader = DirectoryLoader(
        path=KB,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        recursive=True,
        show_progress=True,
    )
    pdf_docs = pdf_loader.load()

    # 加载 txt（指定中文编码）
    txt_loader = DirectoryLoader(
        path=KB,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        recursive=True,
    )
    txt_docs = txt_loader.load()

    # 加载 docx（本知识库暂无，示例保留写法）
    docx_loader = DirectoryLoader(
        path=KB,
        glob="**/*.docx",
        loader_cls=Docx2txtLoader,
        recursive=True,
    )
    docx_docs = docx_loader.load()

    # 合并所有文档
    all_docs = pdf_docs + txt_docs + docx_docs
    print(f"批量加载文档总数：{len(all_docs)}")
    print(f"  PDF 文档数：{len(pdf_docs)}")
    print(f"  TXT 文档数：{len(txt_docs)}")
    print(f"  DOCX 文档数：{len(docx_docs)}")

    # 遍历查看每个文档的来源和类型
    for doc in all_docs:
        print(f"来源：{doc.metadata.get('source')} | 内容长度：{len(doc.page_content)} 字符")
