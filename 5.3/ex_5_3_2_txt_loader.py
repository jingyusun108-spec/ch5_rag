"""例 5.3-2 加载本地 txt 文档（TextLoader）

验证 TextLoader：加载纯文本文档为 List[Document]，自动保留 source 元数据。
运行：python ex_5_3_2_txt_loader.py
"""
import os
from langchain_community.document_loaders import TextLoader

# 知识库根目录（相对本文件定位）
KB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "knowledge_base")


if __name__ == "__main__":
    file_path = os.path.join(KB, "rag_intro.txt")

    # 初始化加载器：指定文件路径，设置编码为 utf-8（解决中文乱码）
    loader = TextLoader(
        file_path=file_path,
        encoding="utf-8",  # 中文场景必配，避免乱码
    )

    # 核心加载方法：返回 List[Document]
    docs = loader.load()

    print(f"加载文档数：{len(docs)}")
    for doc in docs:
        print(f"元数据：{doc.metadata}")  # 自动包含 source（文件路径）
        print(f"文档内容（前 200 字符）：\n{doc.page_content[:200]}...")
