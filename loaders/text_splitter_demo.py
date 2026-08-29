"""文本分割演示（RecursiveCharacterTextSplitter）。

验证递归字符分割器核心能力：
1. 按 Token 精准计数（tiktoken + cl100k_base）
2. 中文专属分隔符 + 重叠 + 起始索引 + 首尾空白清理

运行：python text_splitter_demo.py
"""
import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1. 定义按 Token 计数的函数（适配 OpenAI 模型）
def tiktoken_len(text: str) -> int:
    """按 token 计数。cl100k_base 是 OpenAI 主流分词方案。"""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(text))


# 2. 待分割的 Document 对象（含元数据，RAG 主流使用方式）
docs = [
    Document(
        page_content="""RecursiveCharacterTextSplitter是LangChain最通用的文本分割器，
核心通过递归分隔符实现语义无损分割，支持字符/Token两种计数方式。
生产环境推荐按Token计数，同时配置中文专属分隔符，提升分割效果。""",
        metadata={"source": "LangChain开发文档", "type": "文本分割器"},
    )
]

# 3. 初始化生产环境级分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,                                # 每个片段最大 50 Token
    chunk_overlap=5,                              # 相邻片段重叠 5 Token（10%）
    separators=["\n\n", "\n", "。", "，", " "],   # 中文分隔符
    length_function=tiktoken_len,                 # 替换为 Token 计数
    add_start_index=True,                         # 添加起始索引，便于溯源
    strip_whitespace=True,                        # 清理首尾空白
)

# 4. 分割 Document 对象（保留元数据，推荐）
split_docs = text_splitter.split_documents(docs)

for i, doc in enumerate(split_docs):
    print(f"【片段{i + 1}】")
    print(f"内容：{doc.page_content}")
    print(f"元数据：{doc.metadata}（含起始索引）")
    print(f"Token数：{tiktoken_len(doc.page_content)}\n")
