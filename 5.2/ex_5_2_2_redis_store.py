"""例 5.2-2 Redis 向量入库和查询基础示例

验证 RedisVectorStore 两大核心能力：
1. 文本向量化后持久化入库
2. 语义相似度检索（similarity_search_with_score，分数越小越相似）

前置条件：需运行 Redis Stack（8.0+）
运行：python ex_5_2_2_redis_store.py
"""
import os
import hashlib
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_core.documents import Document

load_dotenv()

INDEX_NAME = "agri_wheat"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# 向量模型
em_llm = OpenAIEmbeddings(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
)

# Redis 向量库配置
config = RedisConfig(
    index_name=INDEX_NAME,          # 索引唯一名称
    redis_url=REDIS_URL,            # 连接地址
    distance_metric="COSINE",       # 余弦相似度
)

vector_store = RedisVectorStore(config=config, embeddings=em_llm)


if __name__ == "__main__":
    texts = [
        "小麦抗旱播种每亩10-12公斤",
        "小麦拔节期每亩施用钾肥10公斤增强抗逆",
        "玉米密植每亩4000-4500株搭配滴灌",
    ]

    # 幂等性去重：按文本内容生成全局唯一 MD5 ID
    docs = [Document(page_content=text) for text in texts]
    ids = [hashlib.md5(doc.page_content.strip().encode("utf-8")).hexdigest() for doc in docs]

    # 添加向量（相同 ID 会自动覆盖，避免重复）
    vector_store.add_documents(documents=docs, ids=ids)
    print("向量入库完成。\n")

    # 查询向量：返回文档 + 相似度分数（距离，越小越相似）
    res = vector_store.similarity_search_with_score(query="小麦播种用量", k=4)
    for doc, score in res:
        print(f"相似度 {score:.2f}  内容：{doc.page_content}")
