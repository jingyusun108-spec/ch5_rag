"""例 5.2-1 文本向量化

验证文本向量化（Vectorization）核心知识点：
1. OpenAIEmbeddings 将文本转换为固定长度向量
2. 语义相近的文本（苹果/香蕉）在向量空间距离更近

运行：python ex_5_2_1_embedding.py
"""
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# 向量模型初始化（硅基流动，需显式指定 base_url / api_key）
em_llm = OpenAIEmbeddings(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),  # 指定向量模型
)


def cos_similarity(a, b):
    """计算两个向量的余弦相似度（0~1，越接近 1 越相似）"""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb)


if __name__ == "__main__":
    texts = ["苹果", "香蕉", "汽车"]

    # 向量计算：批量生成向量
    embeddings = em_llm.embed_documents(texts)

    for text, embedding in zip(texts, embeddings):
        print(f"文本: {text}")
        print(f"向量: {embedding[:5]}... (共 {len(embedding)} 维)")
        print()

    # 相似度对比：验证「苹果 vs 香蕉」比「苹果 vs 汽车」更相似
    sim_apple_banana = cos_similarity(embeddings[0], embeddings[1])
    sim_apple_car = cos_similarity(embeddings[0], embeddings[2])
    print("=" * 50)
    print(f"余弦相似度  苹果 vs 香蕉 = {sim_apple_banana:.4f}")
    print(f"余弦相似度  苹果 vs 汽车 = {sim_apple_car:.4f}")
    print("结论：语义相近（苹果/香蕉）的向量距离更近，相似度更高。")
