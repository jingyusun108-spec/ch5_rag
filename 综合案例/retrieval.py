"""两阶段检索模块（粗召回 + 精排）

bge-m3 对短中文查询的语义区分度有限，直接相似度检索容易误配。
生产标准做法：先向量粗召回 Top-K，再用重排模型精排，取最相关片段。

- 第一阶段：RedisVectorStore.similarity_search_with_score 粗召回 k 条
- 第二阶段：BAAI/bge-reranker-v2-m3 精排，按 relevance_score 降序，
            仅保留 score >= RERANK_THRESHOLD 的片段

依赖：requests（requirements.txt 已含）
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
# 重排分数阈值：bge-reranker 输出 0~1，>= 0.5 视为相关
RERANK_THRESHOLD = float(os.getenv("RERANK_THRESHOLD", "0.5"))


def rerank(query: str, docs, top_n: int = 3):
    """对候选文档精排，返回 (文档, 分数) 列表，按分数降序，仅保留达标项。

    参数:
        query: 用户查询文本
        docs: 候选 Document 列表（来自粗召回）
        top_n: 精排后最多保留条数

    返回:
        list[tuple[Document, float]]，按 relevance_score 降序
    """
    if not docs:
        return []

    base = os.getenv("OPENAI_BASE_URL")
    key = os.getenv("OPENAI_API_KEY")
    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": [d.page_content for d in docs],
        "top_n": min(top_n, len(docs)),
    }
    resp = requests.post(
        f"{base}/rerank",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    scored = []
    for item in data.get("results", []):
        idx = item["index"]
        score = item.get("relevance_score", 0.0)
        if score >= RERANK_THRESHOLD:
            scored.append((docs[idx], score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def search_with_rerank(vector_store, query: str, recall_k: int = 200, top_n: int = 3):
    """两阶段检索入口：粗召回 + 精排。

    说明：bge-m3 对术语密集型短条目的向量区分度有限（正确条目可能排
    在百名开外），故粗召回 k 取大值（默认 200，覆盖小库全量），
    最终由 reranker 精排 + 阈值过滤保证精准。

    参数:
        vector_store: RedisVectorStore 实例
        query: 用户查询
        recall_k: 粗召回候选数（默认 200，覆盖小库全量）
        top_n: 精排后保留数

    返回:
        list[Document]，精排后命中的文档（可能为空）
    """
    # 第一阶段：粗召回（取距离最近的 recall_k 条）
    candidates = vector_store.similarity_search_with_score(query, k=recall_k)
    candidate_docs = [d for d, _ in candidates]
    # 第二阶段：精排 + 阈值过滤
    hits = rerank(query, candidate_docs, top_n=top_n)
    return [d for d, _ in hits]
