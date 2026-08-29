"""基于 LangGraph 的 RAG 条件分支工作流。

实现「有相关检索结果则基于 RAG 生成回答，无则直接生成」的条件分支。

流程（StateGraph）：
  START → retrieve（检索）→ 条件判断 route：
    - 有结果 → generate_rag（RAG 生成）
    - 无结果 → generate_direct（直接生成）
  → END

运行：python rag_graph.py（首次会自动向量化入库）
"""
import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from kb_loader import load_documents, split_documents, build_ids

load_dotenv()

INDEX_NAME = "Pest_Disease_knowledge"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

llm = init_chat_model("Qwen/Qwen3-8B", model_provider="openai", temperature=0.1)
em_llm = OpenAIEmbeddings(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
)
config = RedisConfig(index_name=INDEX_NAME, redis_url=REDIS_URL, distance_metric="COSINE")
vector_store = RedisVectorStore(config=config, embeddings=em_llm)

# 检索器：两阶段检索（粗召回 + bge-reranker 精排 + 阈值过滤）
# 注：langchain-redis 未实现 similarity_score_threshold，改用重排模型精排
def _retrieve(question: str, recall_k: int = 100, top_n: int = 3):
    from retrieval import search_with_rerank
    return search_with_rerank(vector_store, question, recall_k=recall_k, top_n=top_n)

RAG_PROMPT = ChatPromptTemplate.from_template("""请基于提供的上下文信息回答用户的问题，不要编造信息。
上下文：{context}

用户问题：{question}
""")

DIRECT_PROMPT = ChatPromptTemplate.from_template("""你是专业农业技术员，请基于通用农业知识简洁回答用户问题。
用户问题：{question}
""")


# ===== 状态定义 =====
class RAGState(TypedDict):
    question: str
    context: str
    answer: str


# ===== 文档预处理（复用 kb_loader 共享模块）=====
def save_documents():
    docs = load_documents()
    s_docs = split_documents(docs)
    ids = build_ids(s_docs)
    vector_store.add_documents(documents=s_docs, ids=ids)
    return len(s_docs)


# ===== 节点 =====
def retrieve(state: RAGState) -> RAGState:
    """检索节点：从向量库检索相关文档，拼为上下文字符串"""
    docs = _retrieve(state["question"])
    context = "\n".join([d.page_content for d in docs])
    return {"context": context}


def route(state: RAGState) -> str:
    """条件路由：有检索结果走 RAG，无结果走直接生成"""
    if state.get("context", "").strip():
        return "generate_rag"
    return "generate_direct"


def generate_rag(state: RAGState) -> RAGState:
    """RAG 生成：基于检索上下文回答"""
    prompt = RAG_PROMPT.invoke({"context": state["context"], "question": state["question"]})
    answer = llm.invoke(prompt).content
    return {"answer": answer}


def generate_direct(state: RAGState) -> RAGState:
    """直接生成：无匹配资料时用通用知识回答"""
    prompt = DIRECT_PROMPT.invoke({"question": state["question"]})
    answer = llm.invoke(prompt).content
    return {"answer": answer}


# ===== 构建图 =====
graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve)
graph.add_node("generate_rag", generate_rag)
graph.add_node("generate_direct", generate_direct)

graph.add_edge(START, "retrieve")
graph.add_conditional_edges(
    "retrieve", route,
    {"generate_rag": "generate_rag", "generate_direct": "generate_direct"},
)
graph.add_edge("generate_rag", END)
graph.add_edge("generate_direct", END)

app = graph.compile()


def ask(question: str) -> str:
    """执行工作流并返回最终回答"""
    result = app.invoke({"question": question})
    return result["answer"]


if __name__ == "__main__":
    try:
        n = save_documents()
        print(f"向量存储已完成，共入库 {n} 个片段。\n")
    except Exception as e:
        print(f"向量存储失败：{e}\n")

    # 场景 1：库内命中 → 走 RAG 分支
    print("=" * 50)
    print("场景 1：有相关检索结果（走 RAG 分支）")
    print("=" * 50)
    print(ask("水稻稻瘟病怎么防治？"))

    # 场景 2：无匹配 → 走直接生成分支
    print("\n" + "=" * 50)
    print("场景 2：无相关检索结果（走直接生成分支）")
    print("=" * 50)
    print(ask("热带火龙果北方露天种植技术"))
