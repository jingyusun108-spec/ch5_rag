"""基于 LangChain Agent + Redis 的农业病虫害 RAG 问答系统。

核心流程：
1. 批量加载知识库（PDF/TXT/DOCX）→ 分割 → 向量化入 Redis
2. 用户提问后，Agent 自动调用检索工具 search_agri_knowledge
3. 检索结果判断：
   - 命中相似度达标的农技资料 → 基于知识库生成专业回答
   - 无匹配资料 → LLM 直接通用科普回答

运行：python rag_agent.py（首次会自动向量化入库，随后演示问答）
"""
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_redis import RedisConfig, RedisVectorStore
from langchain.tools import tool
from langchain.agents import create_agent

from kb_loader import load_documents, split_documents, build_ids

load_dotenv()

# ===== 基础配置 =====
INDEX_NAME = "Pest_Disease_knowledge"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# ===== 初始化核心组件 =====
# 大语言模型
llm = init_chat_model("Qwen/Qwen3-8B", model_provider="openai", temperature=0.1)

# 向量模型
em_llm = OpenAIEmbeddings(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
)

# 向量库
config = RedisConfig(
    index_name=INDEX_NAME,
    redis_url=REDIS_URL,
    distance_metric="COSINE",
)
vector_store = RedisVectorStore(config=config, embeddings=em_llm)


# ===== 向量化存储 =====
def save_documents():
    """批量加载知识库 → 分割 → 向量化入库（幂等 MD5 去重）"""
    docs = load_documents()
    s_docs = split_documents(docs)
    ids = build_ids(s_docs)
    vector_store.add_documents(documents=s_docs, ids=ids)
    return len(s_docs)


# ===== 自定义检索工具（提供给 Agent 调用）=====
@tool
def search_agri_knowledge(question: str) -> str:
    """检索农业种植、病虫害私有知识库，仅返回相似度达标的农技资料。

    参数:
        question: 用户农技提问

    返回:
        返回匹配文档拼接文本，无匹配返回"未找到相关私有资料"
    """
    # 两阶段检索：粗召回 + bge-reranker 精排 + 阈值过滤
    from retrieval import search_with_rerank
    hits = search_with_rerank(vector_store, question, recall_k=100, top_n=3)
    if not hits:
        return "未在私有知识库中找到与您问题相关的资料。"
    return "\n".join([d.page_content for d in hits])


# ===== Agent 提示词：规定分支判断逻辑 =====
SYSTEM_PROMPT = """你是专业农业技术员，拥有知识库检索工具 search_agri_knowledge。
请根据用户问题，结合私有知识库，回答问题。
1. 用户提问必须先调用工具检索私有农技手册；
2. 如果工具返回"未找到相关私有资料"，说明无相关私有资料，请基于通用农业知识简答；
3. 如果工具返回具体资料，严格依据检索资料作答，禁止编造；
4. 回答简洁贴合田间实操，不冗余。"""

# ===== 创建标准 LangChain Agent =====
agent = create_agent(llm, tools=[search_agri_knowledge], system_prompt=SYSTEM_PROMPT)


def ask(question: str) -> str:
    """调用 Agent 回答问题，返回最终回答文本"""
    result = agent.invoke({"messages": [("user", question)]})
    return result["messages"][-1].content


if __name__ == "__main__":
    # 先入库
    try:
        n = save_documents()
        print(f"向量存储已完成，共入库 {n} 个片段。\n")
    except Exception as e:
        print(f"向量存储失败：{e}\n")

    # 测试 1：库内存在资料
    print("=" * 50)
    print("测试 1：库内存在资料")
    print("=" * 50)
    res1 = ask("小麦黄斑病防治方案")
    print("回答 1：", res1)

    # 测试 2：无匹配私有资料
    print("\n" + "=" * 50)
    print("测试 2：无匹配私有资料")
    print("=" * 50)
    res2 = ask("热带火龙果北方露天种植技术")
    print("回答 2：", res2)
