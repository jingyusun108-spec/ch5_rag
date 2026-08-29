"""检索器创建与基础问答链路演示。

验证 as_retriever() 将向量库转换为检索器，并打通完整 RAG 链路：
检索 → 格式化上下文 → 填充提示词 → LLM 生成 → 输出

前置条件：需运行 Redis Stack，且先执行 redis_vector_store_demo 完成入库
运行：python retriever_rag_demo.py
"""
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

INDEX_NAME = "agri_wheat"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# 聊天模型
llm = init_chat_model("Qwen/Qwen3-8B", model_provider="openai", temperature=0.1)

# 向量模型
em_llm = OpenAIEmbeddings(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
)

# 向量数据库
config = RedisConfig(
    index_name=INDEX_NAME,
    redis_url=REDIS_URL,
    distance_metric="COSINE",
)
vector_store = RedisVectorStore(config=config, embeddings=em_llm)

# 初始化向量检索器：返回 Top3 相关文档
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

prompt = ChatPromptTemplate.from_template("""
请基于提供的上下文信息回答用户的问题，不要编造信息。如果上下文没有相关答案，请说明"暂无相关信息"。

上下文：{context}

用户问题：{question}
""")


def rag_pipeline(question: str) -> str:
    """RAG 流程的逐步执行"""
    # 1. 检索相关文档
    retrieved_docs = retriever.invoke(question)
    # 2. 格式化上下文（将文档列表转换为字符串）
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    # 3. 填充提示词模板
    formatted_prompt = prompt.invoke({"context": context, "question": question})
    # 4. 调用 LLM 生成回答
    llm_response = llm.invoke(formatted_prompt)
    # 5. 输出
    return llm_response.content


if __name__ == "__main__":
    print("======测试1======")
    query1 = "小麦拔节期如何施肥？"
    print(rag_pipeline(query1))

    print("\n======测试2======")
    query2 = "玉米拔节期如何施肥？"
    print(rag_pipeline(query2))
