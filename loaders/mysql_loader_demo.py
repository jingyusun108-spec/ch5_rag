"""加载 MySQL 数据（SQLDatabaseLoader）。

验证 SQLDatabaseLoader：将关系型数据库表数据读取为 Document，向量化入库 Redis。
前置条件：需准备 MySQL 库 agri_db 表 wheat_tech（建表 SQL 见文件底部注释）。

运行：python mysql_loader_demo.py
"""
import os
import hashlib
import tiktoken
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_openai import OpenAIEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain_community.document_loaders import SQLDatabaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_core.documents import Document

load_dotenv()

MYSQL_URL = os.getenv(
    "MYSQL_URL",
    "mysql+pymysql://root:root@192.168.100.101:3306/agri_db",
)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# 向量模型
em_llm = OpenAIEmbeddings(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
)


def tiktoken_len(text: str) -> int:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(text))


def custom_page_content_mapper(row) -> str:
    """将所有业务字段值拼接成一个字符串作为文档内容（排除主键 id）"""
    return "。".join(str(getattr(row, key)) for key in row.keys() if key != "id")


def custom_metadata_mapper(row) -> dict:
    """将所有字段作为元数据"""
    return {key: getattr(row, key) for key in row.keys()}


if __name__ == "__main__":
    # 1. 初始化数据库连接
    engine = create_engine(MYSQL_URL)
    db = SQLDatabase(engine)

    # 2. 初始化 MySQL 加载器（含自定义内容/元数据映射）
    loader = SQLDatabaseLoader(
        query="SELECT id, variety, sowing_dosage, fertilizer_note, disease_note FROM wheat_tech;",
        db=db,
        page_content_mapper=custom_page_content_mapper,
        metadata_mapper=custom_metadata_mapper,
    )

    # 3. 加载数据，得到 Document 列表
    docs = loader.load()
    print(f"读取小麦农技数据库记录条数：{len(docs)}")
    if docs:
        print("单条文档内容：", docs[0].page_content)
        print("元数据：", docs[0].metadata)

    # 4. 文本分割（本章标准分割器）
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,
        chunk_overlap=25,
        separators=["\n\n", "\n", "。", "，", " "],
        length_function=tiktoken_len,
        strip_whitespace=True,
    )
    split_docs = splitter.split_documents(docs)

    # 5. 初始化向量库，批量入库（幂等 MD5 去重）
    redis_config = RedisConfig(
        index_name="wheat_mysql_know",
        redis_url=REDIS_URL,
        distance_metric="COSINE",
    )
    vector_store = RedisVectorStore(config=redis_config, embeddings=em_llm)

    doc_ids = [
        hashlib.md5(d.page_content.strip().encode("utf-8")).hexdigest()
        for d in split_docs
    ]
    vector_store.add_documents(documents=split_docs, ids=doc_ids)
    print("MySQL 农业数据向量化入库完成！")


# ==================== 建表 SQL（供参考）====================
"""
CREATE DATABASE IF NOT EXISTS agri_db DEFAULT CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;
USE agri_db;

CREATE TABLE IF NOT EXISTS wheat_tech (
  id INT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
  variety VARCHAR(64) NOT NULL COMMENT '小麦品种名称',
  sowing_dosage VARCHAR(128) NOT NULL COMMENT '播种用量/播种规范',
  fertilizer_note TEXT COMMENT '施肥管理要点',
  disease_note TEXT COMMENT '病害防治方案'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='小麦种植技术知识库';

INSERT INTO wheat_tech (variety, sowing_dosage, fertilizer_note, disease_note)
VALUES
('济麦44', '旱地每亩播种10-11kg，水浇地9kg，10月中旬适期播种',
 '拔节期亩施钾肥10kg、尿素8kg；灌浆期叶面喷施磷酸二氢钾',
 '小麦锈病：发病初期喷施三唑酮，7天一次，连续2次；避免田间积水降低湿度'),
('山农28', '壤土地块每亩11-12kg，黏土增加1kg，浅播3cm以内',
 '基肥搭配腐熟农家肥，返青肥少施氮肥，防止徒长倒伏',
 '纹枯病：播种前拌种，春季拔节期喷井冈霉素，清除田间杂草减少病菌'),
('鲁麦23', '晚播地块每亩加大至13kg，采用条播均匀下种',
 '扬花期禁止大量施氮肥，易造成穗粒早衰，以磷钾肥为主',
 '白粉病：高湿天气易发，通风降湿，发病用醚菌酯喷雾防治');
"""
