# 《第5章 检索增强生成》代码落地项目

本目录是《第5章 检索增强生成》教材全部例题代码的落地实现。
教材代码以截图形式嵌入，部分截图缺失，本目录代码**依据教材正文的详细文字描述重建**，功能与教材一致，可直接运行学习。

## 目录结构

```
ch5_rag/
├── README.md                  # 本文件（含完整知识点总结 + 课后习题答案）
├── requirements.txt           # 依赖清单
├── .env.example               # 环境变量模板（复制为 .env 填写密钥）
├── gen_pdf.py                 # 生成示例 PDF 知识库（供 PDF 加载器/综合案例使用）
├── knowledge_base/            # 知识库（真实农业病虫害手册 + 结构化条目）
│   ├── agri_pest.pdf          # 农业病虫害手册（可打印细化版，9页）
│   ├── manual_detailed.docx   # 同手册 Word 版（内容最完整）
│   ├── pest_manual_structured.txt  # 结构化条目版（每条病虫害独立成行，检索主数据源）
│   ├── rag_intro.txt          # IPM 问答式农业知识（33条）
│   ├── rag_intro.docx         # 同上 Word 版
│   └── rag_intro.pdf          # 同上 PDF 版
├── 5.1/                       # 概述（理论，无例题代码）
│   └── README.md
├── 5.2/                       # 向量存储
│   ├── ex_5_2_1_embedding.py        # 例5.2-1 文本向量化
│   ├── ex_5_2_2_redis_store.py      # 例5.2-2 Redis向量入库和查询（幂等去重）
│   └── ex_5_2_3_retriever_rag.py    # 例5.2-3 检索器创建与基础问答链路
├── 5.3/                       # 文档检索
│   ├── ex_5_3_1_text_splitter.py    # 例5.3-1 递归字符分割器实操
│   ├── ex_5_3_2_txt_loader.py       # 例5.3-2 加载本地 txt
│   ├── ex_5_3_3_pdf_loader.py       # 例5.3-3 加载本地 PDF
│   ├── ex_5_3_4_web_loader.py       # 例5.3-4 加载网页
│   ├── ex_5_3_5_directory_loader.py # 例5.3-5 批量加载文件
│   └── ex_5_3_6_mysql_loader.py     # 例5.3-6 加载 MySQL 数据
│   └── 综合案例/
    ├── rag_agent.py           # 例5.4-1 LangChain Agent + Redis RAG 问答系统
    ├── rag_graph.py           # 课后综合应用题：LangGraph 条件分支工作流
    ├── retrieval.py           # 两阶段检索模块（全量粗召回 + bge-reranker 精排）
    ├── kb_loader.py           # 共享知识库加载模块（加载/分割/幂等ID）
    ├── main.py                # FastAPI 接口服务
    └── static/index.html      # 前端交互页面
```

## 环境准备

```bash
# 1. 安装依赖（清华镜像源）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 复制 .env.example 为 .env，填入硅基流动 API Key
#    OPENAI_API_KEY=sk-xxxxx
#    OPENAI_BASE_URL=https://api.siliconflow.cn/v1

# 3. 启动 Redis Stack（向量库 + 记忆依赖）
#    Docker: docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
#    或本机安装 Redis Stack 并启动

# 4. 生成示例 PDF 知识库（供 5.3-3 与综合案例加载）
python gen_pdf.py
```

## 模型配置规范

- 聊天模型：`Qwen/Qwen3-8B`（硅基流动），统一 `init_chat_model(..., model_provider="openai")`
- 向量模型：`BAAI/bge-m3`（硅基流动，OpenAIEmbeddings）
- 重排模型：`BAAI/bge-reranker-v2-m3`（硅基流动，两阶段检索精排用）
- temperature：精准问答/工具调用 0.1~0.3
- 向量库：Redis Stack，`distance_metric="COSINE"`

> ⚠️ **重要**：`BAAI/bge-m3` 为「句子/段落检索」优化，对**术语密集的短中文条目**（如「小麦赤霉病」）向量区分度极差（实测正确条目排全量 #89，无关项排 #0）。本项目综合案例采用**两阶段检索**：先向量粗召回大候选集（默认 200，覆盖小库全量），再用 `bge-reranker-v2-m3` 全量精排 + 阈值 0.5 过滤，实测所有病虫害条目精准命中（0.98+），无关查询（量子计算机）正确无命中。

---

## 知识点速查（全章精华）

### 1. 大模型四大固有痛点
| 痛点 | 表现 |
|------|------|
| 知识截止 | 无法获取训练截止后新信息，重训成本极高 |
| 知识幻觉 | 概率生成优先保证通顺，编造不存在的事实/数据 |
| 无法访问私有知识 | 预训练数据仅公网公开内容 |
| 上下文窗口限制 | 单次输入 Token 上限，无法一次性读入海量文档 |

### 2. RAG 核心架构
**先检索、再生成**：检索系统负责「死记硬背」，模型负责「理解 + 整合 + 生成」。
- 根除幻觉（可溯源）/ 实时更新（秒级）/ 接入私有知识 / 突破上下文限制

### 3. RAG vs 模型微调
| 维度 | RAG | 微调 |
|------|-----|------|
| 定位 | 给模型动态知识 | 教模型固定做事方法 |
| 适用 | 知识高频更新、私有知识、事实准确性高 | 输出风格定制、推理逻辑优化、知识固定低延迟 |
| 关系 | 互补，可「微调基座 + RAG 动态知识」结合 |

### 4. 向量化三组件（核心三角）
- **Embeddings**（生产者）：文本 → 高维向量
- **VectorStore**（管理者）：向量的持久化、索引、相似度计算
- **Retriever**（消费者）：标准化检索接口，向量库与业务层桥梁

### 5. Redis 向量库（生产首选）
- `RedisConfig`：`redis_url`（必传）、`index_name`（必传）、`distance_metric`（COSINE/L2/IP）、`embedding_dimensions`、`key_prefix`
- `RedisVectorStore(config=..., embeddings=...)`
- **幂等去重**：`hashlib.md5(page_content).hexdigest()` 生成唯一 ID，`add_documents(ids=ids)` 自动覆盖

### 6. as_retriever() 检索参数
| 参数 | 说明 |
|------|------|
| search_type | similarity（默认）/ mmr / similarity_score_threshold |
| search_kwargs.k | 返回文档数（默认 4） |
| search_kwargs.filter | 元数据过滤 |
| search_kwargs.fetch_k | MMR 候选池（默认 20） |
| search_kwargs.lambda_mult | MMR 多样性权重（0~1，默认 0.5） |
| search_kwargs.score_threshold | 相似度阈值 |

### 6.1 两阶段检索（生产增强，本项目综合案例采用）
术语密集短条目向量区分度弱的问题，用「全量粗召回 + 精排」解决：
1. **粗召回**：`vector_store.similarity_search_with_score(query, k=200)` 取大候选集
2. **精排**：调用 `BAAI/bge-reranker-v2-m3` 全量重排，按 relevance_score 降序，仅保留 `>= 0.5` 的片段

实现见 `综合案例/retrieval.py`，Agent 版与 LangGraph 版均已接入（recall_k=100 起，小库可全量）。

> 关键经验：**bge-m3 的向量排名对术语条目不可靠，reranker 才是最终决策者**。小知识库（<500 片段）直接全量精排最稳妥，避免粗召回漏掉正确候选。

### 7. 文本分割器（RecursiveCharacterTextSplitter 生产首选）
| 参数 | 说明 |
|------|------|
| chunk_size | 片段最大尺寸（默认 1000） |
| chunk_overlap | 相邻片段重叠（默认 200，建议 chunk_size 的 10%~20%） |
| separators | 递归分隔符（中文：`["\n\n","\n","。","，","；","、"," ",""]`） |
| length_function | 长度计数（生产推荐 tiktoken Token 计数） |
| keep_separator / add_start_index / strip_whitespace | 分隔符保留 / 起始索引溯源 / 首尾空白清理 |

### 8. 文档加载器选型
| 加载器 | 场景 |
|--------|------|
| TextLoader | 本地纯文本 .txt/.md/.csv |
| PyPDFLoader | 本地 PDF（每页一个 Document） |
| Docx2txtLoader | 本地 Word .docx |
| WebBaseLoader | 网络网页正文 |
| UnstructuredMarkdownLoader | Markdown 结构保留 |
| DirectoryLoader | 目录批量加载（glob 匹配） |
| SQLDatabaseLoader | 关系型数据库表数据 |

### 9. RAG 五大核心步骤
加载（Document Loader）→ 分割（Text Splitter）→ 向量化存储（Embeddings + VectorStore）→ 检索（Retriever）→ 生成（LLM）

---

## 课后习题答案

### 一、选择题
1. B（知识截止）
2. C（先检索再生成）
3. C（彻底替代模型微调 —— 错误，二者互补）
4. B（向量化）
5. B（RecursiveCharacterTextSplitter）
6. B（余弦相似度 COSINE）
7. A（as_retriever()）
8. A（文本内容哈希生成唯一 ID）
9. C（客服话术统一 —— 输出风格固定，适合微调）
10. B（文档加载器）

### 二、问答题（要点）

**1. 四大核心痛点与 RAG 解决机制**
- 知识截止 → RAG 外部知识库实时更新，无需重训
- 知识幻觉 → RAG 强制基于检索片段生成，可溯源可验证
- 无法访问私有知识 → RAG 将企业文档向量化入私有向量库
- 上下文窗口限制 → RAG 先检索筛选最相关片段，再喂给模型

**2. RAG 与微调的核心区别**
- RAG：知识外置、动态更新、可溯源、成本低，适合「给知识」
- 微调：参数更新、风格定制、低延迟，适合「教做事」
- 配合：微调基座能力 + RAG 提供动态事实知识

**3. RAG 五大核心步骤**
- ①文档加载（Document Loader，多源异构统一为 Document）
- ②文本分割（Text Splitter，语义无损切块）
- ③向量化存储（Embeddings + VectorStore，文本→向量入库）
- ④知识检索（Retriever，语义相似度 Top-K）
- ⑤回答生成（LLM，基于检索上下文生成）

### 三、综合应用题
完整实现见 `综合案例/rag_agent.py`（Agent 版）与 `综合案例/rag_graph.py`（LangGraph 条件分支版）：
- (1) 架构：加载器 → 分割器 → Embeddings → Redis 向量库 → Retriever → LLM；选型理由见知识点速查
- (2) 文档预处理：PyPDFLoader 加载 + RecursiveCharacterTextSplitter（Token 计数 + 中文分隔符）+ MD5 幂等去重
- (3) 向量存储：RedisVectorStore + OpenAIEmbeddings + as_retriever（score_threshold=0.7, k=3）
- (4) 条件分支：LangGraph StateGraph，`route` 函数判断 context 是否为空，分流 RAG 生成 / 直接生成
