# 农业病虫害 RAG 问答系统

基于 **LangChain + Redis + LangGraph** 构建的农业病虫害检索增强生成（RAG）问答系统。

面向农业种植场景，将病虫害防治手册、农技资料等文档向量化后存入 Redis 向量库，用户提问时通过「向量粗召回 + 重排精排」的两阶段检索定位相关资料，再由 LLM 基于检索结果生成专业回答；无匹配资料时自动降级为通用知识科普。

## 核心特性

- **两阶段检索**：`bge-m3` 向量粗召回 + `bge-reranker-v2-m3` 精排 + 阈值过滤，显著提升术语密集短条目的命中精度
- **条件分支工作流**：基于 LangGraph 实现「有检索结果走 RAG 生成 / 无结果直接生成」的自动路由
- **幂等入库**：以正文 MD5 作为文档唯一 ID，重复入库自动覆盖，避免冗余
- **双入口**：标准 LangChain Agent 版 + LangGraph 状态图版，两种实现方式对照
- **Web 服务**：FastAPI 后端 + 原生前端页面，开箱即用的问答界面

## 技术栈

| 层 | 选型 |
|------|------|
| LLM | Qwen/Qwen3-8B（硅基流动） |
| 向量模型 | BAAI/bge-m3 |
| 重排模型 | BAAI/bge-reranker-v2-m3 |
| 向量库 | Redis Stack（COSINE 相似度） |
| 编排 | LangChain / LangGraph |
| 服务 | FastAPI + Uvicorn |

## 目录结构

```
ch5_rag/
├── app/                        # 端到端问答系统
│   ├── rag_agent.py            # LangChain Agent 版（检索工具 + 分支判断）
│   ├── rag_graph.py            # LangGraph 条件分支工作流
│   ├── retrieval.py            # 两阶段检索模块（粗召回 + 重排精排）
│   ├── kb_loader.py            # 知识库加载模块（加载/分割/幂等ID）
│   ├── main.py                 # FastAPI 接口服务
│   └── static/index.html       # 前端交互页面
├── examples/                   # 核心能力示例
│   ├── embedding_demo.py       # 文本向量化
│   ├── redis_vector_store_demo.py  # Redis 向量入库与相似度检索
│   └── retriever_rag_demo.py   # 检索器 + 基础 RAG 链路
├── loaders/                    # 文档加载器示例
│   ├── text_splitter_demo.py   # 递归字符分割器
│   ├── txt_loader_demo.py      # 加载 TXT
│   ├── pdf_loader_demo.py      # 加载 PDF
│   ├── web_loader_demo.py      # 加载网页
│   ├── directory_loader_demo.py    # 批量加载目录
│   └── mysql_loader_demo.py    # 加载 MySQL 数据
├── docs/                       # 背景说明（RAG 概述）
│   └── README.md
├── knowledge_base/             # 知识库（农业病虫害手册 + 结构化条目）
├── gen_pdf.py                  # 生成示例 PDF 知识库
├── requirements.txt            # 依赖清单
└── .env.example                # 环境变量模板
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，填写：

```bash
OPENAI_API_KEY=sk-xxxxx
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
CHAT_MODEL=Qwen/Qwen3-8B
EMBEDDING_MODEL=BAAI/bge-m3
RERANK_MODEL=BAAI/bge-reranker-v2-m3
REDIS_URL=redis://:your_password@your_redis_host:6379
```

### 3. 启动 Redis Stack

```bash
# Docker 方式
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

### 4. 启动 Web 服务

```bash
cd app
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

浏览器访问 `http://127.0.0.1:8000/`，即可在界面提问，例如：

- 「水稻稻瘟病怎么防治？」
- 「小麦赤霉病怎么防治？」
- 「番茄灰霉病怎么治？」

## 两阶段检索设计

`bge-m3` 为「句子/段落检索」优化，对术语密集的短中文条目（如「小麦赤霉病」）向量区分度有限，直接相似度检索容易误配。本项目采用生产标准的**两阶段检索**：

1. **粗召回**：`similarity_search_with_score` 取大候选集（默认 200，覆盖小库全量）
2. **精排**：调用 `bge-reranker-v2-m3` 全量重排，按 relevance_score 降序，仅保留 `>= 0.5` 的片段

实现见 `app/retrieval.py`。实测所有病虫害条目精准命中（0.98+），无关查询正确无命中。

> 关键经验：**向量模型的排名对术语条目不可靠，重排模型才是最终决策者**。小知识库（<500 片段）直接全量精排最稳妥。

## 许可

MIT License
