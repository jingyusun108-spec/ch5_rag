"""综合案例：农业病虫害 RAG 问答系统 —— FastAPI 接口服务

接口：
    GET /rag/ask?question=xxx   返回 JSON 问答结果
    GET /                      服务健康检查

运行：
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from rag_agent import ask

app = FastAPI(title="农业病虫害 RAG 问答系统")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件目录（前端页面）
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/rag/ask")
async def rag_ask(question: str = Query(..., description="用户问题")):
    """RAG 问答接口"""
    try:
        answer = ask(question)
        return {"code": 0, "question": question, "answer": answer}
    except Exception as e:
        return {"code": 1, "question": question, "error": str(e)}


@app.get("/")
async def index():
    """返回前端页面"""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
