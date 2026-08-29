"""从网页加载文档（WebBaseLoader）。

验证 WebBaseLoader：爬取网页正文，自动过滤广告/导航，保留 URL/标题元数据。
运行：python web_loader_demo.py
"""
from langchain_community.document_loaders import WebBaseLoader

# 配置请求头：解决部分网站反爬（模拟浏览器请求）
custom_headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def load_single(url: str):
    """方式 1：爬取单个网页"""
    loader = WebBaseLoader(
        web_path=url,
        requests_kwargs={"headers": custom_headers},  # 可传入 requests.get 所有参数
    )
    docs = loader.load()
    if docs:
        print(f"网页标题：{docs[0].metadata.get('title', 'N/A')}")
        print(f"网页正文：\n{docs[0].page_content[:200]}...\n")
    return docs


def load_batch(urls: list[str]):
    """方式 2：批量爬取多个网页"""
    loader = WebBaseLoader(web_paths=urls, requests_kwargs={"headers": custom_headers})
    batch_docs = loader.load()
    print(f"批量爬取网页数：{len(batch_docs)}")
    for i, doc in enumerate(batch_docs):
        print(f"第{i + 1}个网页正文：\n{doc.page_content[:200]}...\n")
    return batch_docs


if __name__ == "__main__":
    # 示例 URL（可替换为任意可访问网页）
    load_single("https://api-docs.siliconflow.cn/docs/api/chat-completions-post")

    # load_batch([
    #     "https://api-docs.siliconflow.cn/docs/api/messages-post",
    #     "https://api-docs.siliconflow.cn/docs/api/embeddings-post",
    # ])
