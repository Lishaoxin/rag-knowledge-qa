# 智能知识库问答系统（RAG）

上传文档后，用自然语言检索资料并提问。网页流式展示回答，结束时给出块级来源，方便回查原文。项目使用 FastAPI、Chroma、BGE Embedding、BM25、BGE Reranker 和 DeepSeek。

## 试用示例

仓库自带 4 份示例语料。启动后可以问“接入 RAG 后，大模型可以引用什么？”。在归档评测中，这道题的最终检索来源为 `rag.txt_0`。也可以上传 PDF、DOCX、TXT 或 Markdown 文档后再提问。[示例语料](docs/)

## 本地启动

在仓库根目录创建虚拟环境并安装依赖。以下命令以 PowerShell 为例。

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path -LiteralPath .\.env)) {
    Copy-Item -LiteralPath .\.env.example -Destination .\.env
}
```

在 `.env` 中填入 `DEEPSEEK_API_KEY` 和 `SILICONFLOW_API_KEY`，然后启动服务。

```powershell
.\venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

打开 [问答网页](http://127.0.0.1:8000/) 或 [API 文档](http://127.0.0.1:8000/docs)。首次启动会向量化 `docs/` 下的 TXT 文档，随后复用本地 Chroma 索引。问答、首次建库和上传文档需要调用模型服务。

## 执行流程

1. **入库**。提取文本，按句切分为最多 300 字符的片段，批量向量化后写入 Chroma。上传同名文档时更新对应片段，并清理多余旧块。
2. **检索**。向量召回 20 段，BM25 筛出 5 段，BGE Reranker 选出 1 段。文本和块 ID 随检索结果一起传递。
3. **回答**。将本轮检索片段与会话历史交给 DeepSeek，网页逐段显示答案，最后附上本轮来源。

网页使用 `POST /ask_stream` 和 `POST /upload_file`。程序调用也可使用 `POST /ask` 获取 JSON 答案和来源，或使用 `POST /upload` 提交文本。[API 入口](main.py) · [RAG 核心](rag_agent.py) · [网页](index.html)

## 检索评测

在 4 份语料、1,133 个分块和 20 道固定题上，按人工复核的标注证据计算 Top-1 命中率。

| 检索阶段 | Top-1 标注证据命中 |
| --- | --- |
| 向量召回 | 14/20（70%） |
| 向量召回 + BM25 | 13/20（65%） |
| 向量召回 + BM25 + Rerank | 16/20（80%） |

BM25 对照中，两组均命中 16/20。加入 BM25 后，每题送入重排的候选数从 20 段降到 5 段。题集中有 17 道来自 Python 教程，结果主要反映这批语料上的检索表现。[评测方法与复现](evaluation/README.md) · [人工复核报告](evaluation/reports/eval_result_v1_reviewed.json) · [BM25 对照报告](evaluation/reports/eval_compare_bm25_f3dbeb697a84.json)
