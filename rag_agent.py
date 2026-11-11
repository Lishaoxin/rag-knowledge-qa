"""RAG 核心：文档解析、分块、向量化、检索、生成和入库。"""

import io
import json
import os
import time
from pathlib import Path

import chromadb
import docx
import pdfplumber
import requests
from docx.table import Table
from docx.text.paragraph import Paragraph
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY")

client = chromadb.PersistentClient(path=str(BASE_DIR / "chroma_db"))
collection = client.get_or_create_collection("knowledge_base")


def parse_pdf(content_bytes):
    with pdfplumber.open(io.BytesIO(content_bytes)) as pdf:
        text = ""
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text


def parse_docx(content_bytes):
    doc = docx.Document(io.BytesIO(content_bytes))
    parts = []

    for block in doc.iter_inner_content():
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if text:
                parts.append(text)

        elif isinstance(block, Table):
            for row in block.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    parts.append(" | ".join(cells))

    return "\n".join(parts)


def chunk_by_sentence(text, max_chunk=300):
    if max_chunk <= 0:
        raise ValueError("max_chunk 必须大于 0")

    text = text.replace("！", "。").replace("？", "。").replace("\n", "。")
    sentences = text.split("。")
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current = ""

    for s in sentences:
        sentence = s + "。"  # 把句号也算入长度

        if len(current) + len(sentence) <= max_chunk:
            current += sentence
        else:
            if current:
                chunks.append(current)

            # 单句过长：反复切走前 max_chunk 个字符
            while len(sentence) > max_chunk:
                chunks.append(sentence[:max_chunk])
                sentence = sentence[max_chunk:]

            current = sentence

    if current:
        chunks.append(current)

    return chunks


def embed(text):
    resp = requests.post(
        "https://api.siliconflow.cn/v1/embeddings",
        headers={"Authorization": f"Bearer {SILICONFLOW_API_KEY}"},
        json={"model": "BAAI/bge-large-zh-v1.5", "input": text},
        timeout=(10, 120)
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def embed_batch(texts, batch_size=20):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_ok = False

        for attempt in range(3):  # 总共尝试 3 次
            try:
                resp = requests.post(
                    "https://api.siliconflow.cn/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {SILICONFLOW_API_KEY}"
                    },
                    json={
                        "model": "BAAI/bge-large-zh-v1.5",
                        "input": batch
                    },
                    timeout=(10, 120)
                )
                resp.raise_for_status()

                data = resp.json()
                if "data" in data:
                    for item in data["data"]:
                        all_embeddings.append(item["embedding"])

                    batch_ok = True
                    break

            except Exception as e:
                print(
                    f"批次 {i // batch_size} 失败"
                    f"（第 {attempt + 1} 次）：{e}"
                )

            time.sleep(3)

        if not batch_ok:
            raise Exception(f"批次 {i // batch_size} 连续失败 3 次")

        time.sleep(1)
        print(f"已处理 {min(i + batch_size, len(texts))}/{len(texts)} 块")

    return all_embeddings


def keyword_search(query, documents, top_n=10):
    # ① 把所有文档「分词」（简单版：按字符切，中文可用）
    tokenized_docs = [list(doc) for doc in documents]   # 每个字符当一个"词"

    # ② 建 BM25 索引
    bm25 = BM25Okapi(tokenized_docs)

    # ③ 查询
    tokenized_query = list(query)
    scores = bm25.get_scores(tokenized_query)   # 每个文档一个「关键词得分」

    # ④ 返回得分最高的 top_n 个（文档 + 得分）
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def rerank(query, documents, top_n=3):
    resp = requests.post(
        "https://api.siliconflow.cn/v1/rerank",
        headers={"Authorization": f"Bearer {SILICONFLOW_API_KEY}"},
        json={
            "model": "BAAI/bge-reranker-v2-m3",
            "query": query,
            "documents": documents,
            "top_n": top_n
        },
        timeout=(10, 120)
    )
    resp.raise_for_status()

    data = resp.json()
    results = data["results"]

    indices = [result["index"] for result in results]
    docs = [documents[index] for index in indices]

    return docs, indices


def hybrid_search(query, top_n=20, mid_n=5):
    if collection.count() == 0:
        raise ValueError("知识库为空，请先上传文档")

    # ① 向量检索：找出 20 段候选资料
    q_vec = embed(query)
    result = collection.query(
        query_embeddings=[q_vec],
        n_results=top_n
    )
    candidates = result["documents"][0]
    candidate_ids = result["ids"][0]
    if not candidates:
        raise ValueError("没有检索到可用资料，请检查知识库")

    # ② BM25：保留关键词得分最高的 5 段
    kw_ranked = keyword_search(query, candidates, top_n=mid_n)

    mid_docs = []
    mid_ids = []

    for idx, score in kw_ranked:
        mid_docs.append(candidates[idx])
        mid_ids.append(candidate_ids[idx])

    # ③ Rerank：从这 5 段中选出 1 段
    reranked_docs, reranked_indices = rerank(
        query, mid_docs, top_n=1
    )

    best_doc = reranked_docs[0]
    best_id = mid_ids[reranked_indices[0]]

    return best_doc, best_id


def generate(history, context):
    resp = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
        json={
            "model": "deepseek-flash",
            "temperature": 0.3,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是一个知识库问答助手。只回答最新一条用户消息，不要重新回答历史问题。\n"
                        "历史对话仅用于理解当前问题中的指代和上下文。\n"
                        "根据本轮检索资料回答；资料不足时，明确说明无法根据现有资料回答，不要编造。\n"
                        f"\n本轮检索资料：\n{context}"
                    )
                }
            ] + history
        },
        timeout=(10, 120)
    )
    resp.raise_for_status()

    return resp.json()["choices"][0]["message"]["content"]


def generate_stream(history, context):
    resp = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
        json={
            "model": "deepseek-flash",
            "temperature": 0.3,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是一个知识库问答助手。只回答最新一条用户消息，不要重新回答历史问题。\n"
                        "历史对话仅用于理解当前问题中的指代和上下文。\n"
                        "根据本轮检索资料回答；资料不足时，明确说明无法根据现有资料回答，不要编造。\n"
                        f"\n本轮检索资料：\n{context}"
                    )
                }
            ] + history,
            "stream": True
        },
        stream=True,
        timeout=(10, 120)
    )

    try:
        resp.raise_for_status()
        finished = False

        for line in resp.iter_lines():
            if line:
                line = line.decode("utf-8")

                if line.startswith("data: "):
                    data_str = line[6:]

                    if data_str == "[DONE]":
                        finished = True
                        break

                    chunk = json.loads(data_str)
                    content = chunk["choices"][0]["delta"].get("content")

                    if content:
                        yield content

        if not finished:
            raise ValueError("模型输出未完整结束")

    finally:
        resp.close()


def delete_extra_chunks(filename, new_ids):
    if not new_ids:
        raise ValueError("新文档没有可用内容，无法更新")

    stored_ids = collection.get(include=[])["ids"]
    extra_ids = []

    for chunk_id in stored_ids:
        saved_filename = chunk_id.rsplit("_", 1)[0]

        if saved_filename == filename and chunk_id not in new_ids:
            extra_ids.append(chunk_id)

    if extra_ids:
        collection.delete(ids=extra_ids)

def add_document(filename, text):
    """两个上传接口共用同一套切块、向量化和更新流程。"""
    chunks = chunk_by_sentence(text, max_chunk=300)
    if not chunks:
        raise ValueError("文档没有可用文本，请检查文件内容")

    embeddings = embed_batch(chunks)
    ids = [f"{filename}_{i}" for i in range(len(chunks))]
    collection.upsert(ids=ids, documents=chunks, embeddings=embeddings)
    delete_extra_chunks(filename, ids)
    return len(chunks)


def init_knowledge_base():
    """服务启动时调用一次；已有知识库直接复用。"""
    if collection.count() > 0:
        print("✅ 库已存在，跳过建库")
        return

    docs_dir = BASE_DIR / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    all_chunks = []
    all_ids = []

    for filename in os.listdir(docs_dir):
        if filename.endswith(".txt"):
            doc_text = (docs_dir / filename).read_text(encoding="utf-8")
            chunks = chunk_by_sentence(doc_text, 300)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_ids.append(f"{filename}_{i}")

    if not all_chunks:
        print("ℹ️ 暂无可索引文本，请通过页面上传文档")
        return

    embeddings = embed_batch(all_chunks)
    collection.add(ids=all_ids, documents=all_chunks, embeddings=embeddings)
    print("✅ 建库完成，共", len(all_chunks), "块")
