import time
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
import os
import json
import requests
import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY")

def embed(text):   # 单个文本的向量化（/ask 查询时用）
    resp = requests.post(
        "https://api.siliconflow.cn/v1/embeddings",
        headers={"Authorization": f"Bearer {SILICONFLOW_API_KEY}"},
        json={"model": "BAAI/bge-large-zh-v1.5", "input": text}
    )
    return resp.json()["data"][0]["embedding"]

# ===== 【启动时做一次】建库 =====
def embed_batch(texts, batch_size=20):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        batch_ok = False
        for attempt in range(3):          # 最多重试 3 次
            try:
                resp = requests.post(
                    "https://api.siliconflow.cn/v1/embeddings",
                    headers={"Authorization": f"Bearer {SILICONFLOW_API_KEY}"},
                    json={"model": "BAAI/bge-large-zh-v1.5", "input": batch}
                )
                data = resp.json()
                if "data" in data:
                    for item in data["data"]:
                        all_embeddings.append(item["embedding"])
                    batch_ok = True
                    break                # 成功，跳出重试
            except Exception as e:
                print(f"  批次 {i//batch_size} 失败（第{attempt+1}次）：{e}")
            time.sleep(3)                 # 失败后等 3 秒再试

        if not batch_ok:
            raise Exception(f"批次 {i//batch_size} 连续失败 3 次")

        time.sleep(1)                     # 每批之间停 1 秒（原来是 0.5）
        print(f"已处理 {min(i+batch_size, len(texts))}/{len(texts)} 块")

    return all_embeddings

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("knowledge_base")

# docs_dir = "./docs"
# documents = []
# filenames = []
history = []

def chunk_text(text, chunk_size=300):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        if chunk.strip():        # ← 关键：只保留「非空白」的块
            chunks.append(chunk)
    return chunks


# ===== 替换：读文档 + 切块 =====
docs_dir = "./docs"
all_chunks = []   # 所有块
all_ids = []      # 所有块的 id

for filename in os.listdir(docs_dir):          # 列出文件夹里所有文件名
    if filename.endswith(".txt"):              # 只处理 .txt 文件
        with open(os.path.join(docs_dir, filename), "r", encoding="utf-8") as f:
            doc_text = f.read()
        chunks = chunk_text(doc_text, 300)          # 切成块
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{filename}_{i}")        # id = 文件名_块号

# 如果集合是空的，才建库（第一次启动时）
if collection.count() == 0:
    embeddings = embed_batch(all_chunks)    # ← 一次批量搞定
    collection.add(ids=all_ids, documents=all_chunks, embeddings=embeddings)
    print("✅ 建库完成，共", len(all_chunks), "块")

else:
    print("✅ 库已存在，跳过建库")

def generate(history, context):
    resp = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
        json={
            "model": "deepseek-flash",
            "messages": [
                {"role": "system", "content": f"你是一个助手，根据资料回答。\n\n资料：{context}"}
            ] + history
        }
    )
    return resp.json()["choices"][0]["message"]["content"]


# ===== 【每次请求】查询 + 生成 =====
def generate_stream(history, context):
    resp = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
        json={
            "model": "deepseek-flash",
            "messages": [
                {"role": "system", "content": f"你是一个助手，根据资料回答。\n\n资料：{context}"}
            ] + history,
            "stream": True
        },
        stream=True
    )

    for line in resp.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                chunk = json.loads(data_str)
                content = chunk["choices"][0]["delta"].get("content")
                if content:
                    yield content     # ← 关键：不是 return，是 yield

def rerank(query, documents, top_n=3):
    resp = requests.post(
        "https://api.siliconflow.cn/v1/rerank",
        headers={"Authorization": f"Bearer {SILICONFLOW_API_KEY}"},
        json={
            "model": "BAAI/bge-reranker-v2-m3",   # Rerank 模型（和 embedding 不是同一个！）
            "query": query,                        # 用户的问题
            "documents": documents,                # 候选文档列表
            "top_n": top_n                         # 返回前几名
        }
    )
    data = resp.json()
    results = data["results"]                     # 已按相关性排序
    indices = [r["index"] for r in results]        # 重排后的 index（在原列表里的位置）
    docs = [documents[i] for i in indices]          # 对应的文档
    return docs, indices                            # 返回两样


class AskRequest(BaseModel):     # ← 请求体：用户的问题
    question: str

@app.post("/ask")
def ask(req: AskRequest):
    try:                                      # 尝试执行下面这些
        q_vec = embed(req.question)
        result = collection.query(query_embeddings=[q_vec], n_results=10)
        candidates = result["documents"][0]
        candidate_ids = result["ids"][0]
        reranked_docs, reranked_indices = rerank(req.question, candidates, top_n=1)
        context = reranked_docs[0]
        source = candidate_ids[reranked_indices[0]]
        history.append({"role": "user", "content": req.question})
        answer = generate(history, context)
        history.append({"role": "assistant", "content": answer})
        return {"answer": answer, "source": source}             # 成功 → 正常返回
    except Exception as e:                    # 任何一步出错
        return {"error": f"出错了：{e}"}      # 失败 → 返回友好提示，不崩溃

class UploadRequest(BaseModel):     # 上传请求体：文件名 + 内容
    filename: str
    content: str

@app.post("/upload")
def upload(req: UploadRequest):
    try:
        # ① 切块
        chunks = chunk_text(req.content, chunk_size=300)

        # ② 每块向量化
        embeddings = embed_batch(chunks)

        # ③ 生成 id（文件名_块号）
        ids = [f"{req.filename}_{i}" for i in range(len(chunks))]

        # ④ 存进 Chroma
        collection.add(ids=ids, documents=chunks, embeddings=embeddings)

        return {"message": f"已上传 {req.filename}，切成了 {len(chunks)} 块"}
    except Exception as e:
        return {"error": f"出错了：{e}"}

@app.get("/")
def home():
    return FileResponse("index.html")

@app.post("/ask_stream")
def ask_stream(req: AskRequest):
    try:
        # ① 检索（和 /ask 一样）
        q_vec = embed(req.question)
        result = collection.query(query_embeddings=[q_vec], n_results=10)
        candidates = result["documents"][0]
        reranked_docs, reranked_indices = rerank(req.question, candidates, top_n=1)
        context = reranked_docs[0]

        history.append({"role": "user", "content": req.question})

        # ② 流式生成
        return StreamingResponse(generate_stream(history, context), media_type="text/plain")
    except Exception as e:
        return {"error": f"出错了：{e}"}