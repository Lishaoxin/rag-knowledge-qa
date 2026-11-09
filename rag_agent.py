import time
from rank_bm25 import BM25Okapi
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

def hybrid_search(query, top_n=10):
    # ① 向量粗选 top-10
    q_vec = embed(query)
    result = collection.query(query_embeddings=[q_vec], n_results=top_n)
    candidates = result["documents"][0]      # 10 个候选文档
    candidate_ids = result["ids"][0]          # 10 个候选的 id

    # ② 关键词精排：对这 10 个候选用 BM25 打分
    kw_ranked = keyword_search(query, candidates, top_n=top_n)

    # ③ 取关键词得分最高的那个
    best_idx = kw_ranked[0][0]                # 第一个结果的「文档序号」
    return candidates[best_idx], candidate_ids[best_idx]

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
histories = {}

def chunk_text(text, chunk_size=300):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        if chunk.strip():        # ← 关键：只保留「非空白」的块
            chunks.append(chunk)
    return chunks

def chunk_by_sentence(text, max_chunk=300):
    # ① 把各种句子结尾符号，统一换成句号，再按句号切
    text = text.replace("！", "。").replace("？", "。").replace("\n", "。")
    sentences = text.split("。")
    sentences = [s.strip() for s in sentences if s.strip()]   # 去掉空白

    # ② 合并短句，直到接近 max_chunk
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) <= max_chunk:   # 还能装下，就合并
            current += s + "。"
        else:                                    # 装不下了，当前块收工
            if current:
                chunks.append(current)
            current = s + "。"                   # 开新块
    if current:                                  # 最后一块
        chunks.append(current)

    return chunks


# ===== 替换：读文档 + 切块 =====
docs_dir = "./docs"
all_chunks = []   # 所有块
all_ids = []      # 所有块的 id

for filename in os.listdir(docs_dir):          # 列出文件夹里所有文件名
    if filename.endswith(".txt"):              # 只处理 .txt 文件
        with open(os.path.join(docs_dir, filename), "r", encoding="utf-8") as f:
            doc_text = f.read()
        chunks = chunk_by_sentence(doc_text, 300)          # 切成块
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
    user_id: str = "default"

def get_history(user_id):
    if user_id not in histories:      # 这个用户还没有历史
        histories[user_id] = []       # 给他建一个空的
    return histories[user_id]         # 返回他的历史

@app.post("/ask")
def ask(req: AskRequest):
    try:                                      # 尝试执行下面这些
        context, source = hybrid_search(req.question)
        history = get_history(req.user_id)
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
        chunks = chunk_by_sentence(req.content, chunk_size=300)

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
        context, source = hybrid_search(req.question)
        history = get_history(req.user_id)    # ← 先取出来
        history.append({"role": "user", "content": req.question})

        def generate_with_source():
            yield from generate_stream(history, context)   # ← 用这个 history
            yield f"\n\n来源：{source}"

        return StreamingResponse(generate_with_source(), media_type="text/plain")
    except Exception as e:
        return {"error": f"出错了：{e}"}
