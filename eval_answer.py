import os
import requests
import chromadb
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY")

# ===== ① embed（单个文本向量化）=====
def embed(text):
    resp = requests.post(
        "https://api.siliconflow.cn/v1/embeddings",
        headers={"Authorization": f"Bearer {SILICONFLOW_API_KEY}"},
        json={
            "model": "BAAI/bge-large-zh-v1.5",
            "input": text
        },
        timeout=(10, 120)
    )
    resp.raise_for_status()

    return resp.json()["data"][0]["embedding"]

# ===== ② Chroma =====
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("knowledge_base")

# ===== ③ 关键词检索（BM25）=====
def keyword_search(query, documents, top_n=10):
    tokenized_docs = [list(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(list(query))
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]

# ===== ④ 混合检索（向量粗选 + 关键词精排）=====
def hybrid_search(query, top_n=20, mid_n=5):
    # ① 向量粗选 top-20
    q_vec = embed(query)
    result = collection.query(query_embeddings=[q_vec], n_results=top_n)
    candidates = result["documents"][0]
    candidate_ids = result["ids"][0]

    # ② BM25 关键词粗排，选出 top-5
    kw_ranked = keyword_search(query, candidates, top_n=mid_n)
    mid_docs = [candidates[idx] for idx, _ in kw_ranked]
    mid_ids = [candidate_ids[idx] for idx, _ in kw_ranked]

    # ③ Rerank 精排，选出 top-1
    reranked_docs, reranked_indices = rerank(query, mid_docs, top_n=1)
    best_doc = reranked_docs[0]
    best_id = mid_ids[reranked_indices[0]]

    return best_doc, best_id


# ===== ⑤ 生成（DeepSeek）=====
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

# ===== ⑥ LLM 判卷（判断答案对不对）=====
def judge_answer(question, answer, expected):
    prompt = f"""
问题：{question}
标准答案：{expected}
模型回答：{answer}

请判断「模型回答」是否包含了标准答案的核心意思（不必逐字一致，角度可以不同，只要意思对就算对）。只回答「对」或「错」。
"""

    resp = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
        json={
            "model": "deepseek-flash",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        },
        timeout=(10, 120)
    )
    resp.raise_for_status()

    return resp.json()["choices"][0]["message"]["content"]

def rerank(query, documents, top_n=1):
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


# ===== ⑦ 评测集（问题 + 标准答案）=====
test_set = [
    {"question": "什么是函数？", "expected": "函数是封装一段代码、可以重复调用的代码块"},
    {"question": "什么是类？", "expected": "类是创建对象的模板或蓝图"},
    {"question": "list 和 tuple 有什么区别？", "expected": "list 可变，tuple 不可变"},
    {"question": "什么是装饰器？", "expected": "装饰器是用来增强函数功能的函数"},
    {"question": "self 是什么？", "expected": "self 是实例本身，代表当前对象"},
    {"question": "什么是模块？", "expected": "模块是包含 Python 代码的 .py 文件"},
    {"question": "什么是异常处理？", "expected": "用 try except 捕获并处理错误"},
    {"question": "什么是生成器？", "expected": "生成器是边生成边返回值的迭代器"},
]

# ===== ⑧ 跑评测，算答案准确率 =====
correct = 0
for item in test_set:
    try:
        # ① 检索 + 生成（不变）
        context, source = hybrid_search(item["question"])
        answer = generate([{"role": "user", "content": item["question"]}], context)

        # ② 多数投票：判 3 次
        yes_count = 0
        for _ in range(3):
            result = judge_answer(item["question"], answer, item["expected"]).strip()

            if result not in ("对", "错"):
                raise ValueError(f"判卷格式错误：{result}")

            if result == "对":
                yes_count += 1

        # ③ 2 次以上判对，就算对
        if yes_count >= 2:
            correct += 1
            print(f"✅ {item['question']}（{yes_count}/3 票）")
        else:
            print(f"❌ {item['question']}（{yes_count}/3 票）")

    except Exception as e:
        print(f"⚠️ {item['question']} 出错：{e}")

print(f"\n📊 答案准确率：{correct}/{len(test_set)} = {correct/len(test_set)*100:.1f}%")

