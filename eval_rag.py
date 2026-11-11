# 导入、embed、client、collection（复制你会的那些）
import time
import os
import requests
import chromadb
from dotenv import load_dotenv

load_dotenv()
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY")

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

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("knowledge_base")

test_set = [
    # ===== rag.txt（RAG 相关）=====
    {"question": "什么是 RAG？", "expected": "rag.txt"},
    {"question": "RAG 的全称是什么？", "expected": "rag.txt"},
    {"question": "检索增强生成是什么？", "expected": "rag.txt"},
    {"question": "RAG 能解决什么问题？", "expected": "rag.txt"},

    # ===== python.txt（Python 相关）=====
    {"question": "Python 是什么？", "expected": "python.txt"},
    {"question": "Python 适合做什么？", "expected": "python.txt"},
    {"question": "Python 是一门什么样的语言？", "expected": "python.txt"},

    # ===== fastapi.txt（FastAPI 相关）=====
    {"question": "FastAPI 是什么？", "expected": "fastapi.txt"},
    {"question": "FastAPI 用来做什么？", "expected": "fastapi.txt"},
    {"question": "FastAPI 是哪个语言写的框架？", "expected": "fastapi.txt"},

    # ===== 人工智能科普.txt（AI 相关）=====
    {"question": "人工智能是什么？", "expected": "人工智能科普.txt"},
    {"question": "人工智能的起源是什么？", "expected": "人工智能科普.txt"},
    {"question": "什么是图灵测试？", "expected": "人工智能科普.txt"},
    {"question": "机器学习分为哪几类？", "expected": "人工智能科普.txt"},
    {"question": "什么是监督学习？", "expected": "人工智能科普.txt"},
    {"question": "什么是强化学习？", "expected": "人工智能科普.txt"},
    {"question": "深度学习是什么？", "expected": "人工智能科普.txt"},
    {"question": "Transformer 架构擅长什么？", "expected": "人工智能科普.txt"},
    {"question": "大语言模型基于什么架构？", "expected": "人工智能科普.txt"},
    {"question": "人工智能带来哪些挑战？", "expected": "人工智能科普.txt"},
]

correct = 0
for item in test_set:
    # ① 检索
    q_vec = embed(item["question"])
    result = collection.query(query_embeddings=[q_vec], n_results=1)
    source = result["ids"][0][0]          # 检索到的来源，如 "rag.txt_0"
    time.sleep(0.5)
    # ② 提取文档名（去掉 "_块号"）
    doc_name = source.rsplit("_", 1)[0]
    # ③ 判断是否命中
    if doc_name == item["expected"]:
        correct += 1
        print(f"✅ {item['question']} → {source}")
    else:
        print(f"❌ {item['question']} → {source}（期望 {item['expected']}）")

# ④ 算准确率
accuracy = correct / len(test_set)
print(f"\n检索准确率：{accuracy * 100:.1f}%（{correct}/{len(test_set)}）")
