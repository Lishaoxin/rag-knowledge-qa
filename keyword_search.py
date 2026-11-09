from rank_bm25 import BM25Okapi

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

docs = ["Python 是一种编程语言", "FastAPI 是 Web 框架", "RAG 是检索增强生成"]
print(keyword_search("RAG 是什么", docs, top_n=3))
