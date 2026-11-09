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
doc, doc_id = hybrid_search("什么是 RAG？")
print(doc_id, "→", doc[:30])
