import argparse
import json
from hashlib import sha256
from pathlib import Path

import chromadb
import rag_agent as rag

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    case_file = ROOT / "eval_cases_v1.json"
    cases = json.loads(case_file.read_text(encoding="utf-8-sig"))
    paths = sorted((ROOT / "docs").glob("*.txt"))
    files_sha256 = {
        path.relative_to(ROOT).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in [case_file, *paths]
    }

    chunks_by_source = {
        path.name: rag.chunk_by_sentence(path.read_text(encoding="utf-8"), max_chunk=300)
        for path in paths
    }

    if not cases:
        raise ValueError("评测题库为空")

    for case in cases:
        chunks = chunks_by_source.get(case["source"], [])
        evidences = [
            case["evidence"],
            *case.get("alternative_evidence", []),
        ]

        for evidence in evidences:
            if not evidence or not any(evidence in chunk for chunk in chunks):
                raise ValueError(f"找不到完整证据：{case['question']}")

    ids = []
    documents = []
    for source, chunks in chunks_by_source.items():
        ids.extend(f"{source}_{i}" for i in range(len(chunks)))
        documents.extend(chunks)

    print(f"证据检查通过：{len(cases)} 题")
    print(f"固定语料：{len(paths)} 个文件，{len(documents)} 个分块")
    if args.check_only:
            return

    index_config = {
        "embedding_model": "BAAI/bge-large-zh-v1.5",
        "chunks": list(zip(ids, documents)),
    }
    index_key = sha256(
        json.dumps(index_config, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]

    eval_client = chromadb.PersistentClient(path=str(ROOT / "eval_chroma_db"))
    db = eval_client.get_or_create_collection(f"retrieval_{index_key}")

    stored_ids = set(db.get(include=[])["ids"])
    pending = [
        (chunk_id, text)
        for chunk_id, text in zip(ids, documents)
        if chunk_id not in stored_ids
    ]
    print(f"评测索引 {index_key}：需新增 {len(pending)} 个分块")

    for start in range(0, len(pending), 20):
        batch = pending[start:start + 20]
        embeddings = rag.embed_batch([text for _, text in batch])
        db.upsert(
            ids=[chunk_id for chunk_id, _ in batch],
            documents=[text for _, text in batch],
            embeddings=embeddings,
        )
        print(f"评测索引进度：{db.count()}/{len(documents)}")

    if db.count() != len(documents):
        raise ValueError("评测索引分块数与当前语料不一致")

    hits = {"vector": 0, "bm25": 0, "rerank": 0}
    rows = []

    for case in cases:
        evidences = [
            case["evidence"],
            *case.get("alternative_evidence", []),
        ]

        stages, candidates_by_stage = rag.retrieve_stages(case["question"], db)
        row = {
            "case": case,
            "stages": {},
            "candidate_ids": candidates_by_stage,
        }

        for name, (text, chunk_id) in stages.items():
            source = chunk_id.rsplit("_", 1)[0]
            hit = source == case["source"] and any(
                evidence in text for evidence in evidences
            )
            hits[name] += int(hit)
            row["stages"][name] = {
                "id": chunk_id,
                "hit": hit,
                "text": text,
            }

        rows.append(row)

    report = {
        "index_key": index_key,
        "embedding_model": index_config["embedding_model"],
        "files_sha256": files_sha256,
        "total": len(cases),
        "hits": hits,
        "rows": rows,
    }
    report_file = ROOT / "eval_result_v1.json"
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for name, count in hits.items():
        print(f"{name}: {count}/{len(cases)} ({count / len(cases):.1%})")


if __name__ == "__main__":
    main()