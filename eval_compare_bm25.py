import json
from hashlib import sha256
from pathlib import Path

import chromadb
import rag_agent as rag

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT


def save_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def rerank_and_score(case, candidate_ids, text_by_id):
    documents = [text_by_id[chunk_id] for chunk_id in candidate_ids]
    best_documents, best_indices = rag.rerank(
        case["question"], documents, top_n=1
    )
    best_id = candidate_ids[best_indices[0]]
    text = best_documents[0]
    if text != text_by_id[best_id]:
        raise ValueError("重排后的文本和来源 ID 不一致")

    evidences = [case["evidence"], *case.get("alternative_evidence", [])]
    hit = best_id.rsplit("_", 1)[0] == case["source"] and any(
        evidence in text for evidence in evidences
    )
    return {"id": best_id, "text": text, "hit": hit}


def main():
    baseline_path = ROOT / "eval_result_v1_reviewed.json"
    baseline_bytes = baseline_path.read_bytes()
    baseline = json.loads(baseline_bytes.decode("utf-8-sig"))

    for name, expected_hash in baseline["files_sha256"].items():
        if sha256((ROOT / name).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"输入与审核基线不一致：{name}")

    client = chromadb.PersistentClient(path=str(ROOT / "eval_chroma_db"))
    db = client.get_collection(f"retrieval_{baseline['index_key']}")
    stored = db.get(include=["documents"])
    text_by_id = dict(zip(stored["ids"], stored["documents"]))

    for row in baseline["rows"]:
        vector_ids = row["candidate_ids"]["vector"]
        bm25_ids = row["candidate_ids"]["bm25"]
        if not vector_ids or not bm25_ids:
            raise ValueError("基线候选列表为空")
        if not set(bm25_ids).issubset(vector_ids):
            raise ValueError("BM25 候选不属于对应的向量候选")
        for chunk_id in vector_ids:
            if chunk_id not in text_by_id:
                raise ValueError(f"评测索引缺少分块：{chunk_id}")

    inputs = {
        "baseline_sha256": sha256(baseline_bytes).hexdigest(),
        "rag_agent_sha256": sha256((ROOT / "rag_agent.py").read_bytes()).hexdigest(),
        "comparison_script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "index_key": baseline["index_key"],
        "rerank_top_n": 1,
    }
    experiment_key = sha256(
        json.dumps(inputs, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    output_path = OUTPUT_DIR / f"eval_compare_bm25_{experiment_key}.json"

    if output_path.exists():
        report = json.loads(output_path.read_text(encoding="utf-8"))
    else:
        report = {"inputs": inputs, "total": baseline["total"], "rows": []}

    completed = {row["case"]["question"] for row in report["rows"]}
    print(f"已完成 {len(completed)}/{report['total']} 题")

    try:
        for saved_row in baseline["rows"]:
            case = saved_row["case"]
            if case["question"] in completed:
                continue

            # 两个方案使用同一道题和同一份已归档候选。
            with_bm25 = rerank_and_score(
                case, saved_row["candidate_ids"]["bm25"], text_by_id
            )
            without_bm25 = rerank_and_score(
                case, saved_row["candidate_ids"]["vector"], text_by_id
            )
            report["rows"].append({
                "case": case,
                "with_bm25": with_bm25,
                "without_bm25": without_bm25,
                "candidate_ids": saved_row["candidate_ids"],
            })
            save_report(output_path, report)
            print(f"完成 {len(report['rows'])}/{report['total']}：{case['question']}")
    except Exception:
        print(f"已保存 {len(report['rows'])} 题，重跑同一命令可继续。")
        raise

    if len(report["rows"]) != report["total"]:
        raise ValueError("实验题数不完整")

    report["hits"] = {
        method: sum(row[method]["hit"] for row in report["rows"])
        for method in ("with_bm25", "without_bm25")
    }
    report["gains_without_bm25"] = [
        row["case"]["question"] for row in report["rows"]
        if not row["with_bm25"]["hit"] and row["without_bm25"]["hit"]
    ]
    report["losses_without_bm25"] = [
        row["case"]["question"] for row in report["rows"]
        if row["with_bm25"]["hit"] and not row["without_bm25"]["hit"]
    ]
    save_report(output_path, report)

    for method, count in report["hits"].items():
        print(f"{method}: {count}/{report['total']} ({count / report['total']:.1%})")
    print(f"去掉 BM25 后由未命中变为命中：{len(report['gains_without_bm25'])} 题")
    print(f"去掉 BM25 后由命中变为未命中：{len(report['losses_without_bm25'])} 题")
    print(f"报告：{output_path.name}")


if __name__ == "__main__":
    main()
