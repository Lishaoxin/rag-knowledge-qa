# 检索评测

## 固定输入与指标

检索评测版本为 `retrieval_v1_reviewed_2026-11-11`。4 份 [语料](../docs/)切为 1,133 段，每段最多 300 字符。[评测题库](cases/eval_cases_v1.json)共有 20 道题，其中 17 道来自 Python 教程。

Top-1 命中要求第一名片段来自标注文件，并包含至少一条人工审核的证据。等价证据保存在题库的 `alternative_evidence` 字段中。复核是在同一批已保存的检索输出上重新评分。向量 Top 20 中含有有效证据的题目为 20/20，BM25 Top 5 为 18/20，最终重排第一名为 16/20。该指标衡量检索证据命中。

评测使用 `BAAI/bge-large-zh-v1.5` 生成向量，`BAAI/bge-reranker-v2-m3` 重排。归档索引版本为 `6f00bd42abc82068`。报告记录了题库和语料指纹。报告内的历史路径 `eval_cases_v1.json` 对应现在 `cases/` 下的同一份题库。

- [人工复核报告](reports/eval_result_v1_reviewed.json)
- [原始检索报告](reports/eval_result_v1_baseline_20q.json)
- [BM25 对照报告](reports/eval_compare_bm25_f3dbeb697a84.json)

## BM25 对照

两组共享同一批向量候选、相同的 Rerank 模型和证据标注。

| 检索方案 | 每题重排候选数 | Top-1 命中 |
| --- | --- | --- |
| 向量 → BM25 → Rerank | 5 | 16/20 |
| 向量 → Rerank | 20 | 16/20 |

去掉 BM25 后，Python 用途题由未命中变为命中，装饰器题选中了一段标注未覆盖的练习内容，按当前口径计为未命中。另 18 题的命中判定一致。项目保留三阶段检索。本次比较记录的是候选数与命中数。

## 复现

安装项目依赖后，在仓库根目录运行证据检查。该命令校验每条标注证据都能在对应语料的分块中找到，无需 API Key。

```powershell
python -X utf8 -m evaluation.eval_rag --check-only
```

配置模型服务 Key 后，可运行在线检索并执行 BM25 对照。

```powershell
python -X utf8 -m evaluation.eval_rag
python -X utf8 -m evaluation.eval_compare_bm25
```

第一条命令为原始语料构建或复用独立的 `eval_chroma_db/` 索引。第二条命令从该索引按归档候选 ID 取文本，分别调用 Rerank。新报告写到 `evaluation/runs/`，在线结果可能随模型服务输出变化。

可以用下面的命令离线重算归档报告的 Top-1 命中数。

```powershell
@'
import json
from pathlib import Path

path = Path("evaluation/reports/eval_result_v1_reviewed.json")
report = json.loads(path.read_text(encoding="utf-8"))
for stage in ("vector", "bm25", "rerank"):
    hits = 0
    for row in report["rows"]:
        case = row["case"]
        result = row["stages"][stage]
        evidence = [case["evidence"], *case.get("alternative_evidence", [])]
        source = result["id"].rsplit("_", 1)[0]
        hits += int(source == case["source"] and any(e in result["text"] for e in evidence))
    print(f"{stage}: {hits}/{report['total']} ({hits / report['total']:.1%})")
'@ | python -X utf8 -
```
