# 智能知识库问答系统

一个基于 **RAG（检索增强生成）** 的智能知识库问答系统。它能加载文档、自动切块、向量化存储，并根据文档内容回答用户问题，答案**附带引用溯源**。

> 解决的核心问题：企业内部知识散落在文档里，员工查找困难。本系统让 AI 自动检索文档并回答，且答案可溯源。

---

## ✨ 功能特性

- **文档问答**：基于向量检索 + LLM 生成，回答有依据
- **引用溯源**：每个答案标注来源文档（精确到「文档 + 块」），可查证
- **多轮对话**：支持上下文记忆，能理解「它」「这个」等指代
- **三阶段检索**：向量召回 Top 20 → BM25 筛选 Top 5 → Rerank 选取 Top 1
- **动态上传**：支持运行时上传新文档，自动切块、向量化、入库
- **评测**：内置评测集，可量化检索准确率

---

## 🛠 技术栈

| 层 | 技术 |
|---|---|
| Web 框架 | FastAPI |
| 向量数据库 | Chroma（持久化） |
| Embedding | BGE-large-zh（SiliconFlow） |
| Rerank | BGE-reranker-v2-m3（SiliconFlow） |
| 生成模型 | DeepSeek |
| 配置管理 | python-dotenv（环境变量） |

---

## 🏗 架构

```
【建库（启动时 + 上传时）】
文档 → 切块 → 向量化（embedding）→ 存 Chroma

【问答（每次请求）】
问题 → 向量化 → Chroma 召回 Top 20 → BM25 筛选 Top 5 → Rerank 选取 Top 1 → DeepSeek 生成 → 返回（带溯源）
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件，填入你的 API Key：

```
DEEPSEEK_API_KEY=sk-你的DeepSeek-key
SILICONFLOW_API_KEY=sk-你的SiliconFlow-key
```

### 3. 准备文档

在 `docs/` 文件夹放入你的 `.txt` 文档（会自动加载、切块、建库）。

### 4. 启动

```bash
uvicorn main:app --reload
```

### 5. 访问

- 前端界面：http://127.0.0.1:8000
- API 文档：http://127.0.0.1:8000/docs

---

## 📡 接口说明

| 接口 | 方法 | 说明 |
|---|---|---|
| `/` | GET | 前端页面 |
| `/ask` | POST | 问答（返回答案 + 来源） |
| `/upload` | POST | 上传文档（自动切块入库） |

### `/ask` 示例

```json
请求：{"question": "什么是 RAG？"}
响应：{"answer": "RAG 是检索增强生成...", "source": "rag.txt_0"}
```

### `/upload` 示例

```json
请求：{"filename": "新文档.txt", "content": "这是文档内容..."}
响应：{"message": "已上传 新文档.txt，切成了 3 块"}
```

---

## 📊 评测结果

评测版本为 `retrieval_v1_reviewed_2026-11-11`。使用 20 道可回答问题和仓库中的 4 份文本资料，共 1,133 个分块，每块最多 300 个字符。其中 17 道题来自 Python 教程，结果主要反映当前 Python 语料上的检索表现。

指标采用 Top-1 标注证据命中率。检索片段的来源文件与标注一致，并包含任意一条经人工审核的有效证据，即计为命中。等价证据记录在题库的 `alternative_evidence` 字段中。

| 检索方案 | 命中数 | 命中率 |
|---|---:|---:|
| 向量召回的第一名 | 14/20 | 70% |
| 向量召回 + BM25 的第一名 | 13/20 | 65% |
| 向量召回 + BM25 + Rerank 的第一名 | 16/20 | 80% |

向量 Top 20 候选包含有效证据的题目为 20/20，BM25 Top 5 为 18/20。

本表来自对同一批已保存检索输出的离线重新评分，已经人工复核等价证据。归档报告为 `eval_result_v1_reviewed.json`，原始输出保存在 `eval_result_v1_baseline_20q.json`。标注修订前后的分数变化来自评分规则修订。

评测使用 `BAAI/bge-large-zh-v1.5` 生成向量，使用 `BAAI/bge-reranker-v2-m3` 重排。文档索引版本为 `6f00bd42abc82068`。题库、语料文件和代码的 SHA-256 已记录在归档报告中。

检查标注和执行新的在线评测：

```powershell
python -X utf8 eval_rag.py --check-only
python -X utf8 eval_rag.py
```

首次运行构建独立的 `eval_chroma_db/`，之后复用已保存的文档向量。在线重跑仍会调用问题向量化和 Rerank 接口，服务输出可能出现变化，新运行的报告保存为 `eval_result_v1.json`。

离线核对归档报告的命中数：

```powershell
@'
import json
from pathlib import Path

report = json.loads(Path("eval_result_v1_reviewed.json").read_text(encoding="utf-8"))
for stage in ("vector", "bm25", "rerank"):
    hits = 0
    for row in report["rows"]:
        case = row["case"]
        result = row["stages"][stage]
        evidences = [case["evidence"], *case.get("alternative_evidence", [])]
        source = result["id"].rsplit("_", 1)[0]
        hits += int(source == case["source"] and any(e in result["text"] for e in evidences))
    print(f"{stage}: {hits}/{report['total']} ({hits / report['total']:.1%})")
'@ | python -X utf8 -
```

### BM25 筛选对比实验

使用同一份已归档向量候选、同一道问题、相同的 Rerank 模型及冻结后的证据标注，比较两种流程。

| 方案 | 每题送入 Rerank 的片段数 | 标注证据命中数 | 命中率 |
|---|---:|---:|---:|
| 向量 → BM25 → Rerank | 5 | 16/20 | 80% |
| 向量 → Rerank | 20 | 16/20 | 80% |

移除 BM25 后，Python 用途题命中了指定证据。装饰器题选中了另一段描述 decorator 与前后日志的练习内容，由于冻结标注未覆盖该段，在本指标下计为未命中。其余 18 题的命中判定一致。

本轮记录表明，BM25 将重排候选数从 20 个缩减到 5 个，两组的标注证据命中数持平。当前项目继续沿用三阶段流程。

运行 `python -X utf8 eval_compare_bm25.py` 可执行此对比。实验逐题输出保存在 [对比报告](eval_compare_bm25_f3dbeb697a84.json)，包含两组输出、来源 ID 及改善和退步的题目。

---

## 📁 项目结构

```text
├── main.py                         # FastAPI 入口、接口和会话管理
├── rag_agent.py                    # 文档处理、检索、重排和答案生成
├── eval_rag.py                     # 检索评测
├── eval_answer.py                  # 答案评测
├── eval_cases_v1.json              # 已审核的检索题库
├── eval_result_v1_reviewed.json     # 本轮评测归档
├── eval_result_v1_baseline_20q.json # 原始检索输出
├── index.html                     # 前端页面
├── docs/                          # 知识库文档
├── chroma_db/                     # 应用向量数据库
├── eval_chroma_db/                # 独立评测索引
└── .env                           # API Key
```

---

## 🔮 后续计划

- [x] 流式输出（答案逐字返回）
- [x] 混合检索（关键词 + 向量）
- [ ] 语义分块（替代固定长度切块）
- [x] 答案评测（LLM-as-judge）
- [ ] Docker 部署
- [ ] 将本项目定制为自己的专属Agent项目

