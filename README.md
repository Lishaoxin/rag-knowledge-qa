# 智能知识库问答系统

一个基于 **RAG（检索增强生成）** 的智能知识库问答系统。它能加载文档、自动切块、向量化存储，并根据文档内容回答用户问题，答案**附带引用溯源**。

> 解决的核心问题：企业内部知识散落在文档里，员工查找困难。本系统让 AI 自动检索文档并回答，且答案可溯源。

---

## ✨ 功能特性

- **文档问答**：基于向量检索 + LLM 生成，回答有依据
- **引用溯源**：每个答案标注来源文档（精确到「文档 + 块」），可查证
- **多轮对话**：支持上下文记忆，能理解「它」「这个」等指代
- **两段式检索**：向量粗选（Chroma）+ Rerank 精排，检索更精准
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
问题 → 向量化 → Chroma 粗选 top-10 → Rerank 精排 top-1 → DeepSeek 生成 → 返回（带溯源）
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

- 评测集：20 个问题（覆盖 4 篇文档）
- 检索命中率：**100%（20/20）**
- 注：样本较小（20 题），未来会扩充评测集并引入答案准确率评测（LLM-as-judge）

---

## 📁 项目结构

```
├── rag_agent.py      # 主程序（FastAPI + RAG）
├── index.html        # 前端页面
├── main.py           # FastAPI 入口、接口和会话管理
├── rag_agent.py      # 文档处理、检索、重排和答案生成
├── docs/             # 知识库文档（.txt）
├── chroma_db/        # 向量数据库（持久化）
└── .env              # API Key（不上传 GitHub）
```

---

## 🔮 后续计划

- [x] 流式输出（答案逐字返回）
- [x] 混合检索（关键词 + 向量）
- [x] 语义分块（替代固定长度切块）
- [x] 答案评测（LLM-as-judge）
- [ ] Docker 部署
- [ ] 将本项目定制为自己的专属Agent项目

