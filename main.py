"""Web 入口：接收请求、管理会话、调用 RAG，并返回页面或答案。"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import rag_agent as rag

BASE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    rag.init_knowledge_base()
    yield


app = FastAPI(lifespan=lifespan)
histories = {}


class AskRequest(BaseModel):
    question: str
    user_id: str = "default"


class UploadRequest(BaseModel):
    filename: str
    content: str


def get_history(user_id):
    if user_id not in histories:
        histories[user_id] = []
    return histories[user_id]


@app.post("/ask")
def ask(req: AskRequest):
    try:
        context, source = rag.hybrid_search(req.question)
        history = get_history(req.user_id)
        user_message = {"role": "user", "content": req.question}
        request_history = history + [user_message]
        answer = rag.generate(request_history, context)
        history.extend([user_message, {"role": "assistant", "content": answer}])
        return {"answer": answer, "source": source}
    except Exception as e:
        return {"error": f"出错了：{e}"}


@app.post("/upload")
def upload(req: UploadRequest):
    try:
        count = rag.add_document(req.filename, req.content)
        return {"message": f"已上传 {req.filename}，切成了 {count} 块"}
    except Exception as e:
        return {"error": f"出错了：{e}"}


@app.post("/upload_file")
async def upload_file(file: UploadFile):
    filename = file.filename
    content = await file.read()

    if filename.endswith(".pdf"):
        text = rag.parse_pdf(content)
    elif filename.endswith(".docx"):
        text = rag.parse_docx(content)
    elif filename.endswith((".txt", ".md")):
        text = content.decode("utf-8")
    else:
        return {"error": f"不支持的文件格式：{filename}"}

    count = rag.add_document(filename, text)
    return {"message": f"已上传 {filename}，切成 {count} 块"}


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "index.html")


@app.post("/ask_stream")
def ask_stream(req: AskRequest):
    try:
        context, source = rag.hybrid_search(req.question)
        history = get_history(req.user_id)
        user_message = {"role": "user", "content": req.question}
        request_history = history + [user_message]

        def generate_with_source():
            answer_parts = []

            try:
                for content in rag.generate_stream(request_history, context):
                    answer_parts.append(content)
                    yield content

                answer = "".join(answer_parts)
                if not answer.strip():
                    raise ValueError("模型没有返回有效回答")

            except Exception as e:
                print(f"流式生成失败：{type(e).__name__}：{e}")
                yield "\n\n生成失败，请稍后重试。"
                return

            history.extend([
                user_message,
                {"role": "assistant", "content": answer}
            ])
            yield f"\n\n来源：{source}"

        return StreamingResponse(generate_with_source(), media_type="text/plain")
    except Exception as e:
        return {"error": f"出错了：{e}"}
