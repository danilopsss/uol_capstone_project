import asyncio
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Capstone App")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

uploaded_files: list[str] = []


class ChatRequest(BaseModel):
    message: str


MOCK_RESPONSE = (
    "This is a mocked streaming response. "
    "In the future, this will be powered by an LLM "
    "that can analyze your uploaded PDF documents. "
    "For now, it simply echoes back a simulated reply "
    "to demonstrate the streaming chat functionality."
)


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    dest = UPLOAD_DIR / file.filename
    content = await file.read()
    dest.write_bytes(content)
    uploaded_files.append(file.filename)
    return {
        "filename": file.filename,
        "size": len(content),
        "uploaded_files": uploaded_files,
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    async def stream():
        words = MOCK_RESPONSE.split(" ")
        for word in words:
            yield f"data: {word} \n\n"
            await asyncio.sleep(0.06)
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
