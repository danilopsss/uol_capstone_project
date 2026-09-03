import asyncio
from collections.abc import AsyncIterator, Callable
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.core.config.model import settings
from app.core.llm.agent import OllamaAgent
from app.core.llm.tools import TOOLS
from app.core.llm.tools.skills import simplify_legal_doc

app = FastAPI(title="Capstone App")

TOOL_FUNCTIONS: dict[str, Callable[..., object]] = {
    "simplify_legal_doc": simplify_legal_doc
}

agent = OllamaAgent(
    model=settings.name,
    tools=TOOLS,
    tool_functions=TOOL_FUNCTIONS,
)


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    async def stream() -> AsyncIterator[str]:
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def produce() -> None:
            try:
                for token in agent.run_stream(req.message):
                    queue.put_nowait(token)
            except Exception as exc:
                queue.put_nowait(f"\n[Error] {exc}")
            finally:
                queue.put_nowait(None)

        loop = asyncio.get_event_loop()
        task = loop.run_in_executor(None, produce)
        while True:
            token = await queue.get()
            if token is None:
                break
            yield f"data: {token} \n\n"
        await task
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
