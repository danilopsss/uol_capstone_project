import logging
from pathlib import Path

import ollama
from ollama import Message, Options

from app.core.config.model import settings

logger = logging.getLogger("skills")


def _skills_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "skills"


def list_skills() -> list[str]:
    """List the markdown skill files for the legal_simplyfier skill."""
    skills = sorted(_skills_dir().rglob("legal_simplyfier/*.md"))
    return [s.as_posix() for s in skills]


def get_skill(path: str) -> str:
    """Read and return the content of a skill file."""
    return Path(path).read_text()


def simplify_legal_doc(document: str = "") -> str:
    """
    Call the legal_simplyfier skill to simplify the given legal document.
    The skill content is injected as system context and the document is
    passed as the user prompt.
    """
    logger.info("[TOOL CALL] Calling legal_simplyfier skill.")

    paths = list_skills()
    if not paths:
        raise RuntimeError("legal_simplyfier skill has no markdown file")
    skill_content = get_skill(paths[0])

    messages = [
        Message(role="system", content=skill_content),
        Message(role="user", content=document),
    ]
    response = ollama.chat(  # pyright: ignore[reportUnknownMemberType]
        model=settings.name,
        messages=messages,
        think="medium",
        options=Options(temperature=0.6),
    )
    return response.message.content or ""
