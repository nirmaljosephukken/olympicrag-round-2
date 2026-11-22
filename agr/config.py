from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
_DEFAULT_MODEL = {"gemini": "gemini-3.1-flash-lite", "anthropic": "claude-haiku-5-5", "openai": "gpt-5-mini"}


@dataclass(frozen=True)
class Settings:
    tg_host: str = os.getenv("TG_HOST", "").rstrip("/")
    tg_secret: str = os.getenv("TG_SECRET", "")
    tg_token: str = os.getenv("TG_TOKEN", "")
    tg_graph: str = os.getenv("TG_GRAPH", "OlympicRAG2")
    # LLM: LLM_PROVIDER=gemini|anthropic|openai, LLM_MODEL overrides the provider default
    llm_provider: str = _PROVIDER
    gemini_key: str = os.getenv("GEMINI_API_KEY", "")
    anthropic_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    openai_key: str = os.getenv("OPENAI_API_KEY", "")
    model: str = os.getenv("LLM_MODEL") or (os.getenv("GEMINI_MODEL") if _PROVIDER == "gemini" else "") \
        or _DEFAULT_MODEL.get(_PROVIDER, "")
    judge_model: str = os.getenv("JUDGE_MODEL", "") or os.getenv("LLM_MODEL") or \
        (os.getenv("GEMINI_MODEL") if _PROVIDER == "gemini" else "") or _DEFAULT_MODEL.get(_PROVIDER, "")
    rag_k: int = int(os.getenv("RAG_K", "8"))
    max_agent_steps: int = int(os.getenv("MAX_AGENT_STEPS", "8"))


SETTINGS = Settings()
