# -*- coding: utf-8 -*-
"""
系统配置文件：API Key、模型参数、路径等。
请将 OPENAI_API_KEY 设置为你的密钥，或通过环境变量 OPENAI_API_KEY 传入。
"""

import json
import os
from pathlib import Path


def _load_conf(path: Path) -> dict:
	if not path.is_file():
		return {}
	try:
		return json.loads(path.read_text(encoding="utf-8"))
	except json.JSONDecodeError:
		return {}

CONF_PATH = Path(__file__).resolve().parent / "cfg" / "conf.json"
_CONF = _load_conf(CONF_PATH)


def _conf_get(key: str, default: str) -> str:
	return _CONF.get(key, default)

# ---------- OpenAI ----------
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", _conf_get("OPENAI_API_KEY", ""))
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4.1") 
OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0"))
OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_API_EMBEDDING_KEY:str=os.getenv("OPENAI_API_EMBEDDING_KEY", _conf_get("OPENAI_API_EMBEDDING_KEY", OPENAI_API_KEY))
OPENAI_BASE_EMBEDDING_URL:str=os.getenv("OPENAI_BASE_EMBEDDING_URL", "https://api.openai.com/v1/")

# ---------- Judge Eval 配置（用于 judge_eval.py 一致性评测） ----------
JUDGE_API_KEY: str = os.getenv("JUDGE_API_KEY", _conf_get("JUDGE_API_KEY", OPENAI_API_KEY))
JUDGE_BASE_URL: str = os.getenv("JUDGE_BASE_URL", _conf_get("JUDGE_BASE_URL", OPENAI_BASE_URL))
JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", _conf_get("JUDGE_MODEL", "gpt-5-chat-latest"))

# ---------- 项目路径 ----------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "log"
LTM_PATH = DATA_DIR / "ltm.json"
LTM_VECTOR_META_PATH = DATA_DIR / "ltm_store.parquet"
LTM_VECTOR_EMBEDDINGS_PATH = DATA_DIR / "ltm_store_embeddings.npy"
LTM_EMBEDDINGS_PATH = DATA_DIR / "ltm_embeddings.json"
MK_MEMORY_PATH = DATA_DIR / "mk_memory.json"

# ---------- MK 默认问题类型（当无法推断类型时使用） ----------
DEFAULT_QUESTION_TYPE: str = _conf_get("DEFAULT_QUESTION_TYPE", "general")
