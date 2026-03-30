# -*- coding: utf-8 -*-
"""FastAPI 后端：提供 /api/suggest 与 /api/run，并托管前端静态页。"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from openai import APIError, AuthenticationError
from pydantic import BaseModel, Field

from main import get_suggest, run_system
from utils.llm import llm_runtime_context
from utils.logger import get_logger, setup_logging

# 启动时初始化日志（创建 log 目录并配置输出）
setup_logging()

app = FastAPI(title="多 Agent 反思系统 API", version="1.0")
logger = get_logger(__name__)

# 前端静态目录
STATIC_DIR = Path(__file__).resolve().parent / "static"


class SuggestRequest(BaseModel):
    question: str
    api_key: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    model: str | None = None


class RunRequest(BaseModel):
    question: str
    selected_agents: list[str]
    update_ltm: bool = True
    update_mk: bool = True
    api_key: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    model: str | None = None
    # 与 Get Suggest 一致时传入，Run Reflection 将复用该初答而非重新生成
    initial_answer: str | None = None
    question_type: str | None = None
    candidate_question_types: list[str] | None = None


# Agent 显示名（供前端展示）
AGENT_LABELS = {
    "logic_checker": "逻辑检查",
    "clarity_editor": "清晰度编辑",
    "completeness_checker": "完整性检查",
    "evidence_checker": "证据与引用检查",
    "brevity_advisor": "简洁性建议",
    "consistency_checker": "一致性检查",
    "relevancy_checker": "相关性检查",
    "harmlessness_checker": "无害性检查",
    "compliance_checker": "合规性检查",
    "fluency_editor": "流畅自然优化",
    "text_writing_expert": "文案写作专家",
    "summarization_expert": "摘要总结专家",
    "code_development_expert": "代码开发专家",
    "knowledge_qa_expert": "知识问答专家",
    "educational_tutoring_expert": "教育辅导专家",
    "translation_localization_expert": "翻译本地化专家",
    "creative_ideation_expert": "创意构思专家",
    "data_processing_expert": "数据处理专家",
    "role_playing_expert": "角色扮演专家",
    "career_business_expert": "职业商业专家",
    "life_emotional_expert": "生活情感专家",
    "marketing_copywriting_expert": "营销文案专家",
    "logical_reasoning_expert": "逻辑推理专家",
    "math_computation_expert": "数学计算专家",
    "multimodal_expert": "多模态专家",
    "other_general_q_expert": "通用问答专家",
}

INVALID_API_KEY_USER_MESSAGE = (
    "The current API Key is invalid. Please go to the \"Model Settings\" page in the sidebar on the left to replace it with the correct key."
)


def _is_llm_invalid_api_key_error(exc: BaseException) -> bool:
    """判断是否为大模型网关返回的鉴权失败（无效令牌 / 401）。"""
    if isinstance(exc, AuthenticationError):
        return True
    if isinstance(exc, APIError) and getattr(exc, "status_code", None) == 401:
        return True
    msg = str(exc)
    if "Error code: 401" in msg or "error code: 401" in msg.lower():
        return True
    return False

@app.post("/api/suggest")
def api_suggest(req: SuggestRequest):
    """根据问题返回 MK 建议的 Agent 与初答，供前端作为系统建议勾选。"""
    if not (req.question or req.question.strip()):
        logger.warning("api_suggest 请求问题为空")
        raise HTTPException(status_code=400, detail="问题不能为空")
    try:
        logger.info("api_suggest 请求 question=%s", req.question[:80] + "..." if len(req.question) > 80 else req.question)
        with llm_runtime_context(
            api_key=req.api_key,
            temperature=req.temperature,
            model=req.model,
        ):
            out = get_suggest(req.question.strip())
        out["all_agents_with_labels"] = [
            {"id": a, "label": AGENT_LABELS.get(a, a)} for a in out["all_agents"]
        ]
        logger.info("api_suggest 成功 question_type=%s", out.get("question_type"))
        return out
    except Exception as e:
        if _is_llm_invalid_api_key_error(e):
            logger.warning("api_suggest API Key 无效: %s", e)
            raise HTTPException(status_code=401, detail=INVALID_API_KEY_USER_MESSAGE) from e
        logger.exception("api_suggest 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/run")
def api_run(req: RunRequest):
    """使用用户选择的 Agent 进行多轮问询，并按用户选择是否更新 LTM、MK。"""
    if not (req.question or req.question.strip()):
        logger.warning("api_run 请求问题为空")
        raise HTTPException(status_code=400, detail="问题不能为空")
    try:
        logger.info(
            "api_run 请求 question=%s selected_agents=%s update_ltm=%s update_mk=%s",
            req.question[:80] + "..." if len(req.question) > 80 else req.question,
            req.selected_agents,
            req.update_ltm,
            req.update_mk,
        )
        agents = req.selected_agents if req.selected_agents else None
        with llm_runtime_context(
            api_key=req.api_key,
            temperature=req.temperature,
            model=req.model,
        ):
            run_result = run_system(
                req.question.strip(),
                selected_agents=agents,
                do_update_ltm=req.update_ltm,
                do_update_mk=req.update_mk,
                suggested_initial_answer=req.initial_answer,
                suggested_question_type=req.question_type,
                suggested_candidate_question_types=req.candidate_question_types,
            )
        logger.info("api_run 成功")
        return {
            "final_answer": run_result.get("final_answer", ""),
            "expert_feedbacks": run_result.get("expert_feedbacks", []),
            "general_feedbacks": run_result.get("general_feedbacks", []),
            "success": True,
            "update_ltm": req.update_ltm,
            "update_mk": req.update_mk,
        }
    except Exception as e:
        if _is_llm_invalid_api_key_error(e):
            logger.warning("api_run API Key 无效: %s", e)
            raise HTTPException(status_code=401, detail=INVALID_API_KEY_USER_MESSAGE) from e
        logger.exception("api_run 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/")
def index():
    """返回前端页面。"""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="前端页面未找到")
    return FileResponse(index_file)

# 挂载前端静态资源目录（例如 vite build 生成的 assets 等）
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


@app.get("/health")
def health():
    return {"status": "ok"}
