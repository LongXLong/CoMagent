# -*- coding: utf-8 -*-
"""工作记忆 (WM)：运行时上下文与反馈记录。"""

import uuid
from datetime import datetime
from typing import Any


def create_wm(session_id: str | None = None) -> dict[str, Any]:
    """创建空的工作记忆。"""
    sid = session_id or f"{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
    return {
        "session_id": sid,
        "question": "",
        "student_answer": "",
        "agent_feedback": [],
        "agent_feedback_history": [],
        "improved_answer": "",
        "iteration": 0,
    }


def update_wm(
    wm: dict[str, Any],
    question: str = "",
    student_answer: str = "",
    agent_feedback: list | None = None,
    improved_answer: str = "",
    iteration: int = 0,
) -> None:
    """更新工作记忆字段；每次传入的 agent_feedback 会追加到 agent_feedback_history。"""
    if question:
        wm["question"] = question
    if student_answer:
        wm["student_answer"] = student_answer
    if agent_feedback is not None:
        wm["agent_feedback"] = agent_feedback
        wm.setdefault("agent_feedback_history", []).append(agent_feedback)
    if improved_answer:
        wm["improved_answer"] = improved_answer
    wm["iteration"] = iteration
