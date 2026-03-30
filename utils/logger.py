# -*- coding: utf-8 -*-
"""
集中配置日志：输出到 log 目录下的文件，并可选输出到控制台。
"""

import logging
import sys
from pathlib import Path

from config import LOG_DIR

# 日志格式：时间 | 级别 | 模块 | 消息
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_initialized = False


def _ensure_log_dir() -> Path:
    """确保 log 目录存在。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR


def setup_logging(
    *,
    level: int = logging.INFO,
    log_file: str | Path | None = "app.log",
    console: bool = True,
) -> None:
    """
    配置根日志：写入 log 目录下的文件，并可选输出到控制台。
    重复调用不会重复添加 handler。
    """
    global _initialized
    if _initialized:
        return
    _ensure_log_dir()
    root = logging.getLogger()
    root.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    if log_file:
        path = LOG_DIR / log_file if isinstance(log_file, str) else Path(log_file)
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        root.addHandler(ch)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """
    获取模块用 logger。若尚未 setup，会先执行一次 setup_logging()。
    """
    if not _initialized:
        setup_logging()
    return logging.getLogger(name)


def truncate_for_log(text: str | None, max_len: int = 250, suffix: str = "...") -> str:
    """
    截断长文本用于日志输出，便于观察中间结果又不刷屏。
    """
    if text is None:
        return ""
    s = str(text).strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - len(suffix)] + suffix
