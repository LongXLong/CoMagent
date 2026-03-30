# -*- coding: utf-8 -*-
"""固定 16 类长期记忆（LTM）与检索工具。"""

import base64
import hashlib
import json
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from config import (
    LTM_EMBEDDINGS_PATH,
    LTM_PATH,
    LTM_VECTOR_EMBEDDINGS_PATH,
    LTM_VECTOR_META_PATH,
)
from utils.llm import get_embedding, llm_call, parse_json_from_llm
from utils.logger import get_logger

logger = get_logger(__name__)
_LTM_EMBEDDINGS_CACHE_KEY: tuple[str, int, int] | None = None
_LTM_EMBEDDINGS_CACHE_VALUE: dict[str, Any] | None = None
_LTM_VECTOR_STORE_CACHE_KEY: tuple[str, int, int, str, int, int] | None = None
_LTM_VECTOR_STORE_CACHE_VALUE: dict[str, Any] | None = None

QUESTION_TYPES_16 = [
    "TEXT_WRITING",
    "SUMMARIZATION",
    "CODE_DEVELOPMENT",
    "KNOWLEDGE_QA",
    "EDUCATIONAL_TUTORING",
    "TRANSLATION_LOCALIZATION",
    "CREATIVE_IDEATION",
    "DATA_PROCESSING",
    "ROLE_PLAYING",
    "CAREER_BUSINESS",
    "LIFE_EMOTIONAL",
    "MARKETING_COPYWRITING",
    "LOGICAL_REASONING",
    "MATH_COMPUTATION",
    "MULTIMODAL",
    "OTHER_GENERAL_Q",
]

def empty_ltm() -> dict[str, Any]:
    """返回固定 16 类的空知识库结构。"""
    return {
        "version": "2.0",
        "categories": {k: [] for k in QUESTION_TYPES_16},
    }


def _normalize_ltm(data: dict[str, Any]) -> dict[str, Any]:
    out = empty_ltm()
    categories = data.get("categories", {})
    if not isinstance(categories, dict):
        return out
    for category in QUESTION_TYPES_16:
        rows = categories.get(category, [])
        if not isinstance(rows, list):
            continue
        valid_rows: list[dict[str, str]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            q = str(row.get("question", "")).strip()
            a = str(row.get("answer", "")).strip()
            if q and a:
                valid_rows.append({"question": q, "answer": a})
        out["categories"][category] = valid_rows
    return out


def load_ltm(path: Path | None = None) -> dict[str, Any]:
    """加载固定结构 LTM，不合法则回退为空库。"""
    p = path or LTM_PATH
    if not p.exists():
        return empty_ltm()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return empty_ltm()
    return _normalize_ltm(data)


def save_ltm(ltm: dict[str, Any], path: Path | None = None) -> None:
    """保存固定结构 LTM。"""
    p = path or LTM_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(_normalize_ltm(ltm), f, ensure_ascii=False, indent=2)


def empty_ltm_embeddings() -> dict[str, Any]:
    """返回空的 LTM embedding 索引结构。"""
    return {
        "version": "1.0",
        "source_ltm": str(LTM_PATH),
        "entry_count": 0,
        "entries": [],
    }


def empty_ltm_vector_store() -> dict[str, Any]:
    """返回空的运行时 LTM 向量库结构。"""
    return {
        "version": "1.0",
        "source_ltm": str(LTM_PATH),
        "entry_count": 0,
        "entries": [],
        "embeddings_matrix": np.empty((0, 0), dtype=np.float32),
    }


def _make_embeddings_cache_key(path: Path) -> tuple[str, int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    try:
        resolved = str(path.resolve())
    except OSError:
        resolved = str(path)
    return (resolved, stat.st_mtime_ns, stat.st_size)


def _make_dual_file_cache_key(
    meta_path: Path,
    embeddings_path: Path,
) -> tuple[str, int, int, str, int, int] | None:
    meta_key = _make_embeddings_cache_key(meta_path)
    embeddings_key = _make_embeddings_cache_key(embeddings_path)
    if meta_key is None or embeddings_key is None:
        return None
    return meta_key + embeddings_key


def _safe_resolve_str(path_str: str) -> str:
    try:
        return str(Path(path_str).resolve())
    except OSError:
        return str(path_str)


def _encode_store_payload(question: str, answer: str) -> str:
    payload = json.dumps(
        {
            "question": str(question or "").strip(),
            "answer": str(answer or "").strip(),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(zlib.compress(payload)).decode("ascii")


def _decode_store_payload(payload: str) -> tuple[str, str]:
    raw = str(payload or "").strip()
    if not raw:
        return "", ""
    try:
        decoded = base64.urlsafe_b64decode(raw.encode("ascii"))
        obj = json.loads(zlib.decompress(decoded).decode("utf-8"))
    except (ValueError, zlib.error, json.JSONDecodeError):
        return "", ""
    question = str(obj.get("question", "")).strip() if isinstance(obj, dict) else ""
    answer = str(obj.get("answer", "")).strip() if isinstance(obj, dict) else ""
    return question, answer


def _normalize_question_type_label(label: str) -> str:
    raw = (label or "").strip().upper()
    if not raw:
        return ""
    normalized = raw.replace("-", "_").replace(" ", "_")
    if normalized in QUESTION_TYPES_16:
        return normalized
    aliases = {
        "KNOWLEDGE_QA_EXPERT": "KNOWLEDGE_QA",
        "MATH": "MATH_COMPUTATION",
        "CODING": "CODE_DEVELOPMENT",
        "GENERAL": "OTHER_GENERAL_Q",
    }
    return aliases.get(normalized, "")


def _extract_question_type_candidates_from_obj(obj: dict[str, Any] | None) -> list[str]:
    if not isinstance(obj, dict):
        return []

    candidates: list[str] = []

    def _append_candidate(val: Any) -> None:
        if val is None:
            return
        if isinstance(val, list):
            for item in val:
                _append_candidate(item)
            return
        qt = _normalize_question_type_label(str(val))
        if qt and qt not in candidates:
            candidates.append(qt)

    for key in (
        "primary_category",
        "primary_question_type",
        "question_type",
        "type",
        "label",
        "category",
        "top_categories",
        "categories",
        "candidate_categories",
        "candidates",
    ):
        _append_candidate(obj.get(key))
    return candidates


def _extract_question_type_from_obj(obj: dict[str, Any] | None) -> str:
    candidates = _extract_question_type_candidates_from_obj(obj)
    return candidates[0] if candidates else ""


def _build_question_type_prompt(question: str, top_n: int = 2) -> str:
    category_lines = "\n".join(f"- {name}" for name in QUESTION_TYPES_16)
    return f"""
你是一个问题分类器，需要将问题严格划分到下列 16 个类别中最可能的前 {max(1, top_n)} 个。

候选类别：
{category_lines}

要求：
1. 按置信度从高到低返回最多 {max(1, top_n)} 个类别。
2. 返回 JSON，且只能返回 JSON。
3. JSON 格式为：{{"primary_category": "类别名", "categories": ["类别1", "类别2"], "reason": "一句简短原因"}}
4. `categories` 中必须包含 `primary_category`，且所有类别都必须来自候选列表。

待分类问题：
{question}
""".strip()


def _classify_question_types_via_llm(question: str, top_n: int = 2) -> tuple[list[str], str]:
    prompt = _build_question_type_prompt(question, top_n=top_n)
    system_prompt = "你是一个严格的问题分类器，只返回合法 JSON。"
    raw = llm_call(prompt, system_prompt=system_prompt)
    candidates = _extract_question_type_candidates_from_obj(parse_json_from_llm(raw))
    return candidates[: max(1, top_n)], raw


def infer_question_types_for_ltm(question: str, top_n: int = 2) -> list[str]:
    """仅从固定 16 类中判定问题类型，返回按置信度排序的候选类别。"""
    q = (question or "").strip()
    if not q:
        logger.info("[TRACE] classify source=fallback_empty question_types=%s", ["OTHER_GENERAL_Q"])
        return ["OTHER_GENERAL_Q"]

    try:
        candidates, raw = _classify_question_types_via_llm(q, top_n=top_n)
        if candidates:
            logger.info("[TRACE] classify source=llm question_types=%s", candidates)
            return candidates
        logger.warning("infer_question_type_for_ltm 分类结果解析失败，原始输出: %s", raw)
    except Exception as e:
        logger.warning("infer_question_type_for_ltm 失败，回退 OTHER_GENERAL_Q: %s", e)
    logger.info("[TRACE] classify source=fallback_invalid question_types=%s", ["OTHER_GENERAL_Q"])
    return ["OTHER_GENERAL_Q"]


def infer_question_type_for_ltm(question: str) -> str:
    """仅从固定 16 类中判定问题类型，返回最高置信类别。"""
    candidates = infer_question_types_for_ltm(question, top_n=1)
    return candidates[0] if candidates else "OTHER_GENERAL_Q"


def _normalize_text(text: str) -> str:
    return "".join((text or "").strip().lower().split())


def _make_question_hash(text: str) -> str:
    normalized = _normalize_text(text)
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _normalize_scoped_question_types(
    question_types: list[str] | None,
    question_type: str | None = None,
) -> list[str]:
    scoped_types: list[str] = []
    if question_types:
        for item in question_types:
            qt = (item or "").strip().upper()
            if qt in QUESTION_TYPES_16 and qt not in scoped_types:
                scoped_types.append(qt)
        return scoped_types
    if question_type:
        qt = question_type.strip().upper()
        if qt in QUESTION_TYPES_16:
            return [qt]
    return []


def get_all_qa_entries(ltm: dict[str, Any]) -> list[dict[str, str]]:
    """获取全库 QA 条目，附带 category。"""
    out: list[dict[str, str]] = []
    categories = (ltm or {}).get("categories", {})
    if not isinstance(categories, dict):
        return out
    for category in QUESTION_TYPES_16:
        rows = categories.get(category, [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            q = str(row.get("question", "")).strip()
            a = str(row.get("answer", "")).strip()
            if q and a:
                out.append({"category": category, "question": q, "answer": a})
    return out


def build_embedding_text(question: str, answer: str) -> str:
    """构造用于 QA 检索的 embedding 文本。"""
    return f"Question: {str(question or '').strip()}\nAnswer: {str(answer or '').strip()}"


def _normalize_ltm_embeddings(data: dict[str, Any]) -> dict[str, Any]:
    out = empty_ltm_embeddings()
    if not isinstance(data, dict):
        return out

    out["version"] = str(data.get("version") or "1.0")
    out["source_ltm"] = str(data.get("source_ltm") or str(LTM_PATH))
    raw_entries = data.get("entries", [])
    if not isinstance(raw_entries, list):
        return out

    valid_entries: list[dict[str, Any]] = []
    for row in raw_entries:
        if not isinstance(row, dict):
            continue
        category = str(row.get("category", "")).strip().upper()
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        embedding = row.get("embedding")
        if category not in QUESTION_TYPES_16 or not question or not answer:
            continue
        if not isinstance(embedding, list) or not embedding:
            continue
        try:
            embedding_values = [float(x) for x in embedding]
        except (TypeError, ValueError):
            continue
        valid_entries.append(
            {
                "category": category,
                "question": question,
                "answer": answer,
                "embedding": embedding_values,
            }
        )

    out["entries"] = valid_entries
    out["entry_count"] = len(valid_entries)
    return out


def load_ltm_embeddings(path: Path | None = None) -> dict[str, Any]:
    """加载预计算的 LTM embedding 索引。"""
    global _LTM_EMBEDDINGS_CACHE_KEY, _LTM_EMBEDDINGS_CACHE_VALUE
    p = path or LTM_EMBEDDINGS_PATH
    if not p.exists():
        logger.warning("LTM embedding 文件不存在: %s", p)
        return empty_ltm_embeddings()
    cache_key = _make_embeddings_cache_key(p)
    if (
        cache_key is not None
        and _LTM_EMBEDDINGS_CACHE_KEY == cache_key
        and _LTM_EMBEDDINGS_CACHE_VALUE is not None
    ):
        logger.info(
            "LTM embedding 索引命中内存缓存 path=%s entry_count=%s",
            p,
            _LTM_EMBEDDINGS_CACHE_VALUE.get("entry_count", 0),
        )
        return _LTM_EMBEDDINGS_CACHE_VALUE
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    normalized = _normalize_ltm_embeddings(data)
    source_ltm = str(normalized.get("source_ltm") or "").strip()
    if source_ltm:
        resolved_source = _safe_resolve_str(source_ltm)
        resolved_current = _safe_resolve_str(str(LTM_PATH))
        if resolved_source != resolved_current:
            logger.warning(
                "LTM embedding 索引来源与当前 LTM 路径不一致 source_ltm=%s current_ltm=%s",
                source_ltm,
                LTM_PATH,
            )
    logger.info(
        "LTM embedding 索引加载完成 path=%s entry_count=%s source_ltm=%s",
        p,
        normalized.get("entry_count", 0),
        source_ltm or "N/A",
    )
    if cache_key is not None:
        _LTM_EMBEDDINGS_CACHE_KEY = cache_key
        _LTM_EMBEDDINGS_CACHE_VALUE = normalized
    return normalized


def save_ltm_embeddings(index_data: dict[str, Any], path: Path | None = None) -> None:
    """保存预计算的 LTM embedding 索引。"""
    p = path or LTM_EMBEDDINGS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_ltm_embeddings(index_data)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)


def _build_single_entry_embedding(idx: int, row: dict[str, str]) -> tuple[int, dict[str, Any]]:
    """单条 QA 的 embedding 计算，供并行调用。返回 (原始下标, 带 embedding 的条目)。"""
    text = build_embedding_text(row["question"], row["answer"])
    embedding = get_embedding(text)
    return (
        idx,
        {
            "category": row["category"],
            "question": row["question"],
            "answer": row["answer"],
            "embedding": embedding,
        },
    )


def build_ltm_embeddings(
    ltm: dict[str, Any] | None = None,
    *,
    source_path: Path | None = None,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """根据当前 LTM 构建 embedding 索引数据。max_workers>1 时并行计算。"""
    ltm_data = _normalize_ltm(ltm) if ltm is not None else load_ltm(source_path)
    entries = get_all_qa_entries(ltm_data)
    total = len(entries)
    logger.info("开始构建 LTM embedding 索引 entry_count=%s max_workers=%s", total, max_workers)

    if max_workers is not None and max_workers <= 1:
        max_workers = None

    if max_workers is None:
        built_entries = []
        for idx, row in enumerate(entries, start=1):
            _, built = _build_single_entry_embedding(0, row)
            built_entries.append(built)
            if idx == total or idx % 100 == 0:
                logger.info("LTM embedding 构建进度 %s/%s", idx, total)
    else:
        index_to_built: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_build_single_entry_embedding, idx, row): idx
                for idx, row in enumerate(entries)
            }
            done = 0
            for future in as_completed(futures):
                idx, built = future.result()
                index_to_built[idx] = built
                done += 1
                if done == total or done % 100 == 0:
                    logger.info("LTM embedding 构建进度 %s/%s", done, total)
        built_entries = [index_to_built[i] for i in range(len(entries))]

    return {
        "version": "1.0",
        "source_ltm": str(source_path or LTM_PATH),
        "entry_count": len(built_entries),
        "entries": built_entries,
    }


def _normalize_embeddings_matrix(matrix: Any, expected_rows: int) -> np.ndarray:
    if expected_rows <= 0:
        return np.empty((0, 0), dtype=np.float32)
    try:
        arr = np.asarray(matrix, dtype=np.float32)
    except (TypeError, ValueError):
        return np.empty((0, 0), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[0] != expected_rows:
        return np.empty((0, 0), dtype=np.float32)
    return arr


def _load_json_file_with_status(
    path: Path,
    *,
    status_callback: Callable[[str], None] | None = None,
) -> Any:
    if status_callback is None:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    file_size_mb = 0.0
    try:
        file_size_mb = path.stat().st_size / (1024 * 1024)
    except OSError:
        file_size_mb = 0.0

    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}
    done = threading.Event()

    def _worker() -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                result["data"] = json.load(f)
        except BaseException as exc:  # noqa: BLE001
            error["exc"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    start = time.perf_counter()
    status_callback(f"阶段1/3：开始读取旧版 JSON store，文件约 {file_size_mb:.2f} MB")
    while not done.wait(1.0):
        elapsed = int(time.perf_counter() - start)
        status_callback(
            f"阶段1/3：仍在读取旧版 JSON store... 已等待 {elapsed}s，文件约 {file_size_mb:.2f} MB"
        )
    thread.join()
    if "exc" in error:
        raise error["exc"]
    elapsed = time.perf_counter() - start
    status_callback(f"阶段1/3：旧版 JSON store 读取完成，耗时 {elapsed:.2f}s")
    return result.get("data")


def _normalize_ltm_vector_store(
    data: dict[str, Any],
    embeddings_matrix: Any | None = None,
) -> dict[str, Any]:
    out = empty_ltm_vector_store()
    if not isinstance(data, dict):
        return out

    out["version"] = str(data.get("version") or "1.0")
    out["source_ltm"] = str(data.get("source_ltm") or str(LTM_PATH))
    raw_entries = data.get("entries", [])
    if not isinstance(raw_entries, list):
        return out

    valid_entries: list[dict[str, Any]] = []
    for idx, row in enumerate(raw_entries):
        if not isinstance(row, dict):
            continue
        category = str(row.get("category", "")).strip().upper()
        payload = str(row.get("payload", "")).strip()
        question, answer = _decode_store_payload(payload)
        question_hash = str(row.get("question_hash", "")).strip().lower()
        if category not in QUESTION_TYPES_16 or not question or not answer:
            continue
        if not question_hash:
            question_hash = _make_question_hash(question)
        if not question_hash:
            continue
        valid_entries.append(
            {
                "row_index": len(valid_entries),
                "id": str(row.get("id") or f"{category}_{question_hash[:12]}"),
                "category": category,
                "question_hash": question_hash,
                "payload": payload,
                "question": question,
                "answer": answer,
                "source_row_index": idx,
            }
        )

    source_indices = [int(row["source_row_index"]) for row in valid_entries]
    matrix_source = embeddings_matrix if embeddings_matrix is not None else data.get("embeddings_matrix")
    normalized_matrix = _normalize_embeddings_matrix(
        matrix_source,
        len(raw_entries),
    )
    if normalized_matrix.size > 0 and source_indices:
        normalized_matrix = normalized_matrix[source_indices]
    else:
        normalized_matrix = np.empty((0, 0), dtype=np.float32)

    for idx, row in enumerate(valid_entries):
        row["row_index"] = idx

    out["entries"] = valid_entries
    out["entry_count"] = len(valid_entries)
    out["embeddings_matrix"] = normalized_matrix
    return out


def _serialize_ltm_vector_store(
    store_data: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray]:
    raw_entries = store_data.get("entries", [])
    if not isinstance(raw_entries, list):
        raw_entries = []
    raw_matrix = store_data.get("embeddings_matrix")
    normalized_matrix = _normalize_embeddings_matrix(raw_matrix, len(raw_entries))

    meta_rows: list[dict[str, Any]] = []
    kept_vectors: list[np.ndarray] = []
    version = str(store_data.get("version") or "1.0")
    source_ltm = str(store_data.get("source_ltm") or str(LTM_PATH))
    for idx, row in enumerate(raw_entries):
        if not isinstance(row, dict):
            continue
        category = str(row.get("category", "")).strip().upper()
        payload = str(row.get("payload", "")).strip()
        question, answer = _decode_store_payload(payload)
        question_hash = str(row.get("question_hash", "")).strip().lower()
        if category not in QUESTION_TYPES_16 or not question or not answer:
            continue
        if not question_hash:
            question_hash = _make_question_hash(question)
        if not question_hash:
            continue
        if normalized_matrix.size == 0 or idx >= normalized_matrix.shape[0]:
            continue
        meta_rows.append(
            {
                "version": version,
                "source_ltm": source_ltm,
                "id": str(row.get("id") or f"{category}_{question_hash[:12]}"),
                "category": category,
                "question_hash": question_hash,
                "payload": payload,
            }
        )
        kept_vectors.append(normalized_matrix[idx])

    meta_df = pd.DataFrame(
        meta_rows,
        columns=["version", "source_ltm", "id", "category", "question_hash", "payload"],
    )
    if kept_vectors:
        matrix = np.stack(kept_vectors).astype(np.float32, copy=False)
    else:
        matrix = np.empty((0, 0), dtype=np.float32)
    return meta_df, matrix


def load_legacy_json_vector_store(
    path: Path,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """加载旧版 JSON store，并转换为内存中的 parquet+npy 统一结构。"""
    if not path.exists():
        raise FileNotFoundError(f"旧版 LTM store 文件不存在: {path}")
    raw_data = _load_json_file_with_status(path, status_callback=status_callback)
    if not isinstance(raw_data, dict):
        return empty_ltm_vector_store()

    raw_entries = raw_data.get("entries", [])
    if not isinstance(raw_entries, list):
        raw_entries = []

    metadata_entries: list[dict[str, Any]] = []
    embeddings: list[list[float]] = []
    total = len(raw_entries)
    if status_callback is not None:
        status_callback(f"阶段2/3：开始转换条目，共 {total} 条")

    for idx, row in enumerate(raw_entries, start=1):
        if not isinstance(row, dict):
            if progress_callback is not None:
                progress_callback(idx, total)
            continue
        category = str(row.get("category", "")).strip().upper()
        payload = str(row.get("payload", "")).strip()
        question_hash = str(row.get("question_hash", "")).strip().lower()
        question, answer = _decode_store_payload(payload)
        embedding = row.get("embedding")
        if progress_callback is not None:
            progress_callback(idx, total)
        if category not in QUESTION_TYPES_16 or not payload or not question or not answer:
            continue
        if not question_hash:
            question_hash = _make_question_hash(question)
        if not question_hash:
            continue
        if not isinstance(embedding, list) or not embedding:
            continue
        try:
            embedding_values = [float(x) for x in embedding]
        except (TypeError, ValueError):
            continue
        metadata_entries.append(
            {
                "id": str(row.get("id") or f"{category}_{question_hash[:12]}"),
                "category": category,
                "question_hash": question_hash,
                "payload": payload,
            }
        )
        embeddings.append(embedding_values)

    normalized_input = {
        "version": str(raw_data.get("version") or "1.0"),
        "source_ltm": str(raw_data.get("source_ltm") or str(LTM_PATH)),
        "entries": metadata_entries,
    }
    embeddings_matrix = (
        np.asarray(embeddings, dtype=np.float32)
        if embeddings
        else np.empty((0, 0), dtype=np.float32)
    )
    if status_callback is not None:
        status_callback(f"阶段2/3：条目转换完成，有效条目 {len(metadata_entries)} 条")
    return _normalize_ltm_vector_store(normalized_input, embeddings_matrix)


def convert_legacy_json_store_to_parquet_npy(
    json_path: Path,
    *,
    meta_path: Path | None = None,
    embeddings_path: Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """将旧版 JSON store 转换为 parquet+npy 双文件。"""
    store_data = load_legacy_json_vector_store(
        json_path,
        progress_callback=progress_callback,
        status_callback=status_callback,
    )
    if status_callback is not None:
        status_callback("阶段3/3：开始写入 parquet 和 npy 文件")
    save_ltm_vector_store(
        store_data,
        meta_path=meta_path,
        embeddings_path=embeddings_path,
    )
    if status_callback is not None:
        status_callback("阶段3/3：parquet 和 npy 文件写入完成")
    logger.info(
        "旧版 JSON store 转换完成 json_path=%s meta_path=%s embeddings_path=%s entry_count=%s",
        json_path,
        meta_path or LTM_VECTOR_META_PATH,
        embeddings_path or LTM_VECTOR_EMBEDDINGS_PATH,
        store_data.get("entry_count", 0),
    )
    return store_data


def load_ltm_vector_store(
    meta_path: Path | None = None,
    embeddings_path: Path | None = None,
) -> dict[str, Any]:
    """加载运行时 LTM 向量库（Parquet 元数据 + NPY 向量矩阵）。"""
    global _LTM_VECTOR_STORE_CACHE_KEY, _LTM_VECTOR_STORE_CACHE_VALUE
    meta_p = meta_path or LTM_VECTOR_META_PATH
    emb_p = embeddings_path or LTM_VECTOR_EMBEDDINGS_PATH
    if not meta_p.exists() or not emb_p.exists():
        logger.warning(
            "LTM 运行时向量库文件不存在 meta_path=%s embeddings_path=%s",
            meta_p,
            emb_p,
        )
        return empty_ltm_vector_store()
    cache_key = _make_dual_file_cache_key(meta_p, emb_p)
    if (
        cache_key is not None
        and _LTM_VECTOR_STORE_CACHE_KEY == cache_key
        and _LTM_VECTOR_STORE_CACHE_VALUE is not None
    ):
        logger.info(
            "LTM 运行时向量库命中内存缓存 meta_path=%s embeddings_path=%s entry_count=%s",
            meta_p,
            emb_p,
            _LTM_VECTOR_STORE_CACHE_VALUE.get("entry_count", 0),
        )
        return _LTM_VECTOR_STORE_CACHE_VALUE

    meta_df = pd.read_parquet(meta_p)
    if meta_df.empty:
        metadata = {"version": "1.0", "source_ltm": str(LTM_PATH), "entries": []}
    else:
        version = str(meta_df["version"].iloc[0] or "1.0") if "version" in meta_df.columns else "1.0"
        source_ltm = (
            str(meta_df["source_ltm"].iloc[0] or str(LTM_PATH))
            if "source_ltm" in meta_df.columns
            else str(LTM_PATH)
        )
        content_df = meta_df.drop(columns=[x for x in ("version", "source_ltm") if x in meta_df.columns])
        metadata = {
            "version": version,
            "source_ltm": source_ltm,
            "entries": content_df.to_dict(orient="records"),
        }
    embeddings_matrix = np.load(emb_p, allow_pickle=False)
    normalized = _normalize_ltm_vector_store(metadata, embeddings_matrix)
    source_ltm = str(normalized.get("source_ltm") or "").strip()
    logger.info(
        "LTM 运行时向量库首次加载完成 meta_path=%s embeddings_path=%s entry_count=%s build_source_ltm=%s note=%s",
        meta_p,
        emb_p,
        normalized.get("entry_count", 0),
        source_ltm or "N/A",
        "运行时仅加载 parquet+npy；build_source_ltm 只是构建来源记录，不代表当前会读取原始 ltm.json",
    )
    if cache_key is not None:
        _LTM_VECTOR_STORE_CACHE_KEY = cache_key
        _LTM_VECTOR_STORE_CACHE_VALUE = normalized
    return normalized


def save_ltm_vector_store(
    store_data: dict[str, Any],
    meta_path: Path | None = None,
    embeddings_path: Path | None = None,
) -> None:
    """保存运行时 LTM 向量库为 Parquet 元数据和 NPY 向量矩阵。"""
    meta_p = meta_path or LTM_VECTOR_META_PATH
    emb_p = embeddings_path or LTM_VECTOR_EMBEDDINGS_PATH
    meta_p.parent.mkdir(parents=True, exist_ok=True)
    emb_p.parent.mkdir(parents=True, exist_ok=True)
    meta_df, matrix = _serialize_ltm_vector_store(store_data)
    meta_df.to_parquet(meta_p, index=False, engine="pyarrow")
    np.save(emb_p, matrix, allow_pickle=False)


def _build_single_store_entry(idx: int, row: dict[str, str]) -> tuple[int, dict[str, Any]]:
    question = row["question"]
    answer = row["answer"]
    question_hash = _make_question_hash(question)
    text = build_embedding_text(question, answer)
    embedding = get_embedding(text)
    return (
        idx,
        {
            "id": f"{row['category']}_{question_hash[:12]}",
            "category": row["category"],
            "question_hash": question_hash,
            "payload": _encode_store_payload(question, answer),
            "question": question,
            "answer": answer,
            "embedding": [float(x) for x in embedding],
        },
    )


def build_ltm_vector_store(
    ltm: dict[str, Any] | None = None,
    *,
    source_path: Path | None = None,
    max_workers: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """根据当前 LTM 构建运行时向量库。"""
    ltm_data = _normalize_ltm(ltm) if ltm is not None else load_ltm(source_path)
    entries = get_all_qa_entries(ltm_data)
    total = len(entries)
    logger.info("开始构建 LTM 运行时向量库 entry_count=%s max_workers=%s", total, max_workers)

    if max_workers is not None and max_workers <= 1:
        max_workers = None

    if max_workers is None:
        built_entries = []
        for idx, row in enumerate(entries, start=1):
            _, built = _build_single_store_entry(0, row)
            built_entries.append(built)
            if progress_callback is not None:
                progress_callback(idx, total)
            if idx == total or idx % 100 == 0:
                logger.info("LTM 运行时向量库构建进度 %s/%s", idx, total)
    else:
        index_to_built: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_build_single_store_entry, idx, row): idx
                for idx, row in enumerate(entries)
            }
            done = 0
            for future in as_completed(futures):
                idx, built = future.result()
                index_to_built[idx] = built
                done += 1
                if progress_callback is not None:
                    progress_callback(done, total)
                if done == total or done % 100 == 0:
                    logger.info("LTM 运行时向量库构建进度 %s/%s", done, total)
        built_entries = [index_to_built[i] for i in range(len(entries))]

    built_store = {
        "version": "1.0",
        "source_ltm": str(source_path or LTM_PATH),
        "entry_count": len(built_entries),
        "entries": built_entries,
        "embeddings_matrix": (
            np.asarray([row["embedding"] for row in built_entries], dtype=np.float32)
            if built_entries
            else np.empty((0, 0), dtype=np.float32)
        ),
    }
    for row in built_store["entries"]:
        row.pop("embedding", None)
    return built_store


def exact_search_in_vector_store(
    store_data: dict[str, Any],
    question: str,
    question_types: list[str] | None,
) -> list[dict[str, str]]:
    """在运行时向量库内按问题 hash 做精确匹配。"""
    scoped_types = _normalize_scoped_question_types(question_types)
    target_hash = _make_question_hash(question)
    if not target_hash:
        return []
    hits: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in store_data.get("entries", []):
        if not isinstance(row, dict):
            continue
        category = str(row.get("category", "")).strip().upper()
        if scoped_types and category not in scoped_types:
            continue
        if str(row.get("question_hash", "")).strip().lower() != target_hash:
            continue
        question_text = str(row.get("question", "")).strip()
        answer_text = str(row.get("answer", "")).strip()
        if not question_text or not answer_text:
            continue
        key = (category, question_text, answer_text)
        if key in seen:
            continue
        seen.add(key)
        hits.append(
            {
                "category": category,
                "question": question_text,
                "answer": answer_text,
            }
        )
    return hits


def vector_search_in_vector_store_scored(
    question: str,
    *,
    top_k: int = 5,
    question_type: str | None = None,
    question_types: list[str] | None = None,
    store_data: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """在运行时向量库中做向量召回，并返回相似度分数。"""
    q = (question or "").strip()
    if not q:
        return []

    scoped_types = _normalize_scoped_question_types(question_types, question_type)
    loaded_store = (
        store_data
        if isinstance(store_data, dict) and "embeddings_matrix" in store_data
        else _normalize_ltm_vector_store(store_data or {})
        if store_data is not None
        else load_ltm_vector_store()
    )
    indexed_rows = loaded_store.get("entries", [])
    embeddings_matrix = loaded_store.get("embeddings_matrix")
    if not isinstance(indexed_rows, list):
        indexed_rows = []
    indexed_matrix = _normalize_embeddings_matrix(embeddings_matrix, len(indexed_rows))
    if scoped_types:
        scoped_pairs = [
            (idx, row)
            for idx, row in enumerate(indexed_rows)
            if isinstance(row, dict) and row.get("category") in scoped_types
        ]
        scoped_indices = [idx for idx, _ in scoped_pairs]
        indexed_rows = [row for _, row in scoped_pairs]
        if indexed_matrix.size > 0 and scoped_indices:
            indexed_matrix = indexed_matrix[scoped_indices]
        else:
            indexed_matrix = np.empty((0, 0), dtype=np.float32)
    elif question_types is not None or question_type:
        indexed_rows = []
        indexed_matrix = np.empty((0, 0), dtype=np.float32)
    if not indexed_rows:
        logger.warning("运行时向量库检索无可用候选 scoped_types=%s", scoped_types)
        return []

    q_emb = get_embedding(q)
    scored: list[tuple[float, dict[str, Any]]] = []
    if indexed_matrix.size == 0 or indexed_matrix.shape[0] != len(indexed_rows):
        logger.warning("运行时向量库检索向量矩阵不可用 scoped_types=%s", scoped_types)
        return []
    for idx, row in enumerate(indexed_rows):
        scored.append((_cosine(q_emb, indexed_matrix[idx].tolist()), row))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_rows = [
        {
            "category": str(row["category"]),
            "question": str(row["question"]),
            "answer": str(row["answer"]),
            "similarity": score,
        }
        for score, row in scored[: max(1, top_k)]
    ]
    logger.info(
        "运行时向量库检索 top_hits=%s",
        [
            {
                "category": row["category"],
                "similarity": round(float(row["similarity"]), 4),
                "question_preview": row["question"].replace("\n", " ")[:100],
            }
            for row in top_rows[: min(3, len(top_rows))]
        ],
    )
    return top_rows


def retrieve_knowledge_bundle_from_vector_store(
    question: str,
    question_types: list[str] | None,
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    """运行时向量库的统一专家检索入口。"""
    search_types = _normalize_scoped_question_types(question_types)
    store_data = load_ltm_vector_store()
    exact_hits = exact_search_in_vector_store(store_data, question, search_types)
    if exact_hits:
        return {
            "knowledge": format_qa_context(exact_hits),
            "exact_found": True,
            "search_types": search_types,
            "hit_count": len(exact_hits),
            "canonical_answer": str(exact_hits[0].get("answer", "")).strip(),
        }

    vector_hits = vector_search_in_vector_store_scored(
        question,
        top_k=top_k,
        question_types=search_types,
        store_data=store_data,
    )
    return {
        "knowledge": format_qa_context(vector_hits) if vector_hits else "",
        "exact_found": False,
        "search_types": search_types,
        "hit_count": len(vector_hits),
        "canonical_answer": "",
    }


def exact_search_in_category(
    ltm: dict[str, Any],
    question: str,
    question_type: str,
) -> list[dict[str, str]]:
    """在指定类别做精确匹配（规范化后字符串相等）。"""
    qt = (question_type or "").strip().upper()
    if qt not in QUESTION_TYPES_16:
        return []
    target = _normalize_text(question)
    if not target:
        return []
    rows = ((ltm or {}).get("categories", {}) or {}).get(qt, [])
    if not isinstance(rows, list):
        return []
    hits: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        q = str(row.get("question", "")).strip()
        a = str(row.get("answer", "")).strip()
        if q and a and _normalize_text(q) == target:
            hits.append({"category": qt, "question": q, "answer": a})
    return hits


def exact_search_in_categories(
    ltm: dict[str, Any],
    question: str,
    question_types: list[str] | None,
) -> list[dict[str, str]]:
    """在多个指定类别内做精确匹配（规范化后字符串相等）。"""
    ordered_types: list[str] = []
    for item in question_types or []:
        qt = (item or "").strip().upper()
        if qt in QUESTION_TYPES_16 and qt not in ordered_types:
            ordered_types.append(qt)

    hits: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for qt in ordered_types:
        for row in exact_search_in_category(ltm, question, qt):
            key = (row["category"], row["question"], row["answer"])
            if key in seen:
                continue
            seen.add(key)
            hits.append(row)
    return hits


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def vector_search_qa(
    ltm: dict[str, Any],
    question: str,
    *,
    top_k: int = 5,
    question_type: str | None = None,
) -> list[dict[str, str]]:
    """在全库或指定类别做向量检索。"""
    rows = vector_search_qa_scored(
        ltm,
        question,
        top_k=top_k,
        question_type=question_type,
        question_types=None,
    )
    return [
        {
            "category": str(row.get("category", "")),
            "question": str(row.get("question", "")),
            "answer": str(row.get("answer", "")),
        }
        for row in rows
    ]


def vector_search_qa_scored(
    ltm: dict[str, Any],
    question: str,
    *,
    top_k: int = 5,
    question_type: str | None = None,
    question_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    """优先基于预计算 embedding 索引做向量检索，并返回相似度分数。"""
    q = (question or "").strip()
    if not q:
        return []

    scoped_types = _normalize_scoped_question_types(question_types, question_type)
    index_data = load_ltm_embeddings()
    indexed_rows = index_data.get("entries", [])
    if not isinstance(indexed_rows, list):
        indexed_rows = []
    if scoped_types:
        indexed_rows = [
            row for row in indexed_rows
            if isinstance(row, dict) and row.get("category") in scoped_types
        ]
    elif question_types is not None or question_type:
        indexed_rows = []

    q_emb = get_embedding(q)
    scored: list[tuple[float, dict[str, Any]]] = []

    if indexed_rows:
        logger.info(
            "向量检索使用预计算 embedding 索引 scoped_types=%s candidates=%s",
            scoped_types,
            len(indexed_rows),
        )
        for row in indexed_rows:
            embedding = row.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                continue
            scored.append((_cosine(q_emb, embedding), row))
    else:
        all_rows = get_all_qa_entries(ltm)
        if scoped_types:
            all_rows = [x for x in all_rows if x["category"] in scoped_types]
        elif question_types is not None or question_type:
            all_rows = []
        if not all_rows:
            logger.warning("向量检索无可用候选：embedding 索引为空且 LTM 范围内无条目 scoped_types=%s", scoped_types)
            return []
        logger.warning(
            "向量检索回退到在线 embedding 计算 scoped_types=%s candidates=%s",
            scoped_types,
            len(all_rows),
        )
        for row in all_rows:
            text = build_embedding_text(row["question"], row["answer"])
            emb = get_embedding(text)
            scored.append((_cosine(q_emb, emb), row))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_rows = [
        {
            "category": str(row["category"]),
            "question": str(row["question"]),
            "answer": str(row["answer"]),
            "similarity": score,
        }
        for score, row in scored[: max(1, top_k)]
    ]
    logger.info(
        "向量检索 top_hits=%s",
        [
            {
                "category": row["category"],
                "similarity": round(float(row["similarity"]), 4),
                "question_preview": row["question"].replace("\n", " ")[:100],
            }
            for row in top_rows[: min(3, len(top_rows))]
        ],
    )
    return top_rows


def format_qa_context(rows: list[dict[str, str]]) -> str:
    """将检索条目格式化为 LLM 上下文。"""
    if not rows:
        return "（知识库暂无匹配）"
    chunks = []
    for row in rows:
        chunks.append(
            f"[{row['category']}]\nQuestion: {row['question']}\nAnswer: {row['answer']}"
        )
    return "\n\n".join(chunks)
