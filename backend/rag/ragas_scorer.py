"""
ragas_scorer.py — RAGAS evaluation for the CiteRAG pipeline
============================================================

Runs all four RAGAS metrics against a retrieved answer and its supporting
chunks. Ground truth is resolved from `qa_dataset.json` when available.
All matched metrics run in parallel via `asyncio.gather`.

Metrics:
    faithfulness       — measures hallucination; no ground truth required.
    answer_relevancy   — measures topic drift; no ground truth required.
    context_precision  — requires ground truth (skipped if none found).
    context_recall     — requires ground truth (skipped if none found).

Compatible with RAGAS v0.1 and v0.2+; version is detected at runtime.
"""

import json
import logging
import asyncio
import contextlib
import importlib.metadata
import io
import os
from pathlib import Path
from typing import Optional

from datasets import Dataset
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from ragas import evaluate as ragas_evaluate, EvaluationDataset
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from backend.core.config import settings
from backend.core.logger import logger

# ── Path to labeled dataset ───────────────────────────────────────────────────
QA_DATASET_PATH = Path(__file__).parent / "qa_dataset.json"

# ── Singleton state ───────────────────────────────────────────────────────────
_ragas_ready       = False
_faithfulness      = None
_answer_relevancy  = None
_context_precision = None
_context_recall    = None
_ragas_llm         = None
_ragas_emb         = None
_qa_map            = None


# ── Load dataset ──────────────────────────────────────────────────────────────

def _load_qa_dataset() -> dict:
    """
    Load qa_dataset.json and build a normalized lookup dict.
    { "what are the leave policy details": "The company's employees are entitled to..." }
    """
    global _qa_map
    if _qa_map is not None:
        return _qa_map

    try:
        with open(QA_DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        _qa_map = {
            entry["question"].strip().lower(): entry["ground_truth"].strip()
            for entry in data
            if entry.get("question") and entry.get("ground_truth")
        }
        logger.info("QA dataset loaded: %d pairs", len(_qa_map))
    except FileNotFoundError:
        logger.info("Optional qa_dataset.json not found — context_recall will be skipped.")
        _qa_map = {}
    except Exception as e:
        logger.warning("QA dataset load failed: %s", e)
        _qa_map = {}

    return _qa_map


def _lookup_ground_truth(question: str) -> Optional[str]:
    """
    Find ground truth for a question.
    1. Exact normalized match
    2. Keyword overlap fallback (>=40% shared words)
    """
    qa_map = _load_qa_dataset()
    if not qa_map:
        return None

    q_norm = question.strip().lower()

    # Exact match
    if q_norm in qa_map:
        logger.info("GT exact match for: %s", q_norm[:60])
        return qa_map[q_norm]

    # Keyword overlap
    q_words = set(w for w in q_norm.split() if len(w) > 3)
    if not q_words:
        return None

    best_key, best_score = None, 0.0
    for key in qa_map:
        key_words = set(w for w in key.split() if len(w) > 3)
        union = len(q_words | key_words)
        if union == 0:
            continue
        overlap = len(q_words & key_words) / union
        if overlap > best_score and overlap >= 0.40:
            best_score = overlap
            best_key   = key

    if best_key:
        logger.info("GT keyword match (%.0f%%): %s", best_score * 100, best_key[:60])
        return qa_map[best_key]

    logger.info("No GT match — context_recall will be None for: %s", q_norm[:60])
    return None


# ── RAGAS version detection (cached) ───────────────────────────────────────

_ragas_version_cache = None

def _get_ragas_version() -> tuple[int, int]:
    """Return (major, minor) of installed ragas package. Cached after first call."""
    global _ragas_version_cache
    if _ragas_version_cache is not None:
        return _ragas_version_cache
    try:
        v = importlib.metadata.version("ragas")
        parts = v.split(".")
        _ragas_version_cache = (int(parts[0]), int(parts[1]))
    except Exception:
        _ragas_version_cache = (0, 1)  # assume old if unknown
    return _ragas_version_cache


# ── RAGAS init ────────────────────────────────────────────────────────────────

def _init_ragas() -> bool:
    """
    Initialize RAGAS metric singletons with Azure OpenAI LLM and embedding models.

    Detects the installed RAGAS version and initialises the appropriate metric
    objects (v0.2+ class-based API vs v0.1 module-level singleton API).
    Idempotent: returns True immediately if already initialised.

    Returns:
        True if initialisation succeeded, False on import or configuration error.
    """
    global _ragas_ready, _faithfulness, _answer_relevancy
    global _context_precision, _context_recall, _ragas_llm, _ragas_emb

    if _ragas_ready:
        return True

    try:
        major, minor = _get_ragas_version()
        logger.info("Detected RAGAS version %d.%d", major, minor)

        judge_llm = AzureChatOpenAI(
            azure_endpoint=settings.AZURE_LLM_ENDPOINT,
            api_key=settings.AZURE_OPENAI_LLM_KEY,
            azure_deployment=settings.AZURE_LLM_DEPLOYMENT_41_MINI,
            api_version="2024-12-01-preview",
            temperature=0,
        )
        from backend.core.vector import get_embedder
        judge_emb = get_embedder()

        if major == 0 and minor >= 2:
            _ragas_llm = LangchainLLMWrapper(judge_llm)
            _ragas_emb = LangchainEmbeddingsWrapper(judge_emb)

            _faithfulness      = Faithfulness(llm=_ragas_llm)
            _answer_relevancy  = AnswerRelevancy(llm=_ragas_llm, embeddings=_ragas_emb)
            _context_precision = ContextPrecision(llm=_ragas_llm)
            _context_recall    = ContextRecall(llm=_ragas_llm)

        else:
            _ragas_llm = LangchainLLMWrapper(judge_llm)
            _ragas_emb = LangchainEmbeddingsWrapper(judge_emb)

            _faithfulness      = faithfulness
            _answer_relevancy  = answer_relevancy
            _context_precision = context_precision
            _context_recall    = context_recall

            _faithfulness.llm            = _ragas_llm
            _answer_relevancy.llm        = _ragas_llm
            _answer_relevancy.embeddings = _ragas_emb
            _context_precision.llm       = _ragas_llm
            _context_recall.llm          = _ragas_llm

        _ragas_ready = True
        logger.info("RAGAS scorer initialized (v%d.%d)", major, minor)
        return True

    except ImportError as e:
        logger.error("RAGAS import failed — is ragas installed? Error: %s", e)
        return False
    except Exception as e:
        logger.error("RAGAS init failed: %s", e, exc_info=True)
        return False


# ── Single metric runner ──────────────────────────────────────────────────────

def _run_single_metric(metric, data) -> Optional[float]:
    """
    Run one RAGAS metric synchronously inside a thread executor.

    Automatically selects the RAGAS v0.1 or v0.2+ evaluation path based on
    the installed package version. tqdm output is suppressed globally to keep
    logs clean.

    Args:
        metric: An initialised RAGAS metric object.
        data:   For v0.2+, an `EvaluationDataset`; for v0.1, a HuggingFace `Dataset`.

    Returns:
        A float score in [0, 1], or None if evaluation fails.
    """
    os.environ["TQDM_DISABLE"] = "1"

    major, minor = _get_ragas_version()
    metric_name  = getattr(metric, "name", type(metric).__name__)

    try:
        if major == 0 and minor >= 2:
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                result = ragas_evaluate(data, metrics=[metric])

            df = result.to_pandas()
            if metric_name in df.columns:
                col = metric_name
            else:
                lower_map = {c.lower(): c for c in df.columns}
                col = lower_map.get(metric_name.lower())
                if col is None:
                    skip = {"user_input", "response", "retrieved_contexts", "reference"}
                    numeric_cols = [c for c in df.columns if c not in skip]
                    col = numeric_cols[-1] if numeric_cols else df.columns[-1]
            val = df.iloc[0][col]
            return round(float(val), 3)

        else:
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                result = ragas_evaluate(data, metrics=[metric])

            df  = result.to_pandas()
            col = df.columns[-1]
            return round(float(df.iloc[0][col]), 3)

    except Exception as e:
        logger.error("Metric %s failed: %s", metric_name, e, exc_info=True)
        return None


# ── Main scorer ───────────────────────────────────────────────────────────────

async def score(
    question:     str,
    answer:       str,
    chunks:       list,
    ground_truth: Optional[str] = None,
) -> Optional[dict]:
    """
    Run RAGAS evaluation against a retrieved answer and its source chunks.

    `context_precision` and `context_recall` require ground truth; they are
    skipped and set to None when no matching entry is found in `qa_dataset.json`.
    All metrics that can run do so in parallel via `asyncio.gather`.

    Args:
        question:     The user's original question.
        answer:       The LLM-generated answer to evaluate.
        chunks:       Retrieved chunks (list of dicts with a `content` or `text` key).
        ground_truth: Optional override from a labeled evaluation set; if not
                      provided, a match is attempted from `qa_dataset.json`.

    Returns:
        A dict with keys `faithfulness`, `answer_relevancy`, `context_precision`,
        and `context_recall` (float or None), or None if evaluation cannot run.
    """
    if not chunks or not answer or not question:
        return None

    if not _init_ragas():
        return None

    contexts = [
        c.get("content", c.get("text", ""))
        for c in chunks
        if c.get("content") or c.get("text")
    ]
    if not contexts:
        return None

    real_gt = ground_truth or _lookup_ground_truth(question)

    major, minor = _get_ragas_version()
    logger.info("Running RAGAS scoring with v%d.%d", major, minor)

    if major == 0 and minor >= 2:
        eval_no_gt = EvaluationDataset.from_list([{
            "user_input":         question,
            "response":           answer,
            "retrieved_contexts": contexts,
        }])

        eval_with_gt = EvaluationDataset.from_list([{
            "user_input":         question,
            "response":           answer,
            "retrieved_contexts": contexts,
            "reference":          real_gt,
        }]) if real_gt else None

        data_no_gt   = eval_no_gt
        data_with_gt = eval_with_gt
    else:
        data_no_gt = Dataset.from_dict({
            "question": [question],
            "answer":   [answer],
            "contexts": [contexts],
        })

        data_with_gt = Dataset.from_dict({
            "question":     [question],
            "answer":       [answer],
            "contexts":     [contexts],
            "ground_truth": [real_gt],
        }) if real_gt else None

    loop = asyncio.get_event_loop()

    async def _run(metric, dataset):
        if dataset is None:
            return None
        return await loop.run_in_executor(
            None, _run_single_metric, metric, dataset
        )

    # Run all 4 in parallel — context_recall/context_precision get None dataset if no GT
    # NOTE: In v0.2, ContextPrecision also requires reference (ground truth)
    results = await asyncio.gather(
        _run(_faithfulness,      data_no_gt),
        _run(_answer_relevancy,  data_no_gt),
        _run(_context_precision, data_with_gt),   # needs GT in v0.2
        _run(_context_recall,    data_with_gt),
        return_exceptions=True,
    )

    def _unwrap(r):
        return r if not isinstance(r, Exception) else None

    scores = {
        "faithfulness":      _unwrap(results[0]),
        "answer_relevancy":  _unwrap(results[1]),
        "context_precision": _unwrap(results[2]),
        "context_recall":    _unwrap(results[3]),  # None if no GT match
    }

    logger.info(
        "RAGAS scores — faith:%s rel:%s prec:%s rec:%s",
        scores["faithfulness"], scores["answer_relevancy"],
        scores["context_precision"], scores["context_recall"],
    )
    return scores