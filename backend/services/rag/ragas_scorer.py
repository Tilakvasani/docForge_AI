"""
ragas_scorer.py — Inline RAGAS scoring for CiteRAG Lab
────────────────────────────────────────────────────────
ALL 4 metrics are REAL. No proxies. No circular logic.

Ground truth comes from qa_dataset.json — 15 human-written QA pairs
covering every document type in your system. When a user question
matches a dataset entry, all 4 metrics run including real context_recall.
When no match, 3 metrics run (faithfulness, answer_relevancy, context_precision).

All matched metrics run in parallel via asyncio.gather.
"""

import json
import logging
import asyncio
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

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
_qa_map            = None   # { normalized_question: ground_truth }


# ── Load dataset ──────────────────────────────────────────────────────────────

def _load_qa_dataset() -> dict:
    """
    Load qa_dataset.json and build a normalized lookup dict.
    { "what are the leave policy details": "Turabit employees are entitled to..." }
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
        logger.warning("qa_dataset.json not found at %s", QA_DATASET_PATH)
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


# ── RAGAS init ────────────────────────────────────────────────────────────────

def _init_ragas() -> bool:
    global _ragas_ready, _faithfulness, _answer_relevancy
    global _context_precision, _context_recall, _ragas_llm, _ragas_emb

    if _ragas_ready:
        return True

    try:
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
        from backend.core.config import settings

        judge_llm = AzureChatOpenAI(
            azure_endpoint=settings.AZURE_LLM_ENDPOINT,
            api_key=settings.AZURE_OPENAI_LLM_KEY,
            azure_deployment=settings.AZURE_LLM_DEPLOYMENT_41_MINI,
            api_version="2024-12-01-preview",
            temperature=0,
        )
        judge_emb = AzureOpenAIEmbeddings(
            azure_endpoint=settings.AZURE_EMB_ENDPOINT,
            api_key=settings.AZURE_OPENAI_EMB_KEY,
            azure_deployment=settings.AZURE_EMB_DEPLOYMENT,
            api_version=settings.AZURE_EMB_API_VERSION,
        )

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
        logger.info("RAGAS scorer initialized")
        return True

    except ImportError:
        logger.warning("RAGAS not installed. Run: pip install ragas datasets")
        return False
    except Exception as e:
        logger.warning("RAGAS init failed: %s", e)
        return False


# ── Single metric runner ──────────────────────────────────────────────────────

def _run_single_metric(metric, data) -> Optional[float]:
    """Run one RAGAS metric synchronously inside a thread executor."""
    try:
        from ragas import evaluate
        result = evaluate(data, metrics=[metric])
        df  = result.to_pandas()
        col = df.columns[-1]
        return round(float(df.iloc[0][col]), 3)
    except Exception as e:
        logger.warning("Metric %s failed: %s", getattr(metric, "name", str(metric)), e)
        return None


# ── Main scorer ───────────────────────────────────────────────────────────────

async def score(
    question:     str,
    answer:       str,
    chunks:       list,
    ground_truth: Optional[str] = None,
) -> Optional[dict]:
    """
    Run real RAGAS evaluation.

    faithfulness      — always real (no GT needed)
    answer_relevancy  — always real (no GT needed)
    context_precision — always real (no GT needed)
    context_recall    — real when GT found in qa_dataset.json
                        None when question has no dataset match

    All metrics run in parallel via asyncio.gather.

    Args:
        question      — user's original question
        answer        — LLM-generated answer
        chunks        — retrieved chunks (list of dicts with 'content' key)
        ground_truth  — optional override from a labeled eval set
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

    # Resolve ground truth: caller override > dataset lookup > None
    real_gt = ground_truth or _lookup_ground_truth(question)

    # Build datasets
    from datasets import Dataset

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

    # Run all 4 in parallel — context_recall gets None dataset if no GT found
    results = await asyncio.gather(
        _run(_faithfulness,      data_no_gt),
        _run(_answer_relevancy,  data_no_gt),
        _run(_context_precision, data_no_gt),
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