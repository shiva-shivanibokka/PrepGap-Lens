
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

from core.skillmap import SkillMap, Skill


# -----------------------------
# Output types
# -----------------------------

@dataclass(frozen=True)
class SkillEvidence:
    """Why we believe the user has (or doesn't have) a skill."""
    skill_id: str
    matched_phrases: List[str]
    match_count: int


@dataclass(frozen=True)
class ExtractionResult:
    """
    coverage_by_skill: skill_id -> confidence in [0, 1]
    evidence_by_skill: skill_id -> evidence info (what matched)
    """
    coverage_by_skill: Dict[str, float]
    evidence_by_skill: Dict[str, SkillEvidence]


# -----------------------------
# Public API
# -----------------------------

def extract_skill_coverage(
    user_text: str,
    skill_map: SkillMap,
    *,
    synonym_phrases: Optional[Dict[str, List[str]]] = None,
    max_confidence: float = 0.95,
) -> ExtractionResult:
    """
    Extract a rough skill coverage signal from user-provided text (resume paste, skills, projects).
    Hackathon-MVP approach:
      - normalize text
      - phrase/keyword matching per skill
      - convert match count into confidence

    You can later upgrade this to embeddings / LLM-based extraction, but this is reliable and fast.

    Parameters
    ----------
    user_text : str
        Resume or skills text pasted by user.
    skill_map : SkillMap
        Loaded skill map containing skills + keywords.
    synonym_phrases : dict[str, list[str]] | None
        Extra phrases to match per skill_id. Useful when resume wording varies.
        Example: {"cv_leakage": ["data leakage", "leakage", "pipeline"]}.
    max_confidence : float
        Upper cap so we never claim 100% based on simple keyword hits.

    Returns
    -------
    ExtractionResult
    """
    synonym_phrases = synonym_phrases or DEFAULT_SYNONYMS

    text = _normalize(user_text)
    coverage: Dict[str, float] = {}
    evidence: Dict[str, SkillEvidence] = {}

    for skill in skill_map.skills_by_id.values():
        phrases = _skill_phrases(skill, synonym_phrases)
        matched = _find_matches(text, phrases)

        # Simple scoring: more unique matches => higher confidence.
        # Tuned for hackathon demo: 0, 0.35, 0.55, 0.7, 0.82, 0.9-ish
        conf = _confidence_from_matches(len(matched))

        # If user literally mentions the skill name, boost slightly (but still cap).
        if _phrase_in_text(text, _normalize(skill.name)):
            conf = min(max_confidence, conf + 0.10)

        coverage[skill.id] = float(min(max_confidence, conf))
        evidence[skill.id] = SkillEvidence(
            skill_id=skill.id,
            matched_phrases=sorted(matched),
            match_count=len(matched),
        )

    return ExtractionResult(coverage_by_skill=coverage, evidence_by_skill=evidence)


def summarize_top_evidence(
    extraction: ExtractionResult,
    skill_map: SkillMap,
    *,
    min_confidence: float = 0.55,
    top_k: int = 8,
) -> List[str]:
    """
    Return human-readable evidence lines for the demo UI:
      "Model Evaluation: matched {precision, recall, roc auc}"
    """
    items: List[Tuple[float, str]] = []
    for sid, conf in extraction.coverage_by_skill.items():
        if conf < min_confidence:
            continue
        skill = skill_map.skills_by_id[sid]
        ev = extraction.evidence_by_skill[sid]
        if ev.match_count == 0:
            continue
        matches = ", ".join(ev.matched_phrases[:6])
        line = f"{skill.name} (conf {conf:.2f}): matched [{matches}]"
        items.append((conf, line))

    items.sort(key=lambda x: -x[0])
    return [line for _, line in items[: max(0, top_k)]]


# -----------------------------
# Defaults / helpers
# -----------------------------

# Skill-specific extra phrases you expect in resumes (beyond the JSON keywords).
# Keep this small and high-quality. You can add as you test with your own resume text.
DEFAULT_SYNONYMS: Dict[str, List[str]] = {
    "ml_basics": [
        "supervised learning",
        "classification",
        "regression",
        "gradient descent",
        "loss",
        "objective function",
    ],
    "bias_variance": [
        "generalization",
        "regularization",
        "ridge",
        "lasso",
        "overfit",
        "underfit",
    ],
    "metrics": [
        "confusion matrix",
        "roc auc",
        "auc",
        "precision",
        "recall",
        "f1",
        "threshold",
    ],
    "cv_leakage": [
        "cross validation",
        "kfold",
        "stratified kfold",
        "data leakage",
        "train test split",
        "pipeline",
    ],
    "pandas_cleaning": [
        "pandas",
        "data cleaning",
        "imputation",
        "missing values",
        "outliers",
        "data preprocessing",
    ],
    "feature_engineering": [
        "feature engineering",
        "one hot",
        "encoding",
        "standardization",
        "normalization",
        "scaling",
    ],
    "trees_boosting": [
        "random forest",
        "xgboost",
        "lightgbm",
        "decision tree",
        "boosting",
        "bagging",
    ],
    "nn_basics": [
        "neural network",
        "deep learning",
        "backprop",
        "relu",
        "dropout",
    ],
    "ml_systems_basics": [
        "fastapi",
        "flask",
        "api",
        "inference",
        "latency",
        "deployment",
    ],
    "ml_storytelling": [
        "error analysis",
        "tradeoff",
        "stakeholders",
        "model choice",
        "explainability",
    ],
}


def _skill_phrases(skill: Skill, synonym_phrases: Dict[str, List[str]]) -> List[str]:
    phrases: List[str] = []
    phrases.extend(skill.keywords or [])
    phrases.extend(synonym_phrases.get(skill.id, []))

    # Normalize + dedupe
    out: List[str] = []
    seen = set()
    for p in phrases:
        p2 = _normalize(p)
        if not p2:
            continue
        if p2 in seen:
            continue
        seen.add(p2)
        out.append(p2)
    return out


def _normalize(s: str) -> str:
    s = s.lower()
    # normalize common punctuation to spaces
    s = re.sub(r"[^a-z0-9+\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _phrase_in_text(norm_text: str, norm_phrase: str) -> bool:
    if not norm_phrase:
        return False
    # word-boundary-ish match
    return re.search(rf"(^| )({re.escape(norm_phrase)})( |$)", norm_text) is not None


def _find_matches(norm_text: str, norm_phrases: Iterable[str]) -> List[str]:
    matched: List[str] = []
    for p in norm_phrases:
        if _phrase_in_text(norm_text, p):
            matched.append(p)
    return matched


def _confidence_from_matches(unique_match_count: int) -> float:
    """
    Convert number of unique matches into a confidence score.
    Tuned for hackathon: conservative but responsive.
    """
    if unique_match_count <= 0:
        return 0.05
    if unique_match_count == 1:
        return 0.35
    if unique_match_count == 2:
        return 0.55
    if unique_match_count == 3:
        return 0.70
    if unique_match_count == 4:
        return 0.82
    return 0.90
