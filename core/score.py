
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from core.skillmap import SkillMap, Skill
from core.extract import ExtractionResult


# -----------------------------
# Output types
# -----------------------------

@dataclass(frozen=True)
class Gap:
    skill_id: str
    skill_name: str
    category: str
    weight: int
    coverage: float          # [0, 1]
    gap: float               # 1 - coverage
    prereqs_missing: List[str]


@dataclass(frozen=True)
class ScoreReport:
    role: str
    version: str
    readiness_overall: float                     # [0, 1]
    readiness_by_category: Dict[str, float]      # category -> [0, 1]
    coverage_by_skill: Dict[str, float]          # skill_id -> [0, 1]
    top_gaps: List[Gap]


# -----------------------------
# Public API
# -----------------------------

def score_readiness(
    skill_map: SkillMap,
    extraction: ExtractionResult,
    *,
    top_k_gaps: int = 6,
    prereq_penalty: float = 0.10,
) -> ScoreReport:
    """
    Compute readiness for the role based on extracted coverage.

    MVP scoring approach:
      - readiness(skill) = coverage(skill) (capped in extract.py)
      - overall readiness = weighted average across all skills
      - category readiness = weighted average within each category
      - gaps = (1 - coverage) * weight, with extra penalty if prereqs are missing

    prereq_penalty:
      Adds extra gap weight when a skill's prereqs have low coverage (common real-world failure mode).
    """
    coverage_by_skill = dict(extraction.coverage_by_skill)

    # Overall readiness: weighted average
    total_w = 0.0
    total = 0.0
    for skill in skill_map.skills_by_id.values():
        w = float(skill.weight)
        c = float(_clamp01(coverage_by_skill.get(skill.id, 0.0)))
        total_w += w
        total += w * c

    readiness_overall = (total / total_w) if total_w > 0 else 0.0

    # Category readiness: weighted average per category
    readiness_by_cat: Dict[str, float] = {}
    for cat in skill_map.categories:
        skills = [s for s in skill_map.skills_by_id.values() if s.category == cat]
        cw = 0.0
        ct = 0.0
        for s in skills:
            w = float(s.weight)
            c = float(_clamp01(coverage_by_skill.get(s.id, 0.0)))
            cw += w
            ct += w * c
        readiness_by_cat[cat] = (ct / cw) if cw > 0 else 0.0

    # Compute gaps
    gaps = _compute_gaps(
        skill_map=skill_map,
        coverage_by_skill=coverage_by_skill,
        prereq_penalty=prereq_penalty,
    )
    gaps.sort(key=lambda g: _gap_priority(g))
    top_gaps = gaps[: max(0, top_k_gaps)]

    return ScoreReport(
        role=skill_map.role,
        version=skill_map.version,
        readiness_overall=float(_clamp01(readiness_overall)),
        readiness_by_category={k: float(_clamp01(v)) for k, v in readiness_by_cat.items()},
        coverage_by_skill={k: float(_clamp01(v)) for k, v in coverage_by_skill.items()},
        top_gaps=top_gaps,
    )


def recommend_focus_skills(
    report: ScoreReport,
    skill_map: SkillMap,
    *,
    n: int = 3,
) -> List[Skill]:
    """
    Return the top N skills to focus on next (based on report.top_gaps).
    """
    ids = [g.skill_id for g in report.top_gaps[: max(0, n)]]
    return [skill_map.skills_by_id[i] for i in ids if i in skill_map.skills_by_id]


# -----------------------------
# Internal helpers
# -----------------------------

def _compute_gaps(
    skill_map: SkillMap,
    coverage_by_skill: Dict[str, float],
    prereq_penalty: float,
) -> List[Gap]:
    out: List[Gap] = []
    for skill in skill_map.skills_by_id.values():
        c = float(_clamp01(coverage_by_skill.get(skill.id, 0.0)))
        gap = 1.0 - c

        prereqs_missing = []
        prereq_gap_factor = 0.0

        for pre_id in (skill.prereqs or []):
            pre_cov = float(_clamp01(coverage_by_skill.get(pre_id, 0.0)))
            if pre_cov < 0.55:  # threshold: "not really there yet"
                prereqs_missing.append(pre_id)
                prereq_gap_factor += (0.55 - pre_cov) / 0.55  # 0..1-ish

        # Add an extra penalty if prereqs are missing
        # This prioritizes foundational gaps naturally.
        if prereqs_missing:
            gap = min(1.0, gap + prereq_penalty * min(1.0, prereq_gap_factor))

        out.append(
            Gap(
                skill_id=skill.id,
                skill_name=skill.name,
                category=skill.category,
                weight=int(skill.weight),
                coverage=c,
                gap=float(_clamp01(gap)),
                prereqs_missing=prereqs_missing,
            )
        )
    return out


def _gap_priority(g: Gap) -> Tuple[float, int, float, str]:
    """
    Higher priority = bigger impact.
    Sort by:
      - (gap * weight) descending
      - weight descending
      - gap descending
      - name ascending
    """
    impact = g.gap * float(g.weight)
    return (-impact, -g.weight, -g.gap, g.skill_name.lower())


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x
