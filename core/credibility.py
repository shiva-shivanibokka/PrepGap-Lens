
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Literal, Tuple, Dict

from core.skillmap import Skill, Resource


Credibility = Literal["interview-safe", "good-intuition", "misleading"]


# -----------------------------
# Simple credibility helpers
# -----------------------------

CRED_RANK: Dict[str, int] = {
    "interview-safe": 0,
    "good-intuition": 1,
    "misleading": 2,
}


def credibility_rank(label: str) -> int:
    """Lower is better."""
    return CRED_RANK.get(label, 9)


def best_resources(
    skill: Skill,
    *,
    prefer_type: Optional[str] = None,
    max_items: int = 3,
) -> List[Resource]:
    """
    Return best resources for a skill, using credibility first and type preference second.

    prefer_type examples: "youtube", "docs"
    """
    resources = list(skill.resources) if skill.resources else []
    if not resources:
        return []

    def key(r: Resource) -> Tuple[int, int, str]:
        t = 0
        if prefer_type is not None:
            t = 0 if r.type == prefer_type else 1
        return (credibility_rank(r.credibility), t, r.title.lower())

    resources.sort(key=key)
    return resources[: max(0, max_items)]


def label_explanation(label: str) -> str:
    """
    Short explanation you can display in the demo UI.
    """
    if label == "interview-safe":
        return "Accurate + sufficient depth for interviews."
    if label == "good-intuition":
        return "Helpful intuition, may miss edge cases—pair with practice."
    if label == "misleading":
        return "Risky/outdated/oversimplified—use cautiously."
    return "Unknown credibility."


def flag_misleading_resources(skill: Skill) -> List[Resource]:
    """Return resources explicitly labeled as misleading."""
    return [r for r in (skill.resources or []) if r.credibility == "misleading"]


def summarize_resources_for_skill(skill: Skill) -> List[str]:
    """
    Human-readable resource lines for printing.
    """
    lines: List[str] = []
    for r in best_resources(skill, max_items=5):
        lines.append(f"- [{r.credibility}] {r.title} ({r.type}) -> {r.url}")
    return lines
