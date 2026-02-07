
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# -----------------------------
# Data models
# -----------------------------

@dataclass(frozen=True)
class Resource:
    title: str
    type: str                # "youtube" | "docs" | "blog" | ...
    credibility: str         # "interview-safe" | "good-intuition" | "misleading"
    reason: str
    url: str


@dataclass(frozen=True)
class Skill:
    id: str
    name: str
    weight: int
    keywords: List[str]
    prereqs: List[str]
    assessment: List[str]
    resources: List[Resource]
    category: str            # category name


@dataclass(frozen=True)
class SkillMap:
    role: str
    version: str
    categories: List[str]
    skills_by_id: Dict[str, Skill]


# -----------------------------
# Loading + validation
# -----------------------------

def load_skill_map(json_path: str | Path) -> SkillMap:
    """
    Load and lightly validate the skill map JSON.

    Expected schema (high-level):
      {
        "role": "...",
        "version": "...",
        "categories": [
          {
            "name": "Category",
            "skills": [
              {
                "id": "...",
                "name": "...",
                "weight": 1..5,
                "keywords": [...],
                "prereqs": [...],
                "assessment": [...],
                "resources": [
                  {"title": "...", "type": "...", "credibility": "...", "reason": "...", "url": "..."}
                ]
              }
            ]
          }
        ]
      }
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Skill map not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    role = _require_str(data, "role")
    version = _require_str(data, "version")

    raw_categories = data.get("categories")
    if not isinstance(raw_categories, list) or not raw_categories:
        raise ValueError("Skill map must contain non-empty 'categories' list.")

    categories: List[str] = []
    skills_by_id: Dict[str, Skill] = {}

    for cat in raw_categories:
        cat_name = _require_str(cat, "name")
        categories.append(cat_name)

        raw_skills = cat.get("skills")
        if not isinstance(raw_skills, list) or not raw_skills:
            raise ValueError(f"Category '{cat_name}' must contain non-empty 'skills' list.")

        for s in raw_skills:
            skill = _parse_skill(s, category=cat_name)

            if skill.id in skills_by_id:
                raise ValueError(f"Duplicate skill id '{skill.id}' found in skill map.")

            skills_by_id[skill.id] = skill

    # Validate prereqs reference existing skills (soft-check: raises if missing)
    _validate_prereqs(skills_by_id)

    return SkillMap(
        role=role,
        version=version,
        categories=categories,
        skills_by_id=skills_by_id,
    )


def list_skills(skill_map: SkillMap) -> List[Skill]:
    """Return skills sorted by category then by descending weight."""
    skills = list(skill_map.skills_by_id.values())
    skills.sort(key=lambda s: (s.category, -s.weight, s.name))
    return skills


def skills_in_category(skill_map: SkillMap, category_name: str) -> List[Skill]:
    """Return skills for a category sorted by descending weight."""
    out = [s for s in skill_map.skills_by_id.values() if s.category == category_name]
    out.sort(key=lambda s: (-s.weight, s.name))
    return out


def get_skill(skill_map: SkillMap, skill_id: str) -> Skill:
    """Get a skill by id, raising KeyError if not found."""
    return skill_map.skills_by_id[skill_id]


def top_weighted_skills(skill_map: SkillMap, n: int = 5) -> List[Skill]:
    """Return top-N skills by weight (ties broken by name)."""
    skills = list(skill_map.skills_by_id.values())
    skills.sort(key=lambda s: (-s.weight, s.name))
    return skills[: max(0, n)]


# -----------------------------
# Internal helpers
# -----------------------------

def _parse_skill(raw: dict, category: str) -> Skill:
    if not isinstance(raw, dict):
        raise ValueError("Each skill must be an object/dict.")

    sid = _require_str(raw, "id")
    name = _require_str(raw, "name")

    weight = raw.get("weight")
    if not isinstance(weight, int) or not (1 <= weight <= 5):
        raise ValueError(f"Skill '{sid}' weight must be an int in [1,5]. Got: {weight}")

    keywords = _require_list_str(raw, "keywords")
    prereqs = _require_list_str(raw, "prereqs")
    assessment = _require_list_str(raw, "assessment")

    resources_raw = raw.get("resources")
    if not isinstance(resources_raw, list):
        raise ValueError(f"Skill '{sid}' must contain 'resources' list.")

    resources: List[Resource] = []
    for r in resources_raw:
        if not isinstance(r, dict):
            raise ValueError(f"Skill '{sid}' has a resource that is not an object.")
        resources.append(
            Resource(
                title=_require_str(r, "title"),
                type=_require_str(r, "type"),
                credibility=_require_str(r, "credibility"),
                reason=_require_str(r, "reason"),
                url=_require_str(r, "url"),
            )
        )

    return Skill(
        id=sid,
        name=name,
        weight=weight,
        keywords=keywords,
        prereqs=prereqs,
        assessment=assessment,
        resources=resources,
        category=category,
    )


def _validate_prereqs(skills_by_id: Dict[str, Skill]) -> None:
    missing: List[Tuple[str, str]] = []
    for sid, skill in skills_by_id.items():
        for pre in skill.prereqs:
            if pre not in skills_by_id:
                missing.append((sid, pre))

    if missing:
        details = ", ".join([f"{sid} -> {pre}" for sid, pre in missing])
        raise ValueError(f"Prereq references missing skill ids: {details}")


def _require_str(obj: dict, key: str) -> str:
    val = obj.get(key)
    if not isinstance(val, str) or not val.strip():
        raise ValueError(f"Missing or invalid string field '{key}'.")
    return val.strip()


def _require_list_str(obj: dict, key: str) -> List[str]:
    val = obj.get(key)
    if val is None:
        return []
    if not isinstance(val, list):
        raise ValueError(f"Field '{key}' must be a list of strings.")
    out: List[str] = []
    for item in val:
        if not isinstance(item, str):
            raise ValueError(f"Field '{key}' must be a list of strings.")
        if item.strip():
            out.append(item.strip())
    return out
