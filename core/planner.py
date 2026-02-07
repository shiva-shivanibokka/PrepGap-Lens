
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Literal, Tuple
import math
import random

from core.skillmap import SkillMap, Skill, Resource
from core.score import ScoreReport, Gap


TaskType = Literal["learn", "practice", "explain", "review"]
Difficulty = Literal["easy", "ok", "hard"]
Status = Literal["done", "skipped"]


# -----------------------------
# Output models
# -----------------------------

@dataclass(frozen=True)
class PlanItem:
    day: int
    order: int
    skill_id: str
    skill_name: str
    category: str
    task_type: TaskType
    minutes: int
    resource_title: str
    resource_url: str
    resource_type: str
    credibility: str
    success_check: str


@dataclass(frozen=True)
class DayPlan:
    day: int
    total_minutes: int
    items: List[PlanItem]
    notes: List[str]


# -----------------------------
# Adaptation state + feedback
# -----------------------------

@dataclass(frozen=True)
class TaskFeedback:
    status: Status
    difficulty: Difficulty
    confidence: int = 3  # 1..5, optional


def init_mastery_from_report(report: ScoreReport) -> Dict[str, float]:
    """
    Initialize mastery from extracted coverage.
    mastery in [0,1]. You can persist this dict between days.
    """
    # copy to avoid mutation surprises
    return {sid: float(_clamp01(c)) for sid, c in report.coverage_by_skill.items()}


def apply_day_feedback(
    mastery_by_skill: Dict[str, float],
    feedback_by_skill: Dict[str, TaskFeedback],
    *,
    easy_delta: float = 0.12,
    ok_delta: float = 0.08,
    hard_delta: float = 0.04,
) -> Tuple[Dict[str, float], List[str], List[str]]:
    """
    Update mastery based on feedback for the day.

    Returns:
      new_mastery, reinforce_skills, retry_skills
    """
    new_mastery = dict(mastery_by_skill)
    reinforce: List[str] = []
    retry: List[str] = []

    for sid, fb in feedback_by_skill.items():
        m = float(_clamp01(new_mastery.get(sid, 0.05)))

        if fb.status == "skipped":
            retry.append(sid)
            continue

        # done
        if fb.difficulty == "easy":
            m = min(1.0, m + easy_delta)
        elif fb.difficulty == "ok":
            m = min(1.0, m + ok_delta)
        else:
            m = min(1.0, m + hard_delta)
            reinforce.append(sid)

        # small confidence adjustment
        # (kept gentle so it doesn't dominate)
        if fb.confidence <= 2:
            reinforce.append(sid)
        elif fb.confidence >= 5:
            m = min(1.0, m + 0.02)

        new_mastery[sid] = float(_clamp01(m))

    # dedupe preserve order
    reinforce = _dedupe(reinforce)
    retry = _dedupe(retry)
    return new_mastery, reinforce, retry


# -----------------------------
# Planner core
# -----------------------------

def generate_day_plan(
    *,
    day: int,
    report: ScoreReport,
    skill_map: SkillMap,
    mastery_by_skill: Dict[str, float],
    minutes_per_day: int = 120,
    reinforce_skills: Optional[List[str]] = None,
    retry_skills: Optional[List[str]] = None,
    seed: int = 7,
    top_gap_pool: int = 8,
) -> DayPlan:
    """
    Create a single-day plan that:
      - focuses on top gaps (impactful)
      - respects prereqs with small primers
      - balances learn/practice/explain/review
      - adapts based on reinforce/retry queues

    For hackathon MVP:
      - tasks are short (15-30 mins)
      - use assessment prompts as practice/explain
      - prefer interview-safe resources first
    """
    rng = random.Random(seed + day)

    minutes_per_day = int(max(30, minutes_per_day))
    reinforce_skills = reinforce_skills or []
    retry_skills = retry_skills or []

    notes: List[str] = []

    # Step 1: build a prioritized list of candidate skills
    gap_ranked = _rank_gaps_by_mastery(skill_map, mastery_by_skill)

    # Force include: retry first, then reinforce (short reinforcement), then top gaps
    candidate_ids: List[str] = []
    candidate_ids.extend(retry_skills)
    candidate_ids.extend(reinforce_skills)
    candidate_ids.extend([g.skill_id for g in gap_ranked[:top_gap_pool]])

    candidate_ids = _dedupe([sid for sid in candidate_ids if sid in skill_map.skills_by_id])

    if retry_skills:
        notes.append("You skipped some items yesterday — today starts with smaller retries.")
    if reinforce_skills:
        notes.append("Some topics felt hard — today includes quick reinforcement blocks.")

    # Step 2: pick 2 focus skills + 1 review skill (unless time is tiny)
    focus_ids = _pick_focus_skills(candidate_ids, mastery_by_skill, max_focus=2)
    review_id = _pick_review_skill(skill_map, mastery_by_skill, exclude=set(focus_ids))

    if review_id:
        chosen = focus_ids + [review_id]
    else:
        chosen = focus_ids

    if not chosen:
        # fallback: take any low mastery skill
        any_low = min(skill_map.skills_by_id.keys(), key=lambda sid: mastery_by_skill.get(sid, 0.05))
        chosen = [any_low]

    # Step 3: allocate time buckets (learn/practice/explain/review)
    # Default: ~45% learn, ~45% practice, ~10% explain/review
    learn_budget = int(round(minutes_per_day * 0.45))
    practice_budget = int(round(minutes_per_day * 0.45))
    explain_budget = minutes_per_day - learn_budget - practice_budget

    # Step 4: generate tasks
    items: List[PlanItem] = []
    order = 1
    used_minutes = 0

    # Helper to add a task
    def add_task(skill: Skill, task_type: TaskType, minutes: int, success_check: str) -> None:
        nonlocal order, used_minutes
        if minutes <= 0:
            return
        if used_minutes + minutes > minutes_per_day:
            minutes = max(0, minutes_per_day - used_minutes)
        if minutes <= 0:
            return

        res = choose_resource(skill, task_type=task_type, rng=rng)
        items.append(
            PlanItem(
                day=day,
                order=order,
                skill_id=skill.id,
                skill_name=skill.name,
                category=skill.category,
                task_type=task_type,
                minutes=int(minutes),
                resource_title=res.title,
                resource_url=res.url,
                resource_type=res.type,
                credibility=res.credibility,
                success_check=success_check,
            )
        )
        used_minutes += int(minutes)
        order += 1

    # 4a) Retry blocks first (short)
    for sid in retry_skills[:2]:
        if sid not in skill_map.skills_by_id:
            continue
        s = skill_map.skills_by_id[sid]
        add_task(
            s,
            "learn",
            minutes=15,
            success_check="Finish this short retry and write 3 bullet takeaways.",
        )

    # 4b) Reinforcement blocks (short practice)
    for sid in reinforce_skills[:2]:
        if sid not in skill_map.skills_by_id:
            continue
        s = skill_map.skills_by_id[sid]
        add_task(
            s,
            "practice",
            minutes=15,
            success_check=_practice_check(s),
        )

    # 4c) Main focus skills: prereq primer (if needed), then learn + practice + explain
    for sid in focus_ids:
        if sid not in skill_map.skills_by_id:
            continue
        s = skill_map.skills_by_id[sid]

        # prereq primer if any prereq mastery low
        missing_prereqs = [p for p in s.prereqs if mastery_by_skill.get(p, 0.05) < 0.55]
        if missing_prereqs:
            # pick first missing prereq and add a quick primer
            pre = skill_map.skills_by_id.get(missing_prereqs[0])
            if pre:
                add_task(
                    pre,
                    "review",
                    minutes=15,
                    success_check=f"Quick primer: explain {pre.name} in 2–3 sentences.",
                )
                notes.append(f"Added prerequisite primer before '{s.name}'.")

        # Learn block
        learn_minutes = 20 if minutes_per_day >= 90 else 15
        add_task(
            s,
            "learn",
            minutes=min(learn_minutes, learn_budget),
            success_check=_learn_check(s),
        )
        learn_budget -= min(learn_minutes, learn_budget)

        # Practice block
        practice_minutes = 25 if minutes_per_day >= 120 else 20
        add_task(
            s,
            "practice",
            minutes=min(practice_minutes, practice_budget),
            success_check=_practice_check(s),
        )
        practice_budget -= min(practice_minutes, practice_budget)

        # Explain block (Feynman)
        add_task(
            s,
            "explain",
            minutes=min(10, explain_budget),
            success_check=_explain_check(s),
        )
        explain_budget -= min(10, explain_budget)

    # 4d) Review skill (confidence + spacing)
    if review_id and review_id in skill_map.skills_by_id and used_minutes < minutes_per_day:
        s = skill_map.skills_by_id[review_id]
        add_task(
            s,
            "review",
            minutes=min(15, minutes_per_day - used_minutes),
            success_check="Review: write a tiny cheat-sheet (5 bullets).",
        )

    # 4e) If we still have time, add practice on the biggest remaining gap
    if used_minutes < minutes_per_day:
        remaining = minutes_per_day - used_minutes
        # pick next best gap not already used
        used_skills = {it.skill_id for it in items}
        for g in gap_ranked:
            if g.skill_id in used_skills:
                continue
            s = skill_map.skills_by_id[g.skill_id]
            add_task(
                s,
                "practice",
                minutes=min(remaining, 15),
                success_check=_practice_check(s),
            )
            break

    # Final notes
    if not items:
        notes.append("No tasks were generated — check your skill map and inputs.")
    else:
        notes.append("Tip: mark tasks as done + difficulty — tomorrow’s plan adapts automatically.")

    return DayPlan(day=day, total_minutes=min(minutes_per_day, used_minutes), items=items, notes=notes)


# -----------------------------
# Resource selection (credibility-aware)
# -----------------------------

def choose_resource(skill: Skill, task_type: TaskType, rng: random.Random) -> Resource:
    """
    Choose the best resource for a task.
    Preference order (MVP):
      - learn: youtube or docs, prefer interview-safe
      - practice: docs or anything; still prefer interview-safe
      - explain/review: any; still prefer interview-safe
    """
    resources = list(skill.resources) if skill.resources else []

    if not resources:
        # fallback stub
        return Resource(
            title=f"{skill.name} (no resource yet)",
            type="text",
            credibility="good-intuition",
            reason="Placeholder resource.",
            url="",
        )

    # Credibility ranking
    cred_rank = {"interview-safe": 0, "good-intuition": 1, "misleading": 2}
    def rank(res: Resource) -> Tuple[int, int]:
        # type preference by task
        if task_type == "learn":
            type_rank = 0 if res.type == "youtube" else 1 if res.type in ("docs", "blog") else 2
        elif task_type == "practice":
            type_rank = 0 if res.type in ("docs", "blog") else 1
        else:
            type_rank = 0
        return (cred_rank.get(res.credibility, 9), type_rank)

    resources.sort(key=rank)

    # Small randomness among top 2 to avoid same exact plan every time
    top = resources[:2] if len(resources) >= 2 else resources
    return rng.choice(top)


# -----------------------------
# Internal: gap ranking and picking
# -----------------------------

def _rank_gaps_by_mastery(skill_map: SkillMap, mastery_by_skill: Dict[str, float]) -> List[Gap]:
    """
    Recreate a gap list using mastery (post-feedback), independent from initial report.top_gaps.
    """
    gaps: List[Gap] = []
    for s in skill_map.skills_by_id.values():
        c = float(_clamp01(mastery_by_skill.get(s.id, 0.05)))
        g = 1.0 - c
        prereqs_missing = [p for p in s.prereqs if mastery_by_skill.get(p, 0.05) < 0.55]
        gaps.append(
            Gap(
                skill_id=s.id,
                skill_name=s.name,
                category=s.category,
                weight=s.weight,
                coverage=c,
                gap=g,
                prereqs_missing=prereqs_missing,
            )
        )

    # Priority: (gap * weight) descending, then weight, then gap
    gaps.sort(key=lambda x: (-(x.gap * x.weight), -x.weight, -x.gap, x.skill_name.lower()))
    return gaps


def _pick_focus_skills(candidate_ids: List[str], mastery_by_skill: Dict[str, float], max_focus: int = 2) -> List[str]:
    """
    Pick up to max_focus skills with lowest mastery among candidates,
    but avoid picking extremely high mastery items.
    """
    scored = []
    for sid in candidate_ids:
        m = float(mastery_by_skill.get(sid, 0.05))
        scored.append((m, sid))
    scored.sort(key=lambda x: x[0])  # lowest mastery first

    focus: List[str] = []
    for m, sid in scored:
        if len(focus) >= max_focus:
            break
        if m >= 0.80:
            continue
        focus.append(sid)
    return focus


def _pick_review_skill(skill_map: SkillMap, mastery_by_skill: Dict[str, float], exclude: set) -> Optional[str]:
    """
    Pick a medium-mastery skill (0.55..0.80) not in exclude.
    Helps confidence and spaced repetition.
    """
    candidates = []
    for sid in skill_map.skills_by_id.keys():
        if sid in exclude:
            continue
        m = float(mastery_by_skill.get(sid, 0.05))
        if 0.55 <= m <= 0.80:
            candidates.append((abs(m - 0.68), sid))  # prefer around 0.68
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


# -----------------------------
# Success checks (demo-friendly)
# -----------------------------

def _learn_check(skill: Skill) -> str:
    return f"Success: summarize {skill.name} in 3 bullets (focus on interview wording)."


def _practice_check(skill: Skill) -> str:
    if skill.assessment:
        q = skill.assessment[0]
        return f"Practice: answer: “{q}”"
    return "Practice: solve 2 quick questions and rate confidence."


def _explain_check(skill: Skill) -> str:
    return f"Explain: teach {skill.name} as if to a friend (5 sentences max)."


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x
