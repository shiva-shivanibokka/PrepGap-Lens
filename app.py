
from __future__ import annotations

from core.skillmap import load_skill_map
from core.extract import extract_skill_coverage, summarize_top_evidence
from core.score import score_readiness
from core.planner import (
    init_mastery_from_report,
    generate_day_plan,
    apply_day_feedback,
    TaskFeedback,
)
from core.credibility import summarize_resources_for_skill

SKILLMAP_PATH = "data/skill_map_ml_intern.json"


def print_header(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def demo_resume_text() -> str:
    """
    Replace this string with your own resume/skills paste any time.
    Keep it short-ish so the terminal output stays readable.
    """
    return """
    Graduate student in Data Science & AI. Built ML pipelines in Python using pandas and scikit-learn.
    Worked on classification models; evaluated with precision, recall, F1, ROC-AUC.
    Used train-test split and KFold cross-validation. Implemented preprocessing with sklearn Pipeline.
    Experience with feature engineering: one-hot encoding, scaling, imputation.
    Built Random Forest and XGBoost models for tabular prediction.
    Deployed a simple model inference API using FastAPI.
    """


def pretty_print_plan(day_plan) -> None:
    print(f"\nDay {day_plan.day} Plan — Total: {day_plan.total_minutes} minutes")
    for item in day_plan.items:
        print(
            f"{item.order}. [{item.task_type.upper():7}] {item.skill_name} "
            f"({item.minutes}m) | {item.credibility} | {item.resource_title}"
        )
        if item.resource_url:
            print(f"    link:  {item.resource_url}")
        print(f"    check: {item.success_check}")

    if day_plan.notes:
        print("\nNotes:")
        for n in day_plan.notes:
            print(f"- {n}")


def main() -> None:
    print_header("PrepGap-Lens — Demo Runner (ML Engineer Intern)")

    # 1) Load skill map
    skill_map = load_skill_map(SKILLMAP_PATH)
    print(f"Loaded skill map for role: {skill_map.role} (v{skill_map.version})")
    print(f"Skills: {len(skill_map.skills_by_id)} | Categories: {len(skill_map.categories)}")

    # 2) Extract skill coverage from user text
    user_text = demo_resume_text()
    extraction = extract_skill_coverage(user_text, skill_map)

    print_header("Extraction Evidence (what matched in your text)")
    lines = summarize_top_evidence(extraction, skill_map, min_confidence=0.50, top_k=12)
    if not lines:
        print("No strong matches found. Try pasting more resume/project text.")
    else:
        for line in lines:
            print("-", line)

    # 3) Score readiness + top gaps
    report = score_readiness(skill_map, extraction, top_k_gaps=6)

    print_header("Readiness Scores")
    print(f"Overall readiness: {report.readiness_overall * 100:.1f}%\n")
    print("By category:")
    for cat, val in report.readiness_by_category.items():
        print(f"- {cat}: {val * 100:.1f}%")

    print_header("Top Gaps (fastest improvements)")
    for g in report.top_gaps:
        prereq_note = f" | prereqs missing: {g.prereqs_missing}" if g.prereqs_missing else ""
        print(
            f"- {g.skill_name} [{g.category}] "
            f"weight={g.weight} coverage={g.coverage:.2f} gap={g.gap:.2f}{prereq_note}"
        )

    # 4) Show credible resources for top 2 gaps
    print_header("Credible Resources for Top Gaps")
    for g in report.top_gaps[:2]:
        s = skill_map.skills_by_id[g.skill_id]
        print(f"\n{s.name}:")
        for line in summarize_resources_for_skill(s):
            print(line)

    # 5) Generate Day 1 plan
    mastery = init_mastery_from_report(report)

    print_header("Day 1 Plan (generated)")
    day1 = generate_day_plan(
        day=1,
        report=report,
        skill_map=skill_map,
        mastery_by_skill=mastery,
        minutes_per_day=120,
    )
    pretty_print_plan(day1)

    # 6) Simulate feedback → Day 2 adaptation
    # In a real UI, you'll collect this via checkboxes + difficulty dropdowns.
    print_header("Simulated Feedback → Day 2 Plan (adapts)")
    feedback = {}

    # mark first item as done-hard, second item as done-ok, third item skipped (if they exist)
    if len(day1.items) >= 1:
        feedback[day1.items[0].skill_id] = TaskFeedback(status="done", difficulty="hard", confidence=2)
    if len(day1.items) >= 2:
        feedback[day1.items[1].skill_id] = TaskFeedback(status="done", difficulty="ok", confidence=4)
    if len(day1.items) >= 3:
        feedback[day1.items[2].skill_id] = TaskFeedback(status="skipped", difficulty="ok", confidence=3)

    new_mastery, reinforce, retry = apply_day_feedback(mastery, feedback)

    day2 = generate_day_plan(
        day=2,
        report=report,
        skill_map=skill_map,
        mastery_by_skill=new_mastery,
        minutes_per_day=120,
        reinforce_skills=reinforce,
        retry_skills=retry,
    )
    pretty_print_plan(day2)

    print("\nQueues (why it adapted):")
    print("Reinforce:", reinforce)
    print("Retry:", retry)

    print("\nDone. Next step: connect this to a minimal UI (Streamlit or notebook widgets).")


if __name__ == "__main__":
    main()
