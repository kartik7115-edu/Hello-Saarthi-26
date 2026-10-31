"""Grade eval/results/runs.jsonl with an LLM judge and write a metrics report.

For every run the judge checks, against the dataset item:
  * whether the answer abstained (declined / said the materials lack it)
  * which key_points are present and which forbidden_claims appear
  * each factual claim in the answer: supported by the retrieved documents or not

Grades are cached in eval/results/grades.jsonl, so re-running only grades new
runs. The report is printed and saved to eval/results/report.md.

Usage (from the repo root):
  python eval/score_eval.py
  python eval/score_eval.py --judge-model gpt-4.1 --env-file C:\\Projects\\Hello-Saarthi-main\\.env
  python eval/score_eval.py --report-only      # skip grading, rebuild the report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import List

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"
DATASET = EVAL_DIR / "eval_dataset.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
RUNS_FILE = RESULTS_DIR / "runs.jsonl"
GRADES_FILE = RESULTS_DIR / "grades.jsonl"
REPORT_FILE = RESULTS_DIR / "report.md"

MAX_CONTEXT_CHARS = 15000


# --------------------------------------------------------------------------
# Judge
# --------------------------------------------------------------------------

class PointCheck(BaseModel):
    index: int = Field(description="0-based index of the item in the list given")
    present: bool = Field(description="True only if the answer clearly states this")


class ClaimCheck(BaseModel):
    claim: str = Field(description="One atomic factual claim made in the answer")
    supported: bool = Field(description="True only if the retrieved documents support this claim")


class Judgement(BaseModel):
    abstained: bool = Field(description="True if the answer declines or says the information is not available")
    key_points: List[PointCheck]
    forbidden_claims: List[PointCheck]
    claims: List[ClaimCheck]
    rationale: str = Field(description="One or two sentences explaining the grade")


JUDGE_SYSTEM = """You are a strict grader for a Signals & Systems / DSP tutoring assistant.
You receive a question, the assistant's answer, the documents the assistant retrieved,
and grading criteria. Grade only what the answer actually says.

1. abstained: true if the answer declines, says it cannot answer, or says the materials
   do not contain the information, even if it adds some general remarks.
2. key_points: for EACH listed key point (by index), present=true only if the answer
   clearly states it. Equivalent notation or wording counts. Vague gestures do not.
3. forbidden_claims: for EACH listed forbidden claim (by index), present=true if the
   answer asserts it (not if it mentions it only to refute it).
4. claims: split the answer into its atomic factual claims (at most 15, most important
   first). supported=true only if the RETRIEVED DOCUMENTS state or directly imply it.
   General knowledge that is correct but absent from the documents is NOT supported.
   If the answer abstained, return an empty list.
Return every key point and every forbidden claim exactly once."""


def build_judge_prompt(item: dict, run: dict) -> str:
    resp = run["response"] or {}
    context_parts = []
    for r in run.get("retrievals", []):
        for d in r["docs"]:
            context_parts.append(f"[{d.get('source') or 'unknown'}]\n{d.get('content', '')}")
    context = "\n\n---\n\n".join(context_parts) or "(no documents were retrieved)"
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n...[truncated]"

    kp = "\n".join(f"{i}. {p}" for i, p in enumerate(item["key_points"])) or "(none)"
    fc = "\n".join(f"{i}. {c}" for i, c in enumerate(item["forbidden_claims"])) or "(none)"
    return (
        f"QUESTION:\n{item['query']}\n\n"
        f"EXPECTED BEHAVIOUR: {item['expected_behavior']}\n\n"
        f"KEY POINTS:\n{kp}\n\nFORBIDDEN CLAIMS:\n{fc}\n\n"
        f"RETRIEVED DOCUMENTS:\n{context}\n\n"
        f"ASSISTANT ANSWER:\n{resp.get('content', '')}"
    )


def run_hash(run: dict) -> str:
    # Same item + same answer + same evidence => same grade (shared across variants to save cost).
    payload = json.dumps([run["id"], run.get("response"), run.get("retrievals")], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def grade_all(items: dict, runs: list, judge_model: str) -> dict:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    grades = {}
    if GRADES_FILE.exists():
        for line in GRADES_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                g = json.loads(line)
                grades[g["hash"]] = g

    judge = ChatOpenAI(model=judge_model, temperature=0).with_structured_output(
        Judgement, method="json_schema", strict=True
    )
    todo, seen = [], set(grades)
    for r in runs:
        h = run_hash(r)
        if r["error"] is None and h not in seen:
            seen.add(h)
            todo.append(r)
    print(f"Grading {len(todo)} new runs with {judge_model} ({len(grades)} cached)")

    with GRADES_FILE.open("a", encoding="utf-8") as out:
        for n, run in enumerate(todo, 1):
            item = items[run["id"]]
            try:
                j: Judgement = judge.invoke([
                    SystemMessage(content=JUDGE_SYSTEM),
                    HumanMessage(content=build_judge_prompt(item, run)),
                ])
                g = {"hash": run_hash(run), "judge_model": judge_model, **j.model_dump()}
            except Exception as e:
                print(f"  [{n}/{len(todo)}] {run['id']} judge failed: {e}")
                continue
            grades[g["hash"]] = g
            out.write(json.dumps(g, ensure_ascii=False) + "\n")
            out.flush()
            print(f"  [{n}/{len(todo)}] {run['variant']:8} {run['mode']:8} {run['id']}  abstained={g['abstained']}")
    return grades


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def checks_by_index(checks: list, n: int) -> list:
    """Map judge checks back onto the item's list; anything missing counts as False."""
    out = [False] * n
    for c in checks:
        if 0 <= c["index"] < n:
            out[c["index"]] = c["present"]
    return out


def score_run(item: dict, run: dict, grade: dict | None) -> dict:
    s = {"id": item["id"], "expert": item["expert"], "category": item["category"],
         "expected": item["expected_behavior"], "error": run["error"] is not None}
    if run["error"] is not None or grade is None:
        s["graded"] = False
        return s

    resp = run["response"]
    flag_abstain = not resp["is_knowledge_present"]
    abstained = flag_abstain or grade["abstained"]
    kp = checks_by_index(grade["key_points"], len(item["key_points"]))
    fc = checks_by_index(grade["forbidden_claims"], len(item["forbidden_claims"]))
    claims = grade["claims"]

    s.update({
        "graded": True,
        "abstained": abstained,
        "flag_mismatch": flag_abstain != grade["abstained"],
        "confidence": resp["confidence_score"],
        "kp_recall": (sum(kp) / len(kp)) if kp else None,
        "forbidden_hit": any(fc),
        "groundedness": (sum(c["supported"] for c in claims) / len(claims)) if (claims and not abstained) else None,
        "retrieved_sources": sorted({d["source"] for r in run.get("retrievals", []) for d in r["docs"] if d["source"]}),
        "latency_s": run["latency_s"],
    })
    if item["expected_behavior"] == "answer":
        s["correct"] = (not abstained) and all(kp) and not any(fc)
    else:
        s["correct"] = abstained and not any(fc)
    return s


def pct(values: list) -> str:
    return f"{100 * mean(values):.0f}%" if values else "n/a"


def summarise(scores: list) -> dict:
    g = [s for s in scores if s.get("graded")]
    ans = [s for s in g if s["expected"] == "answer"]
    abst = [s for s in g if s["expected"] == "abstain"]
    answered = [s for s in ans if not s["abstained"]]
    conf_ok = [s["confidence"] for s in answered if s["correct"]]
    conf_bad = [s["confidence"] for s in answered if not s["correct"]]
    return {
        "n": len(scores),
        "errors": sum(s["error"] for s in scores),
        "accuracy": pct([s["correct"] for s in ans]),
        "kp_recall": pct([s["kp_recall"] for s in ans if s["kp_recall"] is not None]),
        "groundedness": pct([s["groundedness"] for s in g if s.get("groundedness") is not None]),
        "correct_abstention": pct([s["abstained"] for s in abst]),
        "false_abstention": pct([s["abstained"] for s in ans]),
        "forbidden_rate": pct([s["forbidden_hit"] for s in g]),
        "conf_correct": f"{mean(conf_ok):.2f}" if conf_ok else "n/a",
        "conf_incorrect": f"{mean(conf_bad):.2f}" if conf_bad else "n/a",
        "flag_mismatch": sum(s["flag_mismatch"] for s in g),
        "latency": f"{mean(s['latency_s'] for s in g):.1f}s" if g else "n/a",
    }


COLUMNS = [
    ("accuracy", "Answer accuracy"),
    ("kp_recall", "Key-point recall"),
    ("groundedness", "Groundedness"),
    ("correct_abstention", "Correct abstention"),
    ("false_abstention", "False abstention"),
    ("forbidden_rate", "Forbidden-claim rate"),
    ("conf_correct", "Conf. (correct)"),
    ("conf_incorrect", "Conf. (incorrect)"),
    ("flag_mismatch", "Flag mismatches"),
    ("latency", "Mean latency"),
    ("errors", "Run errors"),
]


def table(rows: dict) -> str:
    keys = list(rows)
    lines = ["| Metric | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for col, label in COLUMNS:
        lines.append(f"| {label} | " + " | ".join(str(rows[k][col]) for k in keys) + " |")
    return "\n".join(lines)


def build_report(items: dict, runs: list, grades: dict) -> str:
    groups = defaultdict(list)
    per_item = defaultdict(dict)
    for run in runs:
        s = score_run(items[run["id"]], run, grades.get(run_hash(run)))
        groups[(run["variant"], run["mode"])].append(s)
        per_item[run["id"]][f"{run['variant']}/{run['mode']}"] = s

    order = [k for k in [("baseline", "rag"), ("branch", "rag"), ("baseline", "planning"), ("branch", "planning")]
             if k in groups]
    out = ["# Response-Quality Evaluation Report", ""]
    out.append(f"Dataset: {len(items)} items. Runs: {len(runs)}. Judge-graded: {sum(1 for r in runs if run_hash(r) in grades)}.")
    out.append("")
    out.append("## Overall")
    out.append(table({f"{v}/{m}": summarise(groups[(v, m)]) for v, m in order}))
    out.append("")

    for expert in sorted({r["expert"] for r in runs}):
        out.append(f"## {expert}")
        out.append(table({f"{v}/{m}": summarise([s for s in groups[(v, m)] if s["expert"] == expert]) for v, m in order}))
        out.append("")

    out.append("## Per item")
    cols = [f"{v}/{m}" for v, m in order]
    out.append("| id | category | expected | " + " | ".join(cols) + " |")
    out.append("|---|---|---|" + "---|" * len(cols))
    for iid in sorted(per_item):
        item = items[iid]
        cells = []
        for c in cols:
            s = per_item[iid].get(c)
            if s is None:
                cells.append("")
            elif s["error"]:
                cells.append("ERROR")
            elif not s.get("graded"):
                cells.append("ungraded")
            else:
                mark = "PASS" if s["correct"] else "FAIL"
                cells.append(f"{mark}{' (abstained)' if s['abstained'] else ''}")
        out.append(f"| {iid} | {item['category']} | {item['expected_behavior']} | " + " | ".join(cells) + " |")
    out.append("")

    out.append("## Retrieved sources (use these to fill `expected_sources`)")
    out.append("| id | sources retrieved in any branch run |")
    out.append("|---|---|")
    for iid in sorted(per_item):
        srcs = sorted({src for k, s in per_item[iid].items() if k.startswith("branch") for src in s.get("retrieved_sources", [])})
        out.append(f"| {iid} | {', '.join(srcs) or '-'} |")
    out.append("")

    out.append("## Notes")
    out.append("- *Flag mismatches*: runs where `is_knowledge_present` disagrees with whether the text actually abstained.")
    out.append("- Groundedness counts only answers that did not abstain. Correct general knowledge absent from the retrieved documents counts as unsupported.")
    out.append("- `key_points` are not yet verified against the KB (see eval/README.md).")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--judge-model", default="gpt-4.1")
    p.add_argument("--env-file", help="Path to .env (default: <repo>/.env)")
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args()

    if not RUNS_FILE.exists():
        sys.exit("No runs found. Run: python eval/run_eval.py")

    items = {i["id"]: i for i in (json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip())}
    runs = {}
    for line in RUNS_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r["id"] in items:
                runs[(r["variant"], r["mode"], r["id"])] = r  # latest run wins
    runs = list(runs.values())

    if args.report_only:
        grades = {}
        if GRADES_FILE.exists():
            for line in GRADES_FILE.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    g = json.loads(line)
                    grades[g["hash"]] = g
    else:
        from dotenv import load_dotenv
        env_path = Path(args.env_file) if args.env_file else ROOT / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("ERROR: OPENAI_API_KEY is not set. Pass --env-file.")
        grades = grade_all(items, runs, args.judge_model)

    report = build_report(items, runs, grades)
    REPORT_FILE.write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\nSaved to {REPORT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
