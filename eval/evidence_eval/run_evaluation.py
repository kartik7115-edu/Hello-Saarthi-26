"""Evaluate recorded answers against claim/evidence annotations, offline (Sprint 6 CLI).

Run from the repository root:
    python -m eval.evidence_eval.run_evaluation --answers answers.jsonl --annotations annotations.jsonl \
        [--retrieval sprint5_results.jsonl] [--out report.json] [--strict]
    python -m eval.evidence_eval.run_evaluation --answers eval/evidence_eval/fixtures/illustrative_answers.jsonl \
        --annotations eval/evidence_eval/fixtures/illustrative_annotations.jsonl --out report.json

It reads recorded answers, their retrieval contexts and human annotations; it never generates an
answer, runs a retriever or calls a model. Questions are also checked against the Sprint 1 dataset
(eval/answer_eval/dataset.jsonl) when a case_id matches one of its items.

Exit codes: 0 report written; 1 an input file is unreadable, or --strict and there were data-quality
problems; 2 the annotation file has invalid records (no evaluation is run on invalid ground truth).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.answer_eval.schema import load_dataset  # noqa: E402
from eval.evidence_eval.annotations import load_annotations  # noqa: E402
from eval.evidence_eval.evaluator import evaluate  # noqa: E402
from eval.evidence_eval.records import load_answers  # noqa: E402
from eval.retrieval_eval.records import load_records  # noqa: E402

DATASET = ROOT / "eval" / "answer_eval" / "dataset.jsonl"


def _fmt(m: dict) -> str:
    st = m["status"]
    if st == "exact":
        return f"{m['value']:.3f} ({m['numerator_lower']}/{m['denominator']})"
    if st == "bounded":
        return f"[{m['lower']:.3f}, {m['upper']:.3f}] ({m['numerator_lower']}-{m['numerator_upper']}/{m['denominator']})"
    return st


def summary_lines(report: dict) -> list[str]:
    lines = ["Offline evaluation of recorded answers (no generation, no retrieval run)."]
    lines += [f"NOTE: {n}" for n in report["evaluation"]["notes"][1:]]
    for rid, groups in report["aggregates"].items():
        tag = "  ILLUSTRATIVE FIXTURE" if report["runs"][rid]["illustrative"] else ""
        lines.append(f"Run {rid}{tag}")
        for g, agg in groups.items():
            lines.append(f"  [{g}] {agg['cases']} answer(s), {agg['claims']} claim(s), "
                         f"{agg['evidence_required_claims']} needing evidence")
            for m, v in agg["metrics"].items():
                lines.append(f"    {m}: {_fmt(v)}")
            outs = {k: v for k, v in agg["counts"]["outcomes"].items() if v}
            if outs:
                lines.append(f"    claim outcomes: {outs}")
    dq = report["data_quality"]
    problems = {k: len(v) for k, v in dq.items() if v}
    lines.append(f"Data-quality problems: {problems or 'none'}")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--answers", type=Path, action="append", required=True, help="recorded answers JSONL")
    ap.add_argument("--annotations", type=Path, required=True, help="claim/evidence annotations JSONL")
    ap.add_argument("--retrieval", type=Path, action="append", default=[], help="Sprint 5 recorded results JSONL")
    ap.add_argument("--out", type=Path, help="write the JSON report here (default: stdout)")
    ap.add_argument("--strict", action="store_true", help="exit 1 if there were any data-quality problems")
    args = ap.parse_args(argv)
    try:
        annotations, ann_errors = load_annotations(args.annotations)
        answers = load_answers(args.answers)
        retrieval = load_records(args.retrieval) if args.retrieval else None
    except OSError as e:
        print(f"Cannot read input: {e}", file=sys.stderr)
        return 1
    if ann_errors:
        print(f"Annotations invalid ({len(ann_errors)} error(s)); not evaluating:", file=sys.stderr)
        for e in ann_errors:
            print(f"  - {e}", file=sys.stderr)
        return 2
    items, _ = load_dataset(DATASET)
    report = evaluate(annotations, answers, retrieval, items)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        for line in summary_lines(report):
            print(line)
        print(f"Report written to {args.out}")
    else:
        print(text)
    problems = any(report["data_quality"].values())
    return 1 if args.strict and problems else 0


if __name__ == "__main__":
    sys.exit(main())
