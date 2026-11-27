"""Test builders for Sprint 6 (spans and hashes are computed, never typed by hand)."""
from __future__ import annotations

import copy
import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.evidence_eval.annotations import parse_annotations  # noqa: E402
from eval.evidence_eval.evaluator import evaluate  # noqa: E402
from eval.evidence_eval.records import answer_sha256, parse_answers  # noqa: E402

TODAY = dt.date.today()
REVIEWED = (TODAY - dt.timedelta(days=3)).isoformat()
RUN = "run-a"


def hit(doc, page=None, chunk=None, **md):
    h = {"document_id": doc}
    if page is not None:
        h["page"] = page
    if chunk is not None:
        h["chunk_id"] = chunk
    if md:
        h["metadata"] = md
    return h


def J(level, id_, relation):
    return {"evidence": {"level": level, "id": id_}, "relation": relation}


def claim(answer, cid, text, *, requires=True, judgements=(), status=None, occurrence=0, **extra):
    start = -1
    for _ in range(occurrence + 1):
        start = answer.index(text, start + 1)
    if status is None:
        rels = [j["relation"] for j in judgements]
        status = "supported" if "supports" in rels else "contradicted" if "contradicts" in rels else "not_assessed"
    out = {"claim_id": cid, "text": text, "span": {"start": start, "end": start + len(text)},
           "requires_evidence": requires, "evidence_judgements": list(judgements), "support_status": status}
    out.update(extra)
    return out


def provenance(status="verified"):
    if status == "verified":
        return {"status": "verified", "version": 1, "annotator": "annotator-1", "reviewer": "reviewer-2",
                "reviewed_on": REVIEWED}
    return {"status": status, "version": 1, "annotator": "annotator-1"}


def annotation(answer, claims, *, case_id="q1", run_id=RUN, status="verified", question="What is it?", **over):
    raw = {"schema_version": "evidence-annotation/1", "annotation_id": f"ann-{run_id}-{case_id}", "run_id": run_id,
           "case_id": case_id, "question": question, "language": "en", "answer_sha256": answer_sha256(answer),
           "annotation": provenance(status), "claims": copy.deepcopy(list(claims))}
    raw.update(over)
    return raw


def answer_record(answer, hits=None, *, case_id="q1", run_id=RUN, numbering="retrieval_order",
                  question="What is it?", **over):
    raw = {"record_type": "answer", "run_id": run_id, "case_id": case_id, "question": question, "answer": answer,
           "citation_numbering": numbering}
    if hits is not None:
        raw["retrieval"] = {"hits": list(hits)}
    raw.update(over)
    return raw


def run_header(run_id=RUN, provenance="recorded"):
    return {"record_type": "run", "run_id": run_id, "provenance": provenance}


def build(answer_rows, annotation_rows, run_provenance="recorded"):
    runs = sorted({r["run_id"] for r in answer_rows})
    answers = parse_answers([(f"a{i}", r) for i, r in enumerate([run_header(x, run_provenance) for x in runs]
                                                                 + list(answer_rows), 1)])
    anns, errs = parse_annotations([(f"n{i}", copy.deepcopy(r)) for i, r in enumerate(annotation_rows, 1)])
    return answers, anns, errs


def run_one(answer, hits, claims, *, numbering="retrieval_order", status="verified", run_provenance=None, **ann_over):
    run_provenance = run_provenance or ("fixture" if status == "fixture" else "recorded")
    answers, anns, errs = build([answer_record(answer, hits, numbering=numbering)],
                                [annotation(answer, claims, status=status, **ann_over)], run_provenance)
    assert errs == [] and answers.errors == [], (errs, answers.errors)
    report = evaluate(anns, answers)
    assert len(report["cases"]) == 1, report["data_quality"]
    return report, report["cases"][0]


def outcomes(case):
    return {c["claim_id"]: c["outcome"] for c in case["claims"]}
