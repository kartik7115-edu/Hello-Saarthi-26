"""Sprint 6 tests: joining, data quality, grouping, determinism, CLI, leakage and compatibility."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from evidence_builders import (J, ROOT, annotation, answer_record, build, claim, hit, run_header, run_one)
from eval.answer_eval.schema import load_dataset
from eval.evidence_eval import run_evaluation as CLI
from eval.evidence_eval.annotations import load_annotations, parse_annotations
from eval.evidence_eval.evaluator import evaluate
from eval.evidence_eval.records import answer_sha256, load_answers, parse_answers
from eval.retrieval_eval.records import parse_records

FIX = Path(__file__).resolve().parents[1] / "fixtures"
FX_ANS, FX_ANN = FIX / "illustrative_answers.jsonl", FIX / "illustrative_annotations.jsonl"
PKG = Path(__file__).resolve().parents[1]
A = "Alpha holds [1]."
C = claim(A, "c1", "Alpha holds", judgements=[J("chunk", "k1", "supports")])


def dq(report):
    return {k: v for k, v in report["data_quality"].items() if v}


# ── Joining and data quality ────────────────────────────────────────────────────────────────

def test_stale_annotation_hash_is_a_data_problem_not_an_evaluation():
    answers, anns, _ = build([answer_record(A + " Edited.", [hit("d", 1, "k1")])], [annotation(A, [C])])
    rep = evaluate(anns, answers)
    assert rep["cases"] == []
    assert "answer_sha256 does not match" in rep["data_quality"]["inconsistent_annotations"][0]["problems"][0]


def test_span_text_and_question_mismatches_are_data_problems():
    bad = copy.deepcopy(C)
    bad["text"] = "Alpha holdz"
    answers, anns, errs = build([answer_record(A, [hit("d", 1, "k1")])], [annotation(A, [bad])])
    assert errs == []
    probs = evaluate(anns, answers)["data_quality"]["inconsistent_annotations"][0]["problems"]
    assert any("text differs from answer[0:11]" in p for p in probs)
    answers, anns, _ = build([answer_record(A, [hit("d", 1, "k1")], question="Other?")], [annotation(A, [C])])
    assert any("question differs" in p for p in evaluate(anns, answers)["data_quality"]["inconsistent_annotations"][0]["problems"])
    far = copy.deepcopy(C)
    far["span"] = {"start": 10, "end": 99}
    answers, anns, _ = build([answer_record(A, [hit("d", 1, "k1")])], [annotation(A, [far])])
    assert any("outside the answer" in p for p in evaluate(anns, answers)["data_quality"]["inconsistent_annotations"][0]["problems"])


def test_fixture_and_ground_truth_cannot_be_mixed():
    answers, anns, _ = build([answer_record(A, [hit("d", 1, "k1")])], [annotation(A, [C], status="fixture")], "recorded")
    assert "fixture annotations may only annotate fixture runs" in \
        evaluate(anns, answers)["data_quality"]["inconsistent_annotations"][0]["problems"]
    answers, anns, _ = build([answer_record(A, [hit("d", 1, "k1")])], [annotation(A, [C], status="verified")], "fixture")
    assert "cannot annotate a fixture run" in evaluate(anns, answers)["data_quality"]["inconsistent_annotations"][0]["problems"][0]


def test_annotation_without_answer_and_answer_without_annotation():
    answers, anns, _ = build([answer_record(A, [hit("d", 1, "k1")], case_id="q9")], [annotation(A, [C])])
    rep = evaluate(anns, answers)
    assert rep["data_quality"]["annotations_without_answer"] == [{"annotation_id": "ann-run-a-q1", "run_id": "run-a", "case_id": "q1"}]
    case = rep["cases"][0]
    assert case["group"] == "unannotated" and case["claims"] == []
    agg = rep["aggregates"]["run-a"]["unannotated"]["metrics"]
    assert set(agg) == {"citation_validity", "citation_resolution"}
    assert case["metrics"]["citation_precision"]["status"] == "not_evaluated"


def test_retrieval_context_sources():
    rs = parse_records([("r", {"record_type": "run", "run_id": "run-a", "provenance": "recorded"}),
                        ("r", {"record_type": "result", "run_id": "run-a", "case_id": "q1", "hits": [hit("d", 1, "k1")]})])
    answers, anns, _ = build([answer_record(A)], [annotation(A, [C])])
    rep = evaluate(anns, answers, rs)
    assert rep["cases"][0]["retrieval_source"] == "sprint5_results" and rep["cases"][0]["claims"][0]["outcome"] == "cited_evidence_supports"
    answers, anns, _ = build([answer_record(A, [hit("d", 1, "k1")])], [annotation(A, [C])])
    rep = evaluate(anns, answers, rs)
    assert rep["cases"] == [] and "both an embedded" in rep["data_quality"]["retrieval_context_problems"][0]["problem"]
    answers, anns, _ = build([answer_record(A)], [annotation(A, [C])])
    rep = evaluate(anns, answers)
    assert rep["data_quality"]["missing_retrieval_context"] == [{"run_id": "run-a", "case_id": "q1"}]
    case = rep["cases"][0]
    assert case["metrics"]["citation_resolution"]["status"] == "not_evaluated"
    # no context is a data gap, not an answer failure (Sprint 6 hardening F3)
    assert case["claims"][0]["outcome"] == "retrieval_context_missing"
    assert case["claims"][0]["coverage_satisfied"] is None
    assert case["metrics"]["citation_precision"]["status"] == "not_evaluated"
    cov = case["metrics"]["required_claim_citation_coverage"]
    assert (cov["status"], cov["numerator_lower"], cov["numerator_upper"], cov["denominator"]) == ("bounded", 0, 1, 1)
    assert rep["evaluation_findings"]["invalid_citation"] == []
    err = parse_records([("r", {"record_type": "run", "run_id": "run-a", "provenance": "recorded"}),
                         ("r", {"record_type": "result", "run_id": "run-a", "case_id": "q1", "error": "Timeout"})])
    rep = evaluate(anns, answers, err)
    assert "reported an error: Timeout" in rep["data_quality"]["retrieval_context_problems"][0]["problem"]


def test_sprint1_question_mismatch_is_flagged():
    items, _ = load_dataset(ROOT / "eval" / "answer_eval" / "dataset.jsonl")
    item = items[0]
    answers, anns, _ = build([answer_record(A, [hit("d", 1, "k1")], case_id=item.id, question="paraphrased?")],
                             [annotation(A, [C], case_id=item.id, question="paraphrased?")])
    rep = evaluate(anns, answers, None, items)
    assert rep["data_quality"]["dataset_question_mismatches"] == [{"run_id": "run-a", "case_id": item.id}]
    answers, anns, _ = build([answer_record(A, [hit("d", 1, "k1")], case_id=item.id, question=item.query)],
                             [annotation(A, [C], case_id=item.id, question=item.query)])
    assert evaluate(anns, answers, None, items)["data_quality"]["dataset_question_mismatches"] == []


# ── Grouping and aggregation ────────────────────────────────────────────────────────────────

def test_groups_are_never_pooled_and_aggregates_are_micro_sums():
    a2 = "Beta holds [1]."
    c2 = claim(a2, "c1", "Beta holds")
    answers, anns, _ = build([answer_record(A, [hit("d", 1, "k1")], case_id="q1"),
                              answer_record(a2, [hit("d", 1, "k1")], case_id="q2"),
                              answer_record("Gamma [1] and [Zed, p. 2].", [hit("d", 1, "k1")], case_id="q3")],
                             [annotation(A, [C], case_id="q1", status="verified"),
                              annotation(a2, [c2], case_id="q2", status="unverified")])
    rep = evaluate(anns, answers)
    agg = rep["aggregates"]["run-a"]
    assert set(agg) == {"verified", "unverified", "unannotated"}
    assert agg["verified"]["metrics"]["citation_precision"]["value"] == 1.0
    assert agg["unverified"]["metrics"]["citation_precision"]["status"] == "bounded"
    res = agg["unannotated"]["metrics"]["citation_resolution"]
    assert (res["numerator_lower"], res["denominator"]) == (1, 2)
    assert agg["verified"]["counts"]["outcomes"]["cited_evidence_supports"] == 1
    assert agg["unverified"]["counts"]["outcomes"]["cited_evidence_supports"] == 0


def test_findings_are_separate_from_data_quality():
    a = "P1 holds. P2 holds [7]. P3 holds [1]."
    _, case = run_one(a, [hit("d", 1, "k1")], [claim(a, "c1", "P1 holds"), claim(a, "c2", "P2 holds"),
                                               claim(a, "c3", "P3 holds", judgements=[J("chunk", "k1", "contradicts")])])
    report, _ = run_one(a, [hit("d", 1, "k1")], [claim(a, "c1", "P1 holds"), claim(a, "c2", "P2 holds"),
                                                 claim(a, "c3", "P3 holds", judgements=[J("chunk", "k1", "contradicts")])])
    f = report["evaluation_findings"]
    assert [x["claim_id"] for x in f["no_citation"]] == ["c1"]
    assert [x["claim_id"] for x in f["invalid_citation"]] == ["c2"]
    assert [x["claim_id"] for x in f["cited_evidence_contradicts"]] == ["c3"]
    assert [x["claim_id"] for x in f["human_verdict_contradicted"]] == ["c3"]
    assert dq(report) == {}


# ── Determinism, provenance labels, CLI ─────────────────────────────────────────────────────

def _rows(path):
    return [(f"{path.name}:{i}", json.loads(l)) for i, l in enumerate(path.read_text().splitlines(), 1) if l.strip()]


def test_report_is_deterministic_and_order_independent():
    first = json.dumps(evaluate(load_annotations(FX_ANN)[0], load_answers([FX_ANS])), sort_keys=True)
    for seed in range(3):
        ann_rows, ans_rows = _rows(FX_ANN), _rows(FX_ANS)
        random.Random(seed).shuffle(ann_rows)
        random.Random(seed + 7).shuffle(ans_rows)
        anns, errs = parse_annotations(ann_rows)
        again = evaluate(anns, parse_answers(ans_rows))
        assert errs == [] and json.dumps(again, sort_keys=True) == first


def test_fixture_report_is_labelled_illustrative_and_matches_hand_computed_values():
    rep = evaluate(load_annotations(FX_ANN)[0], load_answers([FX_ANS]))
    assert rep["evaluation"]["generates_answers"] is False and rep["evaluation"]["runs_retriever"] is False
    assert any("not Saarthi performance" in n for n in rep["evaluation"]["notes"])
    assert rep["runs"]["fixture-gen-a"]["illustrative"] is True and set(rep["aggregates"]["fixture-gen-a"]) == {"fixture"}
    m = rep["aggregates"]["fixture-gen-a"]["fixture"]["metrics"]
    assert (m["citation_precision"]["numerator_lower"], m["citation_precision"]["numerator_upper"],
            m["citation_precision"]["denominator"]) == (3, 4, 6)
    assert (m["required_claim_citation_coverage"]["numerator_lower"], m["required_claim_citation_coverage"]["denominator"]) == (6, 9)
    assert m["citation_resolution"]["value"] == pytest.approx(6 / 8)


def test_cli_writes_report_and_labels_fixtures(tmp_path, capsys):
    out = tmp_path / "r.json"
    assert CLI.main(["--answers", str(FX_ANS), "--annotations", str(FX_ANN), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["runs"]["fixture-gen-a"]["illustrative"] is True
    text = capsys.readouterr().out
    assert "ILLUSTRATIVE FIXTURE" in text and "no generation" in text and "Data-quality problems: none" in text


def test_cli_exit_codes(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"schema_version": "evidence-annotation/1"}\n', encoding="utf-8")
    assert CLI.main(["--answers", str(FX_ANS), "--annotations", str(bad)]) == 2
    assert CLI.main(["--answers", str(tmp_path / "missing.jsonl"), "--annotations", str(FX_ANN)]) == 1
    extra = tmp_path / "ans.jsonl"
    extra.write_text(FX_ANS.read_text() + '{"record_type": "answer", "oops": 1}\n', encoding="utf-8")
    out = tmp_path / "r.json"
    assert CLI.main(["--answers", str(extra), "--annotations", str(FX_ANN), "--out", str(out)]) == 0
    assert CLI.main(["--answers", str(extra), "--annotations", str(FX_ANN), "--out", str(out), "--strict"]) == 1
    assert CLI.main(["--answers", str(FX_ANS), "--annotations", str(FX_ANN), "--out", str(out), "--strict"]) == 0


def test_cli_runs_as_a_module_from_the_repo_root(tmp_path):
    out = tmp_path / "r.json"
    proc = subprocess.run([sys.executable, "-m", "eval.evidence_eval.run_evaluation", "--answers", str(FX_ANS),
                           "--annotations", str(FX_ANN), "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(out.read_text())["evaluation"]["schema"] == "evidence-eval-report/1"


# ── Leakage and compatibility ───────────────────────────────────────────────────────────────

def test_annotation_fields_cannot_travel_in_answer_records():
    for field in ("claims", "evidence_judgements", "support_status", "expected_evidence", "annotation"):
        rec = answer_record(A, [hit("d", 1, "k1")])
        rec[field] = []
        out = parse_answers([("h", run_header()), ("a", rec)])
        assert out.answers == {} and "annotations belong in the annotation file" in out.errors[0]


def test_evaluation_does_not_modify_its_inputs():
    anns, _ = load_annotations(FX_ANN)
    answers = load_answers([FX_ANS])
    before = repr((anns, answers.answers, answers.runs))     # inputs hold read-only mappings
    evaluate(anns, answers)
    assert repr((anns, answers.answers, answers.runs)) == before


def test_no_application_import_and_no_circular_dependency():
    for path in PKG.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module} | \
               {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert not any(m.startswith(("src", "saarthi_backend")) for m in mods), path.name
    for other in ("answer_eval", "retrieval_eval"):
        for path in (ROOT / "eval" / other).rglob("*.py"):
            assert "evidence_eval" not in path.read_text(encoding="utf-8"), path


def test_sprint1_dataset_and_sprint4_registry_are_untouched_by_a_run(tmp_path):
    files = [ROOT / "eval" / "answer_eval" / n for n in ("dataset.jsonl", "claim_rules.jsonl")]
    before = [hashlib.sha256(f.read_bytes()).hexdigest() for f in files]
    CLI.main(["--answers", str(FX_ANS), "--annotations", str(FX_ANN), "--out", str(tmp_path / "r.json")])
    assert [hashlib.sha256(f.read_bytes()).hexdigest() for f in files] == before


def test_answer_hash_is_over_the_exact_text():
    assert answer_sha256("a") != answer_sha256("a ") and len(answer_sha256("")) == 64
