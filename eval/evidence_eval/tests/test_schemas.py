"""Sprint 6 tests: annotation schema, provenance and the recorded-answer format."""
from __future__ import annotations

import copy
import datetime as dt
import json

import pytest

from evidence_builders import J, TODAY, annotation, answer_record, claim, run_header
from eval.evidence_eval.annotations import load_annotations, parse_annotation
from eval.evidence_eval.records import answer_sha256, load_answers, parse_answers

ANS = "Sampling twice the bandwidth avoids aliasing [1]."
C1 = claim(ANS, "c1", "Sampling twice the bandwidth avoids aliasing", judgements=[J("chunk", "k1", "supports")])


def parse(raw):
    return parse_annotation(raw, "t", today=TODAY)


def test_valid_annotation_parses():
    ann, errs = parse(annotation(ANS, [C1]))
    assert errs == [] and ann.claims[0].relation(next(iter(ann.claims[0].judgements))) == "supports"
    assert ann.provenance.status == "verified" and ann.provenance.version == 1


@pytest.mark.parametrize("mutate,needle", [
    (lambda r: r.update(schema_version="evidence-annotation/2"), "schema_version must be"),
    (lambda r: r.update(extra=1), "unknown field"),
    (lambda r: r.pop("claims"), "missing field"),
    (lambda r: r.update(language="hi"), "language must be one of"),
    (lambda r: r.update(language="mr"), "language must be one of"),
    (lambda r: r.update(answer_sha256="abc"), "answer_sha256 must be 64"),
    (lambda r: r.update(annotation_id="bad id"), "annotation_id must match"),
    (lambda r: r.update(question=" "), "question must be"),
    (lambda r: r.update(metadata=[]), "metadata must be an object"),
    (lambda r: r["annotation"].update(status="approved"), "annotation.status must be"),
    (lambda r: r["annotation"].update(version=0), "version must be an integer >= 1"),
    (lambda r: r["annotation"].update(version=True), "version must be an integer >= 1"),
    (lambda r: r["annotation"].update(colour="x"), "unknown field"),
    (lambda r: r["annotation"].pop("reviewer"), "need a reviewer"),
    (lambda r: r["annotation"].update(reviewer="annotator-1"), "must not be its annotator"),
    (lambda r: r["annotation"].update(reviewed_on="20-11-2026"), "ISO date"),
    (lambda r: r["annotation"].update(reviewed_on=(TODAY + dt.timedelta(days=2)).isoformat()), "in the future"),
    (lambda r: r["claims"].append(copy.deepcopy(r["claims"][0])), "duplicate claim_id"),
    (lambda r: r["claims"][0].update(claim_id="bad id"), "claim_id must match"),
    (lambda r: r["claims"][0].update(span={"start": 5}), "span must be exactly"),
    (lambda r: r["claims"][0].update(span={"start": 5, "end": 5}), "0 <= start < end"),
    (lambda r: r["claims"][0].update(requires_evidence="yes"), "requires_evidence must be"),
    (lambda r: r["claims"][0].update(support_status="true"), "support_status must be one of"),
    (lambda r: r["claims"][0].update(support_status="contradicted"), "needs at least one 'contradicts'"),
    (lambda r: r["claims"][0].update(evidence_judgements=[J("chunk", "k1", "maybe")], support_status="not_assessed"), "relation must be one of"),
    (lambda r: r["claims"][0].update(evidence_judgements=[J("chunk", "k1", "supports"), J("chunk", "k1", "irrelevant")]), "judged more than once"),
    (lambda r: r["claims"][0].update(evidence_judgements=[J("section", "k1", "supports")]), "level must be one of"),
    (lambda r: r["claims"][0].update(evidence_judgements=[J("page", "doc page 3", "supports")]), "must look like '<document id>#p<page>'"),
    (lambda r: r["claims"][0].update(evidence_judgements=[{"evidence": {"level": "chunk", "id": "k1", "x": 1}, "relation": "supports"}]), "exactly {\"level\", \"id\"}"),
    (lambda r: r["claims"][0].update(expected_evidence=[{"level": "chunk", "id": "k1"}] * 2), "lists chunk 'k1' twice"),
    (lambda r: r["claims"][0].update(citation_starts=[1, 1]), "lists an offset twice"),
    (lambda r: r["claims"][0].update(citation_starts=[-1]), "character offsets >= 0"),
    (lambda r: r["claims"][0].update(confidence=0.9), "unknown field"),
])
def test_invalid_annotations_are_rejected(mutate, needle):
    raw = annotation(ANS, [C1])
    mutate(raw)
    ann, errs = parse(raw)
    assert ann is None and any(needle in e for e in errs), errs


def test_unverified_and_fixture_provenance_rules():
    raw = annotation(ANS, [C1], status="unverified")
    assert parse(raw)[1] == []
    raw["annotation"]["reviewer"] = "reviewer-2"
    assert any("only allowed on verified" in e for e in parse(raw)[1])
    assert parse(annotation(ANS, [C1], status="fixture"))[1] == []


def test_overlapping_claims_and_shared_citation_offsets_are_rejected():
    a = "Alpha beta gamma [1]."
    c1 = claim(a, "c1", "Alpha beta")
    c2 = claim(a, "c2", "beta gamma")
    assert any("overlapping spans" in e for e in parse(annotation(a, [c1, c2]))[1])
    c3 = claim(a, "c3", "Alpha", citation_starts=[17])
    c4 = claim(a, "c4", "gamma", citation_starts=[17])
    assert any("more than one claim" in e for e in parse(annotation(a, [c3, c4]))[1])


def test_duplicate_annotation_ids_and_pairs(tmp_path):
    a = annotation(ANS, [C1])
    b = copy.deepcopy(a)
    b["case_id"] = "q2"
    c = copy.deepcopy(a)
    c["annotation_id"] = "ann-other"
    p = tmp_path / "ann.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in (a, b, c)) + "\n{bad\n", encoding="utf-8")
    anns, errs = load_annotations(p, today=TODAY)
    assert len(anns) == 1
    assert any("duplicate annotation_id" in e for e in errs)
    assert any("a second annotation for run" in e for e in errs)
    assert any("invalid JSON" in e for e in errs)


# ── Recorded answers ────────────────────────────────────────────────────────────────────────

def answers(*rows, provenance="recorded"):
    return parse_answers([(f"l{i}", r) for i, r in enumerate((run_header(provenance=provenance),) + rows, 1)])


def test_valid_answer_with_embedded_sprint5_retrieval():
    out = answers(answer_record(ANS, [{"document_id": "d1", "page": 2, "chunk_id": "k1", "score": 0.5}]))
    assert out.errors == []
    a = out.answers[("run-a", "q1")]
    assert a.retrieval.ranking("chunk") == ("k1",) and a.sha256 == answer_sha256(ANS)


def test_empty_answer_is_allowed():
    out = answers(answer_record("", []))
    assert out.errors == [] and out.answers[("run-a", "q1")].answer == ""


@pytest.mark.parametrize("over,needle", [
    ({"claims": []}, "annotations belong in the annotation file"),
    ({"support_status": "supported"}, "annotations belong in the annotation file"),
    ({"evidence_judgements": []}, "annotations belong in the annotation file"),
    ({"citation_numbering": "guess"}, "citation_numbering must be one of"),
    ({"answer": None}, "answer must be a string"),
    ({"question": ""}, "question must be a non-empty string"),
    ({"run_id": "missing-run"}, "has no run header"),
    ({"retrieval": {"hits": [{"page": 2}]}}, "needs document_id"),
    ({"retrieval": {"hits": [{"document_id": "d", "score": "high"}]}}, "score must be a finite number"),
    ({"retrieval": {"hits": [], "document_ids": []}}, "not both"),
    ({"retrieval": {"ranking": []}}, "retrieval has unknown field"),
    ({"retrieval": []}, "retrieval must be an object"),
])
def test_malformed_answer_records_are_reported_and_skipped(over, needle):
    rec = answer_record(ANS, [])
    rec.update(over)
    out = answers(rec)
    assert any(needle in e for e in out.errors), out.errors
    assert out.answers == {}


def test_duplicate_answers_and_headers_and_unknown_types(tmp_path):
    p = tmp_path / "a.jsonl"
    rows = [run_header(), answer_record(ANS, []), answer_record("other", []), run_header(), {"record_type": "claim"}]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n{oops\n", encoding="utf-8")
    out = load_answers([p])
    assert out.answers[("run-a", "q1")].answer == ANS
    for needle in ("duplicate answer", "duplicate run header", "record_type must be", "invalid JSON"):
        assert any(needle in e for e in out.errors), needle


def test_fixture_files_are_valid_and_marked():
    from pathlib import Path
    fx = Path(__file__).resolve().parents[1] / "fixtures"
    anns, errs = load_annotations(fx / "illustrative_annotations.jsonl", today=TODAY)
    ans = load_answers([fx / "illustrative_answers.jsonl"])
    assert errs == [] and ans.errors == [] and len(anns) == 3
    assert all(a.provenance.status == "fixture" for a in anns)
    assert all(r.provenance == "fixture" and "ILLUSTRATIVE" in r.description for r in ans.runs.values())
    assert all(a.case_id.startswith("fx-") for a in anns)
