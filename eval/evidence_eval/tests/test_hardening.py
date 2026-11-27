"""Sprint 6 hardening tests (review fixes F1-F5) and further adversarial cases.

F1  a supporting citation never overrides a person's verdict of contradicted / insufficient
F2  citation precision counts distinct (claim, unit) pairs; exclusions are reported
F3  a missing retrieval context is a data gap (retrieval_context_missing), not an invalid citation
F4  annotation fields are rejected anywhere in recorded inputs, metadata included
F5  an unjudged cited unit with a judged related unit gets a diagnostic; nothing is transferred
"""
from __future__ import annotations

import json

import pytest

from evidence_builders import (J, annotation, answer_record, build, claim, hit, outcomes, run_header,  # noqa: F401
                               run_one)
from eval.evidence_eval.evaluator import evaluate
from eval.evidence_eval.records import RESERVED_ANNOTATION_KEYS, annotation_keys_in, parse_answers
from eval.retrieval_eval.records import parse_records


def m(case, name):
    return case["metrics"][name]


def triple(r):
    return (r["status"], r["numerator_lower"], r["numerator_upper"], r["denominator"])


# ── F1: the person's verdict is not overridden ──────────────────────────────────────────────

@pytest.mark.parametrize("verdict", ["contradicted", "insufficient"])
def test_f1_supporting_citation_with_contrary_verdict_is_not_counted_as_supported(verdict):
    A = "The filter is stable [1]. It is causal [2]."
    judg = [J("chunk", "k1", "supports"), J("chunk", "k9", "contradicts")]
    c1 = claim(A, "c1", "The filter is stable", judgements=judg, status=verdict)
    rep, case = run_one(A, [hit("d", 1, "k1"), hit("d", 2, "k2")], [c1])
    cl = case["claims"][0]
    assert cl["outcome"] == "cited_evidence_supports"            # the cited unit itself does support it
    assert cl["human_support_status"] == verdict
    assert any("not counted as supported" in d for d in cl["diagnostics"])
    assert triple(m(case, "required_claim_supported_coverage")) == ("exact", 0, 0, 1)
    finding = rep["evaluation_findings"]["supporting_citation_but_verdict_differs"]
    assert finding == [{"run_id": "run-a", "case_id": "q1", "group": "verified", "claim_id": "c1",
                        "human_support_status": verdict}]


def test_f1_supporting_citation_with_unassessed_verdict_is_undecided_not_supported():
    A = "The filter is stable [1]."
    c1 = claim(A, "c1", "The filter is stable", judgements=[J("chunk", "k1", "supports")], status="not_assessed")
    _, case = run_one(A, [hit("d", 1, "k1")], [c1])
    assert triple(m(case, "required_claim_supported_coverage")) == ("bounded", 0, 1, 1)


def test_f1_agreeing_verdict_is_supported():
    A = "The filter is stable [1]."
    c1 = claim(A, "c1", "The filter is stable", judgements=[J("chunk", "k1", "supports")])
    _, case = run_one(A, [hit("d", 1, "k1")], [c1])
    assert triple(m(case, "required_claim_supported_coverage")) == ("exact", 1, 1, 1)


# ── F2: duplicates and exclusions in citation precision ─────────────────────────────────────

def test_f2_repeating_a_supporting_citation_does_not_inflate_precision():
    A = "Aliasing folds high frequencies [1][1][1][2]."
    c1 = claim(A, "c1", "Aliasing folds high frequencies",
               judgements=[J("chunk", "k1", "supports"), J("chunk", "k2", "irrelevant")])
    _, case = run_one(A, [hit("d", 1, "k1"), hit("d", 2, "k2")], [c1])
    p = m(case, "citation_precision")
    assert triple(p) == ("exact", 1, 1, 2) and p["value"] == 0.5          # not 3/4
    assert "2 distinct (claim, unit) pair(s) from 4 associated resolved reference(s)" in p["reason"]
    assert any("cited more than once" in d for d in case["claims"][0]["diagnostics"])


def test_f2_two_hits_resolving_to_the_same_chunk_are_one_pair():
    A = "Aliasing folds high frequencies [1][2]."
    c1 = claim(A, "c1", "Aliasing folds high frequencies", judgements=[J("chunk", "k1", "supports")])
    _, case = run_one(A, [hit("d", 1, "k1"), hit("d", 1, "k1")], [c1])
    assert triple(m(case, "citation_precision")) == ("exact", 1, 1, 1)


def test_f2_excluded_references_are_counted_in_the_reason():
    A = "Claim one. Claim two [1] and more text [2]."
    c1 = claim(A, "c1", "Claim one", judgements=[J("chunk", "k1", "supports")])
    c2 = claim(A, "c2", "Claim two", judgements=[J("chunk", "k1", "supports")])
    _, case = run_one(A, [hit("d", 1, "k1"), hit("d", 2, "k2")], [c1, c2])
    p = m(case, "citation_precision")
    assert p["denominator"] == 1 and "1 unassociated reference(s) excluded" in p["reason"]


def test_f2_same_unit_for_two_claims_is_two_pairs():
    A = "Claim one [1]. Claim two [1]."
    c1 = claim(A, "c1", "Claim one", judgements=[J("chunk", "k1", "supports")])
    c2 = claim(A, "c2", "Claim two", judgements=[J("chunk", "k1", "irrelevant")])
    _, case = run_one(A, [hit("d", 1, "k1")], [c1, c2])
    assert triple(m(case, "citation_precision")) == ("exact", 1, 1, 2)


# ── F3: a missing context is a data gap ─────────────────────────────────────────────────────

def _no_context(A, claims):
    answers, anns, errs = build([answer_record(A)], [annotation(A, claims)])
    assert errs == []
    rep = evaluate(anns, answers)
    return rep, rep["cases"][0]


def test_f3_no_context_is_not_an_invalid_citation_and_metrics_are_not_evaluated_or_bounded():
    A = "The filter is stable [1]. It is causal [EE210 Module 1, p. 2]."
    c1 = claim(A, "c1", "The filter is stable", judgements=[J("chunk", "k1", "supports")])
    c2 = claim(A, "c2", "It is causal")
    rep, case = _no_context(A, [c1, c2])
    assert outcomes(case) == {"c1": "retrieval_context_missing", "c2": "retrieval_context_missing"}
    assert m(case, "citation_resolution")["status"] == "not_evaluated"
    assert m(case, "citation_precision")["status"] == "not_evaluated"
    assert triple(m(case, "required_claim_citation_coverage")) == ("bounded", 0, 2, 2)
    assert triple(m(case, "required_claim_supported_coverage")) == ("bounded", 0, 2, 2)
    assert rep["data_quality"]["missing_retrieval_context"] == [{"run_id": "run-a", "case_id": "q1"}]
    assert not any(rep["evaluation_findings"][k] for k in ("invalid_citation", "no_citation"))


def test_f3_without_context_a_malformed_citation_and_no_citation_are_still_answer_findings():
    A = "The filter is stable [Source: ]. It is causal."            # a marker the Sprint 2 parser reports as malformed
    c1 = claim(A, "c1", "The filter is stable")
    c2 = claim(A, "c2", "It is causal")
    rep, case = _no_context(A, [c1, c2])
    assert outcomes(case) == {"c1": "invalid_citation", "c2": "no_citation"}   # the answer's fault, context or not
    assert case["citations"][0]["status"] == "malformed"
    assert rep["evaluation_findings"]["invalid_citation"][0]["claim_id"] == "c1"


# ── F4: no annotation fields anywhere in recorded inputs ────────────────────────────────────

@pytest.mark.parametrize("where", ["answer_metadata", "hit_metadata", "retrieval_metadata", "run_generator",
                                   "run_metadata"])
def test_f4_annotation_fields_in_metadata_are_rejected(where):
    A = "The filter is stable [1]."
    h = hit("d", 1, "k1")
    rec = answer_record(A, [h])
    run = run_header()
    leak = {"support_status": "supported"}
    if where == "answer_metadata":
        rec["metadata"] = {"note": "x", "nested": [leak]}
    elif where == "hit_metadata":
        rec["retrieval"]["hits"][0]["metadata"] = {"relation": "supports"}
    elif where == "retrieval_metadata":
        rec["retrieval"]["metadata"] = {"claims": []}
    elif where == "run_generator":
        run["generator"] = {"prompt_extras": {"expected_evidence": ["k1"]}}
    else:
        run["metadata"] = {"annotation_id": "a1"}
    out = parse_answers([("r1", run), ("r2", rec)])
    assert any("annotation field" in e for e in out.errors), out.errors
    assert out.answers == {}


def test_f4_annotation_fields_in_a_sprint5_results_file_are_a_data_problem():
    A = "The filter is stable [1]."
    c1 = claim(A, "c1", "The filter is stable", judgements=[J("chunk", "k1", "supports")])
    answers, anns, _ = build([answer_record(A)], [annotation(A, [c1])])
    rs = parse_records([("s1", {"record_type": "run", "run_id": "run-a", "provenance": "recorded"}),
                        ("s2", {"record_type": "result", "run_id": "run-a", "case_id": "q1",
                                "hits": [{"document_id": "d", "page": 1, "chunk_id": "k1",
                                          "metadata": {"judgements": {"c1": "supports"}}}]})])
    assert rs.errors == []                                   # Sprint 5 itself accepts free-form metadata
    rep = evaluate(anns, answers, rs)
    assert rep["cases"] == []
    assert "annotation field" in rep["data_quality"]["retrieval_context_problems"][0]["problem"]


def test_f4_ordinary_metadata_is_accepted_and_the_reserved_set_is_explicit():
    assert annotation_keys_in({"title": "x", "via": "dense", "score_detail": {"bm25": 1.0}}) == []
    assert annotation_keys_in({"a": [{"b": {"support_status": 1}}]}) == ["a[0].b.support_status"]
    assert {"claims", "support_status", "evidence_judgements", "relation", "expected_evidence"} <= RESERVED_ANNOTATION_KEYS


# ── F5: related judgements are reported, never transferred ──────────────────────────────────

def test_f5_page_judgement_is_reported_but_not_applied_to_a_cited_chunk():
    A = "The filter is stable [1]."
    c1 = claim(A, "c1", "The filter is stable", judgements=[J("page", "d#p1", "supports")], status="supported")
    _, case = run_one(A, [hit("d", 1, "k1")], [c1])
    cl = case["claims"][0]
    assert cl["outcome"] == "cited_evidence_not_assessed"
    assert any("related unit(s) are: page 'd#p1' (supports)" in d and "never transferred" in d for d in cl["diagnostics"])
    assert triple(m(case, "required_claim_supported_coverage")) == ("bounded", 0, 1, 1)


def test_f5_chunk_judgement_is_reported_but_not_applied_to_a_cited_page():
    A = "The filter is stable [d, p. 1]."
    c1 = claim(A, "c1", "The filter is stable", judgements=[J("chunk", "k1", "contradicts")])
    _, case = run_one(A, [hit("d", 1, "k1")], [c1])
    cl = case["claims"][0]
    assert cl["outcome"] == "cited_evidence_not_assessed"
    assert any("chunk 'k1' (contradicts)" in d for d in cl["diagnostics"])


def test_f5_unrelated_judgements_produce_no_related_diagnostic():
    A = "The filter is stable [1]."
    c1 = claim(A, "c1", "The filter is stable", judgements=[J("chunk", "k7", "supports")])
    _, case = run_one(A, [hit("d", 1, "k1"), hit("e", 3, "k7")], [c1])
    assert not any("related unit" in d for d in case["claims"][0]["diagnostics"])


# ── more adversarial attempts to turn an unsupported claim into a supported one ─────────────

def test_adv_supports_judgement_on_a_hit_that_was_not_cited_does_not_help():
    A = "The filter is stable [2]."
    c1 = claim(A, "c1", "The filter is stable", judgements=[J("chunk", "k1", "supports")])
    _, case = run_one(A, [hit("d", 1, "k1"), hit("d", 2, "k2")], [c1])
    assert outcomes(case)["c1"] == "cited_evidence_not_assessed"
    assert triple(m(case, "required_claim_supported_coverage")) == ("bounded", 0, 1, 1)


def test_adv_citation_in_another_claims_span_does_not_support_this_claim():
    A = "The filter is stable. It is causal [1]."
    c1 = claim(A, "c1", "The filter is stable", judgements=[J("chunk", "k1", "supports")])
    c2 = claim(A, "c2", "It is causal", judgements=[J("chunk", "k1", "irrelevant")], status="insufficient")
    _, case = run_one(A, [hit("d", 1, "k1")], [c1, c2])
    assert outcomes(case) == {"c1": "no_citation", "c2": "cited_evidence_irrelevant"}
    assert triple(m(case, "required_claim_supported_coverage")) == ("exact", 0, 0, 2)


def test_adv_out_of_range_number_is_never_wrapped_or_clamped():
    A = "The filter is stable [3]."
    c1 = claim(A, "c1", "The filter is stable", judgements=[J("chunk", "k1", "supports")])
    _, case = run_one(A, [hit("d", 1, "k1"), hit("d", 2, "k2")], [c1])
    assert outcomes(case)["c1"] == "invalid_citation"


def test_adv_fixture_support_never_reaches_a_recorded_run_or_the_verified_group():
    A = "The filter is stable [1]."
    c1 = claim(A, "c1", "The filter is stable", judgements=[J("chunk", "k1", "supports")])
    answers, anns, errs = build([answer_record(A, [hit("d", 1, "k1")])], [annotation(A, [c1], status="fixture")],
                                run_provenance="recorded")
    rep = evaluate(anns, answers)
    assert rep["cases"] == [] and rep["aggregates"] == {}
    assert "fixture annotations may only annotate fixture runs" in rep["data_quality"]["inconsistent_annotations"][0]["problems"]


def test_adv_report_json_contains_no_annotation_status_in_inputs_echo():
    # the report echoes inputs (answer hash, citations) but never feeds annotations back into recorded inputs
    A = "The filter is stable [1]."
    c1 = claim(A, "c1", "The filter is stable", judgements=[J("chunk", "k1", "supports")])
    answers, anns, _ = build([answer_record(A, [hit("d", 1, "k1")])], [annotation(A, [c1])])
    before = {k: (v.answer, v.retrieval) for k, v in answers.answers.items()}
    evaluate(anns, answers)
    assert {k: (v.answer, v.retrieval) for k, v in answers.answers.items()} == before
    assert annotation_keys_in(json.loads(json.dumps(
        [{"answer": v.answer, "hits": [dict(h.metadata) for h in v.retrieval.hits]} for v in answers.answers.values()]))) == []


# ── schema rule: a verdict must be traceable to a judged unit (found by the breakage check) ──

@pytest.mark.parametrize("verdict,judgements,needle", [
    ("supported", [], "'supported' needs at least one 'supports' judgement"),
    ("supported", [J("chunk", "k1", "insufficient")], "'supported' needs at least one 'supports' judgement"),
    ("contradicted", [J("chunk", "k1", "supports")], "'contradicted' needs at least one 'contradicts' judgement"),
])
def test_verdict_without_a_matching_judgement_is_rejected(verdict, judgements, needle):
    from eval.evidence_eval.annotations import parse_annotation
    A = "The filter is stable [1]."
    raw = annotation(A, [claim(A, "c1", "The filter is stable", judgements=judgements, status=verdict)])
    ann, errs = parse_annotation(raw, "t")
    assert ann is None and any(needle in e for e in errs), errs


def test_verdicts_with_a_matching_judgement_are_accepted():
    from eval.evidence_eval.annotations import parse_annotation
    A = "The filter is stable [1]."
    for verdict, rel in (("supported", "supports"), ("contradicted", "contradicts"), ("insufficient", "insufficient"),
                         ("not_assessed", "irrelevant")):
        raw = annotation(A, [claim(A, "c1", "The filter is stable", judgements=[J("chunk", "k1", rel)], status=verdict)])
        assert parse_annotation(raw, "t")[1] == [], verdict
