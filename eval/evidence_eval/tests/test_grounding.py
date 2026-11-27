"""Sprint 6 tests: citation resolution, association, claim outcomes, metrics and adversarial cases."""
from __future__ import annotations

import pytest

from evidence_builders import J, claim, hit, outcomes, run_one
from eval.evidence_eval.annotations import EvidenceRef
from eval.evidence_eval.resolution import EvidenceContext, associate, normalise_name, resolve_citations
from eval.retrieval_eval.records import parse_records

DOC = "EE210 Module 1 Signals and Sampling"
HITS = [hit(DOC, 3, "k1"), hit("Doc B", 2, "k2"), hit("Doc C", 1)]


def ctx(hits=HITS, lists=None):
    body = {"hits": hits} if lists is None else lists
    rs = parse_records([("x", {"record_type": "run", "run_id": "r", "provenance": "recorded"}),
                        ("x", {"record_type": "result", "run_id": "r", "case_id": "c", **body})])
    assert rs.errors == []
    return EvidenceContext.from_result(rs.results[("r", "c")])


def resolve(answer, numbering="retrieval_order", context="default"):
    return resolve_citations(answer, numbering, ctx() if context == "default" else context)


def st(refs):
    return [(r.marker, r.status, r.unit.to_dict() if r.unit else None) for r in refs]


# ── Citation parsing and resolution ─────────────────────────────────────────────────────────

def test_numeric_citations_resolve_to_the_nth_hit_most_specific_unit():
    refs = resolve("a [1] b [2, 3] c [3].")
    assert [(r.number, r.status, r.unit) for r in refs] == [
        (1, "resolved", EvidenceRef("chunk", "k1")), (2, "resolved", EvidenceRef("chunk", "k2")),
        (3, "resolved", EvidenceRef("page", "Doc C#p1")), (3, "resolved", EvidenceRef("page", "Doc C#p1"))]
    assert refs[1].start == refs[2].start and refs[1].marker == "[2, 3]"   # positions are traceable


def test_numeric_citations_are_never_guessed():
    assert [r.status for r in resolve("x [1].", "unspecified")] == ["unresolvable"]
    out = resolve("x [4].")
    assert out[0].status == "unresolvable" and "outside the 3" in out[0].detail
    lists = ctx(lists={"document_ids": [DOC], "chunk_ids": ["k1"]})
    assert resolve("x [1].", context=lists)[0].status == "unresolvable"
    nohit = ctx(hits=[hit(None)])
    assert "no document, page or chunk id" in resolve("x [1].", context=nohit)[0].detail


def test_label_citations_resolve_only_to_retrieved_evidence():
    refs = resolve(f"a [{DOC}, p. 3] b [{DOC}, p. 4] c [Doc D, p. 1] d [Source: Doc B] e [Video: Doc C, 1:02].")
    assert st(refs) == [
        (f"[{DOC}, p. 3]", "resolved", {"level": "page", "id": f"{DOC}#p3"}),
        (f"[{DOC}, p. 4]", "page_not_retrieved", None),
        ("[Doc D, p. 1]", "not_in_context", None),
        ("[Source: Doc B]", "resolved", {"level": "document", "id": "Doc B"}),
        ("[Video: Doc C, 1:02]", "resolved", {"level": "document", "id": "Doc C"}),
    ]
    assert "timestamp 1:02 not checked" in refs[4].detail


def test_title_matching_is_exact_after_normalisation_never_fuzzy():
    c = ctx(hits=[hit(DOC, 3, "k1", source_file="EE210_Module_1_Signals_and_Sampling.pdf")])
    good = resolve("[Source: EE210_Module_1_Signals_and_Sampling.pdf, p.3]", context=c)
    assert good[0].status == "resolved" and good[0].unit == EvidenceRef("page", f"{DOC}#p3")
    assert resolve("[EE210 Module 1, p. 3]", context=c)[0].status == "not_in_context"
    assert normalise_name("  EE210_Module_1.PDF ") == "ee210 module 1"


def test_a_name_shared_by_two_documents_is_ambiguous():
    c = ctx(hits=[hit("d1", 1, title="Notes"), hit("d2", 1, title="Notes")])
    r = resolve("[Notes, p. 1]", context=c)[0]
    assert r.status == "unresolvable" and "names 2 retrieved documents" in r.detail


def test_bare_title_needs_it_to_be_a_retrieved_name():
    assert st(resolve("see [Doc B] and [Random words here].")) == [("[Doc B]", "resolved", {"level": "document", "id": "Doc B"})]


def test_malformed_and_no_context_are_distinct_from_missing_evidence():
    refs = resolve("x [Source: , p. 3] y [Video: no timestamp] z [Doc D, p. 2]")
    assert [r.status for r in refs] == ["malformed", "malformed", "not_in_context"]
    none = resolve_citations("x [1] y [Doc B, p. 2]", "retrieval_order", None)
    assert [r.status for r in none] == ["no_retrieval_context", "no_retrieval_context"]


def test_code_math_and_indexing_are_not_citations():
    assert resolve("`a[1]` and $x[1]$ and x[n] and [0, 1] and [2018]") == []


def test_empty_answer_has_no_citations():
    assert resolve("") == []


# ── Association ─────────────────────────────────────────────────────────────────────────────

def anns(answer, *claims):
    from eval.evidence_eval.annotations import parse_annotation
    from evidence_builders import annotation
    a, errs = parse_annotation(annotation(answer, list(claims)), "t")
    assert errs == [], errs
    return list(a.claims)


def test_positional_association_inside_and_directly_after_a_claim():
    a = "Alpha is true [1]. Beta is true, as shown [2] here. Gamma."
    cl = anns(a, claim(a, "c1", "Alpha is true"), claim(a, "c2", "Beta is true, as shown [2] here"), claim(a, "c3", "Gamma"))
    refs = resolve(a)
    asc = associate(a, refs, cl)
    assert asc.by_claim == {"c1": [0], "c2": [1], "c3": []}


def test_citation_not_adjacent_to_any_claim_is_unassociated():
    a = "Alpha is true. Unrelated words [1]."
    cl = anns(a, claim(a, "c1", "Alpha is true"))
    asc = associate(a, resolve(a), cl)
    assert asc.by_ref[0] == ("unassociated", ())


def test_ambiguous_association_is_reported_not_guessed():
    a = "Alpha [1] Beta"
    cl = anns(a, claim(a, "c1", "Alpha"), claim(a, "c2", "[1] Beta"))
    asc = associate(a, resolve(a), cl)
    assert asc.by_ref[0] == ("ambiguous", ("c1", "c2")) and asc.by_claim == {"c1": [], "c2": []}


def test_explicit_citation_starts_override_position():
    a = "Alpha is true. Beta is true [1]."
    pos = a.index("[1]")
    cl = anns(a, claim(a, "c1", "Alpha is true", citation_starts=[pos]), claim(a, "c2", "Beta is true"))
    asc = associate(a, resolve(a), cl)
    assert asc.by_claim == {"c1": [0], "c2": []}


def test_explicit_offset_matching_no_citation_is_a_data_error():
    a = "Alpha is true [1]."
    cl = anns(a, claim(a, "c1", "Alpha is true", citation_starts=[3]))
    assert any("matches no citation" in e for e in associate(a, resolve(a), cl).errors)


def test_sources_block_citations_are_never_associated():
    a = f"Alpha is true.\n\n**Sources:** [{DOC}, p. 3]"
    cl = anns(a, claim(a, "c1", "Alpha is true"))
    refs = resolve(a)
    asc = associate(a, refs, cl)
    assert refs[0].in_sources_block and asc.by_ref[0] == ("sources_block", ()) and asc.by_claim["c1"] == []


# ── Claim outcomes (distinct, never collapsed) ──────────────────────────────────────────────

def test_every_claim_outcome_is_distinguishable():
    a = ("P1 holds [1]. P2 holds. P3 holds [9]. P4 holds [2]. P5 holds [3]. P6 holds [1]. "
         "P7 holds [2]. P8 holds [1][2]. P9 holds. P10 holds [Doc D, p. 1].")
    hits = [hit("d1", 1, "k1"), hit("d2", 1, "k2"), hit("d3", 1, "k3")]
    claims = [
        claim(a, "c1", "P1 holds", judgements=[J("chunk", "k1", "supports")]),
        claim(a, "c2", "P2 holds"),
        claim(a, "c3", "P3 holds"),
        claim(a, "c4", "P4 holds", judgements=[J("chunk", "k2", "irrelevant")], status="insufficient"),
        claim(a, "c5", "P5 holds", judgements=[J("chunk", "k3", "insufficient")], status="insufficient"),
        claim(a, "c6", "P6 holds", judgements=[J("chunk", "k1", "contradicts")]),
        claim(a, "c7", "P7 holds"),
        claim(a, "c8", "P8 holds", judgements=[J("chunk", "k1", "supports"), J("chunk", "k2", "contradicts")], status="contradicted"),
        claim(a, "c9", "P9 holds", requires=False),
        claim(a, "c10", "P10 holds", judgements=[J("document", "Doc D", "supports")], status="supported"),
    ]
    _, case = run_one(a, hits, claims)
    assert outcomes(case) == {
        "c1": "cited_evidence_supports", "c2": "no_citation", "c3": "invalid_citation",
        "c4": "cited_evidence_irrelevant", "c5": "cited_evidence_insufficient", "c6": "cited_evidence_contradicts",
        "c7": "cited_evidence_not_assessed", "c8": "conflicting_evidence", "c9": "evidence_not_required",
        "c10": "invalid_citation"}
    by = {c["claim_id"]: c for c in case["claims"]}
    assert by["c3"]["invalid_citation_reasons"] == ["[9]: unresolvable ([9] is outside the 3 retrieved hit(s))"]
    assert by["c10"]["invalid_citation_reasons"][0].startswith("[Doc D, p. 1]: not_in_context")
    assert by["c9"]["coverage_satisfied"] is None and by["c2"]["coverage_satisfied"] is False
    assert by["c1"]["resolved_evidence"] == [{"citation_index": 0, "unit": {"level": "chunk", "id": "k1"}, "relation": "supports"}]
    assert by["c8"]["human_support_status"] == "contradicted"


def test_association_unresolved_outcome():
    a = "Alpha [1] Beta"
    _, case = run_one(a, [hit("d", 1, "k1")], [claim(a, "c1", "Alpha"), claim(a, "c2", "[1] Beta")])
    assert outcomes(case) == {"c1": "association_unresolved", "c2": "association_unresolved"}
    assert case["metrics"]["required_claim_citation_coverage"]["status"] == "bounded"


def test_human_verdict_is_reported_unchanged_beside_the_outcome():
    a = "Alpha holds [1]."
    _, case = run_one(a, [hit("d", 1, "k1")], [claim(a, "c1", "Alpha holds", judgements=[J("chunk", "k1", "supports")],
                                                     status="insufficient")])
    c = case["claims"][0]
    assert c["outcome"] == "cited_evidence_supports" and c["human_support_status"] == "insufficient"
    assert any("overall verdict is 'insufficient'" in d for d in c["diagnostics"])


def test_expected_evidence_diagnostics_never_imply_support():
    a = "Alpha holds [1]."
    _, case = run_one(a, [hit("d", 1, "k1"), hit("d", 2, "k2")],
                      [claim(a, "c1", "Alpha holds", expected_evidence=[{"level": "chunk", "id": "k2"},
                                                                        {"level": "chunk", "id": "k9"}])])
    c = case["claims"][0]
    assert c["outcome"] == "cited_evidence_not_assessed"
    assert any("not in the retrieval context: ['k9']" in d for d in c["diagnostics"])
    assert any("retrieved but not cited: ['k2']" in d for d in c["diagnostics"])


def test_duplicate_citations_are_reported_once_per_unit():
    a = "Alpha holds [1] [1]."
    _, case = run_one(a, [hit("d", 1, "k1")], [claim(a, "c1", "Alpha holds", judgements=[J("chunk", "k1", "supports")])])
    assert any("cited more than once" in d for d in case["claims"][0]["diagnostics"])
    assert case["metrics"]["citation_validity"]["denominator"] == 2


# ── Metrics: definitions, denominators, n/a vs zero, bounds ─────────────────────────────────

def m(case, name):
    return case["metrics"][name]


def test_metric_values_and_denominators():
    a = "A holds [1]. B holds [2]. C holds [7]. D holds. E holds [3]."
    hits = [hit("d", 1, "k1"), hit("d", 2, "k2"), hit("d", 3, "k3")]
    claims = [claim(a, "a", "A holds", judgements=[J("chunk", "k1", "supports")]),
              claim(a, "b", "B holds", judgements=[J("chunk", "k2", "irrelevant")], status="insufficient"),
              claim(a, "c", "C holds"), claim(a, "d", "D holds"),
              claim(a, "e", "E holds")]                                       # k3 unjudged
    _, case = run_one(a, hits, claims)
    assert (m(case, "citation_validity")["numerator_lower"], m(case, "citation_validity")["denominator"]) == (4, 4)
    assert m(case, "citation_resolution")["value"] == pytest.approx(3 / 4)
    p = m(case, "citation_precision")
    assert p["status"] == "bounded" and (p["lower"], p["upper"]) == (pytest.approx(1 / 3), pytest.approx(2 / 3))
    assert m(case, "required_claim_citation_coverage")["value"] == pytest.approx(3 / 5)
    s = m(case, "required_claim_supported_coverage")
    assert s["status"] == "bounded" and (s["numerator_lower"], s["numerator_upper"], s["denominator"]) == (1, 2, 5)


def test_zero_denominators_are_not_applicable_not_zero():
    _, case = run_one("No citations here at all.", [hit("d", 1, "k1")], [])
    for name in ("citation_validity", "citation_resolution", "citation_precision",
                 "required_claim_citation_coverage", "required_claim_supported_coverage"):
        assert m(case, name)["status"] == "not_applicable" and m(case, name)["value"] is None


def test_zero_is_reported_as_zero_when_there_is_something_to_measure():
    a = "Alpha holds."
    _, case = run_one(a, [hit("d", 1, "k1")], [claim(a, "c1", "Alpha holds")])
    cov = m(case, "required_claim_citation_coverage")
    assert cov["status"] == "exact" and cov["value"] == 0.0 and cov["denominator"] == 1


def test_empty_answer_and_empty_evidence():
    _, case = run_one("", [], [])
    assert case["citations"] == [] and m(case, "citation_validity")["status"] == "not_applicable"
    a = "Alpha holds [1]."
    _, case = run_one(a, [], [claim(a, "c1", "Alpha holds")])
    assert case["claims"][0]["outcome"] == "invalid_citation"
    assert m(case, "citation_resolution")["value"] == 0.0


def test_claims_not_requiring_evidence_are_outside_coverage_denominators():
    a = "Hello there. Alpha holds [1]."
    _, case = run_one(a, [hit("d", 1, "k1")], [claim(a, "g", "Hello there", requires=False),
                                               claim(a, "c1", "Alpha holds", judgements=[J("chunk", "k1", "supports")])])
    assert m(case, "required_claim_citation_coverage")["denominator"] == 1


# ── Adversarial: ways an unsupported claim could wrongly be marked supported ────────────────

def supported(case, cid="c1"):
    return next(c for c in case["claims"] if c["claim_id"] == cid)["outcome"] == "cited_evidence_supports"


def test_adv_identical_text_in_evidence_is_not_support():
    a = "Alpha holds [1]."
    _, case = run_one(a, [hit("d", 1, "k1", text="Alpha holds")], [claim(a, "c1", "Alpha holds")])
    assert not supported(case) and case["claims"][0]["outcome"] == "cited_evidence_not_assessed"


def test_adv_judgement_for_another_claim_does_not_transfer():
    a = "Alpha holds [1]. Beta holds [1]."
    _, case = run_one(a, [hit("d", 1, "k1")], [claim(a, "c1", "Alpha holds"),
                                               claim(a, "c2", "Beta holds", judgements=[J("chunk", "k1", "supports")])])
    assert not supported(case, "c1") and supported(case, "c2")


def test_adv_page_judgement_does_not_cover_a_cited_chunk_and_vice_versa():
    a = "Alpha holds [1]."
    _, case = run_one(a, [hit("d", 1, "k1")], [claim(a, "c1", "Alpha holds", judgements=[J("page", "d#p1", "supports")])])
    assert not supported(case)
    b = "Alpha holds [d, p. 1]."
    _, case = run_one(b, [hit("d", 1, "k1")], [claim(b, "c1", "Alpha holds", judgements=[J("chunk", "k1", "supports")])])
    assert not supported(case)
    _, case = run_one(b, [hit("d", 1, "k1")], [claim(b, "c1", "Alpha holds", judgements=[J("document", "d", "supports")])])
    assert not supported(case)


def test_adv_supporting_evidence_outside_the_retrieval_context_is_not_support():
    a = "Alpha holds [Doc Z, p. 1]."
    _, case = run_one(a, [hit("d", 1, "k1")], [claim(a, "c1", "Alpha holds", judgements=[J("page", "Doc Z#p1", "supports")])])
    assert case["claims"][0]["outcome"] == "invalid_citation"


def test_adv_undeclared_numbering_is_not_resolved_even_if_the_hit_supports():
    a = "Alpha holds [1]."
    _, case = run_one(a, [hit("d", 1, "k1")], [claim(a, "c1", "Alpha holds", judgements=[J("chunk", "k1", "supports")])],
                      numbering="unspecified")
    assert case["claims"][0]["outcome"] == "invalid_citation"


def test_adv_expected_evidence_and_human_verdict_alone_are_not_citation_support():
    a = "Alpha holds."
    _, case = run_one(a, [hit("d", 1, "k1")], [claim(a, "c1", "Alpha holds", judgements=[J("chunk", "k1", "supports")],
                                                     expected_evidence=[{"level": "chunk", "id": "k1"}])])
    assert case["claims"][0]["outcome"] == "no_citation" and case["claims"][0]["human_support_status"] == "supported"
    assert m(case, "required_claim_supported_coverage")["value"] == 0.0


def test_adv_sources_block_next_to_a_claim_is_not_inline_support():
    a = f"Alpha holds\n\n**Sources:** [{DOC}, p. 3]"
    _, case = run_one(a, [hit(DOC, 3, "k1")], [claim(a, "c1", "Alpha holds", judgements=[J("page", f"{DOC}#p3", "supports")])])
    assert case["claims"][0]["outcome"] == "no_citation"


def test_adv_near_miss_title_does_not_resolve():
    a = "Alpha holds [EE210 Module 1, p. 3]."
    _, case = run_one(a, [hit(DOC, 3, "k1")], [claim(a, "c1", "Alpha holds", judgements=[J("page", f"{DOC}#p3", "supports")])])
    assert case["claims"][0]["outcome"] == "invalid_citation"


def test_adv_contradiction_is_not_hidden_by_a_supporting_citation():
    a = "Alpha holds [1][2]."
    _, case = run_one(a, [hit("d", 1, "k1"), hit("d", 2, "k2")],
                      [claim(a, "c1", "Alpha holds", judgements=[J("chunk", "k1", "supports"), J("chunk", "k2", "contradicts")],
                             status="contradicted")])
    assert case["claims"][0]["outcome"] == "conflicting_evidence"
    assert m(case, "required_claim_supported_coverage")["value"] == 0.0


def test_adv_citation_inside_code_is_not_support():
    a = "Alpha holds `[1]`."
    _, case = run_one(a, [hit("d", 1, "k1")], [claim(a, "c1", "Alpha holds", judgements=[J("chunk", "k1", "supports")])])
    assert case["claims"][0]["outcome"] == "no_citation"


def test_adv_unjudged_citation_never_counts_as_irrelevant_or_supported():
    a = "Alpha holds [1]."
    _, case = run_one(a, [hit("d", 1, "k1")], [claim(a, "c1", "Alpha holds")])
    p = m(case, "citation_precision")
    assert p["status"] == "bounded" and (p["lower"], p["upper"]) == (0.0, 1.0)


def test_adv_unverified_support_is_kept_out_of_the_verified_group():
    a = "Alpha holds [1]."
    report, case = run_one(a, [hit("d", 1, "k1")], [claim(a, "c1", "Alpha holds", judgements=[J("chunk", "k1", "supports")])],
                           status="unverified")
    assert case["group"] == "unverified" and set(report["aggregates"]["run-a"]) == {"unverified"}
    assert any("provisional" in n for n in report["evaluation"]["notes"])
