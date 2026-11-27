"""Join recorded answers with annotations and evaluate them (Sprint 6). Offline; generates nothing.

evaluate(annotations, answers, retrieval=None, dataset_items=None) -> report dict (JSON-safe,
deterministic: cases sorted by run_id then case_id).

Joining, by (run_id, case_id):
  answer + annotation   fully evaluated, if the annotation is consistent with the answer:
                        answer_sha256 equals the hash of the recorded answer text, the questions
                        are identical, every claim span lies inside the answer and its text equals
                        answer[start:end], every explicit citation offset matches a citation, and a
                        fixture annotation sits on a fixture run (and only there). Otherwise the case
                        is a DATA PROBLEM and is not evaluated.
  answer only           citation_validity and citation_resolution only (group "unannotated").
  annotation only       a data problem (no recorded answer to evaluate).
The retrieval context is the answer's embedded "retrieval", or a Sprint 5 result with the same
run_id and case_id; both at once is a data problem.

Results are grouped by annotation status: verified | unverified | fixture | unannotated. Groups are
never pooled: an unverified or fixture judgement is never counted as verified ground truth.
"""
from __future__ import annotations

from collections import Counter
from typing import Sequence

from eval.evidence_eval.annotations import SUPPORT_STATUSES, AnswerAnnotation
from eval.evidence_eval.metrics import (ANNOTATION_DEPENDENT, METRIC_NAMES, OUTCOMES, Ratio, add,
                                        citation_metrics, claim_metrics, evaluate_claim)
from eval.evidence_eval.records import AnswerSet, annotation_keys_in
from eval.evidence_eval.resolution import (AMBIGUOUS, RESOLVED, UNASSOCIATED, EvidenceContext,
                                           associate, resolve_citations)
from eval.retrieval_eval.records import RecordSet

GROUPS = ("verified", "unverified", "fixture", "unannotated")
OFFLINE_NOTE = ("Offline evaluation of RECORDED answers and retrieval contexts. This evaluator does not "
                "generate answers or run a retriever; support comes only from human judgements.")
FIXTURE_NOTE = "Fixture runs or annotations are illustrative example data, not Saarthi performance."
UNVERIFIED_NOTE = "Unverified annotations are provisional; their results are reported in a separate group."


def _context(ans, retrieval: RecordSet | None):
    from_file = retrieval.results.get((ans.run_id, ans.case_id)) if retrieval else None
    if ans.retrieval is not None and from_file is not None:
        return None, "both an embedded retrieval context and a Sprint 5 result were given"
    res = ans.retrieval or from_file
    if res is None:
        return None, None
    if from_file is not None:
        leaked = annotation_keys_in({"metadata": dict(from_file.metadata),
                                     "hits": [dict(h.metadata) for h in (from_file.hits or ())]})
        if leaked:
            return None, (f"the Sprint 5 result ({from_file.where}) contains annotation field(s) {leaked}; "
                          f"recorded inputs may not carry ground truth")
    if res.error is not None:
        return None, f"the recorded retrieval reported an error: {res.error}"
    return (EvidenceContext.from_result(res), "embedded" if ans.retrieval is not None else "sprint5_results"), None


def _consistency(ann: AnswerAnnotation, ans, run_provenance: str) -> list[str]:
    errs = []
    if ann.answer_sha256 != ans.sha256:
        errs.append("answer_sha256 does not match the recorded answer (the annotation is for different text)")
        return errs
    if ann.question != ans.question:
        errs.append("the annotation's question differs from the recorded answer's question")
    for c in ann.claims:
        if c.end > len(ans.answer):
            errs.append(f"claim {c.claim_id!r}: span {c.start}..{c.end} lies outside the answer ({len(ans.answer)} chars)")
        elif ans.answer[c.start:c.end] != c.text:
            errs.append(f"claim {c.claim_id!r}: text differs from answer[{c.start}:{c.end}]")
    st = ann.provenance.status
    if st == "fixture" and run_provenance != "fixture":
        errs.append("fixture annotations may only annotate fixture runs")
    if st != "fixture" and run_provenance == "fixture":
        errs.append(f"{st} annotations cannot annotate a fixture run (fixture output is not ground truth)")
    return errs


def _evaluate_case(ans, ann: AnswerAnnotation | None, ctx_info) -> dict:
    ctx = ctx_info[0] if ctx_info else None
    refs = resolve_citations(ans.answer, ans.citation_numbering, ctx)
    claims = list(ann.claims) if ann else []
    assoc = associate(ans.answer, refs, claims)
    group = ann.provenance.status if ann else "unannotated"
    out = {"run_id": ans.run_id, "case_id": ans.case_id, "question": ans.question, "group": group,
           "answer_sha256": ans.sha256, "answer_chars": len(ans.answer),
           "citation_numbering": ans.citation_numbering,
           "retrieval_source": ctx_info[1] if ctx_info else None,
           "annotation": None if ann is None else {"annotation_id": ann.annotation_id, "language": ann.language,
                                                   **ann.provenance.to_dict()}}
    if assoc.errors:
        return {**out, "data_errors": assoc.errors}
    claims_by_id = {c.claim_id: c for c in claims}
    results = [evaluate_claim(c, refs, assoc, ctx) for c in claims]
    metrics = citation_metrics(refs, assoc, claims_by_id, ctx is not None, ann is not None)
    if ann is not None:
        metrics.update(claim_metrics(results))
    else:
        metrics.update({m: Ratio(0, 0, 0, evaluated=False, reason="no annotation for this answer")
                        for m in ("required_claim_citation_coverage", "required_claim_supported_coverage")})
    by_index = {r.index: r for r in refs}
    cites = []
    for r in refs:
        state, cids = assoc.by_ref[r.index]
        cites.append({**r.to_dict(), "association": state, "claim_ids": list(cids)})
    req = [c for c in claims if c.requires_evidence]
    out.update({
        "citations": cites,
        "claims": [cr.to_dict(by_index) for cr in results],
        "metrics": {m: metrics[m].to_dict() for m in METRIC_NAMES},
        "_ratios": metrics,
        "counts": {"outcomes": {o: sum(1 for r in results if r.outcome == o) for o in OUTCOMES},
                   "human_support_status": {s: sum(1 for c in claims if c.support_status == s) for s in SUPPORT_STATUSES},
                   "required_human_support_status": {s: sum(1 for c in req if c.support_status == s) for s in SUPPORT_STATUSES},
                   "claims": len(claims), "evidence_required_claims": len(req),
                   "citation_references": len(refs)},
    })
    return out


def _findings(cases: list[dict]) -> dict:
    f = {k: [] for k in ("no_citation", "association_unresolved", "invalid_citation", "cited_evidence_irrelevant",
                         "cited_evidence_insufficient", "cited_evidence_contradicts", "conflicting_evidence",
                         "cited_evidence_not_assessed", "human_verdict_contradicted", "human_verdict_insufficient",
                         "supporting_citation_but_verdict_differs",
                         "unassociated_citations", "ambiguous_citations", "unresolved_citations",
                         "duplicate_citations")}
    for c in cases:
        if "claims" not in c:
            continue
        base = {"run_id": c["run_id"], "case_id": c["case_id"], "group": c["group"]}
        for cl in c["claims"]:
            if cl["outcome"] in f:
                f[cl["outcome"]].append({**base, "claim_id": cl["claim_id"], "reasons": cl["invalid_citation_reasons"]}
                                        if cl["outcome"] == "invalid_citation" else {**base, "claim_id": cl["claim_id"]})
            if cl["outcome"] == "cited_evidence_supports" and cl["human_support_status"] != "supported":
                f["supporting_citation_but_verdict_differs"].append({**base, "claim_id": cl["claim_id"],
                                                                     "human_support_status": cl["human_support_status"]})
            if cl["human_support_status"] == "contradicted":
                f["human_verdict_contradicted"].append({**base, "claim_id": cl["claim_id"]})
            if cl["human_support_status"] == "insufficient":
                f["human_verdict_insufficient"].append({**base, "claim_id": cl["claim_id"]})
        keys = Counter()
        for r in c["citations"]:
            if r["association"] == UNASSOCIATED and c["annotation"] is not None:
                f["unassociated_citations"].append({**base, "marker": r["marker"], "start": r["start"]})
            if r["association"] == AMBIGUOUS:
                f["ambiguous_citations"].append({**base, "marker": r["marker"], "start": r["start"],
                                                 "candidate_claims": r["claim_ids"]})
            if r["status"] != RESOLVED:
                f["unresolved_citations"].append({**base, "marker": r["marker"], "start": r["start"],
                                                  "status": r["status"], "detail": r["detail"]})
            if not r["in_sources_block"]:
                keys[(r["kind"], r["number"], r["title"], r["page"])] += 1
        dups = sorted(f"{k[1] if k[1] is not None else (k[2] or '')}{', p.' + k[3] if k[3] else ''}" for k, n in keys.items() if n > 1)
        if dups:
            f["duplicate_citations"].append({**base, "repeated": dups})
    return f


def evaluate(annotations: Sequence[AnswerAnnotation], answers: AnswerSet, retrieval: RecordSet | None = None,
             dataset_items: Sequence | None = None) -> dict:
    data = {"input_errors": list(answers.errors) + (list(retrieval.errors) if retrieval else []),
            "annotations_without_answer": [], "inconsistent_annotations": [],
            "retrieval_context_problems": [], "missing_retrieval_context": [], "dataset_question_mismatches": []}
    ann_by = {(a.run_id, a.case_id): a for a in annotations}
    queries = {i.id: i.query for i in (dataset_items or ())}
    cases = []
    for key in sorted(answers.answers):
        ans = answers.answers[key]
        run = answers.runs[ans.run_id]
        ctx_info, ctx_err = _context(ans, retrieval)
        if ctx_err:
            data["retrieval_context_problems"].append({"run_id": ans.run_id, "case_id": ans.case_id, "problem": ctx_err})
            continue
        if ctx_info is None:
            data["missing_retrieval_context"].append({"run_id": ans.run_id, "case_id": ans.case_id})
        if ans.case_id in queries and queries[ans.case_id] != ans.question:
            data["dataset_question_mismatches"].append({"run_id": ans.run_id, "case_id": ans.case_id})
        ann = ann_by.get(key)
        if ann is not None:
            errs = _consistency(ann, ans, run.provenance)
            if errs:
                data["inconsistent_annotations"].append({"annotation_id": ann.annotation_id, "run_id": ans.run_id,
                                                         "case_id": ans.case_id, "problems": errs})
                continue
        case = _evaluate_case(ans, ann, ctx_info)
        if "data_errors" in case:
            data["inconsistent_annotations"].append({"annotation_id": ann.annotation_id, "run_id": ans.run_id,
                                                     "case_id": ans.case_id, "problems": case["data_errors"]})
            continue
        cases.append(case)
    for key, ann in sorted(ann_by.items()):
        if key not in answers.answers:
            data["annotations_without_answer"].append({"annotation_id": ann.annotation_id, "run_id": key[0], "case_id": key[1]})

    aggregates = {}
    for run_id in sorted({c["run_id"] for c in cases}):
        per = {}
        for g in GROUPS:
            sel = [c for c in cases if c["run_id"] == run_id and c["group"] == g]
            if not sel:
                continue
            metrics = {}
            for m in METRIC_NAMES:
                if g == "unannotated" and m in ANNOTATION_DEPENDENT:
                    continue
                metrics[m] = add([c["_ratios"][m] for c in sel]).to_dict()
            counts = {k: dict(sum((Counter(c["counts"][k]) for c in sel), Counter()))
                      for k in ("outcomes", "human_support_status", "required_human_support_status")}
            for k in counts:
                base = OUTCOMES if k == "outcomes" else SUPPORT_STATUSES
                counts[k] = {name: counts[k].get(name, 0) for name in base}
            per[g] = {"cases": len(sel), "claims": sum(c["counts"]["claims"] for c in sel),
                      "evidence_required_claims": sum(c["counts"]["evidence_required_claims"] for c in sel),
                      "metrics": metrics, "counts": counts}
        aggregates[run_id] = per
    for c in cases:
        del c["_ratios"]
    runs = {rid: {**answers.runs[rid].to_dict(), "illustrative": answers.runs[rid].provenance == "fixture"}
            for rid in sorted(answers.runs)}
    notes = [OFFLINE_NOTE]
    if any(r["illustrative"] for r in runs.values()) or any(c["group"] == "fixture" for c in cases):
        notes.append(FIXTURE_NOTE)
    if any(c["group"] == "unverified" for c in cases):
        notes.append(UNVERIFIED_NOTE)
    return {
        "evaluation": {"schema": "evidence-eval-report/1", "notes": notes, "generates_answers": False,
                       "runs_retriever": False, "groups": list(GROUPS), "metrics": list(METRIC_NAMES)},
        "runs": runs,
        "aggregates": aggregates,
        "evaluation_findings": _findings(cases),
        "data_quality": data,
        "cases": cases,
    }
