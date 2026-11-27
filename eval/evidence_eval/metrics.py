"""Claim outcomes and citation/grounding metrics (Sprint 6). Deterministic, no weighted score.

CLAIM OUTCOMES (per annotated claim; distinct, never collapsed into one failure):
  evidence_not_required     the claim needs no evidence (citations are still listed)
  no_citation               evidence required, no reference is associated with the claim
  association_unresolved    no reference is associated, but an ambiguous reference could belong to it
  invalid_citation          associated references exist, none resolves (malformed, not in context,
                            page not retrieved, unresolvable); the reasons are listed
  retrieval_context_missing associated well-formed references exist but no retrieval context was
                            recorded, so they cannot be resolved. A DATA gap, not an answer failure
  cited_evidence_supports   a resolved cited unit is judged "supports" for THIS claim, and none
                            "contradicts"
  cited_evidence_contradicts  a resolved cited unit is judged "contradicts", and none "supports"
  conflicting_evidence      cited units are judged both "supports" and "contradicts"
  cited_evidence_not_assessed  no cited unit supports or contradicts, and at least one resolved cited
                            unit has no judgement for this claim (it might support it)
  cited_evidence_insufficient  every resolved cited unit is judged, none supports/contradicts, and at
                            least one is "insufficient"
  cited_evidence_irrelevant every resolved cited unit is judged "irrelevant" for this claim
Relations are looked up for the exact unit a reference resolves to (same level, same id). A judgement
of a page is never applied to a chunk on that page, nor the reverse, and text similarity is never used.
When the cited unit is unjudged but a unit containing it or contained in it IS judged, a diagnostic
says so (so the annotator can judge the cited unit); the judgement is still not transferred.

The person's overall verdict (support_status) is reported beside the outcome, unchanged.

METRICS. Each is numerator / denominator with explicit statuses (Sprint 5 vocabulary):
  exact            fully determined
  bounded          some items cannot be decided (unjudged units, ambiguous association):
                   lower = decided successes / denominator, upper = (successes + undecided) / denominator
  not_applicable   the denominator is 0 (nothing to measure); never reported as 0
  not_evaluated    the input needed for the metric is missing (e.g. no retrieval context)

  citation_validity      well-formed references / (well-formed references + malformed markers)
                         All citation markers in the answer, Sources block included. No annotation needed.
  citation_resolution    resolved references / well-formed references. Needs the retrieval context.
  citation_precision     distinct (claim, cited unit) pairs, over associated resolved references, whose
                         unit is judged "supports" for that claim / all such distinct pairs. Repeating a
                         citation does not change it. Unjudged units -> bounded. Ambiguous, unassociated
                         and Sources-block references are outside the denominator; their counts are
                         given in the result. Needs the retrieval context (else not_evaluated).
  required_claim_citation_coverage   evidence-required claims with >= 1 associated resolved reference
                         / evidence-required claims. A claim with only ambiguous candidates -> bounded.
  required_claim_supported_coverage  evidence-required claims whose outcome is cited_evidence_supports
                         AND whose human verdict (support_status) is "supported" / evidence-required
                         claims. A cited supporting unit never overrides a person's verdict of
                         contradicted or insufficient. Undecided claims (outcome not_assessed,
                         association_unresolved or retrieval_context_missing, or a supporting citation
                         with verdict not_assessed) -> bounded.
Claim counts (not rates): outcomes, and the person's support_status (supported, contradicted,
insufficient = not supported by the evidence, not_assessed), for all claims and evidence-required claims.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from eval.evidence_eval.annotations import ClaimAnnotation
from eval.evidence_eval.resolution import AMBIGUOUS, RESOLVED, Association, CitationRef
from eval.retrieval_eval.metrics import BOUNDED, EXACT, NOT_APPLICABLE, NOT_EVALUATED

OUTCOMES = ("evidence_not_required", "no_citation", "association_unresolved", "invalid_citation",
            "retrieval_context_missing",
            "cited_evidence_supports", "cited_evidence_contradicts", "conflicting_evidence",
            "cited_evidence_not_assessed", "cited_evidence_insufficient", "cited_evidence_irrelevant")
METRIC_NAMES = ("citation_validity", "citation_resolution", "citation_precision",
                "required_claim_citation_coverage", "required_claim_supported_coverage")
ANNOTATION_DEPENDENT = {"citation_precision", "required_claim_citation_coverage", "required_claim_supported_coverage"}


@dataclass(frozen=True)
class Ratio:
    """numerator_lower <= true numerator <= numerator_upper, over denominator."""
    numerator_lower: int
    numerator_upper: int
    denominator: int
    evaluated: bool = True
    reason: str = ""

    @property
    def status(self) -> str:
        if not self.evaluated:
            return NOT_EVALUATED
        if self.denominator == 0:
            return NOT_APPLICABLE
        return EXACT if self.numerator_lower == self.numerator_upper else BOUNDED

    def to_dict(self) -> dict:
        st = self.status
        d = self.denominator
        lo = self.numerator_lower / d if d and self.evaluated else None
        hi = self.numerator_upper / d if d and self.evaluated else None
        return {"status": st, "value": lo if st == EXACT else None, "lower": lo, "upper": hi,
                "numerator_lower": self.numerator_lower, "numerator_upper": self.numerator_upper,
                "denominator": d, "reason": self.reason}


def add(ratios: Sequence[Ratio]) -> Ratio:
    """Micro-aggregate: sum numerators and denominators over the ratios that were evaluated."""
    ev = [r for r in ratios if r.evaluated]
    if not ev:
        return Ratio(0, 0, 0, evaluated=False, reason="not evaluated for any case")
    return Ratio(sum(r.numerator_lower for r in ev), sum(r.numerator_upper for r in ev),
                 sum(r.denominator for r in ev),
                 reason="" if len(ev) == len(ratios) else f"{len(ratios) - len(ev)} case(s) not evaluated")


@dataclass(frozen=True)
class ClaimResult:
    claim: ClaimAnnotation
    outcome: str
    refs: tuple[int, ...]                 # associated reference indexes
    resolved_units: tuple                 # (ref index, EvidenceRef, relation or None)
    ambiguous_refs: tuple[int, ...]
    invalid_reasons: tuple[str, ...]
    coverage_satisfied: bool | None       # None: evidence not required, or undecided (no context)
    diagnostics: tuple[str, ...]

    def to_dict(self, refs_by_index) -> dict:
        c = self.claim
        return {
            "claim_id": c.claim_id, "text": c.text, "span": {"start": c.start, "end": c.end},
            "requires_evidence": c.requires_evidence, "outcome": self.outcome,
            "human_support_status": c.support_status,
            "citations": [refs_by_index[i].marker for i in self.refs],
            "citation_indexes": list(self.refs),
            "resolved_evidence": [{"citation_index": i, "unit": u.to_dict(), "relation": rel or "not_assessed"}
                                  for i, u, rel in self.resolved_units],
            "ambiguous_citation_indexes": list(self.ambiguous_refs),
            "invalid_citation_reasons": list(self.invalid_reasons),
            "coverage_satisfied": self.coverage_satisfied,
            "expected_evidence": [r.to_dict() for r in c.expected_evidence],
            "diagnostics": list(self.diagnostics),
        }


def evaluate_claim(claim: ClaimAnnotation, refs: Sequence[CitationRef], assoc: Association,
                   ctx=None) -> ClaimResult:
    """Outcome for one annotated claim. ctx is the EvidenceContext, or None when none was recorded."""
    own = assoc.by_claim.get(claim.claim_id, [])
    ambiguous = tuple(r.index for r in refs
                      if assoc.by_ref.get(r.index, ("", ()))[0] == AMBIGUOUS and claim.claim_id in assoc.by_ref[r.index][1])
    resolved = tuple((i, refs[i].unit, claim.relation(refs[i].unit)) for i in own if refs[i].status == RESOLVED)
    invalid = tuple(f"{refs[i].marker}: {refs[i].status}" + (f" ({refs[i].detail})" if refs[i].detail else "")
                    for i in own if refs[i].status != RESOLVED)
    no_context_refs = [i for i in own if refs[i].status == "no_retrieval_context"]
    diags: list[str] = []
    rels = [rel for _, _, rel in resolved]
    if not claim.requires_evidence:
        outcome = "evidence_not_required"
    elif not own:
        outcome = "association_unresolved" if ambiguous else "no_citation"
    elif not resolved and no_context_refs:
        outcome = "retrieval_context_missing"
    elif not resolved:
        outcome = "invalid_citation"
    elif "supports" in rels and "contradicts" in rels:
        outcome = "conflicting_evidence"
    elif "contradicts" in rels:
        outcome = "cited_evidence_contradicts"
    elif "supports" in rels:
        outcome = "cited_evidence_supports"
    elif None in rels:
        outcome = "cited_evidence_not_assessed"
    elif "insufficient" in rels:
        outcome = "cited_evidence_insufficient"
    else:
        outcome = "cited_evidence_irrelevant"
    if invalid and resolved:
        diags.append(f"{len(invalid)} associated citation(s) did not resolve")
    dup = [u for u in {u for _, u, _ in resolved} if sum(1 for _, x, _ in resolved if x == u) > 1]
    if dup:
        diags.append(f"the same evidence unit is cited more than once: {sorted(u.id for u in dup)}")
    if ctx is not None:
        for _, unit, rel in resolved:
            if rel is not None:
                continue
            related = [(r, claim.relation(r)) for r in ctx.related_units(unit) if claim.relation(r) is not None]
            if related:
                diags.append(f"cited {unit.level} {unit.id!r} is not judged for this claim, but related unit(s) are: "
                             + ", ".join(f"{r.level} {r.id!r} ({rel_})" for r, rel_ in sorted(related))
                             + "; judgements are never transferred between units")
        missing = [r.id for r in claim.expected_evidence if not ctx.contains(r)]
        if missing:
            diags.append(f"expected evidence not in the retrieval context: {missing}")
        cited = {u for _, u, _ in resolved}
        uncited = [r.id for r in claim.expected_evidence if ctx.contains(r) and r not in cited]
        if uncited and claim.requires_evidence:
            diags.append(f"expected evidence was retrieved but not cited: {uncited}")
    if outcome == "cited_evidence_supports" and claim.support_status in ("contradicted", "insufficient"):
        diags.append(f"a cited unit supports the claim but the person's overall verdict is {claim.support_status!r}; "
                     f"the claim is not counted as supported")
    if outcome == "retrieval_context_missing":
        diags.append("no retrieval context was recorded, so the claim's citations cannot be resolved")
    if not claim.requires_evidence:
        coverage = None
    elif resolved:
        coverage = True
    elif outcome == "retrieval_context_missing":
        coverage = None      # undecided: shown as null, counted in the bounded range
    else:
        coverage = False
    return ClaimResult(claim, outcome, tuple(own), resolved, ambiguous, invalid, coverage, tuple(diags))


def is_supported(r: ClaimResult) -> bool:
    """Decided supported: a cited unit is judged to support the claim AND the person's verdict agrees."""
    return r.outcome == "cited_evidence_supports" and r.claim.support_status == "supported"


def may_be_supported(r: ClaimResult) -> bool:
    """Undecided: could turn out supported once judging, association or context is completed."""
    return (r.outcome in ("cited_evidence_not_assessed", "association_unresolved", "retrieval_context_missing")
            or (r.outcome == "cited_evidence_supports" and r.claim.support_status == "not_assessed"))


def citation_metrics(refs: Sequence[CitationRef], assoc: Association, claims_by_id, has_context: bool,
                     annotated: bool) -> dict[str, Ratio]:
    wellformed = [r for r in refs if r.status != "malformed"]
    validity = Ratio(len(wellformed), len(wellformed), len(refs))
    resolved = [r for r in wellformed if r.status == RESOLVED]
    resolution = (Ratio(len(resolved), len(resolved), len(wellformed)) if has_context
                  else Ratio(0, 0, len(wellformed), evaluated=False, reason="no recorded retrieval context"))
    if not annotated:
        precision = Ratio(0, 0, 0, evaluated=False, reason="no annotation for this answer")
    elif not has_context:
        precision = Ratio(0, 0, 0, evaluated=False, reason="no recorded retrieval context")
    else:
        assoc_resolved = [r for r in resolved if assoc.by_ref.get(r.index, ("", ()))[0] == "associated"]
        pairs = sorted({(assoc.by_ref[r.index][1][0], r.unit) for r in assoc_resolved})
        sup = unj = 0
        for cid, unit in pairs:
            rel = claims_by_id[cid].relation(unit)
            sup += rel == "supports"
            unj += rel is None
        body = [r for r in wellformed if not r.in_sources_block]
        excluded = Counter(assoc.by_ref.get(r.index, ("", ()))[0] for r in body
                           if assoc.by_ref.get(r.index, ("", ()))[0] != "associated")
        unresolved_assoc = sum(1 for r in wellformed if r.status != RESOLVED
                               and assoc.by_ref.get(r.index, ("", ()))[0] == "associated")
        parts = [f"{len(pairs)} distinct (claim, unit) pair(s) from {len(assoc_resolved)} associated resolved reference(s)"]
        parts += [f"{n} {state} reference(s) excluded" for state, n in sorted(excluded.items())]
        if unresolved_assoc:
            parts.append(f"{unresolved_assoc} associated reference(s) excluded because they did not resolve")
        precision = Ratio(sup, sup + unj, len(pairs), reason="; ".join(parts))
    return {"citation_validity": validity, "citation_resolution": resolution, "citation_precision": precision}


def claim_metrics(results: Sequence[ClaimResult]) -> dict[str, Ratio]:
    req = [r for r in results if r.claim.requires_evidence]
    covered = sum(1 for r in req if r.coverage_satisfied is True)
    maybe_covered = sum(1 for r in req if r.coverage_satisfied is not True and
                        (r.ambiguous_refs or r.outcome == "retrieval_context_missing"))
    supported = sum(1 for r in req if is_supported(r))
    maybe_sup = sum(1 for r in req if may_be_supported(r))
    return {
        "required_claim_citation_coverage": Ratio(covered, covered + maybe_covered, len(req)),
        "required_claim_supported_coverage": Ratio(supported, supported + maybe_sup, len(req)),
    }
