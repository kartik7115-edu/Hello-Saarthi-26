"""Citation resolution and citation-to-claim association (Sprint 6). Deterministic, never guessing.

Citations are found by the Sprint 2 parser (eval.answer_eval.citation_parser); this module does not
re-implement citation syntax. Each parsed citation becomes one or more CITATION REFERENCES: a numeric
marker "[2, 3]" gives two references (2 and 3) at the same position; every other marker gives one.
Malformed markers reported by the parser become references with status "malformed".

Resolution: a reference resolves only to evidence that is in the recorded retrieval context.
  numeric "[n]"     only when the answer record declares citation_numbering "retrieval_order": the
                    n-th hit (1-based). The cited unit is that hit's chunk, else its page, else its
                    document. n out of range, undeclared numbering or a context without one ordered
                    hit list -> "unresolvable".
  "[Title, p. N]"   the title must equal, after normalisation, the document id or the hit metadata
  "[Source: f, p.N]" title/source_file of a retrieved hit, and must point to exactly one document.
                    Normalisation: lower case, "_" -> space, a trailing ".pdf" removed, whitespace
                    collapsed. No partial or fuzzy matching. The page must also be in the context:
                    otherwise "page_not_retrieved" (the document was retrieved, the page was not).
  "[Title]", "[Source: f]"  -> the document. "[Video: T, 14:22]" -> the document (the timestamp is
                    reported but not checked; the context has no timestamps).
  A title that matches nothing retrieved -> "not_in_context" (valid syntax, evidence not available).
  More than one document for a title -> "unresolvable" (ambiguous).
  No retrieval context at all -> "no_retrieval_context".

Statuses: resolved | page_not_retrieved | not_in_context | unresolvable | no_retrieval_context |
malformed. Only "resolved" references count as resolved anywhere.

Association with annotated claims (only references in the answer body; references in a trailing
"Sources" block, which the application appends, are never associated with a claim):
  explicit    a claim with "citation_starts" owns exactly the references starting at those offsets.
              An offset that matches no citation is a data error.
  positional  every other body reference is attached to a claim WITHOUT explicit starts when it lies
              inside the claim's span, or directly after it with only whitespace, closing punctuation
              (. , ; : ! ? ) ]) and other citation markers in between.
              One candidate -> associated; none -> "unassociated"; several -> "ambiguous" (reported as
              unresolved, never split or guessed).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from eval.answer_eval.citation_parser import parse_citations
from eval.evidence_eval.annotations import ClaimAnnotation, EvidenceRef
from eval.retrieval_eval.cases import page_id
from eval.retrieval_eval.records import RecordedResult

RESOLVED = "resolved"
RESOLUTION_STATUSES = (RESOLVED, "page_not_retrieved", "not_in_context", "unresolvable",
                       "no_retrieval_context", "malformed")
ASSOCIATED, UNASSOCIATED, AMBIGUOUS, SOURCES_BLOCK = "associated", "unassociated", "ambiguous", "sources_block"
_GAP_OK = re.compile(r"^[\s.,;:!?)\]]*$")


def normalise_name(name: str) -> str:
    s = name.strip().lower().replace("_", " ")
    s = re.sub(r"\.pdf$", "", s)
    return " ".join(s.split())


@dataclass(frozen=True)
class EvidenceContext:
    """The recorded retrieval context an answer was generated from."""
    hits: tuple | None                      # ordered Sprint 5 hits, or None if only id lists were given
    documents: frozenset[str]
    pages: frozenset[str]
    chunks: frozenset[str]
    names: Mapping[str, frozenset[str]]     # normalised name -> document ids it can denote
    related: Mapping[EvidenceRef, frozenset] = field(default_factory=dict)  # unit -> units containing it or in it

    @classmethod
    def from_result(cls, result: RecordedResult) -> "EvidenceContext":
        docs, pages, chunks = set(), set(), set()
        names: dict[str, set[str]] = {}
        related: dict[EvidenceRef, set[EvidenceRef]] = {}

        def link(a, b):
            related.setdefault(a, set()).add(b)
            related.setdefault(b, set()).add(a)

        def name(n, doc):
            if isinstance(n, str) and n.strip():
                names.setdefault(normalise_name(n), set()).add(doc)

        if result.hits is not None:
            for h in result.hits:
                if h.document_id:
                    docs.add(h.document_id)
                    name(h.document_id, h.document_id)
                    name(h.metadata.get("title"), h.document_id)
                    name(h.metadata.get("source_file"), h.document_id)
                    doc_u = EvidenceRef("document", h.document_id)
                    if h.page is not None:
                        pages.add(page_id(h.document_id, h.page))
                        page_u = EvidenceRef("page", page_id(h.document_id, h.page))
                        link(doc_u, page_u)
                        if h.chunk_id:
                            link(page_u, EvidenceRef("chunk", h.chunk_id))
                    if h.chunk_id:
                        link(doc_u, EvidenceRef("chunk", h.chunk_id))
                if h.chunk_id:
                    chunks.add(h.chunk_id)
            hits = tuple(result.hits)
        else:
            for d in result.document_ids or ():
                if d:
                    docs.add(d)
                    name(d, d)
            chunks |= {c for c in (result.chunk_ids or ()) if c}
            hits = None
        return cls(hits, frozenset(docs), frozenset(pages), frozenset(chunks),
                   {k: frozenset(v) for k, v in names.items()},
                   {k: frozenset(v) for k, v in related.items()})

    def related_units(self, unit: EvidenceRef) -> frozenset:
        """Units that contain this unit or lie inside it, as recorded in the retrieval hits. Used only
        for diagnostics: a judgement is never transferred from one unit to another."""
        return self.related.get(unit, frozenset())

    def contains(self, unit: EvidenceRef) -> bool:
        return unit.id in {"document": self.documents, "page": self.pages, "chunk": self.chunks}[unit.level]

    def known_titles(self) -> list[str]:
        out = set(self.documents)
        for h in self.hits or ():
            for k in ("title", "source_file"):
                v = h.metadata.get(k)
                if isinstance(v, str) and v.strip():
                    out.add(v)
        return sorted(out)


@dataclass(frozen=True)
class CitationRef:
    index: int                    # 0-based position in the answer's reference list
    marker: str                   # the citation text as written
    kind: str                     # parser kind, or "malformed"
    start: int
    end: int
    in_sources_block: bool
    number: int | None = None
    title: str | None = None
    page: str | None = None
    status: str = ""
    unit: EvidenceRef | None = None
    detail: str = ""
    key: str = ""                 # parser key (+ number) for counting repeats

    def to_dict(self) -> dict:
        return {"index": self.index, "marker": self.marker, "kind": self.kind, "start": self.start, "end": self.end,
                "in_sources_block": self.in_sources_block, "number": self.number, "title": self.title,
                "page": self.page, "status": self.status, "unit": self.unit.to_dict() if self.unit else None,
                "detail": self.detail}


def _hit_unit(hit) -> EvidenceRef | None:
    if hit.chunk_id:
        return EvidenceRef("chunk", hit.chunk_id)
    if hit.document_id and hit.page is not None:
        return EvidenceRef("page", page_id(hit.document_id, hit.page))
    if hit.document_id:
        return EvidenceRef("document", hit.document_id)
    return None


def _resolve_numeric(n: int, numbering: str, ctx: EvidenceContext | None) -> tuple[str, EvidenceRef | None, str]:
    if ctx is None:
        return "no_retrieval_context", None, "no recorded retrieval context for this answer"
    if numbering != "retrieval_order":
        return "unresolvable", None, "the answer record does not declare what numeric citations refer to"
    if ctx.hits is None:
        return "unresolvable", None, "the retrieval context has no single ordered hit list"
    if not 1 <= n <= len(ctx.hits):
        return "unresolvable", None, f"[{n}] is outside the {len(ctx.hits)} retrieved hit(s)"
    unit = _hit_unit(ctx.hits[n - 1])
    if unit is None:
        return "unresolvable", None, f"hit {n} has no document, page or chunk id"
    return RESOLVED, unit, f"hit {n}"


def _resolve_label(title: str, page: str | None, ctx: EvidenceContext | None) -> tuple[str, EvidenceRef | None, str]:
    if ctx is None:
        return "no_retrieval_context", None, "no recorded retrieval context for this answer"
    docs = ctx.names.get(normalise_name(title), frozenset())
    if not docs:
        return "not_in_context", None, f"no retrieved document is named {title!r}"
    if len(docs) > 1:
        return "unresolvable", None, f"{title!r} names {len(docs)} retrieved documents: {sorted(docs)}"
    doc = next(iter(docs))
    if page is None:
        return RESOLVED, EvidenceRef("document", doc), ""
    if not page.isdigit() or int(page) < 1:
        return "unresolvable", None, f"page {page!r} is not a page number"
    pid = page_id(doc, int(page))
    if pid in ctx.pages:
        return RESOLVED, EvidenceRef("page", pid), ""
    return "page_not_retrieved", None, f"{doc!r} was retrieved but page {page} was not"


def resolve_citations(answer: str, numbering: str, ctx: EvidenceContext | None) -> list[CitationRef]:
    """Every citation reference in the answer, in order of position, with its resolution."""
    parsed = parse_citations(answer, known_titles=ctx.known_titles() if ctx else None)
    refs: list[CitationRef] = []
    items = [(c.start, 0, c) for c in parsed.citations] + [(m.start, 1, m) for m in parsed.malformed]
    for _, is_bad, c in sorted(items, key=lambda t: (t[0], t[1])):
        if is_bad:
            in_block = parsed.has_sources_block and any(x.in_sources_block and x.start <= c.start for x in parsed.citations)
            refs.append(CitationRef(len(refs), c.text, "malformed", c.start, c.end, in_block,
                                    status="malformed", detail=c.reason, key="malformed:" + c.text))
            continue
        if c.kind == "numeric":
            for n in c.numbers:
                status, unit, detail = _resolve_numeric(n, numbering, ctx)
                refs.append(CitationRef(len(refs), c.text, c.kind, c.start, c.end, c.in_sources_block, number=n,
                                        status=status, unit=unit, detail=detail, key=f"n:{n}"))
            continue
        page = c.page if c.kind in ("page_label", "source_label") else None
        status, unit, detail = _resolve_label(c.title or "", page, ctx)
        if c.kind == "video_label" and status == RESOLVED:
            detail = f"timestamp {c.timestamp} not checked (the context has no timestamps)"
        refs.append(CitationRef(len(refs), c.text, c.kind, c.start, c.end, c.in_sources_block, title=c.title,
                                page=c.page, status=status, unit=unit, detail=detail, key=c.key))
    return refs


@dataclass
class Association:
    by_ref: dict[int, tuple[str, tuple[str, ...]]] = field(default_factory=dict)   # index -> (state, claim ids)
    by_claim: dict[str, list[int]] = field(default_factory=dict)                   # claim id -> ref indexes
    errors: list[str] = field(default_factory=list)


def associate(answer: str, refs: Sequence[CitationRef], claims: Sequence[ClaimAnnotation]) -> Association:
    out = Association(by_claim={c.claim_id: [] for c in claims})
    explicit_owner: dict[int, str] = {}
    for claim in claims:
        if claim.citation_starts is None:
            continue
        for s in claim.citation_starts:
            hits = [r for r in refs if r.start == s]
            if not hits:
                out.errors.append(f"claim {claim.claim_id!r}: citation_starts offset {s} matches no citation")
                continue
            for r in hits:
                if r.in_sources_block:
                    out.errors.append(f"claim {claim.claim_id!r}: offset {s} is in the Sources block, which "
                                      f"is never associated with a claim")
                    continue
                explicit_owner[r.index] = claim.claim_id
    masked = list(answer)
    for r in refs:
        for i in range(r.start, min(r.end, len(masked))):
            masked[i] = " "
    masked_text = "".join(masked)
    positional = [c for c in claims if c.citation_starts is None]
    for r in refs:
        if r.in_sources_block:
            out.by_ref[r.index] = (SOURCES_BLOCK, ())
            continue
        if r.index in explicit_owner:
            cid = explicit_owner[r.index]
            out.by_ref[r.index] = (ASSOCIATED, (cid,))
            out.by_claim[cid].append(r.index)
            continue
        cands = tuple(c.claim_id for c in positional
                      if c.start <= r.start < c.end or (c.end <= r.start and _GAP_OK.match(masked_text[c.end:r.start])))
        if len(cands) == 1:
            out.by_ref[r.index] = (ASSOCIATED, cands)
            out.by_claim[cands[0]].append(r.index)
        elif not cands:
            out.by_ref[r.index] = (UNASSOCIATED, ())
        else:
            out.by_ref[r.index] = (AMBIGUOUS, cands)
    return out
