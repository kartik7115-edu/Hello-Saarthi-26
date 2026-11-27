"""Human claim/evidence annotations for recorded answers (Sprint 6). Strict, versioned schema.

An annotation file is JSON Lines; each line annotates ONE recorded answer:

  {"schema_version": "evidence-annotation/1",
   "annotation_id": "ann-df-01-hybrid",
   "run_id": "saarthi-hybrid-k10", "case_id": "df-01",
   "question": "...", "language": "en",
   "answer_sha256": "<sha256 of the exact recorded answer text, UTF-8>",
   "annotation": {"status": "verified", "version": 1, "annotator": "kartik",
                  "reviewer": "reviewer-2", "reviewed_on": "2026-11-20", "notes": "..."},
   "claims": [
     {"claim_id": "c1", "text": "The Nyquist rate is twice the highest frequency",
      "span": {"start": 0, "end": 47},
      "requires_evidence": true,
      "expected_evidence": [{"level": "page", "id": "EE210 Module 1 Signals and Sampling#p3"}],
      "evidence_judgements": [{"evidence": {"level": "chunk", "id": "c-17"}, "relation": "supports"}],
      "support_status": "supported",
      "citation_starts": [48]}            # optional explicit association (see resolution.py)
   ],
   "metadata": {...}}

Evidence references are {"level": "document" | "page" | "chunk", "id": ...}; page ids are
"<document id>#p<page>" (the Sprint 5 convention).

Relations a person can record between ONE claim and ONE evidence unit:
  supports      the unit, on its own, supports the claim
  contradicts   the unit contradicts the claim
  insufficient  the unit is on topic but does not establish the claim
  irrelevant    the unit has nothing to do with the claim
A unit with no judgement for a claim is NOT ASSESSED for that claim. It is never treated as irrelevant.

support_status (the person's overall verdict for the claim): supported | contradicted | insufficient |
not_assessed. "supported" needs at least one "supports" judgement and "contradicted" at least one
"contradicts" judgement, so the verdict is always traceable to a judged unit.

annotation.status
  verified    reviewed by a reviewer who is not the annotator, with a non-future ISO reviewed_on
  unverified  written but not reviewed; results are provisional and reported separately
  fixture     example data for tests and documentation only; never ground truth
Nothing promotes an annotation from unverified or fixture to verified except editing this record.

Validation rejects: unknown fields, a wrong schema_version, duplicate annotation ids, duplicate claim
ids, duplicate judgements of one unit for one claim, spans outside the answer or whose text differs
from the claim text (checked once the recorded answer is joined), malformed evidence references,
future review dates and the other rules above.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from eval.retrieval_eval.cases import ID_RE, LANGUAGES, PAGE_ID_RE

SCHEMA_VERSION = "evidence-annotation/1"
ANNOTATION_STATUSES = ("verified", "unverified", "fixture")
LEVELS = ("document", "page", "chunk")
RELATIONS = ("supports", "contradicts", "insufficient", "irrelevant")
SUPPORT_STATUSES = ("supported", "contradicted", "insufficient", "not_assessed")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_TOP_KEYS = {"schema_version", "annotation_id", "run_id", "case_id", "question", "language",
             "answer_sha256", "annotation", "claims"}
_TOP_OPTIONAL = {"metadata"}
_PROV_KEYS = {"status", "version", "annotator"}
_PROV_OPTIONAL = {"reviewer", "reviewed_on", "notes"}
_CLAIM_KEYS = {"claim_id", "text", "span", "requires_evidence", "evidence_judgements", "support_status"}
_CLAIM_OPTIONAL = {"expected_evidence", "citation_starts", "notes"}


@dataclass(frozen=True, order=True)
class EvidenceRef:
    level: str
    id: str

    def to_dict(self) -> dict:
        return {"level": self.level, "id": self.id}


@dataclass(frozen=True)
class Provenance:
    status: str
    version: int
    annotator: str
    reviewer: str | None = None
    reviewed_on: str | None = None
    notes: str = ""

    def to_dict(self) -> dict:
        return {"status": self.status, "version": self.version, "annotator": self.annotator,
                "reviewer": self.reviewer, "reviewed_on": self.reviewed_on, "notes": self.notes}


@dataclass(frozen=True)
class ClaimAnnotation:
    claim_id: str
    text: str
    start: int
    end: int
    requires_evidence: bool
    judgements: Mapping[EvidenceRef, str]          # unit -> relation; unlisted units are not assessed
    support_status: str
    expected_evidence: tuple[EvidenceRef, ...] = ()
    citation_starts: tuple[int, ...] | None = None  # None = positional association
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "judgements", MappingProxyType(dict(self.judgements)))

    def relation(self, unit: EvidenceRef) -> str | None:
        """The judged relation of exactly this unit to this claim, or None (not assessed)."""
        return self.judgements.get(unit)


@dataclass(frozen=True)
class AnswerAnnotation:
    annotation_id: str
    run_id: str
    case_id: str
    question: str
    language: str
    answer_sha256: str
    provenance: Provenance
    claims: tuple[ClaimAnnotation, ...]
    where: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def _is_iso_date(v: Any) -> bool:
    if not isinstance(v, str):
        return False
    try:
        return _dt.date.fromisoformat(v).isoformat() == v
    except ValueError:
        return False


def _nonempty_str(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def parse_ref(raw: Any, w: str) -> tuple[EvidenceRef | None, list[str]]:
    if not isinstance(raw, dict) or set(raw) != {"level", "id"}:
        return None, [f"{w}: an evidence reference must be exactly {{\"level\", \"id\"}}"]
    level, rid = raw["level"], raw["id"]
    errs = []
    if level not in LEVELS:
        errs.append(f"{w}: level must be one of {LEVELS}")
    if not _nonempty_str(rid):
        errs.append(f"{w}: id must be a non-empty string")
    elif level == "page" and not PAGE_ID_RE.match(rid):
        errs.append(f"{w}: page id {rid!r} must look like '<document id>#p<page>'")
    return (None, errs) if errs else (EvidenceRef(level, rid), [])


def _parse_provenance(raw: Any, w: str, today: _dt.date) -> tuple[Provenance | None, list[str]]:
    if not isinstance(raw, dict):
        return None, [f"{w}: annotation must be an object"]
    missing, extra = _PROV_KEYS - set(raw), set(raw) - _PROV_KEYS - _PROV_OPTIONAL
    errs = ([f"{w}: annotation missing field(s) {sorted(missing)}"] if missing else []) + \
           ([f"{w}: annotation has unknown field(s) {sorted(extra)}"] if extra else [])
    if errs:
        return None, errs
    status, version, annotator = raw["status"], raw["version"], raw["annotator"]
    reviewer, reviewed_on, notes = raw.get("reviewer"), raw.get("reviewed_on"), raw.get("notes", "")
    if status not in ANNOTATION_STATUSES:
        errs.append(f"{w}: annotation.status must be one of {ANNOTATION_STATUSES}")
    if not _is_int(version) or version < 1:
        errs.append(f"{w}: annotation.version must be an integer >= 1")
    if not _nonempty_str(annotator):
        errs.append(f"{w}: annotation.annotator must be a non-empty string")
    if not isinstance(notes, str):
        errs.append(f"{w}: annotation.notes must be a string")
    if status == "verified":
        if not _nonempty_str(reviewer):
            errs.append(f"{w}: verified annotations need a reviewer")
        elif isinstance(annotator, str) and reviewer.strip() == annotator.strip():
            errs.append(f"{w}: the reviewer of a verified annotation must not be its annotator")
        if not _is_iso_date(reviewed_on):
            errs.append(f"{w}: verified annotations need reviewed_on as an ISO date (YYYY-MM-DD)")
        elif _dt.date.fromisoformat(reviewed_on) > today:
            errs.append(f"{w}: reviewed_on {reviewed_on} is in the future")
    elif reviewer is not None or reviewed_on is not None:
        errs.append(f"{w}: reviewer/reviewed_on are only allowed on verified annotations")
    if errs:
        return None, errs
    return Provenance(status, version, annotator, reviewer, reviewed_on, notes), []


def _parse_claim(raw: Any, w: str) -> tuple[ClaimAnnotation | None, list[str]]:
    if not isinstance(raw, dict):
        return None, [f"{w}: a claim must be an object"]
    missing, extra = _CLAIM_KEYS - set(raw), set(raw) - _CLAIM_KEYS - _CLAIM_OPTIONAL
    errs = ([f"{w}: claim missing field(s) {sorted(missing)}"] if missing else []) + \
           ([f"{w}: claim has unknown field(s) {sorted(extra)}"] if extra else [])
    if errs:
        return None, errs
    cid = raw["claim_id"]
    if not isinstance(cid, str) or not ID_RE.match(cid):
        errs.append(f"{w}: claim_id must match {ID_RE.pattern}")
    w = f"{w} [{cid}]" if isinstance(cid, str) else w
    if not _nonempty_str(raw["text"]):
        errs.append(f"{w}: text must be a non-empty string")
    span = raw["span"]
    if not isinstance(span, dict) or set(span) != {"start", "end"} or not all(_is_int(span.get(k)) for k in ("start", "end")):
        errs.append(f"{w}: span must be exactly {{\"start\": int, \"end\": int}}")
        start = end = -1
    else:
        start, end = span["start"], span["end"]
        if not 0 <= start < end:
            errs.append(f"{w}: span must satisfy 0 <= start < end")
    if not isinstance(raw["requires_evidence"], bool):
        errs.append(f"{w}: requires_evidence must be true or false")
    if raw["support_status"] not in SUPPORT_STATUSES:
        errs.append(f"{w}: support_status must be one of {SUPPORT_STATUSES}")
    judgements: dict[EvidenceRef, str] = {}
    if not isinstance(raw["evidence_judgements"], list):
        errs.append(f"{w}: evidence_judgements must be a list")
    else:
        for i, j in enumerate(raw["evidence_judgements"], 1):
            jw = f"{w} judgement {i}"
            if not isinstance(j, dict) or set(j) != {"evidence", "relation"}:
                errs.append(f"{jw}: must be exactly {{\"evidence\", \"relation\"}}")
                continue
            ref, e = parse_ref(j["evidence"], jw)
            errs += e
            if j["relation"] not in RELATIONS:
                errs.append(f"{jw}: relation must be one of {RELATIONS}")
                continue
            if ref is not None:
                if ref in judgements:
                    errs.append(f"{jw}: {ref.level} {ref.id!r} is judged more than once for this claim")
                judgements[ref] = j["relation"]
    expected: list[EvidenceRef] = []
    exp_raw = raw.get("expected_evidence", [])
    if not isinstance(exp_raw, list):
        errs.append(f"{w}: expected_evidence must be a list")
    else:
        for i, r in enumerate(exp_raw, 1):
            ref, e = parse_ref(r, f"{w} expected_evidence {i}")
            errs += e
            if ref is not None:
                if ref in expected:
                    errs.append(f"{w}: expected_evidence lists {ref.level} {ref.id!r} twice")
                expected.append(ref)
    starts = raw.get("citation_starts")
    if starts is not None:
        if not isinstance(starts, list) or not all(_is_int(s) and s >= 0 for s in starts):
            errs.append(f"{w}: citation_starts must be a list of character offsets >= 0")
        elif len(set(starts)) != len(starts):
            errs.append(f"{w}: citation_starts lists an offset twice")
    if not isinstance(raw.get("notes", ""), str):
        errs.append(f"{w}: notes must be a string")
    status = raw["support_status"]
    if status == "supported" and "supports" not in judgements.values():
        errs.append(f"{w}: support_status 'supported' needs at least one 'supports' judgement")
    if status == "contradicted" and "contradicts" not in judgements.values():
        errs.append(f"{w}: support_status 'contradicted' needs at least one 'contradicts' judgement")
    if errs:
        return None, errs
    return ClaimAnnotation(cid, raw["text"], start, end, raw["requires_evidence"], judgements, status,
                           tuple(expected), tuple(starts) if starts is not None else None,
                           raw.get("notes", "")), []


def parse_annotation(raw: Any, where: str = "annotation", *, today: _dt.date | None = None) -> tuple[AnswerAnnotation | None, list[str]]:
    today = today or _dt.date.today()
    if not isinstance(raw, dict):
        return None, [f"{where}: must be a JSON object"]
    missing, extra = _TOP_KEYS - set(raw), set(raw) - _TOP_KEYS - _TOP_OPTIONAL
    errs = ([f"{where}: missing field(s) {sorted(missing)}"] if missing else []) + \
           ([f"{where}: unknown field(s) {sorted(extra)}"] if extra else [])
    if errs:
        return None, errs
    if raw["schema_version"] != SCHEMA_VERSION:
        errs.append(f"{where}: schema_version must be {SCHEMA_VERSION!r}")
    aid = raw["annotation_id"]
    if not isinstance(aid, str) or not ID_RE.match(aid):
        errs.append(f"{where}: annotation_id must match {ID_RE.pattern}")
    w = f"{where} [{aid}]" if isinstance(aid, str) else where
    for key in ("run_id", "case_id"):
        if not isinstance(raw[key], str) or not ID_RE.match(raw[key]):
            errs.append(f"{w}: {key} must match {ID_RE.pattern}")
    if not _nonempty_str(raw["question"]):
        errs.append(f"{w}: question must be a non-empty string")
    if raw["language"] not in LANGUAGES:
        errs.append(f"{w}: language must be one of {LANGUAGES} (English/Hinglish scope)")
    if not isinstance(raw["answer_sha256"], str) or not SHA256_RE.match(raw["answer_sha256"]):
        errs.append(f"{w}: answer_sha256 must be 64 lower-case hex characters")
    if not isinstance(raw.get("metadata", {}), dict):
        errs.append(f"{w}: metadata must be an object")
    prov, e = _parse_provenance(raw["annotation"], w, today)
    errs += e
    claims: list[ClaimAnnotation] = []
    if not isinstance(raw["claims"], list):
        errs.append(f"{w}: claims must be a list")
    else:
        seen: set[str] = set()
        for i, c in enumerate(raw["claims"], 1):
            claim, e = _parse_claim(c, f"{w} claim {i}")
            errs += e
            if claim is None:
                continue
            if claim.claim_id in seen:
                errs.append(f"{w}: duplicate claim_id {claim.claim_id!r}")
                continue
            seen.add(claim.claim_id)
            claims.append(claim)
        spans = sorted((c.start, c.end, c.claim_id) for c in claims)
        for (s1, e1, c1), (s2, e2, c2) in zip(spans, spans[1:]):
            if s2 < e1:
                errs.append(f"{w}: claims {c1!r} and {c2!r} have overlapping spans")
        all_starts = [s for c in claims if c.citation_starts for s in c.citation_starts]
        if len(all_starts) != len(set(all_starts)):
            errs.append(f"{w}: the same citation offset is assigned to more than one claim")
    if errs:
        return None, errs
    return AnswerAnnotation(aid, raw["run_id"], raw["case_id"], raw["question"], raw["language"],
                            raw["answer_sha256"], prov, tuple(claims), where, raw.get("metadata", {})), []


def parse_annotations(rows: Iterable[tuple[str, Any]], *, today: _dt.date | None = None) -> tuple[list[AnswerAnnotation], list[str]]:
    out: list[AnswerAnnotation] = []
    errors: list[str] = []
    ids: dict[str, str] = {}
    pairs: dict[tuple[str, str], str] = {}
    for where, raw in rows:
        ann, errs = parse_annotation(raw, where, today=today)
        errors += errs
        if ann is None:
            continue
        if ann.annotation_id in ids:
            errors.append(f"{where}: duplicate annotation_id {ann.annotation_id!r} (first at {ids[ann.annotation_id]})")
            continue
        key = (ann.run_id, ann.case_id)
        if key in pairs:
            errors.append(f"{where}: a second annotation for run {key[0]!r} case {key[1]!r} (first at {pairs[key]})")
            continue
        ids[ann.annotation_id] = where
        pairs[key] = where
        out.append(ann)
    return out, errors


def load_annotations(path: Path | str, *, today: _dt.date | None = None) -> tuple[list[AnswerAnnotation], list[str]]:
    rows, errors = [], []
    p = Path(path)
    for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append((f"{p.name}:{n}", json.loads(line)))
        except json.JSONDecodeError as e:
            errors.append(f"{p.name}:{n}: invalid JSON ({e})")
    anns, errs = parse_annotations(rows, today=today)
    return anns, errors + errs
