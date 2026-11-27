"""Recorded answers and their retrieval context (Sprint 6). What the system produced, nothing more.

An answers file is JSON Lines with two record types.

Run header (one per run_id):
  {"record_type": "run", "run_id": "saarthi-hybrid-k10", "provenance": "recorded",
   "description": "...", "generator": {...}, "metadata": {...}}
  provenance  recorded (captured from a real system) | fixture (illustrative example data)

Answer (at most one per run_id and case_id):
  {"record_type": "answer", "run_id": "saarthi-hybrid-k10", "case_id": "df-01",
   "question": "...", "answer": "The Nyquist rate is ... [1]",
   "citation_numbering": "retrieval_order",
   "retrieval": {"hits": [{"document_id": "...", "page": 3, "chunk_id": "c-17", ...}, ...]},
   "metadata": {...}}

  citation_numbering  how numeric citations such as [1] are meant:
      retrieval_order  [n] is the n-th hit of the retrieval context (1-based), as in Saarthi's
                       knowledge-base prompt, which numbers the excerpts it gives the model
      unspecified      numeric citations cannot be resolved (they are reported, never guessed)
  retrieval  the retrieval context the answer was generated from, in Sprint 5's recorded-result
             format (hits, or document_ids/chunk_ids lists), validated by eval.retrieval_eval.records.
             It may instead come from a Sprint 5 results file with the same run_id and case_id;
             giving both is an error. With neither, citations cannot be resolved.

Leakage guard: an answer record may not carry annotation fields (claims, judgements, support,
expected evidence, ...): the schema is strict, and RESERVED_ANNOTATION_KEYS are also rejected anywhere
inside run, generator, answer, retrieval and hit metadata. A Sprint 5 results file used as a context is
checked the same way by the evaluator. Nothing in this package passes annotations to any generator or
retriever.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from eval.retrieval_eval.cases import ID_RE
from eval.retrieval_eval.records import PROVENANCES, RecordedResult, parse_records

NUMBERING = ("retrieval_order", "unspecified")
_RUN_KEYS = {"record_type", "run_id", "provenance"}
_RUN_OPTIONAL = {"description", "generator", "metadata"}
_ANS_KEYS = {"record_type", "run_id", "case_id", "question", "answer", "citation_numbering"}
_ANS_OPTIONAL = {"retrieval", "metadata"}
_RETRIEVAL_KEYS = {"hits", "document_ids", "chunk_ids", "metadata"}


# Field names that belong to annotations. They may not appear ANYWHERE in recorded inputs, including
# free-form metadata, so ground truth cannot travel with what the system produced.
RESERVED_ANNOTATION_KEYS = frozenset({
    "annotation", "annotations", "annotation_id", "claims", "claim_id", "claim_ids", "support_status",
    "human_support_status", "evidence_judgements", "judgement", "judgements", "relation", "relations",
    "expected_evidence", "supports", "contradicts"})


def annotation_keys_in(obj: Any, path: str = "") -> list[str]:
    """Paths of reserved annotation keys found anywhere inside obj (dicts and lists, recursively)."""
    found: list[str] = []
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            here = f"{path}.{k}" if path else str(k)
            if k in RESERVED_ANNOTATION_KEYS:
                found.append(here)
            found += annotation_keys_in(v, here)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            found += annotation_keys_in(v, f"{path}[{i}]")
    return found


def _leak_error(where: str, obj: Any, label: str) -> list[str]:
    keys = annotation_keys_in(obj)
    return [f"{where}: {label} contains annotation field(s) {keys}; recorded inputs may not carry ground "
            f"truth (annotations belong in the annotation file)"] if keys else []


def answer_sha256(answer: str) -> str:
    return hashlib.sha256(answer.encode("utf-8", "surrogatepass")).hexdigest()


@dataclass(frozen=True)
class AnswerRun:
    run_id: str
    provenance: str
    description: str = ""
    generator: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"run_id": self.run_id, "provenance": self.provenance, "description": self.description,
                "generator": dict(self.generator), "metadata": dict(self.metadata)}


@dataclass(frozen=True)
class RecordedAnswer:
    run_id: str
    case_id: str
    question: str
    answer: str
    citation_numbering: str
    where: str
    retrieval: RecordedResult | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def sha256(self) -> str:
        return answer_sha256(self.answer)


@dataclass
class AnswerSet:
    runs: dict[str, AnswerRun] = field(default_factory=dict)
    answers: dict[tuple[str, str], RecordedAnswer] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def _parse_retrieval(raw: Any, run_id: str, case_id: str, provenance: str, w: str) -> tuple[RecordedResult | None, list[str]]:
    if not isinstance(raw, dict):
        return None, [f"{w}: retrieval must be an object"]
    extra = set(raw) - _RETRIEVAL_KEYS
    if extra:
        return None, [f"{w}: retrieval has unknown field(s) {sorted(extra)}"]
    rs = parse_records([(w, {"record_type": "run", "run_id": run_id, "provenance": provenance}),
                        (w, {"record_type": "result", "run_id": run_id, "case_id": case_id, **raw})])
    if rs.errors:
        return None, [f"{w}: retrieval: {e.split(': ', 1)[-1]}" for e in rs.errors]
    return rs.results[(run_id, case_id)], []


def parse_answers(rows: Iterable[tuple[str, Any]]) -> AnswerSet:
    out = AnswerSet()
    pending: list[tuple[str, dict]] = []
    for where, raw in rows:
        if not isinstance(raw, dict):
            out.errors.append(f"{where}: must be a JSON object")
            continue
        rtype = raw.get("record_type")
        if rtype == "run":
            missing, extra = _RUN_KEYS - set(raw), set(raw) - _RUN_KEYS - _RUN_OPTIONAL
            errs = ([f"{where}: run header missing field(s) {sorted(missing)}"] if missing else []) + \
                   ([f"{where}: run header has unknown field(s) {sorted(extra)}"] if extra else [])
            if not errs:
                if not isinstance(raw["run_id"], str) or not ID_RE.match(raw["run_id"]):
                    errs.append(f"{where}: run_id must match {ID_RE.pattern}")
                if raw["provenance"] not in PROVENANCES:
                    errs.append(f"{where}: provenance must be one of {PROVENANCES}")
                if not isinstance(raw.get("description", ""), str):
                    errs.append(f"{where}: description must be a string")
                for key in ("generator", "metadata"):
                    if not isinstance(raw.get(key, {}), dict):
                        errs.append(f"{where}: {key} must be an object")
                    else:
                        errs += _leak_error(where, raw.get(key, {}), f"run {key}")
            if not errs and raw["run_id"] in out.runs:
                errs.append(f"{where}: duplicate run header for {raw['run_id']!r}")
            if errs:
                out.errors += errs
                continue
            out.runs[raw["run_id"]] = AnswerRun(raw["run_id"], raw["provenance"], raw.get("description", ""),
                                                MappingProxyType(dict(raw.get("generator", {}))),
                                                MappingProxyType(dict(raw.get("metadata", {}))))
        elif rtype == "answer":
            pending.append((where, raw))
        else:
            out.errors.append(f"{where}: record_type must be 'run' or 'answer'")
    for where, raw in pending:
        missing, extra = _ANS_KEYS - set(raw), set(raw) - _ANS_KEYS - _ANS_OPTIONAL
        if extra:
            out.errors.append(f"{where}: answer has unknown field(s) {sorted(extra)} (answer records carry only "
                              f"what the system produced; annotations belong in the annotation file)")
            continue
        if missing:
            out.errors.append(f"{where}: answer missing field(s) {sorted(missing)}")
            continue
        run_id, case_id = raw["run_id"], raw["case_id"]
        errs = []
        for key in ("run_id", "case_id"):
            if not isinstance(raw[key], str) or not ID_RE.match(raw[key]):
                errs.append(f"{where}: {key} must match {ID_RE.pattern}")
        if not isinstance(raw["question"], str) or not raw["question"].strip():
            errs.append(f"{where}: question must be a non-empty string")
        if not isinstance(raw["answer"], str):
            errs.append(f"{where}: answer must be a string (an empty string is allowed)")
        if raw["citation_numbering"] not in NUMBERING:
            errs.append(f"{where}: citation_numbering must be one of {NUMBERING}")
        if not isinstance(raw.get("metadata", {}), dict):
            errs.append(f"{where}: metadata must be an object")
        else:
            errs += _leak_error(where, raw.get("metadata", {}), "answer metadata")
        if "retrieval" in raw:
            errs += _leak_error(where, raw["retrieval"], "retrieval")
        if not errs and run_id not in out.runs:
            errs.append(f"{where}: answer for run {run_id!r}, which has no run header")
        if not errs and (run_id, case_id) in out.answers:
            errs.append(f"{where}: duplicate answer for run {run_id!r} case {case_id!r} "
                        f"(first at {out.answers[(run_id, case_id)].where}; kept the first)")
        retrieval = None
        if not errs and "retrieval" in raw:
            retrieval, e = _parse_retrieval(raw["retrieval"], run_id, case_id, out.runs[run_id].provenance, where)
            errs += e
        if errs:
            out.errors += errs
            continue
        out.answers[(run_id, case_id)] = RecordedAnswer(run_id, case_id, raw["question"], raw["answer"],
                                                        raw["citation_numbering"], where, retrieval,
                                                        MappingProxyType(dict(raw.get("metadata", {}))))
    return out


def load_answers(paths: Iterable[Path | str]) -> AnswerSet:
    rows, errors = [], []
    for path in paths:
        p = Path(path)
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append((f"{p.name}:{n}", json.loads(line)))
            except json.JSONDecodeError as e:
                errors.append(f"{p.name}:{n}: invalid JSON ({e})")
    out = parse_answers(rows)
    out.errors = errors + out.errors
    return out
