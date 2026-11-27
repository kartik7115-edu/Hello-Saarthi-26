# Sprint 6: Evidence grounding and citation quality

This is an offline, deterministic evaluator. It connects three things in a **recorded** answer: its
claims, its citations, and the retrieval context the answer was generated from. Each claim is
checked against **human** claim–evidence judgements.

It does not generate answers or run a retriever. It never calls a model, an NLI system or an API,
and it never decides on its own that a source "supports" a claim. Support only ever comes from a
person's judgement, looked up exactly.

> **Status.** No real Saarthi answers have been recorded or annotated yet. Every result in this
> document and in `fixtures/` is an **illustrative fixture**: invented sources, answers and
> judgements. None of it is Saarthi performance.

## Architecture

```
recorded answers (records.py) ─┐   answer text + its retrieval context (Sprint 5 hit format)
Sprint 5 results (optional) ───┼─▶ resolution.py ─▶ metrics.py ─▶ evaluator.py ─▶ JSON report
annotations (annotations.py) ──┘   parse (Sprint 2 parser) → resolve → associate → claim outcomes
```

| File | Role |
|---|---|
| `annotations.py` | Strict, versioned schema for human annotations (`evidence-annotation/1`) and their provenance |
| `records.py` | Recorded answers and their retrieval context. Rejects any annotation field (the leakage guard) |
| `resolution.py` | Resolves citations against the retrieval context and associates them with claims |
| `metrics.py` | Claim outcomes and the five metrics, each with an explicit numerator and denominator |
| `evaluator.py` | Joins answers with annotations, groups results by annotation status, and separates findings from data-quality problems |
| `run_evaluation.py` | Command-line interface |

Dependencies point one way only. The package imports public pieces of `eval.answer_eval` (the
Sprint 2 citation parser and the Sprint 1 dataset loader) and of `eval.retrieval_eval` (the Sprint 5
recorded-result parser, the page-id convention and the metric status names). Neither of those
imports this package. Nothing here imports the application. No Sprint 1–5 file was changed.

## Input 1: recorded answers (what the system produced)

```json
{"record_type": "run", "run_id": "saarthi-kb-v1", "provenance": "recorded", "description": "...", "generator": {...}}
{"record_type": "answer", "run_id": "saarthi-kb-v1", "case_id": "df-01", "question": "...",
 "answer": "The Nyquist rate is twice the highest frequency [1].",
 "citation_numbering": "retrieval_order",
 "retrieval": {"hits": [{"document_id": "EE210 Module 1 Signals and Sampling", "page": 3, "chunk_id": "4f1c..."}]}}
```

- **`retrieval`.** This uses Sprint 5's recorded-result format and is validated by Sprint 5's parser.
  The context can instead come from a Sprint 5 results file (`--retrieval`) with the same `run_id`
  and `case_id`. Giving both is a data problem.
- **`citation_numbering: retrieval_order`.** This means `[n]` is the n-th hit, matching the way
  `src/knowledge_base/gemini_rag.py` numbers the excerpts it shows the model. With `unspecified`,
  numeric citations are reported as unresolvable and never guessed.
- **Strict schema and leakage guard.** Any other field is rejected, including annotation fields
  such as `claims` or `support_status`. The reserved annotation field names
  (`records.RESERVED_ANNOTATION_KEYS`) are also rejected **anywhere** inside free-form objects:
  run `generator` and `metadata`, answer `metadata`, and the retrieval context, including each
  hit's `metadata`. A Sprint 5 results file used as the context is checked the same way. A hit
  there containing such a field makes the case a data problem, and it isn't evaluated. Ground
  truth therefore cannot travel with the recorded inputs.

## Input 2: annotations (human ground truth), one per recorded answer

```json
{"schema_version": "evidence-annotation/1", "annotation_id": "ann-df-01", "run_id": "saarthi-kb-v1",
 "case_id": "df-01", "question": "...", "language": "en", "answer_sha256": "<sha256 of the exact answer>",
 "annotation": {"status": "verified", "version": 1, "annotator": "a1", "reviewer": "r2", "reviewed_on": "2026-11-20"},
 "claims": [{"claim_id": "c1", "text": "The Nyquist rate is twice the highest frequency",
             "span": {"start": 0, "end": 47}, "requires_evidence": true,
             "expected_evidence": [{"level": "page", "id": "EE210 Module 1 Signals and Sampling#p3"}],
             "evidence_judgements": [{"evidence": {"level": "chunk", "id": "4f1c..."}, "relation": "supports"}],
             "support_status": "supported"}]}
```

### Binding to the answer
- `answer_sha256` ties the annotation to the exact answer text. If the answer changes, the
  annotation no longer applies and the case is reported as a data problem.
- Every `span` must lie inside the answer, and its text must equal `text`.
- The question must equal the recorded question. It is also compared with the Sprint 1 query when
  the `case_id` is a Sprint 1 item; a difference is reported, not fatal.

### What can be recorded
- **Evidence references** are `{"level": "document" | "page" | "chunk", "id": ...}`. Page ids follow
  the Sprint 5 convention, `"<document id>#p<page>"`.
- **Relations** for one claim and one unit: `supports`, `contradicts`, `insufficient` or
  `irrelevant`. A unit that isn't listed is **not assessed** for that claim, never irrelevant.
- **`support_status`** is the person's overall verdict: `supported`, `contradicted`,
  `insufficient` or `not_assessed`. `supported` needs a `supports` judgement, and `contradicted`
  needs a `contradicts` judgement.

### Provenance
| Status | Meaning |
|---|---|
| `verified` | Has a reviewer who is not the annotator, and a `reviewed_on` date that isn't in the future. |
| `unverified` | Written but not reviewed. Its results are provisional and reported in their own group. |
| `fixture` | Illustrative only. It may annotate only fixture runs, and fixture runs may carry only fixture annotations. |

`version` is the annotation's revision number.

### What is rejected
Unknown fields, a wrong `schema_version`, duplicate annotation, claim or judgement ids, a second
annotation for the same answer, overlapping claim spans, a citation offset given to two claims,
malformed references, Hindi or Marathi (`language` must be `en` or `hinglish`), and future review
dates.

### Annotation workflow
1. Record the answers (and their retrieval context) from the real system.
2. An annotator splits each answer into claims, recording exact spans; the spans come from offsets,
   never from retyped text. Then they judge each cited or retrieved unit for each claim.
3. Save the annotation as `unverified`.
4. A second person reviews it. Only then do they set `status: verified`, `reviewer` and
   `reviewed_on`, and increase `version` for any change. Nothing promotes an annotation to
   verified automatically.

## Citation resolution rules

Citations are found by the Sprint 2 parser. Each marker becomes one or more **references**:
`[2, 3]` is two references at the same position, and every reference keeps its exact position and
text. Resolution is exact and only reaches evidence in the recorded context:

| Citation | Resolves to | Otherwise |
|---|---|---|
| `[n]` with `retrieval_order` | Hit *n*'s chunk, else its page, else its document | `unresolvable` (out of range, undeclared numbering, or no ordered hit list) |
| `[Title, p. N]`, `[Source: file, p. N]` | That page, if retrieved | `page_not_retrieved`, `not_in_context` (no retrieved document has that name), `unresolvable` (two documents share the name, or the page is not a number) |
| `[Title]`, `[Source: file]` | The document | `not_in_context` |
| `[Video: Title, m:ss]` | The document (the timestamp is reported, not checked) | `not_in_context` |
| Malformed marker (from the parser) | — | `malformed` |
| Any reference when there is no retrieval context | — | `no_retrieval_context` |

Names match a hit's `document_id`, `metadata.title` or `metadata.source_file` after normalisation
(lower case, `_` becomes a space, a trailing `.pdf` is removed, whitespace is collapsed). There is
no partial or fuzzy matching. Code, maths and indexing such as `x[n]` are never citations, because
the Sprint 2 parser masks them.

**Association with claims** covers only references in the answer body. References in a trailing
"Sources" block, which the application appends, are never associated with a claim.
- **Explicit:** a claim with `citation_starts` owns exactly the references at those offsets. An
  offset that matches no citation is a data error.
- **Positional:** every other reference attaches to the claim whose span contains it, or which it
  directly follows with only whitespace, closing punctuation or other citation markers in between.
  - no candidate: `unassociated`;
  - more than one candidate: `ambiguous`, reported and never guessed.

## Claim outcomes

These are never collapsed into one failure status:

| Outcome | Meaning |
|---|---|
| `evidence_not_required` | The claim needs no evidence. Its citations are still listed. |
| `no_citation` | No reference is associated with the claim. |
| `association_unresolved` | No reference is associated, but an ambiguous one could belong to the claim. |
| `invalid_citation` | References are associated, but none resolves (malformed, not in the context, page not retrieved, out of range or ambiguous). The reasons are listed. |
| `retrieval_context_missing` | Well-formed references are associated, but no retrieval context was recorded, so they can't be resolved. This is a data gap, not an answer failure: coverage becomes a range, and it is reported under `data_quality`, not as an evaluation finding. |
| `cited_evidence_supports` | A resolved cited unit is judged `supports` for **this** claim, and none `contradicts`. |
| `cited_evidence_contradicts` | A cited unit is judged `contradicts`, and none `supports`. |
| `conflicting_evidence` | Cited units are judged both `supports` and `contradicts`. |
| `cited_evidence_not_assessed` | No support or contradiction is judged, and a cited unit is unjudged, so it might still support the claim. |
| `cited_evidence_insufficient` | Every cited unit is judged; at least one is `insufficient`. |
| `cited_evidence_irrelevant` | Every cited unit is judged `irrelevant`. |

- **Exact lookup only.** A relation is looked up for the exact unit a reference resolves to: same
  level, same id. A page judgement is never applied to a chunk on that page (or the reverse), and
  text similarity is never used. The cited unit may be unjudged while a unit containing it, or
  inside it, is judged (according to the recorded hits). A diagnostic then names that unit so the
  annotator can judge the cited one; the judgement is still not transferred.
- **Shown separately.** The person's `support_status` is shown beside the outcome and never
  changed. Expected evidence produces diagnostics ("not retrieved", "retrieved but not cited") and
  never implies support.

## Metrics

Each metric is a numerator over a denominator. A denominator of 0 gives `not_applicable`, never 0.
Missing input gives `not_evaluated`. Undecidable items make a metric `bounded`: the lower bound
counts them as failures, the upper bound as successes. There is no weighted overall score.

| Metric | Numerator / denominator | Needs annotation |
|---|---|---|
| `citation_validity` | well-formed references / (well-formed references + malformed markers), counted over every marker, Sources block included | no |
| `citation_resolution` | resolved references / well-formed references | no (needs the retrieval context) |
| `citation_precision` | distinct (claim, cited unit) pairs from associated, resolved references whose unit is judged `supports` for that claim / all such distinct pairs. Repeating a citation does not change it. Unjudged units make it bounded. Ambiguous, unassociated and Sources-block references are outside the denominator, and the result's `reason` gives their counts. `not_evaluated` without a retrieval context. | yes |
| `required_claim_citation_coverage` | evidence-required claims with at least one associated resolved reference / evidence-required claims. Ambiguous-only claims and `retrieval_context_missing` claims make it bounded. | yes |
| `required_claim_supported_coverage` | evidence-required claims whose outcome is `cited_evidence_supports` **and** whose human verdict is `supported` / evidence-required claims. A cited supporting unit never overrides a person's verdict of `contradicted` or `insufficient`; such claims are listed under `supporting_citation_but_verdict_differs`. Some claims are undecided: outcome `cited_evidence_not_assessed`, `association_unresolved` or `retrieval_context_missing`, or a supporting citation with verdict `not_assessed`. They make it bounded. | yes |

**Counts, not rates:** claim outcomes, and the person's `support_status` (supported, contradicted,
insufficient meaning not supported by the evidence, not_assessed), for all claims and for
evidence-required claims.

**Aggregation:** numerators and denominators are summed per run and per **group**: `verified`,
`unverified`, `fixture` or `unannotated`. Groups are never pooled, so unverified or fixture
judgements can never count as verified ground truth.

## Which results are deterministic and which need human review

| Deterministic from the recorded inputs alone | Depends on human judgements |
|---|---|
| Citation syntax, validity, resolution, association | `citation_precision` |
| `no_citation` and `invalid_citation` outcomes, coverage of evidence-required claims | Every `cited_evidence_*` outcome, `conflicting_evidence` |
| | `required_claim_supported_coverage`, the support-status counts |

Human-dependent results carry their annotation's status. Only the `verified` group is ground truth.

## Running it

```bash
python -m eval.evidence_eval.run_evaluation --answers answers.jsonl --annotations annotations.jsonl --out report.json
python -m eval.evidence_eval.run_evaluation --answers answers.jsonl --retrieval sprint5_results.jsonl \
    --annotations annotations.jsonl --out report.json --strict
python -m pytest -q eval/evidence_eval/tests
```

Exit codes: 0 = report written; 1 = unreadable input, or `--strict` with data-quality problems;
2 = invalid annotation records (nothing is evaluated on invalid ground truth).

The report has six parts:
- `runs`, with provenance and an `illustrative` flag;
- `aggregates`, per run and per group;
- `evaluation_findings`: what the answers did, such as claims without citations, invalid,
  irrelevant or contradicting citations, unassociated, ambiguous or duplicate citations,
  contradicted or insufficient verdicts, and supporting citations whose human verdict differs;
- `data_quality`: problems with the inputs, such as malformed records, stale or inconsistent
  annotations, annotations without answers, missing or conflicting retrieval context, and
  questions that differ from Sprint 1;
- `cases`, with every citation (position, resolution, association) and every claim (outcome,
  references, resolved units with relations, diagnostics);
- notes stating that the run is offline, illustrative or provisional, as applicable.

**Example: illustrative fixtures (invented data, not Saarthi performance)**

```
Run fixture-gen-a  ILLUSTRATIVE FIXTURE
  [fixture] 3 answer(s), 10 claim(s), 9 needing evidence
    citation_validity: 1.000 (8/8)
    citation_resolution: 0.750 (6/8)
    citation_precision: [0.500, 0.667] (3-4/6)
    required_claim_citation_coverage: 0.667 (6/9)
    required_claim_supported_coverage: [0.333, 0.444] (3-4/9)
    claim outcomes: {'evidence_not_required': 1, 'no_citation': 1, 'invalid_citation': 2, 'cited_evidence_supports': 3,
                     'cited_evidence_contradicts': 1, 'cited_evidence_not_assessed': 1, 'cited_evidence_irrelevant': 1}
Data-quality problems: none
```

Below is one claim from that report (abridged). The answer cites a page, the page was retrieved,
and the person judged that it contradicts the claim:

```json
{"claim_id": "c2", "text": "A sampling rate of 10 kHz is enough for a 6 kHz tone", "requires_evidence": true,
 "outcome": "cited_evidence_contradicts", "human_support_status": "contradicted",
 "citations": ["[fixture-doc-sampling, p. 1]"], "coverage_satisfied": true,
 "resolved_evidence": [{"citation_index": 1, "unit": {"level": "page", "id": "fixture-doc-sampling#p1"},
                        "relation": "contradicts"}]}
```

## Connecting real Saarthi outputs later

1. Record each answer together with the exact hits the generator saw, in order. Sprint 5's
   `adapters.hit_from_chunk` already maps Saarthi `Chunk` objects to hits.
2. Set `citation_numbering: retrieval_order` for the knowledge-base path, which numbers excerpts
   `[1]..[n]`, and `unspecified` wherever that is not certain.
3. Annotate the recorded answers as described above. Never edit an answer after annotating it: the
   hash will flag the annotation as stale.
4. Run the CLI. Read the `verified` group as ground truth and the others as provisional or
   illustrative.

The live application is never imported or changed here. A recorded run file is the only interface.

## Known limitations

- There is no real Saarthi data yet; only illustrative fixtures exist.
- Support is never inferred, so a claim whose cited unit has not been judged stays not assessed,
  and the metric is bounded until a person judges it.
- Positional association is a fixed, conservative rule. Claims that share one trailing citation
  need explicit `citation_starts`; otherwise only the claim directly before the citation gets it.
- Page and chunk judgements are not interchangeable. Annotators must judge the unit the citation
  actually points to (the chunk for `[n]`, the page for `[Title, p. N]`).
- Video timestamps are not checked. Numeric ranges such as `[1-3]` and footnotes such as `[^1]`
  are not recognised, which is a documented Sprint 2 parser limitation.
- A malformed marker counts as part of the Sources block if any Sources-block citation starts
  before it. This is an approximation, because the parser does not expose where the block starts.
- The evaluator trusts that the recorded retrieval context is the one the generator actually saw.
  It cannot check this. If a different context is recorded, `[n]` resolves to the wrong hit.
- The leakage guard works by field name (`RESERVED_ANNOTATION_KEYS`). Ground truth disguised under
  another key name, or inside free text, cannot be detected. Recording tools must simply never be
  given the annotation file.
- Factual correctness is not claimed from citations: a cited supporting unit shows that the answer
  is grounded in what was retrieved, not that the course material is right.
