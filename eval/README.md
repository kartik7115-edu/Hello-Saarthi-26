# Saarthi.AI Response-Quality Evaluation Dataset (O4)

Purpose: measure whether expert answers are **correct** and **grounded in retrieved evidence**, and whether the agent **abstains** when it should. It is used to compare `main` (baseline) against `ai-architects/response-quality`.

## Files

| File | Purpose |
|---|---|
| `eval_dataset.jsonl` | The dataset, one JSON object per line (v1: 30 items) |
| `validate_dataset.py` | Schema and category checks. Run `python eval/validate_dataset.py` after every edit. |

## What is evaluated

Only the three **knowledge-base experts** go through the retrieval guards in `src/experts/base.py`:

| Expert | Knowledge base | Topics (from `test_data/queries.txt`) |
|---|---|---|
| `notes_agent` | Course notes | Z-Transform, DFT, FFT, filter design |
| `books_agent` | Oppenheim, *Signals & Systems* | LTI systems, Fourier series/transform, convolution, Laplace |
| `video_agent` | 488 lecture transcripts | General Signals & Systems / DSP lectures |

`calculator_agent`, `saarthi_agent`, `web_agent` and `data_analysis_agent` don't retrieve from a KB and are out of scope for v1.

**Evaluate at the expert level.** Call `run_expert(expert, query, mode)` directly, not the full LangGraph graph. In the graph, an abstaining `notes_agent` falls through to `books_agent` and then to `web_agent` (`route_from_notes` and `route_from_books` in `src/orchestrator/graph.py`), so a correct abstention would be replaced by a web answer and couldn't be measured.

**Mode is a run parameter.** Run every item in both `rag` and `planning`. Those are the two guarded paths, so the results show whether they behave differently.

## Fields

| Field | Type | Description |
|---|---|---|
| `id` | string | Stable ID, e.g. `kb-001`. Never reuse. |
| `expert` | string | `notes_agent`, `books_agent` or `video_agent` (matches `ExpertName`) |
| `category` | string | See below |
| `difficulty` | string | `easy`, `medium`, `hard`, `complex` |
| `query` | string | The user question, exactly as a user would type it |
| `expected_behavior` | string | `answer` or `abstain` |
| `key_points` | list[string] | Atomic facts a correct answer **must** contain (empty for `abstain`) |
| `forbidden_claims` | list[string] | Statements that make the answer wrong (typical hallucinations) |
| `expected_sources` | list[string] | KB filenames that should be retrieved. **Empty in v1, see below.** |
| `source` | string | Origin: `queries.txt <ID>`, `manual`, `user-log` |
| `kb_verified` | bool | `true` once the item has been checked against the actual KB documents |
| `notes` | string | Reviewer notes |

## Categories (v1 counts)

| Category | Expected | Count | What it tests |
|---|---|---|---|
| `in_kb_direct` | answer | 10 | Answer is stated plainly in one KB document |
| `in_kb_synthesis` | answer | 5 | Answer needs combining several passages |
| `in_kb_paraphrased` | answer | 4 | Casual wording or typos (tests retrieval and the empty-search retry) |
| `not_in_kb` | abstain | 5 | Plausible domain question the KB can't answer |
| `out_of_scope` | abstain | 3 | Off-domain question for this expert |
| `false_premise` | answer | 3 | The question assumes something untrue; the answer must correct it |

Split: 22 answer, 8 abstain; `notes_agent` 12, `books_agent` 12, `video_agent` 6.

## v1 limitations: read before trusting results

1. **`key_points` are unverified against the KB.** The KB indexes live in the Supabase bucket `saarthi-kb`, and `knowledge_base/` is not in the repo. The key points are written from standard Signals & Systems / DSP theory, so they are fine for checking **correctness**. **Groundedness**, though, is judged against the documents actually retrieved at run time, not against these points.
2. **`expected_sources` is empty**, so the retrieval hit rate can't be computed yet. Fill it in once filenames are known. The first eval run will log which files each query retrieves, and that is the easiest way to get them.
3. **The `not_in_kb` items are educated guesses.** For example, wavelets and compressed sensing aren't in Oppenheim's *Signals & Systems*, but the notes or transcripts might mention them. If a run finds real supporting evidence for one, recategorise it rather than counting it as a failure.

Flip `kb_verified` to `true` item by item as each is checked.

## Metrics (per expert, per mode)

| Metric | Definition |
|---|---|
| **Answer accuracy** | Share of `answer` items with every `key_point` present and no `forbidden_claim` |
| **Groundedness** | Share of answer claims supported by the retrieved documents |
| **Correct abstention rate** | Share of `abstain` items where `is_knowledge_present == False` |
| **False abstention rate** | Share of `answer` items wrongly abstained. Checks the new guards aren't over-blocking. |
| **Retrieval hit rate** | Share of `answer` items retrieving at least one `expected_source` (blocked until sources are filled) |
| **Confidence vs. correctness** | Mean `confidence_score` on correct vs. incorrect answers |

## Running the evaluation

| File | Purpose |
|---|---|
| `run_eval.py` | Runs each item through `run_expert()` for each variant and mode. Records the answer, the exact documents retrieved, the trace and the latency in `results/runs.jsonl`. Resumable. |
| `score_eval.py` | Grades the runs with an LLM judge (cached in `results/grades.jsonl`) and writes `results/report.md`. |

**Variants:** `baseline` loads `src/experts/base.py` from `main` via `git show`, so there's no branch switching. `branch` is the working tree. Both use the same recording retriever, so the evidence the answers rely on is captured identically.

```bat
set PY=C:\Projects\Hello-Saarthi-main\.venv\Scripts\python.exe
set ENV=C:\Projects\Hello-Saarthi-main\.env

:: 1. Smoke test: 2 items x 2 variants x 2 modes = 8 runs
%PY% eval\run_eval.py --limit 2 --env-file %ENV%
%PY% eval\score_eval.py --env-file %ENV%

:: 2. Full run: 30 x 2 x 2 = 120 runs (resumes; already-finished runs are skipped)
%PY% eval\run_eval.py --env-file %ENV%
%PY% eval\score_eval.py --env-file %ENV%
```

The runner stops if any expert has no usable FAISS index, because every item would then abstain and the numbers would mean nothing. It downloads missing indexes from Supabase the same way the app does at startup, so `.env` needs `OPENAI_API_KEY`, `SUPABASE_URL` and `SUPABASE_SERVICE_KEY`.

**Judge caveat:** the judge (`gpt-4.1` by default) is from the same model family as the experts. Spot-check a sample of grades by hand before quoting the numbers, and use `--judge-model` to try a different judge.

## Writing rules

- Write `key_points` as atomic facts, one claim each.
- Keep queries realistic, including typos and casual phrasing.
- Don't edit an item after a run has used it. Add a new ID instead.
- Run `validate_dataset.py` before committing.
