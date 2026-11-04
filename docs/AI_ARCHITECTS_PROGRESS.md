# Saarthi.AI — AI Architects Progress Log

**Team:** AI Architects / AI Agents
**Repository:** `DevaP06/Hello-Saarthi-26`
**Branch:** `ai-architects/response-quality`
**Last updated:** 2026-11-04

This is a living document. It is updated whenever a meaningful improvement is completed, and records what changed, why, how it was validated, and what remains open.

---

## 1. Scope and responsibilities

| Objective | Responsibility | Direction |
|---|---|---|
| **O5** | AI Response Quality | Improve response reliability; prevent unsupported answers when retrieval fails or the materials don't cover a question. |
| **O4** | Evaluations | Build tests and an evaluation set that systematically verify expert-agent behaviour and catch regressions. |

**Working approach:** Measure → Diagnose → Improve → Validate.

**Current phase:** Phase 1 (failure handling) is complete. Phase 2 (quality measurement) is built but **not yet run**, because no OpenAI API key is available (see §9).

---

## 2. Development environment

| Item | Value |
|---|---|
| Local working copy | `C:\Projects\Hello-Saarthi-26-AI-Architects` |
| Base branch | `main` (at `e053585`) |
| Python | 3.11.9 |
| Interpreter used | `C:\Projects\Hello-Saarthi-main\.venv\Scripts\python.exe` |
| Verified imports | `langchain_core`, `langchain_openai`, `langgraph`, `pydantic` |

**Why a feature branch:** it isolates AI Architects changes from other teams' work, keeps them reviewable, and protects `main`.

**Why reuse the existing venv:** the new working copy had no dependencies installed. Reusing the original project's environment is a development convenience, not a permanent solution (see §9).

**How the knowledge base is stored:** each knowledge-base expert has a FAISS index (`index.faiss` + `index.pkl`) in the Supabase Storage bucket `saarthi-kb`. The app downloads it to `knowledge_base/<expert>/vector_store` at startup (`src/tools/kb_ops_supabase.py`). `knowledge_base/` is gitignored. The indexes were built with OpenAI `text-embedding-3-large`, so **even retrieval needs an OpenAI key**.

---

## 3. Which agents this work covers

Only three experts retrieve from a knowledge base and go through the guards in `src/experts/base.py`:

| Expert | Knowledge base |
|---|---|
| `notes_agent` | Course notes: Z-Transform, DFT, FFT, filter design |
| `books_agent` | Oppenheim, *Signals & Systems* |
| `video_agent` | 488 lecture transcripts |

`run_expert()` is only called in `rag` or `planning` mode in the current app. The `fast` branch exists in code but has no callers.

**Routing note:** in the LangGraph graph, an abstaining `notes_agent` falls through to `books_agent`, then to `web_agent` (`src/orchestrator/graph.py`). A correct "not in the materials" answer can therefore be replaced by a web answer at the app level. This is why evaluation is done per expert (§7).

---

## 4. Phase 1: retrieval failure handling (commits `b6f2ffe`, `c9c2c66`)

All implementation changes are in `src/experts/base.py`.

### 4.1 Planning mode: retrieval-state tracking
Three flags inside the planning path of `run_expert()`:

| Variable | Meaning |
|---|---|
| `knowledge_search_attempted` | A knowledge-search tool was called. |
| `knowledge_search_succeeded` | Usable evidence was obtained. |
| `knowledge_search_failed` | A recognised failure occurred (missing index, exception, missing tool). |

**Why:** "never searched", "searched but found nothing", "search crashed" and "search succeeded" need different handling.

### 4.2 Narrower knowledge-tool identification
`is_knowledge_tool()` now matches only names containing `knowledge_base` (previously also any `search_*` tool).
**Limitation:** depends on the naming convention; an explicit tool registry would be more robust.

### 4.3 Usable-evidence check
`has_usable_evidence(result)` rejects `None`, empty collections, blank content, and the app's own failure messages (refined in §5.2).
**Limitation:** checks *presence* of evidence only, not relevance or correctness.

### 4.4 Missing-tool and exception handling
A tool the LLM requests that doesn't exist, or a retrieval exception, marks the search as failed. Exceptions are logged in full; the agent sees only `Error: The tool failed. No verified result was returned.`

### 4.5 Guards before answer synthesis (planning mode)
Synthesis is blocked via `planning_failure()` (`is_knowledge_present=False`, `confidence_score=0.0`) when no search was attempted, the index is unavailable, retrieval raised an exception, or no usable evidence was found after the retry.

### 4.6 RAG mode safeguards
The response is blocked **before any LLM call** when the retriever is unavailable, retrieval raises, no documents are returned, or all documents have blank `page_content`.

### 4.7 Retry after an empty search (planning mode)
`MAX_EMPTY_SEARCH_RETRIES = 1`. Missing indexes and exceptions stop immediately; an empty result allows one retry; synthesis is blocked if the retry also finds nothing. The retry counter only increments while no evidence has been found yet.

---

## 5. Phase 2a: fixes found while building the evaluation (commit `a6c85f8`)

Reading the code for the evaluation work turned up three problems that undermined the Phase 1 guards.

### 5.1 The planning prompt prevented the retry
**Problem:** the ReAct prompt said *"Search again only when the first search returned usable evidence"*. That is the opposite of the retry logic in §4.7: after an empty search, the model was told not to search again, so the retry rarely happened.
**Fix:** step 3 of the workflow now asks for one more search with different keywords after an empty result (at most two searches for the same missing information). After an empty search, the tool message is replaced with an explicit instruction, `EMPTY_SEARCH_RETRY_HINT`.

### 5.2 Good evidence containing "error:" was rejected
**Problem:** `has_usable_evidence()` rejected a result if `"error:"` (or any other failure marker) appeared *anywhere* in it. The retriever tool returns all retrieved chunks joined into one string, so a single chunk such as *"Quantization error: …"*, which is common in DSP material, caused valid evidence to be discarded and the agent to abstain wrongly.
**Fix:** failure markers now match only at the **start** of a result, since they are whole messages produced by our own code. The `[No Index Found` check uses the same rule.

### 5.3 The shared expert prompt contradicted abstention
**Problem:** the `expert_system` prompt in `src/prompts/loader.py`, used by every knowledge-base expert, said:
- *"Never say … 'I don't have this in my notes'"*: this banned exactly the abstention the guards rely on, and contradicted `notes_agent`'s own extra instruction;
- *"Answer confidently like a tutor who knows the subject deeply"*: this encouraged answering from general knowledge;
- *"Always include … at least one worked example"*: this pushed the model to invent examples the materials don't contain;
- it called answer length *"your most important instruction"*.

**Fix:** a new **GROUNDING** section, which overrides the other instructions when they conflict:
- base every factual claim on the study materials;
- when the materials don't cover a question, say so plainly and set `is_knowledge_present` to false;
- correct false assumptions in the question;
- set `confidence_score` to how fully the materials support the answer.

Worked examples and applications are now included only when the materials provide them. The tone section still avoids filler and internal jargon, but no longer bans saying something isn't covered.

**Impact:** this changes the behaviour of `notes_agent`, `books_agent` and `video_agent`. The team should be told before it is merged.

### 5.4 Citations were invented
**Problem:** the model only ever received the *text* of retrieved passages, never their file names. In RAG mode only `page_content` was joined into the prompt, and the planning search tool (`create_retriever_tool`) returns only the text too. Yet every answer must list `sources` with a file name and page number, so the model had to make them up. Page numbers were always invented, because ingestion records only `{"source": filename}`. Students saw these citations in the app.
**Fix:**
- Every passage now reaches the model labelled `[Source: <file name>]` (`format_documents()`), in both modes. The planning search tool is now a `StructuredTool` that returns labelled passages; its name is unchanged.
- The model is told to cite only those labels, copy snippets word for word, and set `page_number` to 0.
- `finalize_response()` checks every answer afterwards: citations to files that weren't retrieved are dropped, snippets that don't appear in the cited file are replaced with a real excerpt, `page_number` is set to `0` (unknown; the API already defaults to 0, so the frontend is unaffected), and duplicates are removed.

### 5.5 "Not in the materials" answers could still claim confidence and sources
**Problem:** the guards set confidence to 0 only when *retrieval* failed. When the model itself decided the materials didn't cover a question, it could still return a high `confidence_score` and citations.
**Fix:** `finalize_response()` sets confidence to 0 and clears sources whenever `is_knowledge_present` is false, in every mode.

### 5.6 Planning mode hid most of the retrieved evidence
**Problem:** each tool result was cut to 1,500 characters before the model saw it. A search returns up to 5 passages of about 1,000 characters each, so the model saw only the first one or two, both while reasoning and when writing the final answer.
**Fix:** the limit is now `MAX_TOOL_RESULT_CHARS = 6000`, which fits all 5 passages. The trace shown in the UI is unchanged (300 characters per observation).

---

## 6. Automated tests

All tests use mocked tools and mocked LLM responses. **They verify control flow and prompt content, not live LLM or retrieval quality.** None need an API key.

### 6.1 `test_planning_guard.py`: 9 tests

| # | Scenario | Expected | Result |
|---|---|---|---|
| 1 | Retrieval exception | Block synthesis | PASS |
| 2 | Index unavailable | Block synthesis | PASS |
| 3 | Empty retrieval | Block synthesis | PASS |
| 4 | Successful retrieval | Allow synthesis | PASS |
| 5 | Empty → retry → evidence | 2 tool calls, 1 synthesis | PASS |
| 6 | Empty → retry → empty | 2 tool calls, 0 synthesis, confidence 0.0 | PASS |
| 7 | Evidence containing `"error:"` | Allow synthesis (§5.2) | PASS |
| 8 | Failure message returned as a result | Block synthesis | PASS |
| 9 | Retry hint and retry-friendly prompt | Hint sent after empty search; prompt permits retry (§5.1) | PASS |

### 6.2 `test_rag_guard.py`: 5 tests
Retriever unavailable, retrieval exception, empty retrieval, blank document text (all blocked with no LLM call), and successful retrieval (evidence in the prompt). All PASS.

### 6.3 `test_prompt_grounding.py`
Checks that the contradictory instructions are gone, the GROUNDING rules are present, and the prompt template still formats. PASS.

### 6.4 `test_citation_guard.py`: 23 checks
Source labels reach the model in both modes; citations to unretrieved files are dropped; real citations keep a verbatim snippet and get page number 0; invented snippets are replaced with real text; path and case differences still match and duplicates are removed; "not in the materials" answers get confidence 0 and no sources; planning mode passes all 5 retrieved passages through to synthesis. Uses real `ExpertResponse` objects rather than mocks. All PASS.

**Regression value:** tests 7 and 9, the prompt test and the citation tests **fail on the pre-fix code** and pass after the fixes, so they guard against the bugs coming back.

### 6.5 How to run
```bat
set PY=C:\Projects\Hello-Saarthi-main\.venv\Scripts\python.exe

%PY% -m py_compile src\experts\base.py src\prompts\loader.py
%PY% test_planning_guard.py
%PY% test_rag_guard.py
%PY% test_prompt_grounding.py
%PY% test_citation_guard.py
%PY% eval\validate_dataset.py
git diff --check
```

---

## 7. Phase 2b: evaluation framework (commit `f75a318`)

Built to measure answer quality, not just failure handling. All files are in `eval/`; full details are in `eval/README.md`.

### 7.1 Dataset: `eval/eval_dataset.jsonl` (30 items)

| Category | Expected | Count | Tests |
|---|---|---|---|
| `in_kb_direct` | answer | 10 | Answer stated plainly in the materials |
| `in_kb_synthesis` | answer | 5 | Combining several passages |
| `in_kb_paraphrased` | answer | 4 | Casual wording or typos (retrieval and the retry) |
| `not_in_kb` | abstain | 5 | Plausible questions the materials don't cover |
| `out_of_scope` | abstain | 3 | Off-topic questions |
| `false_premise` | answer | 3 | Must correct a false assumption |

Experts: `notes_agent` 12, `books_agent` 12, `video_agent` 6. All 20 knowledge-base questions from `test_data/queries.txt` for these experts are included. `validate_dataset.py` checks the schema.

### 7.2 Runner: `eval/run_eval.py`
- Calls `run_expert()` per expert, not through the graph (see the routing note in §3), in both `rag` and `planning` modes.
- Compares two variants: **baseline** (`base.py` *and* the prompts from `main`, loaded via `git show`) and **branch** (the working tree).
- Records the answer, confidence, latency and the **exact documents retrieved**, via a recording wrapper around `get_retriever()`.
- Downloads indexes from Supabase like the app does, and refuses to run if an index is missing (every item would trivially abstain).
- Resumable: finished runs are skipped.

### 7.3 Scorer: `eval/score_eval.py`
An LLM judge (default `gpt-4.1`) grades each run. Grades are cached and shared across variants when the answer and evidence are identical. It writes `eval/results/report.md` with these metrics:

| Metric | Meaning |
|---|---|
| Answer accuracy | All key points present, no forbidden claim |
| Key-point recall | Partial credit |
| Groundedness | Share of answer claims supported by the retrieved documents |
| Correct abstention | "Not in the materials" items correctly declined |
| False abstention | Answerable items wrongly declined (over-blocking check) |
| Forbidden-claim rate | Known hallucinations asserted |
| Confidence (correct vs. incorrect) | Calibration check |
| Flag mismatches | `is_knowledge_present` disagrees with what the text actually says |

The report also lists the knowledge-base files each question retrieved, to fill `expected_sources`.

### 7.4 Validation status
The runner and scorer were tested end to end against stand-in LLM and retriever modules: both variants, both modes, resume, grading, caching, and the report. Testing caught and fixed a grade-cache bug (two different questions with identical answers could share a grade). **They have not yet run against the real LangChain and OpenAI stack.**

---

## 8. Commit history

| Commit | Description |
|---|---|
| `b6f2ffe` | Add retrieval failure guards for expert responses |
| `c9c2c66` | Improve knowledge retrieval retry safeguards |
| `f75a318` | Add response-quality evaluation dataset, runner and scorer |
| `a6c85f8` | Fix retry prompt, evidence check and grounding prompt |
| *(next)* | Fix invented citations, abstention consistency and planning truncation |

**Repository notes:**
- `docs/` is listed in `.gitignore`, so this file must be added with `git add -f docs/AI_ARCHITECTS_PROGRESS.md`.
- `src/experts/base.py.before-ai-architects` is a local backup, excluded via `.git/info/exclude`.
- `eval/results/*.jsonl` (raw runs and grades) are gitignored; only `report.md` should be committed.

---

## 9. Known limitations and open items

**Blocking**
1. **No OpenAI API key.** Neither the app nor the evaluation can run. Retrieval also needs the key, because the indexes use OpenAI embeddings. All evaluation numbers are pending.

**Evaluation**
2. **Dataset unverified against the knowledge base.** Key points come from standard Signals & Systems / DSP theory; `expected_sources` is empty; the `not_in_kb` items are educated guesses. The chunk text is stored in the Supabase `index.pkl` files, so this can be verified with Supabase access alone, no OpenAI key needed.
3. **Judge bias.** The judge model is from the same family as the experts. Spot-check grades by hand before quoting numbers.

**Response quality**
4. **No independent grounding check at answer time.** The prompt now requires grounding and citations are checked against what was retrieved, but nothing verifies the answer's *claims* before it is returned. The evaluation will measure how often this matters.
4a. **Page numbers are unavailable.** Ingestion stores one document per file, so citations can only name the file. Real page numbers need re-ingestion with page metadata.
4b. **Mind mode citations are unchecked.** `mind_agent` writes its own numbered references when combining the three experts; those are not yet validated.
5. **Confidence calibration unverified.**
6. **App-level fallback.** An abstaining expert falls through to `web_agent` (§3). Whether this is desired is a product decision for the team.

**Engineering**
7. **Tool classification** relies on the `knowledge_base` naming convention.
8. **Environment:** the branch borrows another project's venv.
9. **Test runner:** tests are standalone scripts; moving to `pytest` would enable CI.

**Outside AI Architects scope (raise with the team)**
10. **Security:** the calculator tool (`src/tools/utilities.py`) passes model-generated expressions to Python `eval()`, which allows arbitrary code execution through crafted questions or documents.

---

## 10. Progression

```
Phase 1 — Failure handling             ✅ complete
  Retrieval fails → detect it → block the answer → test it

Phase 2 — Quality measurement           🔧 built, waiting for API key
  2a Fixes that undermined the guards   ✅ retry prompt, evidence check, grounding prompt,
                                           invented citations, abstention consistency, truncation
  2b Evaluation framework               ✅ dataset, runner, scorer (not yet run)
  2c Baseline vs. branch numbers        ⏳ needs OpenAI key

Phase 3 — Evidence-driven improvement   ⏳ planned
  Use evaluation results to guide prompt, retrieval and confidence changes
```

---

## 11. Next steps

1. Push the branch: `git push -u origin ai-architects/response-quality`.
2. Get an OpenAI API key from the project owner, then run the smoke test and the full evaluation (`eval/README.md`).
3. With Supabase access, extract the chunk text and verify the dataset (`kb_verified`, `expected_sources`).
4. Record the baseline vs. branch numbers here and decide on Phase 3 priorities from them.
5. Open a PR to `main`, noting the prompt change in §5.3.
