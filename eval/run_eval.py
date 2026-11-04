"""Run the response-quality evaluation dataset through the expert agents.

Each dataset item is sent to run_expert() for every requested mode and code
variant. The answer, the documents actually retrieved, the ReAct trace and the
latency are saved to eval/results/runs.jsonl. Re-running skips results that
already exist, so an interrupted run can be resumed.

Variants
  branch    src/experts/base.py as it is in the working tree
  baseline  src/experts/base.py from git ref `main` (or --baseline-ref / --baseline-file)

Usage (from the repo root):
  python eval/run_eval.py --limit 2                 # smoke test: first 2 items
  python eval/run_eval.py                           # full run, both variants, both modes
  python eval/run_eval.py --variants branch --modes rag --ids kb-001 kb-020
  python eval/run_eval.py --env-file C:\\Projects\\Hello-Saarthi-main\\.env
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, List

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"
DATASET = EVAL_DIR / "eval_dataset.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
RUNS_FILE = RESULTS_DIR / "runs.jsonl"

KB_EXPERTS = ("notes_agent", "books_agent", "video_agent")


# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------

def setup_environment(env_file: str | None) -> None:
    """Make the repo importable and load API keys the same way the app does."""
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)  # kb_ops uses relative paths: knowledge_base/<expert>/vector_store

    from dotenv import load_dotenv

    env_path = Path(env_file) if env_file else ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment from {env_path}")
    else:
        print(f"WARNING: {env_path} not found; relying on existing environment variables")

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY is not set. Pass --env-file pointing at the project's .env.")


def ensure_indexes(experts: List[str], allow_missing: bool) -> None:
    """Download FAISS indexes from Supabase if needed and confirm each one loads."""
    try:
        from src.tools.kb_ops_supabase import sync_kb_on_startup
    except Exception as e:  # supabase package missing, etc.
        sync_kb_on_startup = None
        print(f"WARNING: Supabase sync unavailable ({e}); using local indexes only")

    from src.tools.kb_ops import get_retriever

    missing = []
    for expert in experts:
        if sync_kb_on_startup is not None:
            try:
                sync_kb_on_startup(expert)
            except Exception as e:
                print(f"WARNING: Supabase sync failed for {expert}: {e}")
        if get_retriever(expert) is None:
            missing.append(expert)
        else:
            print(f"Index OK: {expert}")

    if missing and not allow_missing:
        sys.exit(
            "ERROR: no usable index for " + ", ".join(missing) + ".\n"
            "Every item for these experts would trivially abstain, which makes the "
            "results meaningless. Check SUPABASE_URL / SUPABASE_SERVICE_KEY in .env, "
            "or pass --allow-missing-index to run anyway."
        )


# --------------------------------------------------------------------------
# Code variants
# --------------------------------------------------------------------------

def _git_show(ref: str, path: str) -> str:
    proc = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
    )
    if proc.returncode != 0:
        sys.exit(f"ERROR: could not read {path} from '{ref}': {proc.stderr.strip()}")
    return proc.stdout


def _load_module(name: str, source: str):
    tmp = Path(tempfile.gettempdir()) / f"{name}.py"
    tmp.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, tmp)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_variant(variant: str, baseline_ref: str, baseline_file: str | None):
    """Return the expert module for a variant.

    The baseline uses BOTH base.py and the prompt loader from the baseline ref,
    so prompt changes are part of the comparison, not just code changes.
    """
    if variant == "branch":
        import src.experts.base as module
        return module

    if baseline_file:
        source = Path(baseline_file).read_text(encoding="utf-8")
        origin = baseline_file
    else:
        source = _git_show(baseline_ref, "src/experts/base.py")
        origin = f"git {baseline_ref}:src/experts/base.py"
    module = _load_module("saarthi_eval_baseline_base", source)

    prompts = _load_module("saarthi_eval_baseline_prompts", _git_show(baseline_ref, "src/prompts/loader.py"))
    module.get_prompt = prompts.get_prompt
    print(f"Baseline loaded from {origin} (prompts from git {baseline_ref})")
    return module


# --------------------------------------------------------------------------
# Retrieval recording
# --------------------------------------------------------------------------

def make_recorder():
    """Build a get_retriever() replacement that logs every retrieval.

    Both the RAG path (retriever.invoke) and the planning path
    (the labelled search tool) go through get_retriever(), so wrapping it captures
    exactly which documents each answer was based on.
    """
    from langchain_core.retrievers import BaseRetriever
    from src.tools.kb_ops import get_retriever as real_get_retriever

    log: list = []
    cache: dict = {}

    class RecordingRetriever(BaseRetriever):
        inner: Any

        def _get_relevant_documents(self, query, *, run_manager=None):
            docs = self.inner.invoke(query)
            log.append({
                "query": query,
                "docs": [
                    {
                        "source": (getattr(d, "metadata", {}) or {}).get("source", ""),
                        "content": getattr(d, "page_content", ""),
                    }
                    for d in docs
                ],
            })
            return docs

    def get_retriever(expert_name: str):
        if expert_name not in cache:
            cache[expert_name] = real_get_retriever(expert_name)
        inner = cache[expert_name]
        return None if inner is None else RecordingRetriever(inner=inner)

    return get_retriever, log


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------

def load_dataset(ids: List[str] | None, limit: int | None) -> list:
    items = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]
    if ids:
        wanted = set(ids)
        items = [i for i in items if i["id"] in wanted]
    if limit:
        items = items[:limit]
    return items


def load_done() -> set:
    done = set()
    if RUNS_FILE.exists():
        for line in RUNS_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("error") is None:
                    done.add((r["variant"], r["mode"], r["id"]))
    return done


def run_one(module, recorder_log: list, item: dict, mode: str) -> dict:
    recorder_log.clear()
    messages = [{"role": "user", "content": item["query"]}]  # same shape the app sends
    start = time.time()
    try:
        response, trace = module.run_expert(item["expert"], item["query"], mode=mode, messages=messages)
        return {
            "response": response.model_dump(mode="json"),
            "react_trace": trace,
            "retrievals": list(recorder_log),
            "latency_s": round(time.time() - start, 2),
            "error": None,
        }
    except Exception:
        return {
            "response": None,
            "react_trace": [],
            "retrievals": list(recorder_log),
            "latency_s": round(time.time() - start, 2),
            "error": traceback.format_exc(),
        }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variants", nargs="+", default=["baseline", "branch"], choices=["baseline", "branch"])
    p.add_argument("--modes", nargs="+", default=["rag", "planning"], choices=["rag", "planning"])
    p.add_argument("--ids", nargs="+", help="Only run these item ids")
    p.add_argument("--limit", type=int, help="Only run the first N items")
    p.add_argument("--baseline-ref", default="main", help="Git ref for the baseline (default: main)")
    p.add_argument("--baseline-file", help="Load the baseline from a file instead of git")
    p.add_argument("--env-file", help="Path to .env (default: <repo>/.env)")
    p.add_argument("--allow-missing-index", action="store_true")
    p.add_argument("--fresh", action="store_true", help="Delete previous results first")
    args = p.parse_args()

    setup_environment(args.env_file)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    items = load_dataset(args.ids, args.limit)
    if not items:
        sys.exit("No dataset items selected.")
    experts = sorted({i["expert"] for i in items})
    ensure_indexes(experts, args.allow_missing_index)

    RESULTS_DIR.mkdir(exist_ok=True)
    if args.fresh and RUNS_FILE.exists():
        RUNS_FILE.unlink()
    done = load_done()

    get_retriever, log = make_recorder()
    total = len(items) * len(args.modes) * len(args.variants)
    count = 0

    with RUNS_FILE.open("a", encoding="utf-8") as out:
        for variant in args.variants:
            module = load_variant(variant, args.baseline_ref, args.baseline_file)
            module.get_retriever = get_retriever
            for mode in args.modes:
                for item in items:
                    count += 1
                    key = (variant, mode, item["id"])
                    label = f"[{count}/{total}] {variant:8} {mode:8} {item['id']}"
                    if key in done:
                        print(f"{label}  skip (already run)")
                        continue
                    result = run_one(module, log, item, mode)
                    record = {"variant": variant, "mode": mode, "id": item["id"],
                              "expert": item["expert"], "query": item["query"], **result}
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                    if result["error"]:
                        status = "ERROR " + result["error"].strip().splitlines()[-1][:80]
                    else:
                        r = result["response"]
                        status = (f"knowledge_present={r['is_knowledge_present']} "
                                  f"conf={r['confidence_score']:.2f} "
                                  f"retrievals={len(result['retrievals'])} "
                                  f"{result['latency_s']}s")
                    print(f"{label}  {status}")

    print(f"\nDone. Results in {RUNS_FILE.relative_to(ROOT)}")
    print("Next: python eval/score_eval.py")


if __name__ == "__main__":
    main()
