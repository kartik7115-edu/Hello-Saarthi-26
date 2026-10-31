"""Validate eval/eval_dataset.jsonl against the schema in eval/README.md.

Usage (from the repo root):
    python eval/validate_dataset.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

DATASET = Path(__file__).parent / "eval_dataset.jsonl"

EXPERTS = {"notes_agent", "books_agent", "video_agent"}
CATEGORIES = {
    "in_kb_direct": "answer",
    "in_kb_synthesis": "answer",
    "in_kb_paraphrased": "answer",
    "not_in_kb": "abstain",
    "out_of_scope": "abstain",
    "false_premise": "answer",
}
DIFFICULTIES = {"easy", "medium", "hard", "complex"}
REQUIRED = {
    "id": str, "expert": str, "category": str, "difficulty": str,
    "query": str, "expected_behavior": str, "key_points": list,
    "forbidden_claims": list, "expected_sources": list, "source": str,
    "kb_verified": bool, "notes": str,
}


def validate(path: Path) -> list[str]:
    errors, ids = [], set()
    lines = path.read_text(encoding="utf-8").splitlines()
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {n}: invalid JSON ({e})")
            continue
        where = f"line {n} ({item.get('id', '?')})"
        for field, typ in REQUIRED.items():
            if field not in item:
                errors.append(f"{where}: missing '{field}'")
            elif not isinstance(item[field], typ):
                errors.append(f"{where}: '{field}' must be {typ.__name__}")
        if errors and errors[-1].startswith(where):
            continue
        if item["id"] in ids:
            errors.append(f"{where}: duplicate id")
        ids.add(item["id"])
        if item["expert"] not in EXPERTS:
            errors.append(f"{where}: unknown expert '{item['expert']}'")
        if item["category"] not in CATEGORIES:
            errors.append(f"{where}: unknown category '{item['category']}'")
        elif item["expected_behavior"] != CATEGORIES[item["category"]]:
            errors.append(f"{where}: category '{item['category']}' expects "
                          f"'{CATEGORIES[item['category']]}'")
        if item["difficulty"] not in DIFFICULTIES:
            errors.append(f"{where}: unknown difficulty '{item['difficulty']}'")
        if item["expected_behavior"] == "answer" and not item["key_points"]:
            errors.append(f"{where}: 'answer' items need key_points")
        if item["expected_behavior"] == "abstain" and item["key_points"]:
            errors.append(f"{where}: 'abstain' items should have no key_points")
        if not item["query"].strip():
            errors.append(f"{where}: empty query")
    return errors


def summary(path: Path) -> None:
    items = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"{len(items)} items")
    for label, key in (("category", "category"), ("expert", "expert"),
                       ("behaviour", "expected_behavior")):
        counts = Counter(i[key] for i in items)
        print(f"  by {label}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    verified = sum(i["kb_verified"] for i in items)
    print(f"  KB-verified: {verified}/{len(items)}")


if __name__ == "__main__":
    errs = validate(DATASET)
    if errs:
        print("INVALID")
        for e in errs:
            print("  " + e)
        sys.exit(1)
    print("VALID")
    summary(DATASET)
