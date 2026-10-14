#!/usr/bin/env python3
"""Evaluate the closed-grammar fast path offline, without pending state."""

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from keytao_bot.harness.operation_request import understand_message


WRITE_OPS = frozenset({
    "add", "select_candidate", "confirm", "cancel", "pending_recode",
    "swap", "delete", "recode", "reorder", "move", "batch_replace",
    "draft_submit", "draft_recall", "draft_clear", "draft_keep_only",
    "clear_history",
})


def evaluate(paths):
    expected_counts = Counter()
    matched_counts = Counter()
    total_rows = excluded_ambiguous = fired = correct = 0
    unsafe_direct = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                total_rows += 1
                expected = row["expected"]
                if expected.get("ambiguous", False):
                    excluded_ambiguous += 1
                    continue
                expected_counts[expected["op"]] += 1
                request = understand_message(row["message"])
                matches_op = request.op == expected["op"]
                matched_counts[expected["op"]] += int(matches_op)
                if request.op != "none":
                    fired += 1
                    correct += int(matches_op)
                if (
                    request.op in WRITE_OPS
                    and request.frame == "direct"
                    and expected["frame"] != "direct"
                ):
                    unsafe_direct.append({
                        "id": row["id"],
                        "message": row["message"],
                        "expected": expected,
                        "actual": asdict(request),
                    })
    return {
        "input_files": [str(path) for path in paths],
        "pending_state": None,
        "total_rows": total_rows,
        "excluded_ambiguous": excluded_ambiguous,
        "evaluated_rows": total_rows - excluded_ambiguous,
        "metric_definition": "Coverage and accuracy compare op only; fired means op != none.",
        "per_op_coverage": {
            op: {
                "rows": count,
                "matched": matched_counts[op],
                "fraction": matched_counts[op] / count,
            }
            for op, count in sorted(expected_counts.items())
        },
        "accuracy_when_fired": {
            "fired": fired,
            "correct": correct,
            "fraction": correct / fired if fired else None,
        },
        "unsafe_direct": unsafe_direct,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    routing_dir = REPO_ROOT / "eval" / "routing"
    paths = [routing_dir / "supplement.v1.jsonl"]
    private_gold = routing_dir / "private" / "gold.v1.jsonl"
    if private_gold.is_file():
        paths.append(private_gold)
    report = evaluate(paths)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    accuracy = report["accuracy_when_fired"]
    accuracy_text = "n/a" if accuracy["fraction"] is None else f'{accuracy["fraction"]:.3f}'
    coverage = "; ".join(
        f'{op}={item["matched"]}/{item["rows"]} ({item["fraction"]:.3f})'
        for op, item in report["per_op_coverage"].items()
    )
    unsafe_ids = ", ".join(str(row["id"]) for row in report["unsafe_direct"])
    print(f'Rows: {report["evaluated_rows"]} evaluated, {report["excluded_ambiguous"]} ambiguous excluded')
    print("Pending state: none (ticket context is not evaluated)")
    print(f'Accuracy when fired (op): {accuracy["correct"]}/{accuracy["fired"]} ({accuracy_text})')
    print(f"Per-op coverage: {coverage}")
    print(f'unsafe_direct: {len(report["unsafe_direct"])}; ids: {unsafe_ids or "none"}')
    print(f"Report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
