#!/usr/bin/env python3
"""Evaluate validated semantic parsing from recordings, or an explicit live run."""

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from eval.routing.run_fastpath import WRITE_OPS
from keytao_bot.harness.operation_request import understand_message
from keytao_bot.harness.semantic_parser import parse_operation, validate_operation
from keytao_bot.utils.llm_policy import chat_usage_metrics


def load_rows():
    routing_dir = REPO_ROOT / "eval" / "routing"
    paths = [routing_dir / "supplement.v1.jsonl", routing_dir / "private" / "gold.v1.jsonl"]
    rows = []
    present = []
    for path in paths:
        if not path.is_file():
            continue
        present.append(path)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append((json.loads(line), path.name == "gold.v1.jsonl"))
    return rows, present


def pending_context(row, is_gold):
    if row["expected"].get("refers_to_pending") is not True:
        return "none"
    return json.dumps({
        "type": "pending_from_transcript",
        "bot_reply": row.get("prev_bot_reply", ""),
    }, ensure_ascii=False)


def load_recordings(path):
    """Canonical rows are {id, raw, model, usage}; flat label rows also work."""
    recordings = {}
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            row_id = record.get("id") if isinstance(record, dict) else None
            if isinstance(row_id, bool) or not isinstance(row_id, (str, int)) or row_id == "":
                raise ValueError(f"Invalid recording id at line {number}")
            if row_id in recordings:
                raise ValueError(f"Duplicate recording id at line {number}")
            if "raw" not in record:
                record = dict(record)
                record["raw"] = {
                    key: value for key, value in record.items()
                    if key not in {"id", "model", "usage"}
                }
            recordings[row_id] = record
    return recordings


def evaluate(rows, paths, recordings):
    expected_counts = Counter()
    matched_counts = Counter()
    raw_matched_counts = Counter()
    violation_counts = Counter()
    total_rows = excluded_ambiguous = fired = correct = 0
    frame_correct = pending_correct = raw_pending_correct = missing_recordings = 0
    unsafe_direct = []
    unsafe_direct_dangerous = []
    unsafe_direct_query_as_write = []
    false_downgrades = []
    violation_rows = []
    model_wins = []
    fast_path_wins = []
    for row, _is_gold in rows:
        total_rows += 1
        expected = row["expected"]
        if expected.get("ambiguous", False):
            excluded_ambiguous += 1
            continue
        record = recordings.get(row["id"])
        if record is None:
            missing_recordings += 1
        raw = record.get("raw") if record is not None else None
        if isinstance(raw, dict):
            raw = dict(raw)
            model = raw.get("_model") or (record or {}).get("model") or "replay:unknown"
            raw["_model"] = model if isinstance(model, str) else "replay:unknown"
        # A transcript is not a trusted live ticket; never invent pending state.
        request, violations = validate_operation(raw, row["message"], None)
        if record is None:
            violations = []
        fast_request = understand_message(row["message"])
        matches_op = request.op == expected["op"]
        fast_matches_op = fast_request.op == expected["op"]
        expected_counts[expected["op"]] += 1
        matched_counts[expected["op"]] += int(matches_op)
        raw_matched_counts[expected["op"]] += int(
            isinstance(raw, dict) and raw.get("op") == expected["op"]
        )
        if request.op != "none":
            fired += 1
            correct += int(matches_op)
        frame_correct += int(request.frame == expected["frame"])
        pending_correct += int(request.refers_to_pending == expected.get("refers_to_pending", False))
        raw_pending_correct += int(
            isinstance(raw, dict)
            and raw.get("refers_to_pending") is expected.get("refers_to_pending", False)
        )
        if violations:
            violation_counts.update(violations)
            violation_rows.append({"id": row["id"], "violations": violations})
        detail = {
            "id": row["id"], "message": row["message"],
            "expected": expected, "actual": asdict(request),
        }
        if request.op in WRITE_OPS and request.frame == "direct" and expected["frame"] != "direct":
            unsafe_direct.append(detail)
            if expected["frame"] in {"question", "negated", "reported"}:
                unsafe_direct_dangerous.append(detail)
            elif expected["op"] in {"query", "chat", "help"}:
                unsafe_direct_query_as_write.append(detail)
        if expected["frame"] == "direct" and any(
            violation.startswith("frame_backstop:") for violation in violations
        ):
            false_downgrades.append({**detail, "violations": violations})
        if matches_op and not fast_matches_op:
            model_wins.append({**detail, "fast_path": asdict(fast_request)})
        elif fast_matches_op and not matches_op:
            fast_path_wins.append({**detail, "fast_path": asdict(fast_request)})
    evaluated = total_rows - excluded_ambiguous
    return {
        "input_files": [str(path) for path in paths],
        "pending_state": None,
        "pending_context_limitation": (
            "Gold and supplement ticket answers receive only a synthetic transcript. "
            "Validation always uses pending_state=None, because transcripts are not live tickets."
        ),
        "total_rows": total_rows,
        "excluded_ambiguous": excluded_ambiguous,
        "evaluated_rows": evaluated,
        "missing_recordings": missing_recordings,
        "metric_definition": (
            "Coverage and accuracy compare op only; fired means op != none. "
            "Raw metrics use model output before validation; missing or non-boolean pending flags are incorrect."
        ),
        "per_op_coverage": {
            op: {"rows": count, "matched": matched_counts[op], "fraction": matched_counts[op] / count}
            for op, count in sorted(expected_counts.items())
        },
        "raw_per_op_coverage": {
            op: {
                "rows": expected_counts[op], "matched": raw_matched_counts[op],
                "fraction": raw_matched_counts[op] / expected_counts[op] if expected_counts[op] else None,
            }
            for op in ("confirm", "select_candidate")
        },
        "accuracy_when_fired": {
            "fired": fired, "correct": correct, "fraction": correct / fired if fired else None,
        },
        "unsafe_direct": unsafe_direct,
        "unsafe_direct_dangerous": unsafe_direct_dangerous,
        "unsafe_direct_query_as_write": unsafe_direct_query_as_write,
        "false_downgrades": false_downgrades,
        "frame_accuracy": {
            "correct": frame_correct, "rows": evaluated,
            "fraction": frame_correct / evaluated if evaluated else None,
        },
        "refers_to_pending_accuracy": {
            "correct": pending_correct, "rows": evaluated,
            "fraction": pending_correct / evaluated if evaluated else None,
        },
        "raw_refers_to_pending_accuracy": {
            "correct": raw_pending_correct, "rows": evaluated,
            "fraction": raw_pending_correct / evaluated if evaluated else None,
        },
        "violations": {
            "count": sum(violation_counts.values()), "rows": violation_rows,
            "by_type": dict(sorted(violation_counts.items())),
        },
        "head_to_head": {"model_right_fast_wrong": model_wins, "fast_right_model_wrong": fast_path_wins},
    }


def _live_configuration():
    """Equivalent dedicated-key separation guard, without importing the E2E rig."""
    from dotenv import dotenv_values

    key_path = REPO_ROOT / ".e2e_key"
    try:
        key_text = key_path.read_text(encoding="utf-8").strip()
        key_values = dotenv_values(key_path, interpolate=False)
        api_key = str(key_values.get("E2E_OPENAI_API_KEY") or "").strip()
        if not api_key and key_text and not any(char.isspace() for char in key_text) and "=" not in key_text:
            api_key = key_text
        if not api_key:
            raise ValueError("Repo .e2e_key must contain a dedicated E2E_OPENAI_API_KEY")

        def reject_bot_key(values):
            for name, value in values.items():
                if str(name).lower() == "openai_api_key" and str(value or "").strip() == api_key:
                    raise ValueError("Dedicated E2E key matches a production OPENAI_API_KEY; refusing live mode")

        reject_bot_key(os.environ)
        for directory in {REPO_ROOT.resolve(), Path.cwd().resolve()}:
            for path in sorted(directory.iterdir()):
                if (path.name == ".env" or path.name.startswith(".env.")) and path.is_file():
                    reject_bot_key(dotenv_values(path))
        env_path = REPO_ROOT / ".env"
        if not env_path.is_file():
            raise ValueError("Repo .env is required for live base URL and model")
        values = {str(name).upper(): value for name, value in dotenv_values(env_path).items()}
    except (OSError, UnicodeError):
        raise ValueError("Cannot verify repo dedicated key and production-key separation") from None
    # E2E_BASE_URL / E2E_MODEL point the run at the production provider when the
    # local .env targets a different one.
    base_url = (os.environ.get("E2E_BASE_URL") or str(values.get("OPENAI_BASE_URL") or "")).strip()
    model = (os.environ.get("E2E_MODEL") or next((str(values.get(name) or "").strip() for name in (
        "WORD_QUERY_INTENT_MODEL", "OPENAI_INTENT_MODEL", "GEMINI_INTENT_MODEL", "OPENAI_MODEL",
    ) if str(values.get(name) or "").strip()), "")).strip()
    if not base_url or not model:
        raise ValueError("Repo .env must specify OPENAI_BASE_URL and an intent model or OPENAI_MODEL")
    return api_key, base_url, model


class _UsageClient:
    def __init__(self, client, usage=None):
        self._client = client
        self.usage = Counter() if usage is None else usage
        self.chat = SimpleNamespace(completions=self)

    def with_options(self, **options):
        return _UsageClient(self._client.with_options(**options), self.usage)

    async def create(self, **kwargs):
        response = await self._client.chat.completions.create(**kwargs)
        metrics = chat_usage_metrics(response)
        self.usage.update({
            "prompt_tokens": metrics.get("input_tokens", 0),
            "completion_tokens": metrics.get("output_tokens", 0),
            "cached_tokens": metrics.get("cached_tokens", 0),
        })
        return response


async def collect_live(rows, configuration, concurrency, completed):
    from openai import AsyncOpenAI

    api_key, base_url, model = configuration
    semaphore = asyncio.Semaphore(concurrency)
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=15.0, max_retries=1)

    async def record_one(index, row, is_gold):
        async with semaphore:
            tracked = _UsageClient(client)
            raw = await parse_operation(row["message"], pending_context(row, is_gold), client=tracked, model=model)
            completed[index] = {
                "id": row["id"], "raw": raw, "model": model, "usage": dict(tracked.usage),
            }

    tasks = [asyncio.create_task(record_one(index, row, is_gold)) for index, (row, is_gold) in enumerate(rows)]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await client.close()


def _print_report(report, out):
    def fraction_text(value):
        return "n/a" if value is None else f"{value:.3f}"

    accuracy = report["accuracy_when_fired"]
    coverage = "; ".join(
        f'{op}={item["matched"]}/{item["rows"]} ({item["fraction"]:.3f})'
        for op, item in report["per_op_coverage"].items()
    )
    print(f'Rows: {report["evaluated_rows"]} evaluated, {report["excluded_ambiguous"]} ambiguous excluded')
    print(f'Missing recordings: {report["missing_recordings"]}')
    print(f'Pending state: none; {report["pending_context_limitation"]}')
    print(f'Accuracy when fired (op): {accuracy["correct"]}/{accuracy["fired"]} ({fraction_text(accuracy["fraction"])})')
    print(f"Per-op coverage: {coverage}")
    raw_coverage = "; ".join(
        f'{op}={item["matched"]}/{item["rows"]} ({fraction_text(item["fraction"])})'
        for op, item in report["raw_per_op_coverage"].items()
    )
    print(f"Raw pending-op coverage: {raw_coverage}")
    for name in ("frame_accuracy", "refers_to_pending_accuracy", "raw_refers_to_pending_accuracy"):
        metric = report[name]
        print(f'{name}: {metric["correct"]}/{metric["rows"]} ({fraction_text(metric["fraction"])})')
    for name in ("unsafe_direct", "unsafe_direct_dangerous", "unsafe_direct_query_as_write"):
        unsafe_ids = ", ".join(str(row["id"]) for row in report[name])
        print(f'{name}: {len(report[name])}; ids: {unsafe_ids or "none"}')
    print(f'False downgrades: {len(report["false_downgrades"])}')
    for row in report["false_downgrades"]:
        print(f'  {row["id"]}: {json.dumps(row["message"], ensure_ascii=False)}')
    print(f'Violations: {report["violations"]["count"]}; rows: {len(report["violations"]["rows"])}')
    comparison = report["head_to_head"]
    print(f'Head-to-head: model wins={len(comparison["model_right_fast_wrong"])}; fast path wins={len(comparison["fast_right_model_wrong"])}')
    usage = report["token_usage"]
    print(f'Total tokens: prompt={usage["prompt_tokens"]}; completion={usage["completion_tokens"]}; cached={usage["cached_tokens"]}')
    print(f"Report: {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--replay", type=Path)
    mode.add_argument("--live", action="store_true")
    parser.add_argument("--record", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 4:
        parser.error("--concurrency must be between 1 and 4")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.live and args.record is None:
        parser.error("--live requires --record")
    if args.replay and args.record is not None:
        parser.error("--record is only valid with --live")
    rows, paths = load_rows()
    exit_code = 0
    if args.live:
        try:
            configuration = _live_configuration()
        except ValueError as exc:
            parser.error(str(exc))
        active_rows = [(row, is_gold) for row, is_gold in rows if not row["expected"].get("ambiguous", False)]
        if args.limit is not None:
            active_rows = active_rows[:args.limit]
        completed = {}
        try:
            asyncio.run(collect_live(active_rows, configuration, args.concurrency, completed))
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("Live evaluation interrupted; saving completed recordings", file=sys.stderr)
            exit_code = 130
        except Exception as exc:
            print(f"Live evaluation failed ({type(exc).__name__}); saving completed recordings", file=sys.stderr)
            exit_code = 1
        finally:
            ordered = [completed[index] for index in sorted(completed)]
            args.record.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in ordered), encoding="utf-8")
        recordings = {record["id"]: record for record in ordered}
        if args.limit is not None:
            rows = active_rows
    else:
        recordings = load_recordings(args.replay)
        if args.limit is not None:
            parser.error("--limit is only valid with --live")
    report = evaluate(rows, paths, recordings)
    usage = Counter()
    for record in recordings.values():
        for name in ("prompt_tokens", "completion_tokens", "cached_tokens"):
            count = (record.get("usage") or {}).get(name, 0)
            if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                usage[name] += count
    report["mode"] = "live" if args.live else "replay"
    report["token_usage"] = {name: usage[name] for name in ("prompt_tokens", "completion_tokens", "cached_tokens")}
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _print_report(report, args.out)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
