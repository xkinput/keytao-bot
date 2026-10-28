#!/usr/bin/env python3
"""Replay production messages through real chat stages with in-memory tools."""

import argparse
import asyncio
import copy
import json
import os
import socket
import sqlite3
import sys
from collections import Counter
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = Path(__file__).with_name("prod_cases_2026-10-15.jsonl")
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO_ROOT))


@contextmanager
def memory_only_runtime(record_path=None):
    """Keep import-time stores in RAM and reject every unrequested file write."""
    import hashlib
    import tempfile

    connect = sqlite3.connect
    keepers = {}
    allowed = record_path.resolve() if record_path is not None else None

    def memory_connect(database, *args, **kwargs):
        key = str(database)
        uri = "file:prod-replay-" + hashlib.sha256(key.encode()).hexdigest() + "?mode=memory&cache=shared"
        if key not in keepers:
            keepers[key] = connect(uri, uri=True)
        kwargs["uri"] = True
        return connect(uri, *args, **kwargs)

    def forbid_disk_writes(event, args):
        if event == "open":
            path, _mode, flags = args
            if isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                if allowed is not None and isinstance(path, (str, bytes, os.PathLike)) and Path(path).resolve() == allowed:
                    return
                raise AssertionError("Replay forbids disk writes outside --record")
        elif event in {"os.mkdir", "os.chmod", "os.remove", "os.rename", "os.rmdir", "os.truncate"}:
            raise AssertionError("Replay forbids filesystem mutation")

    # This script owns its process; the audit hook also catches import/cache writes.
    sys.addaudithook(forbid_disk_writes)
    try:
        with (
            patch.object(sqlite3, "connect", side_effect=memory_connect),
            patch.object(Path, "mkdir"),
            patch.object(os, "chmod"),
            patch.object(tempfile, "gettempdir", return_value="/private/tmp"),
        ):
            yield
    finally:
        for connection in keepers.values():
            connection.close()


def load_cases(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Cases must be nonempty and have unique ids")
    return rows


class Replay:
    def __init__(self, row, harness, fake_client):
        from keytao_bot.harness.state import MemoryConversationStateStore

        self.row = row
        self.harness = harness
        self.chat = harness.openai_chat_module
        self.commands = harness.chat_commands_module
        self.client = fake_client()
        self.store = MemoryConversationStateStore()
        self.memory = harness.ChatMemoryContext(
            platform="qq", user_id=row["sender"], space_type="group",
            space_id="prod-replay", speaker_name="Offline owner",
        )
        self.author = self.memory.conversation_address
        self.history = []
        self.calls = []
        self.writes = []
        self.trace = []
        self.prior_record = None
        self.prior_snapshot = None
        self.prior_owner = None
        self.prompt = ""
        self.entries = row.get("dictionary", [])
        self.candidate_facts = row.get("candidate_facts", {})
        self.draft = copy.deepcopy(row.get("draft", []))

    def result(self, **changes):
        return {"success": True, "batchId": "offline-batch", "contentVersion": 1,
                "batchUrl": "https://offline.invalid/draft/offline-batch", **changes}

    async def tool(self, name, args, platform, user_id, **kwargs):
        assert (platform, user_id) == ("qq", self.author.actor_id)
        args = copy.deepcopy(args)
        self.calls.append((name, args))
        if name in {"keytao_lookup_by_word", "keytao_lookup_by_code"}:
            field = "word" if name.endswith("word") else "code"
            result = {"success": True, "phrases": [item for item in self.entries if item[field] == args[field]]}
        elif name == "keytao_lookup_by_words_batch":
            result = {"success": True, "results": [
                {"success": True, "word": word, "phrases": [item for item in self.entries if item["word"] == word]}
                for word in args["words"]
            ]}
        elif name == "keytao_pending_items_by_words":
            result = {"success": True, "complete": True, "items": [
                item for item in self.draft if item["word"] in args["words"]
            ]}
        elif name == "keytao_prepare_reviewed_add":
            word = args["word"]
            facts = self.candidate_facts[word]
            statuses = []
            for code in facts["codes"]:
                occupants = [item["word"] for item in self.entries if item["code"] == code]
                statuses.append({"code": code, "occupied": bool(occupants),
                                 "label": "已占用" if occupants else "空位", "words": occupants})
            result = {
                "success": True, "word": word, "type": "Phrase", "reviewDisposition": "PASS",
                "recommendedCode": facts["recommended"], "needsManualReview": False,
                "preSubmitAudit": {"success": True, "verdict": "pass", "autoApprove": True,
                                   "summary": "Offline reviewed fixture", "issues": []},
                "pronunciations": [{"pinyin": facts["pinyin"],
                                    "recommendedCode": facts["recommended"],
                                    "sources": [{"source": "offline"}], "candidateStatuses": statuses}],
            }
        elif name == "keytao_list_draft_items":
            result = self.result(items=self.draft)
        elif name == "keytao_get_draft_batches":
            result = {"success": True, "batches": []}
        elif name in {"keytao_create_phrase", "keytao_batch_add_to_draft"}:
            if not args.get("confirmed"):
                result = self.result(success=False, requiresConfirmation=True,
                                     warningDigest="a" * 64, warnings=[], warnedCount=0)
            else:
                self.writes.append((name, args))
                items = args.get("items", [{"action": "Create", "word": args.get("word"),
                                           "code": args.get("code"), "type": "Phrase"}])
                self.draft.extend(copy.deepcopy(items))
                result = self.result(successCount=len(items), failedCount=0)
        elif name == "keytao_submit_batch":
            self.writes.append((name, args))
            result = self.result(status="pending")
        else:
            raise AssertionError(f"Unexpected offline tool: {name}")
        return json.dumps(result, ensure_ascii=False)

    def remember(self, key, memory, message, response):
        self.history.extend([{"role": "user", "content": message},
                             {"role": "assistant", "content": str(response)}])

    async def seed(self):
        from keytao_bot.harness.conversation import ConversationAddress
        from keytao_bot.harness.state import PendingAddWord, PendingToolConfirm

        ticket = self.row.get("live_ticket")
        if not ticket:
            return
        owner = ConversationAddress.group("qq", self.author.space_id, ticket.get("owner", self.author.actor_id))
        if ticket["type"] == "choices":
            state = PendingToolConfirm("keytao_choice_offer", {"options": copy.deepcopy(ticket["options"])})
            assert self.commands._pending_choice_options(state) is not None
            assert self.store.set(owner, state, space_key=owner.space_key, owner_label="Offline owner")
        elif ticket["type"] == "add":
            pairs = [(code, False) for code in ticket["codes"]]
            state = PendingAddWord(ticket["word"], ticket["recommended"], pairs,
                                   server_candidates=pairs, needs_manual_review=False)
            assert self.store.set(owner, state, space_key=owner.space_key, owner_label="Offline owner")
        else:
            state = PendingToolConfirm(ticket["function_name"], copy.deepcopy(ticket["args"]))
            assert self.store.set(owner, state, space_key=owner.space_key, owner_label="Offline owner")
        record = self.store.get_record(owner)
        self.prompt = (ticket.get("prompt") or self.chat._render_live_single_candidate_record(record)
                       or self.chat._render_live_batch_record(record))
        if not self.prompt and isinstance(record.state, PendingToolConfirm):
            self.prompt = self.commands.render_pending_choice_offer(record.state)
        assert self.prompt, "A live ticket must have a concrete advertised prompt"
        if not ticket.get("legacy", False):
            assert self.store.bind_origin_prompt_digest(owner, self.chat._prompt_capability_digest(self.prompt))
        self.history[:] = [{"role": "assistant", "content": self.prompt}]
        self.prior_record, self.prior_snapshot, self.prior_owner = record, copy.deepcopy(record), owner

    def patches(self):
        from keytao_bot.harness.state import DraftOperationCoordinator

        return (
            patch.dict(os.environ, {"KEYTAO_SEMANTIC_ROUTE_OPS":
                "add,move,recode,swap,delete,reorder,batch_replace,draft_submit,draft_recall,draft_clear,draft_keep_only,query,chat,help,pending_release,pending_answer,direct_exec",
                "KEYTAO_SEMANTIC_SHADOW": "0"}),
            patch.object(self.chat, "AsyncOpenAI", return_value=self.client),
            patch.object(self.chat, "WORD_QUERY_INTENT_MODEL", "offline-model"),
            patch.object(self.chat, "conversation_state_store", self.store),
            patch.object(self.chat, "draft_operation_coordinator", DraftOperationCoordinator()),
            patch.object(self.chat, "call_tool_function", AsyncMock(side_effect=self.tool)),
            patch.object(self.chat, "get_ai_response_core", AsyncMock(return_value=self.row.get("language_reply", "这是离线解释。"))),
            patch.object(self.chat, "get_history", side_effect=lambda *args: self.history),
            patch.object(self.chat, "to_me", return_value=AsyncMock(
                side_effect=lambda bot, event, state: bool(getattr(event, "to_me", False)))),
            patch.object(self.chat, "remember_conversation", side_effect=self.remember),
            patch.object(self.chat, "schedule_memory_compaction"),
            patch.object(self.chat, "build_reply_context", AsyncMock(return_value="")),
            patch.object(self.chat, "_finish_ai_chat_response", AsyncMock()),
            patch.object(self.chat, "_finish_ai_chat_matcher", AsyncMock()),
            patch.object(self.commands.user_resolver, "resolve_actor_binding", AsyncMock(return_value=True)),
            patch.object(self.commands, "_format_draft_response", AsyncMock(return_value="草稿地址：https://offline.invalid/draft/offline-batch")),
        )

    async def turn(self, turn, raw):
        from keytao_bot.harness import semantic_parser

        message = turn["message"]
        ctx = self.chat.TurnContext(
            bot=SimpleNamespace(), event=SimpleNamespace(to_me=turn.get("to_me", False)),
            platform="qq", user_id=self.author.actor_id,
            message_text=message, normalized_message_text=message, conv_key=self.author,
            space_key=self.author.space_key, memory_context=self.memory, owner_label="Offline owner",
            command_intent_for=AsyncMock(return_value=self.chat.MessageCommandIntent()),
        )
        quote = turn.get("quote_target", "none")
        if quote != "none":
            ctx.reply_reference = self.harness.ReplyReferenceInfo(
                is_reply=True, is_to_bot=True, text=self.prompt if quote == "ticket" else quote,
                mentioned_user_ids=(self.prior_owner.actor_id,) if self.prior_owner else (),
            )
        message_token = self.chat._current_turn_message.set(message)
        turn_token = self.chat._current_conversational_turn.set(None)
        try:
            with patch.object(semantic_parser, "parse_operation", AsyncMock(return_value=raw)):
                stages = self.chat.STAGES
                start = next(i for i, stage in enumerate(stages) if stage.__name__ == "_stage_understand_operation_early")
                for stage in stages[start:]:
                    handled = await stage(ctx)
                    self.trace.append(stage.__name__)
                    if handled:
                        break
                assert handled, "No routing stage completed the turn"
        finally:
            self.chat._current_turn_message.reset(message_token)
            self.chat._current_conversational_turn.reset(turn_token)
        self.last_response = str(ctx.response if ctx.response is not None else ctx.scoped_pending_response)
        return self.outcome(ctx)

    def outcome(self, ctx):
        from keytao_bot.harness.state import PendingAddWord, PendingToolConfirm

        creates, deletes = [], []
        for name, args in self.writes:
            if name == "keytao_create_phrase":
                creates.append([args["word"], args["code"]])
            if name == "keytao_batch_add_to_draft":
                for item in args["items"]:
                    (deletes if item.get("action") == "Delete" else creates).append([item["word"], item["code"]])
        record = self.store.get_record(self.author)
        state = record.state if record else None
        candidate_words = []
        pending_op = None
        if isinstance(state, PendingAddWord):
            candidate_words = [state.word]
        elif isinstance(state, PendingToolConfirm):
            if state.function_name == "keytao_batch_add_to_draft":
                candidate_words = list(state.args.get("_query_words", []))
            pending_op = state.args.get("op") or state.function_name
        submissions = sum(name == "keytao_submit_batch" for name, _args in self.writes)
        kind = ("submitted" if submissions else "replaced" if deletes and creates else
                "added" if creates else "candidates" if candidate_words else
                "understood" if pending_op else "query" if ctx.semantic_nonwrite_route == "query" else "chat")
        looked_up = {args["word"] for name, args in self.calls if name == "keytao_lookup_by_word"}
        looked_up.update(word for name, args in self.calls if name == "keytao_lookup_by_words_batch"
                         for word in args["words"])
        existing = [[item["word"], item["code"]] for item in self.entries
                    if item["word"] in getattr(ctx.operation_request, "operands", ())
                    and item["word"] in looked_up
                    and item["code"] in str(ctx.response)]
        return {
            "kind": kind, "added": creates, "deleted": deletes, "candidates": candidate_words,
            "existing": existing, "submissions": submissions,
            "write_count": len(self.writes), "pending_op": pending_op,
            "understood_ticket": self.commands.render_pending_understood_command(state)
                if isinstance(state, PendingToolConfirm) and state.function_name == "keytao_understood_command" else "",
            "op": getattr(ctx.operation_request, "op", None),
            "frame": getattr(ctx.operation_request, "frame", None),
            "other_ticket_preserved": bool(self.prior_owner and self.prior_owner != self.author
                and self.store.get_record(self.prior_owner) is self.prior_record
                and self.prior_record == self.prior_snapshot),
            "old_ticket_replaced": bool(self.prior_record and record is not self.prior_record),
            "explanation": "众多人才" in str(ctx.response),
        }


def mismatches(expected, actual):
    errors = []
    for key, value in expected.items():
        if key == "any_of":
            alternatives = [mismatches(option, actual) for option in value]
            if not any(not errors for errors in alternatives):
                errors.append(f"No expected alternative matched: {alternatives!r}")
        elif key == "kind" and value == "submit_or_ticket":
            if not (actual["kind"] == "submitted" or actual["kind"] == "understood"
                    and actual["pending_op"] in {"draft_submit", "keytao_submit_batch"}
                    and actual["old_ticket_replaced"]):
                errors.append("Expected submission or a replacement submit ticket")
        elif actual.get(key) != value:
            errors.append(f"{key}: expected {value!r}, got {actual.get(key)!r}")
    return errors


async def run(rows, *, live=False, record_path=None):
    # The established offline fixture imports real routing while replacing the
    # NoneBot runtime. Preserve the real HTTP/OpenAI modules for explicit live mode.
    real_modules = {}
    if live:
        import httpx
        import openai
        real_modules = {"httpx": httpx, "openai": openai}
    import test_state_machine as harness
    from test_semantic_parser import FakeClient
    if live:
        sys.modules.update(real_modules)
    from keytao_bot.harness.semantic_parser import parse_operation
    from eval.routing.run_semantic import _UsageClient, _live_configuration

    client = None
    model = "offline-model"
    completed = []
    usage = Counter()
    failed = 0
    skipped = 0
    if live:
        api_key, base_url, model = _live_configuration()
        client = real_modules["openai"].AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=15.0, max_retries=0,
        )
    try:
        for row in rows:
            if live and (row.get("offline_only") or any(marker in row["id"] for marker in ("recorded", "backstop"))):
                skipped += 1
                print(f'SKIP {row["id"]} offline-only model output replay')
                continue
            replay = Replay(row, harness, FakeClient)
            errors = []
            actual = None
            try:
                with ExitStack() as stack:
                    for context in replay.patches():
                        stack.enter_context(context)
                    await replay.seed()
                    for index, turn in enumerate([row, *row.get("followups", [])]):
                        raw = turn["model_raw"]
                        if live:
                            tracked = _UsageClient(client)
                            current = replay.store.get_record(replay.author)
                            pending_context = replay.chat._pending_context_for_command_intent(current.state if current else None)
                            raw = await asyncio.wait_for(parse_operation(
                                turn["message"], pending_context, client=tracked,
                                model=model, timeout=15.0,
                            ), timeout=15.0)
                            usage.update(tracked.usage)
                            completed.append({"id": row["id"], "turn": index,
                                              "message": turn["message"], "pending_context": pending_context,
                                              "raw": raw, "model": model, "usage": dict(tracked.usage)})
                        actual = await replay.turn(turn, raw)
                        errors.extend(f"turn {index}: {error}" for error in mismatches(turn["expected"], actual))
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
            failed += bool(errors)
            print(f'{"FAIL" if errors else "PASS"} {row["id"]} {json.dumps(actual, ensure_ascii=False, sort_keys=True)}')
            for error in errors:
                print("  " + error)
            if errors:
                print("  response=" + getattr(replay, "last_response", ""))
                print("  tools=" + repr(replay.calls))
        print(f"Replay: {len(rows) - skipped - failed}/{len(rows) - skipped} PASS; {failed} FAIL; {skipped} SKIP")
        print(f'Total tokens: total={usage["prompt_tokens"] + usage["completion_tokens"]}; '
              f'prompt={usage["prompt_tokens"]}; completion={usage["completion_tokens"]}; cached={usage["cached_tokens"]}')
        return int(bool(failed))
    finally:
        if client is not None:
            await client.close()
        if live and record_path is not None:
            record_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in completed), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--live", action="store_true", help="Explicitly enable sequential model calls; tools remain in memory")
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()
    if args.live != bool(args.record):
        parser.error("--live requires --record; --record is only valid with --live")
    # Offline mode also denies accidental model/tool traffic below the fakes.
    with ExitStack() as stack:
        stack.enter_context(memory_only_runtime(args.record))
        if not args.live:
            stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Offline replay forbids network")))
            stack.enter_context(patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Offline replay forbids network")))
        return asyncio.run(run(load_cases(args.cases), live=args.live, record_path=args.record))


if __name__ == "__main__":
    raise SystemExit(main())
