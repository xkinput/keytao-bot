"""Shared offline regression fixtures, moved from the root test modules."""

# Source: test_b10_flykey.py
import copy as copy
import json as json
import os as os
import socket as socket
import tempfile as tempfile
import unittest as unittest
from contextlib import ExitStack as ExitStack
from dataclasses import replace as replace
from types import SimpleNamespace as SimpleNamespace
from unittest.mock import AsyncMock as AsyncMock
from unittest.mock import patch as patch
from tests import test_state_machine as harness
from eval.routing.replay_prod_cases import Replay as Replay
from keytao_bot.harness.semantic_parser import validate_operation as validate_operation
from keytao_bot.harness.state import PendingAddWord as PendingAddWord
from keytao_bot.harness.state import PendingToolConfirm as PendingToolConfirm
from keytao_bot.harness.state import create_pending_trusted_word_record as create_pending_trusted_word_record
from keytao_bot.utils import completed_draft_undo as completed_draft_undo
from keytao_bot.utils.draft_mutation_store import DraftMutationClaimStore as DraftMutationClaimStore
from tests.test_semantic_parser import FakeClient as FakeClient
from tests.test_semantic_parser import raw_operation as raw_operation
b10_flykey_WORD = "赵孟頫"

b10_flykey_CODES = ("qmf", "qmfv", "qmfva", "fmf", "fmfv", "fmfva")

FLY_CODES = b10_flykey_CODES[3:]

class FlyKeyReplay(Replay):
    def __init__(self):
        super().__init__({
            "sender": "b10-owner",
            "dictionary": [
                {"word": "找麻烦", "code": "qmf", "type": "Phrase"},
                {"word": "找麻烦", "code": "fmf", "type": "Phrase"},
                {"word": "殖民者", "code": "fmfv", "type": "Phrase"},
            ],
            "candidate_facts": {
                b10_flykey_WORD: {"recommended": "qmfv", "codes": list(b10_flykey_CODES), "pinyin": "zhao meng fu"},
            },
        }, harness, FakeClient)
        self.fly_codes = list(FLY_CODES)

    async def turn(self, turn, raw):
        """Exercise the real serialized message handler with an offline model."""
        message = turn["message"]
        self.client.create.return_value.choices[0].message.content = json.dumps(raw, ensure_ascii=False)
        event = SimpleNamespace(
            to_me=turn.get("to_me", False), message=[],
            get_plaintext=lambda: message,
        )
        reply = self.harness.ReplyReferenceInfo()
        if turn.get("quote_target"):
            reply = self.harness.ReplyReferenceInfo(
                is_reply=True, is_to_bot=True, text=turn["quote_target"],
            )
        self.trace.clear()
        captured = []

        def traced(stage):
            async def run(ctx):
                captured[:] = [ctx]
                self.trace.append(stage.__name__)
                return await stage(ctx)
            return run

        tokens = (
            (self.chat._current_turn_message, self.chat._current_turn_message.set(message)),
            (self.chat._current_conversational_turn, self.chat._current_conversational_turn.set(None)),
            (self.chat.current_operation_capture, self.chat.current_operation_capture.set(None)),
            (self.chat.current_draft_delivery_claims, self.chat.current_draft_delivery_claims.set([])),
        )
        try:
            with (
                patch.object(self.chat, "STAGES", tuple(traced(stage) for stage in self.chat.STAGES)),
                patch.object(self.chat, "extract_memory_context", AsyncMock(return_value=self.memory)),
                patch.object(self.chat, "extract_reply_reference_info", AsyncMock(return_value=reply)),
                patch.object(self.chat, "_classify_message_command_intent", AsyncMock(return_value=self.chat.MessageCommandIntent())),
            ):
                await self.chat._handle_ai_chat_serialized(SimpleNamespace(), event, "qq", self.author.actor_id)
                ctx = captured[-1]
                response = ctx.response if ctx.response is not None else ctx.scoped_pending_response
                self.last_response = self.chat._prepare_user_facing_reply(str(response), self.memory)
        finally:
            for variable, token in reversed(tokens):
                variable.reset(token)
        ctx = captured[-1]
        return self.outcome(ctx)

    async def tool(self, name, args, *actor, **kwargs):
        if name == "keytao_encode":
            self.calls.append((name, copy.deepcopy(args)))
            return json.dumps({
                "success": True, "word": b10_flykey_WORD, "recommendedCode": "qmfv",
                "candidateCodes": list(b10_flykey_CODES), "codes": list(b10_flykey_CODES[:3]),
                "flyKeyVariants": [{"baseCode": "fmf", "codes": self.fly_codes, "changes": []}],
                "candidateStatuses": self.statuses(),
                "phrasePinyins": ["zhao", "meng", "fu"],
            }, ensure_ascii=False)
        before = copy.deepcopy(self.draft)
        previous_writes = len(self.writes)
        result = await super().tool(name, args, *actor, **kwargs)
        if len(self.writes) > previous_writes:
            capture = self.chat.current_operation_capture.get()

            def snapshot(items, version):
                return {"batchId": "offline-batch", "contentVersion": version, "items": [
                    {"id": index, "action": "Create", "type": "Phrase", **item}
                    for index, item in enumerate(items, 1)
                ]}

            self.claim_store.save_completed_operation(capture.scope, "qq", self.author.actor_id, {
                "operationId": capture.operation_id, "completedAt": self.chat.time.time(),
                "before": snapshot(before, 0), "after": snapshot(self.draft, 1),
                "batchId": "offline-batch", "batchUrl": "https://offline.invalid/draft/offline-batch",
                "submitted": name == "keytao_submit_batch", "tools": [name],
            })
            payload = json.loads(result)
            payload.update(
                writtenItems=snapshot(self.draft, 1)["items"][len(before):],
                updatedItems=[],
            )
            self.commands._capture_successful_draft_write_delivery(name, payload, "qq", self.author.actor_id)
            result = json.dumps(payload, ensure_ascii=False)
        if name == "keytao_prepare_reviewed_add":
            result = json.loads(result)
            result["pronunciations"][0]["candidateStatuses"] = self.statuses()
            return json.dumps(result, ensure_ascii=False)
        return result

    def statuses(self):
        return [{
            "code": code,
            "occupied": any(item["code"] == code for item in self.entries),
            "words": [item["word"] for item in self.entries if item["code"] == code],
            "label": "已占用" if any(item["code"] == code for item in self.entries) else "空位",
            "flyKey": code in self.fly_codes,
        } for code in b10_flykey_CODES]

def B10FlyKeyTests_setUp(self):
    self.replay = FlyKeyReplay()
    stack = self.enterContext(ExitStack())
    directory = stack.enter_context(tempfile.TemporaryDirectory())
    self.replay.claim_store = DraftMutationClaimStore(directory + "/claims.db")
    stack.enter_context(patch.object(completed_draft_undo, "get_default_draft_mutation_claim_store", return_value=self.replay.claim_store))
    stack.enter_context(patch.object(self.replay.chat, "get_default_draft_mutation_claim_store", return_value=self.replay.claim_store))
    for context in self.replay.patches():
        stack.enter_context(context)
    self.replay.chat._finish_ai_chat_response.side_effect = lambda *args: completed_draft_undo.note_completed_operation_reply(
        self.replay.chat.current_draft_delivery_claims.get() or [],
    )
    stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")))
    stack.enter_context(patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Offline only")))
    self.general = stack.enter_context(patch.object(
        self.replay.chat, "get_ai_response_core", AsyncMock(return_value="Unexpected general response"),
    ))

async def B10FlyKeyTests_add_fly_key(self, message=f"新增“{b10_flykey_WORD}”的飞键编码"):
    return await self.replay.turn({"message": message, "to_me": True}, raw_operation(
        op="add", operands=[b10_flykey_WORD], codes=[], code_selector="fly_key",
    ))

async def B10FlyKeyTests_assert_pending_code_allows_add(self, status, source):
    self.replay.draft.append({
        "word": b10_flykey_WORD, "code": "qmfv", "type": "Phrase", "action": "Create",
        "source": source, "batchId": "already-submitted", "batchStatus": status,
        "itemStatus": "Pending", "batchUrl": "https://offline.invalid/batch/already-submitted",
    })
    outcome = await self.add_fly_key()
    self.assertEqual(outcome["added"], [[b10_flykey_WORD, "fmfva"]], self.replay.last_response)
    self.assertEqual(self.replay.draft[0]["code"], "qmfv")
    self.general.assert_not_awaited()

# Source: test_b10_track_a.py
import subprocess as subprocess
import sys as sys
from keytao_bot.utils import community_vote as vote
from tests.test_community_vote import submitted_data as submitted_data
# Source: test_b6_policy.py
import sqlite3 as sqlite3
from contextlib import closing as closing
from datetime import datetime as datetime
from datetime import timedelta as timedelta
from pathlib import Path as Path
from keytao_bot.utils.draft_reminder import DAY_SECONDS as DAY_SECONDS
from keytao_bot.utils.draft_reminder import SHANGHAI as SHANGHAI
from keytao_bot.utils.draft_reminder import DraftReminderStore as DraftReminderStore
def B6ReminderTests_setUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.db_path = str(Path(self.directory.name) / "reminders.db")
    self.store = DraftReminderStore(self.db_path)
    self.now = datetime(2026, 10, 27, 10, tzinfo=SHANGHAI).timestamp()
    self.flag = patch.dict(os.environ, {"KEYTAO_DRAFT_REMINDER": "1"})
    self.flag.start()
    self.addCleanup(self.flag.stop)
    self.read = AsyncMock(side_effect=self.snapshot)
    self.send = AsyncMock(return_value="reminder-message")

def B6ReminderTests_write(self, *, batch="b6-draft", version=1, written=None, actor="owner", group="group", submitted=False, items=None):
    self.store.record_write(
        json.dumps(["qq", "group", group, actor]), "qq", actor,
        {
            "operationId": f"{batch}-{version}",
            "completedAt": self.now - DAY_SECONDS if written is None else written,
            "batchId": batch, "batchUrl": f"https://offline.invalid/draft/{batch}",
            "submitted": submitted,
            "after": {"batchId": batch, "contentVersion": version,
                      "items": [{"id": 1, "word": "旧词"}] if items is None else items},
        },
    )

@staticmethod
async def B6ReminderTests_snapshot(candidate):
    return {
        "success": True, "batchId": candidate["batch_id"], "status": "Draft",
        "contentVersion": candidate["content_version"], "items": [{"id": 1}],
        "batchUrl": candidate["batch_url"],
    }

async def B6ReminderTests_scan(self, now=None, store=None):
    return await (store or self.store).scan(self.read, self.send, now=self.now if now is None else now)

def B6SubmitTests_setUp(self):
    from tests import test_state_machine as harness
    from keytao_bot.utils import http_client
    self.chat = harness.openai_chat_module
    self.http = http_client
    self.new = {"id": 2, "action": "Create", "word": "屡有", "code": "llydav", "type": "Phrase"}
    self.old = {"id": 1, "action": "Create", "word": "驴友", "code": "llyd", "type": "Phrase",
                "needsManualReview": True}
    self.preview = {
        "success": False, "requiresConfirmation": True, "batchId": "b6-draft",
        "contentVersion": 8, "snapshotDigest": "a" * 64,
        "warningDigest": "b" * 64, "auditDigest": "c" * 64,
        "snapshotItems": [self.old, self.new], "warnings": [],
        "batchUrl": "https://offline.invalid/batch/b6-draft",
        "autoReview": {"issues": ["需要管理员审核"], "structuralPreReviewLocked": True},
    }
    self.batch = {
        "id": "b6-draft", "contentVersion": 8, "status": "Draft", "creatorId": 17,
        "pullRequests": [{**item, "userId": 17} for item in self.preview["snapshotItems"]],
    }
    self.tool = AsyncMock(side_effect=self.invoke)
    self.details = AsyncMock(return_value={"batch": self.batch})
    for context in (
        patch.object(self.chat, "call_tool_function", self.tool),
        patch.object(self.http, "keytao_json", self.details),
        patch.dict(os.environ, {"KEYTAO_COMMUNITY_VOTE": "0"}),
    ):
        context.start()
        self.addCleanup(context.stop)

async def B6SubmitTests_invoke(self, name, arguments, platform, user_id):
    self.assertEqual((name, platform, user_id), ("keytao_submit_batch", "qq", "abang"))
    if arguments.get("confirmed"):
        self.assertEqual(arguments["batch_id"], "b6-draft")
        self.assertEqual(arguments["expected_content_version"], 8)
        self.assertEqual(arguments["expected_server_snapshot_digest"], "a" * 64)
        self.assertEqual(arguments["expected_warning_digest"], "b" * 64)
        self.assertEqual(arguments["expected_audit_digest"], "c" * 64)
        self.assertNotIn("preview_only", arguments)
        return json.dumps({"success": True, "batchId": "b6-draft", "contentVersion": 8,
                           "batchUrl": self.preview["batchUrl"], "snapshotItems": self.preview["snapshotItems"]})
    return json.dumps(self.preview)

async def B6SubmitTests_submit(self, **kwargs):
    return await self.chat._perform_submit_current_draft(
        "qq", "abang", auto_confirm=True, authorized_items=[self.new], **kwargs,
    )

def B6ReminderIntegrationTests_setUp(self):
    from tests import test_state_machine as harness
    self.harness = harness
    self.chat = harness.openai_chat_module

B7BlockedToolLoopTests_policy_message = "当前编码与已核验的词条不一致，本次未写入。"

def B7BlockedToolLoopTests_orchestrator(self, calls, results=None):
    from tests import test_state_machine as harness
    from keytao_bot.harness.orchestrator import AgentOrchestrator, AgentRuntimeConfig
    from keytao_bot.harness.tools import ToolExecutor

    responses = [SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="tool_calls", message=SimpleNamespace(content="", tool_calls=[
            SimpleNamespace(type="function", id=f"b7-{index}", function=SimpleNamespace(
                name=name, arguments=json.dumps(arguments, ensure_ascii=False),
            )),
        ]),
    )], usage=None) for index, (name, arguments) in enumerate(calls)]
    responses.append(SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="stop", message=SimpleNamespace(content="核验已结束。", tool_calls=[]),
    )], usage=None))
    create = AsyncMock(side_effect=responses)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    executor = ToolExecutor(lambda name: None, frozenset())
    executor.call = AsyncMock(side_effect=[json.dumps(result, ensure_ascii=False) for result in results]
                              if results is not None else None,
                              return_value=json.dumps({
                                  "success": False, "policyBlocked": True,
                                  "blockReason": "binding_incomplete", "message": self.policy_message,
                              }, ensure_ascii=False))
    tools = [{"type": "function", "function": {"name": name, "parameters": {}}}
             for name in dict(calls)]
    agent = AgentOrchestrator(
        client_factory=lambda: client,
        runtime=AgentRuntimeConfig(model="offline-model", max_tokens=1000, temperature=0.2, timeout=1),
        skills_manager=SimpleNamespace(get_skill_instructions=lambda: "", has_tools=lambda: True,
                                      get_tools=lambda: tools),
        tool_executor=executor, state_store=harness.MemoryConversationStateStore(),
        bind_help_text="Offline binding help", system_prompt_core="Offline system prompt",
    )
    return agent, create, executor.call

async def B7BlockedToolLoopTests_run_loop(self, agent):
    from keytao_bot.harness.orchestrator import AgentRequestContext
    termination = {}
    reply = await agent._run_loop(
        "修改编码：我送给为wsgu，纯音乐为jyya",
        AgentRequestContext(platform="qq", user_id="b7-loop", mutations_allowed=True),
        termination_state=termination,
    )
    return reply, termination

# Source: test_b11_shift.py
from keytao_bot.harness.orchestrator import AgentRequestContext as AgentRequestContext
from keytao_bot.harness.tools import ToolContext as ToolContext
from keytao_bot.harness.tools import ToolExecutor as ToolExecutor
from keytao_bot.harness.authorization_grammar import parse_standalone_shift as parse_standalone_shift
from keytao_bot.harness.authorization_grammar import parse_reviewed_shift_words as parse_reviewed_shift_words
from keytao_bot.harness.authorization_grammar import parse_dictionary_recode_command as parse_dictionary_recode_command
b11_shift_WORD = "爱理不理"

AMBIGUOUS_SHIFT_OPERANDS = (
    "怎么", "为什么", "什么", "哪个", "哪里", "谁", "我", "你", "我们", "这个", "它",
    "帮我", "要", "已经", "可以", "能", "会", "需要", "吗", "呢", "一下", "了",
    "怎么操作", "什么词", "哪个词", "已经完成", "帮我把爱理不理", "爱理不理吗",
    "为啥", "咋", "何时", "帮忙", "还要", "一下吧", "到哪", "请问", "想要", "必须要",
    "什么时候", "什么情况下", "哪天", "哪一条", "哪辆车", "向谁", "有多少",
    "能够", "你的", "一下呗", "即将", "将要", "需不需要",
)

class ShiftReplay(Replay):
    turn = FlyKeyReplay.turn

    def __init__(self):
        super().__init__({
            "sender": "b11-owner",
            "dictionary": [{"word": b11_shift_WORD, "code": "xlbl", "type": "Phrase", "weight": 100}],
            "candidate_facts": {
                b11_shift_WORD: {"recommended": "xlbl", "codes": ["xlbl", "xlblu", "xlbluv", "xbluv"],
                       "pinyin": "ai li bu li"},
            },
        }, harness, FakeClient)
        self.shifted = []

    async def tool(self, name, args, *actor, **kwargs):
        if name == "keytao_encode":
            self.calls.append((name, copy.deepcopy(args)))
            codes = {"取消": ["qxxl", "qxxlu"], "确认": ["qrrf", "qrrfu"]}.get(
                args["word"], ["xlbl", "xlblu", "xlbluv"],
            )
            return json.dumps({"success": True, "word": args["word"],
                               "codes": codes, "candidateCodes": codes})
        if name == "keytao_shift_phrase_code":
            self.calls.append((name, copy.deepcopy(args)))
            result = self.result(shiftPlan={"shifted": self.shifted, "items": []})
            if args.get("confirmed_plan_digest"):
                self.writes.append((name, copy.deepcopy(args)))
            else:
                result.update(success=False, requiresConfirmation=True, confirmationKind="shiftPlan",
                              planDigest="b" * 64, warningDigest="c" * 64, warnings=[])
            return json.dumps(result, ensure_ascii=False)
        return await super().tool(name, args, *actor, **kwargs)

def B11ShiftTests_setUp(self):
    self.replay = ShiftReplay()
    stack = self.enterContext(ExitStack())
    directory = stack.enter_context(tempfile.TemporaryDirectory())
    self.claim_store = DraftMutationClaimStore(directory + "/claims.db")
    stack.enter_context(patch.object(completed_draft_undo, "get_default_draft_mutation_claim_store", return_value=self.claim_store))
    stack.enter_context(patch.object(self.replay.chat, "get_default_draft_mutation_claim_store", return_value=self.claim_store))
    for context in self.replay.patches():
        stack.enter_context(context)
    self.replay.chat._finish_ai_chat_response.side_effect = lambda *args: completed_draft_undo.note_completed_operation_reply(
        self.replay.chat.current_draft_delivery_claims.get() or [], args[3],
    )
    stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")))
    stack.enter_context(patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Offline only")))
    self.general = stack.enter_context(patch.object(self.replay.chat, "get_ai_response_core", AsyncMock(return_value="Unexpected general response")))

async def B11ShiftTests_turn(self, message, op="other", codes=(), **changes):
    return await self.replay.turn({"message": message, "to_me": True}, raw_operation(
        op=op, operands=[b11_shift_WORD], codes=list(codes), **changes,
    ))

def B11ShiftTests_assert_shift(self, code, word=b11_shift_WORD):
    self.assertEqual(len(self.replay.writes), 1, self.replay.last_response)
    name, args = self.replay.writes[0]
    self.assertEqual((name, args["word"], args["target_code"]), ("keytao_shift_phrase_code", word, code))
    self.assertTrue(args["confirmed_plan_digest"])
    self.assertIn("https://", self.replay.last_response)
    self.general.assert_not_awaited()

def B11ShiftTests_seed_receipt(self):
    capture = completed_draft_undo.start_operation_turn(self.replay.author)
    self.claim_store.save_completed_operation(capture.scope, "qq", self.replay.author.actor_id, {
        "operationId": capture.operation_id, "completedAt": self.replay.chat.time.time(),
        "lastReplyOperationId": capture.operation_id,
        "before": {"items": []},
        "after": {"items": [{"id": 1, "action": "Create", "word": "电脑端", "code": "dndi"}]},
    })
    self.replay.history[:] = [{"role": "assistant", "content": "✅ 已加入草稿：电脑端 → dndi"}]

B7BlockedToolLoopTests_policy_message = B7BlockedToolLoopTests_policy_message

b11_shift_draft_tools = harness._draft_tools

real_batch_remove = b11_shift_draft_tools.keytao_batch_remove_draft_items

def B11SourceShiftTests_setUp(self):
    self.live = []
    self.snapshot = {
        "success": True, "batchId": "b11-draft", "contentVersion": 7,
        "status": "Draft", "items": [{
            "id": 42, "action": "Create", "word": b11_shift_WORD, "code": "xlbl",
            "type": "Phrase", "weight": 100, "remark": "Original review",
            "needsManualReview": True,
        }],
    }
    self.cleanup_result = None
    self.after_append = None
    stack = self.enterContext(ExitStack())
    stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")))
    stack.enter_context(patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Offline only")))
    stack.enter_context(patch.object(b11_shift_draft_tools, "_fetch_encode_candidates", AsyncMock(return_value={
        "success": True, "candidateCodes": ["xlbl", "xlblu", "xlbluv"],
    })))
    stack.enter_context(patch.object(b11_shift_draft_tools, "_lookup_words_raw", AsyncMock(side_effect=lambda words: {
        "success": True, "results": [{"word": word, "phrases": copy.deepcopy([
            row for row in self.live if row["word"] == word
        ])} for word in words],
    })))
    stack.enter_context(patch.object(b11_shift_draft_tools, "_lookup_codes_raw", AsyncMock(side_effect=lambda codes: {
        "success": True, "results": [{"code": code, "phrases": copy.deepcopy([
            row for row in self.live if row["code"] == code
        ])} for code in codes],
    })))
    stack.enter_context(patch.object(b11_shift_draft_tools, "keytao_list_draft_items", AsyncMock(
        side_effect=lambda *args, **kwargs: copy.deepcopy(self.snapshot),
    )))
    self.append = stack.enter_context(patch.object(b11_shift_draft_tools, "_keytao_strict_batch_add_to_draft",
                                                AsyncMock(side_effect=self.strict)))
    self.remove = stack.enter_context(patch.object(b11_shift_draft_tools, "keytao_batch_remove_draft_items",
                                                AsyncMock(side_effect=self.cleanup)))

async def B11SourceShiftTests_strict(self, platform, actor, items, **kwargs):
    if not kwargs.get("confirmed"):
        return {"success": False, "requiresConfirmation": True,
                "warningDigest": "a" * 64, "warnings": []}
    self.assertEqual(kwargs["expected_content_version"], self.snapshot["contentVersion"])
    for item in items:
        self.snapshot["items"].append({"id": len(self.snapshot["items"]) + 100,
                                       "weight": 100, **copy.deepcopy(item)})
    self.snapshot["contentVersion"] += 1
    result = {"success": True, "batchId": self.snapshot["batchId"],
              "contentVersion": self.snapshot["contentVersion"],
              "successCount": len(items), "message": "Appended"}
    if self.after_append:
        self.after_append(self.snapshot)
    return result

async def B11SourceShiftTests_cleanup(self, platform, actor, ids, **kwargs):
    if self.cleanup_result is not None:
        return self.cleanup_result
    self.assertEqual(ids, [42])
    self.assertEqual(kwargs["expected_content_version"], 8)
    self.assertEqual(kwargs["expected_targets"], [{
        "id": 42, "word": b11_shift_WORD, "code": "xlbl", "action": "Create", "type": "Phrase",
    }])
    self.assertEqual(len(kwargs["expected_target_digest"]), 64)
    self.snapshot["items"] = [row for row in self.snapshot["items"] if row["id"] not in ids]
    self.snapshot["contentVersion"] += 1
    return {"success": True, "batchId": self.snapshot["batchId"],
            "contentVersion": self.snapshot["contentVersion"], "successCount": len(ids),
            "failedCount": 0, "draftItems": copy.deepcopy(self.snapshot["items"])}

async def B11SourceShiftTests_preview(self, source="xlbl", target="xlblu", **kwargs):
    return await b11_shift_draft_tools.keytao_shift_phrase_code("qq", "b11-owner", b11_shift_WORD, target,
                                                _source_code=source, target_type="Phrase", **kwargs)

async def B11SourceShiftTests_confirm(self, preview, source="xlbl", target="xlblu"):
    return await self.preview(source, target, confirmed_plan_digest=preview["planDigest"],
                              batch_id=preview["batchId"],
                              expected_content_version=preview["contentVersion"],
                              expected_warning_digest=preview["warningDigest"])

def B11ChangeSourceTests_setUp(self):
    B11SourceShiftTests_setUp(self)
    self.snapshot["items"][0].update(action="Change", oldWord="旧词")
    self.live = [{"word": "旧词", "code": "xlbl", "type": "Phrase", "weight": 100,
                  "remark": "Original live remark"}]

async def B11ChangeSourceTests_cleanup(self, platform, actor, ids, **kwargs):
    self.assertEqual(ids, [42])
    self.assertEqual(kwargs["expected_content_version"], 8)
    self.assertEqual(kwargs["expected_targets"], [{
        "id": 42, "word": b11_shift_WORD, "code": "xlbl", "action": "Change", "type": "Phrase",
    }])
    self.snapshot["items"] = [row for row in self.snapshot["items"] if row["id"] not in ids]
    self.snapshot["contentVersion"] += 1
    return {"success": True, "batchId": self.snapshot["batchId"],
            "contentVersion": self.snapshot["contentVersion"], "successCount": 1,
            "failedCount": 0, "draftItems": copy.deepcopy(self.snapshot["items"])}

# Source: test_b12_select.py
import re as re
from keytao_bot.utils.explicit_code import parse_explicit_entry_code_request as parse_explicit_entry_code_request
from keytao_bot.utils.pending_confirmation import shift_plan_warnings as shift_plan_warnings
b12_select_FACTS = {
    "荡刀": {"recommended": "dpdzia", "codes": ["dpdz", "dpdzi", "dpdzia"], "pinyin": "dang dao"},
    "打泡": {"recommended": "dspzi", "codes": ["dspz", "dspzi", "dspzia"], "pinyin": "da pao"},
    "正手刀": {"recommended": "qsdzi", "codes": ["qsdz", "qsdzi", "qsdzia"], "pinyin": "zheng shou dao"},
    "反手刀": {"recommended": "fsdzi", "codes": ["fsdz", "fsdzi", "fsdzia"], "pinyin": "fan shou dao"},
    "点刀": {"recommended": "dmdzia", "codes": ["dmdz", "dmdzi", "dmdzia"], "pinyin": "dian dao"},
    "削刀": {"recommended": "xhdzi", "codes": ["xhdz", "xhdzi", "xhdzia"], "pinyin": "xiao dao"},
}

class SelectionReplay(Replay):
    turn = FlyKeyReplay.turn

    def __init__(self):
        dictionary = [("dpdz", "挡道"), ("dpdzi", "当道"), ("dpdzia", "挡刀"),
                      ("dspz", "大炮"), ("xhdz", "学到"), ("xhdzi", "挨到"),
                      ("dmdz", "点到"), ("dmdzi", "电刀"), ("dmdzia", "颠倒")]
        super().__init__({
            "sender": "b12-owner",
            "dictionary": [{"word": word, "code": code, "type": "Phrase"} for code, word in dictionary],
            "candidate_facts": copy.deepcopy(b12_select_FACTS),
        }, harness, FakeClient)
        self.submit_extra_items = []

    async def tool(self, name, args, *actor, **kwargs):
        if name == "keytao_submit_batch" and self.submit_extra_items and not args.get("confirmed"):
            self.calls.append((name, copy.deepcopy(args)))
            return json.dumps(self.result(
                success=False, requiresConfirmation=True, confirmationKind="submit",
                snapshotDigest="d" * 64, warningDigest="e" * 64, auditDigest="f" * 64,
                warnings=[{"warningType": "manual_review", "message": "旧草稿需要确认"}],
                snapshotItems=copy.deepcopy(self.draft + self.submit_extra_items),
            ), ensure_ascii=False)
        before = copy.deepcopy(self.draft)
        count = len(self.writes)
        result = await super().tool(name, args, *actor, **kwargs)
        if len(self.writes) > count:
            capture = self.chat.current_operation_capture.get()

            def snapshot(items, version):
                return {"batchId": "offline-batch", "contentVersion": version, "items": [
                    {"id": index, "action": "Create", "type": "Phrase", **item}
                    for index, item in enumerate(items, 1)
                ]}

            self.claim_store.save_completed_operation(capture.scope, "qq", self.author.actor_id, {
                "operationId": capture.operation_id, "completedAt": self.chat.time.time(),
                "before": snapshot(before, 0), "after": snapshot(self.draft, 1),
                "batchId": "offline-batch", "batchUrl": "https://offline.invalid/draft/offline-batch",
                "submitted": name == "keytao_submit_batch", "tools": [name],
            })
            payload = json.loads(result)
            payload.update(writtenItems=snapshot(self.draft, 1)["items"][len(before):], updatedItems=[])
            self.commands._capture_successful_draft_write_delivery(name, payload, "qq", self.author.actor_id)
            result = json.dumps(payload, ensure_ascii=False)
        return result

def B12SelectionTests_setUp(self):
    self.replay = SelectionReplay()
    stack = self.enterContext(ExitStack())
    directory = stack.enter_context(tempfile.TemporaryDirectory())
    self.replay.claim_store = DraftMutationClaimStore(directory + "/claims.db")
    stack.enter_context(patch.object(completed_draft_undo, "get_default_draft_mutation_claim_store", return_value=self.replay.claim_store))
    stack.enter_context(patch.object(self.replay.chat, "get_default_draft_mutation_claim_store", return_value=self.replay.claim_store))
    for context in self.replay.patches():
        stack.enter_context(context)
    self.replay.chat._finish_ai_chat_response.side_effect = lambda *args: completed_draft_undo.note_completed_operation_reply(
        self.replay.chat.current_draft_delivery_claims.get() or [], args[3],
    )
    stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")))
    stack.enter_context(patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Offline only")))
    self.general = stack.enter_context(patch.object(self.replay.chat, "get_ai_response_core", AsyncMock(return_value="Unexpected general response")))
    stack.enter_context(patch.object(self.replay.commands.keytao_review, "compare_word_commonness", AsyncMock(return_value={})))
    stack.enter_context(patch.object(self.replay.commands.keytao_review, "assess_candidate_chain_commonness", AsyncMock(return_value=[])))

async def B12SelectionTests_turn(self, message, quote="", reply_id=""):
    if reply_id:
        original = self.replay.harness.ReplyReferenceInfo

        def reference(**kwargs):
            return original(**{**kwargs, "message_id": reply_id})

        with patch.object(self.replay.harness, "ReplyReferenceInfo", side_effect=reference):
            return await self.replay.turn({"message": message, "to_me": True, "quote_target": quote}, raw_operation())
    return await self.replay.turn({"message": message, "to_me": True, "quote_target": quote}, raw_operation())

async def B12SelectionTests_candidates(self, *words):
    await self.turn("加词：" + "，".join(words))
    self.assertIsNotNone(self.replay.store.get_record(self.replay.author), self.replay.last_response)
    self.replay.client.create.reset_mock()
    return self.replay.last_response

def B12SelectionTests_assert_offline_route(self):
    self.general.assert_not_awaited()
    self.replay.client.create.assert_not_awaited()

def B12SelectionTests_created(self):
    result = []
    for name, args in self.replay.writes:
        if name == "keytao_create_phrase":
            result.append((args["word"], args["code"]))
        elif name == "keytao_batch_add_to_draft":
            result.extend((item["word"], item["code"]) for item in args["items"] if item.get("action") != "Delete")
    return result

# Source: test_b13_default_reading.py
from keytao_bot.utils import keytao_review as review
def B13DefaultReadingTests_setUp(self):
    self.encode = {
        "success": True, "word": "大肥鱼", "codes": ["dfy", "dfyv", "dfyvo"],
        "candidateCodes": ["dfy", "dfyv", "dfyvo"],
        "pronunciationSource": "pinyin-pro-context", "standardPronunciationStatus": "absent",
        "phrasePinyins": ["dà", "féi", "yú"], "contextPhrasePinyins": ["dà", "féi", "yú"],
        "chars": [
            {"char": char, "pinyin": readings[0], "pinyins": readings,
             "pronunciationLookupStatus": "found"}
            for char, readings in (("大", ["dà", "dài"]), ("肥", ["féi"]), ("鱼", ["yú"]))
        ],
    }
    self.evidence = {
        "success": True, "groups": [], "sources": [], "lookupComplete": False,
        "rejections": [{"reason": "search_anchor_not_exact_word", "sourceId": "hwxnet"}],
        "sourceOutcomes": [{"sourceId": "so.com", "status": "errored"}],
    }
    self.web = {"status": "no_evidence", "candidates": [], "searchComplete": False}
    self.fetch_encode = AsyncMock(return_value=self.encode)
    tools = harness._review_tools
    for context in (
        patch.object(review, "collect_pronunciation_evidence_limited", AsyncMock(return_value=self.evidence)),
        patch.object(review, "fetch_keytao_encode", self.fetch_encode),
        patch.object(review, "lookup_words", AsyncMock(return_value={})),
        patch.object(review, "lookup_codes", AsyncMock(return_value={})),
        patch.object(review, "detect_polyphonic_characters", return_value={0: ("da", "dai")}),
        patch.object(review, "resolve_compositional_pronunciation", return_value=None),
        patch.object(review, "_search_pronunciation_web_evidence", AsyncMock(return_value=self.web)),
        patch.object(review, "_infer_semantic_pronunciation_for_review", AsyncMock(return_value={"accepted": False})),
        patch.object(review, "_infer_entity_knowledge", AsyncMock(return_value={"recognized": False})),
        patch.object(review, "_contextual_pronunciation_group", AsyncMock(return_value=None)),
        patch.object(tools, "_review_config", return_value=None),
        patch.object(tools, "_build_pre_submit_audit", AsyncMock(return_value={"autoApprove": False})),
        patch.object(tools, "assess_candidate_chain_commonness", AsyncMock(return_value=[])),
        patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")),
        patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Offline only")),
    ):
        self.enterContext(context)

# Source: test_b14_multi_select.py
b14_multi_select_FACTS = {
    "许褚": {"recommended": "xljj", "codes": ["xljj"], "pinyin": "xu chu"},
    "胡桃木": {"recommended": "htm", "codes": ["htm"], "pinyin": "hu tao mu"},
    "灵能": {"recommended": "lgnr", "codes": ["lgnr"], "pinyin": "ling neng"},
    "债市": {"recommended": "qheki", "codes": ["qhek", "qheki", "qhekio", "fhek", "fheki", "fhekio"], "pinyin": "zhai shi"},
    "琴酒": {"recommended": "qbjq", "codes": ["qbjq"], "pinyin": "qin jiu"},
    "数仓": {"recommended": "ejcp", "codes": ["ejcp"], "pinyin": "shu cang"},
}

T3 = "「许褚1，胡桃木1，灵能1，债市2和4， 琴酒1，数仓1」"

b14_multi_select_T2 = "为什么喵喵推荐的「债市」是qheki而不是fhek呢？两个都要加还是加一个，这种飞键要怎么加啊"

ANSWER = "qheki 和 fhek 都是「债市」的候选，fhek 是飞键。"

b14_multi_select_SELECTION_ERROR = "候选编号无效或重复，请按原候选列表选择；本次未写入。"

class MultiSelectionReplay(SelectionReplay):
    def __init__(self):
        Replay.__init__(self, {
            "sender": "b14-owner",
            "dictionary": [{"word": "确实", "code": "qhek", "type": "Phrase"}],
            "candidate_facts": copy.deepcopy(b14_multi_select_FACTS),
        }, harness, FakeClient)
        self.submit_extra_items = []

    async def tool(self, name, args, *actor, **kwargs):
        result = await super().tool(name, args, *actor, **kwargs)
        if name == "keytao_prepare_reviewed_add" and args["word"] == "债市":
            payload = json.loads(result)
            for status in payload["pronunciations"][0]["candidateStatuses"]:
                status["flyKey"] = status["code"].startswith("f")
            return json.dumps(payload, ensure_ascii=False)
        return result

def B14MultiSelectionTests_setUp(self):
    self.replay = MultiSelectionReplay()
    stack = self.enterContext(ExitStack())
    directory = stack.enter_context(tempfile.TemporaryDirectory())
    self.replay.claim_store = DraftMutationClaimStore(directory + "/claims.db")
    stack.enter_context(patch.object(completed_draft_undo, "get_default_draft_mutation_claim_store", return_value=self.replay.claim_store))
    stack.enter_context(patch.object(self.replay.chat, "get_default_draft_mutation_claim_store", return_value=self.replay.claim_store))
    for context in self.replay.patches():
        stack.enter_context(context)
    self.replay.chat._finish_ai_chat_response.side_effect = lambda *args: completed_draft_undo.note_completed_operation_reply(
        self.replay.chat.current_draft_delivery_claims.get() or [], args[3],
    )
    stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")))
    stack.enter_context(patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Offline only")))
    self.general = stack.enter_context(patch.object(self.replay.chat, "get_ai_response_core", AsyncMock(return_value=ANSWER)))
    stack.enter_context(patch.object(self.replay.commands.keytao_review, "compare_word_commonness", AsyncMock(return_value={})))
    stack.enter_context(patch.object(self.replay.commands.keytao_review, "assess_candidate_chain_commonness", AsyncMock(return_value=[])))

async def B14MultiSelectionTests_candidates(self, *words):
    self.replay.store.delete(self.replay.author)
    self.replay.history.clear()
    self.replay.writes.clear()
    self.replay.draft.clear()
    self.general.reset_mock()
    await self.replay.turn({"message": "加词：" + "，".join(words), "to_me": True}, raw_operation())
    record = self.replay.store.get_record(self.replay.author)
    self.assertIsNotNone(record, self.replay.last_response)
    self.replay.store.bind_origin_message_ids(record, [1401])
    self.replay.client.create.reset_mock()
    self.general.reset_mock()
    return self.replay.last_response

async def B14MultiSelectionTests_select(self, message, prompt):
    original = self.replay.harness.ReplyReferenceInfo

    def reference(**kwargs):
        return original(**{**kwargs, "message_id": "1401"})

    with patch.object(self.replay.harness, "ReplyReferenceInfo", side_effect=reference):
        return await self.replay.turn({
            "message": message, "to_me": True, "quote_target": prompt,
        }, raw_operation(op="select_candidate", operands=[], codes=[], refers_to_pending=True))

def B14MultiSelectionTests_assert_offline_selection(self):
    self.general.assert_not_awaited()
    self.replay.client.create.assert_not_awaited()

def B14MultiSelectionTests_created(self):
    result = []
    for name, args in self.replay.writes:
        if name == "keytao_create_phrase":
            result.append((args["word"], args["code"]))
        elif name == "keytao_batch_add_to_draft":
            result.extend((item["word"], item["code"]) for item in args["items"] if item.get("action") != "Delete")
    return result

# Source: test_b15_sibling_and_others.py
import asyncio as asyncio
from keytao_bot.harness.state import pending_duplicate_confirmation_is_complete as pending_duplicate_confirmation_is_complete
from keytao_bot.harness.state import pending_duplicate_confirmation_state as pending_duplicate_confirmation_state
from keytao_bot.utils.pending_confirmation import trusted_pending_word_items as trusted_pending_word_items
b15_sibling_and_others_WORDS = ("球坐标系", "柱坐标系")

b15_sibling_and_others_CODES = ["qzbx", "qzbxv", "qzbxvi"]

OTHER_BATCH = "b1500000-0000-4000-8000-000000000007"

OWN_BATCH = "b1500000-0000-4000-8000-000000000042"

def pending_item(word=b15_sibling_and_others_WORDS[0], code="qzbxi", source="others_submitted", **changes):
    return {
        "source": source, "batchId": OTHER_BATCH,
        "batchUrl": f"https://keytao.rea.ink/batch/{OTHER_BATCH}",
        "batchStatus": "Submitted", "itemStatus": "Pending", "action": "Create",
        "word": word, "code": code, "type": "Phrase", **changes,
    }

def notice(item):
    return f"「{item['word']}」已有其他用户提交（→ {item['code']}，审核中）：{item['batchUrl']}"

class SiblingReplay(SelectionReplay):
    def __init__(self):
        Replay.__init__(self, {
            "sender": "b15-owner", "dictionary": [],
            "candidate_facts": {
                word: {"recommended": "qzbx", "codes": list(b15_sibling_and_others_CODES), "pinyin": pinyin}
                for word, pinyin in zip(b15_sibling_and_others_WORDS, ("qiu zuo biao xi", "zhu zuo biao xi"))
            },
        }, harness, FakeClient)
        self.submit_extra_items = []
        self.pending_items = []
        self.pending_complete = True

    async def tool(self, name, args, *actor, **kwargs):
        if name == "keytao_pending_items_by_words":
            self.calls.append((name, copy.deepcopy(args)))
            return json.dumps({
                "success": self.pending_complete, "complete": self.pending_complete,
                "items": [item for item in self.pending_items
                          if item["word"] in args["words"] or item.get("oldWord") in args["words"]],
            }, ensure_ascii=False)
        if name in {"keytao_create_phrase", "keytao_batch_add_to_draft"}:
            items = args.get("items", [{"action": "Create", **args}])
            with patch.object(harness._draft_tools, "keytao_pending_items_by_words", AsyncMock(return_value={
                "success": True, "complete": True, "items": self.pending_items,
            })):
                guard, _ = await harness._draft_tools._pending_create_guard(
                    "qq", self.author.actor_id, items,
                    duplicate_confirmed=args.get("_pending_submitted_confirmed") is True,
                )
            if guard:
                self.calls.append((name, copy.deepcopy(args)))
                return json.dumps(guard, ensure_ascii=False)
        if name == "keytao_submit_batch" and not args.get("confirmed"):
            self.calls.append((name, copy.deepcopy(args)))
            return json.dumps(self.result(
                success=False, requiresConfirmation=True, confirmationKind="submit",
                snapshotDigest="d" * 64, warningDigest="e" * 64, auditDigest="f" * 64,
                warnings=[], snapshotItems=copy.deepcopy(self.draft),
            ), ensure_ascii=False)
        return await super().tool(name, args, *actor, **kwargs)

def B15ChatTests_setUp(self):
    self.replay = SiblingReplay()
    stack = self.enterContext(ExitStack())
    directory = stack.enter_context(tempfile.TemporaryDirectory())
    self.replay.claim_store = DraftMutationClaimStore(directory + "/claims.db")
    stack.enter_context(patch.object(completed_draft_undo, "get_default_draft_mutation_claim_store", return_value=self.replay.claim_store))
    stack.enter_context(patch.object(self.replay.chat, "get_default_draft_mutation_claim_store", return_value=self.replay.claim_store))
    for context in self.replay.patches():
        stack.enter_context(context)
    stack.enter_context(patch.object(self.replay.commands.tool_executor, "call", AsyncMock(side_effect=self.replay.tool)))
    self.replay.chat._finish_ai_chat_response.side_effect = lambda *args: completed_draft_undo.note_completed_operation_reply(
        self.replay.chat.current_draft_delivery_claims.get() or [], args[3],
    )
    stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")))
    stack.enter_context(patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Offline only")))
    self.general = stack.enter_context(patch.object(self.replay.chat, "get_ai_response_core", AsyncMock(return_value="Unexpected general response")))
    stack.enter_context(patch.object(self.replay.commands.keytao_review, "compare_word_commonness", AsyncMock(return_value={})))
    stack.enter_context(patch.object(self.replay.commands.keytao_review, "assess_candidate_chain_commonness", AsyncMock(return_value=[])))

async def B15ChatTests_turn(self, message, quote=""):
    raw = raw_operation(op="draft_submit", operands=[], codes=[]) if message == "提交" else raw_operation()
    return await self.replay.turn({"message": message, "to_me": True, "quote_target": quote}, raw)

async def B15ChatTests_candidates(self, words=b15_sibling_and_others_WORDS):
    await self.turn("加词：" + "，".join(words))
    self.assertIsNotNone(self.replay.store.get_record(self.replay.author), self.replay.last_response)
    self.replay.client.create.reset_mock()
    self.general.reset_mock()
    return self.replay.last_response

def B15ChatTests_created(self):
    items = []
    for name, args in self.replay.writes:
        if name == "keytao_create_phrase":
            items.append((args["word"], args["code"]))
        elif name == "keytao_batch_add_to_draft":
            items.extend((item["word"], item["code"]) for item in args["items"])
    return items

def B15ChatTests_submissions(self):
    return sum(name == "keytao_submit_batch" for name, _ in self.replay.writes)

def B15ChatTests_reset_ticket(self):
    self.replay.store.delete(self.replay.author)
    self.replay.history.clear()
    self.replay.writes.clear()
    self.replay.draft.clear()

def B15PendingLookupTests_setUp(self):
    self.tools = harness._draft_tools
    self.batches = []
    self.details = {}
    self.detail_calls = []
    self.active_details = 0
    self.maximum_active_details = 0
    self.submit_preview = None
    self.submit_status = 400
    self.submit_posts = []
    self.http_posts = []
    stack = self.enterContext(ExitStack())
    stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")))
    stack.enter_context(patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Offline only")))
    stack.enter_context(patch.object(self.tools, "get_bot_token", return_value="offline-token"))
    stack.enter_context(patch.object(self.tools, "get_keytao_url", return_value="https://offline.invalid"))
    stack.enter_context(patch.object(self.tools, "get_bot_headers", return_value={}))
    stack.enter_context(patch.object(self.tools, "make_batch_url", side_effect=lambda batch: f"https://offline.invalid/batch/{batch}"))
    stack.enter_context(patch.object(self.tools, "keytao_list_draft_items", AsyncMock(return_value={"success": True, "items": []})))
    async def request_once(request, **kwargs):
        return await request()

    stack.enter_context(patch.object(self.tools.http_client, "request_with_retries", AsyncMock(side_effect=request_once)))
    owner = self

    class Response:
        def __init__(self, payload, status=200):
            self.status_code = status
            self.payload = payload

        def json(self):
            return self.payload

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, **kwargs):
            owner.http_posts.append(url)
            if url.endswith("/submit"):
                owner.submit_posts.append((url, json.loads(kwargs["content"])))
                return Response(copy.deepcopy(owner.submit_preview), owner.submit_status)
            return Response({"found": True, "user": {"id": 42}})

        async def get(self, url, **kwargs):
            if url.endswith("/api/batches"):
                return Response({"batches": copy.deepcopy(owner.batches), "pagination": {"page": 1, "totalPages": 1}})
            batch_id = url.rsplit("/", 1)[1]
            owner.detail_calls.append(batch_id)
            owner.active_details += 1
            owner.maximum_active_details = max(owner.maximum_active_details, owner.active_details)
            try:
                await asyncio.sleep(0)
                value = owner.details[batch_id]
                if isinstance(value, Exception):
                    raise value
                return Response({"batch": copy.deepcopy(value)})
            finally:
                owner.active_details -= 1

    stack.enter_context(patch.object(self.tools.httpx, "AsyncClient", side_effect=lambda **kwargs: Client(), create=True))

def B15PendingLookupTests_batch(self, batch_id, creator, items):
    self.batches.append({"id": batch_id, "status": "Submitted", "creator": {"id": creator}})
    self.details[batch_id] = {"id": batch_id, "creatorId": creator, "status": "Submitted", "pullRequests": items}

async def B15PendingLookupTests_lookup(self, words=b15_sibling_and_others_WORDS):
    return await self.tools.keytao_pending_items_by_words("qq", "b15-owner", list(words))

# Source: test_b16_b15_leftovers.py
from collections import OrderedDict as OrderedDict
from keytao_bot.harness.orchestrator import AgentOrchestrator as AgentOrchestrator
from keytao_bot.harness.orchestrator import AgentRuntimeConfig as AgentRuntimeConfig
from keytao_bot.plugins.chat_commands import call_tool_function as call_tool_function
from keytao_bot.utils.pending_confirmation import pending_confirmation_copy as pending_confirmation_copy
from keytao_bot.utils.pending_confirmation import render_server_backed_single_word_lookup as render_server_backed_single_word_lookup
FIRST_WORD_BASELINE = (
    "2 个词的候选：\n1. 「球坐标系」\n"
    "词库暂无收录「球坐标系」：\n"
    "审词：读音 qiu zuo biao xi；来源 offline；\n"
    "自动审核：Offline reviewed fixture，可自动通过\n"
    "候选编码:\n1. qzbx — ✅ 推荐（空位）\n"
    "2. qzbxv — 空位\n3. qzbxvi — 空位\n推荐编码：qzbx\n"
)

FOOTER_BASELINE = (
    "回复「加入」写入草稿，或回复「加入并提交」写入并提交。或「取消」。\n"
    "按词选择（词名加空格，再写编号或编码；多词用逗号分隔）：\n"
    "- 「球坐标系 1，柱坐标系 2」\n"
    "可只写其中部分词；选择后即加入草稿，提交需明确回复提交指令。\n"
    "同一词可选多个编码、其余词用推荐：如「球坐标系 1 3，其他用推荐的 加入并提交」\n"
    "也可选择后直接加入并提交：\n"
    "- 「球坐标系 1，柱坐标系 2，加入并提交」"
)

def B16ItemTwoTests_setUp(self):
    B15ChatTests_setUp(self)
    original_tool = self.replay.tool
    self.write_previews = []

    async def tool(name, args, context):
        result = await original_tool(name, args, context.platform, context.user_id)
        if name in {"keytao_create_phrase", "keytao_batch_add_to_draft"}:
            data = json.loads(result)
            if not data.get("pendingDuplicateConfirmation"):
                words = {item["word"] for item in args.get("items", [args])}
                data["pendingItems"] = [item for item in self.replay.pending_items if item["word"] in words]
                if data.get("requiresConfirmation") is True:
                    self.write_previews.append(data)
                result = json.dumps(data, ensure_ascii=False)
        return result

    self.enterContext(patch.object(self.replay.chat, "call_tool_function", call_tool_function))
    self.enterContext(patch.object(self.replay.commands.tool_executor, "call", AsyncMock(side_effect=tool)))
    self.enterContext(patch.object(self.replay.commands, "_reviewed_add_verdicts", OrderedDict()))

async def B16ItemThreeTests_model_submit(self):
    tool_call = SimpleNamespace(
        id="offline-submit", type="function",
        function=SimpleNamespace(name="keytao_submit_batch", arguments="{}"),
    )
    self.model = harness._FakeClient([
        harness._FakeAIResponse("tool_calls", "", [tool_call]),
        harness._FakeAIResponse("stop", pending_confirmation_copy()),
    ])
    receipt_recorder = AsyncMock()
    orchestrator = AgentOrchestrator(
        client_factory=lambda: self.model,
        runtime=AgentRuntimeConfig("offline-model", 1000, 0.0, 10.0),
        skills_manager=SimpleNamespace(
            get_skill_instructions=lambda: "",
            has_tools=lambda: True,
            get_tools=lambda: [{"type": "function", "function": {
                "name": "keytao_submit_batch", "description": "Submit draft",
                "parameters": {"type": "object", "properties": {}},
            }}],
        ),
        tool_executor=self.replay.commands.tool_executor,
        state_store=self.replay.store,
        bind_help_text="offline", system_prompt_core="offline",
        tool_receipt_recorder=receipt_recorder,
    )
    token = self.replay.chat.current_memory_context.set(self.replay.memory)
    try:
        prompt = await orchestrator.run(
            "提交当前草稿",
            AgentRequestContext(
                platform="qq", user_id=self.replay.author.actor_id,
                space_type="group", space_id=self.replay.memory.space_id,
                speaker_name="Offline owner", mutations_allowed=True,
            ),
        )
    finally:
        self.replay.chat.current_memory_context.reset(token)
    self.replay.history[:] = [{"role": "assistant", "content": prompt}]
    self.assertEqual(self.submissions(), 0, prompt)
    record = self.replay.store.get_record(self.replay.author)
    self.assertIsNotNone(record, prompt)
    self.assertEqual(record.state.function_name, "keytao_submit_batch")
    self.assertEqual(len(self.model.completions.calls), 2)
    model_messages = self.model.completions.calls[-1]["messages"]
    result = json.loads(next(message["content"] for message in model_messages
                             if message["role"] == "tool"))
    self.assertTrue(result["requiresConfirmation"])
    self.assertNotIn("pendingItems", result)
    receipt_recorder.assert_awaited_once()
    self.assertNotIn("pendingItems", receipt_recorder.await_args.args[3])
    return prompt

# Source: test_b17_rest_recommended.py
from keytao_bot.harness import authorization_grammar as grammar
from keytao_bot.harness.operation_request import understand_message as understand_message
from keytao_bot.harness.state import _pending_add_word_payload as _pending_add_word_payload
from keytao_bot.plugins import chat_commands as b17_rest_recommended_commands
from keytao_bot.plugins import chat_routing as b17_rest_recommended_routing
b17_rest_recommended_FACTS = {
    "脱骨": {"recommended": "togju", "codes": ["togj", "togju"], "pinyin": "tuo gu"},
    "去骨": {"recommended": "qjgju", "codes": ["qjgj", "qjgju"], "pinyin": "qu gu"},
    "名创": {"recommended": "mgwx", "codes": ["mgwx", "mgwxi", "mgwxio", "mgwm"], "pinyin": "ming chuang"},
    "足光散": {"recommended": "zjgsi", "codes": ["zjgs", "zjgsi"], "pinyin": "zu guang san"},
    "爆汁": {"recommended": "bzqki", "codes": ["bzqk", "bzqki"], "pinyin": "bao zhi"},
    "脆香米": {"recommended": "cxmji", "codes": ["cxmj", "cxmji"], "pinyin": "cui xiang mi"},
    "尾货": {"recommended": "wwhoi", "codes": ["wwho", "wwhoi"], "pinyin": "wei huo"},
    "刷头": {"recommended": "eqtdi", "codes": ["eqtd", "eqtdi"], "pinyin": "shua tou"},
    "华夫饼": {"recommended": "hfbgi", "codes": ["hfbg", "hfbgi"], "pinyin": "hua fu bing"},
    "脆柿": {"recommended": "cvek", "codes": ["cvek", "cveki"], "pinyin": "cui shi"},
    "圣者": {"recommended": "eger", "codes": ["eger", "egeri", "egerio", "egeriu", "egeruv"], "pinyin": "sheng zhe"},
    "屁垫": {"recommended": "pbdm", "codes": ["pbdm", "pbdmi"], "pinyin": "pi dian"},
    "饰冠": {"recommended": "ekgr", "codes": ["ekgr", "ekgri"], "pinyin": "shi guan"},
    "冠饰": {"recommended": "grek", "codes": ["grek", "greki"], "pinyin": "guan shi"},
    "球坐标系": {"recommended": "qzbx", "codes": ["qzbx", "qzbxv", "qzbxvi"], "pinyin": "qiu zuo biao xi"},
    "柱坐标系": {"recommended": "qzbx", "codes": ["qzbx", "qzbxv", "qzbxvi"], "pinyin": "zhu zuo biao xi"},
}

b17_rest_recommended_TEN_WORDS = tuple(b17_rest_recommended_FACTS)[:10]

b17_rest_recommended_SELECTION_ERROR = "候选编号无效或重复，请按原候选列表选择；本次未写入。"

class RestReplay(SelectionReplay):
    def __init__(self):
        Replay.__init__(self, {
            "sender": "b17-owner", "dictionary": [], "candidate_facts": copy.deepcopy(b17_rest_recommended_FACTS),
        }, harness, FakeClient)
        self.submit_extra_items = []
        self.review_overrides = {}

    async def turn(self, turn, raw):
        # OneBot get_plaintext excludes reply/at segments from wire messages.
        turn = {**turn, "message": re.sub(r"^(?:\[reply:id=\d+\]|\[at:qq=\d+\])+\s*", "", turn["message"])}
        return await super().turn(turn, raw)

    async def tool(self, name, args, *actor, **kwargs):
        result = await super().tool(name, args, *actor, **kwargs)
        if name == "keytao_prepare_reviewed_add":
            payload = json.loads(result)
            payload.update(copy.deepcopy(self.review_overrides.get(args["word"], {})))
            if args["word"] == "名创":
                for status in payload["pronunciations"][0]["candidateStatuses"]:
                    status["flyKey"] = status["code"] == "mgwm"
            return json.dumps(payload, ensure_ascii=False)
        return result

def positive_parser_case(message, action):
    def test(self):
        parsed = grammar.parse_reviewed_selection_command(message)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.pairs, (("名创", "1"), ("名创", "4")))
        self.assertEqual(parsed.action, action)
        self.assertIs(parsed.rest_recommended, True)
    return test

def negative_parser_case(message):
    def test(self):
        self.assertIsNone(grammar.parse_reviewed_selection_command(message))
    return test

def b17_rest_recommended_shift_ticket():
    items, scopes = [], []
    for word, code, candidates, occupied, assessments in (
        ("名创", "mgwx", [("mgwx", False), ("mgwm", False)], {}, []),
        ("刷头", "eqtd", [("eqtd", True), ("eqtdi", False)], {"eqtd": ["占位词"]}, [{
            "verdict": "front_more_common", "newWord": "刷头", "newCode": "eqtd",
            "occupantWord": "占位词", "occupantCode": "eqtd", "freeCode": "eqtdi",
        }]),
    ):
        reviewed = harness.PendingAddWord(
            word, code, candidates, server_candidates=candidates,
            server_occupied_words=occupied, server_ordering_assessments=assessments,
            pronunciation_codes={candidate: "offline reading" for candidate, _ in candidates},
            code_remarks={candidate: "Sealed " + word for candidate, _ in candidates},
            needs_manual_review=False,
        )
        items.append({"action": "Create", "word": word, "code": code, "type": "Phrase", "remark": "Sealed " + word})
        scopes.append({
            "word": word, "candidates": candidates, "occupiedWords": occupied,
            "orderingAssessments": assessments, "reviewedState": _pending_add_word_payload(reviewed),
            "reviewedPrompt": "Offline authoritative review",
        })
    return harness.PendingToolConfirm("keytao_batch_add_to_draft", {
        "items": items, "_candidate_scopes": scopes, "_reviewed_multi_word": True,
        "_query_words": [item["word"] for item in items],
    })

def B17RestRouteTests_setUp(self):
    self.replay = RestReplay()
    stack = self.enterContext(ExitStack())
    directory = stack.enter_context(tempfile.TemporaryDirectory())
    self.replay.claim_store = DraftMutationClaimStore(directory + "/claims.db")
    stack.enter_context(patch.object(completed_draft_undo, "get_default_draft_mutation_claim_store", return_value=self.replay.claim_store))
    stack.enter_context(patch.object(self.replay.chat, "get_default_draft_mutation_claim_store", return_value=self.replay.claim_store))
    for context in self.replay.patches():
        stack.enter_context(context)
    self.replay.chat._finish_ai_chat_response.side_effect = lambda *args: completed_draft_undo.note_completed_operation_reply(
        self.replay.chat.current_draft_delivery_claims.get() or [], args[3],
    )
    stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")))
    stack.enter_context(patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Offline only")))
    self.general = stack.enter_context(patch.object(self.replay.chat, "get_ai_response_core", AsyncMock(return_value="Offline general response")))
    stack.enter_context(patch.object(self.replay.commands.keytao_review, "compare_word_commonness", AsyncMock(return_value={})))
    stack.enter_context(patch.object(self.replay.commands.keytao_review, "assess_candidate_chain_commonness", AsyncMock(return_value=[])))

async def B17RestRouteTests_candidates(self, *words):
    self.replay.store.delete(self.replay.author)
    self.replay.history.clear()
    self.replay.writes.clear()
    self.replay.draft.clear()
    await self.replay.turn({"message": "加词：" + "，".join(words), "to_me": True}, raw_operation())
    record = self.replay.store.get_record(self.replay.author)
    self.assertIsNotNone(record, self.replay.last_response)
    self.replay.store.bind_origin_message_ids(record, [1701])
    self.replay.client.create.reset_mock()
    self.general.reset_mock()
    return self.replay.last_response

async def B17RestRouteTests_select(self, message, prompt=""):
    original = self.replay.harness.ReplyReferenceInfo

    def reference(**kwargs):
        return original(**{**kwargs, "message_id": "1701"})

    with patch.object(self.replay.harness, "ReplyReferenceInfo", side_effect=reference):
        return await self.replay.turn({"message": message, "to_me": True, "quote_target": prompt}, raw_operation(
            op="select_candidate", operands=[], codes=[], refers_to_pending=True,
        ))

def B17RestRouteTests_created(self):
    items = []
    for name, args in self.replay.writes:
        if name == "keytao_create_phrase":
            items.append((args["word"], args["code"]))
        elif name == "keytao_batch_add_to_draft":
            items.extend((item["word"], item["code"]) for item in args["items"] if item.get("action") != "Delete")
    return items

def B17RestRouteTests_submissions(self):
    return sum(name == "keytao_submit_batch" for name, _ in self.replay.writes)

def B17RestRouteTests_assert_offline(self):
    self.general.assert_not_awaited()
    self.replay.client.create.assert_not_awaited()

async def B17RestRouteTests_check_failed_then_corrected(self, failure):
    prompt = await self.candidates(*b17_rest_recommended_TEN_WORDS)
    record = self.replay.store.get_record(self.replay.author)
    before = copy.deepcopy(record)
    await self.select(failure, prompt)
    self.assertEqual(self.replay.writes, [])
    self.assertIs(self.replay.store.get_record(self.replay.author), record)
    self.assertEqual(record, before)
    await self.select("名创 1 4，加入并提交")
    self.assertEqual(self.created(), [("名创", "mgwx"), ("名创", "mgwm")], self.replay.last_response)
    self.assertEqual(self.submissions(), 1)
    self.assertEqual(len(self.replay.store.get_record(self.replay.author).state.args["_query_words"]), 9)
    self.assert_offline()

# Source: test_b18_rest_real_tools.py
from urllib.parse import urlsplit as urlsplit
from keytao_bot.utils.pending_confirmation import render_server_backed_single_word_candidates as render_server_backed_single_word_candidates
TOOLS = harness._draft_tools

REAL_CALL_TOOL = b17_rest_recommended_commands.call_tool_function

REAL_FORMAT_DRAFT = b17_rest_recommended_commands._format_draft_response

b18_rest_real_tools_FACTS = {
    "小美": ("xmww", ["xmww", "xmwwi"], "xiao mei"),
    "小帅": ("xkeu", ["xkeu", "xkeui"], "xiao shuai"),
    "腰线": ("yzxm", ["yzxm", "yzxmi"], "yao xian"),
    "招笑": ("qzxc", ["qzxc", "qzxci", "qzxciv", "fzxc", "fzxci", "fzxciv"], "zhao xiao"),
    "宜居": ("ykjlo", ["ykjl", "ykjlo", "ykjloa"], "yi ju"),
    "挥拍": ("hvpdi", ["hvpd", "hvpdi"], "hui pai"),
    "对练": ("dvlmi", ["dvlm", "dvlmi"], "dui lian"),
    "蹭蹭": ("crcri", ["crcr", "crcri"], "ceng ceng"),
    "蹲蹲": ("dwdwi", ["dwdw", "dwdwi"], "dun dun"),
    "停售": ("tgek", ["tgek", "tgeki"], "ting shou"),
    "马哲": ("msqe", ["msqe", "msqea", "msqeai", "msfe", "msfea"], "ma zhe"),
}

b18_rest_real_tools_TEN_WORDS = tuple(b18_rest_real_tools_FACTS)[:10]

def reviewed_ticket(words, reorder=True):
    items, scopes = [], []
    for word in words:
        code, codes, pinyin = b18_rest_real_tools_FACTS[word]
        occupied = {"ykjlo": ["义举"]} if reorder and word == "宜居" else {}
        candidates = [(value, value in occupied) for value in codes]
        assessments = [{"verdict": "front_more_common", "newWord": word,
                        "newCode": code, "occupantWord": "义举", "occupantCode": code,
                        "freeCode": "ykjloa"}] if occupied else []
        reviewed = harness.PendingAddWord(
            word, code, candidates, server_candidates=candidates,
            server_occupied_words=occupied, server_ordering_assessments=assessments,
            pronunciation_codes={value: pinyin for value in codes},
            code_remarks={value: "Offline sealed review" for value in codes},
            needs_manual_review=True,
        )
        items.append({"action": "Create", "word": word, "code": code, "type": "Phrase",
                      "remark": "Offline sealed review", "needsManualReview": True})
        scopes.append({"word": word, "candidates": candidates, "occupiedWords": occupied,
                       "orderingAssessments": assessments, "reviewedState": _pending_add_word_payload(reviewed),
                       "reviewedPrompt": render_server_backed_single_word_candidates(
                           word, code, candidates, occupied, assessments,
                       )})
    return harness.PendingToolConfirm("keytao_batch_add_to_draft", {
        "items": items, "_candidate_scopes": scopes, "_reviewed_multi_word": True,
        "_query_words": list(words),
    })

class OfflineResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status
        self.is_success = 200 <= status < 300
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return copy.deepcopy(self.payload)

class b18_rest_real_tools_DraftHTTP:
    """Local keytao-next route contracts; only confirmed HTTP writes mutate rows."""

    def __init__(self):
        self.rows, self.requests, self.writes = [], [], []
        self.version = 0
        self.submissions = 0
        self.errors = []
        self.batch_id = "offline-batch"
        self.entries = [{"word": "义举", "code": "ykjlo", "type": "Phrase", "weight": 100}]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def get(self, url, **kwargs):
        return self._response("GET", url, kwargs)

    async def post(self, url, **kwargs):
        return self._response("POST", url, kwargs)

    async def request(self, method, url, **kwargs):
        return self._response(method, url, kwargs)

    def _response(self, method, url, kwargs):
        path = urlsplit(url).path
        body = json.loads(kwargs["content"]) if kwargs.get("content") is not None else kwargs.get("json") or {}
        params = kwargs.get("params") or {}
        self.requests.append((method, path, copy.deepcopy(body or params)))
        common = {"success": True, "batchId": self.batch_id, "contentVersion": self.version}
        if path.endswith("/latest-draft"):
            return OfflineResponse({**common, "exists": True, "pullRequestCount": len(self.rows), "isNew": False})
        if path.endswith("/latest-draft/items"):
            return OfflineResponse({**common, "items": self.rows, "count": len(self.rows)})
        if path.startswith("/api/pull-requests/"):
            row = next(row for row in self.rows if row["id"] == int(path.rsplit("/", 1)[1]))
            return OfflineResponse({"pullRequest": {**row, "batchId": self.batch_id,
                "batch": {"id": self.batch_id, "status": "Submitted" if self.submissions else "Draft"}}})
        if path.endswith("/preview"):
            changes = [{"phraseType": row["type"], "codes": [row["code"]],
                        "diffs": [{"type": "remove" if row["action"] == "Delete" else "add", "phrase": row}],
                        "before": [row] if row["action"] == "Delete" else [],
                        "after": [] if row["action"] == "Delete" else [row], "beforeStartLine": 0}
                       for row in self.rows]
            return OfflineResponse({"preview": {"changes": changes, "rejected": [], "summary": {
                "added": sum(row["action"] == "Create" for row in self.rows), "modified": 0,
                "deleted": sum(row["action"] == "Delete" for row in self.rows), "rejected": 0,
            }}})
        if path.endswith("/user/find"):
            return OfflineResponse({"found": True, "user": {"id": 42}})
        if path == "/api/batches":
            return OfflineResponse({"batches": [], "pagination": {"page": 1, "totalPages": 0}})
        if path.endswith("/by-word/batch") or path.endswith("/by-code/batch"):
            field = "word" if path.endswith("/by-word/batch") else "code"
            values = body[field + "s"]
            return OfflineResponse({"success": True, "count": len(values), "results": [
                {field: value, "phrases": [row for row in self.entries if row[field] == value]} for value in values
            ]})
        if path == "/api/phrases/encode":
            word = params["word"]
            codes = ["ykjl", "ykjlo", "ykjloa"] if word == "义举" else b18_rest_real_tools_FACTS[word][1]
            return OfflineResponse({"input": word, "type": "二字词", "chars": [], "codes": codes,
                                    "altCodes": [], "flyKeyVariants": [], "phrasePinyins": ["yi ju"],
                                    "pronunciationSource": "zdic-phrase", "standardPronunciationStatus": "found",
                                    "semanticPronunciationNeeded": False, "semanticPronunciationAccepted": False})
        if path.endswith("/pull-requests/batch") or path.endswith("/pull-requests/batch-draft"):
            if not body.get("confirmed"):
                payload = {**common, "success": False, "requiresConfirmation": True,
                           "warningDigest": "a" * 64, "warnings": [], "message": "请确认将修改写入草稿"}
                if path.endswith("batch-draft"):
                    payload.update(successCount=0, failedCount=0, skippedCount=0, warnedCount=0,
                                   failed=[], skipped=[], warned=[], draftItems=[], draftTotal=0)
                return OfflineResponse(payload, 400 if path.endswith("batch-draft") else 200)
            assert body.get("expectedContentVersion") == self.version, body
            assert body.get("expectedWarningDigest") == "a" * 64, body
            keys = [(row.get("type", "Phrase"), row["code"], row["word"]) for row in body["items"]]
            assert len(keys) == len(set(keys)), keys
            written = []
            for item in body["items"]:
                row = {**copy.deepcopy(item), "id": len(self.rows) + 1, "status": "Pending"}
                row.setdefault("needsManualReview", True)
                row.setdefault("weight", 100)
                self.rows.append(row)
                written.append(row)
            self.version += 1
            self.writes.append((path, copy.deepcopy(body)))
            payload = {**common, "contentVersion": self.version, "message": "成功创建批次，包含修改提议"}
            if path.endswith("batch-draft"):
                payload.update(successCount=len(written), failedCount=0, skippedCount=0,
                               warnedCount=0, failed=[], skipped=[], warned=[],
                               draftItems=self.rows, draftTotal=len(self.rows))
            else:
                payload["pullRequestCount"] = len(written)
            return OfflineResponse(payload)
        if path.endswith("/submit"):
            if not body.get("confirmed"):
                return OfflineResponse({**common, "success": False, "requiresConfirmation": True,
                                        "warningDigest": "a" * 64, "snapshotDigest": "b" * 64,
                                        "warnings": [], "message": "批次检查完成，请确认后提交"}, 400)
            assert body.get("expectedContentVersion") == self.version, body
            assert body.get("expectedSnapshotDigest") == "b" * 64, body
            assert body.get("expectedWarningDigest") == "a" * 64, body
            self.submissions += 1
            self.version += 1
            self.writes.append((path, copy.deepcopy(body)))
            return OfflineResponse({"success": True, "message": "批次已提交审核", "batch": {
                "id": self.batch_id, "status": "Submitted", "contentVersion": self.version,
            }})
        self.errors.append((method, path, body, params))
        raise AssertionError((method, path, body, params))

class RealToolReplay(Replay):
    turn = FlyKeyReplay.turn

def B18RealToolTests_setUp(self):
    self.replay = RealToolReplay({"sender": "b18-owner", "dictionary": []}, harness, FakeClient)
    self.http = b18_rest_real_tools_DraftHTTP()
    stack = self.enterContext(ExitStack())
    directory = stack.enter_context(tempfile.TemporaryDirectory())
    self.claim_store = DraftMutationClaimStore(directory + "/claims.db")
    stack.enter_context(patch.object(completed_draft_undo, "get_default_draft_mutation_claim_store", return_value=self.claim_store))
    stack.enter_context(patch.object(self.replay.chat, "get_default_draft_mutation_claim_store", return_value=self.claim_store))
    stack.enter_context(patch.object(TOOLS, "get_default_draft_mutation_claim_store", return_value=self.claim_store))
    for context in self.replay.patches():
        stack.enter_context(context)
    self.replay.chat._finish_ai_chat_response.side_effect = lambda *args: completed_draft_undo.note_completed_operation_reply(
        self.replay.chat.current_draft_delivery_claims.get() or [], args[3],
    )
    # Dispatch and tool implementation stay real; the transport is the fake boundary.
    stack.enter_context(patch.object(self.replay.chat, "call_tool_function", REAL_CALL_TOOL))
    stack.enter_context(patch.object(b17_rest_recommended_commands, "_format_draft_response", REAL_FORMAT_DRAFT))
    stack.enter_context(patch.object(b17_rest_recommended_commands.tool_executor, "_get_tool_function", TOOLS.TOOL_FUNCTIONS.get))
    stack.enter_context(patch.object(TOOLS.httpx, "AsyncClient", side_effect=lambda **_kwargs: self.http, create=True))
    stack.enter_context(patch.object(TOOLS.http_client, "_keytao_client", self.http))
    stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")))
    stack.enter_context(patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Offline only")))

def B18RealToolTests_tearDown(self):
    self.assertEqual(self.http.errors, [], "Incomplete offline HTTP contract")

def B18RealToolTests_seed(self, words=b18_rest_real_tools_TEN_WORDS, reorder=True):
    ticket = reviewed_ticket(words, reorder)
    self.assertTrue(self.replay.store.set(self.replay.author, ticket))
    record = self.replay.store.get_record(self.replay.author)
    self.replay.prompt = self.replay.chat._render_live_batch_record(record)
    self.assertTrue(self.replay.prompt)
    self.replay.history[:] = [{"role": "assistant", "content": self.replay.prompt}]
    return record

async def B18RealToolTests_turn(self, message, raw=None):
    return await self.replay.turn({"message": message, "to_me": True}, raw or raw_operation(
        op="select_candidate", refers_to_pending=True,
    ))

def B18RealToolTests_created(self):
    return [(row["word"], row["code"]) for row in self.http.rows if row["action"] == "Create"]

def B18ShiftValidationTests_setUp(self):
    self.enterContext(patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")))
    self.enterContext(patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Offline only")))

# Source: test_b19_partial_reply_binding.py
from types import ModuleType as ModuleType
b19_partial_reply_binding_chat = harness.openai_chat_module

REAL_FINISH = b19_partial_reply_binding_chat._finish_ai_chat_response

REAL_MATCHER = b19_partial_reply_binding_chat._finish_ai_chat_matcher

b19_partial_reply_binding_WORDS = ("马哲", *(word for word in b18_rest_real_tools_TEN_WORDS if word != "宜居"))

REST = b19_partial_reply_binding_WORDS[1:]

REFUSAL = "这条回复未对应当前待确认操作；请引用当前候选消息后再回复。"

class OfflineQQBot:
    __module__ = "nonebot.adapters.onebot.v11"

    def __init__(self):
        self.self_id = "3785773770"
        self.messages = {}
        self.next_id = 1900
        self.send = AsyncMock(side_effect=self.deliver)
        self.get_msg = AsyncMock(side_effect=lambda message_id: copy.deepcopy(self.messages[str(message_id)]))

    async def deliver(self, *, event, message):
        message_id = str(self.next_id)
        self.next_id += 1
        self.messages[message_id] = {
            "sender": {"user_id": self.self_id}, "message": str(message),
        }
        self.last_id = message_id
        return {"message_id": int(message_id)}

class b19_partial_reply_binding_DraftHTTP(b18_rest_real_tools_DraftHTTP):
    def _response(self, method, url, kwargs):
        if method == "GET" and urlsplit(url).path == f"/api/batches/{self.batch_id}":
            self.requests.append((method, urlsplit(url).path, {}))
            return OfflineResponse({"batch": {
                "id": self.batch_id, "status": "Draft", "creatorId": 42,
                "contentVersion": self.version,
                "pullRequests": [{**row, "userId": 42} for row in self.rows],
            }})
        return super()._response(method, url, kwargs)

def B19PartialReplyBindingTests_setUp(self):
    B18RealToolTests_setUp(self)
    self.http = b19_partial_reply_binding_DraftHTTP()
    self.enterContext(patch.object(TOOLS.http_client, "_keytao_client", self.http))
    self.bot = OfflineQQBot()
    module = ModuleType("nonebot.adapters.onebot.v11")
    module.Bot = OfflineQQBot
    self.enterContext(patch.dict("sys.modules", {module.__name__: module}))
    self.enterContext(patch.object(b19_partial_reply_binding_chat, "_finish_ai_chat_response", REAL_FINISH))
    self.enterContext(patch.object(b19_partial_reply_binding_chat, "_finish_ai_chat_matcher", REAL_MATCHER))
    self.enterContext(patch.object(b19_partial_reply_binding_chat.ai_chat, "send", AsyncMock(side_effect=self.matcher_send), create=True))
    self.enterContext(patch.object(b19_partial_reply_binding_chat.ai_chat, "finish", AsyncMock(side_effect=self.matcher_finish)))
    self.info = self.enterContext(patch.object(b19_partial_reply_binding_chat.logger, "info", wraps=b19_partial_reply_binding_chat.logger.info))

async def B19PartialReplyBindingTests_matcher_send(self, response):
    return await self.bot.send(event=None, message=response)

async def B19PartialReplyBindingTests_matcher_finish(self, response=None):
    if response is not None:
        return await self.matcher_send(response)

def B19PartialReplyBindingTests_tearDown(self):
    B18RealToolTests_tearDown(self)

def B19PartialReplyBindingTests_created(self):
    return B18RealToolTests_created(self)

async def B19PartialReplyBindingTests_seed(self, words=b19_partial_reply_binding_WORDS):
    record = B18RealToolTests_seed(self, words, reorder=False)
    await REAL_FINISH(self.bot, SimpleNamespace(message_id=1), self.replay.author.actor_id,
                      self.replay.memory, self.replay.prompt)
    self.original_id = self.bot.last_id
    self.assertEqual(record.origin_message_ids, (self.original_id,))
    self.assertTrue(any("branch=send_backed" in str(call.args[0]) for call in self.info.call_args_list))
    return record

async def B19PartialReplyBindingTests_turn(self, message, reply_id, actor=None, raw=None):
    actor = actor or self.replay.author.actor_id
    memory = replace(self.replay.memory, user_id=actor)
    full_message = f"[reply:id={reply_id}][at:qq={self.bot.self_id}] {message}"
    event = SimpleNamespace(
        to_me=True, group_id=self.replay.author.space_id, user_id=actor, message_id=2,
        raw_message=full_message,
        message=[SimpleNamespace(type="reply", data={"id": reply_id}),
                 SimpleNamespace(type="at", data={"qq": self.bot.self_id}),
                 SimpleNamespace(type="text", data={"text": message})],
        get_plaintext=lambda: message,
    )
    tokens = [(variable, variable.set(value)) for variable, value in (
        (b19_partial_reply_binding_chat._current_turn_message, full_message),
        (b19_partial_reply_binding_chat._current_conversational_turn, None),
        (b19_partial_reply_binding_chat.current_operation_capture, None),
        (b19_partial_reply_binding_chat.current_draft_delivery_claims, []),
    )]
    self.replay.client.create.return_value.choices[0].message.content = json.dumps(
        raw or raw_operation(op="cancel" if message == "取消" else "confirm" if message in (
            "加入", "加入并提交", "其它的用默认编码",
        ) else "select_candidate", refers_to_pending=True, operands=[], codes=[]), ensure_ascii=False,
    )
    before = self.bot.send.await_count
    try:
        with patch.object(b19_partial_reply_binding_chat, "extract_memory_context", AsyncMock(return_value=memory)), patch.object(
            b19_partial_reply_binding_chat, "_classify_message_command_intent", AsyncMock(return_value=b19_partial_reply_binding_chat.MessageCommandIntent()),
        ):
            await b19_partial_reply_binding_chat._handle_ai_chat_serialized(self.bot, event, "qq", actor)
    finally:
        for variable, token in reversed(tokens):
            variable.reset(token)
    self.assertEqual(self.bot.send.await_count, before + 1)
    self.bot.get_msg.assert_awaited_with(message_id=int(reply_id))
    self.replay.last_response = self.bot.messages[self.bot.last_id]["message"]
    return self.bot.last_id

async def B19PartialReplyBindingTests_partial(self):
    original = await self.seed()
    self.info.reset_mock()
    partial_id = await self.turn("马哲 2 5", self.original_id)
    self.assertCountEqual(self.created(), [("马哲", "msqea"), ("马哲", "msfea")], self.replay.last_response)
    remaining = self.replay.store.get_record(self.replay.author)
    self.assertEqual(remaining.state.args["_query_words"], list(REST))
    self.assertIn(self.original_id, remaining.origin_message_ids)
    self.assertEqual((remaining.created_at, remaining.expires_at), (original.created_at, original.expires_at))
    self.assertIn("未选择：", self.replay.last_response)
    return partial_id

async def B19PartialReplyBindingTests_complete(self, message, quote_partial):
    partial_id = await self.partial()
    await self.turn(message, partial_id if quote_partial else self.original_id)
    self.assertCountEqual(self.created(), [("马哲", "msqea"), ("马哲", "msfea")] + [
        (word, b18_rest_real_tools_FACTS[word][0]) for word in REST
    ], self.replay.last_response)
    self.assertEqual(self.http.submissions, int("提交" in message), self.replay.last_response)
    pending = self.replay.store.get_record(self.replay.author)
    if "提交" in message:
        self.assertIsNone(pending, self.replay.last_response)
    else:
        self.assertTrue(pending is None or pending.state.function_name == "keytao_submit_batch", self.replay.last_response)

async def B19PartialReplyBindingTests_assert_refused(self, reply_id, message="小美 2", raw=None):
    record = self.replay.store.get_record(self.replay.author)
    snapshot, writes = copy.deepcopy(record), copy.deepcopy(self.http.writes)
    await self.turn(message, reply_id, raw=raw)
    self.assertEqual(self.http.writes, writes)
    self.assertIs(self.replay.store.get_record(self.replay.author), record)
    self.assertEqual(record, snapshot)
    self.assertEqual(self.replay.last_response, REFUSAL)

# Source: test_b20_display.py
class PendingWordHTTP(b19_partial_reply_binding_DraftHTTP):
    def __init__(self, source):
        super().__init__()
        self.source = source
        self.pending_batch_id = self.batch_id if source == "draft" else source + "-batch"
        self.pending_row = {"id": 1, "word": "名创", "code": "mgwx", "type": "Phrase",
                            "action": "Create", "status": "Pending", "weight": 100}
        if source == "draft":
            self.rows.append(copy.deepcopy(self.pending_row))

    def _response(self, method, url, kwargs):
        path = urlsplit(url).path
        creator = 43 if self.source == "others_submitted" else 42
        if method == "GET" and path == "/api/batches":
            self.requests.append((method, path, copy.deepcopy(kwargs.get("params") or {})))
            batches = [] if self.source == "draft" else [{
                "id": self.pending_batch_id, "status": "Submitted", "creator": {"id": creator},
            }]
            return OfflineResponse({"batches": batches, "pagination": {"page": 1, "totalPages": 1}})
        if (method == "GET" and self.source != "draft"
                and path == f"/api/batches/{self.pending_batch_id}"):
            self.requests.append((method, path, {}))
            return OfflineResponse({"batch": {
                "id": self.pending_batch_id, "status": "Submitted", "creatorId": creator,
                "pullRequests": [copy.deepcopy(self.pending_row)],
            }})
        return super()._response(method, url, kwargs)

async def B20DisplayTests_assert_pending_link(self, source, label):
    tools = TOOLS
    self.http = PendingWordHTTP(source)
    self.enterContext(patch.object(tools.http_client, "_keytao_client", self.http))
    self.enterContext(patch.object(tools, "get_keytao_url", return_value="https://keytao.vercel.app"))
    self.enterContext(patch.dict(b18_rest_real_tools_FACTS, {"名创": ("mgwm", ["mgwm", "mgwx"], "ming chuang")}))
    self.enterContext(patch.dict("os.environ", {"KEYTAO_PUBLIC_BASE_QQ": "https://keytao.rea.ink"}))
    await self.seed(("名创",))
    await self.turn("名创 1", self.original_id)
    expected = f"「名创」{label}（→ mgwx" + ("）：" if source == "draft" else "，审核中）：")
    url = f"https://keytao.rea.ink/batch/{self.http.pending_batch_id}"
    reminders = [line for line in self.replay.last_response.splitlines() if line.startswith(expected)]
    self.assertEqual(reminders, [expected + url], self.replay.last_response)
    self.assertIn(("名创", "mgwm"), self.created(), self.replay.last_response)
    self.assertEqual(len(self.http.writes), 1)
    self.assertNotIn("keytao.vercel.app", self.replay.last_response)

# Source: test_b21_display.py
from keytao_bot.harness.orchestrator import _command_suggestions_match_pending_batch as _command_suggestions_match_pending_batch
from keytao_bot.utils.pending_confirmation import advertised_command_suggestions as advertised_command_suggestions
HINT = "同一词可选多个编码、其余词用推荐：如「{}」"

FOOTER = "可只写其中部分词；选择后即加入草稿，提交需明确回复提交指令。"

def B21DisplayTests_setUp(self):
    B19PartialReplyBindingTests_setUp(self)
    tools = {**TOOLS.TOOL_FUNCTIONS, **harness._review_tools.TOOL_FUNCTIONS}
    self.enterContext(patch.object(b17_rest_recommended_commands.tool_executor, "_get_tool_function", tools.get))
    self.enterContext(patch.dict(b18_rest_real_tools_FACTS, {
        "名创": ("mgwm", ["mgwm", "mgwmi"], "ming chuang"),
    }))

# Source: test_b22_vote.py
from keytao_bot.harness.orchestrator import _strip_unbound_command_suggestions as _strip_unbound_command_suggestions
OWNER = "2416"

VOTER = "1449533601"

class VoteHTTP(b19_partial_reply_binding_DraftHTTP):
    def __init__(self, batch_id):
        super().__init__()
        self.batch_id = batch_id
        self.approvals = []

    def _response(self, method, url, kwargs):
        path = urlsplit(url).path
        if path.endswith("/user/find"):
            body = json.loads(kwargs["content"]) if kwargs.get("content") else kwargs.get("json") or {}
            self.requests.append((method, path, copy.deepcopy(body)))
            return OfflineResponse({"found": True, "user": {"id": int(body["platformId"])}})
        if method == "GET" and path == f"/api/batches/{self.batch_id}":
            self.requests.append((method, path, {}))
            return OfflineResponse({"batch": {
                "id": self.batch_id, "status": "Submitted" if self.submissions else "Draft",
                "creatorId": int(OWNER if self.batch_id == "b22-owner" else VOTER),
                "contentVersion": self.version,
                "pullRequests": [{**row, "userId": int(OWNER if self.batch_id == "b22-owner" else VOTER)}
                                 for row in self.rows],
            }})
        if method == "POST" and path.endswith("/community-approve"):
            body = json.loads(kwargs["content"])
            self.approvals.append((path, body))
            return OfflineResponse({"success": True, "voters": ["Garth", "Two", "Three"]})
        return super()._response(method, url, kwargs)

def B22VoteTests_setUp(self):
    B19PartialReplyBindingTests_setUp(self)
    self.replay.memory = replace(self.replay.memory, user_id=OWNER, space_id="865189947",
                                 speaker_name="～×！@")
    self.replay.author = self.replay.memory.conversation_address
    directory = self.enterContext(tempfile.TemporaryDirectory())
    self.votes = vote.CommunityVoteStore(Path(directory) / "votes.db")
    self.enterContext(patch.object(vote, "_default_store", self.votes))
    self.enterContext(patch.dict("os.environ", {
        "KEYTAO_COMMUNITY_VOTE": "1", "KEYTAO_PUBLIC_BASE_QQ": "https://keytao.rea.ink",
    }))
    self.enterContext(patch.dict(b18_rest_real_tools_FACTS, {
        "玛卡巴卡": ("mkbkv", ["mkbkv", "mkbkva"], "ma ka ba ka"),
        "青桔": ("qkjv", ["qkjv", "qkjva"], "qing ju"),
    }))
    tools = {**TOOLS.TOOL_FUNCTIONS, **harness._review_tools.TOOL_FUNCTIONS}
    self.enterContext(patch.object(b17_rest_recommended_commands.tool_executor, "_get_tool_function", tools.get))
    self.enterContext(patch.object(b19_partial_reply_binding_chat.skills_manager, "get_tool_function", tools.get))
    self.bot.send.side_effect = self.deliver
    self.bot.next_id = 1604886725
    self.http = VoteHTTP("b22-owner")
    self.enterContext(patch.object(TOOLS.http_client, "_keytao_client", self.http))

async def B22VoteTests_deliver(self, *, event, message):
    result = await self.bot.deliver(event=event, message=message)
    await vote._record_sent_receipts(self.bot, None, "send_group_msg", {
        "group_id": self.replay.author.space_id, "message": str(message),
    }, result)
    return result

async def B22VoteTests_submit(self, word="玛卡巴卡"):
    record = B18RealToolTests_seed(self, (word,), reorder=False)
    await REAL_FINISH(self.bot, SimpleNamespace(message_id=1), self.replay.author.actor_id,
                         self.replay.memory, self.replay.prompt)
    candidate_id = self.bot.last_id
    await self.turn("加入", candidate_id)
    self.assertEqual([row["word"] for row in self.http.rows], [word], self.replay.last_response)
    add_id = self.bot.last_id
    await self.turn("提交", add_id, raw=raw_operation(op="draft_submit"))
    self.assertEqual(self.http.submissions, 1, self.replay.last_response)
    self.assertIn("当前投票：0/3", self.replay.last_response)
    receipt_id = self.bot.last_id
    return receipt_id, record

async def B22VoteTests_submitted_pair(self):
    receipt_id, _ = await self.submit()
    self.owner_http = self.http
    self.http = VoteHTTP("b22-voter")
    self.enterContext(patch.object(TOOLS.http_client, "_keytao_client", self.http))
    self.replay.memory = replace(self.replay.memory, user_id=VOTER, speaker_name="Garth")
    self.replay.author = self.replay.memory.conversation_address
    await self.submit("青桔")
    return receipt_id

async def B22VoteTests_assert_vote(self, message, receipt_id, actor=VOTER, count=1):
    writes = copy.deepcopy(self.http.writes)
    raw = (raw_operation(op="confirm", refers_to_pending=True) if message == "同意"
           else raw_operation(op="chat"))
    await self.turn(message, receipt_id, actor, raw=raw)
    self.assertIn(f"当前投票：{count}/3", self.replay.last_response)
    self.assertEqual(self.http.writes, writes)
    self.assertEqual(len(self.votes.get("b22-owner")["voters"]), count)
    b19_partial_reply_binding_chat.get_ai_response_core.assert_not_awaited()

async def B22VoteTests_assert_alias(self, word):
    receipt_id = await self.submitted_pair()
    for index, prefix in enumerate(("", "喵喵，", "@喵喵 ", "[CQ:at,qq=3785773770] ")):
        with self.subTest(prefix=prefix):
            data = self.votes.get("b22-owner")
            if data:
                data["voters"] = []
                self.assertTrue(self.votes.save(data))
            await self.assert_vote(prefix + word + "！。", receipt_id, str(2000 + index))

# Source: test_b23_commonness.py
from keytao_bot.utils.bcc_reference import CHANNELS as CHANNELS
from keytao_bot.utils.bcc_reference import HISTORICAL_CHANNELS as HISTORICAL_CHANNELS
from keytao_bot.utils.bcc_reference import create_schema as create_schema
from keytao_bot.utils.word_commonness import lookup_word_commonness as lookup_word_commonness
def B23CommonnessTests_setUp(self):
    directory = self.enterContext(tempfile.TemporaryDirectory())
    self.db = Path(directory) / "reference.db"
    with closing(sqlite3.connect(self.db)) as connection, connection:
        create_schema(connection)
        connection.execute(
            "CREATE TABLE word_commonness (word TEXT PRIMARY KEY, corpus_frequency INTEGER, "
            "part_of_speech TEXT, dictionary_presence_count INTEGER)"
        )
        connection.execute("INSERT INTO word_commonness VALUES ('羽衣甘蓝', NULL, NULL, 2)")
        for channel in (*CHANNELS, *HISTORICAL_CHANNELS):
            connection.execute("INSERT INTO bcc_dataset VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (channel, channel, "word", "offline", "offline", "offline", 1, 1,
                 1377730, 835646749))
        connection.execute("INSERT INTO bcc_frequency VALUES (?, ?, ?, ?, ?)",
            ("近代汉语", "用过", 1952, 1952 * 1000000 / 835646749, 33200))
    self.enterContext(patch.object(review, "reference_db_path", return_value=self.db))
    self.web = self.enterContext(patch.object(
        review, "_estimate_word_commonness_web_fallback",
        AsyncMock(side_effect=AssertionError("B23 must stay offline")),
    ))
    self.enterContext(patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")))
    self.enterContext(patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Offline only")))

async def B23CommonnessTests_candidate_review(self):
    subject = {
        "success": True, "word": "羽衣甘蓝", "recommendedCode": "yygla",
        "pronunciations": [{
            "pinyin": "yu yi gan lan", "recommendedCode": "yygla",
            "candidateStatuses": [
                {"code": "yygl", "occupied": True, "words": ["用过"], "label": "已有「用过」"},
                {"code": "yygla", "occupied": False, "words": []},
                {"code": "yyglao", "occupied": False, "words": []},
            ],
        }],
    }
    assessments = await review.assess_candidate_chain_commonness(subject)
    review.apply_candidate_ordering_recommendation(subject, assessments)
    prompt = harness.openai_chat_module._format_reviewed_add_prompt(subject)
    self.assertIsNotNone(prompt)
    self.web.assert_not_awaited()
    return subject, assessments[0], prompt

def B23CommonnessTests_seed_historical_chain(self):
    with closing(sqlite3.connect(self.db)) as connection, connection:
        connection.execute("INSERT INTO word_commonness VALUES ('甲词', NULL, NULL, 3)")
        connection.execute("INSERT INTO bcc_frequency VALUES (?, ?, ?, ?, ?)",
            ("近代汉语", "乙词", 1, 1000000 / 835646749, 100))

# Source: test_b23_display.py
# Source: test_b24_displace.py
from keytao_bot.harness.authorization_grammar import explicit_complete_add_item as explicit_complete_add_item
b24_displace_WORD, b24_displace_OCCUPANT = "豪横", "毫亨"

b24_displace_CODES = ["hzhr", "hzhro", "hzhrov"]

REAL_REVIEW_READS = {name: getattr(review, name) for name in (
    "collect_pronunciation_evidence_limited", "fetch_keytao_encode", "lookup_words",
)}

class DisplaceHTTP(b19_partial_reply_binding_DraftHTTP):
    def __init__(self):
        super().__init__()
        self.entries = [{"word": b24_displace_OCCUPANT, "code": b24_displace_CODES[0], "type": "Phrase", "weight": 100}]

    def _response(self, method, url, kwargs):
        if urlsplit(url).path == "/api/phrases/encode":
            word = (kwargs.get("params") or {})["word"]
            self.requests.append((method, "/api/phrases/encode", {"word": word}))
            if word not in {b24_displace_WORD, b24_displace_OCCUPANT}:
                self.errors.append(("unexpected word", word))
                raise AssertionError(word)
            return OfflineResponse({
                "input": word, "type": "二字词", "codes": b24_displace_CODES, "altCodes": [],
                "flyKeyVariants": [], "phrasePinyins": ["hao heng"],
                "pronunciationSource": "zdic-phrase", "standardPronunciationStatus": "found",
                "chars": [{"char": char, "pinyin": pinyin, "pinyins": [pinyin],
                           "pronunciationLookupStatus": "found", "phoneticCode": code, "shapeCode": shape}
                          for char, pinyin, code, shape in zip(word, ("hao", "heng"), ("hz", "hr"), ("o", "ov"))],
            })
        return super()._response(method, url, kwargs)

def B24DisplaceTests_setUp(self):
    B19PartialReplyBindingTests_setUp(self)
    # Earlier concurrent fixtures can leave patched readers behind.
    self.enterContext(patch.multiple(review, **REAL_REVIEW_READS))
    self.http = DisplaceHTTP()
    self.enterContext(patch.object(TOOLS.http_client, "_keytao_client", self.http))
    functions = {**harness._lookup_tools.TOOL_FUNCTIONS,
                 **harness._review_tools.TOOL_FUNCTIONS, **TOOLS.TOOL_FUNCTIONS}
    self.enterContext(patch.object(b19_partial_reply_binding_chat.tool_executor, "_get_tool_function", functions.get))
    self.web = self.enterContext(patch.object(
        review.http_client, "guarded_fetch", AsyncMock(return_value=OfflineResponse({}, 404)),
    ))
    self.enterContext(patch.object(review, "_review_cache", OrderedDict()))
    self.enterContext(patch.object(review, "_evidence_host_failures", {}))
    directory = self.enterContext(tempfile.TemporaryDirectory())
    db = Path(directory) / "reference.db"
    with closing(sqlite3.connect(db)) as connection, connection:
        create_schema(connection)
        connection.execute("CREATE TABLE word_commonness (word TEXT PRIMARY KEY, corpus_frequency INTEGER, "
                           "part_of_speech TEXT, dictionary_presence_count INTEGER)")
        connection.executemany("INSERT INTO word_commonness VALUES (?, 3, NULL, ?)",
                               [(b24_displace_WORD, 2), (b24_displace_OCCUPANT, 0)])
        for channel in (*CHANNELS, *HISTORICAL_CHANNELS):
            connection.execute("INSERT INTO bcc_dataset VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                               (channel, channel, "word", "offline", "offline", "offline", 1, 1, 100, 1000000))
        connection.executemany("INSERT INTO bcc_frequency VALUES (?, ?, 17, 17, 1)",
                               [("多领域", b24_displace_WORD), ("文学", b24_displace_WORD)])
    self.enterContext(patch.object(review, "reference_db_path", return_value=db))

def B24DisplaceTests_tearDown(self):
    B19PartialReplyBindingTests_tearDown(self)

async def B24DisplaceTests_seed(self):
    greeting = await self.bot.deliver(event=None, message="喵喵在。")
    await self.turn("喵喵 豪横", str(greeting["message_id"]), raw=raw_operation(
        op="query", operands=[b24_displace_WORD], codes=[], refers_to_pending=False,
    ))
    self.original_id = self.bot.last_id
    record = self.replay.store.get_record(self.replay.author)
    self.assertIsNotNone(record, self.replay.last_response)
    state = record.state
    self.assertIsInstance(state, harness.PendingAddWord)
    self.assertEqual(state.recommended_code, b24_displace_CODES[1], self.replay.last_response)
    self.assertEqual(state.server_occupied_words[b24_displace_CODES[0]], [b24_displace_OCCUPANT])
    self.assertIn("1. hzhr — 已有「毫亨」", self.replay.last_response)
    self.assertIn("2. hzhro — ✅ 推荐（空位）", self.replay.last_response)
    return state

async def B24DisplaceTests_finish_shift(self, message):
    await self.turn(message, self.original_id)
    record = self.replay.store.get_record(self.replay.author)
    if not self.http.writes:
        self.assertIsNotNone(record, self.replay.last_response)
        self.assertIsInstance(record.state, harness.PendingToolConfirm, self.replay.last_response)
        self.assertEqual(record.state.function_name, "keytao_shift_phrase_code", self.replay.last_response)
        self.assertTrue(record.state.args["confirmed_plan_digest"])
        await self.turn("确认", self.bot.last_id)
    self.assertCountEqual(self.created(), [(b24_displace_WORD, b24_displace_CODES[0]), (b24_displace_OCCUPANT, b24_displace_CODES[1])],
                          self.replay.last_response)
    self.assertEqual([(row["word"], row["code"]) for row in self.http.rows if row["action"] == "Delete"],
                     [(b24_displace_OCCUPANT, b24_displace_CODES[0])])
    self.assertEqual(len(self.http.writes), 1)
    record = self.replay.store.get_record(self.replay.author)
    self.assertTrue(record is None or record.state.function_name == "keytao_submit_batch")

# Source: test_b25_session.py
b25_session_commands = b19_partial_reply_binding_chat._chat_commands

b25_session_WORD, b25_session_OCCUPANT = "守时", "首诗"

b25_session_CODES = ["edek", "edeko", "edekoo"]

b25_session_WORDS = ("公义", "礼义廉耻", "德智体群美", b25_session_WORD, "随时待命")

b25_session_FACTS = {
    "公义": (["gjyk", "gjyko", "gjykoo"], "gong yi"),
    "礼义廉耻": (["lyli", "lylio", "lylioo"], "li yi lian chi"),
    "德智体群美": (["dftm", "dftmo", "dftmoo"], "de zhi ti qun mei"),
    b25_session_WORD: (b25_session_CODES, "shou shi"),
    "收拾": (b25_session_CODES, "shou shi"),
    "首饰": (b25_session_CODES, "shou shi"),
    b25_session_OCCUPANT: (b25_session_CODES, "shou shi"),
    "随时待命": (["sudm", "sudmo", "sudmoo"], "sui shi dai ming"),
    "周内": (["fdnw", "fdnsu"], "zhou nei"),
}

class SessionHTTP(b19_partial_reply_binding_DraftHTTP):
    def __init__(self):
        super().__init__()
        self.batch_id = "40c69d7a"
        self.entries = [
            {"word": word, "code": code, "type": "Phrase", "weight": 100}
            for word, code in (("收拾", b25_session_CODES[0]), ("首饰", b25_session_CODES[1]), (b25_session_OCCUPANT, b25_session_CODES[2]),
                               ("周内", "fdnsu"))
        ]

    def _response(self, method, url, kwargs):
        if urlsplit(url).path == "/api/bot/evidence/fetch":
            return OfflineResponse({}, 404)
        if urlsplit(url).path == "/api/phrases/encode":
            word = (kwargs.get("params") or {})["word"]
            self.requests.append((method, "/api/phrases/encode", {"word": word}))
            if word not in b25_session_FACTS:
                self.errors.append(("unexpected word", word))
                raise AssertionError("Unexpected concatenated/filler lookup: " + word)
            codes, reading = b25_session_FACTS[word]
            return OfflineResponse({
                "input": word, "type": "二字词" if len(word) == 2 else "多字词",
                "codes": codes, "altCodes": [], "flyKeyVariants": [],
                "phrasePinyins": [reading], "pronunciationSource": "zdic-phrase",
                "standardPronunciationStatus": "found",
                "chars": [{"char": char, "pinyin": syllable, "pinyins": [syllable],
                           "pronunciationLookupStatus": "found", "phoneticCode": "ed",
                           "shapeCode": "oo"} for char, syllable in zip(word, reading.split())],
            })
        return super()._response(method, url, kwargs)

def B25SessionTests_setUp(self):
    B24DisplaceTests_setUp(self)
    self.enterContext(patch.dict("os.environ", {"KEYTAO_COMMUNITY_VOTE": "0"}))
    self.http = SessionHTTP()
    self.enterContext(patch.object(TOOLS.http_client, "_keytao_client", self.http))
    self.replay.client.create.side_effect = TimeoutError("B25 semantic parser timeout")
    self.enterContext(patch.object(b19_partial_reply_binding_chat._chat_routing, "AsyncOpenAI", return_value=self.replay.client))
    self.histories = {self.replay.author.actor_id: self.replay.history}
    self.enterContext(patch.object(b19_partial_reply_binding_chat, "get_history", side_effect=self.history_for))
    self.enterContext(patch.object(b19_partial_reply_binding_chat, "remember_conversation", side_effect=self.remember))
    self.tool_errors = self.enterContext(patch.object(b25_session_commands.logger, "error", wraps=b25_session_commands.logger.error))
    with closing(sqlite3.connect(review.reference_db_path())) as connection, connection:
        connection.executemany("INSERT OR REPLACE INTO word_commonness VALUES (?, 3, NULL, 2)",
                               [(word,) for word in b25_session_FACTS])

def B25SessionTests_tearDown(self):
    B19PartialReplyBindingTests_tearDown(self)

def B25SessionTests_history_for(self, key):
    actor = key.actor_id if hasattr(key, "actor_id") else key[1]
    return self.histories.setdefault(actor, [])

def B25SessionTests_remember(self, key, memory, message, response):
    self.history_for(key).extend([{"role": "user", "content": message},
                                 {"role": "assistant", "content": response}])

async def B25SessionTests_turn(self, message, reply_id, actor=None):
    actor = actor or self.replay.author.actor_id
    memory = replace(self.replay.memory, user_id=actor)
    full_message = f"[reply:id={reply_id}][at:qq={self.bot.self_id}] {message}"
    event = SimpleNamespace(
        to_me=True, group_id=self.replay.author.space_id, user_id=actor, message_id=2,
        raw_message=full_message,
        message=[SimpleNamespace(type="reply", data={"id": reply_id}),
                 SimpleNamespace(type="at", data={"qq": self.bot.self_id}),
                 SimpleNamespace(type="text", data={"text": message})],
        get_plaintext=lambda: message,
    )
    tokens = [(variable, variable.set(value)) for variable, value in (
        (b19_partial_reply_binding_chat._current_turn_message, full_message), (b19_partial_reply_binding_chat._current_conversational_turn, None),
        (b19_partial_reply_binding_chat.current_operation_capture, None), (b19_partial_reply_binding_chat.current_draft_delivery_claims, []),
    )]
    before = self.bot.send.await_count
    try:
        with patch.object(b19_partial_reply_binding_chat, "extract_memory_context", AsyncMock(return_value=memory)):
            await b19_partial_reply_binding_chat._handle_ai_chat_serialized(self.bot, event, "qq", actor)
    finally:
        for variable, token in reversed(tokens):
            variable.reset(token)
    self.assertEqual(self.bot.send.await_count, before + 1)
    self.bot.get_msg.assert_awaited_with(message_id=int(reply_id))
    self.replay.last_response = self.bot.messages[self.bot.last_id]["message"]
    return self.bot.last_id

async def B25SessionTests_greeting(self, text="喵喵在。"):
    result = await self.bot.deliver(event=None, message=text)
    return str(result["message_id"])

async def B25SessionTests_seed(self, word=b25_session_WORD):
    await self.turn("喵喵 加词 " + word, await self.greeting())
    self.original_id = self.bot.last_id
    record = self.replay.store.get_record(self.replay.author)
    self.assertIsNotNone(record, self.replay.last_response)
    return record

async def B25SessionTests_partial_session(self):
    await self.seed("、".join(b25_session_WORDS))
    await self.turn("公义 3，礼义廉耻 2，德智体群美 2，守时 3，随时待命 1，加入并提交",
                    self.original_id)
    self.assertCountEqual(self.created(), [(word, b25_session_FACTS[word][0][index]) for word, index in (
        ("公义", 2), ("礼义廉耻", 1), ("德智体群美", 1), ("随时待命", 0),
    )], self.replay.last_response)
    self.assertEqual(self.http.submissions, 1, self.replay.last_response)
    self.assertIn("首诗", self.replay.last_response)
    self.assertIn("未加入", self.replay.last_response)
    return self.bot.last_id

async def B25SessionTests_finish_replacement(self, message, reply_id=None):
    await self.turn(message, reply_id or self.original_id)
    if not self.http.writes:
        record = self.replay.store.get_record(self.replay.author)
        self.assertIsNotNone(record, self.replay.last_response)
        self.assertIsInstance(record.state, harness.PendingToolConfirm, self.replay.last_response)
        await self.turn("确认", self.bot.last_id)
    self.assertIn((b25_session_WORD, b25_session_CODES[2]), self.created(), self.replay.last_response)
    self.assertIn((b25_session_OCCUPANT, b25_session_CODES[2]), [(row["word"], row["code"])
                  for row in self.http.rows if row["action"] == "Delete"], self.replay.last_response)

# Source: test_b26_display.py
from keytao_bot.utils.pending_confirmation import candidate_commonness_guard_copy as candidate_commonness_guard_copy
async def B26DisplayTests_lookup_replacement_product(self, message, word="替代品"):
    facts = {
        "替代品": (["tdpk", "tdpko", "tdpkoo"], "ti dai pin"),
        "查替代品": (["wtdp", "wtdpo", "wtdpoo"], "cha ti dai pin"),
        "查一下替代品": (["wyxt", "wyxto", "wyxtoo"], "cha yi xia ti dai pin"),
    }
    with patch.dict(b25_session_FACTS, facts):
        await self.turn(message, await self.greeting())
    record = self.replay.store.get_record(self.replay.author)
    self.assertIsNotNone(record, self.replay.last_response)
    self.assertIsInstance(record.state, harness.PendingAddWord, self.replay.last_response)
    self.assertEqual(record.state.word, word, self.replay.last_response)
    self.assertIn(facts[word][0][0], self.replay.last_response)
    self.assertEqual(self.http.writes, [])

@staticmethod
def B26DisplayTests_displacement_suggestions(text):
    return re.findall(r"添加 守时 ([a-z]+)，(?:挤掉|顶替) ([^」\n]+)", text)

# Source: test_b27_reading.py
b27_reading_WORD = "周内"

PRIMARY = ["fdns", "fdnsu", "fdnsui"]

ALTERNATE = ["fdnw", "fdnwu", "fdnwui"]

MODIFIERS = ("强制", "强行", "硬加")

class ReadingHTTP(SessionHTTP):
    def __init__(self):
        super().__init__()
        self.extra_readings = [("nei", ALTERNATE)]
        self.entries = [
            {"word": word, "code": code, "type": "Phrase", "weight": 100}
            for word, code in ((b27_reading_WORD, "fdnsu"), ("周纳", "fdns"))
        ]

    def _response(self, method, url, kwargs):
        if urlsplit(url).path in ("/api/phrases/encode", "/api/bot/phrases/encode"):
            word = (kwargs.get("params") or {})["word"]
            if word in (b27_reading_WORD, "周纳"):
                self.requests.append((method, "/api/phrases/encode", {"word": word}))
                extras = self.extra_readings if word == b27_reading_WORD else []
                return OfflineResponse({
                    "input": word, "type": "二字词", "codes": PRIMARY,
                    "altCodes": [code for _reading, codes in extras for code in codes],
                    "candidateCodes": PRIMARY + [code for _reading, codes in extras for code in codes],
                    "alternatePhrasePronunciationCodes": [
                        {"pinyins": ["zhou", reading], "codes": codes}
                        for reading, codes in extras
                    ],
                    "flyKeyVariants": [], "phrasePinyins": ["zhou", "na"],
                    "pronunciationSource": "zdic-phrase", "standardPronunciationStatus": "found",
                    "chars": [
                        {"char": "周", "pinyin": "zhou", "pinyins": ["zhou"],
                         "pronunciationLookupStatus": "found", "phoneticCode": "fd", "shapeCode": "u"},
                        {"char": word[1], "pinyin": "na", "pinyins": ["na", *[r for r, _c in extras]],
                         "pronunciationLookupStatus": "found", "phoneticCode": "ns", "shapeCode": "ui"},
                    ],
                })
        return super()._response(method, url, kwargs)

def B27ReadingTests_setUp(self):
    B25SessionTests_setUp(self)
    self.http = ReadingHTTP()
    self.enterContext(patch.object(TOOLS.http_client, "_keytao_client", self.http))
    self.replay.client.create.side_effect = TimeoutError("B27 semantic parser timeout")
    with closing(sqlite3.connect(review.reference_db_path())) as connection, connection:
        connection.execute("INSERT OR REPLACE INTO word_commonness VALUES (?, 30000, NULL, 3)", ("周纳",))
        connection.execute("UPDATE word_commonness SET corpus_frequency = 1 WHERE word = ?", (b27_reading_WORD,))

def B27ReadingTests_reset_session(self):
    self.replay.store.delete(self.replay.author)
    self.http.rows.clear()
    self.http.writes.clear()
    self.http.version = 0
    self.http.submissions = 0
    self.histories.clear()
    review._review_cache.clear()

async def B27ReadingTests_seed(self, word=b27_reading_WORD):
    await self.turn("喵喵 " + word, await self.greeting())
    self.original_id = self.bot.last_id
    record = self.replay.store.get_record(self.replay.author)
    self.assertIsNotNone(record, self.replay.last_response)
    return record

async def B27ReadingTests_finish_add(self, message):
    await self.turn(message, self.original_id)
    if not self.http.writes:
        record = self.replay.store.get_record(self.replay.author)
        self.assertIsNotNone(record, self.replay.last_response)
        self.assertIsInstance(record.state, harness.PendingToolConfirm, self.replay.last_response)
        await self.turn("确认", self.bot.last_id)
    self.assertEqual(self.created(), [(b27_reading_WORD, ALTERNATE[0])], self.replay.last_response)
    self.assertTrue(self.http.rows[0]["needsManualReview"], self.http.rows)
    self.assertIn("管理员", self.replay.last_response)
    self.assertEqual(self.http.submissions, 0)

async def B27ReadingTests_modifier_adds(self, command):
    for modifier in MODIFIERS:
        with self.subTest(modifier=modifier):
            self.reset_session()
            await self.seed(b27_reading_WORD)
            await self.finish_add(modifier + command)

# Source: test_b5_prod_defects.py
from tests import test_community_vote as vote_fixtures
from keytao_bot.harness.state import DraftOperationCoordinator as DraftOperationCoordinator
from keytao_bot.harness.state import MemoryConversationStateStore as MemoryConversationStateStore
from tests.test_semantic_b5_delete import B5DeleteTests as B5DeleteTests
def B5ReplySafetyTests_setUp(self):
    self.chat = harness.openai_chat_module
    self.store = MemoryConversationStateStore()
    self.memory = harness.ChatMemoryContext(
        platform="qq", user_id="abang", space_type="group", space_id="b5",
        speaker_name="Offline owner",
    )
    self.tool = AsyncMock(return_value=json.dumps({"success": True, "items": []}))
    self.finish = AsyncMock()
    for context in (
        patch.object(self.chat, "conversation_state_store", self.store),
        patch.object(self.chat, "draft_operation_coordinator", DraftOperationCoordinator()),
        patch.object(self.chat, "call_tool_function", self.tool),
        patch.object(self.chat, "_finish_ai_chat_response", self.finish),
        patch.object(self.chat, "remember_conversation"),
        patch.object(self.chat, "get_history", return_value=[]),
        patch.object(self.chat, "to_me", return_value=AsyncMock(
            side_effect=lambda bot, event, state: event.to_me,
        )),
    ):
        context.start()
        self.addCleanup(context.stop)
    self.token = self.chat.current_draft_delivery_claims.set([])
    self.addCleanup(self.chat.current_draft_delivery_claims.reset, self.token)

def B5ReplySafetyTests_context(self, message="确认", *, quoted=False, addressed=True):
    ctx = self.chat.TurnContext(
        bot=SimpleNamespace(), event=SimpleNamespace(to_me=addressed),
        platform="qq", user_id="abang", message_text=message,
        normalized_message_text=message, conv_key=self.memory.conversation_address,
        memory_context=self.memory, space_key=self.memory.conversation_address.space_key,
    )
    if quoted:
        ctx.reply_reference = harness.ReplyReferenceInfo(
            is_reply=True, is_to_bot=True, text="是否继续提交？回复「确认」。",
        )
    return ctx

# Source: test_b7_phrasing.py
from tests import test_semantic_direct_exec as direct
from tests import test_semantic_query_route as query
B7PhrasingRouteTests_asyncTearDown = direct.SemanticDirectExecTests.asyncTearDown

B7PhrasingRouteTests_row = staticmethod(direct.SemanticDirectExecTests.row)

B7PhrasingRouteTests_result = direct.SemanticDirectExecTests.result

B7PhrasingRouteTests_call = direct.SemanticDirectExecTests.call

B7PhrasingRouteTests_stage = direct.SemanticDirectExecTests.stage

B7PhrasingRouteTests_review_word = direct.SemanticDirectExecTests.review_word

B7PhrasingRouteTests_answer = direct.SemanticDirectExecTests.answer

async def B7PhrasingRouteTests_asyncSetUp(self):
    await direct.SemanticDirectExecTests.asyncSetUp(self)
    self.rows = [self.row(word, code) for word, code in (
        ("我送给", "wsg"), ("纯音乐", "jyy"), ("火腿肠", "htj"),
        ("航天局", "htju"), ("富临", "fjlb"),
    )]

def B7PhrasingRouteTests_assert_recodes(self, response, expected):
    self.assertEqual(len(self.writes), 1, (self.calls, response))
    name, args = self.writes[0]
    self.assertEqual(name, "keytao_batch_add_to_draft")
    self.assertEqual([(item["action"], item["word"], item["code"])
                      for item in args["items"]], expected)
    self.assertIn("已加入草稿", response)
    self.assertIsNone(self.store.get_record(self.author))
    self.review.assert_not_awaited()

B7QueryRouteTests_asyncSetUp = query.SemanticQueryRouteTests.asyncSetUp

B7QueryRouteTests_asyncTearDown = query.SemanticQueryRouteTests.asyncTearDown

B7QueryRouteTests_lookup = query.SemanticQueryRouteTests.lookup

B7QueryRouteTests_context = query.SemanticQueryRouteTests.context

B7QueryRouteTests_set_model = query.SemanticQueryRouteTests.set_model

B7QueryRouteTests_ticket = query.SemanticQueryRouteTests.ticket

B7QueryRouteTests_assert_finished = query.SemanticQueryRouteTests.assert_finished

B7QueryRouteTests_assert_preserved = query.SemanticQueryRouteTests.assert_preserved

# Source: test_b7_selection.py
def incident_ticket(words=None):
    inventories = {
        "一夫一妻": ("yfyq", [("yfyq", False)], ""),
        "一夫多妻": ("yfdq", [("yfdq", False)], ""),
        "吴三桂": ("wsg", [("wsg", True), ("wsgo", False)], "我送给"),
        "陈圆圆": ("jyy", [("jyy", True), ("jyya", False)], "纯音乐"),
        "多尔衮": ("deg", [("deg", False), ("dego", False), ("degov", False)], ""),
        "皇太极": ("htju", [("htj", True), ("htju", True), ("htjuv", False), ("htjuvv", False)], "航天局"),
        "福临": ("fjlb", [("fjlb", True), ("fjlbo", False), ("fjlboi", False)], "富临"),
    }
    items, scopes = [], []
    for word in words or inventories:
        code, candidates, occupant = inventories[word]
        occupied = {code: [occupant]} if occupant else {}
        if word == "皇太极":
            occupied["htj"] = ["火腿肠"]
        assessments = [{
            "verdict": "front_more_common", "newWord": word,
            "occupantWord": occupant, "occupantCode": code, "newCode": code,
            "freeCode": next(candidate for candidate, used in candidates if not used),
        }] if occupant else []
        reviewed = PendingAddWord(
            word, code, candidates, server_candidates=candidates,
            server_occupied_words=occupied, server_ordering_assessments=assessments,
            pronunciation_codes={candidate: "offline reading" for candidate, _ in candidates},
            needs_manual_review=False,
        )
        items.append({"action": "Create", "word": word, "code": code})
        scopes.append({
            "word": word, "candidates": candidates, "occupiedWords": occupied,
            "orderingAssessments": assessments, "reviewedState": _pending_add_word_payload(reviewed),
            "reviewedPrompt": "Offline authoritative review",
        })
    return PendingToolConfirm("keytao_batch_add_to_draft", {
        "items": items, "_candidate_scopes": scopes, "_reviewed_multi_word": True,
        "_query_words": [item["word"] for item in items],
    })

def B7SelectionTests_resolve(self, message, ticket=None):
    ticket = ticket or incident_ticket()
    original = copy.deepcopy(ticket)
    derived, intent, error = b17_rest_recommended_routing._resolve_multi_word_pending_candidate_selection(ticket, message)
    self.assertEqual(ticket, original)
    self.assertIsNone(error, error)
    self.assertIsNotNone(derived)
    self.assertEqual(intent.intent, "pending_confirm")
    self.assertTrue(b17_rest_recommended_routing.message_authorizes_live_pending_mutation(message, ticket))
    return derived

# Source: test_b8_cancel.py
harness._fake_nonebot.init = lambda: None

harness._fake_log.logger.exception = harness._fake_log.logger.error

PRESENT = (
    ("喜上眉梢", "xsms"), ("欢天喜地", "htxd"), ("眉开眼笑", "mkyx"),
    ("喜气洋洋", "xqyy"), ("眉飞色舞", "mfsw"), ("愁眉苦脸", "jmkl"),
    ("唉声叹气", "astq"),
)

b8_cancel_WORDS = tuple(word for word, _code in PRESENT) + ("哀声叹气",)

def B8CancelTests_replay(self):
    return Replay({
        "sender": "offline-abang",
        "dictionary": [
            {"word": word, "code": code, "type": "Phrase"}
            for word, code in PRESENT + (("取消", "qa"), ("取消", "qlxc"))
        ],
        "candidate_facts": {
            "哀声叹气": {"recommended": "astqa", "codes": ["astq", "astqa"],
                     "pinyin": "ai sheng tan qi"},
        },
    }, harness, FakeClient)

async def B8CancelTests_offer(self, replay):
    old_ticket = PendingToolConfirm("keytao_submit_batch", {"batch_id": "offline-batch"})
    self.assertTrue(replay.store.set(replay.author, old_ticket))
    replay.history[:] = [{"role": "assistant", "content": "回复「确认」提交，或「取消」。"}]
    message = "加词：" + "，".join(b8_cancel_WORDS)
    await replay.turn({"message": message, "to_me": True}, raw_operation(
        op="add", operands=list(b8_cancel_WORDS), codes=[], frame="direct", refers_to_pending=False,
    ))
    self.assertNotIn("原待确认提交已取消", replay.last_response)
    message_token = replay.chat._current_turn_message.set(message)
    try:
        # Delivery can redraw the prompt after the stage has remembered it.
        replay.last_response = replay.chat._prepare_user_facing_reply(replay.last_response, replay.memory)
    finally:
        replay.chat._current_turn_message.reset(message_token)
    record = replay.store.get_record(replay.author)
    self.assertIsInstance(record.state, PendingToolConfirm)
    self.assertEqual(record.state.function_name, "keytao_batch_add_to_draft")
    self.assertEqual(record.state.args["_query_words"], list(b8_cancel_WORDS))
    self.assertEqual([item["word"] for item in record.state.args["items"]], ["哀声叹气"])
    self.assertIn("加入", replay.last_response)
    self.assertIn("加入并提交", replay.last_response)
    self.assertIn("换 2", replay.last_response)
    self.assertTrue(replay.chat._advertised_reply_matches_live_record(replay.last_response, record))
    self.assertEqual(understand_message("取消", record.state).op, "cancel")
    return record

async def B8CancelTests_assert_unbound_reply(self, message, *, model_op="query", digestless=False):
    replay = self.replay()
    if not any(row["word"] == message for row in replay.entries):
        replay.entries.append({"word": message, "code": "abcd", "type": "Phrase"})
    original_tool = replay.tool

    async def tool(name, args, *positional, **kwargs):
        if name == "keytao_encode":
            replay.calls.append((name, copy.deepcopy(args)))
            return json.dumps({"success": False, "codes": []})
        return await original_tool(name, args, *positional, **kwargs)

    replay.tool = tool
    with ExitStack() as stack:
        for context in replay.patches():
            stack.enter_context(context)
        stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")))
        record = await self.offer(replay)
        if digestless:
            self.assertTrue(replay.store.bind_origin_prompt_digest(replay.author, ""))
        original = copy.deepcopy(record)
        raw_prompt = "Earlier candidate prompt\n" + replay.history[-1]["content"]
        self.assertNotEqual(replay.chat._prompt_capability_digest(raw_prompt), record.origin_prompt_digest)
        replay.client.create.return_value.choices[0].message.content = json.dumps({
            "intent": "word_lookup", "words": [message], "confidence": 0.99,
        })
        stack.enter_context(patch.object(replay.chat, "OPENAI_API_KEY", "offline"))
        replay.calls.clear()
        replay.trace.clear()
        quote = "Another unrelated bot message" if digestless else raw_prompt
        await replay.turn({"message": message, "to_me": True, "quote_target": quote}, raw_operation(
            op=model_op, operands=["哀声叹气"] if model_op == "cancel" else [message],
            codes=[], frame="direct" if model_op == "cancel" else "none",
            refers_to_pending=model_op == "cancel",
        ))
        self.assertEqual(replay.last_response, "这条回复未对应当前待确认操作；请引用当前候选消息后再回复。")
        self.assertIs(replay.store.get_record(replay.author), record)
        self.assertEqual(record, original)
        self.assertEqual(replay.calls, [])
        self.assertEqual(replay.writes, [])
        replay.client.create.assert_not_awaited()

# Source: test_b8_present_words.py
from keytao_bot.utils.pending_confirmation import advertised_batch_binding_pairs as advertised_batch_binding_pairs
from keytao_bot.utils.pending_confirmation import advertised_reply_contract as advertised_reply_contract
from keytao_bot.utils.pending_confirmation import already_existing_word_copy as already_existing_word_copy
B8PresentWordsTests_asyncSetUp = query.SemanticQueryRouteTests.asyncSetUp

B8PresentWordsTests_asyncTearDown = query.SemanticQueryRouteTests.asyncTearDown

B8PresentWordsTests_lookup = query.SemanticQueryRouteTests.lookup

B8PresentWordsTests_present = {
        "喜上眉梢": "xsms", "欢天喜地": "htxd", "眉开眼笑": "mkyx",
        "喜气洋洋": "xqyy", "眉飞色舞": "mfsw", "愁眉苦脸": "jmkl",
        "唉声叹气": "astq",
    }

async def B8PresentWordsTests_add(self, words):
    self.entries.update(self.present)
    self.candidate_codes.update({"哀声叹气": "astq", "发薪": "fsxb"})
    return await self.commands._try_handle_simple_single_word_query(
        "加词：" + "，".join(words), "qq", self.author.actor_id,
        self.author, self.author.space_key, "Offline owner",
    )

def B8PresentWordsTests_assert_present_copy(self, response):
    self.assertEqual(response.splitlines().count(
        "已在词库：" + "、".join(f"{word} {code}" for word, code in self.present.items())
    ), 1)
    self.assertNotIn("可再追加一个编码", response)
    self.assertNotIn("当前没有必须执行的变更", response)

async def B8PresentWordsTests_assert_candidates(self, missing):
    words = (*self.present, *missing)
    response = await self.add(words)
    self.assert_present_copy(response)
    record = self.store.get_record(self.author)
    self.assertIsInstance(record.state, PendingToolConfirm)
    self.assertEqual(record.owner_key, self.author)
    self.assertEqual(record.state.args["_query_words"], list(words))
    self.assertEqual([row["word"] for row in record.state.args["items"]], list(missing))
    self.assertEqual([row["word"] for row in record.state.args["_candidate_scopes"]], list(missing))
    candidates = self.commands.render_server_backed_batch_candidates(
        record.state.args["items"], record.state.args["_candidate_scopes"],
    )
    self.assertTrue(response.endswith(candidates))
    self.assertEqual(
        advertised_reply_contract(response).command_suggestions,
        advertised_reply_contract(candidates).command_suggestions,
    )
    self.assertEqual(advertised_batch_binding_pairs(response), advertised_batch_binding_pairs(candidates))
    self.assertTrue(self.chat._advertised_reply_matches_live_record(response, record))
    self.assertEqual(self.chat._render_live_batch_record(record), response)
    self.main.assert_not_awaited()
    self.client.create.assert_not_awaited()

# Source: test_b9_add_keeps_submit.py
from keytao_bot.utils import http_client as http_client
def AddKeepsSubmitTests_setUp(self):
    self.replay = Replay({
        "sender": "b9-owner",
        "draft": [{"id": 1, "action": "Create", "word": "驴友", "code": "llyd", "type": "Phrase"}],
        "candidate_facts": {
            "屡有": {"recommended": "llydav", "codes": ["llydav"], "pinyin": "lv you"},
            "云翻": {"recommended": "yjfj", "codes": ["yjfj"], "pinyin": "yun fan"},
        },
    }, harness, FakeClient)
    self.old_draft = copy.deepcopy(self.replay.draft)
    self.foreign = False
    self.submitted = []
    self.original_tool = self.replay.tool
    self.replay.tool = self.tool
    self.details = AsyncMock(side_effect=lambda *args, **kwargs: {"batch": {
        "id": "offline-batch", "contentVersion": 1, "status": "Draft", "creatorId": 17,
        "pullRequests": [{**item, "userId": 99 if self.foreign and item["id"] == 1 else 17}
                         for item in self.replay.draft],
    }})
    stack = ExitStack()
    self.addCleanup(stack.close)
    for context in self.replay.patches():
        stack.enter_context(context)
    stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")))
    stack.enter_context(patch.object(http_client, "keytao_json", self.details))
    stack.enter_context(patch.dict("os.environ", {"KEYTAO_COMMUNITY_VOTE": "0"}))
    token = self.replay.chat.current_draft_delivery_claims.set([])
    self.addCleanup(self.replay.chat.current_draft_delivery_claims.reset, token)

def AddKeepsSubmitTests_preview(self):
    return self.replay.result(
        success=False, requiresConfirmation=True, snapshotItems=copy.deepcopy(self.replay.draft),
        snapshotDigest="a" * 64, warningDigest="b" * 64, auditDigest="c" * 64, warnings=[],
    )

async def AddKeepsSubmitTests_tool(self, name, args, platform, user_id, **kwargs):
    if name != "keytao_submit_batch":
        return await self.original_tool(name, args, platform, user_id, **kwargs)
    self.assertEqual((platform, user_id), ("qq", self.replay.author.actor_id))
    self.replay.calls.append((name, copy.deepcopy(args)))
    for index, item in enumerate(self.replay.draft, 1):
        item.setdefault("id", index)
    if not args.get("confirmed"):
        return json.dumps(self.preview())
    self.assertEqual(args, {
        "batch_id": "offline-batch", "confirmed": True, "expected_content_version": 1,
        "expected_server_snapshot_digest": "a" * 64,
        "expected_warning_digest": "b" * 64, "expected_audit_digest": "c" * 64,
    })
    self.replay.writes.append((name, copy.deepcopy(args)))
    self.submitted.append(copy.deepcopy(self.replay.draft))
    return json.dumps(self.replay.result(snapshotItems=copy.deepcopy(self.replay.draft)))

async def AddKeepsSubmitTests_offer(self, words=("屡有", "云翻")):
    replay = self.replay
    pending = replay.commands._pending_state_from_server_warning(
        PendingToolConfirm("keytao_submit_batch", {"batch_id": "offline-batch"}), self.preview(),
    )
    self.assertTrue(replay.store.set(replay.author, pending))
    replay.history[:] = [{"role": "assistant", "content": "回复「确认」提交，或「取消」。"}]
    await replay.turn({"message": "加词：" + "，".join(words), "to_me": True}, raw_operation(
        op="add", operands=list(words), codes=[], frame="direct", refers_to_pending=False,
    ))
    self.assertIn("加入并提交", replay.last_response)
    self.assertEqual(replay.writes, [])
    self.assertEqual(replay.draft, self.old_draft)
    return replay.store.get_record(replay.author)

async def AddKeepsSubmitTests_answer(self, message):
    replay = self.replay
    return await replay.commands.handle_pending_message_core(
        message, "qq", replay.author.actor_id, replay.author,
        space_key=replay.author.space_key, allow_intent_model=False,
    )

# Source: test_b9_msgid.py
from keytao_bot.harness.state import SQLiteConversationStateStore as SQLiteConversationStateStore
b9_msgid_chat = harness.openai_chat_module

b9_msgid_routing = harness.chat_routing_module

def MessageIdBindingTests_setUp(self):
    self.store = harness.MemoryConversationStateStore()
    self.owner = harness.ConversationAddress.group("qq", "b9-group", "b9-owner")
    self.prompt = "Original candidate prompt"
    self.assertTrue(self.store.set(
        self.owner, harness.PendingAddWord(
            "发薪", "fsxb", [("fsxb", False)], server_candidates=[("fsxb", False)],
        ),
        origin_message_id="501",
    ))
    self.record = self.store.get_record(self.owner)
    self.assertTrue(self.store.bind_origin_prompt_digest(
        self.owner, b9_msgid_chat._prompt_capability_digest(self.prompt),
    ))
    self.patcher = patch.object(b9_msgid_chat, "conversation_state_store", self.store)
    self.patcher.start()
    self.addCleanup(self.patcher.stop)
    network = patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only"))
    network.start()
    self.addCleanup(network.stop)

def MessageIdBindingTests_context(self, message_id, text):
    reply = SimpleNamespace(
        is_reply=True, is_to_bot=True, message_id=message_id, text=text,
        mentioned_user_ids=(), sender_id="b9-bot", images=(),
    )
    return b9_msgid_chat.TurnContext(
        bot=object(), event=object(), platform="qq", user_id=self.owner.actor_id,
        conv_key=self.owner, space_key=self.owner.space_key,
        reply_reference=reply,
    )

def MessageIdBindingTests_delivery(self):
    memory = harness.ChatMemoryContext(
        platform="qq", user_id=self.owner.actor_id, space_type="group", space_id=self.owner.space_id,
    )
    prompt = b9_msgid_chat._render_live_single_candidate_record(self.record)
    self.assertTrue(prompt)
    bot = type("Bot", (), {"__module__": "nonebot.adapters.onebot.v11"})()
    bot.send = AsyncMock(return_value={"message_id": 601})
    event = SimpleNamespace(message_id=123)
    segment = SimpleNamespace(reply=lambda value: "[reply]", at=lambda value: "[at]")
    return memory, prompt, bot, event, segment

# Source: test_b9_parallel.py
from keytao_bot.utils.pending_confirmation import UNBOUND_BINDING_PRECHECK_NOTICE as UNBOUND_BINDING_PRECHECK_NOTICE
def B9ParallelTests_setUp(self):
    self.words = ("发薪", "云翻", "没头没尾", "国潮")
    self.replay = Replay({
        "sender": "b9-parallel-owner",
        "candidate_facts": {
            "发薪": {"recommended": "fsxb", "codes": ["fsxb", "fsxba"], "pinyin": "fa xin"},
            "云翻": {"recommended": "yjfj", "codes": ["yjfj", "yjfja"], "pinyin": "yun fan"},
            "没头没尾": {"recommended": "mtmw", "codes": ["mtmw", "mtmwa"], "pinyin": "mei tou mei wei"},
            "国潮": {"recommended": "gljz", "codes": ["gljz", "gljza"], "pinyin": "guo chao"},
        },
    }, harness, FakeClient)
    self.active = 0
    self.peak = 0
    self.completed = []
    self.failures = {}
    self.evidence = {}
    self.unresolved = set()
    self.replay.commands.user_resolver.reset_binding_notice_fact()
    self.addCleanup(self.replay.commands.user_resolver.reset_binding_notice_fact)
    review = self.replay.commands.keytao_review
    token = review.begin_commonness_evidence_turn()
    self.addCleanup(review.end_commonness_evidence_turn, token)
    for context in (
        patch.object(self.replay.commands, "conversation_state_store", self.replay.store),
        patch.object(self.replay.chat, "conversation_state_store", self.replay.store),
        patch.object(self.replay.commands, "call_tool_function", AsyncMock(side_effect=self.tool)),
        patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")),
        patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Offline only")),
    ):
        context.start()
        self.addCleanup(context.stop)

async def B9ParallelTests_tool(self, name, args, *actor, **kwargs):
    if name == "keytao_prepare_reviewed_add":
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            for _ in range(4 if args["word"] == self.words[0] else 1):
                await asyncio.sleep(0)
            self.completed.append(args["word"])
            failure = self.failures.get(args["word"])
            if isinstance(failure, Exception):
                raise failure
            if failure is not None:
                return json.dumps(failure, ensure_ascii=False)
            if args["word"] in self.evidence:
                self.replay.commands.keytao_review.record_commonness_evidence({
                    "summary": self.evidence[args["word"]],
                })
        finally:
            self.active -= 1
    result = await self.replay.tool(name, args, *actor, **kwargs)
    if name == "keytao_prepare_reviewed_add" and args["word"] in self.unresolved:
        payload = json.loads(result)
        payload.update(pronunciationUnresolved=True, message=f"「{args['word']}」读音存在冲突，暂不推荐编码。")
        return json.dumps(payload)
    return result

async def B9ParallelTests_prepare(self):
    return await self.replay.commands._prepare_multi_word_query(
        self.words, "qq", self.replay.author.actor_id, self.replay.author,
        self.replay.author.space_key, "Offline owner", explicit_add=True,
    )

# Source: test_b9_small.py
from keytao_bot.harness.state import server_warning_pending_state as server_warning_pending_state
B9CandidateTests_asyncSetUp = query.SemanticQueryRouteTests.asyncSetUp

B9CandidateTests_asyncTearDown = query.SemanticQueryRouteTests.asyncTearDown

B9CandidateTests_lookup = query.SemanticQueryRouteTests.lookup

async def B9CandidateTests_offer(self, missing=(), *, explicit=True):
    self.entries.update({"喜上眉梢": "xemev", "欢天喜地": "htxda"})
    words = ("喜上眉梢", "欢天喜地", *missing)
    return await self.commands._try_handle_simple_single_word_query(
        ("加词：" if explicit else "") + "，".join(words),
        "qq", self.author.actor_id, self.author, self.author.space_key, "Offline owner",
    )

# Source: test_b9_speed.py
from keytao_bot.utils.observability import begin_turn_metrics as begin_turn_metrics
from keytao_bot.utils.observability import end_turn_metrics as end_turn_metrics
async def B9LookupTests_asyncSetUp(self):
    self.commands = harness.chat_commands_module
    self.store = MemoryConversationStateStore()
    self.calls = []
    self.entries = {"喜上眉梢": "xemev"}
    self.pending = [{
        "source": source, "batchId": batch, "batchUrl": f"https://offline.invalid/batch/{batch}",
        "batchStatus": status, "itemStatus": "Pending", "action": "Create",
        "word": word, "code": code, "type": "Phrase",
    } for source, batch, status, word, code in (
        ("draft", "draft", "Draft", "发薪", "fsxb"),
        ("submitted", "submitted", "Submitted", "云翻", "yjfj"),
    )]
    self.lookup_failure = False
    self.pending_failure = False
    self.patches = (
        patch.object(self.commands, "conversation_state_store", self.store),
        patch.object(self.commands, "call_tool_function", AsyncMock(side_effect=self.tool)),
        patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")),
    )
    for context in self.patches:
        context.start()
        self.addCleanup(context.stop)

def B9LookupTests_row(self, word):
    code = self.entries.get(word)
    return {"success": not self.lookup_failure, "word": word,
            "phrases": [{"word": word, "code": code, "type": "Phrase"}] if code else []}

async def B9LookupTests_tool(self, name, args, *actor, **kwargs):
    self.calls.append((name, args))
    if name == "keytao_lookup_by_word":
        result = self.row(args["word"])
    elif name == "keytao_lookup_by_words_batch":
        result = {"success": not self.lookup_failure,
                  "results": [self.row(word) for word in args["words"]]}
    elif name == "keytao_pending_items_by_words":
        result = {"success": not self.pending_failure, "complete": not self.pending_failure,
                  "message": "Offline pending unavailable",
                  "items": [row for row in self.pending if row["word"] in args["words"]]}
    elif name == "keytao_prepare_reviewed_add":
        result = {"success": False, "message": f"Offline review unavailable: {args['word']}"}
    else:
        raise AssertionError(name)
    return json.dumps(result, ensure_ascii=False)

async def B9LookupTests_prepare(self, words):
    return await self.commands._prepare_multi_word_query(
        words, "qq", "b9-owner", None, None, "Offline owner",
    )

def B9PronunciationEvidenceTests_setUp(self):
    review._clear_review_caches()
    if hasattr(review, "_evidence_host_failures"):
        review._evidence_host_failures.clear()

def B9PronunciationEvidenceTests_timeout_types(self):
    class TransportError(Exception):
        pass

    class TimeoutException(TransportError):
        pass

    class ConnectTimeout(TimeoutException):
        pass

    class ConnectError(TransportError):
        pass

    return TransportError, TimeoutException, ConnectTimeout, ConnectError

# Source: test_hotfix_explicit_add.py
harness._fake_nonebot.init = lambda: None

harness._fake_log.logger.exception = harness._fake_log.logger.error

from keytao_bot.harness.operation_request import OperationRequest as OperationRequest
from keytao_bot.utils.candidate_inventory import select_candidate_inventory as select_candidate_inventory
from keytao_bot.utils.explicit_code import ExplicitEntryCodeRequest as ExplicitEntryCodeRequest
from keytao_bot.utils.keytao_review import _build_statuses_for_codes as _build_statuses_for_codes
hotfix_explicit_add_WORD = "没头没尾"

READING = "mei tou mei wei"

hotfix_explicit_add_CODES = ["mtmw", "mtmwa", "mtmwao"]

ALTERNATE_CODES = ["mtmy", "mtmya", "mtmyao"]

def reviewed_word(layout="mixed"):
    groups = []
    for reading, codes in ((READING, hotfix_explicit_add_CODES), ("mei tou mei yi", ALTERNATE_CODES)):
        group = {
            "pinyin": reading, "normalized": reading.split(), "codes": list(codes),
            "recommendedCode": codes[0], "score": 100, "fallback": False,
            "requiresManualReview": False,
            "sources": [{"source": "汉典（经编码服务）", "url": "https://www.zdic.net/hans/没头没尾"}],
        }
        if layout == "per_group" or (layout == "mixed" and reading == READING):
            group["candidateStatuses"] = _build_statuses_for_codes(codes, {})
        groups.append(group)
    result = {
        "success": True, "word": hotfix_explicit_add_WORD, "type": "Phrase", "existing": [],
        "pronunciations": groups, "recommendedCode": hotfix_explicit_add_CODES[0],
        "reviewDisposition": "PASS", "needsManualReview": False,
        "lookupFailed": False, "autoReviewable": True,
        "pronunciationEvidenceStatus": "found", "pronunciationEvidenceComplete": True,
    }
    if layout != "per_group":
        result["candidateStatuses"] = _build_statuses_for_codes(hotfix_explicit_add_CODES + ALTERNATE_CODES, {})
    return result

async def ExplicitAddHotfixTests_asyncSetUp(self):
    self.chat = harness.openai_chat_module
    self.commands = harness.chat_commands_module
    self.store = MemoryConversationStateStore()
    self.author = harness.ConversationAddress.group("qq", "hotfix-group", "hotfix-owner")
    self.review = reviewed_word()
    self.calls = []
    self.writes = []
    patches = (
        patch.object(self.chat, "conversation_state_store", self.store),
        patch.object(self.chat, "draft_operation_coordinator", DraftOperationCoordinator()),
        patch.object(self.chat, "call_tool_function", side_effect=self.call),
        patch.object(self.chat, "_classify_message_command_intent", AsyncMock(side_effect=AssertionError("No model calls"))),
        patch.object(self.commands.user_resolver, "resolve_actor_binding", AsyncMock(return_value=True)),
        patch.object(self.commands, "_format_draft_response", AsyncMock(return_value="Offline draft receipt")),
    )
    for context in patches:
        context.start()
        self.addCleanup(context.stop)
    token = self.chat._current_conversational_turn.set(None)
    self.addCleanup(self.chat._current_conversational_turn.reset, token)

async def ExplicitAddHotfixTests_call(self, name, args, platform, user_id, **kwargs):
    self.assertEqual((platform, user_id), ("qq", self.author.actor_id))
    self.calls.append((name, copy.deepcopy(args)))
    if name == "keytao_prepare_reviewed_add":
        self.assertEqual(args, {"word": hotfix_explicit_add_WORD, "requested_code": "mtmwi" if self._testMethodName == "test_code_outside_reviewed_groups_is_rejected" else "mtmw"})
        audit = {"success": True, "verdict": "pass", "autoApprove": True,
                 "needsManualReview": False, "summary": "Offline reviewed fixture", "issues": []}
        # Run the real tool wrapper; all evidence, audit and ordering IO is replaced.
        with patch.object(harness._review_tools, "prepare_reviewed_word", AsyncMock(return_value=copy.deepcopy(self.review))), patch.object(
            harness._review_tools, "_build_pre_submit_audit", AsyncMock(return_value=audit),
        ), patch.object(harness._review_tools, "assess_candidate_chain_commonness", AsyncMock(return_value=[])):
            result = await harness._review_tools.keytao_prepare_reviewed_add(hotfix_explicit_add_WORD)
        return json.dumps(result, ensure_ascii=False)
    if name in {"keytao_lookup_by_code", "keytao_lookup_by_word"}:
        return json.dumps({"success": True, "phrases": []})
    if name == "keytao_pending_items_by_words":
        return json.dumps({"success": True, "complete": True, "items": []})
    self.assertEqual(name, "keytao_create_phrase", "Unexpected offline tool")
    result = {"batchId": "offline-hotfix", "contentVersion": 1,
              "batchUrl": "https://offline.invalid/batch/offline-hotfix"}
    if not args.get("confirmed"):
        return json.dumps({**result, "success": False, "requiresConfirmation": True,
                           "warningDigest": "a" * 64, "warnings": []})
    self.writes.append(copy.deepcopy(args))
    return json.dumps({**result, "success": True, "successCount": 1, "failedCount": 0})

async def ExplicitAddHotfixTests_execute(self, route="model", code="mtmw"):
    if route == "model":
        request = OperationRequest(op="add", frame="direct", source="model",
                                   operands=(hotfix_explicit_add_WORD,), codes=(code,))
    else:
        request = understand_message(f"加词 {hotfix_explicit_add_WORD} {code}")
        self.assertEqual((request.op, request.operands, request.codes), ("add", (hotfix_explicit_add_WORD,), (code,)))
        return await self.commands._try_handle_complete_add_command(
            f"加词 {hotfix_explicit_add_WORD} {code}", "qq", self.author.actor_id,
            self.author, self.author.space_key, "Offline owner",
        )
    command = self.commands.render_understood_command(request)
    self.assertTrue(command)
    return await self.commands.execute_understood_command(
        request, command, "qq", self.author.actor_id, self.author, self.author.space_key, "Offline owner",
    )

def ExplicitAddHotfixTests_assert_created(self, reply):
    self.assertEqual(len(self.writes), 1, reply)
    item = self.writes[0]
    self.assertEqual((item.get("action", "Create"), item["word"], item["code"]), ("Create", hotfix_explicit_add_WORD, "mtmw"))
    self.assertIn("读音 " + READING, item["remark"])
    self.assertEqual(sum(name == "keytao_create_phrase" for name, _ in self.calls), 2)

def ExplicitAddHotfixTests_mark_resolvable_ambiguity(self):
    self.review.update(
        pronunciationUnresolved=True, reviewDisposition="BLOCK",
        reviewVerdictSite="pronunciation_unresolved",
        multiSenseChoice={"status": "ambiguous"},
    )

# Source: test_hotfix_multiword.py
from keytao_bot.utils.pending_confirmation import render_remediation_reply as render_remediation_reply
async def HotfixMultiwordTests_asyncSetUp(self):
    self.chat = harness.openai_chat_module
    self.commands = harness.chat_commands_module
    self.store = MemoryConversationStateStore()
    self.memory = harness.ChatMemoryContext(
        platform="qq", user_id="offline-owner", space_type="group",
        space_id="offline-multiword", speaker_name="Offline owner",
    )
    self.author = self.memory.conversation_address
    self.client = FakeClient()
    self.tool = AsyncMock(side_effect=self.lookup)
    self.main = AsyncMock(side_effect=AssertionError("Unexpected model call"))
    self.finish = AsyncMock()
    self.unresolved = {"车损"}
    self.failure_mode = "code_unresolved"
    self.unresolved_pronunciations = []
    self.codes = {
        "没头没尾": ("mtmw", "mtmwa"), "国潮": ("gljz", "glwz"),
        "泥炭": ("nktf", "nktfa"), "棉麻": ("mmms", "mmmsa"),
        "车损": ("wesy", "wesya"),
    }
    patches = (
        patch.dict(os.environ, {
            "KEYTAO_SEMANTIC_ROUTE_OPS": "query", "KEYTAO_SEMANTIC_SHADOW": "0",
        }),
        patch.object(self.chat, "AsyncOpenAI", return_value=self.client),
        patch.object(self.chat, "WORD_QUERY_INTENT_MODEL", "offline-model"),
        patch.object(self.chat, "conversation_state_store", self.store),
        patch.object(self.chat, "call_tool_function", self.tool),
        patch.object(self.chat, "get_ai_response_core", self.main),
        patch.object(self.chat, "get_history", return_value=[]),
        patch.object(self.chat, "remember_conversation"),
        patch.object(self.chat, "schedule_memory_compaction"),
        patch.object(self.chat, "_finish_ai_chat_response", self.finish),
        patch.object(self.commands.user_resolver, "resolve_actor_binding", AsyncMock(return_value=True)),
        patch.object(self.chat._user_resolver, "resolved_binding_for_notice", return_value=True),
    )
    for context in patches:
        context.start()
        self.addCleanup(context.stop)
    for variable, value in (
        (self.chat._current_conversational_turn, None),
        (self.chat.current_draft_delivery_claims, []),
        (self.chat._current_turn_message, ""),
    ):
        token = variable.set(value)
        self.addCleanup(variable.reset, token)

@staticmethod
def HotfixMultiwordTests_unresolved_message(word):
    return f"「{word}」没有通过逐字权威读音交叉校验的候选读音，暂不推荐编码"

async def HotfixMultiwordTests_unresolved_review(self, word, result):
    if self.failure_mode == "code_unresolved":
        # Use the lookup producer and the review failure path, without I/O.
        encoded = harness._normalize_encode_response(word, {"success": False, "codes": []})
        self.assertEqual(encoded["message"], render_remediation_reply(
            "编码服务未能返回有效候选编码", command=f"加词 {word}", words=(word,),
        ))
        with (
            patch.object(review, "collect_pronunciation_evidence_limited", AsyncMock(return_value={})),
            patch.object(review, "fetch_keytao_encode", AsyncMock(return_value=encoded)),
            patch.object(review, "lookup_words", AsyncMock(return_value={})),
        ):
            review_result = await review.prepare_reviewed_word(
                review.ReviewHttpConfig("https://offline.invalid", "offline"), word,
            )
        self.assertFalse(review_result["success"])
        self.assertEqual(review_result["reviewVerdictSite"], "code_unresolved")
        self.assertNotIn("pronunciationUnresolved", review_result)
        return review_result
    if self.failure_mode == "missing_reading":
        result["pronunciations"][0]["pinyin"] = ""
        return result
    result.update(
        pronunciationUnresolved=True, pronunciations=self.unresolved_pronunciations,
        recommendedCode="", message=self.unresolved_message(word),
    )
    return result

async def HotfixMultiwordTests_lookup(self, name, args, platform, user_id, **kwargs):
    self.assertEqual((platform, user_id), ("qq", self.author.actor_id))
    if name == "keytao_pending_items_by_words":
        result = {"success": True, "complete": True, "items": []}
    elif name == "keytao_lookup_by_word":
        result = {"success": True, "phrases": []}
    elif name == "keytao_lookup_by_words_batch":
        result = {"success": True, "results": [
            {"success": True, "word": word, "phrases": []} for word in args["words"]
        ]}
    elif name == "keytao_prepare_reviewed_add":
        word = args["word"]
        manual = word in {"国潮", "棉麻"}
        result = {
            "success": True, "word": word, "type": "Phrase",
            "recommendedCode": self.codes[word][0],
            "needsManualReview": manual, "requiresManualReview": manual,
            "preSubmitAudit": {
                "success": True, "verdict": "needs_admin" if manual else "pass",
                "autoApprove": not manual, "summary": "Offline pronunciation evidence",
                "issues": [],
            },
            "pronunciations": [{
                "pinyin": "offline reading", "recommendedCode": self.codes[word][0],
                "sources": [{"source": "semantic_context" if manual else "offline"}],
                "candidateStatuses": [
                    {"code": code, "occupied": False, "label": "空位", "words": []}
                    for code in self.codes[word]
                ],
            }],
        }
        if word in self.unresolved:
            result = await self.unresolved_review(word, result)
    else:
        self.fail(f"Unexpected offline tool: {name}")
    return json.dumps(result, ensure_ascii=False)

async def HotfixMultiwordTests_query(self, words):
    message = " ".join(words)
    self.chat._current_turn_message.set(message)
    self.client.create.return_value.choices[0].message.content = json.dumps(
        raw_operation(op="query", operands=list(words), codes=[], frame="none"),
        ensure_ascii=False,
    )
    ctx = self.chat.TurnContext(
        bot=None, event=None, platform="qq", user_id=self.author.actor_id,
        message_text=message, normalized_message_text=message,
        conv_key=self.author, space_key=self.author.space_key,
        memory_context=self.memory, owner_label="Offline owner",
        command_intent_for=AsyncMock(return_value=self.chat.MessageCommandIntent()),
    )
    with patch.object(self.commands, "_prepare_multi_word_query", wraps=self.commands._prepare_multi_word_query) as prepare:
        self.assertTrue(await self.chat._stage_understand_operation_early(ctx))
    prepare.assert_awaited_once()
    self.assertEqual(prepare.await_args.args[0], words)
    self.client.create.assert_awaited_once()
    self.main.assert_not_awaited()
    self.finish.assert_awaited_once()
    self.assertEqual([
        call.args[1]["word"] for call in self.tool.await_args_list
        if call.args[0] == "keytao_prepare_reviewed_add"
    ], list(words))
    return ctx.response, self.chat._prepare_user_facing_reply(ctx.response, self.memory)

async def HotfixMultiwordTests_assert_candidates(self, words):
    raw, delivered = await self.query(words)
    self.assertNotIn("当前没有可验证的可执行操作，本次未写入。", delivered)
    self.assertEqual(delivered.splitlines(), [line for line in raw.splitlines() if line])
    record = self.store.get_record(self.author)
    self.assertIsNotNone(record)
    self.assertIsInstance(record.state, PendingToolConfirm)
    resolved = [word for word in words if word not in self.unresolved]
    self.assertEqual(record.state.args["_query_words"], list(words))
    self.assertEqual([row["word"] for row in record.state.args["items"]], resolved)
    self.assertEqual([row["word"] for row in record.state.args["_candidate_scopes"]], resolved)
    for block in record.state.args["_query_other_blocks"]:
        self.assertFalse(self.chat.advertised_reply_contract(block).requires_live_state)
    self.assertTrue(self.chat._advertised_reply_matches_live_record(delivered, record))
    self.assertEqual(
        self.chat._render_live_batch_record(record).splitlines(), raw.splitlines(),
    )
    self.assertEqual(record.owner_key, self.author)
    for command in self.chat.advertised_reply_contract(delivered).command_suggestions:
        self.assertTrue(harness.chat_routing_module.message_authorizes_live_pending_mutation(command, record.state), command)
    selection = "，".join(f"{word} 1" for word in resolved)
    selected, intent, error = self.chat._resolve_multi_word_pending_candidate_selection(record.state, selection)
    self.assertIsNone(error)
    self.assertIsNotNone(intent)
    self.assertEqual([row["word"] for row in selected.args["items"]], resolved)
    for word in words:
        if word in self.unresolved:
            line = f"「{word}」暂时没有可核验的候选，请单独查询此词。"
            self.assertEqual(delivered.splitlines().count(line), 1)
            self.assertNotIn(f"「{word}」候选编码", delivered)
        else:
            self.assertIn(f"「{word}」", delivered)
            for code in self.codes[word]:
                self.assertIn(code, delivered)
    return delivered

async def HotfixMultiwordTests__check_standalone_shift_real_server_plan(self, *, keeper_in_draft):
    tools = harness._draft_tools
    occupant = {"word": "我送给", "code": "wsg", "type": "Phrase", "weight": 100}
    keeper = {"word": "吴三桂", "code": "wsg", "type": "Phrase", "weight": 100,
              "action": "Create", "id": "7", "needsManualReview": False, "remark": "Offline reviewed"}
    snapshot = {"success": True, "batchId": "offline", "contentVersion": 1, "items": [keeper] if keeper_in_draft else []}
    writes = []

    async def words(requested):
        return {"success": True, "results": [
            {"word": word, "phrases": [occupant] if word == "我送给" else [keeper] if not keeper_in_draft else []}
            for word in requested
        ]}

    async def codes(requested):
        return {"success": True, "results": [
            {"code": code, "phrases": ([occupant] + ([] if keeper_in_draft else [keeper])) if code == "wsg" else []}
            for code in requested
        ]}

    async def strict(platform, user, items, **kwargs):
        self.assertNotIn("吴三桂", [row["word"] for row in items])
        if kwargs.get("confirmed"):
            writes.extend(items)
            return {"success": True, "batchId": "offline", "contentVersion": 2, "successCount": len(items)}
        return {"success": False, "requiresConfirmation": True, "warningDigest": "c" * 64, "warnings": []}

    async def call(name, args, platform, user, **kwargs):
        if name == "keytao_lookup_by_words_batch":
            result = await words(args["words"])
        elif name == "keytao_list_draft_items":
            result = snapshot
        elif name == "keytao_lookup_by_code":
            result = {"success": True, "phrases": [occupant] + ([] if keeper_in_draft else [keeper])}
        elif name == "keytao_encode":
            self.assertEqual(args, {"word": "我送给", "requested_code": "wsg"})
            result = {"success": True, "word": "我送给", "candidateCodes": ["wsg", "wsgu", "wsgo"]}
        elif name == "keytao_shift_phrase_code":
            result = await tools.keytao_shift_phrase_code(platform, user, **args)
        else:
            raise AssertionError(name)
        return json.dumps(result)

    with (
        patch.object(self.commands, "call_tool_function", side_effect=call),
        patch.object(self.commands, "_format_draft_response", AsyncMock(return_value="草稿地址：https://offline.invalid/draft")),
        patch.object(tools, "_lookup_words_raw", side_effect=words),
        patch.object(tools, "_lookup_codes_raw", side_effect=codes),
        patch.object(tools, "_fetch_encode_candidates", AsyncMock(return_value={"success": True, "candidateCodes": ["wsg", "wsgu", "wsgo"]})),
        patch.object(tools, "keytao_list_draft_items", AsyncMock(return_value=snapshot)),
        patch.object(tools, "_keytao_strict_batch_add_to_draft", side_effect=strict),
        patch.dict(os.environ, {"KEYTAO_SEMANTIC_ROUTE_OPS": "direct_exec"}),
    ):
        reply = await self.commands.try_handle_advertised_shift(
            "我送给 顺延", "qq", self.author.actor_id, self.author, self.author.space_key, "Offline owner",
        )
    self.assertEqual([(row["action"], row["word"], row["code"]) for row in writes], [
        ("Delete", "我送给", "wsg"), ("Create", "我送给", "wsgu"),
    ])
    self.assertIn("我送给", reply)
    self.assertIn("wsgu", reply)

# Source: test_s54_multiword.py
s54_multiword_chat = harness.openai_chat_module

s54_multiword_commands = harness.chat_commands_module

def s54_multiword_review_payload(word):
    code, reading = {
        "大端": ("dsdt", "dà duān"),
        "小端": ("xcdt", "xiǎo duān"),
        "肌群": ("jkqt", "jī qún"),
    }[word]
    codes = [code, code + "v", code + "vo"]
    return {
        "success": True, "word": word, "recommendedCode": codes[-1],
        "autoReviewable": False, "needsManualReview": True,
        "manualReviewReason": "常用度证据不足",
        "preSubmitAudit": {"summary": "常用度证据不足"},
        "pronunciations": [{
            "pinyin": reading, "recommendedCode": codes[-1],
            "sources": [{"source": "汉典"}],
            "candidateStatuses": [
                {"code": candidate, "occupied": index < 2,
                 "words": ["占位词"] if index < 2 else [],
                 "label": "已有「占位词」" if index < 2 else "空位"}
                for index, candidate in enumerate(codes)
            ],
        }],
    }

async def MultiwordRouteTests_discover(self, message="大端 小端", *, existing=(), store=None):
    store = store or harness.MemoryConversationStateStore()
    key = harness.ConversationAddress.private("qq", "s54-offline")

    async def tool(name, args, *_args, **_kwargs):
        if name == "keytao_lookup_by_word":
            return json.dumps({"success": True, "phrases": [
                {"word": args["word"], "code": "xcdtio", "type": "Phrase"}
            ] if args["word"] in existing else []})
        if name == "keytao_pending_items_by_words":
            return json.dumps({"success": True, "complete": True, "items": []})
        if name == "keytao_prepare_reviewed_add":
            return json.dumps(s54_multiword_review_payload(args["word"]))
        raise AssertionError(name)

    with patch.object(s54_multiword_chat, "conversation_state_store", store), patch.object(
        s54_multiword_chat, "call_tool_function", side_effect=tool,
    ), patch.object(s54_multiword_chat, "_classify_simple_word_query_intent", side_effect=AssertionError("model routing")):
        response = await s54_multiword_commands._try_handle_simple_single_word_query(message, "qq", "s54-offline", key)
    return response, store, key

# Source: test_s54_renderer.py
from keytao_bot.utils.pending_confirmation import front_insert_recommendation_copy as front_insert_recommendation_copy
from keytao_bot.utils.pending_confirmation import render_server_backed_batch_candidates as render_server_backed_batch_candidates
def reviewed_scope(word, codes, recommended, pinyin, occupied):
    candidates = [(code, code in occupied) for code in codes]
    prompt = "\n".join((
        f"词库暂无收录「{word}」：",
        f"审词：读音 {pinyin}；来源 组合推断（{word}模式）；",
        "自动审核：需管理员审核（组合推断仍需人工核验）",
        "候选编码:",
        *(f"{index}. {code} — " + (
            "已有「" + "、".join(occupied[code]) + "」"
            if code in occupied else "空位"
        ) + ("（推荐）" if code == recommended else "")
          for index, code in enumerate(codes, 1)),
        f"• 「{word}」→ {recommended}（推荐）",
        "回复编号或编码选择（可多选，如「添加1、2」）；",
        "回复「加入」写入草稿，或回复「加入并提交」写入并提交。",
    ))
    state = {
        "word": word,
        "recommendedCode": recommended,
        "serverCandidates": candidates,
        "serverOccupiedWords": occupied,
        "serverOrderingAssessments": [],
    }
    return {
        "word": word,
        "candidates": candidates,
        "occupiedWords": occupied,
        "orderingAssessments": [],
        "reviewedState": state,
        "reviewedPrompt": prompt,
    }

def S54RendererTests_setUp(self):
    self.items = [
        {"word": "大端", "code": "dsdtvo", "needsManualReview": True},
        {"word": "小端", "code": "xcdti", "needsManualReview": True},
    ]
    self.scopes = [
        reviewed_scope("大端", ["dsdt", "dsdtv", "dsdtvo"], "dsdtvo",
                       "dà duān", {"dsdt": ["打断"], "dsdtv": ["大段"]}),
        reviewed_scope("小端", ["xcdt", "xcdti", "xcdtio"], "xcdti",
                       "xiǎo duān", {"xcdt": ["小段"]}),
    ]

# Source: test_s54_selection.py
def reviewed_record(words=("大端", "小端")):
    inventories = {
        "大端": [("dsdt", True), ("dsdtv", True), ("dsdtvo", False)],
        "小端": [("xcdt", True), ("xcdti", False), ("xcdtio", False)],
    }
    items = []
    scopes = []
    for word in words:
        candidates = inventories[word]
        code = next(code for code, occupied in candidates if not occupied)
        state = PendingAddWord(
            word=word,
            recommended_code=code,
            candidates=candidates,
            server_candidates=candidates,
            needs_manual_review=True,
            manual_review_reason="Manual review required",
        )
        items.append({"action": "Create", "word": word, "code": code})
        scopes.append({
            "word": word,
            "candidates": [list(pair) for pair in candidates],
            "occupiedWords": {},
            "orderingAssessments": [],
            "reviewedState": _pending_add_word_payload(state),
            "reviewedPrompt": "Reviewed",
        })
    return PendingToolConfirm(
        function_name="keytao_batch_add_to_draft",
        args={"items": items, "_candidate_scopes": scopes,
              "_reviewed_multi_word": True},
    )

def S54SelectionTests_reordered_record(self):
    from tests.test_add_and_select import MultiwordRouteTests
    _, store, key = asyncio.run(MultiwordRouteTests().discover("大端 小端 肌群"))
    record = store.get(key)
    for scope, item, occupant in zip(record.args["_candidate_scopes"], record.args["items"], ("打断", "小段")):
        code = scope["candidates"][0][0]
        assessment = {
            "verdict": "front_more_common", "newWord": scope["word"],
            "occupantWord": occupant, "occupantCode": code,
            "freeCode": item["code"], "newCode": code,
        }
        item["code"] = code
        scope["reviewedState"]["recommendedCode"] = code
        scope["occupiedWords"] = {code: [occupant]}
        scope["reviewedState"]["serverOccupiedWords"] = scope["occupiedWords"]
        scope["orderingAssessments"] = [assessment]
        scope["reviewedState"]["serverOrderingAssessments"] = [assessment]
    return record

# Source: test_s55_failure_boundary.py
import logging as logging
from keytao_bot.utils.observability import current_turn_id as current_turn_id
from keytao_bot.utils.observability import set_turn_flow as set_turn_flow
from keytao_bot.utils.observability import turn_failure_reply as turn_failure_reply
s55_failure_boundary_chat = harness.openai_chat_module

# Source: test_s55_search_breaker.py
import importlib.util
_SPEC = importlib.util.spec_from_file_location(
    "s55_web_search_tools",
    Path(__file__).parents[1] / "keytao_bot/skills/web-search/tools.py",
)

web = importlib.util.module_from_spec(_SPEC)

_SPEC.loader.exec_module(web)

def SearchBreakerTests_setUp(self):
    self.health_patch = patch.object(web, "_SEARCH_BACKEND_HEALTH", {})
    self.status_patch = patch.object(web, "_LAST_BACKEND_STATUS", {})
    self.exa_patch = patch.object(web, "_exa_api_key", return_value=None)
    for active_patch in (self.health_patch, self.status_patch, self.exa_patch):
        active_patch.start()
        self.addCleanup(active_patch.stop)

@staticmethod
def SearchBreakerTests_result(provider):
    return [{"title": "鎗 qiāng", "url": "https://example.com/qiang", "snippet": "鎗 qiāng", "provider": provider}]

@staticmethod
def SearchBreakerTests_trip(backend="so360"):
    for _ in range(web.SEARCH_FAILURE_THRESHOLD):
        generation = web._acquire_search_backend(backend)
        web._record_search_outcome(backend, generation, success=False, elapsed=2.0)

# Source: test_s55_single_pipeline.py
s55_single_pipeline_chat = harness.openai_chat_module

s55_single_pipeline_commands = harness.chat_commands_module

def single_review():
    return {
        "success": True, "word": "鎗", "type": "Single", "encodingType": "单字",
        "recommendedCode": "qx", "needsManualReview": True,
        "variantNote": "「鎗」是「枪」（繁体「槍」）的异体字",
        "manualReviewReason": "异体字需要管理员审核",
        "preSubmitAudit": {"summary": "异体字需要管理员审核"},
        "pronunciations": [{
            "pinyin": "qiāng", "normalized": ["qiang"], "codes": ["qx"],
            "recommendedCode": "qx", "requiresManualReview": True,
            "sources": [{"source": "单字资料"}],
            "candidateStatuses": [{"code": "qx", "occupied": False, "words": []}],
        }],
    }

# Source: test_s55_single_review.py
def single_encoding(word="鎗", *, shape=None, readings=None):
    readings = readings or ["qiāng"]
    codes = ["qx" + (shape or "")[:index] for index in range(len(shape or "") + 1)]
    return {
        "success": True, "input": word, "type": "单字", "codes": codes,
        "candidateCodes": codes, "phrasePinyins": [readings[0]],
        "chars": [{
            "char": word, "pinyin": readings[0], "pinyins": readings,
            "pronunciationLookupStatus": "found", "phoneticCode": "qx",
            "shapeCode": shape, "c1": None, "c2": None,
            "fullCode": "qx" + (shape or ""),
        }],
    }

async def SingleReviewTests_prepare(self, encoding=None, *, occupants=None, existing=None, **kwargs):
    encoding = copy.deepcopy(encoding if encoding is not None else single_encoding())
    with patch.object(review, "fetch_keytao_encode", AsyncMock(return_value=encoding)), patch.object(
        review, "lookup_words", AsyncMock(return_value={"鎗": existing or []}),
    ), patch.object(review, "lookup_codes", AsyncMock(return_value=occupants or {})), patch.object(
        review, "collect_pronunciation_evidence_limited", AsyncMock(side_effect=AssertionError("phrase evidence lane")),
    ):
        return await review.prepare_reviewed_word(review.ReviewHttpConfig("https://fixture.invalid", "fixture"), "鎗", **kwargs)

async def SingleReviewTests_audit_single(self, prepared, *, code="qx"):
    with patch.object(review, "prepare_reviewed_word", AsyncMock(return_value=prepared)), patch.object(
        review, "_assess_semantic_context_auto_pass", return_value={"accepted": False},
    ) as semantic:
        audit = await review.audit_draft_items(
            review.ReviewHttpConfig("https://fixture.invalid", "fixture"),
            [{"id": -55, "action": "Create", "word": "鎗", "code": code, "type": "Single"}],
        )
    return audit, semantic.call_count

# Source: test_s56_commonness.py
from keytao_bot.utils.candidate_inventory import protected_candidate_occupants as protected_candidate_occupants
from keytao_bot.utils.pending_confirmation import advertised_single_word_lookup_codes as advertised_single_word_lookup_codes
from keytao_bot.harness.authorization_grammar import parse_eviction_modified_add as parse_eviction_modified_add
from keytao_bot.plugins.chat_render import _format_reviewed_add_prompt as _format_reviewed_add_prompt
def occupied_review(word="鎗", occupant="强", code="qx", *, phrase_type="Single", free=""):
    statuses = [{
        "code": code, "occupied": True, "words": [occupant],
        "phrases": [{"word": occupant, "code": code, "type": phrase_type, "weight": 10}],
    }]
    if free:
        statuses.append({"code": free, "occupied": False, "words": [], "phrases": []})
    return {
        "success": True, "word": word, "type": phrase_type,
        "recommendedCode": free or code,
        "pronunciations": [{"pinyin": "qiāng", "recommendedCode": free or code,
                            "candidateStatuses": statuses}],
    }

async def S56CommonnessTests_assess(self, subject, verdict):
    compare = AsyncMock(return_value={
        "success": True, "verdict": verdict,
        "summary": "fixture comparison evidence",
    })
    with patch.object(review, "compare_word_commonness", compare):
        assessments = await review.assess_candidate_chain_commonness(subject)
    return review.apply_candidate_ordering_recommendation(subject, assessments), compare

# Source: test_s56_commonness_binding.py
s56_commonness_binding_chat = harness.openai_chat_module

s56_commonness_binding_commands = harness.chat_commands_module

# Source: test_s56_e2e_closure.py
import unicodedata as unicodedata
# Source: test_s56_explicit_code.py
s56_explicit_code_chat = harness.openai_chat_module

s56_explicit_code_commands = harness.chat_commands_module

s56_explicit_code_routing = s56_explicit_code_chat._chat_routing

def reviewed_single():
    return harness.PendingAddWord(
        word="鎗", recommended_code="qx", candidates=[("qx", True)],
        server_candidates=[("qx", True)], occupied_words={"qx": ["强"]},
        server_occupied_words={"qx": ["强"]},
        pronunciation_codes={"qx": "qiāng"}, phrase_type="Single",
        needs_manual_review=True, manual_review_reason="异体字需管理员审核",
    )

# Source: test_s56_generic_eviction.py
s56_generic_eviction_chat = harness.openai_chat_module

s56_generic_eviction_commands = harness.chat_commands_module

async def GenericEvictionTests_run_command(self, message, *, preexisting=False, verdict="behind_more_common"):
    key = harness.ConversationAddress.private("qq", "s56-generic")
    store = harness.MemoryConversationStateStore()

    def reviewed_state():
        return harness.PendingAddWord(
            word="耙耙柑", recommended_code="ppgv",
            candidates=[("ppg", True), ("ppgv", False)],
            server_candidates=[("ppg", True), ("ppgv", False)],
            occupied_words={"ppg": ["琵琶骨"]},
            server_occupied_words={"ppg": ["琵琶骨"]},
            server_ordering_assessments=[{
                "newWord": "耙耙柑", "occupantWord": "琵琶骨", "occupantCode": "ppg",
                "freeCode": "ppgv", "newCode": "ppg" if verdict == "front_more_common" else "ppgv",
                "verdict": verdict,
            }] if verdict else [],
            pronunciation_codes={"ppg": "pa pa gan", "ppgv": "pa pa gan"},
            needs_manual_review=True,
        )

    if preexisting:
        store.set(key, reviewed_state())

    async def review(*_args, **_kwargs):
        store.set(key, reviewed_state())
        return "reviewed fixture"

    async def lookup(name, *_args, **_kwargs):
        self.assertEqual("keytao_lookup_by_code", name)
        return json.dumps({"success": True, "phrases": [
            {"word": "琵琶骨", "code": "ppg", "type": "Phrase", "weight": 100},
        ]})

    ctx = SimpleNamespace(
        response=None, history=[], normalized_message_text=message,
        platform="qq", user_id="s56-generic", conv_key=key, space_key=None, owner_label="",
        current_pending_record=store.get_record(key),
        eviction_modified_add=parse_eviction_modified_add(message), compound_eviction_add_plan=None,
    )
    shift = AsyncMock(return_value="planned fixture")
    add = AsyncMock(side_effect=AssertionError("protected eviction cannot become a plain add"))
    with (
        patch.object(s56_generic_eviction_commands, "conversation_state_store", store),
        patch.object(s56_generic_eviction_commands, "_try_handle_simple_single_word_query", side_effect=review),
        patch.object(s56_generic_eviction_commands, "call_tool_function", side_effect=lookup),
        patch.object(s56_generic_eviction_commands, "_execute_shift_to_code", shift),
        patch.object(s56_generic_eviction_commands, "_execute_add_to_draft", add),
        patch.object(s56_generic_eviction_chat, "_try_handle_explicit_reading_disambiguation", AsyncMock(return_value=None)),
        patch.object(s56_generic_eviction_chat, "_try_handle_compound_shift_modified_add_command", AsyncMock(return_value=None)),
    ):
        await s56_generic_eviction_chat._stage_handle_simple_word_query(ctx)
    self.assertIsNotNone(store.get_record(key))
    self.assertIsNotNone(ctx.response)
    return ctx.response, shift

# Source: test_s56_security.py
from keytao_bot.utils.explicit_code import ExplicitCodeRequest as ExplicitCodeRequest
from keytao_bot.utils.pending_confirmation import advertised_single_word_candidate_codes as advertised_single_word_candidate_codes
s56_security_chat = harness.openai_chat_module

s56_security_commands = harness.chat_commands_module

s56_security_routing = s56_security_chat._chat_routing

# Source: test_s56_undo.py
import hashlib as hashlib
import time as time
s56_undo_chat = harness.openai_chat_module

def s56_undo_row(item_id, word, code, action="Create"):
    return {"id": item_id, "word": word, "code": code, "type": "Single", "action": action,
            "oldWord": None, "weight": 10, "remark": None, "needsManualReview": True}

CHAIN = [s56_undo_row(2, "鎗", "qx"), s56_undo_row(3, "强", "qx", "Delete"), s56_undo_row(4, "强", "qxa"),
         s56_undo_row(5, "戕", "qxa", "Delete"), s56_undo_row(6, "戕", "qxai")]

class DraftServer:
    def __init__(self):
        self.items = [s56_undo_row(1, "旧", "jq")]
        self.version = 1
        self.batch_id = "s56-undo-batch"
        self.url = "https://example.invalid/batch/" + self.batch_id
        self.submitted = False
        self.calls = []
        self.fail_delete = False
        self.remove_old_on_shift = False
        self.modify_old_on_shift = False
        self.restore_failure = ""
        self.restore_calls = []

    def get(self, name):
        async def call(**args):
            self.calls.append((name, args))
            result = {"success": True, "batchId": self.batch_id, "batchUrl": self.url, "contentVersion": self.version}
            if name == "keytao_list_draft_items":
                if self.submitted:
                    raise AssertionError("post-submit draft read")
                return {**result, "items": copy.deepcopy(self.items), "count": len(self.items)}
            if name == "keytao_shift_phrase_code":
                if self.remove_old_on_shift:
                    self.items = []
                elif self.modify_old_on_shift:
                    self.items[0]["weight"] = 12
                self.items.extend(copy.deepcopy(CHAIN))
                self.version += 1
                return {**result, "contentVersion": self.version, "successCount": len(CHAIN)}
            if name == "keytao_submit_batch":
                self.submitted = True
                self.version += 1
                return {**result, "contentVersion": self.version}
            if name == "keytao_recall_batch":
                if "expected_content_version" not in args:
                    return {**result, "success": False, "requiresConfirmation": True}
                if not self.submitted or args["expected_content_version"] != self.version:
                    return {"success": False}
                self.submitted = False
                self.version += 1
                return {**result, "contentVersion": self.version}
            if name == "keytao_batch_remove_draft_items":
                targets = [{field: item[field] for field in ("id", "word", "code", "action", "type")}
                           for item in self.items if item["id"] in args["ids"]]
                digest = hashlib.sha256(json.dumps(targets).encode()).hexdigest()
                if not args.get("expected_target_digest"):
                    return {**result, "success": False, "requiresConfirmation": True, "targets": targets, "targetDigest": digest}
                if self.fail_delete:
                    return {**result, "success": False, "message": "fixture timeout"}
                if args["expected_content_version"] != self.version or args["expected_targets"] != targets:
                    return {"success": False}
                self.items = [item for item in self.items if item["id"] not in args["ids"]]
                self.version += 1
                return {**result, "contentVersion": self.version, "successCount": len(targets)}
            raise AssertionError(name)
        return call

def CompletedUndoTests_setUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.store = DraftMutationClaimStore(self.directory.name + "/journal.db")
    self.store_patch = patch.object(completed_draft_undo, "get_default_draft_mutation_claim_store", return_value=self.store)
    self.store_factory = self.store_patch.start()
    self.capture_token = completed_draft_undo.current_operation_capture.set(None)
    self.key = harness.ConversationAddress.group("qq", "s56-group", "s56-undo")
    self.server = DraftServer()
    async def raw_item(item_id):
        return next(({**copy.deepcopy(item), "batchId": self.server.batch_id} for item in self.server.items if item["id"] == item_id), {})
    self.raw_patch = patch.object(completed_draft_undo, "_fetch_raw_item", side_effect=raw_item)
    self.raw_patch.start()
    async def restore_request(identity, batch_id, items, **ticket):
        self.server.restore_calls.append((items, ticket))
        result = {"success": True, "batchId": batch_id, "contentVersion": self.server.version}
        if ticket.get("confirmed") is not True:
            return {**result, "success": False, "requiresConfirmation": True, "warningDigest": "a" * 64}
        if self.server.restore_failure == "rejected":
            return {**result, "success": False, "message": "fixture rejected"}
        self.assertEqual(ticket["expectedContentVersion"], self.server.version)
        for index, item in enumerate(items):
            self.server.items.append({**{field: None for field in completed_draft_undo._FIELDS}, **item, "id": 100 + index})
        self.server.version += 1
        if self.server.restore_failure == "lost_reply":
            raise TimeoutError("fixture lost reply")
        return {**result, "contentVersion": self.server.version, "pullRequestCount": len(items)}
    self.restore_patch = patch.object(completed_draft_undo, "_restore_request", side_effect=restore_request)
    self.restore_patch.start()

def CompletedUndoTests_tearDown(self):
    completed_draft_undo.current_operation_capture.reset(self.capture_token)
    self.store_patch.stop()
    self.raw_patch.stop()
    self.restore_patch.stop()
    self.directory.cleanup()

async def CompletedUndoTests_write(self, submitted=False):
    completed_draft_undo.start_operation_turn(self.key)
    name = "keytao_shift_phrase_code"
    await completed_draft_undo.invoke_with_operation_journal(name, {"confirmed_plan_digest": "d" * 64}, self.server.get(name), self.server.get)
    if submitted:
        name = "keytao_submit_batch"
        await completed_draft_undo.invoke_with_operation_journal(name, {"confirmed": True, "batch_id": self.server.batch_id}, self.server.get(name), self.server.get)

async def CompletedUndoTests_cancel(self, message="取消", reply=None, key=None):
    key = key or self.key
    completed_draft_undo.start_operation_turn(key)
    return await completed_draft_undo.handle_completed_undo(key, message, reply or SimpleNamespace(), self.server.get)

# Source: test_s56_undo_review.py
# Source: test_s57_claiming.py
s57_claiming_chat = harness.openai_chat_module

# Source: test_s57_e2e_closure_review.py
import ast as ast
from e2e.recording import ArtifactRecorder as ArtifactRecorder
from e2e.scenarios import ScenarioAssertionError as ScenarioAssertionError
from e2e.scenarios import _assert_s56_advertised_reply_closure as _assert_s56_advertised_reply_closure
# Source: test_s57_extra_code.py
from keytao_bot.utils.explicit_code import parse_explicit_code_request as parse_explicit_code_request
from keytao_bot.harness.state import trusted_word_record_is_complete as trusted_word_record_is_complete
s57_extra_code_chat = harness.openai_chat_module

s57_extra_code_commands = harness.chat_commands_module

s57_extra_code_routing = s57_extra_code_chat._chat_routing

def existing_single_review():
    return {
        "success": True, "word": "嘢", "type": "Single", "encodingType": "单字",
        "recommendedCode": "", "needsManualReview": True,
        "manualReviewReason": "单字需管理员审核",
        "existing": [{"word": "嘢", "code": "yeoiav", "type": "Single", "weight": 11}],
        "pronunciations": [{
            "pinyin": "yě", "normalized": ["ye"], "codes": ["ye"],
            "recommendedCode": "", "requiresManualReview": True,
            "candidateStatuses": [{"code": "ye", "occupied": True, "words": ["也", "耶"]}],
        }],
    }

def additional_code_warning():
    """Exact warning shape captured in S57 target 4, tool event 140."""
    return {
        "success": False, "requiresConfirmation": True,
        "batchId": "s57-existing-batch", "contentVersion": 1,
        "warningDigest": "480406e40055ad974b5ecb09ae03f91ff4cf6a514cb335e9d6e337231d65ba2c",
        "warnings": [{
            "index": 0, "warningType": "multiple_code",
            "item": {
                "action": "Create", "word": "嘢", "code": "yeoia", "type": "Single",
                "oldWord": None, "needsManualReview": True,
                "remark": "喵喵审词：读音 yě；形码 oia 未能核验，需管理员复核",
            },
            "message": '词条 "嘢" 已存在于编码 "yeoiav"，将创建多编码词条',
            "existing": {"word": "嘢", "code": "yeoiav", "weight": 11},
        }],
    }

# Source: test_s57_options.py
from keytao_bot.utils.offered_options import FORCE_ASSENT_TEXTS as FORCE_ASSENT_TEXTS
from keytao_bot.utils.offered_options import is_force_assent as is_force_assent
from keytao_bot.utils.offered_options import matches_previous_bot_option as matches_previous_bot_option
from keytao_bot.utils.offered_options import offered_option_intent as offered_option_intent
from keytao_bot.utils.offered_options import option_questions_bind_live_state as option_questions_bind_live_state
from keytao_bot.utils.offered_options import previous_bot_options as previous_bot_options
from keytao_bot.utils.offered_options import render_missing_option_ticket as render_missing_option_ticket
from keytao_bot.utils.offered_options import structural_option_questions as structural_option_questions
s57_options_chat = harness.openai_chat_module

s57_options_routing = s57_options_chat._chat_routing

def ticket(options=None):
    return harness.PendingToolConfirm(
        "keytao_shift_phrase_code",
        {"word": "嘢", "target_code": "yeoiav", "batch_id": "",
         "expected_content_version": 0, "confirmed_plan_digest": "a" * 64,
         "_offered_options": options or {"强制调序": "confirm", "保留现顺序": "cancel"}},
        confirmation_source="server_warning",
    )

# Source: test_s57_reorder.py
s57_reorder_commands = harness.chat_commands_module

s57_reorder_chat = harness.openai_chat_module

s57_reorder_draft_tools = harness._draft_tools

# Source: test_s57_security_review.py
s57_security_review_commands = harness.chat_commands_module

s57_security_review_chat = harness.openai_chat_module

# Source: test_s58_advertisement.py
s58_advertisement_chat = harness.openai_chat_module

def s58_advertisement_shift_ticket():
    return harness.PendingToolConfirm(
        "keytao_shift_phrase_code",
        {"word": "奈飞", "target_code": "nhfw", "batch_id": "",
         "expected_content_version": 0, "confirmed_plan_digest": "a" * 64,
         "_pending_display": {"shiftPlan": {
             "word": "奈飞", "targetCode": "nhfw",
             "currentState": [{"word": "奈飞", "code": "nhfwv"}, {"word": "浓厚氛围", "code": "nhfw"}],
             "proposedState": [{"word": "奈飞", "code": "nhfw"}, {"word": "浓厚氛围", "code": "nhfwa"}],
             "shifted": [{"word": "浓厚氛围", "fromCode": "nhfw", "toCode": "nhfwa"}],
         }}},
        confirmation_source="server_warning",
    )

# Source: test_s58_bridge.py
from keytao_bot.harness.state import server_warning_ticket_is_complete as server_warning_ticket_is_complete
def preview():
    return {
        "success": False, "requiresConfirmation": True,
        "confirmationKind": "shiftPlan", "batchId": "", "contentVersion": 0,
        "planDigest": "a" * 64, "warningDigest": "b" * 64,
        "shiftPlan": {
            "word": "奈飞", "targetCode": "nhfw",
            "items": [
                {"action": "Delete", "word": "奈飞", "code": "nhfwv", "type": "Phrase"},
                {"action": "Create", "word": "奈飞", "code": "nhfw", "type": "Phrase"},
                {"action": "Delete", "word": "浓厚氛围", "code": "nhfw", "type": "Phrase"},
                {"action": "Create", "word": "浓厚氛围", "code": "nhfwa", "type": "Phrase"},
            ],
            "shifted": [{"word": "浓厚氛围", "fromCode": "nhfw", "toCode": "nhfwa"}],
        },
    }

def BridgeTests_setUp(self):
    self.calls = []
    self.proposal = preview()
    self.entries = [{"word": "奈飞", "code": "nhfwv", "type": "Phrase", "weight": 100}]
    self.context = ToolContext(platform="qq", user_id="s58-bridge", current_message="把 奈飞 迁到 nhfw", writes_allowed=False)

    async def dispatch(name, **kwargs):
        self.calls.append((name, kwargs))
        if name == "keytao_lookup_by_words_batch":
            return {"success": True, "results": [{"word": "奈飞", "phrases": copy.deepcopy(self.entries)}]}
        self.assertEqual(name, "keytao_shift_phrase_code")
        self.assertNotIn("confirmed_plan_digest", kwargs)
        self.assertNotIn("expected_warning_digest", kwargs)
        return copy.deepcopy(self.proposal)

    self.executor = ToolExecutor(
        lambda name: lambda **kwargs: dispatch(name, **kwargs),
        frozenset({"keytao_lookup_by_words_batch", "keytao_shift_phrase_code"}),
    )

async def BridgeTests_call(self, args=None, context=None):
    return json.loads(await self.executor.call(
        "keytao_shift_phrase_code", args or {"word": "奈飞", "target_code": "nhfw"}, context or self.context,
    ))

# Source: test_s58_explicit_destination.py
s58_explicit_destination_draft_tools = harness._draft_tools

def reviewed_destination(pinyin="nóng hòu fēn wéi"):
    return {
        "success": True, "word": "浓厚氛围", "type": "Phrase", "recommendedCode": "nhfwa",
        "pronunciations": [{
            "pinyin": pinyin, "recommendedCode": "nhfwa",
            "candidateStatuses": [{"code": code, "occupied": code == "nhfw"} for code in ("nhfw", "nhfwa", "nhfwav")],
        }],
    }

def ExplicitDestinationTests_setUp(self):
    self.rows = [
        {"word": "奈飞", "code": "nhfwv", "type": "Phrase", "weight": 100},
        {"word": "浓厚氛围", "code": "nhfw", "type": "Phrase", "weight": 100},
    ]
    self.candidates = {
        "奈飞": ["nhfw", "nhfwv", "nhfwva"],
        "浓厚氛围": ["nhfw", "nhfwa", "nhfwav"],
        "浓厚风味": ["nhfw", "nhfwa", "nhfwao"],
    }
    self.draft = {"success": True, "batchId": "", "contentVersion": 0, "items": []}
    self.review = reviewed_destination()
    self.strict_calls = []
    self.words = ["奈飞", "浓厚氛围"]

async def ExplicitDestinationTests_encode(self, word, _requested_code=None):
    return {"success": True, "word": word, "candidateCodes": self.candidates[word]}

async def ExplicitDestinationTests_lookup_words(self, words):
    return {"success": True, "results": [
        {"word": word, "phrases": [copy.deepcopy(row) for row in self.rows if row["word"] == word]}
        for word in words
    ]}

async def ExplicitDestinationTests_lookup_codes(self, codes):
    return {"success": True, "results": [
        {"code": code, "phrases": [copy.deepcopy(row) for row in self.rows if row["code"] == code]}
        for code in codes
    ]}

async def ExplicitDestinationTests_strict(self, platform, user, items, **kwargs):
    self.strict_calls.append((copy.deepcopy(items), dict(kwargs)))
    # Exercise the actual final code-validation sink, including stripping
    # private reading capabilities and preserving the manual-review seal.
    valid, failed = await s58_explicit_destination_draft_tools._split_items_by_code_validation(items)
    self.assertEqual(failed, [])
    self.assertEqual(len(valid), len(items))
    if kwargs.get("confirmed"):
        self.draft = {"success": True, "batchId": "s58-local", "contentVersion": 1, "items": valid}
        return {"success": True, "batchId": "s58-local", "contentVersion": 1, "successCount": len(valid)}
    return {"success": False, "requiresConfirmation": True, "warningDigest": "b" * 64, "warnings": []}

async def ExplicitDestinationTests_invoke(self, destination, **kwargs):
    with (
        patch.object(s58_explicit_destination_draft_tools, "_fetch_encode_candidates", self.encode),
        patch.object(s58_explicit_destination_draft_tools, "_lookup_words_raw", self.lookup_words),
        patch.object(s58_explicit_destination_draft_tools, "_lookup_codes_raw", self.lookup_codes),
        patch.object(s58_explicit_destination_draft_tools, "keytao_list_draft_items", AsyncMock(side_effect=lambda *a, **k: copy.deepcopy(self.draft))),
        patch.object(s58_explicit_destination_draft_tools, "prepare_reviewed_word", AsyncMock(side_effect=lambda *a, **k: copy.deepcopy(self.review))),
        patch.object(s58_explicit_destination_draft_tools, "_keytao_strict_batch_add_to_draft", self.strict),
    ):
        return await s58_explicit_destination_draft_tools.keytao_shift_phrase_code(
            "qq", "s58-review", "奈飞", "nhfw",
            ordered_words=self.words, listed_words=self.words,
            expected_codes=["nhfw", destination], **kwargs,
        )

# Source: test_s58_generic_bridge.py
def reviewed(word, code):
    return {
        "type": "Phrase", "pinyin": "nài fēi" if word == "奈飞" else "nóng hòu fēn wéi",
        "candidate_codes": [code], "needs_manual_review": True,
        "remark": "Server-reviewed reading; manual shape review required",
    }

def GenericProposalBridgeTests_setUp(self):
    self.calls = []
    self.preview_override = {}
    self.entries = {"奈飞": [], "浓厚氛围": []}
    self.rows = [
        {"id": 123, "word": "奈飞", "code": "nhfw", "type": "Phrase", "action": "Create"},
        {"id": 124, "word": "浓厚氛围", "code": "nhfwa", "type": "Phrase", "action": "Create"},
    ]
    self.context = ToolContext(
        platform="qq", user_id="s58-generic", writes_allowed=False,
        current_message="安排添加 奈飞 nhfw",
        trusted_reviewed_items_by_key={
            ("奈飞", "nhfw"): reviewed("奈飞", "nhfw"),
            ("浓厚氛围", "nhfwa"): reviewed("浓厚氛围", "nhfwa"),
        },
    )

    async def dispatch(name, **kwargs):
        self.calls.append((name, copy.deepcopy(kwargs)))
        if name == "keytao_lookup_by_words_batch":
            return {"success": True, "results": [
                {"word": word, "phrases": copy.deepcopy(self.entries[word])}
                for word in kwargs["words"]
            ]}
        if name == "keytao_list_draft_items":
            return {"success": True, "batchId": "actor-draft", "contentVersion": 7, "items": copy.deepcopy(self.rows)}
        if name in {"keytao_create_phrase", "keytao_batch_add_to_draft"}:
            self.assertIs(kwargs.get("preview_only"), True)
            self.assertIs(kwargs.get("confirmed"), False)
            self.assertNotIn("expected_warning_digest", kwargs)
            return {
                "success": False, "requiresConfirmation": True,
                "batchId": "actor-draft", "contentVersion": 7,
                "warningDigest": "a" * 64, "message": "Reviewed entry proposal",
                **copy.deepcopy(self.preview_override),
            }
        self.assertIn(name, {"keytao_remove_draft_item", "keytao_batch_remove_draft_items"})
        self.assertNotIn("expected_target_digest", kwargs)
        self.assertNotIn("expected_content_version", kwargs)
        ids = kwargs.get("ids", [kwargs.get("pr_id")])
        return {
            "success": False, "requiresConfirmation": True,
            "confirmationKind": "deleteTargets", "batchId": "actor-draft",
            "contentVersion": 7, "targetDigest": "b" * 64,
            "targets": [copy.deepcopy(next(row for row in self.rows if row["id"] == value)) for value in ids],
            **copy.deepcopy(self.preview_override),
        }

    registered = {
        "keytao_lookup_by_words_batch", "keytao_list_draft_items",
        "keytao_create_phrase", "keytao_batch_add_to_draft",
        "keytao_remove_draft_item", "keytao_batch_remove_draft_items",
    }
    self.executor = ToolExecutor(
        lambda name: (lambda **kwargs: dispatch(name, **kwargs)) if name in registered else None,
        frozenset(registered - {"keytao_lookup_by_words_batch"}),
    )

async def GenericProposalBridgeTests_call(self, name="keytao_create_phrase", args=None, message=None, context=None):
    active = context or self.context
    if message is not None:
        active = replace(active, current_message=message)
    return json.loads(await self.executor.call(
        name, args if args is not None else {"word": "奈飞", "code": "nhfw"}, active,
    ))

def GenericProposalBridgeTests_assert_ticket(self, result, name):
    self.assertTrue(result.get("grammar_gap_bridged"), result)
    state = server_warning_pending_state(
        PendingToolConfirm(function_name=name, args=result["grammarGapArguments"]), result,
    )
    self.assertTrue(server_warning_ticket_is_complete(state))
    self.assertNotIn("preview_only", result["grammarGapArguments"])
    self.assertNotIn("confirmed", result["grammarGapArguments"])
    return state

# Source: test_s58_grammar.py
def MoveToCodeGrammarTests_context(self, message):
    return ToolContext(
        current_message=message,
        trusted_word_lookup_codes_by_word={
            "奈飞": frozenset({"nhfwv"}),
            "浓厚氛围": frozenset({"nhfw"}),
        },
        trusted_entries_by_code={"nhfw": (("浓厚氛围", 100),)},
    )

def MoveToCodeGrammarTests_assert_bound_move(self, message, occupant=""):
    move = grammar.parse_existing_entry_move(message)
    self.assertIsNotNone(move, message)
    self.assertEqual((move.word, move.target_code, move.named_occupant),
                     ("奈飞", "nhfw", occupant))
    self.assertTrue(grammar.message_authorizes_mutation(message), message)
    self.assertIsNone(grammar._validate_current_message_binding(
        "keytao_shift_phrase_code", {"word": "奈飞", "target_code": "nhfw"},
        self.context(message),
    ), message)

# Source: test_s59_commonness_evidence.py
from keytao_bot.utils.word_commonness import validate_commonness_words as validate_commonness_words
def CommonnessEvidenceTests_setUp(self):
    self.directory = tempfile.TemporaryDirectory(prefix="s59-commonness-")
    self.addCleanup(self.directory.cleanup)
    self.path = Path(self.directory.name) / "reference.db"
    with closing(sqlite3.connect(self.path)) as connection, connection:
        connection.execute(
            "CREATE TABLE word_commonness (word TEXT PRIMARY KEY, "
            "corpus_frequency INTEGER, part_of_speech TEXT, "
            "dictionary_presence_count INTEGER)"
        )
    self.path_patch = patch.object(review, "reference_db_path", return_value=self.path)
    self.path_patch.start()
    self.addCleanup(self.path_patch.stop)
    self.web = AsyncMock(side_effect=AssertionError("S59 must stay offline"))
    self.web_patch = patch.object(review, "_estimate_word_commonness_web_fallback", self.web)
    self.web_patch.start()
    self.addCleanup(self.web_patch.stop)

def CommonnessEvidenceTests_insert(self, *rows):
    with closing(sqlite3.connect(self.path)) as connection, connection:
        connection.executemany("INSERT INTO word_commonness VALUES (?, ?, ?, ?)", rows)

# Source: test_s59_commonness_route.py
from unittest.mock import Mock as Mock
s59_commonness_route_chat = harness.openai_chat_module

s59_commonness_route_commands = harness.chat_commands_module

s59_commonness_route_MESSAGE = "请给这几个词的词频排序：耶博 伊莎贝拉 夜泊 一身本领"

s59_commonness_route_WORDS = ("耶博", "伊莎贝拉", "夜泊", "一身本领")

def S59CommonnessRouteTests_setUp(self):
    from keytao_bot.utils import commonness_query as query
    from keytao_bot.utils import keytao_review as review
    self.query = query
    query.reset_commonness_delivery()
    self.addCleanup(query.reset_commonness_delivery)
    self.directory = tempfile.TemporaryDirectory(prefix="s59-route-")
    self.addCleanup(self.directory.cleanup)
    path = Path(self.directory.name) / "reference.db"
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("CREATE TABLE word_commonness (word TEXT PRIMARY KEY, corpus_frequency INTEGER, part_of_speech TEXT, dictionary_presence_count INTEGER)")
        connection.executemany("INSERT INTO word_commonness VALUES (?, ?, ?, ?)", [
            ("耶博", 100, "n", 2), ("伊莎贝拉", 1000, "nr", 3), ("夜泊", 400, "v", 3),
        ])
    patcher = patch.object(review, "reference_db_path", return_value=path)
    patcher.start()
    self.addCleanup(patcher.stop)
    self.owner = harness.ConversationAddress.private("qq", "s59-fixture")

async def S59CommonnessRouteTests_run_incident(self, *, shared=True, missing=False, lookup_failure=False):
    calls = []
    rows = [
        {"word": "耶博", "code": "yebo", "type": "Phrase", "weight": 100},
        {"word": "夜泊", "code": "yebo" if shared else "abcd", "type": "Phrase", "weight": 101},
        {"word": "伊莎贝拉", "code": "yebl", "type": "Phrase", "weight": 100},
    ]
    if missing:
        rows = [row for row in rows if row["word"] != "夜泊"]

    async def dispatch(name, args, *actor):
        calls.append((name, args))
        self.assertEqual(actor, ("qq", self.owner.actor_id))
        if name == "keytao_lookup_by_words_batch":
            self.assertEqual(args, {"words": list(s59_commonness_route_WORDS)})
            result = {"success": not lookup_failure, "results": [
                {"word": word, "phrases": [row for row in rows if row["word"] == word]}
                for word in s59_commonness_route_WORDS
            ]}
        elif name == "keytao_encode":
            self.assertEqual(set(args), {"word"})
            code = "yebo" if missing and args["word"] in ("夜泊", "耶博") else {
                "耶博": "yebo", "夜泊": "abcd", "伊莎贝拉": "yebl", "一身本领": "none",
            }[args["word"]]
            result = {"success": True, "word": args["word"], "codes": [code, code + "a"]}
        else:
            raise AssertionError("S59 dispatched a write or model tool: " + name)
        return json.dumps(result)

    context = SimpleNamespace(conversation_address=self.owner, platform="qq")
    ctx = SimpleNamespace(normalized_message_text=s59_commonness_route_MESSAGE, platform="qq", user_id=self.owner.actor_id,
                          conv_key=self.owner, memory_context=context, bot=object(), event=object(), QQMessageSegment=None)
    delivered = []

    async def finish(_bot, _event, _user, memory, response, _segment):
        delivered.append(s59_commonness_route_chat._prepare_user_facing_reply(response, memory))

    token = s59_commonness_route_chat._current_turn_message.set(s59_commonness_route_MESSAGE)
    try:
        with (patch.object(s59_commonness_route_commands, "call_tool_function", side_effect=dispatch),
              patch.object(s59_commonness_route_chat, "remember_conversation"),
              patch.object(s59_commonness_route_chat, "_finish_ai_chat_response", side_effect=finish),
              patch.object(s59_commonness_route_chat, "get_ai_response_core", side_effect=AssertionError("model is forbidden")),
              patch.object(s59_commonness_route_chat, "_classify_message_command_intent", side_effect=AssertionError("classifier is forbidden"))):
            self.assertTrue(await s59_commonness_route_chat._stage_handle_commonness_query(ctx))
    finally:
        s59_commonness_route_chat._current_turn_message.reset(token)
    self.assertEqual(len(delivered), 1)
    return delivered[0], calls

# Source: test_s60_delta.py
from unittest.mock import MagicMock as MagicMock
def s60_delta_row(item_id, word, code, action="Create", weight=10):
    return {"id": item_id, "word": word, "code": code, "type": "Phrase",
            "action": action, "oldWord": None, "weight": weight,
            "remark": None, "needsManualReview": False}

OLD = s60_delta_row(11, "旧词", "jqck")

CREATES = [s60_delta_row(12, "吊杠", "dcgp"), s60_delta_row(13, "掉岗", "dcgp", "Delete"),
           s60_delta_row(14, "掉岗", "dcgpi"), s60_delta_row(15, "沃集鲜", "wjxa")]

def snapshot(items, version=1):
    return {"batchId": "s60-delta", "contentVersion": version,
            "batchUrl": "https://example.invalid/batch/s60-delta", "items": copy.deepcopy(items)}

def ReceiptDeltaTests_setUp(self):
    self.capture = completed_draft_undo.OperationCapture("s60-scope", "qq", "fixture-actor", "s60-turn", "")
    self.token = completed_draft_undo.current_operation_capture.set(self.capture)
    self.store = MagicMock()
    self.store.actor_has_running_undo.return_value = False
    self.store_patch = patch.object(completed_draft_undo, "get_default_draft_mutation_claim_store", return_value=self.store)
    self.store_patch.start()
    self.listed = AsyncMock(side_effect=AssertionError("unexpected real snapshot read"))
    self.get_tool = MagicMock(return_value=self.listed)

def ReceiptDeltaTests_tearDown(self):
    self.store_patch.stop()
    completed_draft_undo.current_operation_capture.reset(self.token)

async def ReceiptDeltaTests_invoke(self, before, after, *, name="keytao_batch_add_to_draft", args=None, result=None):
    args = args if args is not None else {"confirmed": True, "items": copy.deepcopy(CREATES)}
    result = result if result is not None else {
        "success": True, "batchId": "s60-delta",
        "contentVersion": after["contentVersion"] if isinstance(after, dict) else 2,
    }
    tool = AsyncMock(return_value=result)
    with patch.object(completed_draft_undo, "_read_snapshot", AsyncMock(side_effect=[before, after])) as read:
        returned = await completed_draft_undo.invoke_with_operation_journal(name, args, tool, self.get_tool)
    return returned, read, tool

# Source: test_s60_finalizers.py
from keytao_bot.utils.draft_receipts import merge_receipt_deltas as merge_receipt_deltas
from keytao_bot.utils.draft_receipts import receipt_change_lines as receipt_change_lines
from keytao_bot.utils.draft_receipts import receipt_changes as receipt_changes
render = harness.openai_chat_module._chat_render

s60_finalizers_BATCH_ID = "s60-finalizer-batch"

def pr(identity, word, code, action="Create", **extra):
    return {"id": identity, "action": action, "word": word, "code": code,
            "type": "Phrase", **extra}

def mixed_items():
    return [pr(1, "吊杠", "dcgp"), pr(2, "掉岗", "dcgp", "Delete"),
            pr(3, "掉岗", "dcgpi"), pr(4, "沃集鲜", "wjxa")]

def ReceiptFinalizerTests_assert_exact_created_coverage(self, data, requested_words, expected_word_count):
    original = copy.deepcopy(data)
    changes = receipt_changes(data, requested_words)
    expected = {item["id"] for item in data["writtenItems"]}
    actual = [identity for change in changes for identity in change.created_ids]
    self.assertEqual(set(actual), expected)
    self.assertEqual(len(actual), len(expected), "a created PR is omitted or counted twice")
    self.assertEqual(len(changes), expected_word_count)
    lines = receipt_change_lines(data, requested_words)
    for change in changes:
        self.assertEqual(sum(line.count(change.text) for line in lines), 1)
    self.assertEqual(data, original, "rendering mutated the actual receipt")
    return changes

# Source: test_s60_receipt.py
s60_receipt_commands = harness.chat_commands_module

s60_receipt_chat = harness.openai_chat_module

def mixed_receipt():
    items = [
        {"id": "pr-1", "action": "Create", "word": "吊杠", "code": "dcgp", "type": "Phrase"},
        {"id": "pr-2", "action": "Delete", "word": "掉岗", "code": "dcgp", "type": "Phrase"},
        {"id": "pr-3", "action": "Create", "word": "掉岗", "code": "dcgpi", "type": "Phrase"},
        {"id": "pr-4", "action": "Create", "word": "沃集鲜", "code": "wjxa", "type": "Phrase"},
    ]
    return {
        "success": True, "batchId": "e4d21c4d", "pullRequestCount": 4,
        "writtenItems": items,
        "shiftPlan": {
            "word": "吊杠", "targetCode": "dcgp", "items": copy.deepcopy(items),
            "shifted": [{"word": "掉岗", "fromCode": "dcgp", "toCode": "dcgpi"}],
        },
    }

# Source: test_s60_scenario.py
s60_scenario_chat = harness.openai_chat_module

s60_scenario_commands = harness.chat_commands_module

s60_scenario_BATCH_ID = "s60-e4d21c4d"

PR_ITEMS = [
    {"id": 6001, "action": "Create", "word": "吊杠", "code": "dcgp", "type": "Phrase", "weight": 100},
    {"id": 6002, "action": "Delete", "word": "掉岗", "code": "dcgp", "type": "Phrase", "weight": 100},
    {"id": 6003, "action": "Create", "word": "掉岗", "code": "dcgpi", "type": "Phrase", "weight": 100},
    {"id": 6004, "action": "Create", "word": "沃集鲜", "code": "wjxa", "type": "Phrase", "weight": 100},
]

def s60_scenario_review_payload(word):
    """Use the public reviewed-add schema, without forging a pending record."""
    code, reading = {"吊杠": ("dcgp", "diào gàng"), "沃集鲜": ("wjxa", "wò jí xiān")}[word]
    statuses = [{
        "code": code, "occupied": word == "吊杠",
        "words": ["掉岗"] if word == "吊杠" else [],
        "phrases": [{"word": "掉岗", "code": code, "type": "Phrase", "weight": 100}]
        if word == "吊杠" else [],
    }]
    if word == "吊杠":
        statuses.append({"code": "dcgpi", "occupied": False, "words": [], "phrases": []})
    return {
        "success": True, "word": word, "type": "Phrase", "recommendedCode": code,
        "autoReviewable": False, "needsManualReview": True,
        "manualReviewReason": "Fixture requires manual review",
        "pronunciations": [{
            "pinyin": reading, "recommendedCode": code,
            "sources": [{"source": "汉典"}], "candidateStatuses": statuses,
        }],
        "candidateOrderingAssessments": [{
            "newWord": word, "occupantWord": "掉岗", "occupantCode": "dcgp",
            "freeCode": "dcgpi", "newCode": "dcgp", "recommendedCode": "dcgp",
            "verdict": "front_more_common", "summary": "Fixture comparison evidence",
        }] if word == "吊杠" else [],
    }

async def S60ScenarioTests_replay(self, platform, words=("吊杠", "沃集鲜")):
    actor = "s60-fixture-actor"
    store = harness.MemoryConversationStateStore()
    key = harness.ConversationAddress.group(platform, "s60-fixture-group", actor)
    calls = []
    created = []
    submitted = False
    plan = {
        "word": "吊杠", "targetCode": "dcgp",
        "shifted": [{"word": "掉岗", "fromCode": "dcgp", "toCode": "dcgpi"}],
        "items": [{key: value for key, value in row.items() if key != "id"} for row in PR_ITEMS],
    }

    async def tool(name, args, request_platform, request_actor, **_kwargs):
        nonlocal submitted
        self.assertEqual((request_platform, request_actor), (platform, actor))
        calls.append((name, copy.deepcopy(args)))
        if name == "keytao_lookup_by_word":
            result = {"success": True, "phrases": []}
        elif name == "keytao_pending_items_by_words":
            result = {"success": True, "complete": True, "items": []}
        elif name == "keytao_prepare_reviewed_add":
            result = s60_scenario_review_payload(args["word"])
        elif name == "keytao_shift_phrase_code":
            self.assertEqual((args["word"], args["target_code"]), ("吊杠", "dcgp"))
            self.assertEqual([(row["word"], row["code"]) for row in args["additional_items"]],
                             [("沃集鲜", "wjxa")])
            if not args.get("confirmed_plan_digest"):
                self.assertFalse(created)
                result = {
                    "success": False, "requiresConfirmation": True,
                    "confirmationKind": "shiftPlan", "batchId": "", "contentVersion": 0,
                    "planDigest": "a" * 64, "warningDigest": "b" * 64,
                    "warnings": [], "shiftPlan": plan,
                }
            else:
                self.assertEqual(args["confirmed_plan_digest"], "a" * 64)
                self.assertEqual(args["expected_content_version"], 0)
                self.assertFalse(created, "a confirmed write was replayed twice")
                created.extend(copy.deepcopy(PR_ITEMS))
                result = {
                    "success": True, "batchId": s60_scenario_BATCH_ID, "contentVersion": 1,
                    "pullRequestCount": len(created), "writtenItems": copy.deepcopy(created),
                    "shiftPlan": plan,
                }
        elif name == "keytao_submit_batch":
            self.assertEqual(args["batch_id"], s60_scenario_BATCH_ID)
            self.assertEqual(created, PR_ITEMS)
            if args.get("preview_only"):
                result = {
                    "success": False, "requiresConfirmation": True,
                    "batchId": s60_scenario_BATCH_ID, "contentVersion": 1,
                    "snapshotDigest": "c" * 64, "warningDigest": "d" * 64,
                    "auditDigest": "e" * 64, "snapshotItems": copy.deepcopy(created),
                    "warnings": [],
                }
            else:
                self.assertTrue(args.get("confirmed"))
                self.assertEqual(args["expected_content_version"], 1)
                self.assertEqual(args["expected_server_snapshot_digest"], "c" * 64)
                self.assertEqual(args["expected_warning_digest"], "d" * 64)
                self.assertEqual(args["expected_audit_digest"], "e" * 64)
                self.assertFalse(submitted)
                submitted = True
                result = {"success": True, "status": "Submitted", "batchId": s60_scenario_BATCH_ID,
                          "contentVersion": 2}
        else:
            raise AssertionError(f"Unexpected external tool: {name}")
        return json.dumps(result, ensure_ascii=False)

    render = s60_scenario_commands.render_server_backed_batch_candidates

    def record_first_render(*args, **kwargs):
        record = store.get_record(key)
        self.assertIsNotNone(record, "candidate reply was rendered before persistence")
        self.assertEqual(record.state.args["_query_words"], list(words))
        return render(*args, **kwargs)

    with (
        patch.object(s60_scenario_chat, "conversation_state_store", store),
        patch.object(s60_scenario_chat, "call_tool_function", side_effect=tool),
        patch.object(s60_scenario_chat, "_classify_simple_word_query_intent",
                     AsyncMock(side_effect=AssertionError("S60 cannot invoke a model"))),
        patch.object(s60_scenario_chat, "_classify_message_command_intent",
                     AsyncMock(side_effect=AssertionError("S60 cannot invoke a model"))),
        patch.object(s60_scenario_commands, "render_server_backed_batch_candidates", side_effect=record_first_render),
    ):
        discovery = await s60_scenario_commands._try_handle_simple_single_word_query(
            " ".join(words), platform, actor, key, key.space_key, "S60 fixture",
        )
        self.assertEqual(discovery.count("审词："), 2)
        self.assertIn("占 dcgp", discovery)
        self.assertIn("掉岗」顺延", discovery)
        self.assertIn("wjxa", discovery)
        self.assertTrue(s60_scenario_chat._advertised_reply_matches_live_record(discovery, store.get_record(key)))
        self.assertEqual(s60_scenario_chat._enforce_advertised_reply_contract(discovery, key), discovery)
        self.assertFalse(created, "the discovery turn wrote draft rows")
        self.assertEqual([args["word"] for name, args in calls if name == "keytao_prepare_reviewed_add"],
                         list(words))
        reply = await s60_scenario_commands.handle_pending_message_core(
            "加入并提交", platform, actor, key,
            space_key=key.space_key, owner_label="S60 fixture",
            allow_intent_model=False,
        )
        self.assertTrue(submitted, reply)
        self.assertEqual(created, PR_ITEMS)
        self.assertIsNone(store.get_record(key))
        context = SimpleNamespace(response=reply, conv_key=key)
        await s60_scenario_chat._stage_normalize_response(context)
        delivered = s60_scenario_chat._enforce_advertised_reply_contract(context.response, key)

    self.assertEqual([name for name, _args in calls if name in {
        "keytao_shift_phrase_code", "keytao_submit_batch",
    }], ["keytao_shift_phrase_code", "keytao_shift_phrase_code",
         "keytao_submit_batch", "keytao_submit_batch"])
    self.assertIn("吊杠 → dcgp", delivered)
    self.assertIn("掉岗 dcgp→dcgpi", delivered)
    self.assertIn("沃集鲜 → wjxa", delivered)
    self.assertIn("已提交审核", delivered)
    host = "https://keytao.rea.ink" if platform == "qq" else "https://keytao.vercel.app"
    self.assertEqual(delivered.count(f"{host}/batch/{s60_scenario_BATCH_ID}"), 1, delivered)
    self.assertLess(delivered.index(words[0]), delivered.index(words[1]), delivered)
    print(json.dumps({"scenario": "S60", "mode": "fixture/fake-model", "platform": platform,
                      "query": " ".join(words), "confirmation": "加入并提交",
                      "createdPullRequestIds": [row["id"] for row in created],
                      "paidModelCalls": 0, "realProviderCalls": 0,
                      "receipt": delivered}, ensure_ascii=False), flush=True)

# Source: test_s61_submit_replay.py
import contextlib as contextlib
s61_submit_replay_chat = harness.openai_chat_module

s61_submit_replay_commands = harness.chat_commands_module

s61_submit_replay_BATCH_ID = "s61-3fc424e6"

s61_submit_replay_WORD = "户晨风"

CODE = "hjfo"

OTHER = {"action": "Create", "word": "旧词", "code": "qtxx", "type": "Phrase", "weight": 100}

class FakeKeytaoServer:
    """One draft batch with content-version CAS, previews and an all-or-nothing submit."""

    def __init__(self, existing_items=()):
        self.items = [dict(item) for item in existing_items]
        self.version = 3 if self.items else 0
        self.created = []
        self.submitted = False
        self.calls = []

    async def __call__(self, name, args, platform, user_id, **_kwargs):
        self.calls.append((name, copy.deepcopy(args)))
        handler = getattr(self, f"_{name}", None)
        if handler is None:
            raise AssertionError(f"Unexpected external tool: {name}")
        return json.dumps(handler(args), ensure_ascii=False)

    def _keytao_create_phrase(self, args):
        if not args.get("confirmed"):
            return {
                "success": False, "requiresConfirmation": True, "batchId": s61_submit_replay_BATCH_ID,
                "contentVersion": self.version, "warningDigest": "a" * 64,
                "warnings": [], "warnedCount": 0,
            }
        assert args["batch_id"] == s61_submit_replay_BATCH_ID, args
        assert args["expected_content_version"] == self.version, args
        assert args["expected_warning_digest"] == "a" * 64, args
        assert not self.created, "a confirmed create was replayed twice"
        self.version += 1
        row = {
            "id": 7001, "action": "Create", "word": args["word"],
            "code": args["code"], "type": "Phrase", "weight": 100,
        }
        self.created.append(row)
        self.items.append({key: value for key, value in row.items() if key != "id"})
        return {
            "success": True, "batchId": s61_submit_replay_BATCH_ID, "contentVersion": self.version,
            "pullRequestCount": len(self.created),
            "writtenItems": copy.deepcopy(self.created),
        }

    def _keytao_submit_batch(self, args):
        if not args.get("confirmed"):
            return {
                "success": False, "requiresConfirmation": True, "batchId": s61_submit_replay_BATCH_ID,
                "contentVersion": self.version, "snapshotDigest": "c" * 64,
                "warningDigest": "d" * 64, "auditDigest": "e" * 64,
                "snapshotItems": copy.deepcopy(self.items), "warnings": [],
            }
        assert args["batch_id"] == s61_submit_replay_BATCH_ID, args
        assert args["expected_content_version"] == self.version, args
        assert args["expected_server_snapshot_digest"] == "c" * 64, args
        assert args["expected_warning_digest"] == "d" * 64, args
        assert args["expected_audit_digest"] == "e" * 64, args
        assert not self.submitted, "the batch was submitted twice"
        self.submitted = True
        self.version += 1
        return {
            "success": True, "status": "Submitted", "batchId": s61_submit_replay_BATCH_ID,
            "contentVersion": self.version,
        }

    def tool_names(self):
        return [name for name, _args in self.calls]

class FakeShiftServer(FakeKeytaoServer):
    """The same batch reached through a confirmed ranked-shift plan."""

    PLAN_ITEMS = [{"action": "Create", "word": s61_submit_replay_WORD, "code": CODE, "type": "Phrase"}]

    def _keytao_shift_phrase_code(self, args):
        assert args["confirmed_plan_digest"] == "b" * 64, args
        assert not self.created, "a confirmed shift was replayed twice"
        self.version += 1
        row = {
            "id": 7101, "action": "Create", "word": s61_submit_replay_WORD,
            "code": CODE, "type": "Phrase", "weight": 100,
        }
        self.created.append(row)
        self.items.append({key: value for key, value in row.items() if key != "id"})
        return {
            "success": True, "batchId": s61_submit_replay_BATCH_ID, "contentVersion": self.version,
            "pullRequestCount": len(self.created),
            "writtenItems": copy.deepcopy(self.created),
            "shiftPlan": {
                "word": s61_submit_replay_WORD, "targetCode": CODE, "shifted": [],
                "items": copy.deepcopy(self.PLAN_ITEMS),
            },
        }

class s61_submit_replay_FakeBot:
    def __init__(self):
        self.messages = []

    async def send(self, **kwargs):
        self.messages.append(kwargs.get("message"))

def S61SubmitReplayTests_setUp(self):
    self.actor = "s61-fixture-actor"
    self.platform = "qq"
    self.key = harness.ConversationAddress.group("qq", "865189947", self.actor)
    self.store = harness.MemoryConversationStateStore()
    self.coordinator = harness.DraftOperationCoordinator()
    self.memory_context = harness.ChatMemoryContext(
        platform="qq", user_id=self.actor, space_type="group",
        space_id="865189947", speaker_name="Rea",
    )

@contextlib.contextmanager
def S61SubmitReplayTests_patches(self, server):
    async def tool(name, args, platform, user_id, **kwargs):
        # Mirror the real dispatcher: every tool response is also the
        # trusted receipt the delivery boundary rebuilds its answer from.
        result_json = await server(name, args, platform, user_id, **kwargs)
        s61_submit_replay_commands._capture_successful_draft_write_delivery(
            name, json.loads(result_json), platform, user_id,
        )
        return result_json

    patchers = (
        patch.object(s61_submit_replay_chat, "conversation_state_store", self.store),
        patch.object(s61_submit_replay_chat, "draft_operation_coordinator", self.coordinator),
        patch.object(s61_submit_replay_chat, "call_tool_function", side_effect=tool),
        patch.object(s61_submit_replay_chat, "remember_conversation", return_value=True),
        patch.object(s61_submit_replay_chat, "schedule_memory_compaction"),
        patch.object(s61_submit_replay_chat.memory_store, "capture_generation", return_value=object()),
        patch.object(s61_submit_replay_chat.memory_store, "is_generation_current", return_value=True),
        patch.object(s61_submit_replay_chat.memory_store, "record_tool_receipt"),
        patch.object(s61_submit_replay_chat.history_store, "capture_generation", return_value=object()),
        patch.object(s61_submit_replay_chat.history_store, "is_generation_current", return_value=True),
        patch.object(s61_submit_replay_chat, "_classify_message_command_intent",
                     AsyncMock(side_effect=AssertionError("S61 cannot invoke a model"))),
        patch.object(s61_submit_replay_chat, "_classify_simple_word_query_intent",
                     AsyncMock(side_effect=AssertionError("S61 cannot invoke a model"))),
    )
    with contextlib.ExitStack() as stack:
        for patcher in patchers:
            stack.enter_context(patcher)
        yield

async def S61SubmitReplayTests_run_add_and_submit(self, server, platform="qq"):
    """Replay the production 「加入并提交」 background operation end to end."""
    self.platform = platform
    self.key = harness.ConversationAddress.group(platform, "865189947", self.actor)
    self.memory_context = harness.ChatMemoryContext(
        platform=platform, user_id=self.actor, space_type="group",
        space_id="865189947", speaker_name="Rea",
    )
    bot = s61_submit_replay_FakeBot()
    token = s61_submit_replay_chat._current_turn_message.set("加入并提交")
    try:
        with self.patches(server):
            operation = self.coordinator.begin(
                self.key, "add_and_submit", word=s61_submit_replay_WORD, code=CODE,
            )
            await s61_submit_replay_chat._run_background_draft_operation(
                operation,
                lambda: s61_submit_replay_commands._perform_add_to_draft_and_submit(
                    s61_submit_replay_WORD, CODE, platform, self.actor,
                    needs_manual_review=True, auto_confirm=True,
                ),
                bot, SimpleNamespace(message_id=None), self.actor,
                self.memory_context, "加入并提交", object(), object(),
            )
    finally:
        s61_submit_replay_chat._current_turn_message.reset(token)
    self.assertEqual(len(bot.messages), 1, bot.messages)
    print(json.dumps({
        "scenario": "S61", "mode": "fixture/fake-tools", "platform": platform,
        "instruction": "加入并提交", "batchHeldOtherItems": len(server.items) > 1,
        "submitted": server.submitted, "toolCalls": server.tool_names(),
        "paidModelCalls": 0, "realProviderCalls": 0, "receipt": bot.messages[0],
    }, ensure_ascii=False), flush=True)
    return bot.messages[0]

async def S61SubmitReplayTests_run_shift_and_submit(self, server):
    """Replay the shift-then-submit path that shares the same submit stage."""
    state = harness.PendingToolConfirm(
        function_name="keytao_shift_phrase_code",
        args={
            "word": s61_submit_replay_WORD, "target_code": CODE, "batch_id": "",
            "expected_content_version": 0, "confirmed_plan_digest": "b" * 64,
            "_submit_after": True,
        },
        confirmation_source="server_warning",
    )
    token = s61_submit_replay_chat._current_turn_message.set("加入并提交")
    try:
        with self.patches(server):
            return await s61_submit_replay_commands._execute_confirmed_tool(
                state, self.platform, self.actor, self.key,
                self.key.space_key, "Rea",
            )
    finally:
        s61_submit_replay_chat._current_turn_message.reset(token)

# Source: test_s62_selection.py
def s62_selection_incident_record():
    """Adapt the existing sealed review fixture to the reported word and code."""
    record = reviewed_record()
    serialized = json.dumps(record.args, ensure_ascii=False)
    for old, new in (("大端", "加亮"), ("dsdtvo", "jslxa"),
                     ("dsdtv", "jslxo"), ("dsdt", "jslx")):
        serialized = serialized.replace(old, new)
    record.args = json.loads(serialized)
    return record

async def S62SelectionTests_execute_selection(self, message, record=None):
    store = harness.MemoryConversationStateStore()
    key = harness.ConversationAddress.private("qq", "s62-selection")
    store.set(key, record or s62_selection_incident_record())
    writes = []
    calls = []

    async def tool(name, args, platform, user_id, **_kwargs):
        self.assertEqual((platform, user_id), ("qq", "s62-selection"))
        if name in {"keytao_get_batch_preview", "keytao_list_draft_items"}:
            return json.dumps({
                "success": True, "batchId": "s62-selection-batch",
                "status": "Draft", "count": len(writes), "items": writes,
            })
        self.assertEqual(name, "keytao_batch_add_to_draft")
        calls.append(copy.deepcopy(args))
        if not args.get("confirmed"):
            self.assertTrue(args.get("preview_only"))
            return json.dumps({
                "success": False, "requiresConfirmation": True,
                "batchId": "s62-selection-batch", "contentVersion": 0,
                "warningDigest": "a" * 64, "warnings": [],
            })
        self.assertEqual(args["batch_id"], "s62-selection-batch")
        self.assertEqual(args["expected_content_version"], 0)
        self.assertEqual(args["expected_warning_digest"], "a" * 64)
        self.assertFalse(writes, "A selection must not replay a confirmed write")
        writes.extend(copy.deepcopy(args["items"]))
        return json.dumps({
            "success": True, "batchId": "s62-selection-batch",
            "contentVersion": 1, "successCount": len(writes),
            "failedCount": 0, "pullRequestCount": len(writes),
            "writtenItems": [{"id": 6200 + index, **item}
                             for index, item in enumerate(writes)],
        })

    with patch.object(harness.openai_chat_module, "conversation_state_store", store), patch.object(
        harness.openai_chat_module, "call_tool_function", side_effect=tool,
    ), patch.object(
        harness.openai_chat_module, "_classify_message_command_intent",
        AsyncMock(side_effect=AssertionError("S62 must not invoke a model")),
    ):
        response = await b17_rest_recommended_commands.handle_pending_message_core(
            message, "qq", "s62-selection", key, allow_intent_model=False,
        )
    return writes, calls, response, store.get(key)

# Source: test_s63_fresh_selection.py
from contextlib import contextmanager as contextmanager
s63_fresh_selection_chat = harness.openai_chat_module

BATCH_URL = "https://keytao.rea.ink/batch/s63-fresh"

def FreshSelectionTests_setUp(self):
    self.now = 100.0
    self.store = harness.MemoryConversationStateStore(
        pending_ttl_seconds=10, clock=lambda: self.now,
    )
    self.key = harness.ConversationAddress.private("qq", "s63-fresh")
    self.memory = harness.ChatMemoryContext(platform="qq", user_id=self.key.actor_id)
    self.calls = []
    self.writes = []
    self.create_warnings = []
    self.review = {
        "success": True, "word": "加亮", "type": "Phrase",
        "recommendedCode": "jslxao", "needsManualReview": True,
        "manualReviewReason": "常用度证据不足",
        "pronunciations": [{
            "pinyin": "jiā liàng", "recommendedCode": "jslxao",
            "candidateStatuses": [
                {"code": code, "occupied": False, "words": []}
                for code in ("jslx", "jslxa", "jslxao")
            ],
        }],
    }
    self.lookup = {"success": True, "phrases": []}
    self.pending_complete = True
    self.pending_items = [{
        "source": "draft", "batchId": "s63-fresh", "batchUrl": BATCH_URL,
        "batchStatus": "Draft", "itemStatus": "Pending", "action": "Create",
        "word": "加亮", "code": "jslxa", "type": "Phrase",
    }]

async def FreshSelectionTests_tool(self, name, args, platform, user_id, **kwargs):
    self.assertEqual((platform, user_id), ("qq", self.key.actor_id))
    self.calls.append((name, copy.deepcopy(args)))
    if name == "keytao_pending_items_by_words":
        return json.dumps({"success": True, "complete": self.pending_complete, "items": self.pending_items})
    if name == "keytao_prepare_reviewed_add":
        return json.dumps(self.review)
    if name == "keytao_lookup_by_code":
        return json.dumps(self.lookup)
    if name == "keytao_batch_add_to_draft":
        self.assertEqual([(item["word"], item["code"]) for item in args["items"]], [("加亮", "jslxa")])
        if not args.get("confirmed"):
            self.assertTrue(args["preview_only"])
            return json.dumps({
                "success": False, "requiresConfirmation": True,
                "batchId": "s63-fresh", "contentVersion": 0,
                "warningDigest": "a" * 64, "warnings": [],
            })
        self.assertEqual(args["expected_content_version"], 0)
        self.assertEqual(args["expected_warning_digest"], "a" * 64)
        self.assertFalse(self.writes, "Confirmed batch write replayed")
        self.writes.extend(copy.deepcopy(args["items"]))
        return json.dumps({
            "success": True, "batchId": "s63-fresh", "contentVersion": 1,
            "batchUrl": BATCH_URL, "writtenItems": self.writes,
            "successCount": len(self.writes), "failedCount": 0,
        })
    if name == "keytao_create_phrase":
        self.assertEqual((args["word"], args["code"]), ("加亮", "jslxa"))
        self.assertTrue(args["needs_manual_review"])
        self.assertIn("jiā liàng", args["remark"])
        capability = kwargs["trusted_reviewed_items_by_key"][("加亮", "jslxa")]
        self.assertEqual(capability["pinyin"], "jiā liàng")
        if not args.get("confirmed"):
            self.assertTrue(args["preview_only"])
            return json.dumps({
                "success": False, "requiresConfirmation": True,
                "batchId": "s63-fresh", "contentVersion": 0,
                "warningDigest": "a" * 64, "warnings": self.create_warnings,
            })
        self.assertEqual(args["expected_content_version"], 0)
        self.assertEqual(args["expected_warning_digest"], "a" * 64)
        self.assertFalse(self.writes, "Confirmed write replayed")
        self.writes.append({"word": args["word"], "code": args["code"], "type": "Phrase"})
        return json.dumps({
            "success": True, "batchId": "s63-fresh", "contentVersion": 1,
            "batchUrl": BATCH_URL, "writtenItems": self.writes,
        })
    if name in {"keytao_get_batch_preview", "keytao_list_draft_items"}:
        return json.dumps({
            "success": True, "batchId": "s63-fresh", "batchUrl": BATCH_URL,
            "contentVersion": 1, "status": "Draft", "items": self.writes,
            "count": len(self.writes),
        })
    raise AssertionError("Unexpected external tool: " + name)

@contextmanager
def FreshSelectionTests_runtime(self):
    with ExitStack() as stack:
        for patcher in (
            patch.object(s63_fresh_selection_chat, "conversation_state_store", self.store),
            patch.object(s63_fresh_selection_chat, "draft_operation_coordinator", harness.DraftOperationCoordinator()),
            patch.object(s63_fresh_selection_chat, "call_tool_function", side_effect=self.tool),
            patch.object(s63_fresh_selection_chat, "remember_conversation"),
            patch.object(s63_fresh_selection_chat, "get_history", return_value=[]),
            patch.object(s63_fresh_selection_chat, "_finish_ai_chat_response", AsyncMock()),
            patch.object(s63_fresh_selection_chat, "_finish_ai_chat_matcher", AsyncMock()),
            patch.object(s63_fresh_selection_chat, "get_ai_response_core", AsyncMock(side_effect=AssertionError("General model reached"))),
            patch.object(s63_fresh_selection_chat, "_classify_message_command_intent", AsyncMock(side_effect=AssertionError("Intent model reached"))),
            patch.object(s63_fresh_selection_chat, "_classify_simple_word_query_intent", AsyncMock(side_effect=AssertionError("Word model reached"))),
        ):
            stack.enter_context(patcher)
        token = b17_rest_recommended_commands.current_memory_context.set(self.memory)
        try:
            yield
        finally:
            b17_rest_recommended_commands.current_memory_context.reset(token)

async def FreshSelectionTests_turn(self, message, *, reply_reference=None):
    cache, classifier = s63_fresh_selection_chat.command_intent_memoizer(message, message)
    ctx = s63_fresh_selection_chat.TurnContext(
        bot=object(), event=object(), platform="qq", user_id=self.key.actor_id,
        message_text=message, normalized_message_text=message,
        conv_key=self.key, space_key=self.key.space_key,
        memory_context=self.memory, history=[],
        reply_reference=reply_reference or harness.ReplyReferenceInfo(),
        command_intent_cache=cache, command_intent_for=classifier,
    )
    await s63_fresh_selection_chat._stage_normalize_message_text(ctx)
    ctx.command_intent_cache, ctx.command_intent_for = s63_fresh_selection_chat.command_intent_memoizer(
        message, ctx.normalized_message_text,
    )
    start = s63_fresh_selection_chat.STAGES.index(s63_fresh_selection_chat._stage_resolve_current_pending_scope)
    end = s63_fresh_selection_chat.STAGES.index(s63_fresh_selection_chat._stage_generate_ai_response)
    for stage in s63_fresh_selection_chat.STAGES[start:end + 1]:
        if await stage(ctx):
            break
    return ctx.response

# Source: test_s64_explicit_submit.py
INSTRUCTION = "加亮 jslxa，加入草稿并提交。"

REAL_COMMAND_CLASSIFIER = s63_fresh_selection_chat._classify_message_command_intent

REAL_WORD_CLASSIFIER = s63_fresh_selection_chat._classify_simple_word_query_intent

REAL_TOOL_DISPATCH = b17_rest_recommended_commands.call_tool_function

@contextmanager
def ExplicitSubmitTests_runtime(self):
    client = Mock(side_effect=AssertionError("An explicit word/code command must not call a model"))
    with FreshSelectionTests_runtime(self), patch.object(
        s63_fresh_selection_chat, "_classify_message_command_intent", REAL_COMMAND_CLASSIFIER,
    ), patch.object(
        s63_fresh_selection_chat, "_classify_simple_word_query_intent", REAL_WORD_CLASSIFIER,
    ), patch.object(s63_fresh_selection_chat._chat_routing, "AsyncOpenAI", client), patch.object(
        s63_fresh_selection_chat._chat_routing, "OPENAI_API_KEY", "s64-fixture-key",
    ):
        try:
            yield
        finally:
            client.assert_not_called()

def ExplicitSubmitTests_setUp(self):
    FreshSelectionTests_setUp(self)
    self.pending_items = []
    self.batch_id = "s64-draft"
    self.batch_url = "https://keytao.rea.ink/batch/" + self.batch_id
    self.items = []
    self.version = 0
    self.submitted = False

async def ExplicitSubmitTests_turn(self, message, **kwargs):
    token = s63_fresh_selection_chat._current_turn_message.set(message)
    try:
        return await FreshSelectionTests_turn(self, message, **kwargs)
    finally:
        s63_fresh_selection_chat._current_turn_message.reset(token)

async def ExplicitSubmitTests_tool(self, name, args, platform, user_id, **kwargs):
    self.assertEqual((platform, user_id), ("qq", self.key.actor_id))
    self.calls.append((name, copy.deepcopy(args)))
    if name == "keytao_pending_items_by_words":
        result = {"success": True, "complete": self.pending_complete, "items": self.pending_items}
    elif name == "keytao_prepare_reviewed_add":
        self.assertEqual(args["word"], "加亮")
        result = self.review
    elif name == "keytao_lookup_by_code":
        self.assertEqual(args["code"], "jslxa")
        result = self.lookup
    elif name == "keytao_create_phrase":
        self.assertEqual((args["word"], args["code"]), ("加亮", "jslxa"))
        self.assertTrue(args["needs_manual_review"])
        self.assertIn("jiā liàng", args["remark"])
        capability = kwargs["trusted_reviewed_items_by_key"][("加亮", "jslxa")]
        self.assertEqual(capability["pinyin"], "jiā liàng")
        if not args.get("confirmed"):
            self.assertTrue(args["preview_only"])
            result = {
                "success": False, "requiresConfirmation": True,
                "batchId": self.batch_id, "batchUrl": self.batch_url,
                "contentVersion": self.version, "warningDigest": "a" * 64,
                "warnings": [], "warnedCount": 0,
            }
        else:
            self.assertEqual(args["batch_id"], self.batch_id)
            self.assertEqual(args["expected_content_version"], self.version)
            self.assertEqual(args["expected_warning_digest"], "a" * 64)
            self.assertFalse(self.writes, "The confirmed create must not replay")
            row = {"id": 6401, "action": "Create", "word": "加亮", "code": "jslxa", "type": "Phrase"}
            self.items.append(row)
            self.writes.append(row)
            self.version += 1
            result = {
                "success": True, "batchId": self.batch_id, "batchUrl": self.batch_url,
                "contentVersion": self.version, "writtenItems": copy.deepcopy(self.writes),
                "pullRequestCount": 1,
            }
    elif name in {"keytao_list_draft_items", "keytao_get_batch_preview"}:
        result = {
            "success": True, "batchId": self.batch_id, "batchUrl": self.batch_url,
            "contentVersion": self.version, "status": "Draft",
            "items": copy.deepcopy(self.items), "count": len(self.items),
        }
    elif name == "keytao_submit_batch":
        self.assertEqual(args.get("batch_id"), self.batch_id)
        if not args.get("confirmed"):
            result = {
                "success": False, "requiresConfirmation": True,
                "batchId": self.batch_id, "batchUrl": self.batch_url,
                "contentVersion": self.version, "snapshotDigest": "c" * 64,
                "warningDigest": "d" * 64, "auditDigest": "e" * 64,
                "snapshotItems": copy.deepcopy(self.items), "warnings": [],
            }
        else:
            self.assertEqual(args["expected_content_version"], self.version)
            self.assertEqual(args["expected_server_snapshot_digest"], "c" * 64)
            self.assertEqual(args["expected_warning_digest"], "d" * 64)
            self.assertEqual(args["expected_audit_digest"], "e" * 64)
            self.assertFalse(self.submitted, "The batch must not submit twice")
            self.submitted = True
            self.version += 1
            result = {
                "success": True, "status": "Submitted", "batchId": self.batch_id,
                "batchUrl": self.batch_url, "contentVersion": self.version,
            }
    else:
        raise AssertionError("Unexpected external tool: " + name)
    if kwargs.get("_capture_receipt", True):
        b17_rest_recommended_commands._capture_successful_draft_write_delivery(name, result, platform, user_id)
    return json.dumps(result, ensure_ascii=False)

@contextmanager
def ExplicitSubmitTests_executor_runtime(self):
    async def transport(name, args, context):
        self.assertEqual((context.platform, context.user_id), ("qq", self.key.actor_id))
        if name == "keytao_create_phrase":
            capability = context.trusted_reviewed_items_by_key[("加亮", "jslxa")]
            self.assertEqual(capability["pinyin"], "jiā liàng")
            self.assertEqual(args["_reviewed_pinyin"], "jiā liàng")
            self.assertIn("jslxa", args["_reviewed_candidate_codes"])
        raw = await self.tool(
            name, args, context.platform, context.user_id,
            trusted_reviewed_items_by_key=context.trusted_reviewed_items_by_key,
            _capture_receipt=False,
        )
        return json.loads(raw)

    with self.runtime(), patch.object(
        s63_fresh_selection_chat, "call_tool_function", REAL_TOOL_DISPATCH,
    ), patch.object(
        b17_rest_recommended_commands.tool_executor, "_invoke_effective_tool", side_effect=transport,
    ), patch.object(b17_rest_recommended_commands.memory_store, "record_tool_receipt"):
        yield

def ExplicitSubmitTests_seed_existing_draft(self, *, other_item=False):
    row = {"id": 6401, "action": "Create", "word": "加亮", "code": "jslxa", "type": "Phrase"}
    self.items = [row]
    self.version = 3
    self.pending_items = [{
        **row, "source": "draft", "batchId": self.batch_id, "batchUrl": self.batch_url,
        "batchStatus": "Draft", "itemStatus": "Pending",
    }]
    if other_item:
        self.items.append({"id": 6402, "action": "Create", "word": "旧词", "code": "qtxx", "type": "Phrase"})

def ExplicitSubmitTests_assert_submitted_or_actionable_submit_ticket(self, response):
    self.assertNotIn("本轮没有可执行的已绑定写操作", response)
    if self.submitted:
        self.assertIn("已提交", response)
        return
    ticket = self.store.get(self.key)
    self.assertTrue(server_warning_ticket_is_complete(ticket), response)
    self.assertEqual(ticket.function_name, "keytao_submit_batch")
    self.assertEqual(ticket.args["batch_id"], self.batch_id)
    delivered = s63_fresh_selection_chat._enforce_advertised_reply_contract(response, self.key)
    self.assertIn("确认", delivered)
    self.assertIn(self.batch_url, delivered)

# Source: test_s62_incident.py
def s62_incident_incident_record():
    record = reviewed_record(("小端",))
    payload = json.dumps(record.args, ensure_ascii=False)
    for old, new in (("小端", "敲不死"), ("xcdtio", "qbsoa"),
                     ("xcdti", "qbso"), ("xcdt", "qbs")):
        payload = payload.replace(old, new)
    record.args = json.loads(payload)
    scope = record.args["_candidate_scopes"][0]
    scope["occupiedWords"] = {"qbs": ["敲破死"]}
    scope["reviewedState"]["serverOccupiedWords"] = copy.deepcopy(scope["occupiedWords"])
    scope["reviewedPrompt"] = reviewed_scope("敲不死", ["qbs", "qbso", "qbsoa"], "qbso", "qiāo bù sǐ", scope["occupiedWords"])["reviewedPrompt"]
    record.args["_query_words"] = ["鸡白汤", "敲不死", "乔布斯"]
    record.args["_query_other_blocks"] = ["「鸡白汤」存在多个读音，请先确定读音。", "「乔布斯」已在词库。"]
    return record

async def S62IncidentTests_replay(self, messages, *, discovery=False, delivery=False, executor=False):
    store = harness.MemoryConversationStateStore()
    key = harness.ConversationAddress.group("qq", "865189947", "s62-offline")
    if not discovery:
        store.set(key, s62_incident_incident_record())
    writes, calls, responses = [], [], []
    submitted = False

    async def tool(name, args, platform, actor, **kwargs):
        nonlocal submitted
        self.assertEqual((platform, actor), ("qq", "s62-offline"))
        calls.append((name, copy.deepcopy(args)))
        base = {"success": True, "batchId": "s62-fixture", "contentVersion": len(writes)}
        if name == "keytao_pending_items_by_words":
            return json.dumps({"success": True, "complete": True, "items": []})
        if name == "keytao_lookup_by_word":
            return json.dumps({"success": True, "phrases": [
                {"word": "乔布斯", "code": "qbsk", "type": "Phrase"}
            ] if args["word"] == "乔布斯" else []})
        if name == "keytao_prepare_reviewed_add":
            if args["word"] == "鸡白汤":
                from tests.helpers import ambiguous_review as ambiguous_review
                return json.dumps(ambiguous_review())
            self.assertEqual(args["word"], "敲不死")
            return json.dumps({
                "success": True, "word": "敲不死", "recommendedCode": "qbso",
                "needsManualReview": True, "manualReviewReason": "组合读音需管理员复核",
                "pronunciations": [{"pinyin": "qiāo bù sǐ", "recommendedCode": "qbso",
                    "candidateStatuses": [
                        {"code": code, "occupied": occupied, "words": ["前半生"] if occupied else [],
                         "label": "已有「前半生」" if occupied else "空位"}
                        for code, occupied in (("qbs", True), ("qbso", False), ("qbsoa", False))
                    ]}],
            })
        if name in {"keytao_list_draft_items", "keytao_get_batch_preview"}:
            return json.dumps({**base, "items": writes, "count": len(writes), "status": "Draft"})
        if name == "keytao_batch_add_to_draft":
            if not args.get("confirmed"):
                return json.dumps({**base, "success": False, "requiresConfirmation": True,
                                   "warningDigest": "a" * 64, "warnings": []})
            self.assertEqual(args["expected_content_version"], len(writes))
            self.assertEqual(args["expected_warning_digest"], "a" * 64)
            self.assertFalse(writes)
            writes.extend({"id": 6200 + i, **copy.deepcopy(row)} for i, row in enumerate(args["items"]))
            result = {**base, "contentVersion": len(writes), "writtenItems": writes,
                      "successCount": len(writes), "failedCount": 0}
            b17_rest_recommended_commands._capture_successful_draft_write_delivery(name, result, platform, actor)
            return json.dumps(result)
        if name == "keytao_submit_batch":
            if not args.get("confirmed"):
                return json.dumps({**base, "success": False, "requiresConfirmation": True,
                                   "snapshotDigest": "b" * 64, "warningDigest": "c" * 64,
                                   "auditDigest": "d" * 64, "snapshotItems": writes, "warnings": []})
            self.assertEqual(args["expected_content_version"], len(writes))
            self.assertEqual(args["expected_server_snapshot_digest"], "b" * 64)
            self.assertEqual(args["expected_warning_digest"], "c" * 64)
            self.assertEqual(args["expected_audit_digest"], "d" * 64)
            self.assertFalse(submitted)
            submitted = True
            result = {**base, "status": "Submitted"}
            b17_rest_recommended_commands._capture_successful_draft_write_delivery(name, result, platform, actor)
            return json.dumps(result)
        raise AssertionError("Unexpected tool: " + name)

    async def transport(name, args, context):
        self.assertEqual((context.platform, context.user_id), ("qq", key.actor_id))
        if name == "keytao_batch_add_to_draft":
            self.assertEqual([(row["word"], row["code"]) for row in args["items"]], [("敲不死", "qbso")])
            self.assertEqual(context.trusted_reviewed_items_by_key[("敲不死", "qbso")]["pinyin"], "qiāo bù sǐ")
        return json.loads(await tool(name, args, context.platform, context.user_id))

    with patch.object(harness.openai_chat_module, "conversation_state_store", store), patch.object(
        harness.openai_chat_module, "call_tool_function", side_effect=tool,
    ), patch.object(harness.openai_chat_module, "_classify_message_command_intent",
                    AsyncMock(side_effect=AssertionError("S62 model calls forbidden"))), ExitStack() as stack:
        if executor:
            stack.enter_context(patch.object(harness.openai_chat_module, "call_tool_function", REAL_TOOL_DISPATCH))
            stack.enter_context(patch.object(b17_rest_recommended_commands.tool_executor, "_invoke_effective_tool", side_effect=transport))
            stack.enter_context(patch.object(b17_rest_recommended_commands.memory_store, "record_tool_receipt"))
        if discovery:
            footer = await b17_rest_recommended_commands._try_handle_simple_single_word_query(
                "鸡白汤，敲不死，乔布斯", "qq", key.actor_id, key,
            )
            self.assertIn("鸡白汤", footer)
            self.assertIn("乔布斯", footer)
            self.assertIn("敲不死 2", footer)
            self.assertTrue(harness.openai_chat_module._advertised_reply_matches_live_record(footer, store.get_record(key)))
            responses.append(footer)
        for message in messages:
            chat = harness.openai_chat_module
            memory = harness.ChatMemoryContext(platform="qq", user_id=key.actor_id, space_type="group", space_id=key.space_id)
            message_token = chat._current_turn_message.set(message)
            delivery_token = b17_rest_recommended_commands.current_draft_delivery_claims.set([])
            try:
                ctx = chat.TurnContext(
                    bot=object(), event=SimpleNamespace(message_id=None), platform="qq", user_id=key.actor_id,
                    conv_key=key, space_key=key.space_key, memory_context=memory, normalized_message_text=message,
                    generic_command_intent=harness.MessageCommandIntent(intent="draft_view" if message == "查看草稿" else "none"),
                )
                if message == "查看草稿":
                    await chat._stage_handle_draft_management(ctx)
                else:
                    ctx.response = await b17_rest_recommended_commands.handle_pending_message_core(
                        message, "qq", "s62-offline", key, allow_intent_model=False,
                    )
                if delivery and ctx.response is not None:
                    for stage in (chat._stage_normalize_response, chat._stage_scope_language_only_response,
                                  chat._stage_append_ticket_challenge, chat._stage_enforce_advertised_reply_contract):
                        await stage(ctx)
                    ctx.response = chat._prepare_user_facing_reply(ctx.response, memory)
                responses.append(ctx.response)
            finally:
                b17_rest_recommended_commands.current_draft_delivery_claims.reset(delivery_token)
                chat._current_turn_message.reset(message_token)
    return writes, submitted, responses, calls, store.get(key)

# Source: test_s62_adversarial.py
# Source: test_s62_commonness.py
from keytao_bot.utils.commonness_query import parse_commonness_query as parse_commonness_query
s62_commonness_MESSAGE = "加量，加梁，加亮，加辆，价量，脊索裂畸形，排列以上词的使用频率"

s62_commonness_WORDS = ("加量", "加梁", "加亮", "加辆", "价量", "脊索裂畸形")

# Source: test_s62_draft_flow.py
s62_draft_flow_chat = harness.openai_chat_module

s62_draft_flow_commands = harness.chat_commands_module

BATCH = "s62-draft"

ITEM = {"id": 3397, "action": "Create", "word": "加亮", "code": "jslxa", "type": "Phrase"}

OLD_ITEM = {"id": 3396, "action": "Create", "word": "旧词", "code": "jc", "type": "Phrase"}

QUERY_WORDS = ["加量", "加梁", "加亮", "加辆", "价量", "脊索裂畸形", "排列以上词的使用频率"]

def DraftFlowTests_setUp(self):
    self.key = harness.ConversationAddress.group("qq", "s62-room", "s62-actor")
    self.store = harness.MemoryConversationStateStore()
    self.memory = harness.ChatMemoryContext(
        platform="qq", user_id=self.key.actor_id, space_type="group", space_id="s62-room",
    )
    self.calls = []
    self.items = [copy.deepcopy(ITEM)]
    self.version = 3
    self.submitted = False

async def DraftFlowTests_tool(self, name, args, platform, user_id, **kwargs):
    self.assertEqual((platform, user_id), ("qq", self.key.actor_id))
    self.calls.append((name, copy.deepcopy(args)))
    base = {"success": True, "batchId": BATCH, "contentVersion": self.version,
            "batchUrl": f"https://keytao.rea.ink/batch/{BATCH}"}
    if name == "keytao_submit_batch":
        self.assertEqual(args.get("batch_id"), BATCH)
        if args.get("confirmed"):
            self.assertNotIn("preview_only", args)
            self.assertEqual(args["expected_content_version"], self.version)
            self.assertEqual(args["expected_server_snapshot_digest"], "a" * 64)
            self.assertEqual(args["expected_warning_digest"], "b" * 64)
            self.assertEqual(args["expected_audit_digest"], "c" * 64)
            self.assertFalse(self.submitted, "Submit replayed twice")
            self.submitted = True
            return json.dumps({**base, "status": "Submitted"})
        return json.dumps({**base, "success": False, "requiresConfirmation": True,
                           "snapshotDigest": "a" * 64, "warningDigest": "b" * 64,
                           "auditDigest": "c" * 64, "snapshotItems": self.items, "warnings": []})
    if name == "keytao_batch_add_to_draft":
        self.assertEqual([(i["word"], i["code"]) for i in args["items"]], [("加亮", "jslxa")])
        return json.dumps({**base, "writtenItems": [ITEM], "successCount": 1})
    if name == "keytao_list_draft_items":
        return json.dumps({**base, "items": self.items, "count": len(self.items)})
    if name == "keytao_get_batch_preview":
        return json.dumps({**base, "summary": {"added": len(self.items)}})
    raise AssertionError("Unexpected external tool: " + name)

@contextmanager
def DraftFlowTests_runtime(self):
    with ExitStack() as stack:
        for patcher in (
            patch.object(s62_draft_flow_chat, "conversation_state_store", self.store),
            patch.object(s62_draft_flow_chat, "draft_operation_coordinator", harness.DraftOperationCoordinator()),
            patch.object(s62_draft_flow_chat, "call_tool_function", side_effect=self.tool),
            patch.object(s62_draft_flow_chat, "remember_conversation"),
            patch.object(s62_draft_flow_chat, "schedule_memory_compaction"),
            patch.object(s62_draft_flow_chat, "_finish_ai_chat_matcher", AsyncMock()),
            patch.object(s62_draft_flow_chat, "_classify_message_command_intent", AsyncMock(side_effect=AssertionError("No model"))),
            patch.object(s62_draft_flow_chat, "_classify_simple_word_query_intent", AsyncMock(side_effect=AssertionError("No model"))),
        ):
            stack.enter_context(patcher)
        token = s62_draft_flow_commands.current_draft_delivery_claims.set([])
        memory_token = s62_draft_flow_commands.current_memory_context.set(self.memory)
        try:
            yield
        finally:
            s62_draft_flow_commands.current_memory_context.reset(memory_token)
            s62_draft_flow_commands.current_draft_delivery_claims.reset(token)

def DraftFlowTests_acknowledge_write(self, *, with_delta=True):
    result = {"success": True, "batchId": BATCH, "contentVersion": self.version}
    if with_delta:
        result["writtenItems"] = [copy.deepcopy(ITEM)]
    s62_draft_flow_commands._capture_successful_draft_write_delivery(
        "keytao_batch_add_to_draft", result, "qq", self.key.actor_id,
    )
    s62_draft_flow_commands._acknowledge_delivered_draft_mutations()
    self.assertTrue(self.store.get(self.key).args["_recent_own_write"])

def DraftFlowTests_context(self, message, *, key=None):
    key = key or self.key
    quote = f"已变更：加亮 → jslxa\n草稿地址：https://keytao.rea.ink/batch/{BATCH}"
    return s62_draft_flow_chat.TurnContext(
        bot=object(), event=object(), platform="qq", user_id=key.actor_id,
        normalized_message_text=message, conv_key=key, space_key=key.space_key,
        memory_context=self.memory, history=[{"role": "assistant", "content": quote}],
        reply_reference=harness.ReplyReferenceInfo(is_reply=True, is_to_bot=True, text=quote),
        command_intent_for=AsyncMock(side_effect=AssertionError("No model")),
    )

async def DraftFlowTests_continue_after_write(self, message="加入并提交"):
    ctx = self.context(message)
    await s62_draft_flow_chat._stage_resolve_current_pending_scope(ctx)
    self.assertIsNotNone(self.store.get(self.key), "Recent write discarded before submission")
    self.assertFalse(await s62_draft_flow_chat._stage_apply_scoped_pending_intent(ctx))
    await s62_draft_flow_chat._stage_execute_pending_state(ctx)
    return ctx

# Source: test_s62_reading_resolution.py
s62_reading_resolution_commands = harness.chat_commands_module

s62_reading_resolution_chat = harness.openai_chat_module

def ambiguous_review():
    return {
        "success": True, "word": "鸡白汤", "type": "Phrase",
        "recommendedCode": "", "reviewDisposition": "BLOCK",
        "reviewVerdictSite": "pronunciation_unresolved",
        "pronunciationUnresolved": True, "lookupFailed": False,
        "multiSenseChoice": {"status": "ambiguous"},
        "pronunciations": [
            {"pinyin": pinyin, "codes": [code], "recommendedCode": code,
             "candidateStatuses": [{"code": code, "occupied": False, "words": []}]}
            for pinyin, code in (("jī bái tāng", "jbtaua"),
                                 ("jī bó tāng", "jbtaua"),
                                 ("jī zì tāng", "jztaua"))
        ],
    }

async def ReadingResolutionTests_prepare_ambiguous(self, *, same_code):
    from keytao_bot.utils import keytao_review as review_module

    groups = [
        {"pinyin": pinyin, "normalized": normalized, "sources": [], "score": 0,
         "requiresManualReview": True}
        for pinyin, normalized in (("jī bái tāng", ["ji", "bai", "tang"]),
                                   ("jī bó tāng", ["ji", "bo", "tang"]))
    ]
    alternate_code = "jbtaua" if same_code else "jbtauo"
    encode = {
        "success": True, "word": "鸡白汤", "codes": ["jbtaua"],
        "candidateCodes": ["jbtaua", alternate_code],
        "pronunciationSource": "pinyin-pro-context", "standardPronunciationStatus": "absent",
        "phrasePinyins": ["jī", "bái", "tāng"], "contextPhrasePinyins": ["jī", "bái", "tāng"],
        "alternatePhrasePronunciationCodes": [{
            "char": "白", "charIndex": 1, "pinyin": "bó", "codes": [alternate_code],
        }],
        "chars": [
            {"char": char, "pinyin": readings[0], "pinyins": readings,
             "pronunciationLookupStatus": "found"}
            for char, readings in (("鸡", ["jī"]), ("白", ["bái", "bó"]), ("汤", ["tāng"]))
        ],
    }
    evidence = {"success": True, "groups": groups, "sources": [], "lookupComplete": True}
    with patch.object(review_module, "collect_pronunciation_evidence_limited", AsyncMock(return_value=evidence)), patch.object(
        review_module, "fetch_keytao_encode", AsyncMock(return_value=encode),
    ), patch.object(review_module, "lookup_words", AsyncMock(return_value={})), patch.object(
        review_module, "lookup_codes", AsyncMock(return_value={}),
    ), patch.object(review_module, "_resolve_multi_sense_pronunciation_choice", AsyncMock(return_value=(groups, {"status": "ambiguous"}))):
        return await review_module.prepare_reviewed_word(None, "鸡白汤")

async def ReadingResolutionTests_execute(self, message, review=None):
    request = parse_explicit_entry_code_request(message)
    self.assertIsNotNone(request, message)
    store = harness.MemoryConversationStateStore()
    key = harness.ConversationAddress.private("qq", "s62-reading")
    review = copy.deepcopy(review or ambiguous_review())
    calls = []

    async def tool(name, args, platform, user_id):
        calls.append((name, args))
        if name == "keytao_prepare_reviewed_add":
            return json.dumps(review)
        if name == "keytao_lookup_by_code":
            return json.dumps({"success": True, "phrases": []})
        raise AssertionError("Unexpected external tool: " + name)

    execute = AsyncMock(return_value="fixture submitted")
    with patch.object(s62_reading_resolution_chat, "conversation_state_store", store), patch.object(
        s62_reading_resolution_chat, "call_tool_function", side_effect=tool,
    ), patch.object(s62_reading_resolution_commands, "_execute_add_to_draft_and_submit", execute), patch.object(
        s62_reading_resolution_chat, "_classify_message_command_intent",
        AsyncMock(side_effect=AssertionError("No model calls")),
    ):
        response = await s62_reading_resolution_commands._execute_explicit_entry_code_request(
            request, message, "qq", key.actor_id, key, None, "",
        )
    return response, execute, calls

def ExplicitReadingReplayTests_setUp(self):
    ExplicitSubmitTests_setUp(self)
    self.review = ambiguous_review()

async def ExplicitReadingReplayTests_tool(self, name, args, platform, user_id, **kwargs):
    self.assertEqual((platform, user_id), ("qq", self.key.actor_id))
    self.calls.append((name, copy.deepcopy(args)))
    if name == "keytao_pending_items_by_words":
        result = {"success": True, "complete": True, "items": []}
    elif name == "keytao_prepare_reviewed_add":
        self.assertEqual(args["word"], "鸡白汤")
        result = self.review
    elif name == "keytao_lookup_by_code":
        self.assertEqual(args["code"], "jbtaua")
        result = {"success": True, "phrases": []}
    elif name == "keytao_create_phrase":
        self.assertEqual((args["word"], args["code"]), ("鸡白汤", "jbtaua"))
        self.assertIs(args["needs_manual_review"], True)
        self.assertIn("jī bái tāng", args["remark"])
        self.assertIn("管理员复核", args["remark"])
        capability = kwargs["trusted_reviewed_items_by_key"][("鸡白汤", "jbtaua")]
        self.assertEqual(capability["pinyin"], "jī bái tāng")
        if not args.get("confirmed"):
            self.assertTrue(args["preview_only"])
            result = {"success": False, "requiresConfirmation": True,
                      "batchId": self.batch_id, "batchUrl": self.batch_url,
                      "contentVersion": self.version, "warningDigest": "a" * 64,
                      "warnings": [], "warnedCount": 0}
        else:
            self.assertEqual(args["batch_id"], self.batch_id)
            self.assertEqual(args["expected_content_version"], self.version)
            self.assertEqual(args["expected_warning_digest"], "a" * 64)
            self.assertFalse(self.writes)
            row = {"id": 6201, "action": "Create", "word": args["word"],
                   "code": args["code"], "type": "Phrase",
                   "needsManualReview": args["needs_manual_review"], "remark": args["remark"]}
            self.items.append(row)
            self.writes.append(copy.deepcopy(row))
            self.version += 1
            result = {"success": True, "batchId": self.batch_id, "batchUrl": self.batch_url,
                      "contentVersion": self.version, "writtenItems": copy.deepcopy(self.writes),
                      "pullRequestCount": 1}
    elif name in {"keytao_list_draft_items", "keytao_get_batch_preview"}:
        result = {"success": True, "batchId": self.batch_id, "batchUrl": self.batch_url,
                  "contentVersion": self.version, "status": "Draft",
                  "items": copy.deepcopy(self.items), "count": len(self.items)}
    elif name == "keytao_submit_batch":
        self.assertEqual(args["batch_id"], self.batch_id)
        if not args.get("confirmed"):
            result = {"success": False, "requiresConfirmation": True,
                      "batchId": self.batch_id, "batchUrl": self.batch_url,
                      "contentVersion": self.version, "snapshotDigest": "c" * 64,
                      "warningDigest": "d" * 64, "auditDigest": "e" * 64,
                      "snapshotItems": copy.deepcopy(self.items), "warnings": []}
        else:
            self.assertEqual(args["expected_content_version"], self.version)
            self.assertEqual(args["expected_server_snapshot_digest"], "c" * 64)
            self.assertEqual(args["expected_warning_digest"], "d" * 64)
            self.assertEqual(args["expected_audit_digest"], "e" * 64)
            self.assertFalse(self.submitted)
            self.submitted = True
            self.version += 1
            result = {"success": True, "status": "Submitted", "batchId": self.batch_id,
                      "batchUrl": self.batch_url, "contentVersion": self.version}
    else:
        raise AssertionError("Unexpected external tool: " + name)
    s62_reading_resolution_commands._capture_successful_draft_write_delivery(name, result, platform, user_id)
    return json.dumps(result)

# Source: test_s62_stale_draft.py
s62_stale_draft_chat = harness.openai_chat_module

class s62_stale_draft_FakeBot:
    def __init__(self):
        self.messages = []

    async def send(self, **kwargs):
        self.messages.append(kwargs["message"])

# Source: test_s63_bcc.py
from itertools import product as product
PAIRS = (('情报所', '敲不死'), ('环境法', '户晨风'), ('五角星', '沃集鲜'))

s63_bcc_FIXTURE = Path(__file__).parents[1] / 'e2e/fixtures/bcc/s63.json'

def seed_historical(db, channel='古代汉语', counts=None, kind='word', total=10000, rows=100):
    """Synthetic historical evidence, separate from the official corpus slice."""
    prefix = 'classical_chinese' if channel == '古代汉语' else 'modern_chinese'
    dataset = f'{prefix}_{kind}_freq.txt'
    with closing(sqlite3.connect(db)) as connection, connection:
        connection.execute('DELETE FROM bcc_frequency WHERE dataset=?', (dataset,))
        connection.execute('INSERT OR REPLACE INTO bcc_dataset VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (dataset, channel, kind, 'synthetic', 'synthetic', 'synthetic', 1, 1, rows, total))
        for token, (count, rank) in (counts or {}).items():
            connection.execute('INSERT INTO bcc_frequency VALUES (?, ?, ?, ?, ?)',
                               (dataset, token, count, count * 1000000 / total, rank))

def BccCommonnessTests_setUp(self):
    directory = tempfile.TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    self.db = Path(directory.name) / 'reference.db'
    fixture = json.loads(s63_bcc_FIXTURE.read_text())
    with closing(sqlite3.connect(self.db)) as connection, connection:
        create_schema(connection)
        connection.execute('CREATE TABLE word_commonness (word TEXT PRIMARY KEY, corpus_frequency INTEGER, part_of_speech TEXT, dictionary_presence_count INTEGER)')
        for table, rows in (('bcc_dataset', fixture['datasets']), ('bcc_frequency', fixture['rows']), ('word_commonness', fixture['jieba'])):
            for row in rows:
                connection.execute(f'INSERT INTO {table} ({",".join(row)}) VALUES ({",".join("?" for _ in row)})', tuple(row.values()))
    path_patch = patch.object(review, 'reference_db_path', return_value=self.db)
    path_patch.start()
    self.addCleanup(path_patch.stop)
    fallback = patch.object(review, '_estimate_word_commonness_web_fallback', AsyncMock(side_effect=AssertionError('S63 must be local')))
    self.web = fallback.start()
    self.addCleanup(fallback.stop)
    review._clear_review_caches()

# Source: test_s63_bcc_delivery.py
from keytao_bot.skills import SkillsManager as SkillsManager
def BccDeliveryTests_setUp(self):
    BccCommonnessTests_setUp(self)

def BccDeliveryTests_run_stage(self, message, *, historical):
    chat, commands = harness.openai_chat_module, harness.chat_commands_module
    owner = harness.ConversationAddress.private('qq', 's63-fixture')
    memory = SimpleNamespace(conversation_address=owner, platform='qq')
    ctx = SimpleNamespace(normalized_message_text=message, platform='qq', user_id=owner.actor_id,
                          conv_key=owner, memory_context=memory, bot=object(), event=object(), QQMessageSegment=None)
    delivered = []

    async def dispatch(name, args, *actor):
        self.assertIn(name, {'keytao_lookup_by_words_batch', 'keytao_encode'})
        return json.dumps({'success': False})

    async def finish(_bot, _event, _user, context, response, _segment):
        delivered.append(chat._prepare_user_facing_reply(response, context))

    async def run():
        token = chat._current_turn_message.set(message)
        try:
            with (patch.object(commands, 'call_tool_function', side_effect=dispatch),
                  patch.object(chat, 'remember_conversation'),
                  patch.object(chat, '_finish_ai_chat_response', side_effect=finish),
                  patch.object(chat, 'get_ai_response_core', side_effect=AssertionError('No model calls')),
                  patch.object(chat, '_classify_message_command_intent', side_effect=AssertionError('No model classifier'))):
                self.assertTrue(await chat._stage_handle_commonness_query(ctx))
        finally:
            chat._current_turn_message.reset(token)

    asyncio.run(run())
    self.assertEqual(len(delivered), 1)
    self.assertIn('BCC', delivered[0])
    if historical:
        self.assertIn('现代四频道均未收录', delivered[0])
        self.assertIn('按古代汉语频次', delivered[0])
        self.assertIn('1,234', delivered[0])
        self.assertIn('「古例甲」较「古例乙」更常用', delivered[0])
    else:
        self.assertIn('多领域 34（每百万 0.08）', delivered[0])
        self.assertIn('单边 BCC 收录不决定高低', delivered[0])
        self.assertNotIn('常用度信号不足', delivered[0])
    self.assertIn('古代汉语', delivered[0])
    self.assertIn('近代汉语', delivered[0])
    self.assertIn('历史补充', delivered[0])
    self.assertNotIn('「情报所」较「敲不死」更常用', delivered[0])
    for field in ('perMillion', 'raw_count', 'bcc_frequency', 'keytao_word_commonness'):
        self.assertNotIn(field, delivered[0])

# Source: test_s63_bcc_ingest.py
import io as io
import zipfile as zipfile
from scripts import ingest_bcc as ingest
def BccIngestTests_cached_corpus(self, root):
    db, cache = root / 'reference.db', root / 'cache'
    cache.mkdir()
    with closing(sqlite3.connect(db)) as connection, connection:
        connection.execute('CREATE TABLE word_commonness (word TEXT PRIMARY KEY, corpus_frequency INTEGER, part_of_speech TEXT, dictionary_presence_count INTEGER)')
    datasets = []
    for channel, prefix in ingest.CHANNEL_PREFIXES.items():
        for kind in ('char', 'word'):
            name = f'{prefix}_{kind}_freq.txt'
            # Different denominators and table populations; deliberately unsorted.
            rows = [('乙', 100), ('甲', 900)] if kind == 'char' else [
                *[(f'对照{index}', 10) for index in range(9)], ('甲词', 20)]
            payload = ('token,count\n' + ''.join(f'{token},{count}\n' for token, count in rows)).encode()
            with zipfile.ZipFile(cache / (name + '.zip'), 'w') as archive:
                archive.writestr(name, payload)
            datasets.append(dict(filename=name, channel=channel, token_type=kind,
                updated_at='fixture', format='txt', freq_type='freq',
                token_column='token', count_column='count', size_bytes=len(payload)))
    return db, cache, datasets

# Source: test_s63_general_reply.py
from keytao_bot.harness.authorization_grammar import message_authorizes_mutation as message_authorizes_mutation
NO_PLAN = "这次调整还没有可确认的计划，本次未写入。"

def policy_agent(calls, final_text="Fixture response"):
    client = harness._FakeClient([
        harness._FakeAIResponse("tool_calls", "", calls),
        harness._FakeAIResponse("stop", final_text),
    ])
    properties = {
        "word": {"type": "string"}, "code": {"type": "string"},
        "items": {"type": "array", "items": {"type": "object", "properties": {
            "word": {"type": "string"}, "code": {"type": "string"}, "action": {"type": "string"},
        }}},
    }
    schemas = [{"type": "function", "function": {
        "name": name, "description": "Fixture", "parameters": {"type": "object", "properties": properties},
    }} for name in {item.function.name for item in calls}]
    manager = SimpleNamespace(get_skill_instructions=lambda: "", has_tools=lambda: True, get_tools=lambda: schemas)

    async def forbidden(**_kwargs):
        raise AssertionError("No real sink may execute")

    agent = harness.AgentOrchestrator(
        client_factory=lambda: client,
        runtime=harness.AgentRuntimeConfig(model="fake-model", max_tokens=1000, temperature=0.0, timeout=10),
        skills_manager=manager,
        tool_executor=harness.ToolExecutor(lambda _name: forbidden, frozenset()),
        state_store=harness.MemoryConversationStateStore(), bind_help_text="fixture", system_prompt_core="fixture",
    )
    return agent, client

def policy_call(name, args, index):
    return SimpleNamespace(id=f"s63-policy-{index}", type="function", function=SimpleNamespace(
        name=name, arguments=json.dumps(args),
    ))

async def GeneralReplyTests_reply(self, message, raw, pending=None):
    client = harness._FakeClient([harness._FakeAIResponse("stop", raw)])
    store = harness.MemoryConversationStateStore()
    context = harness.AgentRequestContext(
        platform="qq", user_id="s63-general",
        mutations_allowed=message_authorizes_mutation(message),
    )
    if pending is not None:
        store.set(context.conversation_address, pending)

    def forbidden_tool(_name):
        raise AssertionError("A text-only reply must not dispatch tools")

    orchestrator = harness.AgentOrchestrator(
        client_factory=lambda: client,
        runtime=harness.AgentRuntimeConfig(
            model="fake-model", max_tokens=1000, temperature=0.0, timeout=10.0,
        ),
        skills_manager=harness._FakeSkillsManager(),
        tool_executor=harness.ToolExecutor(forbidden_tool, frozenset()),
        state_store=store, bind_help_text="fixture", system_prompt_core="fixture",
    )
    response = await orchestrator.run(message, context)
    self.assertEqual(len(client.completions.calls), 1)
    return response

# Source: test_s64_existing_query.py
import shutil as shutil
s64_existing_query_chat = harness.openai_chat_module

s64_existing_query_commands = harness.chat_commands_module

CORPUS_MODES = ("absent", "installed_miss", "modern_hit", "historical_hit")

def lookup_fixture(word="单份", code="dffn", phrase_type="Phrase"):
    row = {"word": word, "code": code, "weight": 100, "type": phrase_type}
    codes = [code, code + "o", code + "i"]
    if word == "单份":
        codes = ["dffn", "dffno", "dffnoi", "effn", "effno", "effnoi"]
    statuses = [{"code": candidate, "occupied": candidate == code,
                 "words": [word] if candidate == code else [],
                 "entries": [(word, 100)] if candidate == code else [],
                 "label": f"已有「{word}」" if candidate == code else "空位"}
                for candidate in codes]
    encoding = {"success": True, "word": word, "type": "二字词" if len(word) == 2 else "fixture",
                "baseCode": code, "recommendedCode": codes[1],
                "candidateCodes": codes, "candidateStatuses": statuses}
    return {
        "keytao_lookup_by_word": {"success": True, "word": word, "phrases": [row]},
        "keytao_list_draft_items": {"success": True, "count": 0, "items": []},
        "keytao_pending_items_by_words": {"success": True, "complete": True, "items": []},
        "keytao_lookup_by_words_batch": {"success": True, "count": 1, "results": [{"word": word, "phrases": [row]}]},
        "keytao_lookup_by_code": {"success": True, "code": code, "count": 1, "phrases": [row]},
        "keytao_encode": encoding,
    }

def pending_candidate(fixture):
    encoding = fixture["keytao_encode"]
    word = encoding["word"]
    return harness.AgentOrchestrator._trusted_single_pending_add(
        {word: tuple((row["code"], row["occupied"]) for row in encoding["candidateStatuses"])},
        {word: tuple(encoding["candidateStatuses"])},
        {word: encoding["recommendedCode"]}, {},
    )

def ExistingQueryTests_setUp(self):
    BccCommonnessTests_setUp(self)

@contextmanager
def ExistingQueryTests_corpus(self, mode):
    with tempfile.TemporaryDirectory() as directory:
        db = Path(directory) / "reference.db"
        shutil.copyfile(self.db, db)
        with closing(sqlite3.connect(db)) as connection, connection:
            if mode == "absent":
                connection.execute("DROP TABLE bcc_frequency")
                connection.execute("DROP TABLE bcc_dataset")
            elif mode == "modern_hit":
                # Synthetic attestation tests installation behavior, not real counts.
                connection.execute("INSERT INTO bcc_frequency SELECT filename, '单份', 100, 10, 1 FROM bcc_dataset WHERE channel='多领域' AND token_type='word'")
        if mode == "historical_hit":
            seed_historical(db, counts={"单份": (100, 1)})
            seed_historical(db, channel="近代汉语", counts={"单份": (50, 2)})
        with patch.object(s64_existing_query_commands.keytao_review, "reference_db_path", return_value=db):
            s64_existing_query_commands.keytao_review._clear_review_caches()
            evidence = s64_existing_query_commands.keytao_review._query_commonness_reference("单份")["bcc"]
            self.assertEqual(evidence["available"], mode != "absent")
            self.assertEqual(evidence["attested"], mode == "modern_hit")
            if mode == "historical_hit":
                self.assertEqual([row["count"] for row in evidence["historicalChannels"]], [100, 50])
            yield
        s64_existing_query_commands.keytao_review._clear_review_caches()

# Source: test_s65_commonness_render.py
from keytao_bot.plugins import chat_render as chat_render
from keytao_bot.utils import commonness_copy as commonness_copy
from keytao_bot.utils import commonness_query as commonness_query
from keytao_bot.utils import pending_confirmation as pending
from keytao_bot.utils import same_code_reorder as same_code_reorder
def candidate_fixture(comparison, code="qbs"):
    word, occupant = comparison["frontWord"], comparison["behindWord"]
    assessment = review._candidate_commonness_assessment({
        "newWord": word, "occupantWord": occupant,
        "occupantCode": code, "freeCode": code + "o",
    }, comparison)
    statuses = [
        {"code": code, "occupied": True, "words": [occupant], "label": f"已有「{occupant}」"},
        {"code": code + "o", "occupied": False, "words": [], "label": "空位"},
    ]
    recommended = assessment["recommendedCode"]
    payload = {
        "success": True, "word": word, "type": "Phrase", "recommendedCode": recommended,
        "candidateOrderingAssessments": [assessment],
        "pronunciations": [{"pinyin": "fixture", "recommendedCode": recommended,
                            "candidateStatuses": statuses}],
    }
    state = harness.AgentOrchestrator._trusted_single_pending_add(
        {word: tuple((row["code"], row["occupied"]) for row in statuses)},
        {word: tuple(statuses)}, {word: recommended}, {},
        candidate_ordering_by_word={word: [assessment]},
    )
    return assessment, payload, state

def CommonnessRenderTests_setUp(self):
    BccCommonnessTests_setUp(self)

def CommonnessRenderTests_local_comparison(self, first, second):
    return review._compare_reference_commonness(
        first, second, review._query_commonness_reference(first),
        review._query_commonness_reference(second),
    )

def CommonnessRenderTests_seed_both(self, second_rate=0.0371):
    # Synthetic counts/denominator reproduce the supplied ppm pair exactly.
    # They are not presented as production channel/count observations.
    with closing(sqlite3.connect(self.db)) as connection, connection:
        connection.execute("UPDATE bcc_dataset SET total_count=10000000000 WHERE filename='multi_domain_total_word_freq.txt'")
        connection.executemany(
            "INSERT INTO bcc_frequency VALUES ('multi_domain_total_word_freq.txt', ?, ?, ?, ?)",
            [("蛋粉", 944, 0.0944, 1), ("单份", round(second_rate * 10000), second_rate, 2)],
        )

def CommonnessRenderTests_assert_delivery(self, comparison, code="qbs"):
    assessment, payload, state = candidate_fixture(comparison, code)
    self.assertEqual(assessment["summary"], comparison["summary"])
    self.assertEqual(assessment["decisionReason"], comparison["decisionReason"])
    chat = harness.openai_chat_module
    store = harness.MemoryConversationStateStore()
    key = harness.ConversationAddress.private("qq", "s65-render")
    store.set(key, state)
    scope = {"word": state.word, "candidates": state.server_candidates,
             "occupiedWords": state.server_occupied_words,
             "orderingAssessments": state.server_ordering_assessments}
    item = {"word": state.word, "code": assessment["recommendedCode"], "needsManualReview": True}
    # Once the assessment exists, no renderer may query again.
    with patch.object(review, "_query_commonness_reference", side_effect=AssertionError("Render must use the snapshot")):
        prompt = chat_render._format_reviewed_add_prompt(payload)
        rebuilt = pending.render_server_backed_single_word_candidates(
            state.word, assessment["recommendedCode"], state.server_candidates,
            state.server_occupied_words, state.server_ordering_assessments,
        )
        lookup = pending.render_server_backed_single_word_lookup(
            state.word, assessment["recommendedCode"], state.server_candidates,
            state.server_occupied_words, ordering_assessments=state.server_ordering_assessments,
            reviewed_prompt=prompt, actionable_controls=True,
        )
        second = {"word": "测试另词", "candidates": [("abcd", False)],
                  "occupiedWords": {}, "orderingAssessments": []}
        batch = pending.render_server_backed_batch_candidates(
            [item, {"word": "测试另词", "code": "abcd"}], [scope, second],
        )
        for name, text in (("reviewed", prompt), ("rebuilt", rebuilt), ("lookup", lookup), ("batch", batch)):
            with self.subTest(path=name, verdict=comparison["verdict"]):
                self.assertIn(comparison["summary"], text)
        self.assertIn(comparison["summary"], review.build_review_note({
            "commonnessComparisons": [{"frontWord": state.word,
                "behindWord": assessment["occupantWord"], "code": code, "result": comparison}],
        }))
        review.record_commonness_evidence(comparison)
        with patch.object(chat, "conversation_state_store", store):
            delivered = chat._enforce_advertised_reply_contract(prompt, key)
            self.assertIn(comparison["summary"], delivered)
            self.assertIn(comparison["summary"], chat._prepare_user_facing_reply(
                delivered, SimpleNamespace(conversation_address=key, platform="qq")))
    self.web.assert_not_called()
    return chat_render._format_candidate_ordering_assessment(assessment, {code: 1, code + "o": 2})

CommonnessRenderArchitectureTests_VERDICT_OWNERS = {
        ("utils/keytao_review.py", "_reference_comparison_summary"),
        ("utils/keytao_review.py", "_web_fallback_summary"),
        ("utils/commonness_copy.py", "render_commonness_summary"),
    }

CommonnessRenderArchitectureTests_NON_RENDERERS = {
        ("plugins/chat_routing.py", "_extract_referenced_word_targets"),
        ("utils/keytao_batch_review.py", "_call_llm"),
        ("utils/keytao_review.py", "can_llm_override_audit_issues"),
        ("skills/keytao-draft/tools.py", "_fallback_draft_audit_with_encode"),
    }

CommonnessRenderArchitectureTests_SITES = {
        "plugins/chat_render.py": {"_format_candidate_ordering_assessment", "_format_reviewed_add_prompt"},
        "utils/pending_confirmation.py": {
            "front_insert_recommendation_copy", "candidate_commonness_guard_copy",
            "render_server_backed_single_word_candidates", "render_server_backed_single_word_lookup",
            "render_server_backed_batch_candidates", "render_server_backed_batch_lookup",
        },
        "utils/same_code_reorder.py": {"_advisory"},
        "utils/commonness_query.py": {"render_commonness_table"},
        "utils/keytao_review.py": {"_chain_recommendation_text", "build_review_note"},
        "plugins/chat_commands.py": {
            "_protected_eviction_response", "_generate_usage_comparison_note",
            "_format_code_chain_reorder_confirmation", "_reorder_evidence_lines",
            "_format_reorder_noop",
        },
    }

# Source: test_s66_readings.py
from keytao_bot.harness.authorization_grammar import normalize_mutation_command_source as normalize_mutation_command_source
from keytao_bot.utils import observability as observability
from keytao_bot.utils.reading_request import parse_parenthesised_readings as parse_parenthesised_readings
T1 = "@喵喵 一力降十会（yi2 li4 xiang3 shi2 hui），大力出奇迹（ da4 li4 chu1 qi2 ji1）"

s66_readings_T2 = "@喵喵 加词 @喵喵 一力降十会（yi2 li4 xiang2 shi2 hui4），大力出奇迹（ da4 li4 chu1 qi2 ji1）"

s66_readings_WORDS = ("一力降十会", "大力出奇迹")

READINGS = ("yī lì xiáng shí huì", "dà lì chū qí jì")

s66_readings_CODES = (("ylxh", "ylxhu", "ylxhui"), ("dlcj", "dlcju", "dlcjui"))

async def prepared_review(word, requested):
    index = s66_readings_WORDS.index(word)
    syllables = READINGS[index].split()
    alternate = ["yljh", "yljhu", "yljhui"] if index == 0 else []
    encode = {
        "success": True, "word": word, "codes": list(s66_readings_CODES[index]),
        "candidateCodes": [*s66_readings_CODES[index], *alternate],
        "pronunciationSource": "pinyin-pro-context", "standardPronunciationStatus": "absent",
        "phrasePinyins": syllables, "contextPhrasePinyins": syllables,
        "alternatePhrasePronunciationCodes": ([{
            "char": "降", "charIndex": 2, "pinyin": "jiàng", "codes": alternate,
        }] if alternate else []),
        "chars": [{"char": char, "pinyin": syllable,
                   "pinyins": [syllable, "jiàng"] if char == "降" else [syllable],
                   "pronunciationLookupStatus": "found"}
                  for char, syllable in zip(word, syllables)],
    }
    with patch.object(review, "collect_pronunciation_evidence_limited", AsyncMock(return_value={
        "success": True, "groups": [], "sources": [], "lookupComplete": True,
    })), patch.object(review, "fetch_keytao_encode", AsyncMock(return_value=encode)) as encoder, patch.object(
        review, "lookup_words", AsyncMock(return_value={}),
    ), patch.object(review, "lookup_codes", AsyncMock(return_value={})), patch.object(
        review, "AsyncOpenAI", Mock(side_effect=AssertionError("Review model forbidden")),
    ) as model:
        result = await review.prepare_reviewed_word(None, word, requested_reading=requested)
        encoder.assert_awaited_once()
        model.assert_not_called()
    return result

def TranscriptTests_setUp(self):
    FreshSelectionTests_setUp(self)
    self.key = harness.ConversationAddress.group("qq", "865189947", "s66-offline")
    self.memory = harness.ChatMemoryContext(
        platform="qq", user_id=self.key.actor_id, space_type="group", space_id=self.key.space_id,
    )
    self.items = []
    self.submitted = False
    self.version = 0
    self.batch_id = "s66-fixture"
    self.batch_url = "https://keytao.rea.ink/batch/s66-fixture"
    self.trace = []

async def TranscriptTests_delivered_turn(self, message):
    token = observability.begin_turn_metrics("qq", "group")
    try:
        raw_response = response = await self.turn(message)
        response = s63_fresh_selection_chat._enforce_advertised_reply_contract(response, self.key)
        response = s63_fresh_selection_chat._prepare_user_facing_reply(response, self.memory)
        metrics = observability.current_turn_metrics()
        self.trace.append({"message": message, "flow": metrics.flow,
                           "modelCalls": metrics.model_calls, "rawReply": raw_response,
                           "reply": response})
        return response
    finally:
        observability.end_turn_metrics(token)

async def TranscriptTests_tool(self, name, args, platform, user_id, **kwargs):
    self.calls.append((name, copy.deepcopy(args)))
    if name == "keytao_pending_items_by_words":
        result = {"success": True, "complete": True, "items": []}
    elif name == "keytao_lookup_by_word":
        result = {"success": True, "phrases": []}
    elif name == "keytao_prepare_reviewed_add":
        result = await prepared_review(args["word"], args.get("requested_reading", ""))
    elif name == "keytao_batch_add_to_draft":
        capabilities = kwargs["trusted_reviewed_items_by_key"]
        self.assertEqual(set(capabilities), {(word, codes[0]) for word, codes in zip(s66_readings_WORDS, s66_readings_CODES)})
        for word, codes, reading in zip(s66_readings_WORDS, s66_readings_CODES, READINGS):
            self.assertEqual(capabilities[(word, codes[0])]["pinyin"], reading)
        if not args.get("confirmed"):
            self.assertTrue(args["preview_only"])
            result = {"success": False, "requiresConfirmation": True,
                      "warningDigest": "a" * 64, "warnings": []}
        else:
            self.assertEqual(args["expected_content_version"], self.version)
            self.assertEqual(args["expected_warning_digest"], "a" * 64)
            self.assertFalse(self.writes)
            self.writes = [{"id": 6600 + index, **copy.deepcopy(row)}
                           for index, row in enumerate(args["items"])]
            self.items = copy.deepcopy(self.writes)
            self.version += 1
            result = {"success": True, "writtenItems": self.writes,
                      "successCount": 2, "failedCount": 0}
    elif name in {"keytao_list_draft_items", "keytao_get_batch_preview"}:
        result = {"success": True, "items": self.items, "count": len(self.items), "status": "Draft"}
    elif name == "keytao_submit_batch":
        if not args.get("confirmed"):
            result = {"success": False, "requiresConfirmation": True,
                      "snapshotDigest": "b" * 64, "warningDigest": "c" * 64,
                      "auditDigest": "d" * 64, "snapshotItems": self.items, "warnings": []}
        else:
            self.assertEqual(args["expected_content_version"], self.version)
            self.assertEqual(args["expected_server_snapshot_digest"], "b" * 64)
            self.assertEqual(args["expected_warning_digest"], "c" * 64)
            self.assertEqual(args["expected_audit_digest"], "d" * 64)
            self.assertFalse(self.submitted)
            self.submitted = True
            result = {"success": True, "status": "Submitted"}
    else:
        raise AssertionError("Unexpected tool: " + name)
    result.update(batchId=self.batch_id, batchUrl=self.batch_url, contentVersion=self.version)
    if kwargs.get("_capture_receipt", True):
        b17_rest_recommended_commands._capture_successful_draft_write_delivery(name, result, platform, user_id)
    return json.dumps(result, ensure_ascii=False)

@contextmanager
def TranscriptTests_executor_runtime(self):
    async def transport(name, args, context):
        self.assertEqual((context.platform, context.user_id), ("qq", self.key.actor_id))
        return json.loads(await self.tool(
            name, args, context.platform, context.user_id,
            trusted_reviewed_items_by_key=context.trusted_reviewed_items_by_key,
            _capture_receipt=False,
        ))

    with self.runtime(), patch.object(s63_fresh_selection_chat, "call_tool_function", REAL_TOOL_DISPATCH), patch.object(
        b17_rest_recommended_commands.tool_executor, "_invoke_effective_tool", side_effect=transport,
    ), patch.object(b17_rest_recommended_commands.memory_store, "record_tool_receipt"):
        yield

# Source: test_s67_ranking.py
from keytao_bot.harness.authorization_grammar import strip_self_mention_tokens as strip_self_mention_tokens
from keytao_bot.utils.bcc_reference import format_frequency as format_frequency
MESSAGES = ("喵喵 词频排序：咽 嘢 强 鎗", "喵喵 词频排序：之乎 鎗 敲不死")

COMMAND = "把 咽 yeoiav 排到 嘢 yeoiav 前面"

def S67RankingTests_setUp(self):
    BccCommonnessTests_setUp(self)
    fixture = json.loads((Path(__file__).parents[1] / "e2e/fixtures/bcc/s67.json").read_text())
    with closing(sqlite3.connect(self.db)) as connection, connection:
        for table, rows in (("bcc_frequency", fixture["rows"]), ("word_commonness", fixture["jieba"])):
            for row in rows:
                connection.execute(f'INSERT INTO {table} ({",".join(row)}) VALUES ({",".join("?" for _ in row)})', tuple(row.values()))
    commonness_query.reset_commonness_delivery()
    self.addCleanup(commonness_query.reset_commonness_delivery)
    self.owner = harness.ConversationAddress.private("qq", "s67-fixture")

async def S67RankingTests_reply(self, message=MESSAGES[0], *, weights=(10, 11), entries=None):
    message = strip_self_mention_tokens(message)
    words = commonness_query.parse_commonness_query(message)
    rows = entries if entries is not None else [
        {"word": word, "code": "yeoiav", "type": "Single", "weight": weight}
        for word, weight in zip(("咽", "嘢"), weights)
    ]
    calls = []

    async def dispatch(name, args, *actor):
        calls.append(name)
        self.assertEqual(actor, ("qq", self.owner.actor_id))
        if name == "keytao_lookup_by_words_batch":
            self.assertEqual(args, {"words": list(words)})
            return json.dumps({"success": True, "results": [
                {"word": word, "phrases": [row for row in rows if row["word"] == word]}
                for word in words
            ]})
        if name == "keytao_encode":
            return json.dumps({"success": True, "word": args["word"], "codes": []})
        raise AssertionError("Unexpected external or mutation tool: " + name)

    chat = harness.openai_chat_module
    memory = SimpleNamespace(conversation_address=self.owner, platform="qq")
    ctx = SimpleNamespace(normalized_message_text=message, platform="qq", user_id=self.owner.actor_id,
                          conv_key=self.owner, memory_context=memory, bot=object(), event=object(), QQMessageSegment=None)
    delivered = []

    async def finish(_bot, _event, _user, context, response, _segment):
        delivered.append(chat._prepare_user_facing_reply(response, context))

    token = chat._current_turn_message.set(message)
    try:
        with (patch.object(harness.chat_commands_module, "call_tool_function", side_effect=dispatch),
              patch.object(chat, "remember_conversation"),
              patch.object(chat, "_finish_ai_chat_response", side_effect=finish),
              patch.object(chat, "get_ai_response_core", side_effect=AssertionError("Model forbidden")),
              patch.object(chat, "_classify_message_command_intent", side_effect=AssertionError("Model forbidden"))):
            self.assertTrue(await chat._stage_handle_commonness_query(ctx))
    finally:
        chat._current_turn_message.reset(token)
    self.assertEqual(len(delivered), 1)
    self.assertTrue(commonness_query.matches_commonness_delivery(delivered[0], self.owner, message))
    self.web.assert_not_called()
    return delivered[0]

def S67RankingTests_cells(self, text, word):
    return next(line.split(" | ") for line in text.splitlines() if f" | {word} | " in line)

# Source: test_s68_selection.py
from keytao_bot.utils.pending_confirmation import parse_pending_candidate_selection as parse_pending_candidate_selection
s68_selection_FIXTURE = {
    "success": True, "word": "乐句", "type": "Phrase",
    "recommendedCode": "yhjluu", "needsManualReview": True,
    "preSubmitAudit": {"autoApprove": False, "summary": "Fixture review requires manual approval"},
    # Ratios are supplied incident evidence; the alternate chain is synthetic.
    "candidateOrderingAssessments": [
        {"newWord": "乐句", "occupantWord": occupant, "occupantCode": code,
         "freeCode": "yhjluu", "verdict": "behind_more_common",
         "summary": f"BCC：「{occupant}」的词频为「乐句」的 {ratio} 倍"}
        for occupant, code, ratio in (("越剧", "yhjl", "11.25"), ("粤剧", "yhjlu", "6.28"))
    ],
    "pronunciations": [
        {"pinyin": "yuè jù", "recommendedCode": "yhjluu", "candidateStatuses": [
            {"code": "yhjl", "occupied": True, "words": ["越剧"]},
            {"code": "yhjlu", "occupied": True, "words": ["粤剧"]},
            {"code": "yhjluu", "occupied": False, "words": []},
        ]},
        {"pinyin": "lè jù", "recommendedCode": "lejluu", "candidateStatuses": [
            {"code": code, "occupied": False, "words": []}
            for code in ("lejl", "lejlu", "lejluu")
        ]},
    ],
}

def SelectionTests_setUp(self):
    FreshSelectionTests_setUp(self)
    self.key = harness.ConversationAddress.group("qq", "865189947", "s68-offline")
    self.memory = harness.ChatMemoryContext(
        platform="qq", user_id=self.key.actor_id, space_type="group", space_id=self.key.space_id,
    )
    self.trace = []

async def SelectionTests_tool(self, name, args, platform, user_id, **kwargs):
    self.assertEqual((platform, user_id), ("qq", self.key.actor_id))
    self.calls.append((name, copy.deepcopy(args)))
    if name == "keytao_prepare_reviewed_add":
        result = copy.deepcopy(s68_selection_FIXTURE)
        if args["word"] == "小端":
            result.update(word="小端", recommendedCode="xcdti", pronunciations=[{
                "pinyin": "xiǎo duān", "recommendedCode": "xcdti", "candidateStatuses": [
                    {"code": "xcdti", "occupied": False, "words": []},
                ],
            }])
    elif name == "keytao_lookup_by_word":
        result = {"success": True, "phrases": []}
    elif name == "keytao_pending_items_by_words":
        result = {"success": True, "complete": True, "items": []}
    elif name in {"keytao_create_phrase", "keytao_batch_add_to_draft"}:
        items = args.get("items", [{"word": args.get("word"), "code": args.get("code")}])
        if not args.get("confirmed"):
            self.assertIs(args.get("preview_only"), True)
            result = {"success": False, "requiresConfirmation": True,
                      "warningDigest": "a" * 64, "warnings": []}
        else:
            self.assertEqual(args["expected_content_version"], 0)
            self.assertEqual(args["expected_warning_digest"], "a" * 64)
            self.assertFalse(self.writes)
            self.writes.extend(copy.deepcopy(items))
            result = {"success": True, "writtenItems": self.writes,
                      "successCount": len(items), "failedCount": 0}
    elif name in {"keytao_get_batch_preview", "keytao_list_draft_items"}:
        result = {"success": True, "status": "Draft", "items": self.writes, "count": len(self.writes)}
    else:
        raise AssertionError("Unexpected tool: " + name)
    result.update(batchId="s68-fixture", contentVersion=int(bool(self.writes)))
    return json.dumps(result, ensure_ascii=False)

async def SelectionTests_prepare(self, message="@喵喵 乐句"):
    # T1 intent is a fixture; T2 uses the real production classifier.
    with patch.object(s63_fresh_selection_chat, "_classify_message_command_intent", AsyncMock(return_value=harness.MessageCommandIntent())), patch.object(
        s63_fresh_selection_chat, "_get_simple_word_query_words", AsyncMock(return_value=("乐句",)),
    ):
        return await self.delivered_turn(message)

# Source: test_s69_fly_ticket.py
from keytao_bot.utils.explicit_code import validate_explicit_code as validate_explicit_code
s69_fly_ticket_commands = harness.chat_commands_module

s69_fly_ticket_chat = harness.openai_chat_module

ENCODE = {
    "success": True, "input": "折煞", "codes": ["eees", "eeesi", "eeesiu"],
    "altCodes": [], "flyKeyVariants": [], "phrasePinyins": ["shé", "shā"],
    "pronunciationSource": "pinyin-pro-context", "standardPronunciationStatus": "absent",
    "chars": [
        {"char": "折", "pinyin": "shé", "pinyins": ["shé", "zhé", "zhē"],
         "phoneticCode": "ee", "shapeCode": "iu", "pronunciationLookupStatus": "found"},
        {"char": "煞", "pinyin": "shā", "pinyins": ["shā", "shà"],
         "phoneticCode": "es", "shapeCode": "uu", "pronunciationLookupStatus": "found"},
    ],
    "requestedCodeAnalysis": {"code": "fees", "supported": False, "matchType": "unsupported"},
}

async def prepare_fly_review(existing=False):
    groups = [{"pinyin": reading, "normalized": plain.split(), "sources": [], "score": 0,
               "requiresManualReview": True}
              for reading, plain in (("shé shā", "she sha"), ("zhé shā", "zhe sha"))]
    with patch.object(review, "collect_pronunciation_evidence_limited", AsyncMock(return_value={
        "success": True, "groups": groups, "sources": [], "lookupComplete": True,
    })), patch.object(review, "fetch_keytao_encode", AsyncMock(return_value=copy.deepcopy(ENCODE))), patch.object(
        review, "lookup_words", AsyncMock(return_value={"折煞": [{"word": "折煞", "code": "qees", "type": "Phrase"}]} if existing else {}),
    ), patch.object(review, "lookup_codes", AsyncMock(return_value={
        "eees": [{"word": "射杀", "code": "eees", "type": "Phrase"}],
        "qees": [{"word": "折煞" if existing else "折杀", "code": "qees", "type": "Phrase"}],
    })), patch.object(review, "_resolve_multi_sense_pronunciation_choice", AsyncMock(return_value=(groups, {"status": "resolved"}))), patch.object(
        review, "AsyncOpenAI", side_effect=AssertionError("Model calls forbidden"),
    ):
        return await review.prepare_reviewed_word(None, "折煞")

def FlyTranscriptTests_setUp(self):
    FreshSelectionTests_setUp(self)
    self.trace = []

async def FlyTranscriptTests_tool(self, name, args, platform, user_id, **kwargs):
    self.calls.append((name, copy.deepcopy(args)))
    if name == "keytao_prepare_reviewed_add":
        return json.dumps(self.review)
    if name == "keytao_lookup_by_word":
        return json.dumps({"success": True, "phrases": []})
    if name == "keytao_pending_items_by_words":
        return json.dumps({"success": True, "complete": True, "items": []})
    raise AssertionError("Unexpected tool: " + name)

def LiveTicketTests_setUp(self):
    FreshSelectionTests_setUp(self)
    self.trace = []
    self.submitted = False
    pairs = [("qngf", True), ("qngfv", False), ("qngfvv", False)]
    self.original = harness.PendingAddWord(
        word="震感", recommended_code="qngf", candidates=pairs,
        server_candidates=pairs, occupied_words={"qngf": ["真敢"]},
        server_occupied_words={"qngf": ["真敢"]},
        pronunciation_codes={code: "zhèn gǎn" for code, _ in pairs},
        needs_manual_review=True,
        server_ordering_assessments=[{
            "newWord": "震感", "occupantWord": "真敢", "occupantCode": "qngf",
            "newCode": "qngf", "freeCode": "qngfv", "verdict": "front_more_common",
            "summary": "Fixture commonness supports front insertion",
        }],
    )
    self.store.set(self.key, self.original)

async def LiveTicketTests_tool(self, name, args, platform, user_id, **kwargs):
    self.calls.append((name, copy.deepcopy(args)))
    result = {"batchId": "s69-fixture", "contentVersion": 0, "warningDigest": "a" * 64}
    if name == "keytao_shift_phrase_code":
        self.assertFalse(args.get("confirmed_plan_digest"), "Old reorder plan executed")
        result.update(success=False, requiresConfirmation=True, planDigest="b" * 64,
                      shiftPlan={"word": "震感", "targetCode": "qngf", "items": [
                          {"word": "震感", "code": "qngf", "type": "Phrase", "action": "Create"},
                          {"word": "真敢", "code": "qngf", "type": "Phrase", "action": "Delete"},
                          {"word": "真敢", "code": "qngfu", "type": "Phrase", "action": "Create"},
                      ], "shifted": [{"word": "真敢", "fromCode": "qngf", "toCode": "qngfu"}]})
    elif name == "keytao_create_phrase":
        self.assertEqual((args["word"], args["code"]), ("震感", "qngfv"))
        if args.get("confirmed"):
            self.assertEqual(args["expected_warning_digest"], "a" * 64)
            self.writes.append({"id": 69, "action": "Create", "word": "震感", "code": "qngfv", "type": "Phrase"})
            result.update(success=True, writtenItems=copy.deepcopy(self.writes), contentVersion=1)
        else:
            self.assertIs(args.get("preview_only"), True)
            result.update(success=False, requiresConfirmation=True, warnings=[])
    elif name in {"keytao_list_draft_items", "keytao_get_batch_preview"}:
        result.update(success=True, items=self.writes, count=len(self.writes), status="Draft")
    elif name == "keytao_submit_batch":
        if args.get("confirmed"):
            self.submitted = True
            result.update(success=True, status="Submitted")
        else:
            result.update(success=False, requiresConfirmation=True, snapshotDigest="c" * 64,
                          auditDigest="d" * 64, snapshotItems=self.writes, warnings=[])
    else:
        raise AssertionError("Unexpected tool: " + name)
    return json.dumps(result, ensure_ascii=False)

async def LiveTicketTests_prepare_ticket(self):
    result = await s69_fly_ticket_commands.handle_pending_message_core("加入并提交", "qq", self.key.actor_id, self.key, allow_intent_model=False)
    self.assertIn("真敢", result)
    self.assertTrue(server_warning_ticket_is_complete(self.store.get(self.key)))
    return result

# Source: test_s70_smalltalk.py
from keytao_bot.utils.conversational_reply import enforce_conversational_reply as enforce_conversational_reply
s70_smalltalk_chat = harness.openai_chat_module

REAL_CORE = s70_smalltalk_chat.get_ai_response_core

NARRATION = (
    "用户发的这条消息只是「早」，是在打招呼，没有任何词库操作请求，"
    "也没有对应的可执行命令。\n不需要补充信息，等有具体需求再说即可。"
)

def SmallTalkTests_setUp(self):
    FreshSelectionTests_setUp(self)
    self.history = []

async def SmallTalkTests_tool(self, name, args, platform, user_id, **kwargs):
    if name == "keytao_lookup_by_words_batch":
        self.calls.append((name, args))
        return json.dumps({"success": True, "results": []})
    return await FreshSelectionTests_tool(self, name, args, platform, user_id, **kwargs)

async def SmallTalkTests_delivered(self, message, raw, *, inject_boundary=False, history=None, has_operation_intent=False):
    async def fake_model(text, *_args, **_kwargs):
        from tests.test_routing_regressions import GeneralReplyTests
        return await GeneralReplyTests().reply(text, raw)

    async def fake_intent(text, *_args):
        return s70_smalltalk_chat._chat_routing.parse_draft_view_command(text) or harness.MessageCommandIntent(
            has_operation_intent=has_operation_intent,
        )

    token = s70_smalltalk_chat._current_turn_message.set(message)
    conversation_token = s70_smalltalk_chat._current_conversational_turn.set(None)
    try:
        with self.runtime(), patch.object(s70_smalltalk_chat, "get_ai_response_core", side_effect=fake_model), patch.object(
            s70_smalltalk_chat, "_classify_message_command_intent", side_effect=fake_intent,
        ), patch.object(
            s70_smalltalk_chat, "_classify_simple_word_query_intent", AsyncMock(return_value=harness.SimpleWordQueryIntent(False)),
        ), patch.object(s70_smalltalk_chat, "get_history", return_value=history or []):
            cache, classifier = s70_smalltalk_chat.command_intent_memoizer(message, message)
            ctx = s70_smalltalk_chat.TurnContext(
                bot=object(), event=object(), platform="qq", user_id=self.key.actor_id,
                message_text=message, normalized_message_text=message, conv_key=self.key,
                space_key=self.key.space_key, memory_context=self.memory, history=history or [],
                reply_reference=harness.ReplyReferenceInfo(), command_intent_cache=cache,
                command_intent_for=classifier,
            )
            await s70_smalltalk_chat._stage_normalize_message_text(ctx)
            start = s70_smalltalk_chat.STAGES.index(s70_smalltalk_chat._stage_resolve_current_pending_scope)
            end = s70_smalltalk_chat.STAGES.index(s70_smalltalk_chat._stage_enforce_advertised_reply_contract)
            for stage in s70_smalltalk_chat.STAGES[start:end + 1]:
                if await stage(ctx):
                    break
            if inject_boundary:
                ctx.response = raw
            response = s70_smalltalk_chat._prepare_user_facing_reply(ctx.response, self.memory)
            self.assertEqual(response, s70_smalltalk_chat._prepare_user_facing_reply(response, self.memory))
            return response
    finally:
        s70_smalltalk_chat._current_conversational_turn.reset(conversation_token)
        s70_smalltalk_chat._current_turn_message.reset(token)

def SmallTalkTests_assert_natural(self, reply):
    self.assertTrue(reply)
    self.assertNotIn("\n", reply)
    self.assertLess(len(reply), 80)
    for text in ("用户", "命令", "候选", "草稿", "证据", "可发送", "未写入", "无需", "不需要", "具体需求"):
        self.assertNotIn(text, reply)

# Source: test_s70b_query_precedence.py
async def QueryPrecedenceTests_tool(self, name, args, platform, user_id, **kwargs):
    if name == "keytao_lookup_by_word":
        self.calls.append((name, args))
        return json.dumps({"success": True, "phrases": [{
            "word": args["word"], "code": "abcd", "type": "Phrase",
        }] if not getattr(self, "missing_word", False) else []})
    return await SmallTalkTests_tool(self, name, args, platform, user_id, **kwargs)

async def QueryPrecedenceTests_delivered(self, message, raw=NARRATION):
    client = harness._FakeClient([harness._FakeAIResponse("stop", raw)])
    token = s70_smalltalk_chat._current_turn_message.set(message)
    conversation_token = s70_smalltalk_chat._current_conversational_turn.set(None)
    try:
        with self.runtime(), patch.object(
            s70_smalltalk_chat, "get_ai_response_core", side_effect=REAL_CORE,
        ), patch.object(s70_smalltalk_chat, "OPENAI_API_KEY", "s70b-fixture"), patch.object(
            s70_smalltalk_chat, "AsyncOpenAI", return_value=client,
        ), patch.object(s70_smalltalk_chat, "skills_manager", harness._FakeToolSkillsManager()), patch.object(
            s70_smalltalk_chat, "_classify_message_command_intent", AsyncMock(return_value=harness.MessageCommandIntent(
                has_operation_intent=False,
            )),
        ) as classifier, patch.object(
            s70_smalltalk_chat, "_classify_simple_word_query_intent", AsyncMock(return_value=harness.SimpleWordQueryIntent(False)),
        ):
            ctx = s70_smalltalk_chat.TurnContext(
                bot=object(), event=object(), platform="qq", user_id=self.key.actor_id,
                message_text=message, normalized_message_text=message, conv_key=self.key,
                space_key=self.key.space_key, memory_context=self.memory, history=[],
                reply_reference=harness.ReplyReferenceInfo(),
            )
            await s70_smalltalk_chat._stage_normalize_message_text(ctx)
            ctx.command_intent_cache, ctx.command_intent_for = s70_smalltalk_chat.command_intent_memoizer(
                message, ctx.normalized_message_text,
            )
            start = s70_smalltalk_chat.STAGES.index(s70_smalltalk_chat._stage_resolve_current_pending_scope)
            end = s70_smalltalk_chat.STAGES.index(s70_smalltalk_chat._stage_enforce_advertised_reply_contract)
            for stage in s70_smalltalk_chat.STAGES[start:end + 1]:
                if await stage(ctx):
                    break
            response = s70_smalltalk_chat._prepare_user_facing_reply(ctx.response, self.memory)
            self.assertGreater(classifier.await_count, 0)
            return response, client.completions.calls
    finally:
        s70_smalltalk_chat._current_conversational_turn.reset(conversation_token)
        s70_smalltalk_chat._current_turn_message.reset(token)

# Source: test_s71_stale_ticket_release.py
from keytao_bot.utils.observability import current_turn_metrics as current_turn_metrics
s71_stale_ticket_release_chat = harness.openai_chat_module

s71_stale_ticket_release_commands = harness.chat_commands_module

QUOTE = "是啊，欲买桂花同载酒，终不似少年游，好句子喵～"

MODEL_REPLY = "Fixture fresh request reached."

REVIEW_REPLY = "Fixture reviewed add reached."

REVIEWS = {
    word: {
        "success": True, "word": word, "type": "Phrase",
        "recommendedCode": code, "needsManualReview": True,
        "manualReviewReason": "Fixture manual review required",
        "pronunciations": [{
            "pinyin": pinyin, "recommendedCode": code,
            "candidateStatuses": [
                {"code": candidate, "occupied": False, "words": []}
                for candidate in candidates
            ],
        }],
    }
    for word, code, candidates, pinyin in (
        ("你都没有上手", "ndme", ("ndme", "ndmev", "ndmevo"), "nǐ dōu méi yǒu shàng shǒu"),
        ("有点不切实际", "ydbjvi", ("ydbj", "ydbjv", "ydbjvi"), "yǒu diǎn bù qiè shí jì"),
    )
}

async def StaleTicketReleaseTests_replay(self, message, quote="", *, live=True, model_failure=False):
    store = harness.MemoryConversationStateStore()
    memory = harness.ChatMemoryContext(
        platform="qq", user_id="s71-owner", space_type="group", space_id="s71-group",
    )
    key = memory.conversation_address
    stages, delivered, calls = [], [], []
    classifier = AsyncMock(return_value=harness.MessageCommandIntent())
    word_classifier = AsyncMock(return_value=harness.SimpleWordQueryIntent(False))
    model = AsyncMock(
        side_effect=s71_stale_ticket_release_chat.get_ai_response_core if model_failure else None,
        return_value=MODEL_REPLY,
    )
    completion = AsyncMock(return_value=SimpleNamespace(
        model="fixture-model", usage={}, choices=[SimpleNamespace(
            finish_reason="length", message=SimpleNamespace(
                content="", tool_calls=[], reasoning_content="",
            ),
        )],
    ))
    client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=completion),
    ))
    fallback_pending = AsyncMock(wraps=s71_stale_ticket_release_chat.handle_pending_message_core)
    execute = AsyncMock(return_value="Fixture ticket executed.")
    submit = AsyncMock(return_value=harness.DraftActionResult("Fixture ticket submitted."))
    schedule = AsyncMock(side_effect=AssertionError("Background work forbidden"))

    async def tool(name, args, platform, user_id, **_kwargs):
        self.assertEqual((platform, user_id), ("qq", key.actor_id))
        calls.append((name, copy.deepcopy(args)))
        if name == "keytao_lookup_by_word":
            return json.dumps({"success": True, "phrases": []})
        if name == "keytao_lookup_by_words_batch":
            return json.dumps({"success": True, "results": []})
        if name == "keytao_pending_items_by_words":
            return json.dumps({"success": True, "complete": True, "items": []})
        if name == "keytao_prepare_reviewed_add":
            if args["word"] in REVIEWS:
                return json.dumps(REVIEWS[args["word"]])
            return json.dumps({"success": False, "message": REVIEW_REPLY})
        raise AssertionError("Unexpected external tool: " + name)

    async def finish(*args, **_kwargs):
        text = args[4] if len(args) >= 5 else args[0]
        delivered.append(s71_stale_ticket_release_chat._enforce_advertised_reply_contract(text, key))

    token = harness.begin_turn_metrics("qq", "group")
    try:
        with ExitStack() as stack:
            for patcher in (
                patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")),
                patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Network forbidden")),
                patch.object(socket, "create_connection", side_effect=AssertionError("Network forbidden")),
                patch.object(socket, "getaddrinfo", side_effect=AssertionError("DNS forbidden")),
                patch.object(s71_stale_ticket_release_chat, "AsyncOpenAI", side_effect=AssertionError("Real model forbidden")),
                patch.object(s71_stale_ticket_release_chat, "conversation_state_store", store),
                patch.object(s71_stale_ticket_release_chat, "draft_operation_coordinator", harness.DraftOperationCoordinator()),
                patch.object(s71_stale_ticket_release_chat, "extract_memory_context", AsyncMock(return_value=memory)),
                patch.object(s71_stale_ticket_release_chat, "get_history", return_value=[]),
                patch.object(s71_stale_ticket_release_chat, "remember_conversation"),
                patch.object(s71_stale_ticket_release_chat, "schedule_memory_compaction"),
                patch.object(s71_stale_ticket_release_chat, "_classify_message_command_intent", classifier),
                patch.object(s71_stale_ticket_release_chat, "_classify_simple_word_query_intent", word_classifier),
                patch.object(s71_stale_ticket_release_chat, "get_ai_response_core", model),
                patch.object(s71_stale_ticket_release_chat, "call_tool_function", side_effect=tool),
                patch.object(s71_stale_ticket_release_commands, "_execute_confirmed_tool", execute),
                patch.object(s71_stale_ticket_release_commands, "_perform_batch_add_to_draft_and_submit", submit),
                patch.object(s71_stale_ticket_release_chat, "_schedule_background_draft_operation", schedule),
                patch.object(s71_stale_ticket_release_chat, "_finish_ai_chat_response", side_effect=finish),
                patch.object(s71_stale_ticket_release_chat, "_finish_ai_chat_matcher", side_effect=finish),
            ):
                stack.enter_context(patcher)
            initial = s71_stale_ticket_release_chat.TurnContext(
                bot=object(), event=object(), platform="qq", user_id=key.actor_id,
                message_text="你都没有上手 有点不切实际",
                reply_reference=harness.ReplyReferenceInfo(),
            )
            for stage in s71_stale_ticket_release_chat.STAGES[s71_stale_ticket_release_chat.STAGES.index(s71_stale_ticket_release_chat._stage_normalize_message_text):]:
                if await stage(initial):
                    break
            self.assertIsInstance(initial.response, s71_stale_ticket_release_chat.ServerBackedQueryReply)
            self.assertEqual(len(delivered), 1)
            record = store.get_record(key)
            self.assertIsInstance(record.state, harness.PendingToolConfirm)
            self.assertTrue(record.state.args["_reviewed_multi_word"])
            self.assertFalse(record.origin_prompt_digest)
            old_render = delivered[0]
            self.assertTrue(s71_stale_ticket_release_chat._advertised_reply_matches_live_record(old_render, record))
            self.assertEqual(old_render, s71_stale_ticket_release_chat._render_live_batch_record(record))
            self.assertEqual(
                [args["word"] for name, args in calls if name == "keytao_prepare_reviewed_add"],
                list(REVIEWS),
            )
            model.assert_not_awaited()
            execute.assert_not_awaited()
            submit.assert_not_awaited()
            schedule.assert_not_called()
            before = copy.deepcopy(record)
            own_quote = quote == "own-command"
            if quote == "ticket":
                quote = old_render
            elif own_quote:
                quote = "加词 你都没有上手 有点不切实际"
            if not live:
                store.delete(key)
            ctx = s71_stale_ticket_release_chat.TurnContext(
                bot=object(), event=object(), platform="qq", user_id=key.actor_id,
                message_text=message, reply_reference=harness.ReplyReferenceInfo(
                    is_reply=bool(quote), is_to_bot=bool(quote) and not own_quote,
                    sender_id=key.actor_id if own_quote else "s71-bot", text=quote,
                ),
            )
            delivered.clear()
            calls.clear()
            harness.end_turn_metrics(token)
            token = harness.begin_turn_metrics("qq", "group")
            if model_failure:
                for patcher in (
                    patch.object(s71_stale_ticket_release_chat, "AsyncOpenAI", return_value=client),
                    patch.object(s71_stale_ticket_release_chat, "OPENAI_API_KEY", "fixture-key"),
                    patch.object(s71_stale_ticket_release_chat.memory_store, "get_context_block", return_value=""),
                    patch.object(s71_stale_ticket_release_chat, "get_group_history_context", return_value=""),
                    patch.object(s71_stale_ticket_release_chat, "skills_manager", SimpleNamespace(
                        get_skill_instructions=lambda: "", has_tools=lambda: False,
                        get_tools=lambda: [], get_tool_function=lambda _name: None,
                    )),
                    patch.object(s71_stale_ticket_release_chat, "handle_pending_message_core", fallback_pending),
                ):
                    stack.enter_context(patcher)
            for stage in s71_stale_ticket_release_chat.STAGES[s71_stale_ticket_release_chat.STAGES.index(s71_stale_ticket_release_chat._stage_normalize_message_text):]:
                stages.append(stage.__name__)
                if await stage(ctx):
                    break
            flow = current_turn_metrics().flow
    finally:
        harness.end_turn_metrics(token)
    return dict(ctx=ctx, stages=stages, delivered=delivered, calls=calls, flow=flow,
                record=store.get_record(key), before=before, old_render=old_render,
                model=model, classifier=classifier, execute=execute, submit=submit,
                schedule=schedule, quote=quote, completion=completion,
                fallback_pending=fallback_pending)

async def StaleTicketReleaseTests_assert_fresh_request(self, message, quote="", *, reviewed_word=""):
    result = await self.replay(message, quote)
    self.assertNotEqual(result["flow"], "pending-confirmation")
    self.assertNotEqual(result["stages"][-1], "_stage_finish_scoped_pending_response")
    self.assertTrue(result["ctx"].generic_intent_is_fresh_command)
    self.assertEqual(result["record"], result["before"])
    self.assertFalse(result["record"].execution_id)
    result["execute"].assert_not_awaited()
    result["submit"].assert_not_awaited()
    result["schedule"].assert_not_called()
    self.assertNotIn(result["old_render"], result["delivered"])
    self.assertEqual(result["delivered"], [REVIEW_REPLY if reviewed_word else MODEL_REPLY])
    self.assertEqual(result["ctx"].reply_reference.text, result["quote"])
    self.assertFalse(result["ctx"].verified_current_pending_reply)
    if reviewed_word:
        self.assertIn(("keytao_prepare_reviewed_add", {"word": reviewed_word}), result["calls"])
        result["model"].assert_not_awaited()
    else:
        result["model"].assert_awaited_once()
        self.assertIn(result["quote"], result["model"].await_args.args[4])
        self.assertTrue(all(name == "keytao_lookup_by_words_batch" for name, _args in result["calls"]))
    # The same message without a live ticket reaches the same external boundary.
    baseline = await self.replay(message, quote, live=False)
    self.assertEqual(result["delivered"], baseline["delivered"])
    self.assertEqual(result["calls"], baseline["calls"])
    self.assertEqual(result["model"].await_count, baseline["model"].await_count)

# Source: test_s72_review_reason.py
BODY = "本次权威来源查询未完成（汉典（经编码服务）、汉典、百度百科、汉语辞典）"

INCIDENT = f"{BODY}，本轮仍需管理员审核"

WRAPPED = f"自动审核：该词需管理员审核（{INCIDENT}）"

PROBES = (
    (INCIDENT, BODY, BODY),
    (f"{BODY}，仍需管理员审核", BODY, BODY),
    (WRAPPED, BODY, WRAPPED),
    ("「田调」加词预审已标记为需管理员审核",) * 3,
    ("重码添加需管理员审核",) * 3,
    ("纯删除需要管理员确认",) * 3,
    (BODY,) * 3,
)

def capture_review(reason):
    reviewed = {}
    harness.AgentOrchestrator._update_trusted_capabilities(
        tool_name="keytao_prepare_reviewed_add",
        arguments={"word": "田调"},
        result={
            "success": True, "word": "田调", "recommendedCode": "tmtn",
            "preSubmitAudit": {"autoApprove": False, "issues": [reason]},
        },
        codes_by_word={}, word_lookup_codes_by_word={}, entries_by_code={},
        draft_words_by_id={}, draft_items_by_id={}, phrase_types_by_key={},
        reviewed_items_by_key=reviewed, candidate_slots_by_word={},
    )
    return reviewed[("田调", "tmtn")]

# Source: test_s73_reading_rereview_persistence.py
s73_reading_rereview_persistence_chat = harness.openai_chat_module

s73_reading_rereview_persistence_commands = harness.chat_commands_module

OLD_CODES = ("tmtc", "tmtci", "tmtcio")

NEW_CODES = ("tmdc", "tmdci", "tmdcio")

REREVIEW = "读音是 tian diao，是田野调查的缩写，请重新编码"

REVIEW_ARGS = {
    "word": "田调", "requested_reading": "tian diao",
    "requested_meaning": "田野调查的缩写",
}

REPLIES = {
    "dash": "「田调」候选编码：\n1. tmdc — 空位（推荐）\n2. tmdci — 空位\n3. tmdcio — 空位\n回复「加入」写入草稿。",
    "colon": "「田调」候选编码：\n1. tmdc：空位（推荐）\n2. tmdci：空位\n3. tmdcio：空位\n回复「加入」写入草稿。",
    "bold": "「田调」候选编码：\n1. **tmdc**：空位（推荐）\n2. `tmdci`：空位\n3. **tmdcio**：空位\n回复「加入」写入草稿。",
    "prose": "「田调」已按新读音重新审核，回复「加入」写入草稿，或「加入并提交」写入并提交。",
}

def review_fixture(*, corrected=False, word="田调"):
    codes = NEW_CODES if corrected else OLD_CODES
    pinyin = "tián diào" if corrected else "tian tiao"
    recommended = codes[0] if corrected else codes[1]
    reason = "用户明确选择编码服务候选读音" if corrected else "本次权威来源查询未完成"
    # The incident log truncates before occupancy; model the new chain as empty.
    statuses = [
        {"code": code, "occupied": not corrected and index == 0,
         "words": ["天条"] if not corrected and index == 0 else []}
        for index, code in enumerate(codes)
    ]
    return {
        "success": True, "word": word, "type": "Phrase", "existing": [],
        "recommendedCode": recommended, "needsManualReview": True,
        "manualReviewReason": reason,
        "pronunciations": [{
            "pinyin": pinyin, "normalized": ["tian", "diao" if corrected else "tiao"],
            "codes": list(codes),
            "sources": [], "score": 0, "fallback": not corrected,
            "semanticPronunciation": corrected, "requiresManualReview": True,
            "sourceSummary": reason, "recommendedCode": recommended,
            "candidateStatuses": statuses,
        }],
        "preSubmitAudit": {"autoApprove": False, "issues": [reason]},
    }

async def ReadingRereviewPersistenceTests_asyncSetUp(self):
    self.store = harness.MemoryConversationStateStore()
    self.memory = harness.ChatMemoryContext(
        platform="qq", user_id="s73-owner", space_type="group", space_id="s73-group",
    )
    self.key = self.memory.conversation_address
    self.calls, self.writes, self.delivered = [], [], []
    self.review = review_fixture()
    self.reviews = {}
    self.client = harness._FakeClient([])
    self.schedule = AsyncMock(side_effect=AssertionError("Background work forbidden"))

    async def review_tool(word, platform, platform_id, **kwargs):
        return json.loads(await self.tool(
            "keytao_prepare_reviewed_add", {"word": word, **kwargs}, platform, platform_id,
        ))

    skills = SimpleNamespace(
        get_skill_instructions=lambda: "", has_tools=lambda: True,
        get_tools=lambda: [{"type": "function", "function": {
            "name": "keytao_prepare_reviewed_add", "description": "Review one word",
            "parameters": {"type": "object", "properties": {
                key: {"type": "string"} for key in REVIEW_ARGS
            }, "required": ["word"]},
        }}],
        get_tool_function=lambda name: review_tool if name == "keytao_prepare_reviewed_add" else None,
    )

    async def finish(*args, **kwargs):
        response = args[4] if len(args) >= 5 else args[0]
        self.delivered.append(s73_reading_rereview_persistence_chat._enforce_advertised_reply_contract(response, self.key))

    self.stack = ExitStack()
    self.addCleanup(self.stack.close)
    for patcher in (
        patch.object(s73_reading_rereview_persistence_chat, "conversation_state_store", self.store),
        patch.object(s73_reading_rereview_persistence_chat, "draft_operation_coordinator", harness.DraftOperationCoordinator()),
        patch.object(s73_reading_rereview_persistence_chat, "extract_memory_context", AsyncMock(side_effect=lambda *_: self.memory)),
        patch.object(s73_reading_rereview_persistence_chat, "get_history", return_value=[]),
        patch.object(s73_reading_rereview_persistence_chat, "remember_conversation"),
        patch.object(s73_reading_rereview_persistence_chat, "schedule_memory_compaction"),
        patch.object(s73_reading_rereview_persistence_chat.memory_store, "get_context_block", return_value=""),
        patch.object(s73_reading_rereview_persistence_chat, "get_group_history_context", return_value=""),
        patch.object(s73_reading_rereview_persistence_chat, "_classify_message_command_intent", AsyncMock(return_value=harness.MessageCommandIntent())),
        patch.object(s73_reading_rereview_persistence_chat, "_classify_simple_word_query_intent", AsyncMock(return_value=harness.SimpleWordQueryIntent(False))),
        patch.object(s73_reading_rereview_persistence_chat, "AsyncOpenAI", return_value=self.client),
        patch.object(s73_reading_rereview_persistence_chat, "OPENAI_API_KEY", "fixture-key"),
        patch.object(s73_reading_rereview_persistence_chat, "skills_manager", skills),
        patch.object(s73_reading_rereview_persistence_chat, "tool_executor", harness.ToolExecutor(
            skills.get_tool_function, frozenset({"keytao_prepare_reviewed_add"}),
        )),
        patch.object(s73_reading_rereview_persistence_chat, "call_tool_function", side_effect=self.tool),
        patch.object(s73_reading_rereview_persistence_commands.user_resolver, "resolve_actor_binding", AsyncMock(return_value=True)),
        patch.object(s73_reading_rereview_persistence_chat, "_schedule_background_draft_operation", self.schedule),
        patch.object(s73_reading_rereview_persistence_chat, "_finish_ai_chat_response", side_effect=finish),
        patch.object(s73_reading_rereview_persistence_chat, "_finish_ai_chat_matcher", side_effect=finish),
    ):
        self.stack.enter_context(patcher)

async def ReadingRereviewPersistenceTests_tool(self, name, args, platform, user_id, **kwargs):
    self.assertEqual((platform, user_id), ("qq", self.key.actor_id))
    self.calls.append((name, copy.deepcopy(args)))
    if name == "keytao_lookup_by_word":
        result = {"success": True, "phrases": []}
    elif name == "keytao_lookup_by_words_batch":
        result = {"success": True, "results": []}
    elif name == "keytao_pending_items_by_words":
        result = {"success": True, "complete": True, "items": []}
    elif name == "keytao_prepare_reviewed_add":
        result = self.reviews.get(args["word"], self.review)
    elif name == "keytao_create_phrase":
        self.writes.append((copy.deepcopy(args), copy.deepcopy(kwargs)))
        result = {"success": True, "word": args["word"], "code": args["code"],
                  "batchId": "s73-draft", "message": "已加入草稿"}
    elif name in {"keytao_get_batch_preview", "keytao_list_draft_items"}:
        result = {"success": True, "batchId": "s73-draft", "count": len(self.writes), "items": [
            {"word": args["word"], "code": args["code"], "action": "Create"}
            for args, _kwargs in self.writes
        ]}
    else:
        raise AssertionError("Unexpected external tool: " + name)
    return json.dumps(result, ensure_ascii=False)

async def ReadingRereviewPersistenceTests_turn(self, message, quote=""):
    ctx = s73_reading_rereview_persistence_chat.TurnContext(
        bot=object(), event=object(), platform="qq", user_id=self.key.actor_id,
        message_text=message, reply_reference=harness.ReplyReferenceInfo(
            is_reply=bool(quote), is_to_bot=bool(quote), sender_id="s73-bot", text=quote,
        ),
    )
    token = harness.begin_turn_metrics("qq", "group")
    self.stages = []
    try:
        for stage in s73_reading_rereview_persistence_chat.STAGES[s73_reading_rereview_persistence_chat.STAGES.index(s73_reading_rereview_persistence_chat._stage_normalize_message_text):]:
            self.stages.append(stage.__name__)
            if await stage(ctx):
                break
        self.flow = current_turn_metrics().flow
    finally:
        harness.end_turn_metrics(token)
    self.schedule.assert_not_called()
    return self.delivered[-1]

async def ReadingRereviewPersistenceTests_initial_review(self, word="田调"):
    self.review = review_fixture(word=word)
    text = await self.turn(word)
    self.assertEqual(self.flow, "word-discovery")
    record = self.store.get_record(self.key)
    self.assertIsInstance(record.state, harness.PendingAddWord)
    self.assertEqual(record.state.word, word)
    self.assertEqual(record.state.recommended_code, "tmtci")
    self.assertFalse(record.origin_prompt_digest)
    self.assertEqual(self.client.completions.calls, [])
    self.assertEqual(self.writes, [])
    return text

async def ReadingRereviewPersistenceTests_rereview(self, style, quote, message=REREVIEW, word="田调"):
    review_args = {**REVIEW_ARGS, "word": word}
    self.client.completions.responses.extend([
        harness._FakeAIResponse("tool_calls", "", [SimpleNamespace(
            id="s73-review", type="function", function=SimpleNamespace(
                name="keytao_prepare_reviewed_add", arguments=json.dumps(review_args),
            ),
        )]),
        harness._FakeAIResponse("stop", REPLIES[style].replace("田调", word)),
    ])
    create = self.client.completions.create

    async def complete(**kwargs):
        if len(self.client.completions.calls) == 1:
            self.before_final_reply = copy.deepcopy(self.store.get_record(self.key))
        return await create(**kwargs)

    with patch.object(self.client.completions, "create", side_effect=complete):
        text = await self.turn(message, quote)
    self.assertEqual(self.flow, "general")
    self.assertIn(("keytao_prepare_reviewed_add", review_args), self.calls,
                  (self.stages, text, self.client.completions.calls))
    self.assertEqual(len(self.client.completions.calls), 2)
    self.assertEqual(self.writes, [])
    return text

async def ReadingRereviewPersistenceTests_assert_incident(self, style):
    quote = await self.initial_review()
    self.review = review_fixture(corrected=True)
    text = await self.rereview(style, quote)
    self.assertIn("tián diào", text)
    for code in NEW_CODES:
        self.assertIn(code, text)
    for stale in (*OLD_CODES, "tian tiao"):
        self.assertNotIn(stale, text)
    pending = self.store.get(self.key)
    self.assertIsInstance(pending, harness.PendingAddWord)
    self.assertEqual(pending.server_candidates, [(code, False) for code in NEW_CODES])
    self.assertEqual(pending.pronunciation_codes, dict.fromkeys(NEW_CODES, "tián diào"))
    self.assertEqual(pending.recommended_code, "tmdc")
    record = self.store.get_record(self.key)
    print(f"S73 {style}: digest_matches_delivered="
          f"{record.origin_prompt_digest == s73_reading_rereview_persistence_commands._prompt_capability_digest(text)}")
    await self.turn("加入")
    self.assertEqual(len(self.writes), 1)
    args, kwargs = self.writes[0]
    self.assertEqual((args["word"], args["code"]), ("田调", "tmdc"))
    self.assertEqual(len(self.client.completions.calls), 2)
    self.assertEqual(
        kwargs["trusted_reviewed_items_by_key"][("田调", "tmdc")]["pinyin"], "tián diào",
    )

async def ReadingRereviewPersistenceTests_assert_new_word_dash(self, message):
    before = copy.deepcopy(self.store.get_record(self.key))
    self.review = review_fixture(corrected=True, word="天调")
    text = await self.rereview("dash", "", message=message, word="天调")
    self.assertEqual(self.before_final_reply, before)
    self.assertIn("天调", text)
    self.assertIn("tián diào", text)
    for code in NEW_CODES:
        self.assertIn(code, text)
    for stale in (*OLD_CODES, "田调", "地调", "tian tiao"):
        self.assertNotIn(stale, text)
    self.assertEqual(self.store.get(self.key).word, "天调")
    await self.turn("加入")
    self.assertEqual([(args["word"], args["code"]) for args, _ in self.writes],
                     [("天调", "tmdc")])

async def ReadingRereviewPersistenceTests_assert_batch_release_dash(self, message):
    self.reviews["地调"] = review_fixture(corrected=True, word="地调")
    await self.turn("田调 地调")
    record = self.store.get_record(self.key)
    self.assertIsInstance(record.state, harness.PendingToolConfirm)
    self.assertTrue(record.state.args["_reviewed_multi_word"])
    self.assertFalse(record.origin_prompt_digest)
    self.assertEqual(self.client.completions.calls, [])
    await self.assert_new_word_dash(message)


__all__ = ['ALTERNATE', 'ALTERNATE_CODES', 'AMBIGUOUS_SHIFT_OPERANDS', 'ANSWER', 'AddKeepsSubmitTests_answer', 'AddKeepsSubmitTests_offer', 'AddKeepsSubmitTests_preview', 'AddKeepsSubmitTests_setUp', 'AddKeepsSubmitTests_tool', 'AgentOrchestrator', 'AgentRequestContext', 'AgentRuntimeConfig', 'ArtifactRecorder', 'AsyncMock', 'B10FlyKeyTests_add_fly_key', 'B10FlyKeyTests_assert_pending_code_allows_add', 'B10FlyKeyTests_setUp', 'B11ChangeSourceTests_cleanup', 'B11ChangeSourceTests_setUp', 'B11ShiftTests_assert_shift', 'B11ShiftTests_seed_receipt', 'B11ShiftTests_setUp', 'B11ShiftTests_turn', 'B11SourceShiftTests_cleanup', 'B11SourceShiftTests_confirm', 'B11SourceShiftTests_preview', 'B11SourceShiftTests_setUp', 'B11SourceShiftTests_strict', 'B12SelectionTests_assert_offline_route', 'B12SelectionTests_candidates', 'B12SelectionTests_created', 'B12SelectionTests_setUp', 'B12SelectionTests_turn', 'B13DefaultReadingTests_setUp', 'B14MultiSelectionTests_assert_offline_selection', 'B14MultiSelectionTests_candidates', 'B14MultiSelectionTests_created', 'B14MultiSelectionTests_select', 'B14MultiSelectionTests_setUp', 'B15ChatTests_candidates', 'B15ChatTests_created', 'B15ChatTests_reset_ticket', 'B15ChatTests_setUp', 'B15ChatTests_submissions', 'B15ChatTests_turn', 'B15PendingLookupTests_batch', 'B15PendingLookupTests_lookup', 'B15PendingLookupTests_setUp', 'B16ItemThreeTests_model_submit', 'B16ItemTwoTests_setUp', 'B17RestRouteTests_assert_offline', 'B17RestRouteTests_candidates', 'B17RestRouteTests_check_failed_then_corrected', 'B17RestRouteTests_created', 'B17RestRouteTests_select', 'B17RestRouteTests_setUp', 'B17RestRouteTests_submissions', 'B18RealToolTests_created', 'B18RealToolTests_seed', 'B18RealToolTests_setUp', 'B18RealToolTests_tearDown', 'B18RealToolTests_turn', 'B18ShiftValidationTests_setUp', 'B19PartialReplyBindingTests_assert_refused', 'B19PartialReplyBindingTests_complete', 'B19PartialReplyBindingTests_created', 'B19PartialReplyBindingTests_matcher_finish', 'B19PartialReplyBindingTests_matcher_send', 'B19PartialReplyBindingTests_partial', 'B19PartialReplyBindingTests_seed', 'B19PartialReplyBindingTests_setUp', 'B19PartialReplyBindingTests_tearDown', 'B19PartialReplyBindingTests_turn', 'B20DisplayTests_assert_pending_link', 'B21DisplayTests_setUp', 'B22VoteTests_assert_alias', 'B22VoteTests_assert_vote', 'B22VoteTests_deliver', 'B22VoteTests_setUp', 'B22VoteTests_submit', 'B22VoteTests_submitted_pair', 'B23CommonnessTests_candidate_review', 'B23CommonnessTests_seed_historical_chain', 'B23CommonnessTests_setUp', 'B24DisplaceTests_finish_shift', 'B24DisplaceTests_seed', 'B24DisplaceTests_setUp', 'B24DisplaceTests_tearDown', 'B25SessionTests_finish_replacement', 'B25SessionTests_greeting', 'B25SessionTests_history_for', 'B25SessionTests_partial_session', 'B25SessionTests_remember', 'B25SessionTests_seed', 'B25SessionTests_setUp', 'B25SessionTests_tearDown', 'B25SessionTests_turn', 'B26DisplayTests_displacement_suggestions', 'B26DisplayTests_lookup_replacement_product', 'B27ReadingTests_finish_add', 'B27ReadingTests_modifier_adds', 'B27ReadingTests_reset_session', 'B27ReadingTests_seed', 'B27ReadingTests_setUp', 'B5DeleteTests', 'B5ReplySafetyTests_context', 'B5ReplySafetyTests_setUp', 'B6ReminderIntegrationTests_setUp', 'B6ReminderTests_scan', 'B6ReminderTests_setUp', 'B6ReminderTests_snapshot', 'B6ReminderTests_write', 'B6SubmitTests_invoke', 'B6SubmitTests_setUp', 'B6SubmitTests_submit', 'B7BlockedToolLoopTests_orchestrator', 'B7BlockedToolLoopTests_policy_message', 'B7BlockedToolLoopTests_run_loop', 'B7PhrasingRouteTests_answer', 'B7PhrasingRouteTests_assert_recodes', 'B7PhrasingRouteTests_asyncSetUp', 'B7PhrasingRouteTests_asyncTearDown', 'B7PhrasingRouteTests_call', 'B7PhrasingRouteTests_result', 'B7PhrasingRouteTests_review_word', 'B7PhrasingRouteTests_row', 'B7PhrasingRouteTests_stage', 'B7QueryRouteTests_assert_finished', 'B7QueryRouteTests_assert_preserved', 'B7QueryRouteTests_asyncSetUp', 'B7QueryRouteTests_asyncTearDown', 'B7QueryRouteTests_context', 'B7QueryRouteTests_lookup', 'B7QueryRouteTests_set_model', 'B7QueryRouteTests_ticket', 'B7SelectionTests_resolve', 'B8CancelTests_assert_unbound_reply', 'B8CancelTests_offer', 'B8CancelTests_replay', 'B8PresentWordsTests_add', 'B8PresentWordsTests_assert_candidates', 'B8PresentWordsTests_assert_present_copy', 'B8PresentWordsTests_asyncSetUp', 'B8PresentWordsTests_asyncTearDown', 'B8PresentWordsTests_lookup', 'B8PresentWordsTests_present', 'B9CandidateTests_asyncSetUp', 'B9CandidateTests_asyncTearDown', 'B9CandidateTests_lookup', 'B9CandidateTests_offer', 'B9LookupTests_asyncSetUp', 'B9LookupTests_prepare', 'B9LookupTests_row', 'B9LookupTests_tool', 'B9ParallelTests_prepare', 'B9ParallelTests_setUp', 'B9ParallelTests_tool', 'B9PronunciationEvidenceTests_setUp', 'B9PronunciationEvidenceTests_timeout_types', 'BATCH', 'BATCH_URL', 'BODY', 'BccCommonnessTests_setUp', 'BccDeliveryTests_run_stage', 'BccDeliveryTests_setUp', 'BccIngestTests_cached_corpus', 'BridgeTests_call', 'BridgeTests_setUp', 'CHAIN', 'CHANNELS', 'CODE', 'COMMAND', 'CORPUS_MODES', 'CREATES', 'CommonnessEvidenceTests_insert', 'CommonnessEvidenceTests_setUp', 'CommonnessRenderArchitectureTests_NON_RENDERERS', 'CommonnessRenderArchitectureTests_SITES', 'CommonnessRenderArchitectureTests_VERDICT_OWNERS', 'CommonnessRenderTests_assert_delivery', 'CommonnessRenderTests_local_comparison', 'CommonnessRenderTests_seed_both', 'CommonnessRenderTests_setUp', 'CompletedUndoTests_cancel', 'CompletedUndoTests_setUp', 'CompletedUndoTests_tearDown', 'CompletedUndoTests_write', 'DAY_SECONDS', 'DisplaceHTTP', 'DraftFlowTests_acknowledge_write', 'DraftFlowTests_context', 'DraftFlowTests_continue_after_write', 'DraftFlowTests_runtime', 'DraftFlowTests_setUp', 'DraftFlowTests_tool', 'DraftMutationClaimStore', 'DraftOperationCoordinator', 'DraftReminderStore', 'DraftServer', 'ENCODE', 'ExistingQueryTests_corpus', 'ExistingQueryTests_setUp', 'ExitStack', 'ExplicitAddHotfixTests_assert_created', 'ExplicitAddHotfixTests_asyncSetUp', 'ExplicitAddHotfixTests_call', 'ExplicitAddHotfixTests_execute', 'ExplicitAddHotfixTests_mark_resolvable_ambiguity', 'ExplicitCodeRequest', 'ExplicitDestinationTests_encode', 'ExplicitDestinationTests_invoke', 'ExplicitDestinationTests_lookup_codes', 'ExplicitDestinationTests_lookup_words', 'ExplicitDestinationTests_setUp', 'ExplicitDestinationTests_strict', 'ExplicitEntryCodeRequest', 'ExplicitReadingReplayTests_setUp', 'ExplicitReadingReplayTests_tool', 'ExplicitSubmitTests_assert_submitted_or_actionable_submit_ticket', 'ExplicitSubmitTests_executor_runtime', 'ExplicitSubmitTests_runtime', 'ExplicitSubmitTests_seed_existing_draft', 'ExplicitSubmitTests_setUp', 'ExplicitSubmitTests_tool', 'ExplicitSubmitTests_turn', 'FIRST_WORD_BASELINE', 'FLY_CODES', 'FOOTER', 'FOOTER_BASELINE', 'FORCE_ASSENT_TEXTS', 'FakeClient', 'FakeKeytaoServer', 'FakeShiftServer', 'FlyKeyReplay', 'FlyTranscriptTests_setUp', 'FlyTranscriptTests_tool', 'FreshSelectionTests_runtime', 'FreshSelectionTests_setUp', 'FreshSelectionTests_tool', 'FreshSelectionTests_turn', 'GeneralReplyTests_reply', 'GenericEvictionTests_run_command', 'GenericProposalBridgeTests_assert_ticket', 'GenericProposalBridgeTests_call', 'GenericProposalBridgeTests_setUp', 'HINT', 'HISTORICAL_CHANNELS', 'HotfixMultiwordTests__check_standalone_shift_real_server_plan', 'HotfixMultiwordTests_assert_candidates', 'HotfixMultiwordTests_asyncSetUp', 'HotfixMultiwordTests_lookup', 'HotfixMultiwordTests_query', 'HotfixMultiwordTests_unresolved_message', 'HotfixMultiwordTests_unresolved_review', 'INCIDENT', 'INSTRUCTION', 'ITEM', 'LiveTicketTests_prepare_ticket', 'LiveTicketTests_setUp', 'LiveTicketTests_tool', 'MESSAGES', 'MODEL_REPLY', 'MODIFIERS', 'MagicMock', 'MemoryConversationStateStore', 'MessageIdBindingTests_context', 'MessageIdBindingTests_delivery', 'MessageIdBindingTests_setUp', 'Mock', 'ModuleType', 'MoveToCodeGrammarTests_assert_bound_move', 'MoveToCodeGrammarTests_context', 'MultiSelectionReplay', 'MultiwordRouteTests_discover', 'NARRATION', 'NEW_CODES', 'NO_PLAN', 'OLD', 'OLD_CODES', 'OLD_ITEM', 'OTHER', 'OTHER_BATCH', 'OWNER', 'OWN_BATCH', 'OfflineQQBot', 'OfflineResponse', 'OperationRequest', 'OrderedDict', 'PAIRS', 'PRESENT', 'PRIMARY', 'PROBES', 'PR_ITEMS', 'Path', 'PendingAddWord', 'PendingToolConfirm', 'PendingWordHTTP', 'QUERY_WORDS', 'QUOTE', 'QueryPrecedenceTests_delivered', 'QueryPrecedenceTests_tool', 'READING', 'READINGS', 'REAL_CALL_TOOL', 'REAL_COMMAND_CLASSIFIER', 'REAL_CORE', 'REAL_FINISH', 'REAL_FORMAT_DRAFT', 'REAL_MATCHER', 'REAL_REVIEW_READS', 'REAL_TOOL_DISPATCH', 'REAL_WORD_CLASSIFIER', 'REFUSAL', 'REPLIES', 'REREVIEW', 'REST', 'REVIEWS', 'REVIEW_ARGS', 'REVIEW_REPLY', 'ReadingHTTP', 'ReadingRereviewPersistenceTests_assert_batch_release_dash', 'ReadingRereviewPersistenceTests_assert_incident', 'ReadingRereviewPersistenceTests_assert_new_word_dash', 'ReadingRereviewPersistenceTests_asyncSetUp', 'ReadingRereviewPersistenceTests_initial_review', 'ReadingRereviewPersistenceTests_rereview', 'ReadingRereviewPersistenceTests_tool', 'ReadingRereviewPersistenceTests_turn', 'ReadingResolutionTests_execute', 'ReadingResolutionTests_prepare_ambiguous', 'RealToolReplay', 'ReceiptDeltaTests_invoke', 'ReceiptDeltaTests_setUp', 'ReceiptDeltaTests_tearDown', 'ReceiptFinalizerTests_assert_exact_created_coverage', 'Replay', 'RestReplay', 'S54RendererTests_setUp', 'S54SelectionTests_reordered_record', 'S56CommonnessTests_assess', 'S59CommonnessRouteTests_run_incident', 'S59CommonnessRouteTests_setUp', 'S60ScenarioTests_replay', 'S61SubmitReplayTests_patches', 'S61SubmitReplayTests_run_add_and_submit', 'S61SubmitReplayTests_run_shift_and_submit', 'S61SubmitReplayTests_setUp', 'S62IncidentTests_replay', 'S62SelectionTests_execute_selection', 'S67RankingTests_cells', 'S67RankingTests_reply', 'S67RankingTests_setUp', 'SHANGHAI', 'SQLiteConversationStateStore', 'ScenarioAssertionError', 'SearchBreakerTests_result', 'SearchBreakerTests_setUp', 'SearchBreakerTests_trip', 'SelectionReplay', 'SelectionTests_prepare', 'SelectionTests_setUp', 'SelectionTests_tool', 'SessionHTTP', 'ShiftReplay', 'SiblingReplay', 'SimpleNamespace', 'SingleReviewTests_audit_single', 'SingleReviewTests_prepare', 'SkillsManager', 'SmallTalkTests_assert_natural', 'SmallTalkTests_delivered', 'SmallTalkTests_setUp', 'SmallTalkTests_tool', 'StaleTicketReleaseTests_assert_fresh_request', 'StaleTicketReleaseTests_replay', 'T1', 'T3', 'TOOLS', 'ToolContext', 'ToolExecutor', 'TranscriptTests_delivered_turn', 'TranscriptTests_executor_runtime', 'TranscriptTests_setUp', 'TranscriptTests_tool', 'UNBOUND_BINDING_PRECHECK_NOTICE', 'VOTER', 'VoteHTTP', 'WRAPPED', '_SPEC', '_assert_s56_advertised_reply_closure', '_build_statuses_for_codes', '_command_suggestions_match_pending_batch', '_format_reviewed_add_prompt', '_pending_add_word_payload', '_strip_unbound_command_suggestions', 'additional_code_warning', 'advertised_batch_binding_pairs', 'advertised_command_suggestions', 'advertised_reply_contract', 'advertised_single_word_candidate_codes', 'advertised_single_word_lookup_codes', 'already_existing_word_copy', 'ambiguous_review', 'ast', 'asyncio', 'b10_flykey_CODES', 'b10_flykey_WORD', 'b11_shift_WORD', 'b11_shift_draft_tools', 'b12_select_FACTS', 'b14_multi_select_FACTS', 'b14_multi_select_SELECTION_ERROR', 'b14_multi_select_T2', 'b15_sibling_and_others_CODES', 'b15_sibling_and_others_WORDS', 'b17_rest_recommended_FACTS', 'b17_rest_recommended_SELECTION_ERROR', 'b17_rest_recommended_TEN_WORDS', 'b17_rest_recommended_commands', 'b17_rest_recommended_routing', 'b17_rest_recommended_shift_ticket', 'b18_rest_real_tools_DraftHTTP', 'b18_rest_real_tools_FACTS', 'b18_rest_real_tools_TEN_WORDS', 'b19_partial_reply_binding_DraftHTTP', 'b19_partial_reply_binding_WORDS', 'b19_partial_reply_binding_chat', 'b24_displace_CODES', 'b24_displace_OCCUPANT', 'b24_displace_WORD', 'b25_session_CODES', 'b25_session_FACTS', 'b25_session_OCCUPANT', 'b25_session_WORD', 'b25_session_WORDS', 'b25_session_commands', 'b27_reading_WORD', 'b8_cancel_WORDS', 'b9_msgid_chat', 'b9_msgid_routing', 'begin_turn_metrics', 'call_tool_function', 'candidate_commonness_guard_copy', 'candidate_fixture', 'capture_review', 'chat_render', 'closing', 'commonness_copy', 'commonness_query', 'completed_draft_undo', 'contextlib', 'contextmanager', 'copy', 'create_pending_trusted_word_record', 'create_schema', 'current_turn_id', 'current_turn_metrics', 'datetime', 'direct', 'end_turn_metrics', 'enforce_conversational_reply', 'existing_single_review', 'explicit_complete_add_item', 'format_frequency', 'front_insert_recommendation_copy', 'grammar', 'harness', 'hashlib', 'hotfix_explicit_add_CODES', 'hotfix_explicit_add_WORD', 'http_client', 'importlib', 'incident_ticket', 'ingest', 'io', 'is_force_assent', 'json', 'logging', 'lookup_fixture', 'lookup_word_commonness', 'matches_previous_bot_option', 'merge_receipt_deltas', 'message_authorizes_mutation', 'mixed_items', 'mixed_receipt', 'negative_parser_case', 'normalize_mutation_command_source', 'notice', 'observability', 'occupied_review', 'offered_option_intent', 'option_questions_bind_live_state', 'os', 'parse_commonness_query', 'parse_dictionary_recode_command', 'parse_eviction_modified_add', 'parse_explicit_code_request', 'parse_explicit_entry_code_request', 'parse_parenthesised_readings', 'parse_pending_candidate_selection', 'parse_reviewed_shift_words', 'parse_standalone_shift', 'patch', 'pending', 'pending_candidate', 'pending_confirmation_copy', 'pending_duplicate_confirmation_is_complete', 'pending_duplicate_confirmation_state', 'pending_item', 'policy_agent', 'policy_call', 'positive_parser_case', 'pr', 'prepare_fly_review', 'prepared_review', 'preview', 'previous_bot_options', 'product', 'protected_candidate_occupants', 'query', 'raw_operation', 're', 'real_batch_remove', 'receipt_change_lines', 'receipt_changes', 'render', 'render_missing_option_ticket', 'render_remediation_reply', 'render_server_backed_batch_candidates', 'render_server_backed_single_word_candidates', 'render_server_backed_single_word_lookup', 'replace', 'review', 'review_fixture', 'reviewed', 'reviewed_destination', 'reviewed_record', 'reviewed_scope', 'reviewed_single', 'reviewed_ticket', 'reviewed_word', 's54_multiword_chat', 's54_multiword_commands', 's54_multiword_review_payload', 's55_failure_boundary_chat', 's55_single_pipeline_chat', 's55_single_pipeline_commands', 's56_commonness_binding_chat', 's56_commonness_binding_commands', 's56_explicit_code_chat', 's56_explicit_code_commands', 's56_explicit_code_routing', 's56_generic_eviction_chat', 's56_generic_eviction_commands', 's56_security_chat', 's56_security_commands', 's56_security_routing', 's56_undo_chat', 's56_undo_row', 's57_claiming_chat', 's57_extra_code_chat', 's57_extra_code_commands', 's57_extra_code_routing', 's57_options_chat', 's57_options_routing', 's57_reorder_chat', 's57_reorder_commands', 's57_reorder_draft_tools', 's57_security_review_chat', 's57_security_review_commands', 's58_advertisement_chat', 's58_advertisement_shift_ticket', 's58_explicit_destination_draft_tools', 's59_commonness_route_MESSAGE', 's59_commonness_route_WORDS', 's59_commonness_route_chat', 's59_commonness_route_commands', 's60_delta_row', 's60_finalizers_BATCH_ID', 's60_receipt_chat', 's60_receipt_commands', 's60_scenario_BATCH_ID', 's60_scenario_chat', 's60_scenario_commands', 's60_scenario_review_payload', 's61_submit_replay_BATCH_ID', 's61_submit_replay_FakeBot', 's61_submit_replay_WORD', 's61_submit_replay_chat', 's61_submit_replay_commands', 's62_commonness_MESSAGE', 's62_commonness_WORDS', 's62_draft_flow_chat', 's62_draft_flow_commands', 's62_incident_incident_record', 's62_reading_resolution_chat', 's62_reading_resolution_commands', 's62_selection_incident_record', 's62_stale_draft_FakeBot', 's62_stale_draft_chat', 's63_bcc_FIXTURE', 's63_fresh_selection_chat', 's64_existing_query_chat', 's64_existing_query_commands', 's66_readings_CODES', 's66_readings_T2', 's66_readings_WORDS', 's68_selection_FIXTURE', 's69_fly_ticket_chat', 's69_fly_ticket_commands', 's70_smalltalk_chat', 's71_stale_ticket_release_chat', 's71_stale_ticket_release_commands', 's73_reading_rereview_persistence_chat', 's73_reading_rereview_persistence_commands', 'same_code_reorder', 'seed_historical', 'select_candidate_inventory', 'server_warning_pending_state', 'server_warning_ticket_is_complete', 'set_turn_flow', 'shift_plan_warnings', 'shutil', 'single_encoding', 'single_review', 'snapshot', 'socket', 'sqlite3', 'strip_self_mention_tokens', 'structural_option_questions', 'submitted_data', 'subprocess', 'sys', 'tempfile', 'ticket', 'time', 'timedelta', 'trusted_pending_word_items', 'trusted_word_record_is_complete', 'turn_failure_reply', 'understand_message', 'unicodedata', 'unittest', 'urlsplit', 'validate_commonness_words', 'validate_explicit_code', 'validate_operation', 'vote', 'vote_fixtures', 'web', 'zipfile']
