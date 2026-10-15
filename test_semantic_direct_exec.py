"""Offline direct-execution checks through the real semantic and write handlers."""

import asyncio
import copy
import json
import os
import unittest
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from test_semantic_parser import FakeClient, raw_operation

from keytao_bot.harness.conversation import ConversationAddress
from keytao_bot.harness.operation_request import OperationRequest
from keytao_bot.harness.state import (
    DraftOperationCoordinator,
    MemoryConversationStateStore,
    PendingAddWord,
    PendingToolConfirm,
)


class SemanticDirectExecTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.chat = harness.openai_chat_module
        self.commands = harness.chat_commands_module
        self.store = MemoryConversationStateStore()
        self.author = ConversationAddress.group("qq", "direct-exec-group", "direct-exec-owner")
        self.other = ConversationAddress.group("qq", "direct-exec-group", "other-owner")
        self.client = FakeClient()
        self.rows = [self.row("吃饱了撑", "wblj")]
        self.calls = []
        self.writes = []
        self.warning = None
        self.batch_preview = False
        self.preview_changes = {}
        self.omit_warnings = False
        self.submit_preview = False
        self.shifted = []
        self.draft_items = None
        self.draft_version = 1
        self.review = AsyncMock(side_effect=self.review_word)
        self.finish = AsyncMock()
        patches = (
            patch.dict(os.environ, {
                "KEYTAO_SEMANTIC_ROUTE_OPS": "move,recode,swap,delete,reorder,batch_replace,draft_submit,draft_recall,draft_clear,draft_keep_only,direct_exec",
                "KEYTAO_SEMANTIC_SHADOW": "0",
            }),
            patch.object(self.chat, "AsyncOpenAI", return_value=self.client),
            patch.object(self.chat, "WORD_QUERY_INTENT_MODEL", "offline-model"),
            patch.object(self.chat, "conversation_state_store", self.store),
            patch.object(self.chat, "draft_operation_coordinator", DraftOperationCoordinator()),
            patch.object(self.chat, "call_tool_function", side_effect=self.call),
            patch.object(self.chat, "remember_conversation"),
            patch.object(self.chat, "_finish_ai_chat_response", self.finish),
            patch.object(self.commands, "_try_handle_simple_single_word_query", self.review),
            patch.object(self.commands, "_format_draft_response", AsyncMock(return_value="草稿地址：https://offline.invalid/draft/offline-batch")),
        )
        for context in patches:
            context.start()
            self.addCleanup(context.stop)

    async def asyncTearDown(self):
        tasks = tuple(self.chat._semantic_shadow_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.chat._semantic_shadow_tasks.clear()

    @staticmethod
    def row(word, code, weight=100):
        return {"word": word, "code": code, "type": "Phrase", "weight": weight}

    async def review_word(self, message, platform, user_id, conv_key, space_key, owner_label, **kwargs):
        self.assertEqual(message, "加词 我本来就")
        self.assertTrue(self.store.set(
            conv_key,
            PendingAddWord(
                "我本来就", "wblj", [("wblj", True)],
                server_candidates=[("wblj", True)],
                server_occupied_words={"wblj": ["吃饱了撑"]},
                pronunciation_codes={"wblj": "wo ben lai jiu"},
                needs_manual_review=False,
            ),
            space_key=space_key, owner_label=owner_label,
        ))
        return "offline pronunciation review"

    def result(self, **changes):
        data = {
            "success": True, "batchId": "offline-batch", "contentVersion": 1,
            "batchUrl": "https://offline.invalid/draft/offline-batch",
        }
        data.update(changes)
        return data

    async def call(self, name, args, platform, user_id, **kwargs):
        self.assertEqual((platform, user_id), ("qq", self.author.actor_id))
        self.calls.append((name, copy.deepcopy(args)))
        if name == "keytao_lookup_by_words_batch":
            data = {"success": True, "results": [
                {"word": word, "phrases": [row for row in self.rows if row["word"] == word]}
                for word in args["words"]
            ]}
        elif name == "keytao_lookup_by_word":
            data = {"success": True, "phrases": [
                row for row in self.rows if row["word"] == args["word"]
            ]}
        elif name == "keytao_lookup_by_code":
            data = {"success": True, "phrases": [
                row for row in self.rows if row["code"] == args["code"]
            ]}
        elif name == "keytao_list_draft_items":
            data = (
                {"success": True, "batchId": "", "contentVersion": 0, "items": []}
                if self.draft_items is None
                else self.result(items=self.draft_items, contentVersion=self.draft_version)
            )
        elif name == "keytao_batch_remove_draft_items":
            targets = [
                {key: item[key] for key in ("id", "word", "code", "action", "type")}
                for item in self.draft_items if item["id"] in args["ids"]
            ]
            if not args.get("expected_target_digest"):
                data = self.result(
                    success=False, requiresConfirmation=True,
                    contentVersion=self.draft_version, targetDigest="a" * 64, warningDigest="a" * 64,
                    targets=targets, warnings=[self.warning] if self.warning else [],
                )
            else:
                self.writes.append((name, copy.deepcopy(args)))
                self.draft_items = [item for item in self.draft_items if item["id"] not in args["ids"]]
                self.draft_version += 1
                data = self.result(successCount=len(args["ids"]), failedCount=0, contentVersion=self.draft_version)
        elif name == "keytao_recall_batch":
            if "expected_content_version" not in args:
                data = self.result(
                    success=False, requiresConfirmation=True,
                    warningDigest="a" * 64,
                    warnings=[self.warning] if self.warning else [],
                )
            else:
                self.writes.append((name, copy.deepcopy(args)))
                data = self.result()
        elif name == "keytao_batch_add_to_draft":
            if (self.batch_preview or self.warning) and not args.get("confirmed"):
                data = self.result(
                    success=False, requiresConfirmation=True,
                    warningDigest="a" * 64, warnings=[self.warning] if self.warning else [],
                )
            else:
                self.writes.append((name, copy.deepcopy(args)))
                data = self.result(successCount=len(args["items"]), failedCount=0)
        elif name == "keytao_shift_phrase_code":
            if not args.get("confirmed_plan_digest"):
                data = self.result(
                    success=False, requiresConfirmation=True,
                    confirmationKind="shiftPlan", planDigest="b" * 64,
                    warningDigest="c" * 64,
                    warnings=[self.warning] if self.warning else [],
                    shiftPlan={"shifted": self.shifted, "items": []},
                )
            else:
                self.writes.append((name, copy.deepcopy(args)))
                data = self.result(shiftPlan={"shifted": self.shifted, "items": []})
        elif name == "keytao_submit_batch":
            if self.submit_preview and not args.get("confirmed"):
                data = self.result(
                    success=False, requiresConfirmation=True,
                    snapshotItems=[{"action": "Create", "word": "午时", "code": "wj", "type": "Phrase"}],
                    snapshotDigest="d" * 64, warningDigest="e" * 64,
                    auditDigest="f" * 64, warnings=[self.warning] if self.warning else [],
                )
            else:
                self.writes.append((name, copy.deepcopy(args)))
                data = self.result()
        else:
            raise AssertionError(f"Unexpected offline tool: {name} {args!r}")
        if data.get("requiresConfirmation"):
            data.update(self.preview_changes)
            if self.omit_warnings:
                data.pop("warnings", None)
        return json.dumps(data, ensure_ascii=False)

    async def stage(self, message, op="move", operands=("吃饱了撑", "我本来就"), codes=(), **changes):
        self.client.create.return_value.choices[0].message.content = json.dumps(
            raw_operation(op=op, operands=list(operands), codes=list(codes), **changes),
            ensure_ascii=False,
        )
        ctx = self.chat.TurnContext(
            bot=None, event=None, platform="qq", user_id=self.author.actor_id,
            normalized_message_text=message, conv_key=self.author,
            space_key=self.author.space_key, owner_label="Offline owner",
        )
        handled = await self.chat._stage_understand_operation(ctx)
        return handled, ctx

    async def answer(self, message, actor=None):
        actor = actor or self.author
        return await self.commands.handle_pending_message_core(
            message, "qq", actor.actor_id, actor,
            space_key=actor.space_key, allow_intent_model=False,
        )

    def assert_replace_write(self, response):
        self.assertEqual(len(self.writes), 1, (self.calls, response))
        name, args = self.writes[0]
        self.assertEqual(name, "keytao_batch_add_to_draft")
        self.assertEqual([
            {key: item[key] for key in ("action", "word", "code", "type")}
            for item in args["items"]
        ], [
            {"action": "Delete", "word": "吃饱了撑", "code": "wblj", "type": "Phrase"},
            {"action": "Create", "word": "我本来就", "code": "wblj", "type": "Phrase"},
        ])
        self.assertIn("删除「吃饱了撑」", response)
        self.assertIn("创建「我本来就」", response)
        self.assertIn("https://offline.invalid/draft/offline-batch", response)
        self.assertNotIn("回复「确认」", response)
        self.assertIsNone(self.store.get_record(self.author), response)

    def assert_server_confirmation(self, response):
        self.assertEqual(self.writes, [], response)
        record = self.store.get_record(self.author)
        self.assertIsNotNone(record, response)
        self.assertEqual(record.owner_key, self.author)
        self.assertEqual(record.state.confirmation_source, "server_warning")
        self.assertIn("确认", response)

    async def test_three_production_replacements_execute_exact_delete_create_once(self):
        cases = (
            ("把吃饱了撑换成我本来就", "move", ("吃饱了撑", "我本来就"), ()),
            ("把“吃饱了撑 wblj”换成“我本来就 wblj”", "recode", ("吃饱了撑", "我本来就"), ("wblj",)),
            ("顶替wblj为我本来就", "move", ("我本来就",), ("wblj",)),
        )
        for message, op, operands, codes in cases:
            with self.subTest(message=message):
                self.writes.clear()
                self.calls.clear()
                self.client.create.reset_mock()
                handled, ctx = await self.stage(message, op, operands, codes)
                self.assertTrue(handled, ctx.response)
                self.assert_replace_write(ctx.response)
                self.client.create.assert_awaited_once()

    async def test_without_direct_exec_ticket_shows_resolved_replace_and_confirm_writes(self):
        os.environ["KEYTAO_SEMANTIC_ROUTE_OPS"] = "move,recode"
        handled, ctx = await self.stage("把吃饱了撑换成我本来就")
        self.assertTrue(handled)
        self.assertEqual(self.writes, [])
        record = self.store.get_record(self.author)
        self.assertIsInstance(record.state, PendingToolConfirm)
        self.assertEqual(record.owner_key, self.author)
        self.assertEqual(record.state.function_name, "keytao_understood_command")
        self.assertEqual(record.state.args["command_text"], "修改 wblj：吃饱了撑 → 我本来就")
        self.assertIn("修改 wblj：吃饱了撑 → 我本来就", ctx.response)
        self.assertIsNone(await self.answer("确认", self.other))
        self.assertEqual(self.writes, [])
        self.assertIs(self.store.get_record(self.author), record)
        self.assert_replace_write(await self.answer("确认"))
        self.assertIsNone(await self.answer("确认"))
        self.assertEqual(len(self.writes), 1)

    async def test_ambiguous_old_word_offers_numbered_author_owned_options(self):
        self.rows.append(self.row("吃饱了撑", "wblja"))
        handled, ctx = await self.stage("把吃饱了撑换成我本来就")
        self.assertTrue(handled)
        self.assertEqual(self.writes, [])
        self.assertRegex(ctx.response, r"(?m)^1[.、．)]")
        self.assertRegex(ctx.response, r"(?m)^2[.、．)]")
        self.assertIn("wblj", ctx.response)
        self.assertIn("wblja", ctx.response)
        record = self.store.get_record(self.author)
        self.assertIsNotNone(record, ctx.response)
        self.assertEqual(record.owner_key, self.author)
        self.assertIsNone(await self.answer("1", self.other))
        self.assertEqual(self.writes, [])
        self.assertIs(self.store.get_record(self.author), record)
        self.assert_replace_write(await self.answer("1"))
        self.assertIsNone(await self.answer("1"))
        self.assertEqual(len(self.writes), 1)

    async def test_question_frame_preserves_preview_without_direct_write(self):
        for frame in ("question", "direct"):
            with self.subTest(model_frame=frame):
                handled, ctx = await self.stage("能把吃饱了撑换成我本来就吗", frame=frame)
                self.assertTrue(handled)
                self.assertEqual(self.writes, [])
                self.assertIn("吃饱了撑", ctx.response)
                self.assertIsNotNone(self.store.get_record(self.author), ctx.response)

    async def test_pending_reference_cannot_use_direct_execution(self):
        pending = PendingAddWord("我本来就", "wblj", [("wblj", True)])
        self.assertTrue(self.store.set(self.author, pending))
        record = self.store.get_record(self.author)
        handled, ctx = await self.stage(
            "把吃饱了撑换成我本来就", refers_to_pending=True,
        )
        self.assertFalse(handled)
        self.assertEqual(self.writes, [])
        self.assertIs(self.store.get_record(self.author), record)
        self.review.assert_not_awaited()

    async def test_swap_with_model_replace_relation_uses_real_replace_handler(self):
        handled, ctx = await self.stage(
            "把吃饱了撑换成我本来就", "swap", relation="replace",
        )
        self.assertTrue(handled)
        self.assert_replace_write(ctx.response)

    async def test_unresolved_or_multiple_code_occupants_never_write(self):
        for rows in (
            [],
            [self.row("吃饱了撑", "wblj"), self.row("其他词条", "wblj")],
        ):
            with self.subTest(rows=rows):
                self.rows = rows
                handled, ctx = await self.stage("顶替wblj为我本来就", "move", ("我本来就",), ("wblj",))
                self.assertTrue(handled)
                self.assertEqual(self.writes, [])
                self.assertIn("未写入", ctx.response)
        self.review.assert_not_awaited()

    async def test_literal_operand_and_code_validation_still_blocks_writes(self):
        for operands, codes in (
            (("吃饱了撑", "不存在"), ()),
            (("吃饱了撑", "我本来就"), ("othercode",)),
        ):
            with self.subTest(operands=operands, codes=codes):
                handled, ctx = await self.stage("把吃饱了撑换成我本来就", "move", operands, codes)
                self.assertFalse(handled)
                self.assertEqual(self.writes, [])
                self.assertIsNone(self.store.get_record(self.author))
        self.review.assert_not_awaited()

    async def test_negated_and_reported_frames_never_write(self):
        for message, frame in (
            ("不要把吃饱了撑换成我本来就", "negated"),
            ("他说把吃饱了撑换成我本来就", "reported"),
            ("不要把吃饱了撑换成我本来就", "direct"),
            ("他说把吃饱了撑换成我本来就", "direct"),
        ):
            with self.subTest(message=message, model_frame=frame):
                handled, ctx = await self.stage(message, frame=frame)
                self.assertFalse(handled)
                self.assertEqual(self.writes, [])
                self.assertIsNone(self.store.get_record(self.author))
        self.review.assert_not_awaited()

    async def test_direct_delete_writes_one_typed_item_without_ticket(self):
        handled, ctx = await self.stage(
            "把吃饱了撑 wblj 删掉", "delete", ("吃饱了撑",), ("wblj",),
        )
        self.assertTrue(handled)
        self.assertEqual(len(self.writes), 1, ctx.response)
        self.assertEqual(self.writes[0][0], "keytao_batch_add_to_draft")
        self.assertEqual(self.writes[0][1]["items"], [
            {"action": "Delete", "word": "吃饱了撑", "code": "wblj", "type": "Phrase"},
        ])
        self.assertIn("已加入草稿", ctx.response)
        self.assertIsNone(self.store.get_record(self.author))

    async def test_ambiguous_delete_numbered_choice_is_author_owned_and_writes_once(self):
        self.rows.append(self.row("吃饱了撑", "wblja"))
        handled, ctx = await self.stage("删掉吃饱了撑", "delete", ("吃饱了撑",), ())
        self.assertTrue(handled)
        self.assertEqual(self.writes, [])
        self.assertRegex(ctx.response, r"(?m)^1[.、．)]")
        self.assertRegex(ctx.response, r"(?m)^2[.、．)]")
        record = self.store.get_record(self.author)
        self.assertIsNotNone(record, ctx.response)
        self.assertEqual(record.owner_key, self.author)
        self.assertIsNone(await self.answer("2", self.other))
        self.assertEqual(self.writes, [])
        self.assertIs(self.store.get_record(self.author), record)
        response = await self.answer("2")
        self.assertEqual(len(self.writes), 1, response)
        self.assertEqual(self.writes[0][1]["items"], [
            {"action": "Delete", "word": "吃饱了撑", "code": "wblja", "type": "Phrase"},
        ])
        self.assertIsNone(self.store.get_record(self.author))
        self.assertIsNone(await self.answer("2"))
        self.assertEqual(len(self.writes), 1)

    async def test_existing_new_word_moves_to_code_instead_of_creating_replacement(self):
        self.rows.append(self.row("我本来就", "wblja", 101))
        handled, ctx = await self.stage("把吃饱了撑 wblj 换成我本来就", codes=("wblj",))
        self.assertTrue(handled)
        self.assertEqual(len(self.writes), 1, ctx.response)
        name, args = self.writes[0]
        self.assertEqual(name, "keytao_shift_phrase_code")
        self.assertEqual((args["word"], args["target_code"]), ("我本来就", "wblj"))
        self.assertNotIn("items", args)
        self.review.assert_not_awaited()
        self.assertIsNone(self.store.get_record(self.author))

    async def test_existing_word_moves_to_empty_code_without_replace_review(self):
        handled, ctx = await self.stage(
            "把吃饱了撑调到wblja", "move", ("吃饱了撑",), ("wblja",),
        )
        self.assertTrue(handled)
        self.assertEqual(len(self.writes), 1, ctx.response)
        name, args = self.writes[0]
        self.assertEqual(name, "keytao_shift_phrase_code")
        self.assertEqual((args["word"], args["target_code"]), ("吃饱了撑", "wblja"))
        self.assertNotIn("items", args)
        self.review.assert_not_awaited()
        self.assertIsNone(self.store.get_record(self.author))

    async def test_direct_recode_and_batch_replace_each_write_one_exact_batch(self):
        cases = (
            (
                "吃饱了撑的 wblj 改成 wblja", "recode", ("吃饱了撑",), ("wblj", "wblja"),
                [("Delete", "吃饱了撑", "wblj"), ("Create", "吃饱了撑", "wblja")],
            ),
            (
                "把午替换成晚，源条目午时 wj", "batch_replace", ("午", "晚", "午时"), ("wj",),
                [("Change", "晚时", "wj")],
            ),
        )
        for message, op, operands, codes, expected in cases:
            with self.subTest(op=op):
                self.writes.clear()
                handled, ctx = await self.stage(message, op, operands, codes)
                self.assertTrue(handled)
                self.assertEqual(len(self.writes), 1, ctx.response)
                name, args = self.writes[0]
                self.assertEqual(name, "keytao_batch_add_to_draft")
                self.assertEqual([(item["action"], item["word"], item["code"]) for item in args["items"]], expected)
                self.assertIsNone(self.store.get_record(self.author))

    async def test_failed_batch_replace_save_never_executes_or_consumes_existing_ticket(self):
        request = OperationRequest(
            op="batch_replace", operands=("午", "晚", "午时"), codes=("wj",),
        )
        command = self.commands.render_understood_command(request)
        stale_states = (
            PendingToolConfirm("keytao_batch_add_to_draft", {"items": [
                {"action": "Create", "word": "其他词条", "code": "qtct"},
            ]}),
            PendingToolConfirm("keytao_batch_add_to_draft", {"items": [
                {"action": "Change", "old_word": "午时", "word": "晚时", "code": "wj"},
            ]}),
            PendingAddWord("我本来就", "wblj", [("wblj", True)]),
        )
        for stale in stale_states:
            with self.subTest(stale=type(stale).__name__, args=getattr(stale, "args", None)):
                self.assertTrue(self.store.set(self.author, stale))
                record = self.store.get_record(self.author)
                token = self.commands._semantic_direct_words.set(request.operands)
                try:
                    with patch.object(self.store, "set", return_value=False):
                        response = await self.commands.execute_understood_command(
                            request, command, "qq", self.author.actor_id,
                            self.author, self.author.space_key, "Offline owner",
                        )
                finally:
                    self.commands._semantic_direct_words.reset(token)
                self.assertIn("替换进度无法保存", response)
                self.assertEqual(self.writes, [])
                self.assertIs(self.store.get_record(self.author), record)
                self.assertEqual(record.execution_id, "")
                self.assertIs(record.state, stale)

    async def test_direct_same_code_reorder_writes_one_exact_weight_plan(self):
        self.rows = [self.row("午时", "wj", 101), self.row("务实", "wj", 100)]
        with patch("keytao_bot.utils.same_code_reorder._advisory", return_value="offline ordering evidence"):
            handled, ctx = await self.stage("午时优先于务实", "reorder", ("午时", "务实"), ())
        self.assertTrue(handled)
        self.assertEqual(len(self.writes), 1, ctx.response)
        name, args = self.writes[0]
        self.assertEqual(name, "keytao_shift_phrase_code")
        self.assertEqual(args["ordered_words"], ["午时", "务实"])
        self.assertEqual(args["expected_codes"], ["wj", "wj"])
        self.assertEqual(args["expected_weights"], [100, 101])
        self.assertIsNone(self.store.get_record(self.author))

    async def test_direct_draft_clear_recall_and_keep_only_each_write_once(self):
        for message, op, operands, expected_tool, expected_ids in (
            ("清空当前草稿", "draft_clear", (), "keytao_batch_remove_draft_items", [1, 2]),
            ("撤回当前提交", "draft_recall", (), "keytao_recall_batch", None),
            ("草稿只保留午时", "draft_keep_only", ("午时",), "keytao_batch_remove_draft_items", [2]),
        ):
            with self.subTest(op=op):
                self.store.delete(self.author)
                self.writes.clear()
                self.draft_version = 1
                self.draft_items = [
                    {"id": 1, "action": "Create", "word": "午时", "code": "wj", "type": "Phrase"},
                    {"id": 2, "action": "Create", "word": "务实", "code": "wja", "type": "Phrase"},
                ]
                handled, ctx = await self.stage(message, op, operands, ())
                self.assertTrue(handled)
                self.assertEqual(len(self.writes), 1, ctx.response)
                name, args = self.writes[0]
                self.assertEqual(name, expected_tool)
                if expected_ids is not None:
                    self.assertEqual(args["ids"], expected_ids)
                self.assertIsNone(self.store.get_record(self.author), ctx.response)
                self.assertIn("https://offline.invalid/draft/offline-batch", ctx.response)

    async def test_direct_draft_submit_writes_once_without_ticket(self):
        handled, ctx = await self.stage("直接提交现在草稿", "draft_submit", (), ())
        self.assertTrue(handled)
        self.assertEqual([name for name, args in self.writes], ["keytao_submit_batch"])
        self.assertIn("提交审核", ctx.response)
        self.assertIn("https://offline.invalid/draft/offline-batch", ctx.response)
        self.assertIsNone(self.store.get_record(self.author))

    async def test_direct_swap_confirms_exact_server_plan_with_one_write(self):
        self.rows = [self.row("午时", "wj", 100), self.row("务实", "wja", 101)]
        handled, ctx = await self.stage("把午时和务实的编码对调", "swap", ("午时", "务实"), ())
        self.assertTrue(handled)
        self.assertEqual(len(self.writes), 1, (self.calls, ctx.response))
        name, args = self.writes[0]
        self.assertEqual(name, "keytao_shift_phrase_code")
        self.assertEqual(args["confirmed_plan_digest"], "b" * 64)
        self.assertEqual(args["ordered_words"], ["午时", "务实"])
        self.assertEqual(args["expected_codes"], ["wja", "wj"])
        self.assertIsNone(self.store.get_record(self.author))
        self.assertNotIn("回复「确认」", ctx.response)

    async def test_direct_submit_preview_confirms_only_without_unnamed_warnings(self):
        self.submit_preview = True
        for warning in (None, {"warningType": "conflict", "word": "未提到的词", "code": "wj"}):
            with self.subTest(warning=warning):
                self.store.delete(self.author)
                self.writes.clear()
                self.warning = warning
                handled, ctx = await self.stage("直接提交现在草稿", "draft_submit", (), ())
                self.assertTrue(handled)
                if warning is None:
                    self.assertEqual(len(self.writes), 1, ctx.response)
                    self.assertTrue(self.writes[0][1]["confirmed"])
                    self.assertIsNone(self.store.get_record(self.author))
                else:
                    self.assertEqual(self.writes, [], ctx.response)
                    record = self.store.get_record(self.author)
                    self.assertIsNotNone(record, ctx.response)
                    self.assertEqual(record.owner_key, self.author)
                    self.assertEqual(record.state.confirmation_source, "server_warning")

    async def test_direct_swap_does_not_auto_confirm_an_unnamed_shift(self):
        self.rows = [self.row("午时", "wj", 100), self.row("务实", "wja", 101)]
        self.shifted = [{"word": "额外词条", "fromCode": "wj", "toCode": "wjaa"}]
        handled, ctx = await self.stage("把午时和务实的编码对调", "swap", ("午时", "务实"), ())
        self.assertTrue(handled)
        self.assertEqual(self.writes, [])
        record = self.store.get_record(self.author)
        self.assertIsNotNone(record, ctx.response)
        self.assertEqual(record.owner_key, self.author)
        self.assertEqual(record.state.confirmation_source, "server_warning")

    async def test_named_replace_warning_auto_confirms_but_unnamed_warning_seals_ticket(self):
        for warning_word, direct in (("吃饱了撑", True), ("外部词条", False)):
            with self.subTest(warning_word=warning_word):
                self.store.delete(self.author)
                self.writes.clear()
                self.warning = {
                    "warningType": "duplicate_code",
                    "item": {"action": "Delete", "word": warning_word, "code": "wblj"},
                }
                handled, ctx = await self.stage("把吃饱了撑换成我本来就")
                self.assertTrue(handled)
                if direct:
                    self.assert_replace_write(ctx.response)
                    self.assertTrue(self.writes[0][1]["confirmed"])
                else:
                    self.assertEqual(self.writes, [])
                    record = self.store.get_record(self.author)
                    self.assertIsNotNone(record, ctx.response)
                    self.assertEqual(record.owner_key, self.author)
                    self.assertEqual(record.state.confirmation_source, "server_warning")
                    self.assertIn("确认", ctx.response)

    async def test_code_only_replace_auto_confirms_warning_only_for_target_occupant(self):
        self.warning = {
            "warningType": "duplicate_code",
            "item": {"action": "Delete", "word": "吃饱了撑", "code": "wblj"},
        }
        handled, ctx = await self.stage("顶替wblj为我本来就", "move", ("我本来就",), ("wblj",))
        self.assertTrue(handled)
        self.assert_replace_write(ctx.response)
        self.assertTrue(self.writes[0][1]["confirmed"])

        for word, code in (("吃饱了撑", "wblja"), ("外部词条", "wblj")):
            with self.subTest(word=word, code=code):
                self.writes.clear()
                self.warning["item"].update(word=word, code=code)
                handled, ctx = await self.stage("顶替wblj为我本来就", "move", ("我本来就",), ("wblj",))
                self.assertTrue(handled)
                self.assert_server_confirmation(ctx.response)
                self.store.delete(self.author)

    async def test_warning_existing_word_list_includes_unnamed_entries_in_ownership_check(self):
        for existing, direct in ((["吃饱了撑"], True), (["吃饱了撑", "外部词条"], False)):
            with self.subTest(existing=existing):
                self.store.delete(self.author)
                self.writes.clear()
                self.warning = {
                    "warningType": "duplicate_code", "word": "我本来就",
                    "code": "wblj", "existing": existing,
                }
                handled, ctx = await self.stage("把吃饱了撑换成我本来就")
                self.assertTrue(handled)
                if direct:
                    self.assert_replace_write(ctx.response)
                else:
                    self.assertEqual(self.writes, [])
                    record = self.store.get_record(self.author)
                    self.assertIsNotNone(record, ctx.response)
                    self.assertEqual(record.owner_key, self.author)
                    self.assertEqual(record.state.confirmation_source, "server_warning")

    async def test_direct_replace_rejects_unsafe_or_incomplete_server_preview(self):
        self.batch_preview = True
        cases = [
            {"conflicts": [{"word": "别人的词", "code": "wblj"}]},
            *({flag: True} for flag in (
                "uncertain", "staleConfirmation", "batchStateChanged", "contentVersionConflict",
                "partialWrite", "failed", "skipped",
            )),
            {"failedCount": 1}, {"skippedCount": 1}, {"warnedCount": 1},
            {"warnings": None}, {"warnings": {}}, {"warnings": "unknown"},
            {},
        ]
        for index, changes in enumerate(cases):
            with self.subTest(changes=changes, missing_warnings=index == len(cases) - 1):
                self.store.delete(self.author)
                self.writes.clear()
                self.calls.clear()
                self.preview_changes = changes
                self.omit_warnings = index == len(cases) - 1
                handled, ctx = await self.stage(
                    "把吃饱了撑 wblj 换成我本来就", codes=("wblj",),
                )
                self.assertTrue(handled)
                self.assert_server_confirmation(ctx.response)
                self.assertFalse(any(args.get("confirmed") for _, args in self.calls))

    async def test_direct_warning_rejects_unknown_types_codes_and_shapes(self):
        cases = [
            {"warningType": kind, "word": "我本来就", "code": "wblj"}
            for kind in ("x", "dictionary_delete", "dictionary_entry_deleted", "code_conflict", None, [], {})
        ] + [
            {"warningType": "duplicate_code", "word": "我本来就", "code": "wblja"},
            {"warningType": "duplicate_code", "word": ["我本来就"], "code": "wblj"},
            {"warningType": "duplicate_code", "word": "我本来就", "code": ["wblj"]},
            {"warningType": "duplicate_code", "word": {"word": "我本来就", "code": "wblj"}},
            {"warningType": "duplicate_code", "word": "我本来就", "code": "wblj",
             "conflictWith": {"phrase": "某词", "code": "wblj"}},
            {"warningType": "duplicate_code", "word": "我本来就", "code": "wblj",
             "conflictWith": {"phrase": {"word": "吃饱了撑", "code": "wblja"}}},
            {"warningType": "duplicate_code", "word": "我本来就", "code": "wblj",
             "existing": ["吃饱了撑", 42]},
            {"warningType": "duplicate_code", "word": "我本来就", "code": "wblj",
             "conflictWith": {"unknown": "某词"}},
            {"warningType": "duplicate_code", "word": "我本来就", "code": "wblj",
             "conflictWith": "某词"},
            {"warningType": "duplicate_code", "word": "我本来就", "code": "wblj",
             "mystery": [{"word": "吃饱了撑", "code": "wblj"}]},
        ]
        for warning in cases:
            with self.subTest(warning=warning):
                self.store.delete(self.author)
                self.writes.clear()
                self.warning = warning
                handled, ctx = await self.stage("把吃饱了撑换成我本来就")
                self.assertTrue(handled)
                self.assert_server_confirmation(ctx.response)

    async def test_direct_warning_accepts_named_nested_phrases_at_exact_code(self):
        for existing in (
            {"phrase": "吃饱了撑", "code": "wblj"},
            {"phrase": {"word": "吃饱了撑", "code": "wblj"}},
            [{"phrase": "吃饱了撑", "code": "wblj"}],
        ):
            with self.subTest(existing=existing):
                self.warning = {
                    "warningType": "code_chain_priority", "word": "我本来就",
                    "code": "wblj", "conflictWith": existing,
                }
                self.writes.clear()
                handled, ctx = await self.stage("把吃饱了撑换成我本来就")
                self.assertTrue(handled)
                self.assert_replace_write(ctx.response)

    async def test_unsafe_previews_cannot_auto_confirm_other_direct_routes(self):
        for message, op, operands, codes in (
            ("把午时和务实的编码对调", "swap", ("午时", "务实"), ()),
            ("直接提交现在草稿", "draft_submit", (), ()),
            ("清空当前草稿", "draft_clear", (), ()),
            ("撤回当前提交", "draft_recall", (), ()),
        ):
            for missing in (False, True):
                with self.subTest(op=op, missing_warnings=missing):
                    self.store.delete(self.author)
                    self.writes.clear()
                    self.rows = [self.row("午时", "wj", 100), self.row("务实", "wja", 101)]
                    self.submit_preview = True
                    self.draft_items = (
                        None if op == "swap" else
                        [{"id": 1, "action": "Create", "word": "午时", "code": "wj", "type": "Phrase"}]
                    )
                    self.preview_changes = {} if missing else {"conflicts": [{"word": "别人的词", "code": "wj"}]}
                    self.omit_warnings = missing
                    handled, ctx = await self.stage(message, op, operands, codes)
                    self.assertTrue(handled)
                    self.assert_server_confirmation(ctx.response)

    async def test_code_only_replace_does_not_authorize_chain_shifts_or_leak_to_next_turn(self):
        handled, ctx = await self.stage("顶替wblj为我本来就", "move", ("我本来就",), ("wblj",))
        self.assertTrue(handled)
        self.assert_replace_write(ctx.response)
        self.rows.append(self.row("我本来就", "wblja", 101))
        self.writes.clear()
        self.shifted = [{"word": "吃饱了撑", "fromCode": "wblj", "toCode": "wbljaa"}]
        handled, ctx = await self.stage("顶替wblj为我本来就", "move", ("我本来就",), ("wblj",))
        self.assertTrue(handled)
        self.assert_server_confirmation(ctx.response)

    async def test_code_only_replace_at_empty_code_suggests_normal_add(self):
        self.rows = []
        handled, ctx = await self.stage("顶替wblj为我本来就", "move", ("我本来就",), ("wblj",))
        self.assertTrue(handled)
        self.assertEqual(self.writes, [])
        self.assertIn("wblj", ctx.response)
        self.assertIn("没有", ctx.response)
        self.assertIn("加词", ctx.response)
        self.assertNotIn("不唯一", ctx.response)
        self.assertIsNone(self.store.get_record(self.author))
        self.review.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
