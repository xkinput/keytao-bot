"""Offline S63 fresh word/code turns exercise the production stage chain."""

import copy
import json
import unittest
from contextlib import ExitStack, contextmanager
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from test_s62_selection import incident_record
from keytao_bot.harness.state import server_warning_ticket_is_complete
from keytao_bot.plugins import chat_commands as commands


chat = harness.openai_chat_module
BATCH_URL = "https://keytao.rea.ink/batch/s63-fresh"


class FreshSelectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
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

    async def tool(self, name, args, platform, user_id, **kwargs):
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
    def runtime(self):
        with ExitStack() as stack:
            for patcher in (
                patch.object(chat, "conversation_state_store", self.store),
                patch.object(chat, "draft_operation_coordinator", harness.DraftOperationCoordinator()),
                patch.object(chat, "call_tool_function", side_effect=self.tool),
                patch.object(chat, "remember_conversation"),
                patch.object(chat, "get_history", return_value=[]),
                patch.object(chat, "_finish_ai_chat_response", AsyncMock()),
                patch.object(chat, "_finish_ai_chat_matcher", AsyncMock()),
                patch.object(chat, "get_ai_response_core", AsyncMock(side_effect=AssertionError("General model reached"))),
                patch.object(chat, "_classify_message_command_intent", AsyncMock(side_effect=AssertionError("Intent model reached"))),
                patch.object(chat, "_classify_simple_word_query_intent", AsyncMock(side_effect=AssertionError("Word model reached"))),
            ):
                stack.enter_context(patcher)
            token = commands.current_memory_context.set(self.memory)
            try:
                yield
            finally:
                commands.current_memory_context.reset(token)

    async def turn(self, message, *, reply_reference=None):
        cache, classifier = chat.command_intent_memoizer(message, message)
        ctx = chat.TurnContext(
            bot=object(), event=object(), platform="qq", user_id=self.key.actor_id,
            message_text=message, normalized_message_text=message,
            conv_key=self.key, space_key=self.key.space_key,
            memory_context=self.memory, history=[],
            reply_reference=reply_reference or harness.ReplyReferenceInfo(),
            command_intent_cache=cache, command_intent_for=classifier,
        )
        await chat._stage_normalize_message_text(ctx)
        ctx.command_intent_cache, ctx.command_intent_for = chat.command_intent_memoizer(
            message, ctx.normalized_message_text,
        )
        start = chat.STAGES.index(chat._stage_resolve_current_pending_scope)
        end = chat.STAGES.index(chat._stage_generate_ai_response)
        for stage in chat.STAGES[start:end + 1]:
            if await stage(ctx):
                break
        return ctx.response

    async def test_fresh_pair_reads_own_draft_and_returns_its_real_link_without_model(self):
        with self.runtime():
            response = await self.turn("加亮 jslxa")
        self.assertIn("已在当前草稿", response)
        self.assertIn(BATCH_URL, response)
        self.assertEqual([name for name, _ in self.calls], ["keytao_pending_items_by_words"])
        self.assertIsNone(self.store.get(self.key))

    async def test_textual_bot_mention_reaches_the_same_fresh_draft_lookup(self):
        with self.runtime():
            response = await self.turn("@喵喵 加亮 jslxa")
        self.assertIn("已在当前草稿", response)
        self.assertIn(BATCH_URL, response)
        self.assertEqual([name for name, _ in self.calls], ["keytao_pending_items_by_words"])
        self.assertIsNone(self.store.get(self.key))

    async def test_real_ttl_expiry_uses_fresh_pending_facts_without_model(self):
        self.store.set(self.key, incident_record())
        self.now += 11
        with self.runtime():
            response = await self.turn("加亮 jslxa")
        self.assertIn(BATCH_URL, response)
        self.assertEqual([name for name, _ in self.calls], ["keytao_pending_items_by_words"])
        self.assertIsNone(self.store.get(self.key))

    async def test_new_pair_prepares_exact_code_then_real_confirmation_writes_once(self):
        self.pending_items = []
        with self.runtime():
            response = await self.turn("加亮 jslxa")
            self.assertIn("尚未写入", response)
            self.assertIn("jslxa — 空位（推荐）", response)
            self.assertNotIn("jslxao", response)
            self.assertIn("jiā liàng", response)
            self.assertIn("需要管理员审核", response)
            state = self.store.get(self.key)
            self.assertIsInstance(state, harness.PendingAddWord)
            self.assertEqual(state.recommended_code, "jslxa")
            self.assertEqual(state.server_candidates, [("jslxa", False)])
            self.assertTrue(state.needs_manual_review)
            self.assertEqual(self.writes, [])
            self.assertEqual([name for name, _ in self.calls], [
                "keytao_pending_items_by_words", "keytao_prepare_reviewed_add", "keytao_lookup_by_code",
            ])
            delivered = chat._enforce_advertised_reply_contract(response, self.key)
            self.assertIn("jslxa", delivered)
            self.assertNotIn("jslxao", delivered)
            confirmed = await commands.handle_pending_message_core(
                "确认", "qq", self.key.actor_id, self.key, allow_intent_model=False,
            )
            self.assertIn("已", confirmed)
            await commands.handle_pending_message_core(
                "确认", "qq", self.key.actor_id, self.key, allow_intent_model=False,
            )
        self.assertEqual([(item["word"], item["code"]) for item in self.writes], [("加亮", "jslxa")])
        self.assertEqual(sum(name == "keytao_create_phrase" for name, _ in self.calls), 2)

    async def test_blocked_review_never_creates_a_confirmation_record(self):
        self.pending_items = []
        self.review["reviewDisposition"] = "BLOCK"
        with self.runtime():
            response = await self.turn("加亮 jslxa")
        self.assertIn("未写入", response)
        self.assertIsNone(self.store.get(self.key))
        self.assertEqual([name for name, _ in self.calls], [
            "keytao_pending_items_by_words", "keytao_prepare_reviewed_add",
        ])

    async def test_submitted_same_pair_is_reported_without_review_or_write(self):
        self.pending_items[0].update(source="submitted", batchStatus="Submitted")
        with self.runtime():
            response = await self.turn("加亮 JSLXA")
        self.assertIn("待审核批次", response)
        self.assertIn(BATCH_URL, response)
        self.assertEqual([name for name, _ in self.calls], ["keytao_pending_items_by_words"])

    async def test_unknown_pending_facts_never_become_an_empty_draft(self):
        for complete, items in ((False, []), (True, None), (True, {})):
            with self.subTest(complete=complete, items=items):
                self.setUp()
                self.pending_complete, self.pending_items = complete, items
                with self.runtime():
                    response = await self.turn("加亮 jslxa")
                self.assertIn("无法核验", response)
                self.assertIsNone(self.store.get(self.key))
                self.assertEqual([name for name, _ in self.calls], ["keytao_pending_items_by_words"])

    async def test_invalid_code_or_unreviewed_reading_never_arms_a_ticket(self):
        for code, changes in (
            ("evil", {}), ("jslxaaa", {}),
            ("jslxa", {"pronunciations": []}),
            ("jslxa", {"pronunciationUnresolved": True}),
            ("jslxa", {"word": "加量"}),
            ("jslxa", {"type": "Single"}),
        ):
            with self.subTest(code=code, changes=changes):
                self.setUp()
                self.pending_items = []
                self.review.update(changes)
                with self.runtime():
                    response = await self.turn("加亮 " + code)
                self.assertIn("未写入", response)
                self.assertIsNone(self.store.get(self.key))
                self.assertEqual([name for name, _ in self.calls], [
                    "keytao_pending_items_by_words", "keytao_prepare_reviewed_add",
                ])

    async def test_dictionary_same_identity_refuses_a_duplicate_after_exact_lookup(self):
        self.pending_items = []
        self.lookup["phrases"] = [{"word": "加亮", "code": "jslxa", "type": "Phrase", "weight": 100}]
        with self.runtime():
            response = await self.turn("加亮 jslxa")
        self.assertIn("不能重复添加", response)
        self.assertIsNone(self.store.get(self.key))
        self.assertEqual(self.writes, [])

    async def test_other_code_advisory_cannot_override_the_selected_code(self):
        self.pending_items = []
        self.review["pronunciations"][0]["candidateStatuses"][0].update(occupied=True, words=["加量"])
        self.review["candidateOrderingAssessments"] = [{
            "verdict": "front_more_common", "newWord": "加亮", "occupantWord": "加量",
            "occupantCode": "jslx", "freeCode": "jslxao", "newCode": "jslx",
        }]
        with self.runtime():
            response = await self.turn("加亮 jslxa")
            response = chat._enforce_advertised_reply_contract(response, self.key)
        self.assertIn("jslxa — 空位（推荐）", response)
        self.assertNotIn("需重排", response)
        self.assertEqual(self.store.get(self.key).recommended_code, "jslxa")
        self.assertEqual(self.store.get(self.key).server_ordering_assessments, [])

    async def test_group_additional_code_warning_keeps_a_real_actor_ticket(self):
        self.key = harness.ConversationAddress.group("qq", "s63-room", "s63-fresh")
        self.memory = harness.ChatMemoryContext(
            platform="qq", user_id=self.key.actor_id, space_type="group", space_id="s63-room",
        )
        self.pending_items = []
        self.review["existing"] = [{"word": "加亮", "code": "jslx", "type": "Phrase"}]
        self.create_warnings = [{
            "warningType": "multiple_code", "message": "该词已有其他编码，请核对是否继续添加。",
            "item": {"action": "Create", "word": "加亮", "code": "jslxa", "type": "Phrase"},
            "existing": self.review["existing"][0],
        }]
        with self.runtime():
            await self.turn("加亮 jslxa")
            response = await commands.handle_pending_message_core(
                "确认", "qq", self.key.actor_id, self.key,
                space_key=self.key.space_key, allow_intent_model=False,
            )
            self.assertEqual(self.writes, [])
            self.assertIn("其他编码", response)
            ticket = self.store.get(self.key)
            self.assertTrue(server_warning_ticket_is_complete(ticket))
            self.assertEqual(ticket.args["code"], "jslxa")
            self.assertIsNone(self.store.get(harness.ConversationAddress.private("qq", self.key.actor_id)))
            response = chat._enforce_advertised_reply_contract(response, self.key)
            self.assertIn("确认", response)
            await commands.handle_pending_message_core(
                "确认", "qq", self.key.actor_id, self.key,
                space_key=self.key.space_key, allow_intent_model=False,
            )
        self.assertEqual([(item["word"], item["code"]) for item in self.writes], [("加亮", "jslxa")])
        self.assertEqual(len([name for name, _args in self.calls if name == "keytao_create_phrase"]), 2)

    async def test_foreign_actor_ticket_is_never_used_for_a_fresh_pair(self):
        self.key = harness.ConversationAddress.group("qq", "s63-room", "s63-fresh")
        self.memory = harness.ChatMemoryContext(
            platform="qq", user_id=self.key.actor_id, space_type="group", space_id="s63-room",
        )
        other = harness.ConversationAddress.group("qq", "s63-room", "s63-other")
        self.store.set(other, incident_record())
        original = self.store.get_record(other)
        with self.runtime():
            response = await self.turn("加亮 jslxa")
        self.assertIn(BATCH_URL, response)
        self.assertIs(self.store.get_record(other), original)
        self.assertIsNone(self.store.get(self.key))
        self.assertEqual([name for name, _ in self.calls], ["keytao_pending_items_by_words"])

    async def test_negative_quoted_and_extra_clause_pairs_never_enter_preparation(self):
        self.assertFalse(chat._chat_routing.message_authorizes_mutation("加亮 jslxa"))
        for message in (
            "不要加亮 jslxa", "别加亮 jslxa", "他说加亮 jslxa",
            "据说加亮 jslxa", "「加亮 jslxa」", "加亮 jslxa？",
            "加亮 jslxa，然后删除", "加亮 jslxa；提交", "加亮 3",
            "加亮 jslxa，小端 xcdti",
        ):
            with self.subTest(message=message):
                self.setUp()
                self.assertIsNone(commands.fresh_entry_code_selection(message))
                with self.runtime(), patch.object(
                    chat, "_classify_message_command_intent", AsyncMock(return_value=harness.MessageCommandIntent()),
                ), patch.object(
                    chat, "_classify_simple_word_query_intent", AsyncMock(return_value=[]),
                ), patch.object(chat, "get_ai_response_core", AsyncMock(return_value="只读说明")):
                    await self.turn(message)
                self.assertEqual(self.calls, [])
                self.assertIsNone(self.store.get(self.key))

    async def test_unrelated_live_ticket_is_not_replaced_by_fresh_preparation(self):
        pending = harness.PendingToolConfirm("keytao_submit_batch", {"batch_id": "s63-unrelated"})
        self.store.set(self.key, pending)
        original = self.store.get_record(self.key)
        with self.runtime(), patch.object(
            chat, "_classify_message_command_intent", AsyncMock(return_value=harness.MessageCommandIntent()),
        ), patch.object(chat, "get_ai_response_core", AsyncMock(return_value="只读说明")):
            await self.turn("加亮 jslxa")
        self.assertIs(self.store.get_record(self.key), original)
        self.assertEqual(self.calls, [])

    async def test_quoted_old_display_does_not_create_a_new_ticket(self):
        reply = harness.ReplyReferenceInfo(
            is_reply=True, is_to_bot=True,
            text=chat.render_server_backed_single_word_candidates(
                "加亮", "jslxa", [("jslxa", False)], {},
            ),
        )
        with self.runtime(), patch.object(
            chat, "_classify_message_command_intent", AsyncMock(return_value=harness.MessageCommandIntent()),
        ), patch.object(chat, "get_ai_response_core", AsyncMock(return_value="只读说明")):
            await self.turn("加亮 jslxa", reply_reference=reply)
        self.assertEqual(self.calls, [])
        self.assertIsNone(self.store.get(self.key))

    async def test_live_s62_pair_keeps_its_existing_sealed_selection_executor(self):
        self.store.set(self.key, incident_record())
        with self.runtime():
            response = await self.turn("加亮 jslxa")
        self.assertIn("已", response)
        self.assertEqual([(item["word"], item["code"]) for item in self.writes], [("加亮", "jslxa")])
        self.assertNotIn("keytao_pending_items_by_words", [name for name, _ in self.calls])
        self.assertEqual(sum(name == "keytao_batch_add_to_draft" for name, _ in self.calls), 2)


if __name__ == "__main__":
    unittest.main()
