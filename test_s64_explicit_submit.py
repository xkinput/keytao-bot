"""Offline S64 replay of a fresh explicit word/code add-and-submit turn."""

import copy
import json
import unittest
from contextlib import contextmanager
from unittest.mock import Mock, patch

import test_s63_fresh_selection as fresh


INSTRUCTION = "加亮 jslxa，加入草稿并提交。"
REAL_COMMAND_CLASSIFIER = fresh.chat._classify_message_command_intent
REAL_WORD_CLASSIFIER = fresh.chat._classify_simple_word_query_intent
REAL_TOOL_DISPATCH = fresh.commands.call_tool_function


class ExplicitSubmitTests(unittest.IsolatedAsyncioTestCase):

    @contextmanager
    def runtime(self):
        client = Mock(side_effect=AssertionError("An explicit word/code command must not call a model"))
        with fresh.FreshSelectionTests.runtime(self), patch.object(
            fresh.chat, "_classify_message_command_intent", REAL_COMMAND_CLASSIFIER,
        ), patch.object(
            fresh.chat, "_classify_simple_word_query_intent", REAL_WORD_CLASSIFIER,
        ), patch.object(fresh.chat._chat_routing, "AsyncOpenAI", client), patch.object(
            fresh.chat._chat_routing, "OPENAI_API_KEY", "s64-fixture-key",
        ):
            try:
                yield
            finally:
                client.assert_not_called()

    def setUp(self):
        fresh.FreshSelectionTests.setUp(self)
        self.pending_items = []
        self.batch_id = "s64-draft"
        self.batch_url = "https://keytao.rea.ink/batch/" + self.batch_id
        self.items = []
        self.version = 0
        self.submitted = False

    async def turn(self, message, **kwargs):
        token = fresh.chat._current_turn_message.set(message)
        try:
            return await fresh.FreshSelectionTests.turn(self, message, **kwargs)
        finally:
            fresh.chat._current_turn_message.reset(token)

    async def tool(self, name, args, platform, user_id, **kwargs):
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
            fresh.commands._capture_successful_draft_write_delivery(name, result, platform, user_id)
        return json.dumps(result, ensure_ascii=False)

    @contextmanager
    def executor_runtime(self):
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
            fresh.chat, "call_tool_function", REAL_TOOL_DISPATCH,
        ), patch.object(
            fresh.commands.tool_executor, "_invoke_effective_tool", side_effect=transport,
        ), patch.object(fresh.commands.memory_store, "record_tool_receipt"):
            yield

    def seed_existing_draft(self, *, other_item=False):
        row = {"id": 6401, "action": "Create", "word": "加亮", "code": "jslxa", "type": "Phrase"}
        self.items = [row]
        self.version = 3
        self.pending_items = [{
            **row, "source": "draft", "batchId": self.batch_id, "batchUrl": self.batch_url,
            "batchStatus": "Draft", "itemStatus": "Pending",
        }]
        if other_item:
            self.items.append({"id": 6402, "action": "Create", "word": "旧词", "code": "qtxx", "type": "Phrase"})

    def assert_submitted_or_actionable_submit_ticket(self, response):
        self.assertNotIn("本轮没有可执行的已绑定写操作", response)
        if self.submitted:
            self.assertIn("已提交", response)
            return
        ticket = self.store.get(self.key)
        self.assertTrue(fresh.server_warning_ticket_is_complete(ticket), response)
        self.assertEqual(ticket.function_name, "keytao_submit_batch")
        self.assertEqual(ticket.args["batch_id"], self.batch_id)
        delivered = fresh.chat._enforce_advertised_reply_contract(response, self.key)
        self.assertIn("确认", delivered)
        self.assertIn(self.batch_url, delivered)

    async def test_exact_fresh_instruction_does_not_fall_through_to_intent_model(self):
        with self.runtime():
            response = await self.turn(INSTRUCTION)
            self.assert_submitted_or_actionable_submit_ticket(response)
        names = [name for name, _ in self.calls]
        self.assertIn("keytao_prepare_reviewed_add", names)
        self.assertIn("keytao_lookup_by_code", names)
        self.assertEqual(sum(name == "keytao_create_phrase" for name in names), 2)
        self.assertEqual([(row["word"], row["code"]) for row in self.writes], [("加亮", "jslxa")])
        self.assertIn("keytao_submit_batch", names)

    async def test_expired_candidate_requires_fresh_review_and_preserves_submit_intent(self):
        self.store.set(self.key, fresh.incident_record())
        self.now += 11
        with self.runtime():
            response = await self.turn(INSTRUCTION)
            self.assert_submitted_or_actionable_submit_ticket(response)
        self.assertIn("keytao_prepare_reviewed_add", [name for name, _ in self.calls])
        self.assertEqual([(row["word"], row["code"]) for row in self.writes], [("加亮", "jslxa")])

    async def test_existing_same_draft_is_submitted_without_a_duplicate_create(self):
        self.seed_existing_draft()
        with self.runtime():
            response = await self.turn(INSTRUCTION)
            self.assert_submitted_or_actionable_submit_ticket(response)
        names = [name for name, _ in self.calls]
        self.assertIn("keytao_submit_batch", names)
        self.assertNotIn("keytao_create_phrase", names)
        self.assertEqual(self.writes, [])

    async def test_existing_draft_with_other_words_requires_a_real_snapshot_confirmation(self):
        self.seed_existing_draft(other_item=True)
        with self.runtime():
            response = await self.turn(INSTRUCTION)
            self.assertFalse(self.submitted)
            self.assert_submitted_or_actionable_submit_ticket(response)
            self.assertIn("旧词", response)
            self.assertIn("qtxx", response)
            confirmed = await fresh.commands.handle_pending_message_core(
                "确认", "qq", self.key.actor_id, self.key,
                space_key=self.key.space_key, allow_intent_model=False,
            )
            self.assertTrue(self.submitted, confirmed)
        self.assertEqual(self.writes, [])

    async def test_group_mention_uses_the_speakers_own_empty_draft(self):
        self.key = fresh.harness.ConversationAddress.group("qq", "s64-room", "s64-rea")
        self.memory = fresh.harness.ChatMemoryContext(
            platform="qq", user_id=self.key.actor_id, space_type="group", space_id="s64-room",
        )
        other_key = fresh.harness.ConversationAddress.group("qq", "s64-room", "s64-other")
        self.store.set(other_key, fresh.incident_record())
        other_record = self.store.get_record(other_key)
        with self.runtime():
            response = await self.turn("@喵喵 " + INSTRUCTION)
            self.assert_submitted_or_actionable_submit_ticket(response)
        self.assertIs(self.store.get_record(other_key), other_record)
        self.assertIsNone(self.store.get(fresh.harness.ConversationAddress.private("qq", self.key.actor_id)))

    async def test_action_first_form_keeps_the_existing_reviewed_submit_path(self):
        with self.runtime():
            response = await self.turn("添加 加亮 jslxa 并提交")
            self.assert_submitted_or_actionable_submit_ticket(response)
        self.assertEqual([(row["word"], row["code"]) for row in self.writes], [("加亮", "jslxa")])

    async def test_blocked_or_unresolved_review_never_arms_or_writes(self):
        for change in ({"reviewDisposition": "BLOCK"}, {"pronunciationUnresolved": True}):
            with self.subTest(change=change):
                self.setUp()
                self.review.update(change)
                with self.runtime():
                    response = await self.turn(INSTRUCTION)
                self.assertIn("未写入", response)
                self.assertEqual(self.writes, [])
                self.assertFalse(self.submitted)
                self.assertIsNone(self.store.get(self.key))
                self.assertEqual([name for name, _ in self.calls], [
                    "keytao_pending_items_by_words", "keytao_prepare_reviewed_add",
                ])

    async def test_unknown_pending_facts_cannot_be_treated_as_an_empty_draft(self):
        for complete, items in ((False, []), (True, None), (True, {})):
            with self.subTest(complete=complete, items=items):
                self.setUp()
                self.pending_complete, self.pending_items = complete, items
                with self.runtime():
                    response = await self.turn(INSTRUCTION)
                self.assertIn("无法核验", response)
                self.assertEqual(self.writes, [])
                self.assertFalse(self.submitted)
                self.assertIsNone(self.store.get(self.key))
                self.assertEqual([name for name, _ in self.calls], ["keytao_pending_items_by_words"])

    async def test_new_word_preserves_other_draft_items_and_requires_snapshot_confirmation(self):
        old_row = {"id": 6402, "action": "Create", "word": "旧词", "code": "qtxx", "type": "Phrase"}
        self.items = [copy.deepcopy(old_row)]
        self.version = 3
        with self.runtime():
            response = await self.turn(INSTRUCTION)
            self.assertFalse(self.submitted)
            self.assert_submitted_or_actionable_submit_ticket(response)
            self.assertIn("旧词", response)
            self.assertIn("qtxx", response)
            self.assertEqual(self.items[0], old_row)
            confirmed = await fresh.commands.handle_pending_message_core(
                "确认", "qq", self.key.actor_id, self.key,
                space_key=self.key.space_key, allow_intent_model=False,
            )
            self.assertTrue(self.submitted, confirmed)
        self.assertEqual([(row["word"], row["code"]) for row in self.writes], [("加亮", "jslxa")])
        self.assertEqual(self.items[0], old_row)

    async def test_live_review_keeps_the_explicit_code_instead_of_its_recommendation(self):
        candidates = [("jslx", False), ("jslxa", False), ("jslxao", False)]
        state = fresh.harness.PendingAddWord(
            word="加亮", recommended_code="jslxao", candidates=candidates,
            server_candidates=list(candidates), phrase_type="Phrase", needs_manual_review=True,
            manual_review_reason="常用度证据不足",
            pronunciation_codes={code: "jiā liàng" for code, _ in candidates},
            code_remarks={code: "喵喵审词：读音 jiā liàng；常用度证据不足" for code, _ in candidates},
        )
        self.store.set(self.key, state)
        with self.runtime():
            response = await self.turn(INSTRUCTION)
            self.assert_submitted_or_actionable_submit_ticket(response)
        self.assertEqual([(row["word"], row["code"]) for row in self.writes], [("加亮", "jslxa")])

    async def test_real_dispatcher_and_executor_create_and_submit_cas_with_reviewed_capability(self):
        with self.executor_runtime():
            response = await self.turn(INSTRUCTION)
            self.assertTrue(self.submitted, response)
            self.assertIn("已提交", response)
        self.assertEqual([name for name, _ in self.calls], [
            "keytao_pending_items_by_words", "keytao_prepare_reviewed_add", "keytao_lookup_by_code",
            "keytao_create_phrase", "keytao_create_phrase", "keytao_submit_batch", "keytao_submit_batch",
        ])
        self.assertEqual([(row["word"], row["code"]) for row in self.writes], [("加亮", "jslxa")])

    async def test_real_dispatcher_keeps_existing_other_items_behind_submit_confirmation(self):
        self.seed_existing_draft(other_item=True)
        with self.executor_runtime():
            response = await self.turn(INSTRUCTION)
            self.assertFalse(self.submitted)
            self.assert_submitted_or_actionable_submit_ticket(response)
            self.assertIn("旧词", response)
            confirmed = await fresh.commands.handle_pending_message_core(
                "确认", "qq", self.key.actor_id, self.key,
                space_key=self.key.space_key, allow_intent_model=False,
            )
            self.assertTrue(self.submitted, confirmed)
        self.assertEqual([name for name, _ in self.calls], [
            "keytao_pending_items_by_words", "keytao_submit_batch", "keytao_submit_batch",
        ])
        self.assertEqual(self.writes, [])

    async def test_submit_ticket_save_failure_keeps_reason_and_link_without_confirmation_advertisement(self):
        from keytao_bot.utils.pending_confirmation import advertised_command_suggestions

        self.seed_existing_draft(other_item=True)
        with self.runtime(), patch.object(self.store, "set", return_value=False):
            response = await self.turn(INSTRUCTION)
            delivered = fresh.chat._enforce_advertised_reply_contract(response, self.key)
        self.assertFalse(self.submitted)
        self.assertEqual(self.writes, [])
        self.assertIsNone(self.store.get(self.key))
        self.assertIn("提交确认记录未能保存", delivered)
        self.assertIn(self.batch_url, delivered)
        self.assertFalse(advertised_command_suggestions(delivered))

    async def test_deferred_negative_prefixes_cannot_submit_a_matching_literal_draft(self):
        from keytao_bot.utils.explicit_code import parse_explicit_entry_code_request

        for word in ("先不加亮", "暂不加亮", "暂时不加亮", "请先不加亮"):
            with self.subTest(word=word):
                self.setUp()
                self.seed_existing_draft()
                self.items[0]["word"] = word
                self.pending_items[0]["word"] = word
                message = word + " jslxa，加入草稿并提交。"
                self.assertIsNone(parse_explicit_entry_code_request(message))
                with self.runtime():
                    response = await fresh.commands.try_handle_explicit_entry_code_command(
                        message, "qq", self.key.actor_id, self.key,
                        self.key.space_key, "Rea",
                    )
                self.assertIsNone(response)
                self.assertEqual(self.calls, [])
                self.assertEqual(self.writes, [])
                self.assertFalse(self.submitted)
                self.assertIsNone(self.store.get(self.key))


class ExplicitSubmitGrammarTests(unittest.TestCase):
    def test_named_pair_and_action_forms_preserve_submit_intent(self):
        from keytao_bot.utils.explicit_code import parse_explicit_entry_code_request

        for message in (
            INSTRUCTION, "加亮 jslxa,加入草稿并提交。", "加亮 jslxa 加入草稿并提交",
            "加亮 jslxa，添加到草稿并提交。", "添加 加亮 jslxa 并提交",
        ):
            with self.subTest(message=message):
                request = parse_explicit_entry_code_request(message)
                self.assertIsNotNone(request)
                self.assertEqual((request.word, request.code, request.submit_after), ("加亮", "jslxa", True))

    def test_negative_reported_quoted_and_extra_target_forms_are_not_commands(self):
        from keytao_bot.utils.explicit_code import parse_explicit_entry_code_request

        for message in (
            "不要加亮 jslxa，加入草稿并提交。", "他说加亮 jslxa，加入草稿并提交。",
            "「加亮 jslxa，加入草稿并提交。」", "加亮 jslxa，加入草稿并提交？",
            "加亮 jslxa，小端 xcdti，加入草稿并提交。",
            "加亮 jslxa，加入草稿并提交，然后删除加量。",
            "加亮 jslxa，加入到",
        ):
            with self.subTest(message=message):
                self.assertIsNone(parse_explicit_entry_code_request(message))


if __name__ == "__main__":
    unittest.main()
