"""Offline model-first add routing through the real reviewed write handlers."""

import asyncio
import copy
import json
import os
import unittest
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from test_semantic_parser import FakeClient, raw_operation

from keytao_bot.harness.conversation import ConversationAddress
from keytao_bot.harness.operation_request import OperationRequest, understand_message
from keytao_bot.harness.state import (
    DraftOperationCoordinator,
    MemoryConversationStateStore,
    PendingAddWord,
    PendingToolConfirm,
)


class SemanticAddRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.chat = harness.openai_chat_module
        self.commands = harness.chat_commands_module
        self.store = MemoryConversationStateStore()
        self.author = ConversationAddress.group("qq", "semantic-add-group", "semantic-add-owner")
        self.other = ConversationAddress.group("qq", "semantic-add-group", "other-owner")
        self.client = FakeClient()
        self.finish = AsyncMock()
        self.calls = []
        self.writes = []
        self.warning = None
        self.review_changes = {}
        self.codes = {"燃面": "rfmm", "晕碳": "ywtfo", "清波": "qgbla"}
        self.readings = {"燃面": "ran mian", "晕碳": "yun tan", "清波": "qing bo"}
        self.tool = AsyncMock(side_effect=self.call)
        patches = (
            patch.dict(os.environ, {
                "KEYTAO_SEMANTIC_ROUTE_OPS": "add,direct_exec",
                "KEYTAO_SEMANTIC_SHADOW": "0",
            }),
            patch.object(self.chat, "AsyncOpenAI", return_value=self.client),
            patch.object(self.chat, "WORD_QUERY_INTENT_MODEL", "offline-model"),
            patch.object(self.chat, "conversation_state_store", self.store),
            patch.object(self.chat, "draft_operation_coordinator", DraftOperationCoordinator()),
            patch.object(self.chat, "call_tool_function", self.tool),
            patch.object(self.chat, "remember_conversation"),
            patch.object(self.chat, "_finish_ai_chat_response", self.finish),
            patch.object(self.chat.logger, "info"),
            patch.object(self.chat.logger, "warning"),
            patch.object(self.commands.user_resolver, "resolve_actor_binding", AsyncMock(return_value=True)),
            patch.object(self.commands, "_format_draft_response", AsyncMock(return_value="Offline draft receipt")),
        )
        for context in patches:
            context.start()
            self.addCleanup(context.stop)
        token = self.chat._current_conversational_turn.set(None)
        self.addCleanup(self.chat._current_conversational_turn.reset, token)

    async def asyncTearDown(self):
        tasks = tuple(self.chat._semantic_shadow_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.chat._semantic_shadow_tasks.clear()

    def result(self, **changes):
        return {
            "success": True, "batchId": "offline-add-batch", "contentVersion": 1,
            "batchUrl": "https://offline.invalid/draft/offline-add-batch", **changes,
        }

    async def call(self, name, args, platform, user_id, **kwargs):
        self.assertEqual((platform, user_id), ("qq", self.author.actor_id))
        self.calls.append((name, copy.deepcopy(args)))
        if name == "keytao_prepare_reviewed_add":
            self.assertEqual(set(args), {"word"})
            word = args["word"]
            code = self.codes[word]
            return json.dumps({
                "success": True, "word": word, "type": "Phrase",
                "reviewDisposition": "PASS", "recommendedCode": code,
                "needsManualReview": False,
                "preSubmitAudit": {
                    "success": True, "verdict": "pass", "autoApprove": True,
                    "summary": "Offline reviewed fixture", "issues": [],
                },
                "pronunciations": [{
                    "pinyin": self.readings[word], "recommendedCode": code,
                    "sources": [{"source": "offline"}],
                    "candidateStatuses": [
                        {"code": value, "occupied": False, "label": "空位", "words": []}
                        for value in (code, code + "o")
                    ],
                }],
                **self.review_changes,
            }, ensure_ascii=False)
        if name in {"keytao_lookup_by_word", "keytao_lookup_by_code"}:
            return json.dumps({"success": True, "phrases": []})
        if name == "keytao_pending_items_by_words":
            return json.dumps({"success": True, "complete": True, "items": []})
        self.assertIn(name, {"keytao_create_phrase", "keytao_batch_add_to_draft", "keytao_submit_batch"},
                      "Unexpected offline tool")
        if name == "keytao_submit_batch":
            if not args.get("confirmed"):
                return json.dumps(self.result(
                    success=False, requiresConfirmation=True,
                    snapshotItems=[{
                        "action": "Create", "word": "燃面", "code": "rfmm", "type": "Phrase",
                    }],
                    snapshotDigest="b" * 64, warningDigest="c" * 64,
                    auditDigest="d" * 64, warnings=[],
                ))
            self.writes.append((name, copy.deepcopy(args)))
            return json.dumps(self.result())
        if not args.get("confirmed"):
            return json.dumps(self.result(
                success=False, requiresConfirmation=True, warningDigest="a" * 64,
                warnedCount=1 if self.warning else 0,
                warnings=[self.warning] if self.warning else [],
            ), ensure_ascii=False)
        self.writes.append((name, copy.deepcopy(args)))
        return json.dumps(self.result(
            successCount=len(args["items"]) if "items" in args else 1,
            failedCount=0,
        ))

    def set_model(self, operands=("燃面",), codes=("rfmm",), **changes):
        self.client.create.side_effect = None
        payload = raw_operation(op="add", operands=list(operands), codes=list(codes), **changes)
        self.client.create.return_value.choices[0].message.content = json.dumps(payload, ensure_ascii=False)
        return payload

    def context(self, message):
        return self.chat.TurnContext(
            bot=None, event=None, platform="qq", user_id=self.author.actor_id,
            message_text=message, normalized_message_text=message,
            conv_key=self.author, space_key=self.author.space_key,
            owner_label="Offline owner",
        )

    async def stage(self, message):
        ctx = self.context(message)
        return await self.chat._stage_understand_operation_early(ctx), ctx

    async def answer(self, message, actor=None):
        actor = actor or self.author
        return await self.commands.handle_pending_message_core(
            message, "qq", actor.actor_id, actor, space_key=actor.space_key,
            allow_intent_model=False,
        )

    def added_pairs(self):
        pairs = []
        for name, args in self.writes:
            if name == "keytao_create_phrase":
                pairs.append((args["word"], args["code"]))
            elif name == "keytao_batch_add_to_draft":
                pairs.extend((item["word"], item["code"]) for item in args["items"])
        return pairs

    async def assert_fallback(self, message):
        handled, ctx = await self.stage(message)
        self.assertFalse(handled, ctx.response)
        self.assertEqual(self.writes, [])
        self.assertEqual(self.calls, [])
        self.assertIsNone(self.store.get_record(self.author))
        return ctx

    async def test_glued_production_pairs_write_one_exact_batch_without_candidates(self):
        pairs = (("燃面", "rfmm"), ("晕碳", "ywtfo"), ("清波", "qgbla"))
        self.set_model(tuple(word for word, _ in pairs), tuple(code for _, code in pairs))
        with patch.object(self.commands, "_try_handle_simple_single_word_query", AsyncMock()) as candidates:
            handled, ctx = await self.stage("@喵喵 加词 燃面rfmm 晕碳ywtfo 清波qgbla")
        self.assertTrue(handled, ctx.response)
        self.assertEqual(self.added_pairs(), list(pairs))
        self.assertEqual([name for name, _ in self.writes], ["keytao_batch_add_to_draft"])
        candidates.assert_not_awaited()
        self.assertEqual([args["word"] for name, args in self.calls if name == "keytao_prepare_reviewed_add"],
                         [word for word, _ in pairs])
        self.assertIsNone(self.store.get_record(self.author))
        self.client.create.assert_awaited_once()

    async def test_natural_single_pair_with_extra_word_characters_falls_back(self):
        self.set_model()
        await self.assert_fallback("把燃面加到rfmm")

    async def test_single_word_without_code_uses_normal_candidate_ticket(self):
        self.set_model(codes=())
        handled, ctx = await self.stage("加个词 燃面")
        self.assertTrue(handled, ctx.response)
        self.assertEqual(self.writes, [])
        self.assertIsInstance(self.store.get(self.author), PendingAddWord)
        self.assertEqual(self.store.get(self.author).word, "燃面")
        self.assertIn("1. rfmm", ctx.response)
        self.assertTrue(self.chat._advertised_reply_matches_live_record(ctx.response, self.store.get_record(self.author)))

    async def test_words_without_codes_share_normal_multiword_ticket(self):
        self.set_model(("燃面", "晕碳"), ())
        handled, ctx = await self.stage("加词 燃面 晕碳")
        self.assertTrue(handled, ctx.response)
        self.assertEqual(self.writes, [])
        state = self.store.get(self.author)
        self.assertIsInstance(state, PendingToolConfirm)
        self.assertEqual(state.args["_query_words"], ["燃面", "晕碳"])
        self.assertTrue(self.chat._advertised_reply_matches_live_record(ctx.response, self.store.get_record(self.author)))

    async def test_mixed_attached_pair_and_uncoded_word_keep_exact_scope(self):
        self.set_model(("燃面", "晕碳"), ("rfmm",))
        handled, ctx = await self.stage("加词 燃面rfmm 晕碳")
        self.assertTrue(handled, ctx.response)
        self.assertEqual(self.added_pairs(), [("燃面", "rfmm")])
        state = self.store.get(self.author)
        self.assertIsInstance(state, PendingAddWord)
        self.assertEqual(state.word, "晕碳")
        self.assertIn("1. ywtfo", ctx.response)

    async def test_attached_pairs_override_model_code_order(self):
        self.set_model(("燃面", "晕碳"), ("ywtfo", "rfmm"))
        handled, ctx = await self.stage("加词 燃面rfmm 晕碳ywtfo")
        self.assertTrue(handled, ctx.response)
        self.assertEqual(self.added_pairs(), [("燃面", "rfmm"), ("晕碳", "ywtfo")])

    async def test_separate_equal_order_lists_pair_exactly(self):
        self.set_model(("燃面", "晕碳"), ("rfmm", "ywtfo"))
        handled, ctx = await self.stage("添加 燃面、晕碳，rfmm、ywtfo")
        self.assertTrue(handled, ctx.response)
        self.assertEqual(self.added_pairs(), [("燃面", "rfmm"), ("晕碳", "ywtfo")])

    async def test_separate_code_list_with_unequal_counts_falls_back(self):
        self.set_model(("燃面", "晕碳", "清波"), ("rfmm", "ywtfo"))
        await self.assert_fallback("添加燃面、晕碳、清波，编码是rfmm、ywtfo")

    async def test_glued_pair_conflict_cannot_be_overridden_by_model_list_order(self):
        self.set_model(("燃面", "晕碳"), ("rfmm", "ywtfo"))
        await self.assert_fallback("加词 燃面 晕碳rfmm ywtfo")

    async def test_unknown_shape_suffix_keeps_manual_review_reason_in_one_batch_write(self):
        self.set_model(("燃面", "晕碳"), ("rfmmx", "ywtfo"))
        handled, ctx = await self.stage("加词 燃面rfmmx 晕碳ywtfo")
        self.assertTrue(handled, ctx.response)
        self.assertEqual(self.added_pairs(), [("燃面", "rfmmx"), ("晕碳", "ywtfo")])
        self.assertEqual([name for name, _ in self.writes], ["keytao_batch_add_to_draft"])
        item = self.writes[0][1]["items"][0]
        self.assertIs(item["needsManualReview"], True)
        self.assertIn("ran mian", item["remark"])
        self.assertIn("形码 x 未能核验，需管理员复核", item["remark"])

    async def test_literal_word_and_code_hallucinations_fall_back(self):
        for operands, codes in ((("燃面", "清波"), ("rfmm",)), (("燃面",), ("ywtfo",))):
            with self.subTest(operands=operands, codes=codes):
                self.set_model(operands, codes)
                await self.assert_fallback("加词 燃面rfmm")

    async def test_one_word_cannot_consume_two_codes(self):
        self.set_model(("燃面",), ("rfmm", "ywtfo"))
        await self.assert_fallback("加词 燃面 rfmm ywtfo")

    async def test_one_code_cannot_be_reused_for_two_words(self):
        self.set_model(("燃面", "晕碳"), ("rfmm", "rfmm"))
        await self.assert_fallback("加词 燃面rfmm 晕碳rfmm")

    async def test_code_substring_does_not_authorize_a_shorter_model_code(self):
        self.set_model()
        await self.assert_fallback("加词 燃面rfmmo")

    async def test_omitted_code_from_separate_lists_falls_back(self):
        self.set_model(("燃面", "晕碳"), ("rfmm",))
        await self.assert_fallback("加词 燃面 晕碳 rfmm ywtfo")

    async def test_omitted_second_code_for_one_word_falls_back(self):
        self.set_model()
        await self.assert_fallback("加词 燃面 rfmm ywtfo")

    async def test_truncated_glued_operand_falls_back(self):
        self.set_model()
        await self.assert_fallback("加词 燃面皮rfmm")

    async def test_latin_prefixed_operand_falls_back(self):
        self.set_model()
        await self.assert_fallback("AB燃面 rfmm")

    async def test_omitted_tokens_are_checked_case_insensitively_and_when_glued(self):
        for message in ("加词 燃面 rfmm YwTfO", "加词 燃面rfmm 晕碳ywtfo", "加词 AB燃面 rfmm"):
            with self.subTest(message=message):
                self.set_model()
                await self.assert_fallback(message)

    async def test_omitted_code_and_operand_from_separate_lists_fall_back(self):
        self.set_model(("晕碳",), ("rfmm",))
        await self.assert_fallback("加词 燃面 晕碳 rfmm ywtfo")

    async def test_word_tokens_must_match_whole_operands(self):
        for message in (
            "加词 燃面皮 rfmm", "加词 皮燃面rfmm", "加词 燃面2rfmm",
            "加词燃面rfmm", "添加燃面 rfmm", "加词 燃面rfmm 清波",
        ):
            with self.subTest(message=message):
                self.set_model()
                await self.assert_fallback(message)

    async def test_order_pairing_cannot_ignore_extra_word_tokens(self):
        for message in (
            "加词 燃面 晕碳 清波 rfmm ywtfo",
            "加词 燃面晕碳 rfmm ywtfo",
            "添加燃面、晕碳，编码依次是rfmm、ywtfo",
        ):
            with self.subTest(message=message):
                self.set_model(("燃面", "晕碳"), ("rfmm", "ywtfo"))
                await self.assert_fallback(message)

    async def test_complete_case_insensitive_codes_and_punctuation_keep_exact_pairs(self):
        for message in ("@喵喵 添加：燃面，RFMM", "加词 燃面rfmm"):
            with self.subTest(message=message):
                self.writes.clear()
                self.set_model()
                handled, ctx = await self.stage(message)
                self.assertTrue(handled, ctx.response)
                self.assertEqual(self.added_pairs(), [("燃面", "rfmm")])

    async def test_latin_tokens_with_no_model_codes_fall_back(self):
        self.set_model(codes=())
        await self.assert_fallback("加个词 燃面rfmm")

    async def test_question_negated_and_reported_frames_never_write(self):
        cases = (
            ("加词 燃面rfmm", "question"),
            ("可以把燃面加到rfmm吗？", "direct"),
            ("不要加词 燃面rfmm", "negated"),
            ("不要加词 燃面rfmm", "direct"),
            ("他说加词 燃面rfmm", "direct"),
        )
        for message, frame in cases:
            with self.subTest(message=message, frame=frame):
                self.set_model(frame=frame)
                await self.assert_fallback(message)

    async def test_pending_reference_cannot_route_as_fresh_add(self):
        self.set_model(refers_to_pending=True)
        await self.assert_fallback("加词 燃面rfmm")

    async def test_invalid_model_boolean_fields_fail_closed(self):
        for field, value in (("refers_to_pending", "false"), ("submit_after", "false")):
            with self.subTest(field=field):
                self.set_model(**{field: value})
                await self.assert_fallback("加词 燃面rfmm")

    async def test_missing_and_nonexact_add_tokens_keep_today_path(self):
        for token in (None, "", "adds,direct_exec", "ADD,direct_exec", "add;direct_exec", "direct_exec"):
            with self.subTest(token=token):
                if token is None:
                    os.environ.pop("KEYTAO_SEMANTIC_ROUTE_OPS", None)
                else:
                    os.environ["KEYTAO_SEMANTIC_ROUTE_OPS"] = token
                self.set_model()
                self.client.create.reset_mock()
                await self.assert_fallback("加词 燃面rfmm")
                self.client.create.assert_not_awaited()

    async def test_model_failure_and_invalid_output_keep_today_path(self):
        for scenario in ("timeout", "invalid_json", "invalid_op"):
            with self.subTest(scenario=scenario):
                self.set_model()
                if scenario == "timeout":
                    self.client.create.side_effect = asyncio.TimeoutError()
                elif scenario == "invalid_json":
                    self.client.create.return_value.choices[0].message.content = "not JSON"
                else:
                    self.client.create.return_value.choices[0].message.content = '{"op":"ADD"}'
                await self.assert_fallback("加词 燃面rfmm")

    async def test_without_direct_exec_seals_exact_ticket_then_author_executes_once(self):
        os.environ["KEYTAO_SEMANTIC_ROUTE_OPS"] = "add"
        self.set_model(("燃面", "晕碳"), ("rfmm", "ywtfo"))
        handled, ctx = await self.stage("加词 燃面rfmm 晕碳ywtfo")
        self.assertTrue(handled, ctx.response)
        self.assertEqual(self.calls, [])
        record = self.store.get_record(self.author)
        self.assertIsInstance(record.state, PendingToolConfirm)
        self.assertEqual(record.state.function_name, "keytao_understood_command")
        canonical = record.state.args["command_text"]
        self.assertEqual(canonical, "添加 燃面 rfmm；添加 晕碳 ywtfo")
        self.assertIsNone(await self.answer("确认", self.other))
        self.assertIs(self.store.get_record(self.author), record)
        self.assertIn("已加入草稿", await self.answer("确认"))
        self.assertEqual(self.added_pairs(), [("燃面", "rfmm"), ("晕碳", "ywtfo")])
        self.assertIsNone(await self.answer("确认"))
        self.assertEqual(len(self.writes), 1)

    def test_canonical_add_commands_round_trip_through_closed_parser(self):
        for words, codes in ((("燃面",), ("rfmm",)), (("燃面", "晕碳"), ("rfmm", "ywtfo"))):
            with self.subTest(words=words):
                request = OperationRequest(op="add", operands=words, codes=codes)
                command = self.commands.render_understood_command(request)
                self.assertTrue(command)
                parsed = understand_message(command)
                self.assertEqual((parsed.op, parsed.operands, parsed.codes, parsed.submit_after),
                                 ("add", words, codes, False))

    def test_single_pair_submit_canonical_round_trips(self):
        request = OperationRequest(op="add", operands=("燃面",), codes=("rfmm",), submit_after=True)
        command = self.commands.render_understood_command(request)
        self.assertTrue(command)
        parsed = understand_message(command)
        self.assertEqual((parsed.op, parsed.operands, parsed.codes, parsed.submit_after),
                         ("add", ("燃面",), ("rfmm",), True))

    async def test_single_and_batch_keep_unacknowledged_server_warning(self):
        self.warning = {
            "warningType": "manual_review_required", "word": "燃面", "code": "rfmm",
            "message": "Offline warning requires confirmation",
        }
        for operands, codes, message in (
            (("燃面",), ("rfmm",), "加词 燃面rfmm"),
            (("燃面", "晕碳"), ("rfmm", "ywtfo"), "加词 燃面rfmm 晕碳ywtfo"),
        ):
            with self.subTest(operands=operands):
                self.store.delete(self.author)
                self.writes.clear()
                self.set_model(operands, codes)
                handled, ctx = await self.stage(message)
                self.assertTrue(handled, ctx.response)
                self.assertEqual(self.writes, [], ctx.response)
                record = self.store.get_record(self.author)
                self.assertIsNotNone(record, ctx.response)
                self.assertEqual(record.owner_key, self.author)
                self.assertEqual(record.state.confirmation_source, "server_warning")
                self.assertIn("确认", ctx.response)
                self.assertIsNone(await self.answer("确认", self.other))
                self.assertIs(self.store.get_record(self.author), record)
                self.assertIsNotNone(await self.answer("确认"))
                self.assertEqual(len(self.writes), 1)
                self.assertEqual(self.added_pairs(), list(zip(operands, codes)))
                self.assertIsNone(await self.answer("确认"))

    async def test_blocked_review_prevents_any_add(self):
        self.review_changes = {"reviewDisposition": "BLOCK", "message": "Offline review blocked"}
        self.set_model()
        handled, ctx = await self.stage("加词 燃面rfmm")
        self.assertTrue(handled, ctx.response)
        self.assertEqual(self.writes, [])
        self.assertEqual([name for name, _ in self.calls], ["keytao_prepare_reviewed_add"])

    async def test_batch_second_review_failure_prevents_partial_add(self):
        for failure in ("block", "malformed_json"):
            with self.subTest(failure=failure):
                self.calls.clear()

                async def review_failure(name, args, *identity, **kwargs):
                    response = await self.call(name, args, *identity, **kwargs)
                    if name == "keytao_prepare_reviewed_add" and args["word"] == "晕碳":
                        if failure == "malformed_json":
                            return "not JSON"
                        payload = json.loads(response)
                        payload["reviewDisposition"] = "BLOCK"
                        return json.dumps(payload)
                    return response

                self.tool.side_effect = review_failure
                self.set_model(("燃面", "晕碳"), ("rfmm", "ywtfo"))
                handled, ctx = await self.stage("加词 燃面rfmm 晕碳ywtfo")
                self.assertTrue(handled, ctx.response)
                self.assertEqual(self.writes, [])
                self.assertEqual([name for name, _ in self.calls],
                                 ["keytao_prepare_reviewed_add", "keytao_prepare_reviewed_add"])

    async def test_multi_pair_submit_suffix_with_extra_word_tokens_falls_back(self):
        self.set_model(("燃面", "晕碳"), ("rfmm", "ywtfo"), submit_after=True)
        await self.assert_fallback("加词 燃面rfmm 晕碳ywtfo，加入并提交")

    async def test_single_pair_submit_suffix_with_extra_word_tokens_falls_back(self):
        self.set_model(submit_after=True)
        await self.assert_fallback("加词 燃面rfmm并提交")


if __name__ == "__main__":
    unittest.main()
