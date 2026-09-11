"""S62 offline incident replay through the real selection parser and executor."""

import copy
import json
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from test_s54_selection import fixtures, reviewed_record
from keytao_bot.harness import authorization_grammar as grammar
from keytao_bot.plugins import chat_commands as commands, chat_routing as routing
from keytao_bot.utils.pending_confirmation import advertised_command_suggestions, render_server_backed_batch_candidates
from test_s54_renderer import reviewed_scope
from test_s64_explicit_submit import REAL_TOOL_DISPATCH


def incident_record():
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


class S62IncidentTests(unittest.IsolatedAsyncioTestCase):
    async def replay(self, messages, *, discovery=False, delivery=False, executor=False):
        store = fixtures.MemoryConversationStateStore()
        key = fixtures.ConversationAddress.group("qq", "865189947", "s62-offline")
        if not discovery:
            store.set(key, incident_record())
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
                    from test_s62_reading_resolution import ambiguous_review
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
                commands._capture_successful_draft_write_delivery(name, result, platform, actor)
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
                commands._capture_successful_draft_write_delivery(name, result, platform, actor)
                return json.dumps(result)
            raise AssertionError("Unexpected tool: " + name)

        async def transport(name, args, context):
            self.assertEqual((context.platform, context.user_id), ("qq", key.actor_id))
            if name == "keytao_batch_add_to_draft":
                self.assertEqual([(row["word"], row["code"]) for row in args["items"]], [("敲不死", "qbso")])
                self.assertEqual(context.trusted_reviewed_items_by_key[("敲不死", "qbso")]["pinyin"], "qiāo bù sǐ")
            return json.loads(await tool(name, args, context.platform, context.user_id))

        with patch.object(fixtures.openai_chat_module, "conversation_state_store", store), patch.object(
            fixtures.openai_chat_module, "call_tool_function", side_effect=tool,
        ), patch.object(fixtures.openai_chat_module, "_classify_message_command_intent",
                        AsyncMock(side_effect=AssertionError("S62 model calls forbidden"))), ExitStack() as stack:
            if executor:
                stack.enter_context(patch.object(fixtures.openai_chat_module, "call_tool_function", REAL_TOOL_DISPATCH))
                stack.enter_context(patch.object(commands.tool_executor, "_invoke_effective_tool", side_effect=transport))
                stack.enter_context(patch.object(commands.memory_store, "record_tool_receipt"))
            if discovery:
                footer = await commands._try_handle_simple_single_word_query(
                    "鸡白汤，敲不死，乔布斯", "qq", key.actor_id, key,
                )
                self.assertIn("鸡白汤", footer)
                self.assertIn("乔布斯", footer)
                self.assertIn("敲不死 2", footer)
                self.assertTrue(fixtures.openai_chat_module._advertised_reply_matches_live_record(footer, store.get_record(key)))
                responses.append(footer)
            for message in messages:
                chat = fixtures.openai_chat_module
                memory = fixtures.ChatMemoryContext(platform="qq", user_id=key.actor_id, space_type="group", space_id=key.space_id)
                message_token = chat._current_turn_message.set(message)
                delivery_token = commands.current_draft_delivery_claims.set([])
                try:
                    ctx = chat.TurnContext(
                        bot=object(), event=SimpleNamespace(message_id=None), platform="qq", user_id=key.actor_id,
                        conv_key=key, space_key=key.space_key, memory_context=memory, normalized_message_text=message,
                        generic_command_intent=fixtures.MessageCommandIntent(intent="draft_view" if message == "查看草稿" else "none"),
                    )
                    if message == "查看草稿":
                        await chat._stage_handle_draft_management(ctx)
                    else:
                        ctx.response = await commands.handle_pending_message_core(
                            message, "qq", "s62-offline", key, allow_intent_model=False,
                        )
                    if delivery and ctx.response is not None:
                        for stage in (chat._stage_normalize_response, chat._stage_scope_language_only_response,
                                      chat._stage_append_ticket_challenge, chat._stage_enforce_advertised_reply_contract):
                            await stage(ctx)
                        ctx.response = chat._prepare_user_facing_reply(ctx.response, memory)
                    responses.append(ctx.response)
                finally:
                    commands.current_draft_delivery_claims.reset(delivery_token)
                    chat._current_turn_message.reset(message_token)
        return writes, submitted, responses, calls, store.get(key)

    async def test_turn_two_to_seven_fixture_branches_reach_final_delivery(self):
        for selection in (
            "鸡白汤 jbtaua, 敲不死 qbso，加入草稿并提交",
            "鸡白汤 jbtaua, 敲不死 qbso; 加入草稿并提交.",
            "鸡白汤 jī bái tāng：jbtaua，敲不死 qbso; 加入并提交",
            "敲不死 2 加入并提交",
        ):
            with self.subTest(selection=selection):
                writes, submitted, replies, calls, _ = await self.replay(
                    [selection, "查看草稿"], discovery=True, delivery=True,
                )
                self.assertEqual([(row["word"], row["code"]) for row in writes], [("敲不死", "qbso")], replies)
                self.assertTrue(submitted, replies)
                self.assertIn("已提交审核", replies[1])
                if "鸡白汤" in selection:
                    self.assertIn("鸡白汤", replies[1])
                    self.assertIn("读音", replies[1])
                    self.assertIn("未加入", replies[1])
                self.assertIn("当前草稿", replies[2])
                self.assertNotIn("候选", replies[2])
                self.assertNotIn("鸡白汤", replies[2])
                self.assertEqual(sum(name == "keytao_batch_add_to_draft" for name, _ in calls), 2)
                self.assertEqual(sum(name == "keytao_submit_batch" for name, _ in calls), 2)

    async def test_real_dispatcher_and_executor_bind_the_mixed_selection(self):
        writes, submitted, replies, _, _ = await self.replay(
            ["鸡白汤 jbtaua, 敲不死 qbso，加入草稿并提交"],
            discovery=True, delivery=True, executor=True,
        )
        self.assertEqual([(row["word"], row["code"]) for row in writes], [("敲不死", "qbso")], replies)
        self.assertTrue(submitted, replies)
        self.assertIn("鸡白汤", replies[-1])
        self.assertIn("未加入", replies[-1])

    async def test_combined_action_forms_and_separators(self):
        for action in ("加入", "加入草稿", "加到草稿", "写入草稿", "加入并提交", "加入草稿并提交", "加到草稿并提交"):
            for separator in (" ", "，", ",", "、", "；", ";"):
                for message in (f"敲不死 2{separator}{action}。", f"{action}{separator}敲不死 qbso."):
                    with self.subTest(message=message):
                        parsed = grammar.parse_reviewed_selection_command(message)
                        self.assertIsNotNone(parsed)
                        self.assertTrue(routing.message_authorizes_live_pending_mutation(message, incident_record()))
                        writes, submitted, replies, _, _ = await self.replay([message])
                        self.assertEqual([(r["word"], r["code"]) for r in writes], [("敲不死", "qbso")], replies)
                        self.assertEqual(submitted, "提交" in action, replies)

    async def test_mixed_message_keeps_bindable_selection(self):
        for message in ("鸡白汤 jbtaua, 敲不死 qbso，加入草稿并提交",
                        "鸡白汤 jbtaua, 敲不死 qbso; 加入草稿并提交."):
            writes, submitted, replies, _, _ = await self.replay([message])
            self.assertEqual([(r["word"], r["code"]) for r in writes], [("敲不死", "qbso")], replies)
            self.assertTrue(submitted, replies)
            self.assertIn("鸡白汤", replies[0])
            self.assertIn("读音", replies[0])
            self.assertIn("未加入", replies[0])

    async def test_bare_selection_writes_once_with_receipt_and_never_submits(self):
        for message in ("敲不死 2", "鸡白汤 jbtaua, 敲不死 qbso"):
            with self.subTest(message=message):
                writes, submitted, replies, calls, state = await self.replay(
                    [message], discovery=True, delivery=True, executor=True,
                )
                self.assertEqual([(r["word"], r["code"]) for r in writes], [("敲不死", "qbso")], replies)
                self.assertFalse(submitted)
                self.assertIsNone(state)
                self.assertEqual(sum(name == "keytao_batch_add_to_draft" for name, _ in calls), 2)
                self.assertFalse(any(name == "keytao_submit_batch" for name, _ in calls))
                self.assertTrue(writes[0]["needsManualReview"])
                self.assertIn("敲不死 → qbso", replies[-1])
                self.assertIn("https://keytao.rea.ink/batch/s62-fixture", replies[-1])
                self.assertNotIn("已提交", replies[-1])
                self.assertNotIn("尚未加入", replies[-1])
                if "鸡白汤" in message:
                    self.assertIn("「鸡白汤」还没确定读音，未加入", replies[-1])

    async def test_nonrecommended_selection_writes_and_mixed_errors_do_not_poison_valid_choice(self):
        writes, submitted, replies, _, _ = await self.replay(["敲不死 3"], delivery=True)
        self.assertEqual([(r["word"], r["code"]) for r in writes], [("敲不死", "qbsoa")], replies)
        self.assertFalse(submitted)
        for invalid, reason in (("9", "超出"), ("evil", "不在当前候选")):
            state = reviewed_record()
            selected, intent, error = routing._resolve_multi_word_pending_candidate_selection(
                state, f"小端 {invalid}，大端 3，加入并提交",
            )
            self.assertIsNone(error)
            self.assertEqual([(row["word"], row["code"]) for row in selected.args["items"]], [("大端", "dsdtvo")])
            self.assertEqual(intent.intent, "pending_add_and_submit")
            self.assertIn(reason, selected.args["_selection_skipped"][0])

    async def test_complete_envelope_and_review_seals_still_required(self):
        for message in (
            "他说鸡白汤 jbtaua, 敲不死 qbso，加入并提交",
            "不要鸡白汤 jbtaua, 敲不死 qbso，加入并提交",
            "鸡白汤 jbtaua, 敲不死 qbso，加入并提交？",
            "敲不死 2，加入并提交，然后删除", "敲不死 2，敲不死 3，加入",
            "他说「敲不死 2，加入并提交」", "「『敲不死 2，加入并提交』」",
        ):
            with self.subTest(message=message):
                writes, _, _, calls, _ = await self.replay([message])
                self.assertEqual(writes, [])
                self.assertEqual(calls, [])
        for mutation in ("missing", "candidate", "warning"):
            record = incident_record()
            if mutation == "missing":
                record.args["_candidate_scopes"][0].pop("reviewedState")
            elif mutation == "candidate":
                record.args["_candidate_scopes"][0]["candidates"][1][0] = "evil"
            else:
                record.confirmation_source = "server_warning"
            self.assertFalse(routing.message_authorizes_live_pending_mutation("敲不死 2，加入并提交", record))

    def test_footer_and_added_commands_close_over_parser_and_binding(self):
        record = incident_record()
        footer = render_server_backed_batch_candidates(record.args["items"], record.args["_candidate_scopes"])
        controls = advertised_command_suggestions(footer)
        self.assertEqual(controls, ("加入", "加入并提交", "敲不死 2", "敲不死 2，加入并提交"))
        for command in controls:
            with self.subTest(command=command):
                if command == "取消":
                    self.assertEqual(routing._pending_tool_assent_intent(record, command).intent, "pending_cancel")
                elif grammar.parse_reviewed_multi_word_selection(command):
                    selected, intent, error = routing._resolve_multi_word_pending_candidate_selection(record, command)
                    self.assertIsNotNone(selected, error)
                    self.assertTrue(routing.message_authorizes_live_pending_mutation(command, record))
                else:
                    self.assertTrue(routing.message_authorizes_live_pending_mutation(command, record))


if __name__ == "__main__":
    unittest.main()
