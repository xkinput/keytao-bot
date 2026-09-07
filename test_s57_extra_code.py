"""Existing entries can gain an explicitly requested, reading-bound code."""

import asyncio
import json
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from keytao_bot.utils.explicit_code import parse_explicit_code_request
from keytao_bot.harness.state import SQLiteConversationStateStore, trusted_word_record_is_complete

chat = harness.openai_chat_module
commands = harness.chat_commands_module
routing = chat._chat_routing


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


class ExtraCodeTests(unittest.TestCase):
    def test_additional_code_warning_rejects_foreign_targets_mixed_risks_and_incomplete_cas(self):
        async def run():
            cases = (
                ("foreign-word", lambda p: p["warnings"][0]["item"].update(word="咽")),
                ("foreign-code", lambda p: p["warnings"][0]["item"].update(code="yeoiu")),
                ("foreign-type", lambda p: p["warnings"][0]["item"].update(type="Phrase")),
                ("foreign-existing-word", lambda p: p["warnings"][0]["existing"].update(word="咽")),
                ("foreign-existing-code", lambda p: p["warnings"][0]["existing"].update(code="yeoiu")),
                ("duplicate-identity", lambda p: p["warnings"][0]["existing"].update(code="yeoia")),
                ("foreign-existing-type", lambda p: p["warnings"][0]["existing"].update(type="Phrase")),
                ("changed-remark", lambda p: p["warnings"][0]["item"].update(remark="")),
                ("lost-seal", lambda p: p["warnings"][0]["item"].update(needsManualReview=False)),
                ("mixed-risk", lambda p: p["warnings"].append({**p["warnings"][0], "warningType": "duplicate_code"})),
                ("missing-digest", lambda p: p.pop("warningDigest")),
                ("boolean-version", lambda p: p.update(contentVersion=True)),
                ("stale-version", lambda p: p.update(contentVersionConflict=True)),
            )
            for label, mutate in cases:
                with self.subTest(label=label):
                    key = harness.ConversationAddress.private("qq", "s57-warning-" + label)
                    chat.conversation_state_store.delete(key)
                    warning = additional_code_warning()
                    mutate(warning)
                    creates = []

                    async def tool(name, args, *_args, **kwargs):
                        if name == "keytao_prepare_reviewed_add":
                            return json.dumps(existing_single_review())
                        if name == "keytao_lookup_by_code":
                            return json.dumps({"success": True, "phrases": []})
                        if name == "keytao_create_phrase":
                            creates.append(dict(args))
                            return json.dumps(warning)
                        raise AssertionError(name)

                    with patch.object(chat, "call_tool_function", side_effect=tool):
                        await commands.try_handle_explicit_entry_code_command(
                            '添加单字"嘢"，编码为"yeoia"', "qq", key.actor_id, key,
                        )
                    self.assertEqual(len(creates), 1)
                    self.assertIsNot(creates[0].get("confirmed"), True)
        asyncio.run(run())

    def test_additional_code_capability_is_cleared_when_confirmed_execution_raises(self):
        async def run():
            key = harness.ConversationAddress.private("qq", "s57-warning-error")
            chat.conversation_state_store.delete(key)

            async def tool(name, args, *_args, **kwargs):
                if name == "keytao_prepare_reviewed_add":
                    return json.dumps(existing_single_review())
                if name == "keytao_lookup_by_code":
                    return json.dumps({"success": True, "phrases": []})
                if name == "keytao_create_phrase":
                    if args.get("confirmed"):
                        raise RuntimeError("fixture interrupted confirmed dispatch")
                    return json.dumps(additional_code_warning())
                raise AssertionError(name)

            with patch.object(chat, "call_tool_function", side_effect=tool):
                with self.assertRaisesRegex(RuntimeError, "fixture interrupted"):
                    await commands.try_handle_explicit_entry_code_command(
                        '添加单字"嘢"，编码为"yeoia"', "qq", key.actor_id, key,
                    )
            self.assertIsNone(commands._current_explicit_additional_code.get())
            self.assertFalse(commands._create_preview_can_auto_confirm(
                additional_code_warning(), {"word": "嘢", "code": "yeoia"},
            ))
        asyncio.run(run())

    def test_real_create_executor_replays_exact_multiple_code_warning_only_for_explicit_request(self):
        async def run():
            for explicit_request, expected_writes in ((True, 1), (False, 0)):
                key = harness.ConversationAddress.private("qq", "s57-multiple-warning")
                chat.conversation_state_store.delete(key)
                creates = []

                async def tool(name, args, *_args, **kwargs):
                    if name == "keytao_prepare_reviewed_add":
                        return json.dumps(existing_single_review())
                    if name == "keytao_lookup_by_code":
                        return json.dumps({"success": True, "phrases": []})
                    if name == "keytao_create_phrase":
                        creates.append((dict(args), kwargs))
                        if args.get("confirmed") is True:
                            return json.dumps({"success": True, "batchId": "s57-existing-batch", "contentVersion": 2})
                        return json.dumps(additional_code_warning())
                    raise AssertionError(name)

                with patch.object(chat, "call_tool_function", side_effect=tool), patch.object(
                    commands, "_format_draft_response", AsyncMock(return_value="Single 嘢 yeoia"),
                ):
                    if explicit_request:
                        reply = await commands.try_handle_explicit_entry_code_command(
                            '添加单字"嘢"，编码为"yeoia"', "qq", key.actor_id, key,
                        )
                    else:
                        reply = await commands._execute_add_to_draft(
                            "嘢", "yeoia", "qq", key.actor_id,
                            reviewed_pinyin="yě", reviewed_candidate_codes=("ye", "yeoia"),
                        )
                actual_writes = [call for call in creates if call[0].get("confirmed") is True]
                self.assertEqual(len(actual_writes), expected_writes, reply)
                self.assertTrue(creates[0][0].get("preview_only"))
                if explicit_request:
                    self.assertEqual(len(creates), 2)
                    arguments, kwargs = actual_writes[0]
                    self.assertEqual((arguments["word"], arguments["code"]), ("嘢", "yeoia"))
                    self.assertEqual(arguments["batch_id"], "s57-existing-batch")
                    self.assertEqual(arguments["expected_content_version"], 1)
                    self.assertEqual(arguments["expected_warning_digest"], additional_code_warning()["warningDigest"])
                    self.assertEqual(kwargs["trusted_reviewed_items_by_key"][("嘢", "yeoia")]["type"], "Single")
                    self.assertIn("形码 oia 未能核验", arguments["remark"])
                    self.assertTrue(arguments["needs_manual_review"])
                    self.assertIn("已", reply)
                else:
                    self.assertIn("确认", reply)
        asyncio.run(run())

    def test_quoted_and_natural_forms_bind_to_the_same_word(self):
        for message in (
            '添加单字"嘢"，编码为"yeoia"', '添加 单字 “嘢” ，编码为 “yeoia”',
            '添加单字「嘢」，编码为「yeoia」', '添加单字＂嘢＂，编码为＂yeoia＂',
            '给 嘢 加一个码 yeoia', '嘢 也放到 yeoia', '加入编码yeoia',
        ):
            with self.subTest(message=message):
                request = parse_explicit_code_request(message, "嘢")
                self.assertIsNotNone(request, message)
                self.assertEqual(request.code, "yeoia")
                state = harness.PendingAddWord(
                    word="嘢", recommended_code="ye", candidates=[("ye", False)],
                    server_candidates=[("ye", False)], pronunciation_codes={"ye": "yě"},
                    phrase_type="Single",
                )
                self.assertTrue(routing.message_authorizes_live_pending_mutation(message, state), message)
                self.assertIsNone(routing._pending_assent_rejection_response(state, message))

    def test_explicit_submit_suffix_is_not_part_of_the_code(self):
        for message in ('添加单字"嘢"，编码为"yeoia"并提交', "给 嘢 加一个码 yeoia 并提交"):
            request = parse_explicit_code_request(message, "嘢")
            self.assertEqual(request.code, "yeoia")
            self.assertTrue(request.submit_after)

    def test_wrong_word_and_non_commands_never_gain_explicit_authority(self):
        for message in (
            '添加单字"咽"，编码为"yeoia"', '他说添加单字"嘢"，编码为"yeoia"',
            '不要给 嘢 加一个码 yeoia', '嘢 也放到 yeoia？',
            '给 嘢 加一个码 yeoia 删除也', '添加单字"嘢"，编码为"yeoia”，提交',
        ):
            self.assertIsNone(parse_explicit_code_request(message, "嘢"), message)

    def test_cold_existing_single_add_uses_new_code_occupancy_and_manual_seal(self):
        async def run():
            key = harness.ConversationAddress.private("qq", "s57-extra-cold")
            chat.conversation_state_store.delete(key)
            calls = []

            async def tool(name, args, *_args, **_kwargs):
                calls.append((name, args))
                if name == "keytao_prepare_reviewed_add":
                    return json.dumps(existing_single_review())
                if name == "keytao_lookup_by_code":
                    return json.dumps({"success": True, "phrases": []})
                raise AssertionError(name)

            execute = AsyncMock(return_value="已加入草稿。")
            with patch.object(chat, "call_tool_function", side_effect=tool), patch.object(
                commands, "_execute_add_to_draft", execute,
            ):
                reply = await commands.try_handle_explicit_entry_code_command(
                    '添加单字"嘢"，编码为"yeoia"', "qq", "s57-extra-cold", key,
                )
            self.assertIn("已加入草稿", reply)
            self.assertIn("形码 oia 未能核验，需管理员复核", reply)
            self.assertNotIn("也", reply)
            self.assertNotIn("耶", reply)
            self.assertEqual(execute.await_count, 1)
            self.assertEqual(execute.call_args.args[:2], ("嘢", "yeoia"))
            self.assertIs(execute.call_args.args[7], True)
            self.assertEqual(execute.call_args.kwargs["reviewed_pinyin"], "yě")
            self.assertIn("yeoia", execute.call_args.kwargs["reviewed_candidate_codes"])
            self.assertEqual([args["code"] for name, args in calls if name == "keytao_lookup_by_code"], ["yeoia"])
        asyncio.run(run())

    def test_review_identity_and_wrong_prefix_refuse_before_occupancy(self):
        async def run():
            for changes, code, expected in (
                ({"word": "咽"}, "yeoia", "已审读音"),
                ({"type": "Phrase"}, "yeoia", "正确词条类型"),
                ({}, "aboia", "音码前缀"),
                ({}, "yeoiaaa", "最多 6 位"),
                ({}, "yeoi1", "小写字母"),
            ):
                key = harness.ConversationAddress.private("qq", "s57-extra-invalid")
                chat.conversation_state_store.delete(key)
                tool = AsyncMock(return_value=json.dumps({**existing_single_review(), **changes}))
                execute = AsyncMock(side_effect=AssertionError("unexpected write"))
                with patch.object(chat, "call_tool_function", tool), patch.object(commands, "_execute_add_to_draft", execute):
                    reply = await commands.try_handle_explicit_entry_code_command(
                        f'添加单字"嘢"，编码为"{code}"', "qq", key.actor_id, key,
                    )
                self.assertIn(expected, reply)
                self.assertEqual(tool.await_count, 1)
                self.assertEqual(execute.await_count, 0)
        asyncio.run(run())

    def test_multi_code_existing_context_survives_reload_without_row_action_authority(self):
        async def run(db_path):
            store = SQLiteConversationStateStore(db_path=db_path)
            key = harness.ConversationAddress.private("qq", "s57-extra-multiple")

            async def tool(name, args, *_args, **_kwargs):
                if name == "keytao_lookup_by_word":
                    return json.dumps({"success": True, "phrases": [
                        {"word": "嘢", "code": code, "type": "Single", "weight": 11}
                        for code in ("yeoiav", "yeoi")
                    ]})
                if name == "keytao_pending_items_by_words":
                    return json.dumps({"success": True, "complete": True, "items": []})
                raise AssertionError(name)

            with patch.object(chat, "conversation_state_store", store), patch.object(
                chat, "call_tool_function", side_effect=tool,
            ), patch.object(chat, "_classify_simple_word_query_intent", AsyncMock(
                return_value=harness.SimpleWordQueryIntent(True, ("嘢",), "word_lookup", 1.0),
            )):
                reply = await commands._try_handle_simple_single_word_query("嘢", "qq", key.actor_id, key)
            reloaded = SQLiteConversationStateStore(db_path=db_path)
            record = reloaded.get_record(key)
            self.assertTrue(record.state.context_only)
            self.assertTrue(trusted_word_record_is_complete(record.state))
            self.assertFalse(trusted_word_record_is_complete(replace(record.state, context_only=False)))
            self.assertFalse(trusted_word_record_is_complete(replace(record.state, context_only=1)))
            self.assertIn("追加一个编码", reply)
            self.assertTrue(routing.message_authorizes_live_pending_mutation("加入编码yeoia", record.state))
            for action in ("删除", "换码", "确认", "硬来"):
                self.assertFalse(routing.message_authorizes_live_pending_mutation(action, record.state), action)
            self.assertTrue(chat._advertised_reply_matches_live_record(reply, record))
        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(run(directory + "/state.db"))

    def test_occupied_new_code_preserves_guard_and_force_binds_only_selected_slot(self):
        async def run():
            key = harness.ConversationAddress.private("qq", "s57-extra-protected")
            chat.conversation_state_store.delete(key)

            async def tool(name, args, *_args, **_kwargs):
                if name == "keytao_prepare_reviewed_add":
                    return json.dumps(existing_single_review())
                if name == "keytao_lookup_by_code":
                    self.assertEqual(args["code"], "yeoia")
                    return json.dumps({"success": True, "phrases": [{
                        "word": "咽", "code": "yeoia", "type": "Single", "weight": 10,
                    }]})
                raise AssertionError(name)

            shift = AsyncMock(return_value="shifted")
            with patch.object(chat, "call_tool_function", side_effect=tool), patch.object(
                commands.keytao_review, "assess_candidate_chain_commonness", AsyncMock(return_value=[]),
            ), patch.object(commands, "_execute_shift_to_code", shift):
                reply = await commands.try_handle_explicit_entry_code_command(
                    "给 嘢 加一个码 yeoia", "qq", key.actor_id, key,
                )
                self.assertIn("本次未写入", reply)
                self.assertEqual(shift.await_count, 0)
                record = chat.conversation_state_store.get_record(key)
                self.assertEqual(record.state.recommended_code, "yeoia")
                self.assertTrue(routing.message_authorizes_live_pending_mutation("硬加", record.state))
                reply = await commands.handle_pending_message_core(
                    "硬加", "qq", key.actor_id, key, allow_intent_model=False,
                )
            self.assertEqual(reply, "shifted")
            self.assertEqual(shift.await_count, 1)
            self.assertEqual(shift.call_args.args[:2], ("嘢", "yeoia"))
            self.assertTrue(shift.call_args.kwargs["auto_confirm_shift_plan"])
            self.assertEqual(shift.call_args.kwargs["target_item"]["type"], "Single")
        asyncio.run(run())

    def test_contextual_existing_record_rechecks_reading_before_add(self):
        async def run():
            key = harness.ConversationAddress.private("qq", "s57-extra-context")
            state = commands.create_pending_trusted_word_record("嘢", "yeoiav", "Single")
            chat.conversation_state_store.set(key, state)
            execute = AsyncMock(return_value="已加入草稿。")

            async def tool(name, args, *_args, **_kwargs):
                if name == "keytao_prepare_reviewed_add":
                    self.assertEqual(args, {"word": "嘢"})
                    return json.dumps(existing_single_review())
                if name == "keytao_lookup_by_code":
                    self.assertEqual(args, {"code": "yeoia"})
                    return json.dumps({"success": True, "phrases": []})
                raise AssertionError(name)

            with patch.object(chat, "call_tool_function", side_effect=tool), patch.object(
                commands, "_execute_add_to_draft", execute,
            ):
                reply = await commands.handle_pending_message_core(
                    "加入编码yeoia", "qq", "s57-extra-context", key, allow_intent_model=False,
                )
            self.assertIn("已加入草稿", reply)
            self.assertEqual(execute.await_count, 1)
        asyncio.run(run())

    def test_existing_phrase_keeps_its_table_and_reviewed_phonetic_prefix(self):
        async def run():
            key = harness.ConversationAddress.private("qq", "s57-extra-phrase")
            chat.conversation_state_store.delete(key)
            review = {
                "success": True, "word": "财宝", "type": "Phrase", "recommendedCode": "cdbz",
                "needsManualReview": False,
                "existing": [{"word": "财宝", "code": "cdbz", "type": "Phrase", "weight": 100}],
                "pronunciations": [{
                    "pinyin": "cái bǎo", "recommendedCode": "cdbz", "codes": ["cdbz"],
                    "candidateStatuses": [{"code": "cdbz", "occupied": True, "words": ["财宝"]}],
                }],
            }

            async def tool(name, args, *_args, **_kwargs):
                if name == "keytao_prepare_reviewed_add":
                    return json.dumps(review)
                if name == "keytao_lookup_by_code":
                    self.assertEqual(args["code"], "cdbzio")
                    return json.dumps({"success": True, "phrases": []})
                raise AssertionError(name)

            async def execute(word, code, *_args, **kwargs):
                state = chat.conversation_state_store.get(key)
                self.assertEqual(state.phrase_type, "Phrase")
                self.assertEqual((word, code), ("财宝", "cdbzio"))
                self.assertEqual(kwargs["reviewed_pinyin"], "cái bǎo")
                self.assertTrue(state.needs_manual_review)
                return "已加入草稿。"

            with patch.object(chat, "call_tool_function", side_effect=tool), patch.object(
                commands, "_execute_add_to_draft", side_effect=execute,
            ):
                reply = await commands.try_handle_explicit_entry_code_command(
                    '添加词组「财宝」，编码为「cdbzio」', "qq", key.actor_id, key,
                )
            self.assertIn("形码 io 未能核验，需管理员复核", reply)
            self.assertIn("已加入草稿", reply)
        asyncio.run(run())

    def test_exact_identity_is_rejected_while_other_table_is_ignored(self):
        async def run():
            for occupant_type, writes in (("Single", 0), ("Phrase", 1)):
                key = harness.ConversationAddress.private("qq", "s57-extra-identity-" + occupant_type)
                chat.conversation_state_store.delete(key)
                execute = AsyncMock(return_value="已加入草稿。")

                async def tool(name, args, *_args, **_kwargs):
                    if name == "keytao_prepare_reviewed_add":
                        return json.dumps(existing_single_review())
                    if name == "keytao_lookup_by_code":
                        return json.dumps({"success": True, "phrases": [{
                            "word": "嘢", "code": "yeoia", "type": occupant_type, "weight": 11,
                        }]})
                    raise AssertionError(name)

                with patch.object(chat, "call_tool_function", side_effect=tool), patch.object(
                    commands, "_execute_add_to_draft", execute,
                ):
                    reply = await commands.try_handle_explicit_entry_code_command(
                        "给 嘢 加一个码 yeoia", "qq", key.actor_id, key,
                    )
                self.assertEqual(execute.await_count, writes)
                self.assertIn("不能重复添加" if not writes else "已加入草稿", reply)
        asyncio.run(run())

    def test_completed_receipt_allows_new_explicit_code_but_unfinished_ticket_does_not(self):
        async def run():
            for recent, writes in ((True, 1), (False, 0)):
                key = harness.ConversationAddress.private("qq", "s57-extra-receipt")
                chat.conversation_state_store.set(key, harness.PendingToolConfirm(
                    function_name="keytao_submit_batch",
                    args={"batch_id": "s57-batch", "_recent_own_write": recent},
                ))

                async def tool(name, args, *_args, **_kwargs):
                    if name == "keytao_prepare_reviewed_add":
                        return json.dumps(existing_single_review())
                    if name == "keytao_lookup_by_code":
                        return json.dumps({"success": True, "phrases": []})
                    raise AssertionError(name)

                execute = AsyncMock(return_value="已加入草稿。")
                with patch.object(chat, "call_tool_function", side_effect=tool), patch.object(
                    commands, "_execute_add_to_draft", execute,
                ):
                    reply = await commands.try_handle_explicit_entry_code_command(
                        '添加单字"嘢"，编码为"yeoia"', "qq", key.actor_id, key,
                    )
                self.assertIsNotNone(reply)
                self.assertEqual(execute.await_count, writes)
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
