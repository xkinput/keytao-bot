"""Offline S71 routing through the production STAGES and delivery contract."""

import copy
import json
import socket
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from keytao_bot.utils.observability import current_turn_metrics


chat = harness.openai_chat_module
commands = harness.chat_commands_module
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


class StaleTicketReleaseTests(unittest.IsolatedAsyncioTestCase):
    async def replay(self, message, quote="", *, live=True, model_failure=False):
        store = harness.MemoryConversationStateStore()
        memory = harness.ChatMemoryContext(
            platform="qq", user_id="s71-owner", space_type="group", space_id="s71-group",
        )
        key = memory.conversation_address
        stages, delivered, calls = [], [], []
        classifier = AsyncMock(return_value=harness.MessageCommandIntent())
        word_classifier = AsyncMock(return_value=harness.SimpleWordQueryIntent(False))
        model = AsyncMock(
            side_effect=chat.get_ai_response_core if model_failure else None,
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
        fallback_pending = AsyncMock(wraps=chat.handle_pending_message_core)
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
            delivered.append(chat._enforce_advertised_reply_contract(text, key))

        token = harness.begin_turn_metrics("qq", "group")
        try:
            with ExitStack() as stack:
                for patcher in (
                    patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")),
                    patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Network forbidden")),
                    patch.object(socket, "create_connection", side_effect=AssertionError("Network forbidden")),
                    patch.object(socket, "getaddrinfo", side_effect=AssertionError("DNS forbidden")),
                    patch.object(chat, "AsyncOpenAI", side_effect=AssertionError("Real model forbidden")),
                    patch.object(chat, "conversation_state_store", store),
                    patch.object(chat, "draft_operation_coordinator", harness.DraftOperationCoordinator()),
                    patch.object(chat, "extract_memory_context", AsyncMock(return_value=memory)),
                    patch.object(chat, "get_history", return_value=[]),
                    patch.object(chat, "remember_conversation"),
                    patch.object(chat, "schedule_memory_compaction"),
                    patch.object(chat, "_classify_message_command_intent", classifier),
                    patch.object(chat, "_classify_simple_word_query_intent", word_classifier),
                    patch.object(chat, "get_ai_response_core", model),
                    patch.object(chat, "call_tool_function", side_effect=tool),
                    patch.object(commands, "_execute_confirmed_tool", execute),
                    patch.object(commands, "_perform_batch_add_to_draft_and_submit", submit),
                    patch.object(chat, "_schedule_background_draft_operation", schedule),
                    patch.object(chat, "_finish_ai_chat_response", side_effect=finish),
                    patch.object(chat, "_finish_ai_chat_matcher", side_effect=finish),
                ):
                    stack.enter_context(patcher)
                initial = chat.TurnContext(
                    bot=object(), event=object(), platform="qq", user_id=key.actor_id,
                    message_text="你都没有上手 有点不切实际",
                    reply_reference=harness.ReplyReferenceInfo(),
                )
                for stage in chat.STAGES[chat.STAGES.index(chat._stage_normalize_message_text):]:
                    if await stage(initial):
                        break
                self.assertIsInstance(initial.response, chat.ServerBackedQueryReply)
                self.assertEqual(len(delivered), 1)
                record = store.get_record(key)
                self.assertIsInstance(record.state, harness.PendingToolConfirm)
                self.assertTrue(record.state.args["_reviewed_multi_word"])
                self.assertFalse(record.origin_prompt_digest)
                old_render = delivered[0]
                self.assertTrue(chat._advertised_reply_matches_live_record(old_render, record))
                self.assertEqual(old_render, chat._render_live_batch_record(record))
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
                ctx = chat.TurnContext(
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
                        patch.object(chat, "AsyncOpenAI", return_value=client),
                        patch.object(chat, "OPENAI_API_KEY", "fixture-key"),
                        patch.object(chat.memory_store, "get_context_block", return_value=""),
                        patch.object(chat, "get_group_history_context", return_value=""),
                        patch.object(chat, "skills_manager", SimpleNamespace(
                            get_skill_instructions=lambda: "", has_tools=lambda: False,
                            get_tools=lambda: [], get_tool_function=lambda _name: None,
                        )),
                        patch.object(chat, "handle_pending_message_core", fallback_pending),
                    ):
                        stack.enter_context(patcher)
                for stage in chat.STAGES[chat.STAGES.index(chat._stage_normalize_message_text):]:
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

    async def assert_fresh_request(self, message, quote="", *, reviewed_word=""):
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

    async def test_incident_unrelated_bot_quote(self):
        await self.assert_fresh_request("把这个加进词库", QUOTE)

    async def test_deictic_ticket_commands_keep_head_behavior(self):
        # Recorded on HEAD 573a37e: all ten turns reject and deliver the live render.
        for message in (
            "加进草稿", "把这两个加进词库", "加入这两个词", "加上吧", "都加进词库吧",
        ):
            for quote in ("", "ticket"):
                with self.subTest(message=message, quote=quote):
                    result = await self.replay(message, quote)
                    self.assertEqual(
                        (result["flow"], result["stages"][-1], result["delivered"]),
                        ("pending-confirmation", "_stage_finish_scoped_pending_response",
                         [result["old_render"]]),
                    )
                    self.assertEqual(result["record"], result["before"])
                    result["execute"].assert_not_awaited()
                    result["submit"].assert_not_awaited()
                    result["model"].assert_not_awaited()
                    self.assertEqual(result["calls"], [])

    async def test_deictic_add_commands_reach_model_without_ticket(self):
        for message in ("把这个加进词库", "把这句话加进词库", "加上这个"):
            for quote in ("", QUOTE):
                with self.subTest(message=message, quote=quote):
                    result = await self.replay(message, quote, live=False)
                    result["model"].assert_awaited_once()
                    self.assertIn(quote, result["model"].await_args.args[4])
                    self.assertEqual(result["delivered"], [MODEL_REPLY])
                    self.assertFalse(any(name == "keytao_prepare_reviewed_add"
                                         for name, _args in result["calls"]))

    async def test_quoted_ticket_words_and_candidate_codes_keep_head_behavior(self):
        for quote in ("own-command", "你都没有上手", "候选 NDMEV", "候选ｙｄｂｊｖ"):
            with self.subTest(quote=quote):
                result = await self.replay("加上吧", quote)
                self.assertEqual(
                    (result["flow"], result["stages"][-1], result["delivered"]),
                    ("pending-confirmation", "_stage_finish_scoped_pending_response",
                     [result["old_render"]]),
                )
                self.assertFalse(result["ctx"].verified_current_pending_reply)
                self.assertEqual(result["record"], result["before"])
                result["execute"].assert_not_awaited()
                result["submit"].assert_not_awaited()
                result["model"].assert_not_awaited()
                self.assertEqual(result["calls"], [])

    async def test_other_quoted_deictic_add_commands_are_released(self):
        for message in (
            "加进草稿", "把这两个加进词库", "加入这两个词", "加上吧", "都加进词库吧",
            "把这句话加进词库", "加上这个",
        ):
            with self.subTest(message=message):
                await self.assert_fresh_request(message, QUOTE)

    async def test_released_model_failure_does_not_reenter_ticket(self):
        message = "加欲买桂花同载酒，终不似少年游！这句话"
        result = await self.replay(message, "ticket", model_failure=True)
        baseline = await self.replay(message, "ticket", live=False, model_failure=True)
        self.assertTrue(result["ctx"].released_pending_turn)
        result["model"].assert_awaited_once()
        self.assertEqual(result["completion"].await_count, 2)
        result["fallback_pending"].assert_not_awaited()
        baseline["fallback_pending"].assert_awaited_once()
        self.assertEqual(result["delivered"], baseline["delivered"])
        self.assertEqual(result["delivered"], ["这次服务没有完成处理，本次未写入。"])
        self.assertNotIn(result["old_render"], result["delivered"])
        self.assertEqual(result["record"], result["before"])
        result["execute"].assert_not_awaited()
        result["submit"].assert_not_awaited()
        result["schedule"].assert_not_called()
        self.assertTrue(all(name == "keytao_lookup_by_words_batch"
                            for name, _args in result["calls"]))

    async def test_matched_assent_to_other_quote_does_not_bind_ticket(self):
        result = await self.replay("把这个加入草稿", QUOTE)
        self.assertTrue(result["ctx"].released_pending_turn)
        self.assertNotEqual(result["flow"], "pending-confirmation")
        self.assertEqual(result["delivered"], [MODEL_REPLY])
        result["model"].assert_awaited_once()
        self.assertIn(QUOTE, result["model"].await_args.args[4])
        self.assertEqual(result["record"], result["before"])
        result["execute"].assert_not_awaited()
        result["submit"].assert_not_awaited()
        result["schedule"].assert_not_called()
        self.assertFalse(any(name == "keytao_prepare_reviewed_add"
                             for name, _args in result["calls"]))

    async def test_incident_new_poem_quoting_old_ticket(self):
        await self.assert_fresh_request("加欲买桂花同载酒，终不似少年游！这句话", "ticket")

    async def test_incident_explicit_new_word(self):
        await self.assert_fresh_request("加词 测试词", reviewed_word="测试词")

    async def test_ticket_assent_and_selections_remain_bound(self):
        # Exact HEAD 573a37e delivery, including selection receipts.
        for message, expected, receipt in (
            ("加入", [("你都没有上手", "ndme"), ("有点不切实际", "ydbjvi")], ""),
            ("都加", [("你都没有上手", "ndme"), ("有点不切实际", "ydbjvi")], ""),
            ("你都没有上手 1，有点不切实际 3", [("你都没有上手", "ndme"), ("有点不切实际", "ydbjvi")],
             "词：你都没有上手、有点不切实际\n"),
            ("有点不切实际 3", [("有点不切实际", "ydbjvi")],
             "词：有点不切实际\n未选择：「你都没有上手」；这些词本次未添加。\n"),
        ):
            for quote in ("", "ticket", QUOTE, "own-command"):
                with self.subTest(message=message, quote=quote):
                    result = await self.replay(message, quote)
                    self.assertEqual(result["flow"], "pending-confirmation")
                    self.assertFalse(result["ctx"].generic_intent_is_fresh_command)
                    self.assertFalse(getattr(result["ctx"], "released_pending_turn", False))
                    self.assertFalse(result["ctx"].verified_current_pending_reply)
                    result["execute"].assert_awaited_once()
                    items = result["execute"].await_args.args[0].args["items"]
                    self.assertEqual([(item["word"], item["code"]) for item in items], expected)
                    self.assertEqual(result["delivered"], [receipt + "Fixture ticket executed."])
                    result["model"].assert_not_awaited()
                    result["submit"].assert_not_awaited()
                    self.assertEqual(result["calls"], [])

    async def test_ticket_add_and_submit_remains_bound(self):
        for quote in ("", "ticket", QUOTE, "own-command"):
            with self.subTest(quote=quote):
                result = await self.replay("加入并提交", quote)
                self.assertEqual(result["flow"], "pending-confirmation")
                self.assertFalse(getattr(result["ctx"], "released_pending_turn", False))
                result["submit"].assert_awaited_once()
                self.assertEqual(result["submit"].await_args.args[0], result["before"].state.args["items"])
                self.assertEqual(result["delivered"], ["Fixture ticket submitted."])
                result["model"].assert_not_awaited()
                result["execute"].assert_not_awaited()
                self.assertEqual(result["calls"], [])

    async def test_ticket_cancel_remains_bound(self):
        result = await self.replay("取消")
        self.assertEqual(result["delivered"], ["已取消。"])
        self.assertIsNone(result["record"])
        result["execute"].assert_not_awaited()
        result["model"].assert_not_awaited()

    async def test_malformed_ticket_reference_is_not_released(self):
        for message in (
            "加入 你都没有上手 abc", "加入你都没有上手额外内容",
            "加入编码ndme额外内容", "加入编码YDBJV额外内容", "加入编码ｙｄｂｊｖ额外内容",
            "加入吗？", "他说加入", "不要加入其他词", "加入并删除其他词",
        ):
            with self.subTest(message=message):
                result = await self.replay(message)
                self.assertEqual(result["flow"], "pending-confirmation")
                self.assertEqual(result["stages"][-1], "_stage_finish_scoped_pending_response")
                self.assertEqual(result["record"], result["before"])
                result["execute"].assert_not_awaited()
                result["model"].assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
