"""S57 option advertisements and answer ownership regressions."""

import asyncio
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from keytao_bot.harness.state import SQLiteConversationStateStore
from keytao_bot.utils.offered_options import (
    FORCE_ASSENT_TEXTS,
    is_force_assent,
    matches_previous_bot_option,
    offered_option_intent,
    option_questions_bind_live_state,
    previous_bot_options,
    render_missing_option_ticket,
    structural_option_questions,
)


chat = harness.openai_chat_module
routing = chat._chat_routing


def ticket(options=None):
    return harness.PendingToolConfirm(
        "keytao_shift_phrase_code",
        {"word": "嘢", "target_code": "yeoiav", "batch_id": "",
         "expected_content_version": 0, "confirmed_plan_digest": "a" * 64,
         "_offered_options": options or {"强制调序": "confirm", "保留现顺序": "cancel"}},
        confirmation_source="server_warning",
    )


class S57OptionBoundaryTests(unittest.TestCase):
    def test_real_orchestrator_refuses_unbacked_write_option_question(self):
        async def run():
            raw = "请问是要强制调序，还是保留现顺序？"
            client = harness._FakeClient([harness._FakeAIResponse("stop", raw)])
            orchestrator = harness.AgentOrchestrator(
                client_factory=lambda: client,
                runtime=harness.AgentRuntimeConfig(model="fake-model", max_tokens=1000, temperature=0.0, timeout=10.0),
                skills_manager=harness._FakeSkillsManager(),
                tool_executor=harness.ToolExecutor(lambda _name: None, frozenset()),
                state_store=harness.MemoryConversationStateStore(),
                bind_help_text="bind help", system_prompt_core="system",
            )
            reply = await orchestrator.run(
                "把 嘢 排到 咽 前面",
                harness.AgentRequestContext(platform="qq", user_id="s57-options", mutations_allowed=True),
            )
            self.assertIn("未写入", reply)
            self.assertNotIn("还是", reply)
            self.assertEqual(len(client.completions.calls), 1)
        asyncio.run(run())

    def test_no_ticket_option_question_is_refused_at_delivery(self):
        store = harness.MemoryConversationStateStore()
        key = harness.ConversationAddress.group("qq", "s57-options", "s57-owner")
        raw = "本次未调整。请问是要强制调序，还是保留现顺序？"
        with patch.object(chat, "conversation_state_store", store):
            actual = chat._enforce_advertised_reply_contract(raw, key)
        self.assertNotEqual(raw, actual)
        self.assertNotIn("还是", actual)
        self.assertIn("未写入", actual)

    def test_bound_option_question_round_trips_every_choice(self):
        store = harness.MemoryConversationStateStore()
        key = harness.ConversationAddress.group("qq", "s57-options", "s57-owner")
        state = ticket()
        store.set(key, state)
        raw = "请问是要强制调序，还是保留现顺序？"
        for option, expected in state.args["_offered_options"].items():
            with self.subTest(option=option):
                parsed = routing._pending_tool_assent_intent(state, option)
                self.assertEqual(f"pending_{expected}", parsed.intent)
                self.assertTrue(routing._message_authorizes_pending_state_control(state, option, parsed))
        self.assertTrue(option_questions_bind_live_state(raw, state))
        with patch.object(chat, "conversation_state_store", store):
            self.assertEqual(raw, chat._enforce_advertised_reply_contract(raw, key))

    def test_lexicon_is_exact_and_cannot_bind_incomplete_or_foreign_ticket(self):
        raw = "强制调序还是保留现顺序？"
        malformed = ticket()
        malformed.args.pop("confirmed_plan_digest")
        extra_alias = ticket({"强制调序": "confirm", "保留现顺序": "cancel", "偷偷执行": "confirm"})
        wrong_alias = ticket({"强制": "confirm", "保留现顺序": "cancel"})
        local = ticket()
        local.confirmation_source = "local_preview"
        for state in (None, malformed, local, extra_alias, wrong_alias):
            with self.subTest(state=state):
                self.assertFalse(option_questions_bind_live_state(raw, state))
        store = harness.MemoryConversationStateStore()
        owner = harness.ConversationAddress.group("qq", "s57-group", "owner")
        other = harness.ConversationAddress.group("qq", "s57-group", "other")
        another_group = harness.ConversationAddress.group("qq", "s57-another", "owner")
        store.set(owner, ticket())
        with patch.object(chat, "conversation_state_store", store):
            for address in (other, another_group):
                self.assertNotEqual(raw, chat._enforce_advertised_reply_contract(raw, address))
            store.get_record(owner).execution_id = "already-claimed"
            self.assertNotEqual(raw, chat._enforce_advertised_reply_contract(raw, owner))

    def test_arbitrary_trusted_choices_bind_after_sqlite_reload(self):
        raw = "现在落实，或维持原样？"
        with tempfile.TemporaryDirectory() as directory:
            path = directory + "/pending.db"
            key = harness.ConversationAddress.private("qq", "s57-durable")
            SQLiteConversationStateStore(path).set(key, ticket({"现在落实": "confirm", "维持原样": "cancel"}))
            state = SQLiteConversationStateStore(path).get_record(key).state
            self.assertTrue(option_questions_bind_live_state(raw, state))
            self.assertEqual("confirm", offered_option_intent("现在落实", state))
            self.assertEqual("cancel", offered_option_intent("维持原样", state))

    def test_read_only_question_can_offer_explanatory_topics(self):
        token = chat._current_turn_message.set("介绍编码方案")
        try:
            raw = "你想先了解音码，还是形码？"
            self.assertEqual(raw, chat._prepare_user_facing_reply(raw, None))
        finally:
            chat._current_turn_message.reset(token)

    def test_unrelated_live_ticket_does_not_turn_explanation_into_write_flow(self):
        store = harness.MemoryConversationStateStore()
        key = harness.ConversationAddress.private("qq", "s57-read-live")
        store.set(key, ticket())
        token = chat._current_turn_message.set("介绍编码方案")
        try:
            raw = "你想先了解音码，还是形码？"
            with patch.object(chat, "conversation_state_store", store):
                self.assertEqual(raw, chat._enforce_advertised_reply_contract(raw, key))
        finally:
            chat._current_turn_message.reset(token)

    def test_missing_ticket_recovery_survives_real_final_delivery(self):
        for history in ([], [{"role": "assistant", "content": "请问是要强制调序，还是保留现顺序？"}],
                        [{"role": "assistant", "content": "将把 嘢 排到 咽 之前：嘢 11→10、咽 10→11"}]):
            token = chat._current_turn_message.set("强制调序")
            try:
                raw = render_missing_option_ticket(history)
                actual = chat._prepare_user_facing_reply(raw, None)
                self.assertEqual(raw, actual)
                self.assertIn("引用原提议", actual)
                self.assertIn("未写入", actual)
                if history:
                    self.assertIn("上一条提议", actual)
            finally:
                chat._current_turn_message.reset(token)


class S57OptionStructureTests(unittest.TestCase):
    def test_structural_question_forms_without_confirmation_keywords(self):
        for text, expected in (
            ("请问是要强制调序，还是保留现顺序？", ("强制调序", "保留现顺序")),
            ("「现在落实」或「维持原样」?", ("现在落实", "维持原样")),
            ("选择 A. 现在落实或者 B. 维持原样？", ("现在落实", "维持原样")),
            ("是否进行上述调整？", ()),
        ):
            with self.subTest(text=text):
                parsed = structural_option_questions(text)
                self.assertEqual(len(parsed), 1)
                self.assertEqual(parsed[0].options, expected)
        for text in ("现在落实或维持原样。", "这是含有或字的普通说明", "怎样查看草稿？"):
            self.assertFalse(structural_option_questions(text), text)

    def test_force_words_are_closed_controls_not_questions_or_negated_text(self):
        for text in FORCE_ASSENT_TEXTS:
            self.assertTrue(is_force_assent(text), text)
            self.assertTrue(is_force_assent("  " + text + "！"), text)
        for text in ("不要强制", "强制？", "强制吗", "他说强制", "「强制」", "硬加安全吗", "强制加错字"):
            self.assertFalse(is_force_assent(text), text)

    def test_only_immediately_preceding_bot_options_claim_the_answer(self):
        history = [{"role": "assistant", "content": "强制调序还是保留现顺序？"}]
        self.assertEqual(previous_bot_options(history), ("强制调序", "保留现顺序"))
        self.assertTrue(matches_previous_bot_option("「强制 调序」", history))
        self.assertFalse(matches_previous_bot_option("强制调序？", history))
        self.assertFalse(matches_previous_bot_option("强制调序", history + [{"role": "assistant", "content": "好的。"}]))
        self.assertFalse(matches_previous_bot_option("强制调序", history + [{"role": "user", "content": "还在吗"}]))


class S57OptionClaimTests(unittest.TestCase):
    def test_live_force_and_exact_options_skip_intent_models_in_both_stages(self):
        async def run():
            state = ticket({"现在落实": "confirm", "维持原样": "cancel"})
            for message in (*FORCE_ASSENT_TEXTS, "现在落实", "维持原样"):
                with self.subTest(message=message):
                    ctx = SimpleNamespace(
                        current_pending_record=SimpleNamespace(state=state),
                        normalized_message_text=message, eviction_modified_add=None,
                        compound_eviction_add_plan=None, scoped_pending_intent=None,
                        resolved_advertised_words=(), quoted_pending_add_control=False,
                        command_intent_for=AsyncMock(side_effect=AssertionError("generic intent model called")),
                    )
                    self.assertFalse(await chat._stage_apply_scoped_pending_intent(ctx))
                    expected = "pending_cancel" if message == "维持原样" else "pending_confirm"
                    self.assertEqual(ctx.generic_command_intent.intent, expected)
                    ctx.command_intent_for.assert_not_awaited()
                    with patch.object(routing, "_classify_message_command_intent", AsyncMock(side_effect=AssertionError("pending intent model called"))):
                        _cache, classify = routing.command_intent_memoizer(message, message)
                        self.assertEqual((await classify(state)).intent, expected)
        asyncio.run(run())

    def test_no_ticket_offered_answer_is_claimed_before_word_discovery(self):
        async def run():
            key = harness.ConversationAddress.group("qq", "s57-claim", "owner")
            history = [{"role": "assistant", "content": "现在落实还是维持原样？"}]
            ctx = SimpleNamespace(
                conv_key=key, normalized_message_text="现在落实", memory_context=None,
                platform="qq", user_id="owner", bot=None, event=None, QQMessageSegment=None,
            )
            with patch.object(chat, "conversation_state_store", harness.MemoryConversationStateStore()), patch.object(
                chat, "get_history", return_value=history,
            ), patch.object(chat, "remember_conversation"), patch.object(chat, "_finish_ai_chat_response", AsyncMock()) as finish:
                self.assertTrue(await chat._stage_claim_offered_answer(ctx))
                self.assertIn("未写入", ctx.response)
                self.assertEqual(finish.await_count, 1)
        asyncio.run(run())

    def test_genuine_four_character_query_without_options_remains_unclaimed(self):
        async def run():
            key = harness.ConversationAddress.private("qq", "s57-word")
            ctx = SimpleNamespace(conv_key=key, normalized_message_text="海阔天空")
            with patch.object(chat, "get_history", return_value=[{"role": "assistant", "content": "请发要查询的词。"}]):
                self.assertFalse(await chat._stage_claim_offered_answer(ctx))
            calls = []

            async def tool(name, args, *_args, **_kwargs):
                calls.append((name, args))
                if name == "keytao_lookup_by_word":
                    return json.dumps({"success": True, "phrases": []})
                if name == "keytao_pending_items_by_words":
                    return json.dumps({"success": True, "complete": True, "items": []})
                if name == "keytao_prepare_reviewed_add":
                    return json.dumps({
                        "success": True, "word": "海阔天空", "recommendedCode": "hktk",
                        "needsManualReview": True, "manualReviewReason": "离线核验",
                        "pronunciations": [{"pinyin": "hǎi kuò tiān kōng", "recommendedCode": "hktk",
                            "sources": [{"source": "汉典"}], "candidateStatuses": [
                                {"code": "hktk", "occupied": False, "words": [], "label": "空位"},
                                {"code": "hktkv", "occupied": False, "words": [], "label": "空位"},
                            ]}],
                    })
                raise AssertionError(name)

            store = harness.MemoryConversationStateStore()
            with patch.object(chat, "conversation_state_store", store), patch.object(
                chat, "call_tool_function", side_effect=tool,
            ), patch.object(chat, "_classify_simple_word_query_intent", AsyncMock(return_value=harness.SimpleWordQueryIntent(
                True, ("海阔天空",), "word_lookup", 1.0,
            ))):
                reply = await harness.chat_commands_module._try_handle_simple_single_word_query(
                    "海阔天空", "qq", "s57-word", key,
                )
            self.assertIn("审词", reply)
            self.assertIsInstance(store.get(key), harness.PendingAddWord)
            self.assertIn(("keytao_prepare_reviewed_add", {"word": "海阔天空"}), calls)
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
