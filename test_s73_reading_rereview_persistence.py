"""Offline S73 rereviews through real stages, orchestration, and assent."""

import copy
import json
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from keytao_bot.utils.observability import current_turn_metrics


chat = harness.openai_chat_module
commands = harness.chat_commands_module
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


class ReadingRereviewPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
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
            self.delivered.append(chat._enforce_advertised_reply_contract(response, self.key))

        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for patcher in (
            patch.object(chat, "conversation_state_store", self.store),
            patch.object(chat, "draft_operation_coordinator", harness.DraftOperationCoordinator()),
            patch.object(chat, "extract_memory_context", AsyncMock(side_effect=lambda *_: self.memory)),
            patch.object(chat, "get_history", return_value=[]),
            patch.object(chat, "remember_conversation"),
            patch.object(chat, "schedule_memory_compaction"),
            patch.object(chat.memory_store, "get_context_block", return_value=""),
            patch.object(chat, "get_group_history_context", return_value=""),
            patch.object(chat, "_classify_message_command_intent", AsyncMock(return_value=harness.MessageCommandIntent())),
            patch.object(chat, "_classify_simple_word_query_intent", AsyncMock(return_value=harness.SimpleWordQueryIntent(False))),
            patch.object(chat, "AsyncOpenAI", return_value=self.client),
            patch.object(chat, "OPENAI_API_KEY", "fixture-key"),
            patch.object(chat, "skills_manager", skills),
            patch.object(chat, "tool_executor", harness.ToolExecutor(
                skills.get_tool_function, frozenset({"keytao_prepare_reviewed_add"}),
            )),
            patch.object(chat, "call_tool_function", side_effect=self.tool),
            patch.object(commands.user_resolver, "resolve_actor_binding", AsyncMock(return_value=True)),
            patch.object(chat, "_schedule_background_draft_operation", self.schedule),
            patch.object(chat, "_finish_ai_chat_response", side_effect=finish),
            patch.object(chat, "_finish_ai_chat_matcher", side_effect=finish),
        ):
            self.stack.enter_context(patcher)

    async def tool(self, name, args, platform, user_id, **kwargs):
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

    async def turn(self, message, quote=""):
        ctx = chat.TurnContext(
            bot=object(), event=object(), platform="qq", user_id=self.key.actor_id,
            message_text=message, reply_reference=harness.ReplyReferenceInfo(
                is_reply=bool(quote), is_to_bot=bool(quote), sender_id="s73-bot", text=quote,
            ),
        )
        token = harness.begin_turn_metrics("qq", "group")
        self.stages = []
        try:
            for stage in chat.STAGES[chat.STAGES.index(chat._stage_normalize_message_text):]:
                self.stages.append(stage.__name__)
                if await stage(ctx):
                    break
            self.flow = current_turn_metrics().flow
        finally:
            harness.end_turn_metrics(token)
        self.schedule.assert_not_called()
        return self.delivered[-1]

    async def initial_review(self, word="田调"):
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

    async def rereview(self, style, quote, message=REREVIEW, word="田调"):
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

    async def assert_incident(self, style):
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
              f"{record.origin_prompt_digest == commands._prompt_capability_digest(text)}")
        await self.turn("加入")
        self.assertEqual(len(self.writes), 1)
        args, kwargs = self.writes[0]
        self.assertEqual((args["word"], args["code"]), ("田调", "tmdc"))
        self.assertEqual(len(self.client.completions.calls), 2)
        self.assertEqual(
            kwargs["trusted_reviewed_items_by_key"][("田调", "tmdc")]["pinyin"], "tián diào",
        )

    async def test_incident_dash(self):
        await self.assert_incident("dash")

    async def test_incident_colon(self):
        await self.assert_incident("colon")

    async def test_incident_bold(self):
        await self.assert_incident("bold")

    async def test_incident_prose(self):
        await self.assert_incident("prose")

    async def test_bound_prose_still_renders_review(self):
        with patch.dict(REPLIES, {"prose": "• 「田调」→ tmdc（推荐）\n回复「加入」写入草稿。"}):
            await self.assert_incident("prose")

    async def test_other_word_ticket_is_untouched(self):
        quote = await self.initial_review("天条")
        before = copy.deepcopy(self.store.get_record(self.key))
        self.review = review_fixture(corrected=True)
        await self.rereview("dash", quote)
        self.assertEqual(self.before_final_reply, before)
        self.assertEqual(self.store.get(self.key).word, "田调")

    async def test_other_actor_ticket_is_untouched(self):
        quote = await self.initial_review()
        other_key = self.key
        before = copy.deepcopy(self.store.get_record(other_key))
        self.memory = harness.ChatMemoryContext(
            platform="qq", user_id="s73-other", space_type="group", space_id="s73-group",
        )
        self.key = self.memory.conversation_address
        self.review = review_fixture(corrected=True)
        await self.rereview("dash", quote)
        self.assertEqual(self.store.get_record(other_key), before)

    async def test_tool_confirmation_is_untouched(self):
        self.store.set(self.key, harness._sealed_submit_advertisement_ticket())
        before = copy.deepcopy(self.store.get_record(self.key))
        self.review = review_fixture(corrected=True)
        await self.rereview("dash", "田调，" + REREVIEW, message="把这个加进词库")
        self.assertEqual(self.before_final_reply, before)
        self.assertIsInstance(self.store.get(self.key), harness.PendingAddWord)
        self.assertEqual(self.store.get(self.key).word, "田调")

    async def assert_new_word_dash(self, message):
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

    async def test_other_word_dash_replaces_ticket_and_adds_new_word(self):
        await self.initial_review()
        await self.assert_new_word_dash("帮我把天调加进词库，读音 tian diao")

    async def assert_batch_release_dash(self, message):
        self.reviews["地调"] = review_fixture(corrected=True, word="地调")
        await self.turn("田调 地调")
        record = self.store.get_record(self.key)
        self.assertIsInstance(record.state, harness.PendingToolConfirm)
        self.assertTrue(record.state.args["_reviewed_multi_word"])
        self.assertFalse(record.origin_prompt_digest)
        self.assertEqual(self.client.completions.calls, [])
        await self.assert_new_word_dash(message)

    async def test_batch_ticket_releases_join_word_dash(self):
        await self.assert_batch_release_dash("加入天调")

    async def test_batch_ticket_releases_add_word_dash(self):
        await self.assert_batch_release_dash("帮我加个词：天调")

    async def test_other_word_prose_does_not_render_old_candidates(self):
        await self.initial_review()
        before = copy.deepcopy(self.store.get_record(self.key))
        self.review = review_fixture(corrected=True, word="天调")
        text = await self.rereview(
            "prose", "", message="帮我把天调加进词库，读音 tian diao", word="天调",
        )
        self.assertEqual(self.before_final_reply, before)
        self.assertEqual(text, "当前没有可验证的可执行操作，本次未写入。")
        self.assertEqual(self.store.get(self.key), before.state)

    async def test_prose_without_fresh_review_does_not_render_old_candidates(self):
        quote = await self.initial_review()
        before = copy.deepcopy(self.store.get_record(self.key))
        self.client.completions.responses.append(harness._FakeAIResponse("stop", REPLIES["prose"]))
        text = await self.turn(REREVIEW, quote)
        self.assertEqual(self.flow, "general")
        self.assertEqual(text, "当前没有可验证的可执行操作，本次未写入。")
        self.assertEqual(self.store.get_record(self.key).nonce, before.nonce)
        self.assertEqual(self.store.get(self.key), before.state)
        self.assertEqual(self.writes, [])

    async def test_other_word_review_with_old_word_prose_requires_fresh_nonce(self):
        for reply in (
            "你说的是「天调」不是「田调」吧？「天调」已审核，回复「加入」写入草稿。",
            "「田调」已审核，回复「加入」写入草稿。",
        ):
            with self.subTest(reply=reply):
                self.store.delete(self.key)
                self.client.completions.calls.clear()
                self.client.completions.responses.clear()
                self.calls.clear()
                self.writes.clear()
                await self.initial_review()
                before = copy.deepcopy(self.store.get_record(self.key))
                self.assertEqual(
                    [code for code, _ in before.state.server_candidates], list(OLD_CODES),
                )
                self.review = review_fixture(corrected=True, word="天调")
                review_args = {"word": "天调", "requested_reading": "tian diao"}
                self.client.completions.responses.extend([
                    harness._FakeAIResponse("tool_calls", "", [SimpleNamespace(
                        id="s73-review", type="function", function=SimpleNamespace(
                            name="keytao_prepare_reviewed_add", arguments=json.dumps(review_args),
                        ),
                    )]),
                    harness._FakeAIResponse("stop", reply),
                ])
                create = self.client.completions.create

                async def complete(**kwargs):
                    if len(self.client.completions.calls) == 1:
                        self.before_final_reply = copy.deepcopy(self.store.get_record(self.key))
                    return await create(**kwargs)

                with patch.object(self.client.completions, "create", side_effect=complete):
                    text = await self.turn("帮我把天调加进词库，读音 tian diao")
                self.assertEqual(self.flow, "general")
                self.assertIn(("keytao_prepare_reviewed_add", review_args), self.calls)
                self.assertEqual(len(self.client.completions.calls), 2)
                self.assertEqual(self.before_final_reply, before)
                for stale in (*OLD_CODES, "tian tiao"):
                    self.assertNotIn(stale, text)
                self.assertEqual(self.store.get_record(self.key).nonce, before.nonce)
                self.assertEqual(self.store.get(self.key), before.state)
                self.assertEqual(self.writes, [])

    async def test_fresh_review_does_not_back_another_word_prose(self):
        quote = await self.initial_review()
        self.review = review_fixture(corrected=True)
        with patch.dict(REPLIES, {"prose": REPLIES["prose"].replace("田调", "天调")}):
            text = await self.rereview("prose", quote)
        self.assertEqual(text, "当前没有可验证的可执行操作，本次未写入。")
        self.assertEqual(self.store.get(self.key).recommended_code, "tmdc")

    async def test_unresolved_review_removes_stale_ticket(self):
        quote = await self.initial_review()
        self.review = {
            "success": True, "word": "田调", "pronunciationUnresolved": True,
            "recommendedCode": "", "message": "读音尚未完成核验",
        }
        text = await self.rereview("colon", quote)
        self.assertIsNone(self.store.get_record(self.key))
        for stale in (*OLD_CODES, "tian tiao"):
            self.assertNotIn(stale, text)

    async def test_short_inventory_removes_stale_ticket(self):
        quote = await self.initial_review()
        self.review = review_fixture(corrected=True)
        self.review["pronunciations"][0]["candidateStatuses"] = [
            {"code": "tmdc", "occupied": False, "words": []},
        ]
        text = await self.rereview("prose", quote)
        self.assertIsNone(self.store.get_record(self.key))
        self.assertNotIn("tmtc", text)

    async def test_unresolved_inventory_cannot_reestablish_ticket(self):
        quote = await self.initial_review()
        self.review = review_fixture(corrected=True)
        self.review["pronunciationUnresolved"] = True
        text = await self.rereview("dash", quote)
        self.assertIsNone(self.store.get_record(self.key))
        self.assertNotIn("tmtc", text)

    async def test_invalid_status_removes_stale_ticket(self):
        quote = await self.initial_review()
        self.review = review_fixture(corrected=True)
        self.review["pronunciations"][0]["candidateStatuses"][0]["occupied"] = "unknown"
        text = await self.rereview("colon", quote)
        self.assertIsNone(self.store.get_record(self.key))
        self.assertNotIn("tmtc", text)

    async def test_prose_without_state_still_fails_honestly(self):
        text = chat._enforce_advertised_reply_contract(REPLIES["prose"], self.key)
        self.assertEqual(text, "当前没有可验证的可执行操作，本次未写入。")


if __name__ == "__main__":
    unittest.main()
