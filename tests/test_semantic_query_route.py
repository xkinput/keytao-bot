"""Offline model-first query, conversation and pending-ticket isolation checks."""

import asyncio
import copy
import json
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from tests import test_state_machine as harness
from tests.test_semantic_parser import FakeClient, raw_operation

from keytao_bot.harness.conversation import ConversationAddress
from keytao_bot.harness.state import (
    DraftOperationCoordinator,
    MemoryConversationStateStore,
    PendingAddWord,
    PendingToolConfirm,
)


class SemanticQueryRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.chat = harness.openai_chat_module
        self.commands = harness.chat_commands_module
        self.store = MemoryConversationStateStore()
        self.memory = harness.ChatMemoryContext(
            platform="qq", user_id="semantic-query-owner", space_type="group",
            space_id="semantic-query-group", speaker_name="Offline owner",
        )
        self.author = self.memory.conversation_address
        self.other = ConversationAddress.group("qq", "semantic-query-group", "other-user")
        self.client = FakeClient()
        self.finish = AsyncMock()
        self.real_main = self.chat.get_ai_response_core
        self.main = AsyncMock(return_value="这是离线解释。")
        self.remember = MagicMock()
        self.tool = AsyncMock(side_effect=self.lookup)
        self.entries = {"济济": "jjes", "积极": "jjuu", "午时": "wj"}
        self.pending_items = []
        self.candidate_codes = {"云翻": "yjfj", "发薪": "fsxb", "发薪日": "fxr", "济济": "jjes"}
        self.history = []
        self.patches = (
            patch.dict(os.environ, {
                "KEYTAO_SEMANTIC_ROUTE_OPS": "query,chat,help",
                "KEYTAO_SEMANTIC_SHADOW": "0",
            }),
            patch.object(self.chat, "AsyncOpenAI", return_value=self.client),
            patch.object(self.chat, "WORD_QUERY_INTENT_MODEL", "offline-model"),
            patch.object(self.chat, "conversation_state_store", self.store),
            patch.object(self.chat, "draft_operation_coordinator", DraftOperationCoordinator()),
            patch.object(self.chat, "call_tool_function", self.tool),
            patch.object(self.chat, "get_ai_response_core", self.main),
            patch.object(self.chat, "build_reply_context", AsyncMock(return_value="Offline quote context")),
            patch.object(self.chat, "get_history", side_effect=lambda *args: self.history),
            patch.object(self.chat, "remember_conversation", self.remember),
            patch.object(self.chat, "schedule_memory_compaction"),
            patch.object(self.chat, "_finish_ai_chat_response", self.finish),
            patch.object(self.chat.logger, "info"),
            patch.object(self.chat.logger, "warning"),
            patch.object(self.commands.user_resolver, "resolve_actor_binding", AsyncMock(return_value=True)),
        )
        for context in self.patches:
            context.start()
            self.addCleanup(context.stop)
        self.turn_token = self.chat._current_conversational_turn.set(None)
        self.addCleanup(self.chat._current_conversational_turn.reset, self.turn_token)

    async def asyncTearDown(self):
        tasks = tuple(self.chat._semantic_shadow_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.chat._semantic_shadow_tasks.clear()

    async def lookup(self, name, args, platform, user_id, **kwargs):
        self.assertEqual((platform, user_id), ("qq", self.author.actor_id))
        if name == "keytao_encode":
            return json.dumps({"success": False, "word": args["word"]})
        if name == "keytao_lookup_by_words_batch":
            return json.dumps({"success": True, "results": [
                {"success": True, "word": word, "phrases": [
                    {"word": word, "code": self.entries[word], "type": "Phrase"},
                ] if word in self.entries else []}
                for word in args["words"]
            ]}, ensure_ascii=False)
        if name == "keytao_pending_items_by_words":
            self.assertEqual(set(args), {"words"})
            return json.dumps({
                "success": True, "complete": True,
                "items": [row for row in self.pending_items if row["word"] in args["words"]],
            }, ensure_ascii=False)
        if name == "keytao_prepare_reviewed_add":
            self.assertEqual(set(args), {"word"})
            word = args["word"]
            code = self.candidate_codes[word]
            return json.dumps({
                "success": True, "word": word, "type": "Phrase",
                "recommendedCode": code, "needsManualReview": False,
                "preSubmitAudit": {
                    "success": True, "verdict": "pass", "autoApprove": True,
                    "summary": "Offline reviewed fixture", "issues": [],
                },
                "pronunciations": [{
                    "pinyin": "offline reading", "recommendedCode": code,
                    "sources": [{"source": "offline"}],
                    "candidateStatuses": [
                        {"code": value, "occupied": False, "label": "空位", "words": []}
                        for value in (code, code + "a")
                    ],
                }],
            }, ensure_ascii=False)
        self.assertEqual(name, "keytao_lookup_by_word", "Unexpected offline tool")
        self.assertEqual(set(args), {"word"})
        word = args["word"]
        code = self.entries.get(word)
        return json.dumps({
            "success": True,
            "phrases": [{"word": word, "code": code, "type": "Phrase"}] if code else [],
        }, ensure_ascii=False)

    def context(self, message):
        return self.chat.TurnContext(
            bot=None, event=None, platform="qq", user_id=self.author.actor_id,
            message_text=message, normalized_message_text=message,
            conv_key=self.author, space_key=self.author.space_key,
            memory_context=self.memory, owner_label="Offline owner",
            command_intent_for=AsyncMock(return_value=self.chat.MessageCommandIntent()),
        )

    def set_model(self, op="query", operands=("济济",), **changes):
        self.client.create.side_effect = None
        raw = raw_operation(op=op, operands=list(operands), codes=[], frame="none", **changes)
        self.client.create.return_value.choices[0].message.content = json.dumps(raw, ensure_ascii=False)
        return raw

    def ticket(self, owner, word="午时", code="wj"):
        pending = PendingAddWord(
            word, code, [(code, False), (code + "a", False)],
            server_candidates=[(code, False), (code + "a", False)],
        )
        self.assertTrue(self.store.set(owner, pending, space_key=owner.space_key, owner_label="Offline owner"))
        record = self.store.get_record(owner)
        prompt = self.chat._render_live_single_candidate_record(record)
        self.assertTrue(self.store.bind_origin_prompt_digest(owner, self.chat._prompt_capability_digest(prompt)))
        self.history = [{"role": "assistant", "content": prompt}]
        return record, copy.deepcopy(record), prompt

    def assert_preserved(self, owner, record, original):
        self.assertIs(self.store.get_record(owner), record)
        self.assertEqual(record, original)

    def assert_main(self, message, *lookup_fragments):
        self.main.assert_awaited_once()
        call = self.main.await_args
        self.assertEqual(call.args[0], message)
        context = call.args[4] if len(call.args) > 4 else call.kwargs["reply_context"]
        for fragment in lookup_fragments:
            self.assertIn(fragment, context)
        self.assertTrue(call.kwargs["released_pending_turn"])
        self.assertTrue(call.kwargs["preserve_pending"])
        self.assertTrue(call.kwargs["conversation_only"])

    async def assert_finished(self, ctx):
        self.assertTrue(await self.chat._stage_understand_operation_early(ctx))
        self.finish.assert_awaited_once()
        self.assertEqual(self.finish.await_args.args[4], ctx.response)
        self.remember.assert_called_once_with(self.author, self.memory, ctx.normalized_message_text, ctx.response)

    async def test_quoted_other_owner_meaning_query_looks_up_only_operand_and_calls_main(self):
        record, original, prompt = self.ticket(self.other)
        ctx = self.context("济济啥意思")
        ctx.reply_reference = harness.ReplyReferenceInfo(
            is_reply=True, is_to_bot=True, mentioned_user_ids=(self.other.actor_id,), text=prompt,
        )
        self.set_model()
        await self.assert_finished(ctx)
        self.tool.assert_awaited_once_with("keytao_lookup_by_word", {"word": "济济"}, "qq", self.author.actor_id)
        self.assert_main("济济啥意思", "济济", "jjes")
        self.assertIsNone(self.store.get_record(self.author))
        self.assert_preserved(self.other, record, original)

    async def test_code_question_uses_operand_lookup_and_passes_codes_to_main(self):
        ctx = self.context("积极怎么打")
        self.main.return_value = "积极的编码是 jjuu。"
        self.set_model(operands=("积极",))
        await self.assert_finished(ctx)
        self.tool.assert_awaited_once_with("keytao_lookup_by_word", {"word": "积极"}, "qq", self.author.actor_id)
        self.assert_main("积极怎么打", "积极", "jjuu")
        self.assertIn("jjuu", ctx.response)
        self.assertIsNone(self.store.get_record(self.author))

    async def test_exact_word_returns_existing_code_without_main_or_ticket(self):
        ctx = self.context("积极")
        self.set_model(operands=("积极",))
        await self.assert_finished(ctx)
        self.assertIn("积极", ctx.response)
        self.assertIn("jjuu", ctx.response)
        self.main.assert_not_awaited()
        self.tool.assert_awaited_once()
        self.assertIsNone(self.store.get_record(self.author))

    async def test_missing_word_reuses_normal_single_word_candidates_and_ticket(self):
        ctx = self.context("云翻")
        self.set_model(operands=("云翻",))
        await self.assert_finished(ctx)
        self.assertIn("云翻", ctx.response)
        self.assertIn("1. yjfj", ctx.response)
        record = self.store.get_record(self.author)
        self.assertIsInstance(record.state, PendingAddWord)
        self.assertEqual(record.state.word, "云翻")
        self.assertEqual(record.owner_key, self.author)
        self.assertTrue(self.chat._advertised_reply_matches_live_record(ctx.response, record))
        self.assertEqual(self.prepared_words(), ["云翻"])
        self.main.assert_not_awaited()
        self.client.create.assert_awaited_once()

    async def test_missing_meaning_keeps_explanation_when_main_echoes_candidates(self):
        self.entries.pop("济济")
        ctx = self.context("济济什么意思")
        self.set_model()
        self.main.return_value = (
            "济济指众多人才聚集。\n「济济」候选编码：\n"
            "1. jjes — 空位（推荐）\n2. jjesa — 空位"
        )
        await self.assert_finished(ctx)
        self.assertIn("济济指众多人才聚集。", ctx.response)
        self.assertEqual(ctx.response.count("1. jjes"), 1)
        self.assertEqual(self.prepared_words(), ["济济"])
        self.assertTrue(self.chat._advertised_reply_matches_live_record(ctx.response, self.store.get_record(self.author)))

    async def test_multiple_missing_meaning_operands_keep_explanation_in_normal_batch_reply(self):
        ctx = self.context("发薪 发薪日什么意思")
        self.set_model(operands=("发薪", "发薪日"))
        await self.assert_finished(ctx)
        self.assertIn("这是离线解释。", ctx.response)
        self.assertEqual(self.prepared_words(), ["发薪", "发薪日"])
        self.assertIsInstance(self.store.get(self.author), PendingToolConfirm)
        self.assertTrue(self.chat._advertised_reply_matches_live_record(ctx.response, self.store.get_record(self.author)))
        self.assert_main(ctx.normalized_message_text, "发薪", "发薪日", "未在词库")

    async def test_present_code_survives_alongside_multiple_missing_candidates(self):
        ctx = self.context("积极 发薪 发薪日")
        self.set_model(operands=("积极", "发薪", "发薪日"))
        await self.assert_finished(ctx)
        self.assertIn("jjuu", ctx.response)
        self.assertEqual(self.prepared_words(), ["发薪", "发薪日"])
        self.assertTrue(self.chat._advertised_reply_matches_live_record(ctx.response, self.store.get_record(self.author)))
        self.main.assert_not_awaited()

    def prepared_words(self):
        return [call.args[1]["word"] for call in self.tool.await_args_list
                if call.args[0] == "keytao_prepare_reviewed_add"]

    async def test_a4_mixed_code_query_reviews_only_han_without_empty_explanation(self):
        self.candidate_codes["大绿"] = "dsllv"
        for existing in (True, False):
            with self.subTest(existing=existing):
                self.store.delete(self.author)
                self.tool.reset_mock()
                self.finish.reset_mock()
                self.remember.reset_mock()
                self.main.reset_mock()
                self.main.return_value = "候选编码：dsllva，词库查询完成。"
                if existing:
                    self.entries["大绿"] = "dsllv"
                else:
                    self.entries.pop("大绿", None)
                self.set_model(operands=("ds", "大绿"))
                ctx = self.context("ds大绿")
                await self.assert_finished(ctx)
                self.assertEqual(self.prepared_words(), [] if existing else ["大绿"])
                self.assertIn("大绿", ctx.response)
                self.assertIn("dsllv", ctx.response)
                self.assertNotIn("这条回复没有留下", ctx.response)
                self.assertNotIn("读音服务", ctx.response)
                self.assertNotIn("「ds」", ctx.response)
                self.assertTrue(all(
                    "ds" not in (call.args[1].get("words") or [call.args[1].get("word")])
                    for call in self.tool.await_args_list
                ))

    async def test_b2_bare_lexical_list_precedes_semantic_after_write(self):
        self.candidate_codes.update({"气笑了": "qxle", "看笑了": "kxle"})
        for recent_write in (False, True):
            with self.subTest(recent_write=recent_write):
                self.store.delete(self.author)
                self.tool.reset_mock()
                self.finish.reset_mock()
                self.remember.reset_mock()
                self.client.create.reset_mock()
                self.main.reset_mock()
                if recent_write:
                    self.store.set(self.author, PendingToolConfirm(
                        "keytao_submit_batch", {"_recent_own_write": True},
                    ), space_key=self.author.space_key)
                self.set_model("chat", ())
                ctx = self.context("气笑了 看笑了")
                await self.assert_finished(ctx)
                self.assertEqual(ctx.operation_request.op, "query")
                self.assertEqual(ctx.operation_request.operands, ("气笑了", "看笑了"))
                self.assertEqual(self.prepared_words(), ["气笑了", "看笑了"])
                self.assertIn("气笑了", ctx.response)
                self.assertIn("看笑了", ctx.response)
                self.assertIsInstance(self.store.get(self.author), PendingToolConfirm)
                self.client.create.assert_not_awaited()
                self.main.assert_not_awaited()
        fallback = self.chat.AgentOrchestrator._finalize_reply(
            "气笑了 看笑了", "请原样重新发送这条指令。", {},
        )
        self.assertIn("还没有确定词条和编码", fallback)
        self.assertNotIn("已绑定写操作", fallback)

    async def test_b3_coordinated_query_preserves_live_ticket_and_looks_up_all_words(self):
        self.candidate_codes.update({"禁塑": "jbsi", "禁塑令": "jsl", "加班": "jsbj", "休息": "xqxy",
                                     "加速度": "jsd", "加速器": "jsq"})
        for message, op, operands, expected_words, live_ticket in (
            ("加禁塑还是禁塑令呢", "chat", (), ("禁塑", "禁塑令"), True),
            ("加禁塑还是禁塑令呢", "add", ("禁塑",), ("禁塑", "禁塑令"), False),
            ("禁塑和禁塑令", "chat", (), ("禁塑", "禁塑令"), True),
            ("禁塑或禁塑令", "query", ("禁塑", "禁塑令"), ("禁塑", "禁塑令"), True),
            ("比较一下禁塑、禁塑令", "query", ("禁塑", "禁塑令"), ("禁塑", "禁塑令"), True),
            ("加班还是休息", "chat", (), ("加班", "休息"), False),
            ("加速度和加速器", "chat", (), ("加速度", "加速器"), False),
            ("加速度或加速器", "query", ("加速度", "加速器"), ("加速度", "加速器"), False),
        ):
            with self.subTest(message=message):
                self.store.delete(self.author)
                self.tool.reset_mock()
                self.finish.reset_mock()
                self.remember.reset_mock()
                self.main.reset_mock()
                if live_ticket:
                    record, original, _prompt = self.ticket(self.author, "禁塑", "jbsi")
                else:
                    self.history.clear()
                self.set_model(op, operands)
                ctx = self.context(message)
                await self.assert_finished(ctx)
                self.assertEqual(self.prepared_words(), list(expected_words))
                for word in expected_words:
                    self.assertIn(word, ctx.response)
                self.assertFalse(str(ctx.response).startswith("- "))
                if live_ticket:
                    self.assert_preserved(self.author, record, original)
                else:
                    self.assertIsNone(self.store.get_record(self.author))
                self.main.assert_not_awaited()
        self.assertEqual(
            self.chat._chat_routing._coordinated_word_query_words("把禁塑和禁塑令删除"), (),
        )
        self.assertEqual(
            self.chat._chat_routing._coordinated_word_query_words("午时和务实就按这个组合吧"), (),
        )

    async def test_c5_bare_single_word_overrides_chat_text(self):
        self.assertFalse(self.chat._chat_routing._is_chat_word_query_target("不行"))
        self.entries["看不懂"] = "kbdi"
        for op, reply in (
            ("chat", "之后想再加词，直接发「加词 + 词语」就行。"),
            ("chat", "暂时没有更多信息。"),
            ("other", "暂时没有更多信息。"),
            ("swap", "暂时没有更多信息。"),
        ):
            for live_ticket in (False,):
                with self.subTest(op=op, reply=reply, live_ticket=live_ticket):
                    self.store.delete(self.author)
                    self.tool.reset_mock()
                    self.finish.reset_mock()
                    self.remember.reset_mock()
                    self.main.reset_mock()
                    self.main.return_value = reply
                    if live_ticket:
                        record, original, _prompt = self.ticket(self.author)
                    self.set_model(op, ())
                    ctx = self.context("看不懂")
                    await self.assert_finished(ctx)
                    self.assertIn("看不懂", ctx.response)
                    self.assertIn("kbdi", ctx.response)
                    self.main.assert_not_awaited()
                    if live_ticket:
                        self.assert_preserved(self.author, record, original)

    async def test_c6_explicit_query_list_drops_leading_verb(self):
        self.entries.update({"查报": "wsbz", "茶包": "wsbz"})
        for message in ("查词 查报 茶包", "查询：查报、茶包", "查:查报，茶包"):
            with self.subTest(message=message):
                self.tool.reset_mock()
                self.finish.reset_mock()
                self.remember.reset_mock()
                self.set_model(operands=("查词", "查报", "茶包"))
                ctx = self.context(message)
                await self.assert_finished(ctx)
                self.assertIn("查报 wsbz", ctx.response)
                self.assertIn("茶包 wsbz", ctx.response)
                self.assertEqual(
                    await self.chat._get_simple_word_query_words(message), ("查报", "茶包"),
                )
                self.assertEqual(
                    self.tool.await_args_list[0].args[1], {"words": ["查报", "茶包"]},
                )
                self.assertNotIn("「查词」", ctx.response)

    async def test_repeated_missing_operand_creates_one_single_word_ticket(self):
        ctx = self.context("发薪 发薪")
        self.set_model(operands=("发薪", "发薪"))
        await self.assert_finished(ctx)
        self.assertEqual(self.prepared_words(), ["发薪"])
        self.assertIsInstance(self.store.get(self.author), PendingAddWord)
        self.assertEqual(self.store.get(self.author).word, "发薪")
        self.assertEqual(ctx.response.count("「发薪」未收录："), 1)
        self.assertIn("1. fsxb", ctx.response)
        self.main.assert_not_awaited()

    async def test_two_missing_operands_share_one_normal_batch_ticket(self):
        other_record, original, _prompt = self.ticket(self.other)
        ctx = self.context("发薪 发薪日")
        self.set_model(operands=("发薪", "发薪日"))
        with patch.object(self.store, "set", wraps=self.store.set) as persist:
            await self.assert_finished(ctx)
        self.assertEqual(persist.call_count, 1)
        record = self.store.get_record(self.author)
        self.assertIsInstance(record.state, PendingToolConfirm)
        self.assertEqual(record.state.function_name, "keytao_batch_add_to_draft")
        self.assertEqual(record.owner_key, self.author)
        self.assertEqual(record.state.args["_query_words"], ["发薪", "发薪日"])
        self.assertEqual([scope["word"] for scope in record.state.args["_candidate_scopes"]], ["发薪", "发薪日"])
        self.assertEqual(self.prepared_words(), ["发薪", "发薪日"])
        self.assertEqual(ctx.response.count("1. fsxb"), 1)
        self.assertEqual(ctx.response.count("1. fxr"), 1)
        self.assertNotIn("「发薪 发薪日」", ctx.response)
        self.assertTrue(self.chat._advertised_reply_matches_live_record(ctx.response, record))
        self.assert_preserved(self.other, other_record, original)
        self.main.assert_not_awaited()
        self.client.create.assert_not_awaited()
        execute = AsyncMock(return_value="Offline batch saved")
        stranger = ConversationAddress.group("qq", self.author.space_id, "stranger")
        with patch.object(self.commands, "_execute_confirmed_tool", execute):
            self.assertIsNone(await self.chat.handle_pending_message_core(
                "加入", "qq", stranger.actor_id, stranger,
                history=[], space_key=stranger.space_key, allow_intent_model=False,
            ))
            execute.assert_not_awaited()
            result = await self.chat.handle_pending_message_core(
                "加入", "qq", self.author.actor_id, self.author,
                history=[], space_key=self.author.space_key, allow_intent_model=False,
            )
        self.assertIn("Offline batch saved", result)
        execute.assert_awaited_once()
        self.assertEqual([row["word"] for row in execute.await_args.args[0].args["items"]], ["发薪", "发薪日"])
        self.assertIsNone(self.store.get_record(self.author))

    async def test_mixed_present_and_missing_operands_keep_code_and_normal_selection(self):
        ctx = self.context("积极 发薪")
        self.set_model(operands=("积极", "发薪"))
        await self.assert_finished(ctx)
        self.assertIn("jjuu", ctx.response)
        self.assertIn("1. fsxb", ctx.response)
        self.assertEqual(self.prepared_words(), ["发薪"])
        record = self.store.get_record(self.author)
        self.assertIsInstance(record.state, PendingToolConfirm)
        self.assertEqual(record.state.args["_query_words"], ["发薪"])
        execute = AsyncMock(return_value="Offline addition")
        with patch.object(self.commands, "_execute_confirmed_tool", execute):
            result = await self.chat.handle_pending_message_core(
                "发薪 添加2", "qq", self.author.actor_id, self.author,
                history=[], space_key=self.author.space_key, allow_intent_model=False,
            )
        self.assertIn("Offline addition", result)
        execute.assert_awaited_once()
        self.assertEqual(execute.await_args.args[0].args["items"][0]["code"], "fsxba")
        self.assertIsNone(self.store.get_record(self.author))
        self.main.assert_not_awaited()

    async def test_missing_meaning_operand_keeps_explanation_and_only_operand_candidates(self):
        self.entries.pop("济济")
        ctx = self.context("济济啥意思")
        self.set_model()
        await self.assert_finished(ctx)
        self.assert_main("济济啥意思", "济济", "未在词库")
        self.assertNotIn("候选编码", self.main.await_args.args[4])
        self.assertIn("这是离线解释。", ctx.response)
        self.assertIn("1. jjes", ctx.response)
        self.assertNotIn("「济济啥意思」", ctx.response)
        self.assertEqual(self.prepared_words(), ["济济"])
        self.assertEqual(self.store.get(self.author).word, "济济")
        self.assertTrue(self.chat._advertised_reply_matches_live_record(ctx.response, self.store.get_record(self.author)))
        self.client.create.assert_awaited_once()

    async def test_missing_operands_keep_existing_draft_reminder(self):
        self.pending_items = [{
            "source": "draft", "word": "发薪日", "code": "fxr", "action": "Create",
            "batchId": "offline-draft", "batchStatus": "Draft",
            "batchUrl": "https://offline.invalid/batch/offline-draft",
        }]
        ctx = self.context("发薪 发薪日")
        self.set_model(operands=("发薪", "发薪日"))
        await self.assert_finished(ctx)
        self.assertIn("当前草稿", ctx.response)
        self.assertIn("发薪日", ctx.response)
        self.assertEqual(self.prepared_words(), ["发薪"])
        self.assertEqual([row["word"] for row in self.store.get(self.author).args["items"]], ["发薪"])

    async def test_single_missing_operand_already_in_draft_creates_no_ticket(self):
        self.pending_items = [{
            "source": "draft", "word": "发薪日", "code": "fxr", "action": "Create",
            "batchId": "offline-draft", "batchStatus": "Draft",
            "batchUrl": "https://offline.invalid/batch/offline-draft",
        }]
        ctx = self.context("发薪日")
        self.set_model(operands=("发薪日",))
        await self.assert_finished(ctx)
        self.assertIn("当前草稿", ctx.response)
        self.assertEqual(self.prepared_words(), [])
        self.assertIsNone(self.store.get_record(self.author))

    async def test_multiword_query_uses_only_each_literal_operand_and_preserves_ticket(self):
        record, original, _prompt = self.ticket(self.author)
        ctx = self.context("济济 积极")
        self.set_model(operands=("济济", "积极"))
        await self.assert_finished(ctx)
        self.assertEqual([call.args[1] for call in self.tool.await_args_list], [{"word": "济济"}, {"word": "积极"}])
        self.assertIn("jjes", ctx.response)
        self.assertIn("jjuu", ctx.response)
        self.main.assert_not_awaited()
        self.assert_preserved(self.author, record, original)

    async def test_chat_with_own_live_ticket_and_release_enabled_calls_only_main(self):
        os.environ["KEYTAO_SEMANTIC_ROUTE_OPS"] = "chat,pending_release"
        record, original, _prompt = self.ticket(self.author)
        ctx = self.context("哈哈好的")
        self.set_model("chat", ())
        await self.assert_finished(ctx)
        self.assert_main("哈哈好的")
        self.tool.assert_not_awaited()
        self.assert_preserved(self.author, record, original)

    async def test_help_quoting_other_ticket_reaches_main_without_claiming_it(self):
        record, original, prompt = self.ticket(self.other)
        ctx = self.context("应该怎么问你")
        ctx.reply_reference = harness.ReplyReferenceInfo(
            is_reply=True, is_to_bot=True, mentioned_user_ids=(self.other.actor_id,), text=prompt,
        )
        self.set_model("help", ())
        await self.assert_finished(ctx)
        self.assert_main("应该怎么问你")
        self.tool.assert_not_awaited()
        self.assertIsNone(self.store.get_record(self.author))
        self.assert_preserved(self.other, record, original)

    async def test_real_main_core_disables_tools_and_structural_fallback_in_isolated_state(self):
        record, original, _prompt = self.ticket(self.author)
        ctx = self.context("清空草稿")
        self.set_model("help", ())
        seen = []
        forbidden = AsyncMock(side_effect=AssertionError("A semantic conversation must never execute tools"))
        self.tool.side_effect = forbidden.side_effect

        async def isolated_main(orchestrator, *, message, context, **kwargs):
            seen.append(context)
            self.assertEqual(message, "清空草稿")
            self.assertTrue(context.conversation_only)
            self.assertFalse(context.mutations_allowed)
            self.assertIsNot(orchestrator._state_store, self.store)
            self.assertIsNone(orchestrator._state_store.get_record(self.author))
            sandbox = PendingAddWord("云帆", "yjfj", [("yjfj", False)])
            self.assertTrue(orchestrator._state_store.set(self.author, sandbox))
            orchestrator._state_store.delete(self.author)
            self.assert_preserved(self.author, record, original)
            self.assertEqual(self.chat._structural_draft_management_intent(message).intent, "draft_clear")
            self.assertIsNone(await orchestrator._deterministic_fallback_handler(message, context))
            return "请发送完整的词条请求。"

        with (
            patch.object(self.chat, "get_ai_response_core", self.real_main),
            patch.object(self.chat, "OPENAI_API_KEY", "offline-key"),
            patch.object(self.chat.AgentOrchestrator, "run", isolated_main),
            patch.object(self.chat, "handle_pending_message_core", forbidden),
            patch.object(self.chat, "_try_handle_draft_management_command", forbidden),
            patch.object(self.chat, "_try_handle_encoding_rule_question", forbidden),
            patch.object(self.chat.tool_executor, "call", forbidden),
            patch.object(self.chat.tool_executor, "_invoke_effective_tool", forbidden),
            patch.object(self.chat.tool_executor, "replay_shift_plan", forbidden),
            patch.object(self.chat.tool_executor, "replay_routed_warning", forbidden),
            patch.object(self.chat.memory_store, "get_context_block", return_value=""),
            patch.object(self.chat, "get_group_history_context", return_value=""),
        ):
            await self.assert_finished(ctx)
        self.assertEqual(len(seen), 1)
        self.assertEqual(ctx.response, "请发送完整的词条请求。")
        forbidden.assert_not_awaited()
        self.tool.assert_not_awaited()
        self.assert_preserved(self.author, record, original)

    async def assert_legacy_path(self, ctx):
        self.assertFalse(await self.chat._stage_understand_operation_early(ctx))
        self.assertFalse(getattr(ctx, "semantic_nonwrite_route", ""))
        self.main.assert_not_awaited()
        self.tool.assert_not_awaited()
        self.finish.assert_not_awaited()
        legacy = AsyncMock(return_value="Offline legacy reply")
        with (
            patch.object(self.chat, "_set_conversational_turn"),
            patch.object(self.chat, "_try_handle_explicit_reading_disambiguation", AsyncMock(return_value=None)),
            patch.object(self.chat, "_try_handle_compound_shift_modified_add_command", AsyncMock(return_value=None)),
            patch.object(self.chat, "_try_handle_shift_modified_add_command", AsyncMock(return_value=None)),
            patch.object(self.chat, "_try_handle_complete_add_command", AsyncMock(return_value=None)),
            patch.object(self.chat, "_try_recover_reviewed_add_from_history", AsyncMock(return_value=None)),
            patch.object(self.chat, "_try_handle_simple_single_word_query", legacy),
        ):
            self.assertFalse(await self.chat._stage_handle_simple_word_query(ctx))
        legacy.assert_awaited_once_with(
            ctx.normalized_message_text, "qq", self.author.actor_id,
            self.author, self.author.space_key, "Offline owner", actionable_lookup=False,
        )
        self.assertEqual(ctx.response, "Offline legacy reply")

    async def test_operand_absent_from_message_keeps_legacy_input_and_kwargs(self):
        self.set_model(operands=("济济", "不存在"))
        await self.assert_legacy_path(self.context("济济啥意思"))

    async def test_absent_or_nonexact_tokens_keep_legacy_path(self):
        for token in (None, "", "queries,chats,helps", "QUERY,CHAT,HELP", "query;chat;help"):
            with self.subTest(token=token):
                if token is None:
                    os.environ.pop("KEYTAO_SEMANTIC_ROUTE_OPS", None)
                else:
                    os.environ["KEYTAO_SEMANTIC_ROUTE_OPS"] = token
                self.client.create.reset_mock()
                self.set_model()
                await self.assert_legacy_path(self.context("济济啥意思"))
                self.client.create.assert_not_awaited()

    async def test_each_exact_token_enables_only_its_own_route(self):
        for token, op, operands in (("chat", "query", ("济济",)), ("query", "chat", ()), ("help", "chat", ())):
            with self.subTest(token=token, op=op):
                os.environ["KEYTAO_SEMANTIC_ROUTE_OPS"] = token
                self.set_model(op, operands)
                await self.assert_legacy_path(self.context("济济啥意思"))

    async def test_lookup_failure_stays_read_only_without_candidate_fallback(self):
        self.tool.side_effect = None
        self.tool.return_value = json.dumps({"success": False, "error": "Offline unavailable"})
        ctx = self.context("济济")
        self.set_model()
        await self.assert_finished(ctx)
        self.tool.assert_awaited_once_with("keytao_lookup_by_word", {"word": "济济"}, "qq", self.author.actor_id)
        self.main.assert_not_awaited()
        self.assertIsNone(self.store.get_record(self.author))

    async def test_model_failures_keep_legacy_path(self):
        for scenario in ("timeout", "invalid_json", "invalid_op"):
            with self.subTest(scenario=scenario):
                self.set_model()
                if scenario == "timeout":
                    self.client.create.side_effect = asyncio.TimeoutError()
                elif scenario == "invalid_json":
                    self.client.create.return_value.choices[0].message.content = "not JSON"
                else:
                    self.client.create.return_value.choices[0].message.content = json.dumps({"op": "QUERY"})
                await self.assert_legacy_path(self.context("济济啥意思"))

    async def test_query_requires_at_least_one_operand(self):
        self.set_model(operands=())
        await self.assert_legacy_path(self.context("济济啥意思"))

    async def test_raw_boolean_and_frame_validation_fail_closed(self):
        for field, value in (
            ("refers_to_pending", None), ("refers_to_pending", "false"),
            ("refers_to_pending", 0), ("submit_after", None),
            ("submit_after", "false"), ("frame", "invalid"),
        ):
            with self.subTest(field=field, value=value):
                raw = self.set_model()
                if value is None:
                    raw.pop(field)
                else:
                    raw[field] = value
                self.client.create.return_value.choices[0].message.content = json.dumps(raw, ensure_ascii=False)
                await self.assert_legacy_path(self.context("济济啥意思"))

    async def test_pending_reference_cannot_route_as_an_unrelated_query(self):
        self.set_model(refers_to_pending=True)
        await self.assert_legacy_path(self.context("济济啥意思"))

    async def test_frame_backstop_question_still_routes_query(self):
        raw = self.set_model()
        raw["frame"] = "direct"
        self.client.create.return_value.choices[0].message.content = json.dumps(raw, ensure_ascii=False)
        ctx = self.context("济济怎么打？")
        await self.assert_finished(ctx)
        self.assertIn("frame_backstop:question", ctx.operation_violations)
        self.assert_main("济济怎么打？", "济济", "jjes")

if __name__ == "__main__":
    unittest.main()
