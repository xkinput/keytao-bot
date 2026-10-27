"""Offline checks for affirmative replies bound to an author's live prompt."""

import asyncio
import copy
import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import test_chat_word_discovery as discovery_fixtures
import test_state_machine as harness
from test_semantic_parser import FakeClient, raw_operation

from keytao_bot.harness import word_discovery as discovery
from keytao_bot.harness.conversation import ConversationAddress
from keytao_bot.harness.state import (
    DraftOperationCoordinator, MemoryConversationStateStore, PendingAddWord, PendingToolConfirm,
)
from keytao_bot.plugins import chat_routing
from keytao_bot.utils import community_vote as vote
from eval.routing.replay_prod_cases import DEFAULT_CASES, Replay, load_cases


AFFIRMATIVES = (
    "确认", "同意", "好", "好的", "好滴", "可以", "行", "加入", "加", "嗯", "嗯嗯",
    "对", "是", "是的", "ok", "OK", "没问题", "就这样", "就这个", "确定",
)


class SemanticAdvertisedAnswerTests(unittest.IsolatedAsyncioTestCase):
    def replay(self):
        row = next(row for row in load_cases(DEFAULT_CASES) if row["id"] == "12-scoped-second-candidate")
        return Replay(row, harness, FakeClient)

    def context(self, replay, message, *, quoted=None):
        ctx = replay.chat.TurnContext(
            bot=SimpleNamespace(), event=SimpleNamespace(to_me=False),
            platform="qq", user_id=replay.author.actor_id,
            message_text=message, normalized_message_text=message,
            conv_key=replay.author, space_key=replay.author.space_key,
            memory_context=replay.memory, owner_label="Offline owner",
            command_intent_for=AsyncMock(return_value=replay.chat.MessageCommandIntent()),
        )
        if quoted is not None:
            ctx.reply_reference = harness.ReplyReferenceInfo(is_reply=True, is_to_bot=True, text=quoted)
        return ctx

    async def offer(self, replay, *, advertise_scoped_format=True):
        await replay.seed()
        actual = await replay.turn(replay.row, replay.row["model_raw"])
        self.assertEqual((actual["kind"], actual["candidates"], actual["write_count"]),
                         ("candidates", ["燃面", "晕碳"], 0))
        self.assertIn("燃面 1，晕碳 1", replay.last_response)
        if advertise_scoped_format:
            prompt = replay.last_response + "\n回复「晕碳 添加1」，多选回复「晕碳 添加1、2」。"
            replay.history[-1]["content"] = prompt
            replay.prompt = prompt
            self.assertTrue(replay.store.bind_origin_prompt_digest(
                replay.author, replay.chat._prompt_capability_digest(prompt),
            ))

    async def test_no_live_multiword_add_keeps_original_candidate_offer(self):
        replay = self.replay()
        with ExitStack() as stack:
            for context in replay.patches():
                stack.enter_context(context)
            self.assertIsNone(replay.store.get_record(replay.author))
            await self.offer(replay, advertise_scoped_format=False)
            self.assertEqual(replay.writes, [])
            self.assertEqual([args["word"] for name, args in replay.calls if name == "keytao_prepare_reviewed_add"],
                             ["燃面", "晕碳"])

    async def test_run5_and_run6_scoped_answer_bypass_model_and_add_second_candidate_once(self):
        recorded = (
            {
                "op": "select_candidate", "operands": [], "codes": [], "submit_after": False,
                "refers_to_pending": True, "frame": "direct", "confidence": "low",
                "note": "“晕碳 添加2”疑似在待确认工单中选择第2项（晕碳）进行添加，但表述含糊",
                "relation": "", "_model": "glm-5.3",
            },
            {
                "op": "add", "operands": ["晕碳"], "codes": [], "submit_after": False,
                "refers_to_pending": True, "frame": "direct", "confidence": "low",
                "note": "消息提到pending票中的词“晕碳”，“添加2”可能是选择第2项或添加该词，票类型非候选选择，故按add处理并降低置信度",
                "_model": "glm-5.3",
            },
        )
        for run, raw in zip((5, 6), recorded):
            with self.subTest(run=run), ExitStack() as stack:
                replay = self.replay()
                for context in replay.patches():
                    stack.enter_context(context)
                await self.offer(replay)
                ctx = self.context(replay, "晕碳 添加2")
                parse = stack.enter_context(patch(
                    "keytao_bot.harness.semantic_parser.parse_operation", AsyncMock(return_value=raw),
                ))
                self.assertTrue(await replay.chat._stage_understand_operation_early(ctx))
                parse.assert_not_awaited()
                actual = replay.outcome(ctx)
                self.assertEqual(actual["added"], [["晕碳", "ywtfo"]])
                self.assertEqual(actual["write_count"], 1)
                self.assertEqual(actual["submissions"], 0)
                self.assertIsNone(replay.store.get_record(replay.author))
                await replay.chat._stage_understand_operation_early(ctx)
                self.assertEqual(replay.outcome(ctx)["write_count"], 1)

    async def test_scoped_multi_selection_and_cancel_bypass_model(self):
        for message, expected in (("晕碳 添加1、2", [["晕碳", "ywtf"], ["晕碳", "ywtfo"]]),
                                  ("取消", [])):
            with self.subTest(message=message), ExitStack() as stack:
                replay = self.replay()
                for context in replay.patches():
                    stack.enter_context(context)
                await self.offer(replay)
                parse = stack.enter_context(patch(
                    "keytao_bot.harness.semantic_parser.parse_operation", AsyncMock(return_value=raw_operation()),
                ))
                ctx = self.context(replay, message)
                self.assertTrue(await replay.chat._stage_understand_operation_early(ctx))
                parse.assert_not_awaited()
                self.assertEqual(replay.outcome(ctx)["added"], expected)
                self.assertIsNone(replay.store.get_record(replay.author))

    async def test_scoped_answer_quoting_another_message_preserves_ticket(self):
        replay = self.replay()
        with ExitStack() as stack:
            for context in replay.patches():
                stack.enter_context(context)
            await self.offer(replay)
            record = replay.store.get_record(replay.author)
            original = copy.deepcopy(record)
            stack.enter_context(patch("keytao_bot.harness.semantic_parser.parse_operation", AsyncMock(
                return_value=raw_operation(op="chat", operands=[], codes=[], frame="none"),
            )))
            ctx = self.context(replay, "晕碳 添加2", quoted="Another bot message")
            await replay.chat._stage_understand_operation_early(ctx)
            self.assertIs(replay.store.get_record(replay.author), record)
            self.assertEqual(record, original)
            self.assertEqual(replay.writes, [])


class SemanticAssentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.chat = harness.openai_chat_module
        self.commands = harness.chat_commands_module
        self.store = MemoryConversationStateStore()
        self.memory = harness.ChatMemoryContext(
            platform="qq", user_id="assent-owner", space_type="group",
            space_id="assent-group", speaker_name="Offline owner",
        )
        self.author = self.memory.conversation_address
        self.client = FakeClient(json.dumps(raw_operation(
            op="chat", operands=[], codes=[], frame="none", refers_to_pending=False,
        )))
        self.finish = AsyncMock()
        self.add = AsyncMock(return_value="offline addition")
        self.submit = AsyncMock(return_value="offline addition and submission")
        self.tool = AsyncMock(side_effect=AssertionError("Unexpected offline tool call"))
        self.history = []
        patches = (
            patch.dict(os.environ, {
                "KEYTAO_SEMANTIC_ROUTE_OPS": "pending_release,pending_answer",
                "KEYTAO_SEMANTIC_SHADOW": "0",
            }),
            patch.object(self.chat, "AsyncOpenAI", return_value=self.client),
            patch.object(self.chat, "WORD_QUERY_INTENT_MODEL", "offline-model"),
            patch.object(self.chat, "conversation_state_store", self.store),
            patch.object(self.chat, "draft_operation_coordinator", DraftOperationCoordinator()),
            patch.object(self.chat, "_execute_add_to_draft", self.add),
            patch.object(self.chat, "_execute_add_to_draft_and_submit", self.submit),
            patch.object(self.chat, "call_tool_function", self.tool),
            patch.object(self.chat, "_finish_ai_chat_response", self.finish),
            patch.object(self.chat, "remember_conversation"),
            patch.object(self.chat, "get_history", side_effect=lambda *args: self.history),
            patch.object(self.chat, "to_me", return_value=AsyncMock(
                side_effect=lambda bot, event, state: bool(getattr(event, "to_me", False)),
            )),
            patch.object(self.chat.logger, "info"),
            patch.object(self.chat.logger, "warning"),
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
        self.tool.assert_not_awaited()

    def ticket(self, *, recommended="wj", legacy=False):
        self.add.reset_mock()
        self.submit.reset_mock()
        self.client.create.reset_mock()
        self.assertTrue(self.store.set(
            self.author,
            PendingAddWord("午时", recommended, [("wj", False), ("wja", False)],
                           server_candidates=[("wj", False), ("wja", False)]),
            space_key=self.author.space_key, owner_label="Offline owner",
        ))
        record = self.store.get_record(self.author)
        prompt = self.chat._render_live_single_candidate_record(record)
        self.assertIn("加入", prompt)
        if not legacy:
            self.assertTrue(self.store.bind_origin_prompt_digest(
                self.author, self.chat._prompt_capability_digest(prompt),
            ))
        self.history = [{"role": "assistant", "content": prompt}]
        return record, prompt

    def context(self, message, *, quoted=None, addressed=False, actor=None,
                platform=None, space_key=None):
        actor = actor or self.author
        ctx = self.chat.TurnContext(
            bot=SimpleNamespace(), event=SimpleNamespace(to_me=addressed),
            platform=platform or actor.platform, user_id=actor.actor_id,
            normalized_message_text=self.chat._strip_command_message_prefixes(message),
            message_text=message, conv_key=actor,
            space_key=space_key or actor.space_key, memory_context=self.memory,
            owner_label="Offline owner", command_intent_for=AsyncMock(
                return_value=self.chat.MessageCommandIntent(),
            ),
        )
        if quoted is not None:
            ctx.reply_reference = harness.ReplyReferenceInfo(
                is_reply=True, is_to_bot=True, text=quoted,
            )
        return ctx

    def assert_untouched(self, record, original):
        self.assertIs(self.store.get_record(self.author), record)
        self.assertEqual(record, original)
        self.add.assert_not_awaited()
        self.submit.assert_not_awaited()

    async def assert_added_once(self, ctx, *, code="wj", response="offline addition"):
        self.assertTrue(await self.chat._stage_understand_operation_early(ctx), ctx.message_text)
        self.assertEqual(ctx.response, response)
        self.add.assert_awaited_once()
        self.assertEqual(self.add.await_args.args[:2], ("午时", code))
        self.submit.assert_not_awaited()
        self.client.create.assert_not_awaited()
        self.assertIsNone(self.store.get_record(self.author))
        await self.chat._stage_understand_operation_early(ctx)
        self.add.assert_awaited_once()

    def test_shared_affirmative_set_matches_spec(self):
        self.assertEqual(set(chat_routing.AFFIRMATIVE_REPLY_TEXTS), set(AFFIRMATIVES))

    async def test_every_affirmative_quotes_own_prompt_or_addresses_latest(self):
        for message in AFFIRMATIVES:
            for binding in ("quote", "addressed"):
                with self.subTest(message=message, binding=binding):
                    _record, prompt = self.ticket()
                    ctx = self.context(message, quoted=prompt if binding == "quote" else None,
                                       addressed=binding == "addressed")
                    await self.assert_added_once(ctx)

    async def test_bot_name_mentions_and_trailing_decorations_are_affirmative(self):
        for message in ("@喵喵 确认", "喵喵同意", "@喵喵 好滴～~!！。😂", " 好 的 ！ "):
            with self.subTest(message=message):
                _record, prompt = self.ticket()
                await self.assert_added_once(self.context(message, quoted=prompt))
        for message in ("@喵喵 确认", "喵喵 同意"):
            with self.subTest(addressed_by_name=message):
                self.ticket()
                await self.assert_added_once(self.context(message))

    async def test_assent_uses_recommended_code_without_implicit_submission(self):
        _record, prompt = self.ticket(recommended="wja")
        await self.assert_added_once(self.context("确认", quoted=prompt), code="wja")

    async def test_listed_index_code_and_add_ordinal_bypass_model(self):
        for message, code in (("2", "wja"), ("wja", "wja"), ("添加2", "wja")):
            with self.subTest(message=message):
                _record, prompt = self.ticket()
                await self.assert_added_once(
                    self.context(message, quoted=prompt), code=code,
                    response="offline addition\n未选择：1. 午时 → wj；这些候选本次未添加。",
                )

    async def test_multiword_assent_keeps_only_each_advertised_default(self):
        defaults = (("午时", "wj", "wja"), ("务实", "we", "wea"))
        scopes = []
        for word, short, recommended in defaults:
            candidates = [(short, False), (recommended, False)]
            reviewed = PendingAddWord(word, recommended, candidates, server_candidates=candidates)
            scopes.append({
                "word": word, "candidates": candidates, "occupiedWords": {},
                "orderingAssessments": [], "reviewedState": self.commands._pending_add_word_payload(reviewed),
            })
        state = PendingToolConfirm("keytao_batch_add_to_draft", {
            "items": [{"action": "Create", "word": word, "code": code} for word, _short, code in defaults],
            "_candidate_scopes": scopes, "_reviewed_multi_word": True,
            "_query_words": [word for word, _short, _code in defaults],
        })
        self.assertTrue(self.store.set(self.author, state))
        prompt = "午时 wja，务实 wea。回复「加入」写入草稿，或「加入并提交」。"
        self.assertTrue(self.store.bind_origin_prompt_digest(self.author, self.chat._prompt_capability_digest(prompt)))
        self.history = [{"role": "assistant", "content": prompt}]
        execute = AsyncMock(return_value="offline multiword addition")
        with patch.object(self.commands, "_execute_confirmed_tool", execute):
            self.assertTrue(await self.chat._stage_understand_operation_early(self.context("同意", quoted=prompt)))
        execute.assert_awaited_once()
        self.assertEqual([(item["word"], item["code"]) for item in execute.await_args.args[0].args["items"]],
                         [(word, code) for word, _short, code in defaults])
        self.assertFalse(execute.await_args.kwargs.get("submit_after", False))
        self.client.create.assert_not_awaited()
        self.submit.assert_not_awaited()

    async def test_choice_offer_assent_preserves_choice_without_guessing(self):
        state = PendingToolConfirm("keytao_choice_offer", {"options": [
            {"label": "A", "description": "Adjust the draft", "plan": None},
            {"label": "B", "description": "Keep the draft", "plan": None},
        ]})
        self.assertTrue(self.store.set(self.author, state))
        record = self.store.get_record(self.author)
        prompt = self.commands.render_pending_choice_offer(state)
        self.assertTrue(self.store.bind_origin_prompt_digest(self.author, self.chat._prompt_capability_digest(prompt)))
        self.history = [{"role": "assistant", "content": prompt}]
        ctx = self.context("同意", quoted=prompt)
        self.assertTrue(await self.chat._stage_understand_operation_early(ctx))
        self.assertEqual(ctx.response, prompt)
        self.assertIs(self.store.get_record(self.author), record)
        self.client.create.assert_not_awaited()
        self.add.assert_not_awaited()
        self.submit.assert_not_awaited()

    async def test_assent_overrides_chat_route_and_works_with_semantic_flags_off(self):
        for route_ops in ("pending_release,pending_answer,chat", ""):
            with self.subTest(route_ops=route_ops):
                _record, prompt = self.ticket()
                with patch.dict(os.environ, {"KEYTAO_SEMANTIC_ROUTE_OPS": route_ops}):
                    await self.assert_added_once(self.context("同意", quoted=prompt))

    async def test_matching_quote_can_bind_older_prompt(self):
        _record, prompt = self.ticket()
        self.history.append({"role": "assistant", "content": "A newer unrelated response"})
        await self.assert_added_once(self.context("好", quoted=prompt))

    async def test_digestless_legacy_quote_uses_existing_latest_binding(self):
        _record, prompt = self.ticket(legacy=True)
        await self.assert_added_once(self.context("同意", quoted=prompt))

    async def test_unbound_wrong_quote_and_nonlatest_addressed_messages_do_not_assent(self):
        for scenario in ("unaddressed", "wrong_quote", "quote_other_owner", "nonlatest"):
            with self.subTest(scenario=scenario):
                record, _prompt = self.ticket()
                if scenario == "nonlatest":
                    self.history.append({"role": "assistant", "content": "A newer unrelated response"})
                original = copy.deepcopy(record)
                ctx = self.context("好", quoted="A different bot message" if scenario == "wrong_quote" else None,
                                   addressed=scenario != "unaddressed")
                if scenario == "quote_other_owner":
                    ctx.reply_reference = harness.ReplyReferenceInfo(
                        is_reply=True, is_to_bot=True, text=_prompt,
                        mentioned_user_ids=("another-owner",),
                    )
                self.assertFalse(await self.chat._stage_understand_operation_early(ctx))
                self.assert_untouched(record, original)

    async def test_other_owner_foreign_platform_space_and_executing_ticket_do_not_assent(self):
        for scenario in ("other_owner", "platform", "space", "executing"):
            with self.subTest(scenario=scenario):
                record, prompt = self.ticket()
                actor = self.author
                options = {}
                if scenario == "other_owner":
                    actor = ConversationAddress.group("qq", "assent-group", "another-owner")
                elif scenario == "platform":
                    options["platform"] = "telegram"
                elif scenario == "space":
                    actor = ConversationAddress.group("qq", "another-group", self.author.actor_id)
                else:
                    self.assertTrue(self.store.begin_execution(record))
                original = copy.deepcopy(record)
                ctx = self.context("好", quoted=prompt, addressed=True, actor=actor, **options)
                if actor != self.author:
                    ctx.memory_context = harness.ChatMemoryContext(
                        platform=actor.platform, user_id=actor.actor_id,
                        space_type="group", space_id=actor.space_id, speaker_name="Offline owner",
                    )
                with patch.object(self.chat, "get_history", side_effect=lambda key: (
                    self.history if key == self.author else []
                )):
                    handled = await self.chat._stage_understand_operation_early(ctx)
                self.assertEqual(handled, scenario in {"other_owner", "space"})
                if handled:
                    self.assertEqual(ctx.response, "这条提议不是发给你的，本次未执行。")
                    self.client.create.assert_not_awaited()
                self.tool.assert_not_awaited()
                self.assert_untouched(record, original)

    async def test_non_affirmative_extra_content_never_uses_bound_assent(self):
        for message in ("好的呀哈哈", "好，再改个编码", "同意吗？", "不行", "ok next", "确认提交"):
            with self.subTest(message=message):
                record, prompt = self.ticket()
                original = copy.deepcopy(record)
                self.assertFalse(await self.chat._stage_understand_operation_early(
                    self.context(message, quoted=prompt),
                ))
                self.assert_untouched(record, original)

    async def test_vote_receipt_is_consumed_during_initialization_before_ticket_assent(self):
        record, _prompt = self.ticket()
        original = copy.deepcopy(record)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"KEYTAO_COMMUNITY_VOTE": "1", "KEYTAO_WORD_DISCOVERY": "1"}),
        ):
            votes = vote.CommunityVoteStore(Path(directory) / "votes.db")
            batch_id = "bc32871a-1111-2222-3333-444444444444"
            submitted = {
                "success": True, "autoApproved": False, "batchId": batch_id,
                "batchUrl": f"https://keytao.rea.ink/batch/{batch_id}", "contentVersion": 1,
                "snapshotItems": [{"id": 1, "action": "Create", "word": "开团", "code": "khtt", "type": "Phrase"}],
            }
            self.assertTrue(votes.bind_receipt(submitted, "qq", "another-owner", "assent-group", "vote-receipt"))
            receipt = vote.submitted_vote_receipt(submitted)
            self.assertTrue(receipt)
            offers = discovery.DiscoveryStore(Path(directory) / "offers.db")
            self.assertTrue(offers.claim(self.author, "午时"))
            offers.bind_offer(self.author, "午时", record.nonce, "offer-receipt")
            worker = discovery.WordDiscovery(offers)
            event = discovery_fixtures.FakeEvent(
                "同意", actor=self.author.actor_id, group=self.author.space_id,
            )
            event.reply = SimpleNamespace(message_id="vote-receipt")
            bot = SimpleNamespace(self_id="999", send=AsyncMock())
            approve = AsyncMock()
            ctx = self.context("同意", quoted=receipt)
            with (
                discovery_fixtures.qq_runtime(self.chat, bot) as qq_bot,
                patch.object(discovery, "_discovery", worker),
                patch.object(vote, "get_store", return_value=votes),
                patch.object(vote, "find_user_by_platform", AsyncMock(
                    side_effect=lambda platform, actor: {"id": 1 if actor == self.author.actor_id else 2},
                )),
                patch.object(self.chat, "extract_memory_context", AsyncMock(return_value=self.memory)),
                patch.object(self.chat, "extract_onebot_reply_id", return_value="vote-receipt"),
                patch.object(self.chat.skills_manager, "get_tool_function", return_value=approve),
            ):
                self.assertTrue(await self.chat.should_handle(qq_bot, event))
                self.assert_untouched(record, original)
                self.assertTrue(offers.is_offer(self.author, record.nonce, "offer-receipt"))
                ctx.bot, ctx.event = qq_bot, event
                self.assertTrue(await self.chat._stage_initialize_conversation(ctx))
            self.assertIn("当前投票：1/3", ctx.response)
            self.assertEqual(len(votes.get(batch_id)["voters"]), 1)
            approve.assert_not_awaited()
            self.client.create.assert_not_awaited()
        self.assert_untouched(record, original)

    async def test_discovery_offers_admit_every_bound_affirmative(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, message in enumerate(AFFIRMATIVES):
                for binding in ("quote", "addressed"):
                    with self.subTest(message=message, binding=binding):
                        record, prompt = self.ticket()
                        offers = discovery.DiscoveryStore(Path(directory) / f"{index}-{binding}.db")
                        self.assertTrue(offers.claim(self.author, "午时"))
                        offers.bind_offer(self.author, "午时", record.nonce, "offer-receipt")
                        worker = discovery.WordDiscovery(offers)
                        event = discovery_fixtures.FakeEvent(
                            message, actor=self.author.actor_id, group=self.author.space_id,
                        )
                        event.to_me = binding == "addressed"
                        if binding == "quote":
                            event.reply = SimpleNamespace(message_id="offer-receipt")
                        with (
                            patch.dict(os.environ, {"KEYTAO_WORD_DISCOVERY": "1"}),
                            patch.object(discovery, "_discovery", worker),
                        ):
                            self.assertTrue(await discovery.discovery_reply_route(SimpleNamespace(), event, self.chat))
                        await self.assert_added_once(self.context(
                            message, quoted=prompt if binding == "quote" else None,
                            addressed=binding == "addressed",
                        ))

    async def test_discovery_binding_rejects_other_quote_owner_and_stale_address(self):
        with tempfile.TemporaryDirectory() as directory:
            for scenario in ("other_quote", "other_owner", "nonlatest", "executing", "older_own_quote"):
                with self.subTest(scenario=scenario):
                    record, _prompt = self.ticket()
                    offers = discovery.DiscoveryStore(Path(directory) / f"{scenario}.db")
                    self.assertTrue(offers.claim(self.author, "午时"))
                    offers.bind_offer(self.author, "午时", record.nonce, "offer-receipt")
                    event = discovery_fixtures.FakeEvent(
                        "好", actor="another-owner" if scenario == "other_owner" else self.author.actor_id,
                        group=self.author.space_id,
                    )
                    event.to_me = True
                    if scenario != "nonlatest":
                        event.reply = SimpleNamespace(
                            message_id="unrelated" if scenario == "other_quote" else "offer-receipt",
                        )
                    if scenario in {"nonlatest", "older_own_quote"}:
                        self.history.append({"role": "assistant", "content": "A newer unrelated response"})
                    if scenario == "executing":
                        self.assertTrue(self.store.begin_execution(record))
                    original = copy.deepcopy(record)
                    with (
                        patch.dict(os.environ, {"KEYTAO_WORD_DISCOVERY": "1"}),
                        patch.object(discovery, "_discovery", discovery.WordDiscovery(offers)),
                    ):
                        admitted = await discovery.discovery_reply_route(SimpleNamespace(), event, self.chat)
                    self.assertEqual(admitted, None if scenario == "other_owner" else scenario == "older_own_quote")
                    self.assert_untouched(record, original)


if __name__ == "__main__":
    unittest.main()
