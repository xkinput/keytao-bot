"""Offline B16 regressions for give-way review lines and submit receipts."""

import asyncio
import json
import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import test_b15_sibling_and_others as b15
from keytao_bot.harness.orchestrator import (
    AgentOrchestrator, AgentRequestContext, AgentRuntimeConfig,
)
from keytao_bot.plugins.chat_commands import call_tool_function
from keytao_bot.utils.pending_confirmation import (
    pending_confirmation_copy, render_server_backed_single_word_lookup,
)


FIRST_WORD_BASELINE = (
    "2 个词的候选：\n1. 「球坐标系」\n"
    "词库暂无收录「球坐标系」：\n"
    "审词：读音 qiu zuo biao xi；来源 offline；\n"
    "自动审核：Offline reviewed fixture，可自动通过\n"
    "候选编码:\n1. qzbx — ✅ 推荐（空位）\n"
    "2. qzbxv — 空位\n3. qzbxvi — 空位\n推荐编码：qzbx\n"
)
FOOTER_BASELINE = (
    "回复「加入」写入草稿，或回复「加入并提交」写入并提交。或「取消」。\n"
    "按词选择（词名加空格，再写编号或编码；多词用逗号分隔）：\n"
    "- 「球坐标系 1，柱坐标系 2」\n"
    "可只写其中部分词；选择后即加入草稿，提交需明确回复提交指令。\n"
    "也可选择后直接加入并提交：\n"
    "- 「球坐标系 1，柱坐标系 2，加入并提交」"
)


class B16ItemOneTests(unittest.IsolatedAsyncioTestCase):
    setUp = b15.B15ChatTests.setUp
    turn = b15.B15ChatTests.turn
    candidates = b15.B15ChatTests.candidates
    created = b15.B15ChatTests.created

    async def test_f1_give_way_keeps_review_lines_without_recommendation_arrow(self):
        prompt = await self.candidates()
        first, separator, second = prompt.partition("2. 「柱坐标系」\n")
        self.assertTrue(separator)
        self.assertEqual(first.encode(), FIRST_WORD_BASELINE.encode())
        self.assertTrue(second.endswith(FOOTER_BASELINE), second)
        candidate_list = (
            "「柱坐标系」候选编码：\n"
            "1. qzbx — 本次推荐给「球坐标系」\n"
            "2. qzbxv — ✅ 推荐（空位）\n3. qzbxvi — 空位"
        )
        self.assertIn(candidate_list, second)
        self.assertIn("推荐编码：qzbxv", second)
        with self.subTest(part="review_prefix"):
            self.assertTrue(second.startswith(
                "词库暂无收录「柱坐标系」：\n"
                "审词：读音 zhu zuo biao xi；来源 offline；\n"
                "自动审核：Offline reviewed fixture，可自动通过\n"
                + candidate_list
            ), second)
        with self.subTest(part="recommendation_arrow"):
            self.assertNotIn("• 「柱坐标系」→ qzbxv（推荐）", second)
        record = self.replay.store.get_record(self.replay.author)
        self.assertEqual(
            [(item["word"], item["code"]) for item in record.state.args["items"]],
            [(b15.WORDS[0], "qzbx"), (b15.WORDS[1], "qzbxv")],
        )
        self.assertEqual(record.state.args["_candidate_scopes"][1]["siblingRecommendations"],
                         {"qzbx": b15.WORDS[0]})
        await self.turn("加入", prompt)
        self.assertEqual(self.created(), [(b15.WORDS[0], "qzbx"), (b15.WORDS[1], "qzbxv")])
        self.assertIsNone(self.replay.store.get_record(self.replay.author))
        self.general.assert_not_awaited()
        self.replay.client.create.assert_not_awaited()

    async def test_manual_give_way_shows_reason_and_writes_manual_review_flag(self):
        reason = "「柱坐标系」@qzbx 证据不足"
        expected_reason = "「柱坐标系」@qzbxv 证据不足"
        original_tool = self.replay.tool

        async def tool(name, args, context):
            result = await original_tool(name, args, context.platform, context.user_id)
            if name == "keytao_prepare_reviewed_add" and args["word"] == b15.WORDS[1]:
                review = json.loads(result)
                review.update(needsManualReview=True, manualReviewReason=reason,
                              reviewDisposition="SEAL")
                review["preSubmitAudit"].update(autoApprove=False, verdict="manual",
                                                summary=reason, issues=[reason])
                result = json.dumps(review, ensure_ascii=False)
            return result

        with (
            patch.object(self.replay.chat, "call_tool_function", call_tool_function),
            patch.object(self.replay.commands.tool_executor, "call", AsyncMock(side_effect=tool)),
            patch.object(self.replay.commands, "_reviewed_add_verdicts", OrderedDict()),
        ):
            prompt = await self.candidates()
            record = self.replay.store.get_record(self.replay.author)
            pending_item = record.state.args["items"][1]
            self.assertEqual(pending_item["code"], "qzbxv")
            self.assertIs(pending_item["needsManualReview"], True)
            self.assertEqual(pending_item["manualReviewReason"], expected_reason)
            with self.subTest(part="visible_reason"):
                self.assertIn(f"自动审核：{expected_reason}，需要管理员审核", prompt)
            await self.turn("加入", prompt)
        written_item = next(item for item in self.replay.draft if item["word"] == b15.WORDS[1])
        self.assertEqual(written_item, pending_item)
        self.assertIs(written_item["needsManualReview"], True)
        self.assertEqual(written_item["manualReviewReason"], expected_reason)
        self.assertEqual(self.created(), [(b15.WORDS[0], "qzbx"), (b15.WORDS[1], "qzbxv")])
        self.general.assert_not_awaited()
        self.replay.client.create.assert_not_awaited()

    async def test_give_way_rewrites_only_exact_code_tokens_and_keeps_review_line(self):
        review = json.loads(await self.replay.tool(
            "keytao_prepare_reviewed_add", {"word": b15.WORDS[1]},
            "qq", self.replay.author.actor_id,
        ))
        for code in ("qzbx", "QZBX", "qzbxv", "qzbxvi", "qzbx_extra", "qzbx1"):
            for sealed in (False, True):
                with self.subTest(code=code, sealed=sealed):
                    reason = f"「柱坐标系」@{code} 证据不足"
                    review.update(needsManualReview=sealed, manualReviewReason=reason)
                    review["preSubmitAudit"].update(autoApprove=False, issues=[reason])
                    reviewed_prompt = self.replay.commands._format_reviewed_add_prompt(review)
                    rendered = render_server_backed_single_word_lookup(
                        b15.WORDS[1], "qzbxv", [(code, False) for code in b15.CODES], {},
                        reviewed_prompt=reviewed_prompt,
                        sibling_recommendations={"qzbx": b15.WORDS[0]},
                        actionable_controls=True, include_controls=False,
                    )
                    audit_line = f"自动审核：{reason}，需要管理员审核"
                    self.assertIn(audit_line, reviewed_prompt)
                    self.assertIn("词库暂无收录「柱坐标系」：", rendered)
                    self.assertIn("审词：读音 zhu zuo biao xi；来源 offline；", rendered)
                    expected_code = "qzbxv" if code.lower() == "qzbx" else code
                    self.assertIn(f"自动审核：「柱坐标系」@{expected_code} 证据不足，需要管理员审核", rendered)
                    self.assertIn("推荐编码：qzbxv", rendered)
                    self.assertNotIn("• 「柱坐标系」→ qzbxv（推荐）", rendered)

    def test_give_way_preserves_plain_syllable_and_longer_code_tokens(self):
        rendered = render_server_backed_single_word_lookup(
            "安", "anv", [("an", False), ("anv", False)], {},
            reviewed_prompt=(
                "词库暂无收录「安」：\n审词：读音 an；来源 offline；\n"
                "自动审核：@an 证据不足；另见 @anv、@anvi、@an_extra、@an1\n"
                "候选编码:\n1. an — 空位\n2. anv — 空位"
            ),
            sibling_recommendations={"an": "鞍"},
            actionable_controls=True, include_controls=False,
        )
        self.assertIn("审词：读音 an；来源 offline；", rendered)
        self.assertIn("自动审核：@anv 证据不足；另见 @anv、@anvi、@an_extra、@an1", rendered)

    async def test_polyphonic_give_way_has_no_stale_reading_heading(self):
        review = json.loads(await self.replay.tool(
            "keytao_prepare_reviewed_add", {"word": b15.WORDS[1]},
            "qq", self.replay.author.actor_id,
        ))
        review["pronunciations"].append({
            **review["pronunciations"][0], "pinyin": "chu zuo biao xi",
            "candidateStatuses": [{"code": "czbx", "occupied": False, "words": [], "phrases": []}],
        })
        reviewed_prompt = self.replay.commands._format_reviewed_add_prompt(review)
        self.assertIn("候选编码（读音 1）:", reviewed_prompt)
        rendered = render_server_backed_single_word_lookup(
            b15.WORDS[1], "qzbxv", [(code, False) for code in [*b15.CODES, "czbx"]], {},
            reviewed_prompt=reviewed_prompt, sibling_recommendations={"qzbx": b15.WORDS[0]},
            actionable_controls=True, include_controls=False,
        )
        self.assertNotIn("候选编码（读音", rendered)
        self.assertIn("1. qzbx — 本次推荐给「球坐标系」", rendered)
        self.assertIn("4. czbx — 空位", rendered)


class B16ItemTwoTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        b15.B15ChatTests.setUp(self)
        original_tool = self.replay.tool
        self.write_previews = []

        async def tool(name, args, context):
            result = await original_tool(name, args, context.platform, context.user_id)
            if name in {"keytao_create_phrase", "keytao_batch_add_to_draft"}:
                data = json.loads(result)
                if not data.get("pendingDuplicateConfirmation"):
                    words = {item["word"] for item in args.get("items", [args])}
                    data["pendingItems"] = [item for item in self.replay.pending_items if item["word"] in words]
                    if data.get("requiresConfirmation") is True:
                        self.write_previews.append(data)
                    result = json.dumps(data, ensure_ascii=False)
            return result

        self.enterContext(patch.object(self.replay.chat, "call_tool_function", call_tool_function))
        self.enterContext(patch.object(self.replay.commands.tool_executor, "call", AsyncMock(side_effect=tool)))
        self.enterContext(patch.object(self.replay.commands, "_reviewed_add_verdicts", OrderedDict()))

    turn = b15.B15ChatTests.turn
    candidates = b15.B15ChatTests.candidates
    created = b15.B15ChatTests.created
    submissions = b15.B15ChatTests.submissions
    reset_ticket = b15.B15ChatTests.reset_ticket

    async def test_shown_overlap_preview_receipt_omits_notice(self):
        item = b15.pending_item()
        self.replay.pending_items = [item]
        for words, answer in (
            ((b15.WORDS[0],), "加入"),
            ((b15.WORDS[0],), "加入并提交"),
            (b15.WORDS, "加入并提交"),
            (b15.WORDS, "球坐标系 1，柱坐标系 2，加入并提交"),
        ):
            with self.subTest(words=words, answer=answer):
                self.reset_ticket()
                self.write_previews.clear()
                prompt = await self.candidates(words)
                self.assertIn(b15.notice(item), prompt)
                await self.turn(answer, prompt)
                self.assertTrue(self.write_previews)
                for preview in self.write_previews:
                    self.assertIs(preview["success"], False)
                    self.assertIs(preview["requiresConfirmation"], True)
                    self.assertIn(item, preview["pendingItems"])
                expected = [(b15.WORDS[0], "qzbx")]
                if len(words) == 2:
                    expected.append((b15.WORDS[1], "qzbxv"))
                self.assertEqual(self.created(), expected, self.replay.last_response)
                self.assertEqual(self.submissions(), int("提交" in answer), self.replay.last_response)
                self.assertIsNone(self.replay.store.get_record(self.replay.author))
                self.assertNotIn(b15.notice(item), self.replay.last_response)
                self.general.assert_not_awaited()
                self.replay.client.create.assert_not_awaited()

    async def test_shown_same_word_and_code_still_requires_confirmation_without_write(self):
        item = b15.pending_item(code="qzbx")
        self.replay.pending_items = [item]
        for words, answer in (
            ((b15.WORDS[0],), "加入"),
            ((b15.WORDS[0],), "加入并提交"),
        ):
            with self.subTest(words=words, answer=answer):
                self.reset_ticket()
                prompt = await self.candidates(words)
                self.assertIn(b15.notice(item), prompt)
                await self.turn(answer, prompt)
                self.assertIn(b15.notice(item), self.replay.last_response)
                self.assertIn("确认", self.replay.last_response)
                record = self.replay.store.get_record(self.replay.author)
                self.assertIsNotNone(record)
                self.assertTrue(b15.pending_duplicate_confirmation_is_complete(record.state))
                self.assertEqual(record.state.args["_pending_display"]["pendingItems"], [item])
                self.assertEqual(self.replay.writes, [])
                self.general.assert_not_awaited()
                self.replay.client.create.assert_not_awaited()

    async def test_shown_overlap_add_and_submit_receipt_omits_notice(self):
        item = b15.pending_item()
        for words, answer in (
            ((b15.WORDS[0],), "加入并提交"),
            (b15.WORDS, "加入并提交"),
            (b15.WORDS, "球坐标系 1，柱坐标系 2，加入并提交"),
            ((b15.WORDS[0],), "加入"),
            (b15.WORDS, "加入"),
        ):
            with self.subTest(words=words, answer=answer):
                self.reset_ticket()
                self.replay.pending_items = []
                prompt = await self.candidates(words)
                await self.turn(answer, prompt)
                baseline_receipt = self.replay.last_response

                self.reset_ticket()
                self.replay.pending_items = [item]
                prompt = await self.candidates(words)
                self.assertIn(b15.notice(item), prompt)
                await self.turn(answer, prompt)
                self.assertEqual(self.submissions(), int("提交" in answer), self.replay.last_response)
                self.assertIsNone(self.replay.store.get_record(self.replay.author))
                self.assertNotIn(b15.notice(item), self.replay.last_response)
                self.assertEqual(self.replay.last_response, baseline_receipt)
                self.general.assert_not_awaited()
                self.replay.client.create.assert_not_awaited()

    async def test_first_plain_add_keeps_unseen_notice_once(self):
        item = b15.pending_item()
        self.replay.pending_items = [item]
        self.assertIsNone(self.replay.store.get_record(self.replay.author))
        await self.turn("加词 球坐标系 qzbx")
        self.assertEqual(self.created(), [(b15.WORDS[0], "qzbx")], self.replay.last_response)
        self.assertEqual(self.replay.last_response.count(b15.notice(item)), 1, self.replay.last_response)
        self.general.assert_not_awaited()

    async def test_unseen_overlap_prompts_then_confirm_receipt_omits_notice(self):
        draft = [{"action": "Create", "word": b15.WORDS[0], "code": "qzbx", "type": "Phrase"}]
        self.replay.draft = list(draft)
        item = b15.pending_item()
        self.replay.pending_items = [item]
        await self.turn("提交")
        prompt = self.replay.last_response
        self.assertEqual(self.submissions(), 0, prompt)
        self.assertEqual(prompt.count(b15.notice(item)), 1, prompt)
        record = self.replay.store.get_record(self.replay.author)
        self.assertIsNotNone(record)
        self.assertEqual(record.state.function_name, "keytao_submit_batch")
        self.replay.client.create.reset_mock()
        await self.turn("确认", prompt)
        self.assertEqual(self.submissions(), 1, self.replay.last_response)
        self.assertIsNone(self.replay.store.get_record(self.replay.author))
        self.assertNotIn(b15.notice(item), self.replay.last_response)
        self.assertEqual(self.replay.last_response, (
            "✅ 草稿已提交审核。\n草稿地址：https://offline.invalid/draft/offline-batch\n"
            "✅ 批次已提交审核。\n草稿地址：https://keytao.rea.ink/batch/offline-batch"
        ))
        self.general.assert_not_awaited()
        self.replay.client.create.assert_not_awaited()


class B16ItemThreeTests(unittest.IsolatedAsyncioTestCase):
    setUp = b15.B15ChatTests.setUp
    turn = b15.B15ChatTests.turn
    candidates = b15.B15ChatTests.candidates
    submissions = b15.B15ChatTests.submissions
    reset_ticket = b15.B15ChatTests.reset_ticket

    async def model_submit(self):
        tool_call = SimpleNamespace(
            id="offline-submit", type="function",
            function=SimpleNamespace(name="keytao_submit_batch", arguments="{}"),
        )
        self.model = b15.harness._FakeClient([
            b15.harness._FakeAIResponse("tool_calls", "", [tool_call]),
            b15.harness._FakeAIResponse("stop", pending_confirmation_copy()),
        ])
        receipt_recorder = AsyncMock()
        orchestrator = AgentOrchestrator(
            client_factory=lambda: self.model,
            runtime=AgentRuntimeConfig("offline-model", 1000, 0.0, 10.0),
            skills_manager=SimpleNamespace(
                get_skill_instructions=lambda: "",
                has_tools=lambda: True,
                get_tools=lambda: [{"type": "function", "function": {
                    "name": "keytao_submit_batch", "description": "Submit draft",
                    "parameters": {"type": "object", "properties": {}},
                }}],
            ),
            tool_executor=self.replay.commands.tool_executor,
            state_store=self.replay.store,
            bind_help_text="offline", system_prompt_core="offline",
            tool_receipt_recorder=receipt_recorder,
        )
        token = self.replay.chat.current_memory_context.set(self.replay.memory)
        try:
            prompt = await orchestrator.run(
                "提交当前草稿",
                AgentRequestContext(
                    platform="qq", user_id=self.replay.author.actor_id,
                    space_type="group", space_id=self.replay.memory.space_id,
                    speaker_name="Offline owner", mutations_allowed=True,
                ),
            )
        finally:
            self.replay.chat.current_memory_context.reset(token)
        self.replay.history[:] = [{"role": "assistant", "content": prompt}]
        self.assertEqual(self.submissions(), 0, prompt)
        record = self.replay.store.get_record(self.replay.author)
        self.assertIsNotNone(record, prompt)
        self.assertEqual(record.state.function_name, "keytao_submit_batch")
        self.assertEqual(len(self.model.completions.calls), 2)
        model_messages = self.model.completions.calls[-1]["messages"]
        result = json.loads(next(message["content"] for message in model_messages
                                 if message["role"] == "tool"))
        self.assertTrue(result["requiresConfirmation"])
        self.assertNotIn("pendingItems", result)
        receipt_recorder.assert_awaited_once()
        self.assertNotIn("pendingItems", receipt_recorder.await_args.args[3])
        return prompt

    async def test_model_submit_unseen_overlap_notice_then_confirm_or_cancel(self):
        item = b15.pending_item()
        for answer, expected_submissions in (("确认", 1), ("取消", 0)):
            with self.subTest(answer=answer):
                self.reset_ticket()
                self.replay.calls.clear()
                self.replay.draft = [{"action": "Create", "word": b15.WORDS[0],
                                      "code": "qzbx", "type": "Phrase"}]
                self.replay.pending_items = [item]
                prompt = await self.model_submit()
                with self.subTest(part="notice"):
                    self.assertTrue(prompt.startswith(b15.notice(item) + "\n"), prompt)
                    self.assertEqual(prompt.count(b15.notice(item)), 1)
                with self.subTest(part="lookup_trace"):
                    self.assertEqual([name for name, _ in self.replay.calls],
                                     ["keytao_submit_batch", "keytao_pending_items_by_words"])
                await self.turn(answer, prompt)
                self.assertEqual(self.submissions(), expected_submissions,
                                 self.replay.last_response)
                self.assertIsNone(self.replay.store.get_record(self.replay.author))
                self.assertNotIn(b15.notice(item), self.replay.last_response)
                if answer == "确认":
                    self.assertIn("已提交审核", self.replay.last_response)
                else:
                    self.assertEqual(len(self.replay.draft), 1)
                self.general.assert_not_awaited()
                self.replay.client.create.assert_not_awaited()

    async def test_model_submit_omits_overlap_already_shown_on_ticket(self):
        item = b15.pending_item()
        self.replay.pending_items = [item]
        candidate_prompt = await self.candidates((b15.WORDS[0],))
        self.assertIn(b15.notice(item), candidate_prompt)
        self.replay.draft = [{"action": "Create", "word": b15.WORDS[0],
                              "code": "qzbx", "type": "Phrase"}]
        prompt = await self.model_submit()
        self.assertNotIn(b15.notice(item), prompt)
        await self.turn("确认", prompt)
        self.assertEqual(self.submissions(), 1, self.replay.last_response)

    async def test_model_submit_lookup_failure_keeps_reply_and_confirmation(self):
        baseline = None
        for failure in (None, "incomplete", "exception"):
            with self.subTest(failure=failure):
                self.reset_ticket()
                self.replay.draft = [{"action": "Create", "word": b15.WORDS[0],
                                      "code": "qzbx", "type": "Phrase"}]
                self.replay.pending_complete = failure != "incomplete"
                self.replay.pending_items = [b15.pending_item(
                    word="坐标系" if failure is None else b15.WORDS[0],
                )]

                async def tool(name, args, *actor):
                    if failure == "exception" and name == "keytao_pending_items_by_words":
                        raise RuntimeError("Offline lookup unavailable")
                    return await self.replay.tool(name, args, *actor)

                with patch.object(self.replay.commands.tool_executor, "call",
                                  AsyncMock(side_effect=tool)):
                    prompt = await self.model_submit()
                if baseline is None:
                    baseline = prompt
                self.assertEqual(prompt, baseline)
                self.assertNotIn("已有其他用户提交", prompt)
                await self.turn("确认", prompt)
                self.assertEqual(self.submissions(), 1, self.replay.last_response)
                self.assertIsNone(self.replay.store.get_record(self.replay.author))


class B16ItemFourTests(unittest.IsolatedAsyncioTestCase):
    setUp = b15.B15ChatTests.setUp
    turn = b15.B15ChatTests.turn
    submissions = b15.B15ChatTests.submissions
    reset_ticket = b15.B15ChatTests.reset_ticket
    model_submit = B16ItemThreeTests.model_submit

    async def test_over_cap_overlap_only_prompt_keeps_ticket_confirm_or_cancel(self):
        for answer, expected_submissions in (("确认", 1), ("取消", 0)):
            with self.subTest(answer=answer):
                self.reset_ticket()
                words = [b15.WORDS[0], *[f"offline-word-{index}" for index in range(1, 151)]]
                self.replay.draft = [
                    {"action": "Create", "word": word, "code": "qzbx", "type": "Phrase"}
                    for word in words
                ]
                self.replay.pending_items = [
                    b15.pending_item(), b15.pending_item(word=words[1]),
                ]
                formatter = self.replay.commands._format_server_warning_confirmation
                with patch.object(self.replay.commands, "_format_server_warning_confirmation",
                                  wraps=formatter) as formatted:
                    await self.turn("提交")
                self.assertGreater(len(formatter(*formatted.call_args.args)),
                                   self.replay.commands.MAX_REPLACE_CONFIRMATION_CHARS)
                prompt = self.replay.last_response
                notices = [b15.notice(item) for item in self.replay.pending_items]
                self.assertEqual(prompt, "\n".join([
                    *notices, pending_confirmation_copy(),
                    "草稿地址：https://offline.invalid/draft/offline-batch",
                ]))
                for line in notices:
                    self.assertEqual(prompt.count(line), 1)
                self.assertEqual(self.submissions(), 0)
                record = self.replay.store.get_record(self.replay.author)
                self.assertIsNotNone(record)
                self.assertTrue(self.replay.commands.server_warning_ticket_is_complete(record.state))
                self.assertEqual(record.state.function_name, "keytao_submit_batch")
                self.assertEqual(record.state.args["expected_content_version"], 1)
                self.assertEqual(record.state.args["expected_server_snapshot_digest"], "d" * 64)
                self.assertEqual(record.state.args["expected_warning_digest"], "e" * 64)
                self.assertEqual(record.state.args["expected_audit_digest"], "f" * 64)
                await self.turn(answer, prompt)
                self.assertEqual(self.submissions(), expected_submissions, self.replay.last_response)
                self.assertIsNone(self.replay.store.get_record(self.replay.author))
                for line in notices:
                    self.assertNotIn(line, self.replay.last_response)
                if answer == "取消":
                    self.assertEqual(len(self.replay.draft), len(words))
                self.general.assert_not_awaited()

    async def test_over_cap_server_warning_still_requires_smaller_draft(self):
        for entrypoint, overlap in (("command", False), ("command", True), ("legacy", True)):
            with self.subTest(entrypoint=entrypoint, overlap=overlap):
                self.reset_ticket()
                self.replay.draft = [
                    {"action": "Create", "word": f"offline-word-{index}",
                     "code": "qzbx", "type": "Phrase"}
                    for index in range(151)
                ]
                item = b15.pending_item(word=self.replay.draft[0]["word"])
                self.replay.pending_items = [item] if overlap else []

                async def tool(name, args, *actor):
                    result = await self.replay.tool(name, args, *actor)
                    if name == "keytao_submit_batch" and not args.get("confirmed"):
                        data = json.loads(result)
                        data["warnings"] = [{
                            "warningType": "server_risk", "message": "Offline server risk requires review",
                        }]
                        return json.dumps(data)
                    return result

                formatter = self.replay.commands._format_server_warning_confirmation
                with (
                    patch.object(self.replay.chat, "call_tool_function", AsyncMock(side_effect=tool)),
                    patch.object(self.replay.commands, "_format_server_warning_confirmation",
                                 wraps=formatter) as formatted,
                ):
                    if entrypoint == "command":
                        await self.turn("提交")
                    else:
                        self.assertIsNone(self.replay.commands._semantic_direct_words.get())
                        result = await self.replay.commands._perform_submit_current_draft(
                            "qq", self.replay.author.actor_id,
                            auto_confirm=True, authorize_current_draft=True,
                        )
                        self.assertIsNone(result.pending_state)
                        self.replay.last_response = self.replay.chat._prepare_user_facing_reply(
                            result.text, self.replay.memory,
                        )
                self.assertGreater(len(formatter(*formatted.call_args.args)),
                                   self.replay.commands.MAX_REPLACE_CONFIRMATION_CHARS)
                expected = (
                    "提交内容过长，本次未确认；请缩小范围。\n- 「查看草稿」\n"
                    "草稿地址：https://offline.invalid/draft/offline-batch"
                )
                if overlap and entrypoint == "command":
                    expected = b15.notice(item) + "\n" + expected
                self.assertEqual(self.replay.last_response, expected)
                self.assertIsNone(self.replay.store.get_record(self.replay.author))
                self.assertEqual(self.submissions(), 0)
                self.assertEqual(len(self.replay.draft), 151)
                self.general.assert_not_awaited()

    async def test_total_lookup_timeout_keeps_finished_chunks_and_submit_works(self):
        for entrypoint in ("command", "model"):
            for keep_finished_chunk in (False, True):
                with self.subTest(entrypoint=entrypoint, keep_finished_chunk=keep_finished_chunk):
                    self.reset_ticket()
                    self.replay.client.create.reset_mock()
                    self.general.reset_mock()
                    words = [b15.WORDS[0], *[f"offline-word-{index}" for index in range(1, 151)]]
                    self.replay.draft = [
                        {"action": "Create", "word": word, "code": "qzbx", "type": "Phrase"}
                        for word in words
                    ]
                    finished_item = b15.pending_item()
                    unfinished_item = b15.pending_item(word=words[50])
                    self.replay.pending_items = [finished_item, unfinished_item]
                    chunks, completed, cancelled = [], [], []
                    active = maximum_active = 0

                    async def slow_tool(name, args, context):
                        nonlocal active, maximum_active
                        if name != "keytao_pending_items_by_words":
                            return await self.replay.tool(name, args, context.platform, context.user_id)
                        chunk = args["words"]
                        chunks.append(chunk)
                        active += 1
                        maximum_active = max(maximum_active, active)
                        try:
                            await asyncio.sleep(0.10 if len(chunks) == 1 and keep_finished_chunk else 0.15)
                            result = await self.replay.tool(name, args, context.platform, context.user_id)
                            completed.append(chunk)
                            return result
                        except asyncio.CancelledError:
                            cancelled.append(chunk)
                            raise
                        finally:
                            active -= 1

                    with (
                        patch.object(self.replay.commands, "SUBMIT_OTHERS_LOOKUP_TIMEOUT_SECONDS",
                                     0.20 if keep_finished_chunk else 0.05, create=True),
                        patch.object(self.replay.commands.tool_executor, "call", AsyncMock(side_effect=slow_tool)),
                        patch.object(self.replay.commands.logger, "warning") as warning,
                    ):
                        if entrypoint == "model":
                            prompt = await self.model_submit()
                        else:
                            await self.turn("提交")
                            prompt = self.replay.last_response
                        expected_completed = 1 if keep_finished_chunk else 0
                        self.assertEqual(completed, [words[:50]] if keep_finished_chunk else [])
                        self.assertEqual(cancelled, [words[50:100] if keep_finished_chunk else words[:50]])
                        self.assertEqual(len(chunks), expected_completed + 1)
                        self.assertEqual(maximum_active, 1)
                        self.assertEqual(active, 0)
                        warning.assert_called_once()
                        self.assertIn("timed out", warning.call_args.args[0])
                        self.assertIn(f"{len(words) - expected_completed * 50} words were not checked",
                                      warning.call_args.args[0])
                        if keep_finished_chunk:
                            self.assertTrue(prompt.startswith(b15.notice(finished_item) + "\n"), prompt)
                            self.assertEqual(prompt.count(b15.notice(finished_item)), 1)
                        else:
                            self.assertNotIn(b15.notice(finished_item), prompt)
                        self.assertNotIn(b15.notice(unfinished_item), prompt)
                        if keep_finished_chunk or entrypoint == "model":
                            self.assertEqual(self.submissions(), 0, prompt)
                            await self.turn("确认", prompt)
                        self.assertEqual(self.submissions(), 1, self.replay.last_response)
                        self.assertIsNone(self.replay.store.get_record(self.replay.author))
                        self.assertNotIn(b15.notice(finished_item), self.replay.last_response)
                        self.general.assert_not_awaited()
                        if entrypoint == "command":
                            self.replay.client.create.assert_awaited_once()
                            self.assertEqual(self.replay.client.create.await_args.kwargs["response_format"],
                                             {"type": "json_object"})
                            self.assertEqual(self.replay.client.create.await_args.kwargs["messages"][-1], {
                                "role": "user", "content": "message: 提交\npending_context: none",
                            })
                        else:
                            self.replay.client.create.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
