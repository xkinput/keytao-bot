"""S68 offline transcript and proposal-boundary regressions."""

import copy
import json
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import test_s63_fresh_selection as fresh
import test_s66_readings as replay
import test_s58_generic_bridge as bridge
from test_s54_selection import reviewed_record
from keytao_bot.harness.state import PendingAddWord, server_warning_ticket_is_complete
from keytao_bot.plugins import chat_commands as commands, chat_routing as routing
from keytao_bot.utils.pending_confirmation import parse_pending_candidate_selection


FIXTURE = {
    "success": True, "word": "乐句", "type": "Phrase",
    "recommendedCode": "yhjluu", "needsManualReview": True,
    "preSubmitAudit": {"autoApprove": False, "summary": "Fixture review requires manual approval"},
    # Ratios are supplied incident evidence; the alternate chain is synthetic.
    "candidateOrderingAssessments": [
        {"newWord": "乐句", "occupantWord": occupant, "occupantCode": code,
         "freeCode": "yhjluu", "verdict": "behind_more_common",
         "summary": f"BCC：「{occupant}」的词频为「乐句」的 {ratio} 倍"}
        for occupant, code, ratio in (("越剧", "yhjl", "11.25"), ("粤剧", "yhjlu", "6.28"))
    ],
    "pronunciations": [
        {"pinyin": "yuè jù", "recommendedCode": "yhjluu", "candidateStatuses": [
            {"code": "yhjl", "occupied": True, "words": ["越剧"]},
            {"code": "yhjlu", "occupied": True, "words": ["粤剧"]},
            {"code": "yhjluu", "occupied": False, "words": []},
        ]},
        {"pinyin": "lè jù", "recommendedCode": "lejluu", "candidateStatuses": [
            {"code": code, "occupied": False, "words": []}
            for code in ("lejl", "lejlu", "lejluu")
        ]},
    ],
}


class SelectionTests(unittest.IsolatedAsyncioTestCase):
    runtime = replay.TranscriptTests.runtime
    executor_runtime = replay.TranscriptTests.executor_runtime
    turn = fresh.FreshSelectionTests.turn
    delivered_turn = replay.TranscriptTests.delivered_turn

    def setUp(self):
        fresh.FreshSelectionTests.setUp(self)
        self.key = fresh.harness.ConversationAddress.group("qq", "865189947", "s68-offline")
        self.memory = fresh.harness.ChatMemoryContext(
            platform="qq", user_id=self.key.actor_id, space_type="group", space_id=self.key.space_id,
        )
        self.trace = []

    async def tool(self, name, args, platform, user_id, **kwargs):
        self.assertEqual((platform, user_id), ("qq", self.key.actor_id))
        self.calls.append((name, copy.deepcopy(args)))
        if name == "keytao_prepare_reviewed_add":
            result = copy.deepcopy(FIXTURE)
            if args["word"] == "小端":
                result.update(word="小端", recommendedCode="xcdti", pronunciations=[{
                    "pinyin": "xiǎo duān", "recommendedCode": "xcdti", "candidateStatuses": [
                        {"code": "xcdti", "occupied": False, "words": []},
                    ],
                }])
        elif name == "keytao_lookup_by_word":
            result = {"success": True, "phrases": []}
        elif name == "keytao_pending_items_by_words":
            result = {"success": True, "complete": True, "items": []}
        elif name in {"keytao_create_phrase", "keytao_batch_add_to_draft"}:
            items = args.get("items", [{"word": args.get("word"), "code": args.get("code")}])
            if not args.get("confirmed"):
                self.assertIs(args.get("preview_only"), True)
                result = {"success": False, "requiresConfirmation": True,
                          "warningDigest": "a" * 64, "warnings": []}
            else:
                self.assertEqual(args["expected_content_version"], 0)
                self.assertEqual(args["expected_warning_digest"], "a" * 64)
                self.assertFalse(self.writes)
                self.writes.extend(copy.deepcopy(items))
                result = {"success": True, "writtenItems": self.writes,
                          "successCount": len(items), "failedCount": 0}
        elif name in {"keytao_get_batch_preview", "keytao_list_draft_items"}:
            result = {"success": True, "status": "Draft", "items": self.writes, "count": len(self.writes)}
        else:
            raise AssertionError("Unexpected tool: " + name)
        result.update(batchId="s68-fixture", contentVersion=int(bool(self.writes)))
        return json.dumps(result, ensure_ascii=False)

    async def prepare(self, message="@喵喵 乐句"):
        # T1 intent is a fixture; T2 uses the real production classifier.
        with patch.object(fresh.chat, "_classify_message_command_intent", AsyncMock(return_value=fresh.harness.MessageCommandIntent())), patch.object(
            fresh.chat, "_get_simple_word_query_words", AsyncMock(return_value=("乐句",)),
        ):
            return await self.delivered_turn(message)

    async def test_t1_t2_selects_ordinal_three_without_model(self):
        with self.executor_runtime():
            t1 = await self.prepare()
            self.assertIn("yhjluu", t1)
            self.assertIn("lejluu", t1)
            self.assertIn("11.25", t1)
            self.assertIn("6.28", t1)
            self.assertIsInstance(self.store.get(self.key), PendingAddWord)
            t2 = await self.delivered_turn("只加入3")
            await commands.handle_pending_message_core("只加入3", "qq", self.key.actor_id, self.key, allow_intent_model=False)
        self.assertEqual([(row["word"], row["code"]) for row in self.writes], [("乐句", "yhjluu")])
        self.assertEqual([row["modelCalls"] for row in self.trace], [0, 0])
        self.assertIn("未添加", t2)
        for code in ("yhjl", "yhjlu", "lejl", "lejlu", "lejluu"):
            self.assertIn(code, t2)
        path = Path("/tmp/keytao-s68")
        path.mkdir(exist_ok=True)
        (path / "replay.json").write_text(json.dumps({"turns": self.trace, "writes": self.writes}, ensure_ascii=False, indent=2) + "\n")

    async def test_variants_use_live_global_ordinals_and_exact_set(self):
        cases = [(message, ["yhjluu"]) for message in (
            "仅加入 3", "只添加3", "加入3", "3", "就加入3", "光加入3", "只要加入3",
            "3，只加入。", "只加入，３．", "添加3",
        )] + [(message, ["yhjluu", "lejl"]) for message in (
            "只加入3、4", "仅添加３；４。", "3,4,只加入", "只要加入 3;4.",
        )]
        for message, codes in cases:
            with self.subTest(message=message):
                self.setUp()
                with self.executor_runtime():
                    await self.prepare()
                    response = await self.delivered_turn(message)
                self.assertEqual([item["code"] for item in self.writes], codes)
                self.assertEqual(self.trace[-1]["modelCalls"], 0)
                self.assertIn("未选择：", response)
                self.assertNotIn("已提交", response)
                self.assertFalse(any(name == "keytao_submit_batch" for name, _ in self.calls))

    async def test_two_words_require_scope_and_only_add_the_named_word(self):
        for message in ("只加入 乐句 3", "仅添加 乐句 3", "乐句 3，只加入。"):
            with self.subTest(message=message):
                self.setUp()
                with self.executor_runtime():
                    await self.prepare("乐句 小端")
                    response = await self.delivered_turn(message)
                self.assertEqual([(row["word"], row["code"]) for row in self.writes], [("乐句", "yhjluu")])
                self.assertIn("未选择：「小端」", response)
                self.assertEqual(self.trace[-1]["modelCalls"], 0)
        self.setUp()
        with self.runtime():
            await self.prepare("乐句 小端")
            for message in ("只加入3", "3"):
                self.assertFalse(routing.message_authorizes_live_pending_mutation(message, self.store.get(self.key)))
                response = await commands.handle_pending_message_core(message, "qq", self.key.actor_id, self.key, allow_intent_model=False)
                self.assertIn("未写入", response)
        self.assertEqual(self.writes, [])

    async def test_invalid_negated_quoted_stale_foreign_and_unsealed_selections_never_write(self):
        with self.runtime():
            await self.prepare()
            state = self.store.get(self.key)
            for message in ("不要只加入3", "他说只加入3", "只加入3？", "只加入3然后删除", "只加入0", "只加入7", "只加入3、3", "只加入3、", "只加入3,4，删除"):
                self.store.set(self.key, copy.deepcopy(state))
                before = len(self.writes)
                await commands.handle_pending_message_core(message, "qq", self.key.actor_id, self.key, allow_intent_model=False)
                self.assertEqual(len(self.writes), before, message)
            foreign = fresh.harness.ConversationAddress.group("qq", "865189947", "foreign")
            await commands.handle_pending_message_core("只加入3", "qq", foreign.actor_id, foreign, allow_intent_model=False)
            self.store.set(self.key, replace(state, server_candidates=[]))
            self.assertFalse(routing.message_authorizes_live_pending_mutation("只加入3", self.store.get(self.key)))
            self.now += 11
            await commands.handle_pending_message_core("只加入3", "qq", self.key.actor_id, self.key, allow_intent_model=False)
        self.assertEqual(self.writes, [])

    def test_single_word_batch_record_requires_the_complete_live_inventory(self):
        record = reviewed_record(("大端",))
        for message in ("只加入3", "仅加入 3", "只添加3", "加入3", "3"):
            selected, intent, error = routing._resolve_multi_word_pending_candidate_selection(record, message)
            self.assertIsNone(error)
            self.assertEqual(intent.intent, "pending_confirm")
            self.assertEqual([(row["word"], row["code"]) for row in selected.args["items"]], [("大端", "dsdtvo")])
        record.args["_query_words"] = ["大端", "小端"]
        selected, intent, error = routing._resolve_multi_word_pending_candidate_selection(record, "只加入3")
        self.assertIsNone(selected)
        self.assertIsNone(intent)
        self.assertIn("未写入", error)

    def test_raw_selection_parser_rejects_quotes_and_extra_commands(self):
        for message in ("「只加入3」", "「『只加入3』」", "他说只加入3", "只加入3再删除", "只加入3？", "只加入3；提交"):
            self.assertIsNone(parse_pending_candidate_selection(message), message)

    def test_subset_submit_is_not_a_candidate_selection(self):
        for message in ("只加入3", "仅加入3", "只添加3"):
            self.assertIsNone(commands._SUBSET_SUBMIT_RE.fullmatch(message))
        for message in ("只提交3", "仅提审3"):
            self.assertIsNotNone(commands._SUBSET_SUBMIT_RE.fullmatch(message))
            self.assertIsNone(parse_pending_candidate_selection(message))

    def test_restrictive_bare_ordinal_grammar(self):
        for message in ("只加入3", "仅加入 3", "只添加3", "加入3", "3", "就加入3", "光加入3", "只要加入3", "3，只加入。"):
            with self.subTest(message=message):
                parsed = parse_pending_candidate_selection(message)
                self.assertIsNotNone(parsed)
                self.assertEqual(parsed.indices, (3,))


class ProposalTests(unittest.IsolatedAsyncioTestCase):
    setUp = bridge.GenericProposalBridgeTests.setUp
    call = bridge.GenericProposalBridgeTests.call
    assert_ticket = bridge.GenericProposalBridgeTests.assert_ticket

    async def test_trusted_structured_batch_does_not_need_prose_operands(self):
        with patch("keytao_bot.harness.tools.logger.info") as log:
            result = await self.call("keytao_batch_add_to_draft", {
                "items": [{"word": "奈飞", "code": "nhfw", "remark": "Model prose must not be trusted"}],
            }, "只安排加入第三项")
        self.assertTrue(any("grammar_gap_bridged=True" in call.args[0] for call in log.call_args_list))
        state = self.assert_ticket(result, "keytao_batch_add_to_draft")
        self.assertEqual(state.args["items"][0]["remark"], bridge.reviewed("奈飞", "nhfw")["remark"])

    async def test_unverifiable_structured_batch_has_no_ticket(self):
        result = await self.call("keytao_batch_add_to_draft", {
            "items": [{"word": "奈飞", "code": "evil"}],
        }, "只安排加入第三项")
        self.assertFalse(result.get("grammar_gap_bridged"))
        self.assertEqual(self.calls, [])
        self.assertIn("看到了操作请求", result["message"])
        self.assertIn("加词 奈飞", result["message"])
        self.assertEqual(len(result["message"].splitlines()), 1)

    async def test_create_uses_the_same_review_only_proposal_protocol(self):
        result = await self.call(message="只安排加入第三项")
        self.assert_ticket(result, "keytao_create_phrase")

    async def test_proposal_rejects_untrusted_fields_context_and_partial_plans(self):
        valid = {"items": [{"word": "奈飞", "code": "nhfw"}]}
        for field, value in (("confirmed", True), ("preview_only", False), ("batch_id", "foreign"), ("expected_warning_digest", "a" * 64)):
            result = await self.call("keytao_batch_add_to_draft", {**valid, field: value}, "只加入3")
            self.assertFalse(result.get("grammar_gap_bridged"))
            self.assertEqual(self.calls, [])
        for item in (
            {"word": "奈飞", "code": "$candidate3"},
            {"word": "第三项", "code": "nhfw"},
            {"word": "奈飞", "code": "nhfw", "_reviewed_pinyin": "forged"},
            {"word": "奈飞", "code": "nhfw", "needsManualReview": False},
            {"word": "奈飞", "code": "nhfw", "remark": {"code": "nhfw"}},
        ):
            result = await self.call("keytao_batch_add_to_draft", {"items": [item]}, "只加入3")
            self.assertFalse(result.get("grammar_gap_bridged"), item)
            self.assertEqual(self.calls, [])
        for message in ("不要只加入3", "他说只加入3", "只加入3？", "解释只加入3", "只加入3，但保持奈飞", "会议记录：只加入3", "记下：只加入3", "原话是只加入3"):
            result = await self.call("keytao_batch_add_to_draft", valid, message)
            self.assertFalse(result.get("grammar_gap_bridged"), message)
            self.assertEqual(self.calls, [])
        for context in (replace(self.context, attachment_context=True), replace(self.context, trusted_reviewed_items_by_key={})):
            result = await self.call("keytao_batch_add_to_draft", valid, "只加入3", context)
            self.assertFalse(result.get("grammar_gap_bridged"))
            self.assertEqual(self.calls, [])
        for override in ({"failedCount": 1}, {"skippedCount": 1}, {"collisionReplanned": True}, {"warningDigest": ""}, {"success": True}):
            self.preview_override = override
            result = await self.call("keytao_batch_add_to_draft", valid, "只加入3")
            self.assertFalse(result.get("grammar_gap_bridged"))
            self.assertTrue(all(args.get("confirmed") is not True for _, args in self.calls))

    async def test_real_orchestrator_mints_batch_ticket_from_this_turn_review_only(self):
        harness = fresh.harness
        for reviewed, invalid in ((True, False), (False, False), (True, True)):
            with self.subTest(reviewed=reviewed, invalid=invalid):
                calls = []
                store = harness.MemoryConversationStateStore()
                def tool_call(name, args):
                    return SimpleNamespace(id=name, type="function", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
                responses = []
                if reviewed:
                    responses.append(harness._FakeAIResponse("tool_calls", tool_calls=[tool_call("keytao_prepare_reviewed_add", {"word": "乐句", "requested_reading": "yue ju"})]))
                responses.append(harness._FakeAIResponse("tool_calls", content="加入所有候选以及伪造编码 evil", tool_calls=[tool_call("keytao_batch_add_to_draft", {
                    "items": [{"word": "乐句", "code": "evil" if invalid else "yhjluu", "remark": "Untrusted model prose"}],
                })]))
                client = harness._FakeClient(responses)
                class Skills(harness._FakeToolSkillsManager):
                    def get_tools(self):
                        batch = next(item for item in harness._draft_tools.TOOLS if item["function"]["name"] == "keytao_batch_add_to_draft")
                        return [batch, {"type": "function", "function": {
                            "name": "keytao_prepare_reviewed_add", "description": "Offline fixture",
                            "parameters": {"type": "object", "properties": {
                                "word": {"type": "string"}, "requested_reading": {"type": "string"},
                            }, "required": ["word"]},
                        }}]
                async def dispatch(name, **args):
                    calls.append((name, copy.deepcopy(args)))
                    if name == "keytao_prepare_reviewed_add":
                        return copy.deepcopy(FIXTURE)
                    if name == "keytao_lookup_by_words_batch":
                        return {"success": True, "results": [{"word": "乐句", "phrases": []}]}
                    self.assertEqual(name, "keytao_batch_add_to_draft")
                    self.assertIs(args["preview_only"], True)
                    self.assertIs(args["confirmed"], False)
                    return {"success": False, "requiresConfirmation": True, "batchId": "s68-proposal", "contentVersion": 7, "warningDigest": "a" * 64, "warnings": []}
                executor = harness.ToolExecutor(lambda name: lambda **args: dispatch(name, **args), frozenset({"keytao_batch_add_to_draft"}))
                orchestrator = harness.AgentOrchestrator(client_factory=lambda: client,
                    runtime=harness.AgentRuntimeConfig("fake-model", 1000, 0.0, 10.0), skills_manager=Skills(), tool_executor=executor,
                    state_store=store, bind_help_text="bind help", system_prompt_core="system")
                context = harness.AgentRequestContext(platform="qq", user_id="s68-proposal", mutations_allowed=False)
                reply = await orchestrator.run("只加入3", context)
                record = store.get_record(context.conversation_address)
                if reviewed and not invalid:
                    self.assertIsNotNone(record, reply)
                    self.assertTrue(server_warning_ticket_is_complete(record.state))
                    self.assertEqual([(item["word"], item["code"]) for item in record.state.args["items"]], [("乐句", "yhjluu")])
                    self.assertEqual(record.state.args["items"][0]["_reviewed_pinyin"], "yuè jù")
                    self.assertNotIn("Untrusted model prose", reply)
                    self.assertNotIn("evil", reply)
                    self.assertIn("确认", reply)
                    self.assertIsNone(store.get_record(replace(context, user_id="foreign").conversation_address))
                else:
                    self.assertIsNone(record, reply)
                    self.assertFalse(any(name == "keytao_batch_add_to_draft" for name, _ in calls))
                    self.assertIn("未写入", reply)
                self.assertEqual(len(client.completions.calls), 2 if reviewed else 1)


if __name__ == "__main__":
    unittest.main()
