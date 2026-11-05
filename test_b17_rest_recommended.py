"""Offline B17 regressions for named selections plus sealed recommendations."""

import copy
import json
import re
import socket
import tempfile
import unittest
from contextlib import ExitStack
from collections import OrderedDict
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from eval.routing.replay_prod_cases import Replay
from keytao_bot.harness import authorization_grammar as grammar
from keytao_bot.harness.operation_request import understand_message
from keytao_bot.harness.semantic_parser import validate_operation
from keytao_bot.harness.state import _pending_add_word_payload
from keytao_bot.plugins import chat_commands as commands, chat_routing as routing
from keytao_bot.plugins.chat_commands import call_tool_function
from keytao_bot.utils import completed_draft_undo
from keytao_bot.utils.draft_mutation_store import DraftMutationClaimStore
from test_b12_select import SelectionReplay
from test_semantic_parser import FakeClient, raw_operation


FACTS = {
    "脱骨": {"recommended": "togju", "codes": ["togj", "togju"], "pinyin": "tuo gu"},
    "去骨": {"recommended": "qjgju", "codes": ["qjgj", "qjgju"], "pinyin": "qu gu"},
    "名创": {"recommended": "mgwx", "codes": ["mgwx", "mgwxi", "mgwxio", "mgwm"], "pinyin": "ming chuang"},
    "足光散": {"recommended": "zjgsi", "codes": ["zjgs", "zjgsi"], "pinyin": "zu guang san"},
    "爆汁": {"recommended": "bzqki", "codes": ["bzqk", "bzqki"], "pinyin": "bao zhi"},
    "脆香米": {"recommended": "cxmji", "codes": ["cxmj", "cxmji"], "pinyin": "cui xiang mi"},
    "尾货": {"recommended": "wwhoi", "codes": ["wwho", "wwhoi"], "pinyin": "wei huo"},
    "刷头": {"recommended": "eqtdi", "codes": ["eqtd", "eqtdi"], "pinyin": "shua tou"},
    "华夫饼": {"recommended": "hfbgi", "codes": ["hfbg", "hfbgi"], "pinyin": "hua fu bing"},
    "脆柿": {"recommended": "cvek", "codes": ["cvek", "cveki"], "pinyin": "cui shi"},
    "圣者": {"recommended": "eger", "codes": ["eger", "egeri", "egerio", "egeriu", "egeruv"], "pinyin": "sheng zhe"},
    "屁垫": {"recommended": "pbdm", "codes": ["pbdm", "pbdmi"], "pinyin": "pi dian"},
    "饰冠": {"recommended": "ekgr", "codes": ["ekgr", "ekgri"], "pinyin": "shi guan"},
    "冠饰": {"recommended": "grek", "codes": ["grek", "greki"], "pinyin": "guan shi"},
    "球坐标系": {"recommended": "qzbx", "codes": ["qzbx", "qzbxv", "qzbxvi"], "pinyin": "qiu zuo biao xi"},
    "柱坐标系": {"recommended": "qzbx", "codes": ["qzbx", "qzbxv", "qzbxvi"], "pinyin": "zhu zuo biao xi"},
}
TEN_WORDS = tuple(FACTS)[:10]
SELECTION_ERROR = "候选编号无效或重复，请按原候选列表选择；本次未写入。"


class RestReplay(SelectionReplay):
    def __init__(self):
        Replay.__init__(self, {
            "sender": "b17-owner", "dictionary": [], "candidate_facts": copy.deepcopy(FACTS),
        }, harness, FakeClient)
        self.submit_extra_items = []
        self.review_overrides = {}

    async def turn(self, turn, raw):
        # OneBot get_plaintext excludes reply/at segments from wire messages.
        turn = {**turn, "message": re.sub(r"^(?:\[reply:id=\d+\]|\[at:qq=\d+\])+\s*", "", turn["message"])}
        return await super().turn(turn, raw)

    async def tool(self, name, args, *actor, **kwargs):
        result = await super().tool(name, args, *actor, **kwargs)
        if name == "keytao_prepare_reviewed_add":
            payload = json.loads(result)
            payload.update(copy.deepcopy(self.review_overrides.get(args["word"], {})))
            if args["word"] == "名创":
                for status in payload["pronunciations"][0]["candidateStatuses"]:
                    status["flyKey"] = status["code"] == "mgwm"
            return json.dumps(payload, ensure_ascii=False)
        return result


class B17ParserTests(unittest.TestCase):
    def test_plain_selection_defaults_rest_to_false(self):
        parsed = grammar.parse_reviewed_selection_command("名创 1 4，加入并提交")
        self.assertEqual(parsed.pairs, (("名创", "1"), ("名创", "4")))
        self.assertIs(getattr(parsed, "rest_recommended", None), False)

    def test_compatibility_projection_never_drops_rest_metadata(self):
        self.assertIsNone(grammar.parse_reviewed_multi_word_selection("名创 1 4，其他用推荐的 加入并提交"))
        self.assertEqual(grammar.parse_reviewed_multi_word_selection("名创 1 4"), (("名创", "1"), ("名创", "4")))

    def test_fresh_selection_cannot_drop_rest_without_ticket(self):
        for message in ("名创 1，其他用推荐的", "名创 mgwx，其他用推荐的", "名创 mgwx，其他用推荐的 加入并提交"):
            with self.subTest(message=message):
                self.assertIsNone(commands.fresh_entry_code_selection(message))

    def test_operation_projection_fails_closed_for_multiword_rest(self):
        ticket = shift_ticket()
        request = understand_message("名创 1，其他用推荐的 加入并提交", ticket)
        self.assertEqual(request.op, "none")

    def test_semantic_model_cannot_authorize_named_subset_of_rest(self):
        for op in ("add", "select_candidate"):
            with self.subTest(op=op):
                request, _ = validate_operation(raw_operation(
                    op=op, operands=["名创"], codes=["mgwx"], refers_to_pending=True,
                ), "名创 mgwx，其他用推荐的 加入并提交", shift_ticket())
                self.assertTrue(request.op == "none" or request.frame != "direct")

    def test_complete_rest_selection_is_not_mutation_grammar_gap(self):
        self.assertFalse(grammar.looks_like_mutation_grammar_gap("名创 1，其他用推荐的 加入并提交"))

    def test_semantic_pending_renderer_never_drops_rest(self):
        from keytao_bot.harness.operation_request import OperationRequest
        request = OperationRequest(op="select_candidate", operands=("名创",), codes=("mgwx",),
                                   frame="direct", refers_to_pending=True, submit_after=True)
        message = "名创 mgwx，其他用推荐的 加入并提交"
        rendered = commands.render_understood_pending_answer(request, shift_ticket(), "", message)
        self.assertTrue(not rendered or rendered == message, rendered)


def positive_parser_case(message, action):
    def test(self):
        parsed = grammar.parse_reviewed_selection_command(message)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.pairs, (("名创", "1"), ("名创", "4")))
        self.assertEqual(parsed.action, action)
        self.assertIs(parsed.rest_recommended, True)
    return test


for case_name, message, action in (
    ("required_other", "名创 1 4，其他用推荐的 加入并提交", "加入并提交"),
    ("required_remainder", "名创 1 4，其余用推荐 加入", "加入"),
    ("required_default", "名创 1 4 其他默认", ""),
    ("required_remaining", "名创 1 4，剩下的都用推荐的，加入并提交", "加入并提交"),
    ("variant_other", "名创 1 4，其它的词全按默认编码", ""),
    ("variant_rest", "名创 1 4，剩下走推荐的编码，加入草稿", "加入草稿"),
    ("variant_words", "名创 1 4，其余词都用默认的", ""),
    ("variant_action", "名创 1 4，其他推荐 加到草稿并提交", "加到草稿并提交"),
):
    setattr(B17ParserTests, "test_r1_positive_" + case_name, positive_parser_case(message, action))


def negative_parser_case(message):
    def test(self):
        self.assertIsNone(grammar.parse_reviewed_selection_command(message))
    return test


for case_name, message in (
    ("negated", "名创 1 4，其他不用推荐的"),
    ("question", "名创 1 4，其他用推荐的吗"),
    ("question_mark", "名创 1 4，其他用推荐的？"),
    ("prefix", "其他用推荐的，名创 1 4"),
    ("middle", "名创 1，其他用推荐的，去骨 2"),
    ("missing_named", "其他用推荐的 加入并提交"),
    ("shift_only", "「占位词」顺延，其他用推荐的"),
    ("missing_separator", "名创 1 4其他默认"),
    ("extra_clause", "名创 1 4，其他用推荐的 然后删除全部"),
    ("repeated_rest", "名创 1 4，其他用推荐的，其余默认"),
    ("unknown_adverb", "名创 1 4，其他大概用推荐的"),
):
    setattr(B17ParserTests, "test_r1_negative_" + case_name, negative_parser_case(message))


def shift_ticket():
    items, scopes = [], []
    for word, code, candidates, occupied, assessments in (
        ("名创", "mgwx", [("mgwx", False), ("mgwm", False)], {}, []),
        ("刷头", "eqtd", [("eqtd", True), ("eqtdi", False)], {"eqtd": ["占位词"]}, [{
            "verdict": "front_more_common", "newWord": "刷头", "newCode": "eqtd",
            "occupantWord": "占位词", "occupantCode": "eqtd", "freeCode": "eqtdi",
        }]),
    ):
        reviewed = harness.PendingAddWord(
            word, code, candidates, server_candidates=candidates,
            server_occupied_words=occupied, server_ordering_assessments=assessments,
            pronunciation_codes={candidate: "offline reading" for candidate, _ in candidates},
            code_remarks={candidate: "Sealed " + word for candidate, _ in candidates},
            needs_manual_review=False,
        )
        items.append({"action": "Create", "word": word, "code": code, "type": "Phrase", "remark": "Sealed " + word})
        scopes.append({
            "word": word, "candidates": candidates, "occupiedWords": occupied,
            "orderingAssessments": assessments, "reviewedState": _pending_add_word_payload(reviewed),
            "reviewedPrompt": "Offline authoritative review",
        })
    return harness.PendingToolConfirm("keytao_batch_add_to_draft", {
        "items": items, "_candidate_scopes": scopes, "_reviewed_multi_word": True,
        "_query_words": [item["word"] for item in items],
    })


class B17RestRouteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.replay = RestReplay()
        stack = self.enterContext(ExitStack())
        directory = stack.enter_context(tempfile.TemporaryDirectory())
        self.replay.claim_store = DraftMutationClaimStore(directory + "/claims.db")
        stack.enter_context(patch.object(completed_draft_undo, "get_default_draft_mutation_claim_store", return_value=self.replay.claim_store))
        stack.enter_context(patch.object(self.replay.chat, "get_default_draft_mutation_claim_store", return_value=self.replay.claim_store))
        for context in self.replay.patches():
            stack.enter_context(context)
        self.replay.chat._finish_ai_chat_response.side_effect = lambda *args: completed_draft_undo.note_completed_operation_reply(
            self.replay.chat.current_draft_delivery_claims.get() or [], args[3],
        )
        stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")))
        stack.enter_context(patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Offline only")))
        self.general = stack.enter_context(patch.object(self.replay.chat, "get_ai_response_core", AsyncMock(return_value="Offline general response")))
        stack.enter_context(patch.object(self.replay.commands.keytao_review, "compare_word_commonness", AsyncMock(return_value={})))
        stack.enter_context(patch.object(self.replay.commands.keytao_review, "assess_candidate_chain_commonness", AsyncMock(return_value=[])))

    async def candidates(self, *words):
        self.replay.store.delete(self.replay.author)
        self.replay.history.clear()
        self.replay.writes.clear()
        self.replay.draft.clear()
        await self.replay.turn({"message": "加词：" + "，".join(words), "to_me": True}, raw_operation())
        record = self.replay.store.get_record(self.replay.author)
        self.assertIsNotNone(record, self.replay.last_response)
        self.replay.store.bind_origin_message_ids(record, [1701])
        self.replay.client.create.reset_mock()
        self.general.reset_mock()
        return self.replay.last_response

    async def select(self, message, prompt=""):
        original = self.replay.harness.ReplyReferenceInfo

        def reference(**kwargs):
            return original(**{**kwargs, "message_id": "1701"})

        with patch.object(self.replay.harness, "ReplyReferenceInfo", side_effect=reference):
            return await self.replay.turn({"message": message, "to_me": True, "quote_target": prompt}, raw_operation(
                op="select_candidate", operands=[], codes=[], refers_to_pending=True,
            ))

    def created(self):
        items = []
        for name, args in self.replay.writes:
            if name == "keytao_create_phrase":
                items.append((args["word"], args["code"]))
            elif name == "keytao_batch_add_to_draft":
                items.extend((item["word"], item["code"]) for item in args["items"] if item.get("action") != "Delete")
        return items

    def submissions(self):
        return sum(name == "keytao_submit_batch" for name, _ in self.replay.writes)

    def assert_offline(self):
        self.general.assert_not_awaited()
        self.replay.client.create.assert_not_awaited()

    async def test_r2_named_first_then_rest_in_ticket_order_and_submit(self):
        prompt = await self.candidates(*TEN_WORDS)
        await self.select("名创 1 4，其他用推荐的 加入并提交", prompt)
        self.assertEqual(self.created(), [("名创", "mgwx"), ("名创", "mgwm")] + [
            (word, FACTS[word]["recommended"]) for word in TEN_WORDS if word != "名创"
        ], self.replay.last_response)
        self.assertEqual(self.submissions(), 1)
        self.assertIsNone(self.replay.store.get_record(self.replay.author))
        self.assert_offline()

    async def test_r2_without_action_writes_draft_only(self):
        prompt = await self.candidates("脱骨", "名创", "去骨")
        await self.select("名创 1 4 其他默认", prompt)
        self.assertEqual(self.created(), [("名创", "mgwx"), ("名创", "mgwm"), ("脱骨", "togju"), ("去骨", "qjgju")], self.replay.last_response)
        self.assertEqual(self.submissions(), 0)
        self.assert_offline()

    async def test_r2_rest_matches_bare_add_sealed_items(self):
        prompt = await self.candidates("脱骨", "名创", "去骨")
        baseline = copy.deepcopy(self.replay.store.get_record(self.replay.author).state.args["items"])
        await self.select("加入", prompt)
        self.assertEqual(self.replay.draft, baseline)
        prompt = await self.candidates("脱骨", "名创", "去骨")
        sealed = copy.deepcopy(self.replay.store.get_record(self.replay.author).state.args["items"])
        await self.select("名创 1，其他用推荐的", prompt)
        self.assertEqual(self.replay.draft, [sealed[1], sealed[0], sealed[2]], self.replay.last_response)
        self.assert_offline()

    async def test_r2_give_way_preserves_sealed_manual_review_and_remark(self):
        reason = "「柱坐标系」@qzbx 证据不足"
        self.replay.review_overrides["柱坐标系"] = {
            "needsManualReview": True, "manualReviewReason": reason, "reviewDisposition": "SEAL",
            "preSubmitAudit": {"success": True, "autoApprove": False, "verdict": "manual", "summary": reason, "issues": [reason]},
        }
        async def tool(name, args, context):
            return await self.replay.tool(name, args, context.platform, context.user_id)

        with patch.object(self.replay.chat, "call_tool_function", call_tool_function), patch.object(
            self.replay.commands.tool_executor, "call", AsyncMock(side_effect=tool),
        ), patch.object(self.replay.commands, "_reviewed_add_verdicts", OrderedDict()):
            prompt = await self.candidates("球坐标系", "柱坐标系")
            sealed = copy.deepcopy(self.replay.store.get_record(self.replay.author).state.args["items"])
            self.assertEqual(sealed[1]["code"], "qzbxv")
            self.assertTrue(sealed[1]["needsManualReview"])
            self.assertIn("qzbxv", sealed[1]["manualReviewReason"])
            await self.select("球坐标系 1，其他用推荐的", prompt)
        self.assertEqual(self.replay.draft, sealed, self.replay.last_response)
        self.assert_offline()

    async def test_r2_no_recommendation_stays_on_original_ticket(self):
        self.replay.candidate_facts["去骨"]["recommended"] = ""
        self.replay.entries.extend({"word": "占位" + str(index), "code": code, "type": "Phrase"} for index, code in enumerate(FACTS["去骨"]["codes"]))
        prompt = await self.candidates("名创", "去骨", "脱骨")
        initial = self.replay.store.get_record(self.replay.author)
        self.assertEqual([item["word"] for item in initial.state.args["items"]], ["名创", "脱骨"])
        await self.select("名创 1 4，其他用推荐的", prompt)
        self.assertEqual(self.created(), [("名创", "mgwx"), ("名创", "mgwm"), ("脱骨", "togju")], self.replay.last_response)
        remaining = self.replay.store.get_record(self.replay.author)
        self.assertIsNotNone(remaining)
        self.assertEqual(remaining.state.args["_query_words"], ["去骨"])
        self.assertIn("去骨", self.replay.last_response)
        self.assertIn("未选择：「去骨」；这些词本次未添加。", self.replay.last_response)
        self.assertEqual(remaining.origin_message_ids, initial.origin_message_ids)
        self.assert_offline()

    async def test_r2_duplicate_check_includes_rest_and_keeps_ticket(self):
        prompt = await self.candidates("球坐标系", "柱坐标系")
        record = self.replay.store.get_record(self.replay.author)
        before = copy.deepcopy(record)
        await self.select("球坐标系 qzbxv，其他用推荐的", prompt)
        self.assertEqual(self.replay.writes, [])
        self.assertEqual(self.replay.last_response, "球坐标系、柱坐标系 选了同一个编码 qzbxv，请改选；本次未写入。")
        self.assertIs(self.replay.store.get_record(self.replay.author), record)
        self.assertEqual(record, before)
        self.assert_offline()

    async def test_r2_invalid_named_selector_keeps_whole_ticket_and_no_write(self):
        prompt = await self.candidates("名创", "脱骨")
        record = self.replay.store.get_record(self.replay.author)
        before = copy.deepcopy(record)
        await self.select("名创 9，其他用推荐的 加入并提交", prompt)
        self.assertEqual(self.replay.writes, [])
        self.assertEqual(self.replay.last_response, SELECTION_ERROR)
        self.assertIs(self.replay.store.get_record(self.replay.author), record)
        self.assertEqual(record, before)
        self.assert_offline()

    async def test_r3_single_ticket_rest_adds_nothing_and_preserves_multiselect(self):
        prompt = await self.candidates("名创")
        await self.select("名创 1 4，其他用推荐的 加入并提交", prompt)
        self.assertEqual(self.created(), [("名创", "mgwx"), ("名创", "mgwm")], self.replay.last_response)
        self.assertEqual(self.submissions(), 1)
        self.assert_offline()

    async def test_r3_single_ticket_structural_normalize_and_canonicalize(self):
        await self.candidates("名创")
        state = self.replay.store.get_record(self.replay.author).state
        message = "名创 1，其他用推荐的 加入并提交"
        intent = routing._structural_pending_add_word_intent(message, state)
        self.assertIsNotNone(intent)
        self.assertEqual((intent.intent, intent.choice_index), ("pending_add_and_submit", 1))
        self.assertEqual(routing._normalize_pending_candidate_selection_text(message, state.word), "1，加入并提交")
        canonical, error = await commands._canonicalize_pending_ticket_intent(state, message, intent, "qq", self.replay.author.actor_id)
        self.assertIsNone(error)
        self.assertEqual(canonical.choice_index, 1)
        self.assertTrue(routing.message_authorizes_live_pending_mutation(message, state))
        for reading, accepted in (("ming chuang", True), ("ming chong", False)):
            for action in ("", "，加入并提交"):
                with self.subTest(reading=reading, action=action):
                    prompt = await self.candidates("名创")
                    record = self.replay.store.get_record(self.replay.author)
                    before = copy.deepcopy(record)
                    await self.select(f"名创 {reading}: mgwx，其他用推荐的" + action, prompt)
                    self.assertEqual(self.created(), [("名创", "mgwx")] if accepted else [], self.replay.last_response)
                    self.assertEqual(self.submissions(), int(accepted and bool(action)))
                    if not accepted:
                        self.assertIs(self.replay.store.get_record(self.replay.author), record)
                        self.assertEqual(record, before)
                    self.assert_offline()

    async def test_r3_no_ticket_never_writes(self):
        await self.select("名创 1 4，其他用推荐的 加入并提交")
        self.assertEqual(self.replay.writes, [])
        self.assertIsNone(self.replay.store.get_record(self.replay.author))

    async def test_r3_rest_cannot_authorize_generic_named_only_tool_without_ticket(self):
        from keytao_bot.harness.tools import ToolContext, ToolExecutor

        for message in (
            "名创 mgwx，其他用推荐的，加入并提交",
            "名创 mgwx，其他用推荐的，加入",
            "名创 mgwx，其他用推荐的",
            "名创 1，其他用推荐的，加入并提交",
            "名创 1 4，其他用推荐的 加入并提交",
        ):
            with self.subTest(message=message):
                allowed = grammar.message_authorizes_mutation(message)
                create = AsyncMock(return_value=json.dumps({"success": True}))
                executor = ToolExecutor(
                    lambda name: create if name == "keytao_create_phrase" else None,
                    frozenset(),
                )
                result = json.loads(await executor.call(
                    "keytao_create_phrase", {"word": "名创", "code": "mgwx"},
                    ToolContext(platform="qq", user_id="b17-owner", current_message=message,
                                writes_allowed=allowed),
                ))
                self.assertFalse(allowed)
                self.assertTrue(result.get("policyBlocked"), result)
                create.assert_not_awaited()

    async def test_r3_risk_confirmation_rejects_rest_reselection(self):
        await self.candidates("名创", "脱骨")
        state = copy.deepcopy(self.replay.store.get_record(self.replay.author).state)
        state.confirmation_source = "server_warning"
        derived, intent, error = routing._resolve_multi_word_pending_candidate_selection(state, "名创 1，其他用推荐的")
        self.assertIsNone(derived)
        self.assertIsNone(intent)
        self.assertIn("当前已进入风险确认", error or "")

    async def test_r3_unquoted_openai_route_honors_rest(self):
        await self.candidates("脱骨", "名创", "去骨")
        await self.select("名创 1 4，其他用推荐的 加入并提交")
        self.assertEqual(self.created(), [("名创", "mgwx"), ("名创", "mgwm"), ("脱骨", "togju"), ("去骨", "qjgju")], self.replay.last_response)
        self.assertEqual(self.submissions(), 1)
        self.assert_offline()

    async def test_r4_four_words_two_saint_codes_submit(self):
        prompt = await self.candidates("圣者", "屁垫", "饰冠", "冠饰")
        await self.select("圣者 2 5，屁垫 2，饰冠 2，冠饰 2，加入并提交", prompt)
        self.assertEqual(self.created(), [("圣者", "egeri"), ("圣者", "egeruv"), ("屁垫", "pbdmi"), ("饰冠", "ekgri"), ("冠饰", "greki")], self.replay.last_response)
        self.assertEqual(self.submissions(), 1)
        self.assert_offline()

    async def test_r4_ten_words_two_name_codes_keep_remainder(self):
        prompt = await self.candidates(*TEN_WORDS)
        self.assertIn("飞键", prompt)
        await self.select("名创 1 4，加入并提交", prompt)
        self.assertEqual(self.created(), [("名创", "mgwx"), ("名创", "mgwm")], self.replay.last_response)
        self.assertEqual(self.submissions(), 1)
        remaining = self.replay.store.get_record(self.replay.author)
        self.assertEqual(remaining.state.args["_query_words"], [word for word in TEN_WORDS if word != "名创"])
        self.assert_offline()

    async def test_r4_reply_and_at_prefix_complete_route(self):
        prompt = await self.candidates(*TEN_WORDS)
        await self.select("[reply:id=1701][at:qq=3785773770] 名创 1 4，加入并提交", prompt)
        self.assertEqual(self.created(), [("名创", "mgwx"), ("名创", "mgwm")], self.replay.last_response)
        self.assertEqual(self.submissions(), 1)
        self.assertEqual(len(self.replay.store.get_record(self.replay.author).state.args["_query_words"]), 9)
        self.assert_offline()

    async def check_failed_then_corrected(self, failure):
        prompt = await self.candidates(*TEN_WORDS)
        record = self.replay.store.get_record(self.replay.author)
        before = copy.deepcopy(record)
        await self.select(failure, prompt)
        self.assertEqual(self.replay.writes, [])
        self.assertIs(self.replay.store.get_record(self.replay.author), record)
        self.assertEqual(record, before)
        await self.select("名创 1 4，加入并提交")
        self.assertEqual(self.created(), [("名创", "mgwx"), ("名创", "mgwm")], self.replay.last_response)
        self.assertEqual(self.submissions(), 1)
        self.assertEqual(len(self.replay.store.get_record(self.replay.author).state.args["_query_words"]), 9)
        self.assert_offline()

    async def test_r5_invalid_index_keeps_ticket_corrected_without_query(self):
        await self.check_failed_then_corrected("名创 9，加入并提交")

    async def test_r5_duplicate_selector_keeps_ticket_corrected_without_query(self):
        await self.check_failed_then_corrected("名创 1 1")


class B17ShiftPlanTests(unittest.IsolatedAsyncioTestCase):
    async def test_r2_rest_reorder_dispatch_matches_bare_add_plan(self):
        ticket = shift_ticket()
        baseline = commands._pending_batch_front_insert_plan(ticket)
        self.assertIsNotNone(baseline)
        self.assertEqual(baseline["expectedOccupantsByCode"], {"eqtd": ["占位词"]})
        derived, intent, error = routing._resolve_multi_word_pending_candidate_selection(ticket, "名创 1，其他用推荐的")
        self.assertIsNone(error, error)
        self.assertIsNotNone(derived)
        self.assertEqual(intent.intent, "pending_confirm")
        self.assertEqual(commands._pending_batch_front_insert_plan(derived), baseline)
        calls = []
        key = harness.ConversationAddress.private("qq", "b17-shift")
        for message in ("加入", "名创 1，其他用推荐的"):
            store = harness.MemoryConversationStateStore()
            store.set(key, copy.deepcopy(ticket))
            execute = AsyncMock(return_value="Offline reorder receipt")
            with patch.object(harness.openai_chat_module, "conversation_state_store", store), patch.object(commands, "_execute_confirmed_tool", execute):
                response = await commands.handle_pending_message_core(message, "qq", "b17-shift", key, allow_intent_model=False)
            self.assertIn("Offline reorder receipt", response)
            execute.assert_awaited_once()
            sent = execute.await_args.args[0]
            self.assertEqual(sent.function_name, "keytao_shift_phrase_code")
            self.assertEqual((sent.args["word"], sent.args["target_code"]), ("刷头", "eqtd"))
            self.assertEqual(sent.args["_auto_confirm_expected_shifted_words"], ["占位词"])
            self.assertTrue(execute.await_args.kwargs["auto_confirm_shift_plan"])
            calls.append((sent.args, execute.await_args.kwargs))
        self.assertEqual(calls[0], calls[1])


if __name__ == "__main__":
    unittest.main()
