"""Offline regressions for B14 bound multi-candidate selections and questions."""

import copy
import json
import socket
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from eval.routing.replay_prod_cases import Replay
from keytao_bot.utils import completed_draft_undo
from keytao_bot.utils.draft_mutation_store import DraftMutationClaimStore
from test_b12_select import SelectionReplay
from test_semantic_parser import FakeClient, raw_operation


FACTS = {
    "许褚": {"recommended": "xljj", "codes": ["xljj"], "pinyin": "xu chu"},
    "胡桃木": {"recommended": "htm", "codes": ["htm"], "pinyin": "hu tao mu"},
    "灵能": {"recommended": "lgnr", "codes": ["lgnr"], "pinyin": "ling neng"},
    "债市": {"recommended": "qheki", "codes": ["qhek", "qheki", "qhekio", "fhek", "fheki", "fhekio"], "pinyin": "zhai shi"},
    "琴酒": {"recommended": "qbjq", "codes": ["qbjq"], "pinyin": "qin jiu"},
    "数仓": {"recommended": "ejcp", "codes": ["ejcp"], "pinyin": "shu cang"},
}
T3 = "「许褚1，胡桃木1，灵能1，债市2和4， 琴酒1，数仓1」"
T2 = "为什么喵喵推荐的「债市」是qheki而不是fhek呢？两个都要加还是加一个，这种飞键要怎么加啊"
ANSWER = "qheki 和 fhek 都是「债市」的候选，fhek 是飞键。"
SELECTION_ERROR = "候选编号无效或重复，请按原候选列表选择；本次未写入。"


class MultiSelectionReplay(SelectionReplay):
    def __init__(self):
        Replay.__init__(self, {
            "sender": "b14-owner",
            "dictionary": [{"word": "确实", "code": "qhek", "type": "Phrase"}],
            "candidate_facts": copy.deepcopy(FACTS),
        }, harness, FakeClient)
        self.submit_extra_items = []

    async def tool(self, name, args, *actor, **kwargs):
        result = await super().tool(name, args, *actor, **kwargs)
        if name == "keytao_prepare_reviewed_add" and args["word"] == "债市":
            payload = json.loads(result)
            for status in payload["pronunciations"][0]["candidateStatuses"]:
                status["flyKey"] = status["code"].startswith("f")
            return json.dumps(payload, ensure_ascii=False)
        return result


class B14MultiSelectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.replay = MultiSelectionReplay()
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
        self.general = stack.enter_context(patch.object(self.replay.chat, "get_ai_response_core", AsyncMock(return_value=ANSWER)))
        stack.enter_context(patch.object(self.replay.commands.keytao_review, "compare_word_commonness", AsyncMock(return_value={})))
        stack.enter_context(patch.object(self.replay.commands.keytao_review, "assess_candidate_chain_commonness", AsyncMock(return_value=[])))

    async def candidates(self, *words):
        self.replay.store.delete(self.replay.author)
        self.replay.history.clear()
        self.replay.writes.clear()
        self.replay.draft.clear()
        self.general.reset_mock()
        await self.replay.turn({"message": "加词：" + "，".join(words), "to_me": True}, raw_operation())
        record = self.replay.store.get_record(self.replay.author)
        self.assertIsNotNone(record, self.replay.last_response)
        self.replay.store.bind_origin_message_ids(record, [1401])
        self.replay.client.create.reset_mock()
        self.general.reset_mock()
        return self.replay.last_response

    async def select(self, message, prompt):
        original = self.replay.harness.ReplyReferenceInfo

        def reference(**kwargs):
            return original(**{**kwargs, "message_id": "1401"})

        with patch.object(self.replay.harness, "ReplyReferenceInfo", side_effect=reference):
            return await self.replay.turn({
                "message": message, "to_me": True, "quote_target": prompt,
            }, raw_operation(op="select_candidate", operands=[], codes=[], refers_to_pending=True))

    def assert_offline_selection(self):
        self.general.assert_not_awaited()
        self.replay.client.create.assert_not_awaited()

    def created(self):
        result = []
        for name, args in self.replay.writes:
            if name == "keytao_create_phrase":
                result.append((args["word"], args["code"]))
            elif name == "keytao_batch_add_to_draft":
                result.extend((item["word"], item["code"]) for item in args["items"] if item.get("action") != "Delete")
        return result

    async def test_r1_exact_t3_selects_seven_bound_creates(self):
        prompt = await self.candidates(*FACTS)
        record = self.replay.store.get_record(self.replay.author)
        self.assertIsInstance(record.state, harness.PendingToolConfirm)
        self.assertEqual(len(record.state.args["_candidate_scopes"]), 6)
        self.assertIn("飞键", prompt)
        await self.select(T3, prompt)
        self.assertEqual(self.created(), [
            ("许褚", "xljj"), ("胡桃木", "htm"), ("灵能", "lgnr"),
            ("债市", "qheki"), ("债市", "fhek"), ("琴酒", "qbjq"), ("数仓", "ejcp"),
        ], self.replay.last_response)
        self.assertFalse(any(name == "keytao_submit_batch" for name, _ in self.replay.writes))
        self.assert_offline_selection()

    async def test_r1_single_and_multiword_multiselection_forms(self):
        for words in (("债市",), tuple(FACTS)):
            for message in ("债市2和4", "债市 2、4", "债市 2,4", "债市 2 4", "“债市qheki、fhek”", "「债市2和4」"):
                with self.subTest(words=words, message=message):
                    prompt = await self.candidates(*words)
                    await self.select(message, prompt)
                    self.assertEqual(self.created(), [("债市", "qheki"), ("债市", "fhek")], self.replay.last_response)
                    self.assertFalse(any(name == "keytao_submit_batch" for name, _ in self.replay.writes))
                    self.assert_offline_selection()

    async def test_r1_single_no_space_number_and_code(self):
        for words in (("债市",), tuple(FACTS)):
            for message in ("债市2", "债市qheki", "“债市2”"):
                with self.subTest(words=words, message=message):
                    prompt = await self.candidates(*words)
                    await self.select(message, prompt)
                    self.assertEqual(self.created(), [("债市", "qheki")], self.replay.last_response)
                    self.assert_offline_selection()

    async def test_r1_multiselection_keeps_add_and_submit_verbs(self):
        for suffix, submissions in (("，加入", 0), ("，加入并提交", 1)):
            with self.subTest(suffix=suffix):
                prompt = await self.candidates(*FACTS)
                await self.select("债市2、4" + suffix, prompt)
                self.assertEqual(self.created(), [("债市", "qheki"), ("债市", "fhek")], self.replay.last_response)
                self.assertEqual(sum(name == "keytao_submit_batch" for name, _ in self.replay.writes), submissions)
                self.assert_offline_selection()

    async def test_r1_unbound_tokens_reject_the_whole_message(self):
        for message in (
            "债市 2，未知词 4", "债市2和7", "债市2和0", "债市2和2",
            "债市qheki，许褚qheki", "债市 2 garbage", "债市2，胡桃木1，孤词",
            "债市2 然后", "债市2，未知词qheki", "债市2qheki",
            "不要债市2和4", "他说债市2和4", "债市2，债市4",
        ):
            with self.subTest(message=message):
                prompt = await self.candidates(*FACTS)
                record = self.replay.store.get_record(self.replay.author)
                before = copy.deepcopy(record)
                await self.select(message, prompt)
                self.assertEqual(self.replay.writes, [], self.replay.last_response)
                self.assertTrue(
                    self.replay.last_response == SELECTION_ERROR
                    or "看到了你的选词请求" in self.replay.last_response,
                    self.replay.last_response,
                )
                self.assertIs(self.replay.store.get_record(self.replay.author), record)
                self.assertEqual(record, before)
                self.general.assert_not_awaited()

    async def test_r1_single_word_unbound_tokens_reject_the_whole_message(self):
        for message in (
            "债市2和7", "债市2和0", "债市2和2", "债市 2 garbage", "债市2 然后",
            "债市2，未知词qheki", "债市2qheki", "不要债市2和4", "他说债市2和4", "债市2，债市4",
        ):
            with self.subTest(message=message):
                prompt = await self.candidates("债市")
                record = self.replay.store.get_record(self.replay.author)
                before = copy.deepcopy(record)
                await self.select(message, prompt)
                self.assertEqual(self.replay.writes, [], self.replay.last_response)
                self.assertEqual(self.replay.last_response, SELECTION_ERROR)
                self.assertIs(self.replay.store.get_record(self.replay.author), record)
                self.assertEqual(record, before)
                self.general.assert_not_awaited()

    async def test_r2_exact_t2_answer_keeps_live_ticket_and_bindings(self):
        prompt = await self.candidates(*FACTS)
        record = self.replay.store.get_record(self.replay.author)
        before = copy.deepcopy(record)
        await self.replay.turn({"message": T2}, raw_operation(
            op="query", frame="question", operands=["债市"], codes=["qheki", "fhek"], refers_to_pending=True,
        ))
        self.assertEqual(self.replay.last_response, ANSWER)
        self.assertNotEqual(self.replay.last_response, prompt)
        self.general.assert_awaited_once()
        self.assertEqual(self.replay.writes, [])
        self.assertIs(self.replay.store.get_record(self.replay.author), record)
        self.assertEqual(record, before)
        self.assertIn("1401", record.origin_message_ids)
        self.assertTrue(record.origin_prompt_digest)

    async def test_unpunctuated_code_queries_keep_live_ticket_and_reach_general_agent(self):
        for message, operands, codes in (
            ("查一下fhek", [], ["fhek"]),
            ("看看qheki", [], ["qheki"]),
            ("确实qhek", ["确实"], ["qhek"]),
            ("查一下fhek 的意思", [], ["fhek"]),
            ("看看 qheki", [], ["qheki"]),
            ("查一下 fhek 的意思", [], ["fhek"]),
            ("查一下债市qheki", ["债市"], ["qheki"]),
            ("看看债市qheki", ["债市"], ["qheki"]),
        ):
            with self.subTest(message=message):
                await self.candidates(*FACTS)
                record = self.replay.store.get_record(self.replay.author)
                before = copy.deepcopy(record)
                await self.replay.turn({"message": message}, raw_operation(
                    op="query", frame="none", operands=operands, codes=codes,
                ))
                self.assertEqual(self.replay.last_response, ANSWER)
                self.general.assert_awaited_once()
                self.assertEqual(self.replay.writes, [])
                self.assertIs(self.replay.store.get_record(self.replay.author), record)
                self.assertEqual(record, before)

    async def test_single_and_multiword_unpunctuated_word_code_queries_reach_general_agent(self):
        for words in (("债市",), tuple(FACTS)):
            for message in ("查一下债市fhek", "我觉得债市fhek不错", "债市fhek是什么意思"):
                with self.subTest(words=words, message=message):
                    await self.candidates(*words)
                    record = self.replay.store.get_record(self.replay.author)
                    self.assertIsInstance(
                        record.state, harness.PendingAddWord if len(words) == 1 else harness.PendingToolConfirm,
                    )
                    await self.replay.turn({"message": message}, raw_operation(
                        op="query", frame="none", operands=["债市"], codes=["fhek"],
                    ))
                    self.general.assert_awaited_once()
                    self.assertNotEqual(self.replay.last_response, SELECTION_ERROR)
                    self.assertEqual(self.replay.writes, [])

    async def test_r2_action_advertisement_uses_short_bound_selection_line(self):
        for message, codes, selection, selected_codes in (
            (T2, ["qheki", "fhek"], "2、4", "qheki 和 fhek"),
            ("为什么债市推荐qheki？", ["qheki"], "2、3", "qheki 和 qhekio"),
            ("债市这种飞键要怎么加啊？", [], "2、3", "qheki 和 qhekio"),
        ):
            with self.subTest(message=message):
                prompt = await self.candidates(*FACTS)
                record = self.replay.store.get_record(self.replay.author)
                before = copy.deepcopy(record)
                self.general.return_value = "回复「加入」。"
                await self.replay.turn({"message": message}, raw_operation(
                    op="query", frame="question", operands=["债市"], codes=codes, refers_to_pending=True,
                ))
                self.assertEqual(self.replay.last_response, f"「债市」可多选：回复「债市 {selection}」同时加入 {selected_codes}。")
                self.assertNotEqual(self.replay.last_response, prompt)
                self.general.assert_awaited_once()
                self.assertEqual(self.replay.writes, [])
                self.assertIs(self.replay.store.get_record(self.replay.author), record)
                self.assertEqual(record, before)


if __name__ == "__main__":
    unittest.main()
