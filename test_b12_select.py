"""Offline regressions for the B12 candidate-selection timeline."""

import copy
import json
import re
import socket
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from eval.routing.replay_prod_cases import Replay
from keytao_bot.utils import completed_draft_undo
from keytao_bot.utils.draft_mutation_store import DraftMutationClaimStore
from keytao_bot.utils.explicit_code import parse_explicit_entry_code_request
from keytao_bot.utils.pending_confirmation import shift_plan_warnings
from test_b10_flykey import FlyKeyReplay
from test_b11_shift import ShiftReplay, WORD as SHIFT_WORD
from test_semantic_parser import FakeClient, raw_operation


FACTS = {
    "荡刀": {"recommended": "dpdzia", "codes": ["dpdz", "dpdzi", "dpdzia"], "pinyin": "dang dao"},
    "打泡": {"recommended": "dspzi", "codes": ["dspz", "dspzi", "dspzia"], "pinyin": "da pao"},
    "正手刀": {"recommended": "qsdzi", "codes": ["qsdz", "qsdzi", "qsdzia"], "pinyin": "zheng shou dao"},
    "反手刀": {"recommended": "fsdzi", "codes": ["fsdz", "fsdzi", "fsdzia"], "pinyin": "fan shou dao"},
    "点刀": {"recommended": "dmdzia", "codes": ["dmdz", "dmdzi", "dmdzia"], "pinyin": "dian dao"},
    "削刀": {"recommended": "xhdzi", "codes": ["xhdz", "xhdzi", "xhdzia"], "pinyin": "xiao dao"},
}


class SelectionReplay(Replay):
    turn = FlyKeyReplay.turn

    def __init__(self):
        dictionary = [("dpdz", "挡道"), ("dpdzi", "当道"), ("dpdzia", "挡刀"),
                      ("dspz", "大炮"), ("xhdz", "学到"), ("xhdzi", "挨到"),
                      ("dmdz", "点到"), ("dmdzi", "电刀"), ("dmdzia", "颠倒")]
        super().__init__({
            "sender": "b12-owner",
            "dictionary": [{"word": word, "code": code, "type": "Phrase"} for code, word in dictionary],
            "candidate_facts": copy.deepcopy(FACTS),
        }, harness, FakeClient)
        self.submit_extra_items = []

    async def tool(self, name, args, *actor, **kwargs):
        if name == "keytao_submit_batch" and self.submit_extra_items and not args.get("confirmed"):
            self.calls.append((name, copy.deepcopy(args)))
            return json.dumps(self.result(
                success=False, requiresConfirmation=True, confirmationKind="submit",
                snapshotDigest="d" * 64, warningDigest="e" * 64, auditDigest="f" * 64,
                warnings=[{"warningType": "manual_review", "message": "旧草稿需要确认"}],
                snapshotItems=copy.deepcopy(self.draft + self.submit_extra_items),
            ), ensure_ascii=False)
        before = copy.deepcopy(self.draft)
        count = len(self.writes)
        result = await super().tool(name, args, *actor, **kwargs)
        if len(self.writes) > count:
            capture = self.chat.current_operation_capture.get()

            def snapshot(items, version):
                return {"batchId": "offline-batch", "contentVersion": version, "items": [
                    {"id": index, "action": "Create", "type": "Phrase", **item}
                    for index, item in enumerate(items, 1)
                ]}

            self.claim_store.save_completed_operation(capture.scope, "qq", self.author.actor_id, {
                "operationId": capture.operation_id, "completedAt": self.chat.time.time(),
                "before": snapshot(before, 0), "after": snapshot(self.draft, 1),
                "batchId": "offline-batch", "batchUrl": "https://offline.invalid/draft/offline-batch",
                "submitted": name == "keytao_submit_batch", "tools": [name],
            })
            payload = json.loads(result)
            payload.update(writtenItems=snapshot(self.draft, 1)["items"][len(before):], updatedItems=[])
            self.commands._capture_successful_draft_write_delivery(name, payload, "qq", self.author.actor_id)
            result = json.dumps(payload, ensure_ascii=False)
        return result


class B12SelectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.replay = SelectionReplay()
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
        self.general = stack.enter_context(patch.object(self.replay.chat, "get_ai_response_core", AsyncMock(return_value="Unexpected general response")))
        stack.enter_context(patch.object(self.replay.commands.keytao_review, "compare_word_commonness", AsyncMock(return_value={})))
        stack.enter_context(patch.object(self.replay.commands.keytao_review, "assess_candidate_chain_commonness", AsyncMock(return_value=[])))

    async def turn(self, message, quote="", reply_id=""):
        if reply_id:
            original = self.replay.harness.ReplyReferenceInfo

            def reference(**kwargs):
                return original(**{**kwargs, "message_id": reply_id})

            with patch.object(self.replay.harness, "ReplyReferenceInfo", side_effect=reference):
                return await self.replay.turn({"message": message, "to_me": True, "quote_target": quote}, raw_operation())
        return await self.replay.turn({"message": message, "to_me": True, "quote_target": quote}, raw_operation())

    async def candidates(self, *words):
        await self.turn("加词：" + "，".join(words))
        self.assertIsNotNone(self.replay.store.get_record(self.replay.author), self.replay.last_response)
        self.replay.client.create.reset_mock()
        return self.replay.last_response

    def assert_offline_route(self):
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

    def test_r1_digits_are_never_explicit_entry_codes(self):
        for message in ("削刀 3", "削刀 3，加入", "削刀 3，加入并提交", "加入 削刀 3", "给削刀加入编码3"):
            with self.subTest(message=message):
                self.assertIsNone(parse_explicit_entry_code_request(message))

    async def test_r1_single_word_number_selects_empty_candidate(self):
        prompt = await self.candidates("削刀")
        await self.turn("削刀 3，加入并提交", prompt)
        self.assertEqual(self.created(), [("削刀", "xhdzia")], self.replay.last_response)
        self.assertEqual(sum(name == "keytao_submit_batch" for name, _ in self.replay.writes), 1)
        self.assert_offline_route()

    async def test_r1_bare_word_number_selects_without_submitting(self):
        prompt = await self.candidates("削刀")
        await self.turn("削刀 3", prompt)
        self.assertEqual(self.created(), [("削刀", "xhdzia")], self.replay.last_response)
        self.assertFalse(any(name == "keytao_submit_batch" for name, _ in self.replay.writes))
        self.assert_offline_route()

    async def test_r1_multi_word_number_selects_named_candidate(self):
        prompt = await self.candidates("正手刀", "削刀")
        await self.turn("削刀 3，加入", prompt)
        self.assertEqual(self.created(), [("削刀", "xhdzia")], self.replay.last_response)
        self.assert_offline_route()

    async def test_r2_partial_selection_keeps_original_message_binding(self):
        prompt = await self.candidates("正手刀", "反手刀", "点刀", "削刀")
        self.replay.store.bind_origin_message_ids(self.replay.store.get_record(self.replay.author), [701])
        await self.turn("正手刀 3，反手刀 3，加入并提交", "Rewritten original prompt", "701")
        record = self.replay.store.get_record(self.replay.author)
        self.assertIsNotNone(record, self.replay.last_response)
        self.assertEqual([scope["word"] for scope in record.state.args["_candidate_scopes"]], ["点刀", "削刀"])
        self.assertIn("701", record.origin_message_ids)
        await self.turn("削刀 3，加入并提交", "Rewritten original prompt", "701")
        self.assertEqual(self.created(), [("正手刀", "qsdzia"), ("反手刀", "fsdzia"), ("削刀", "xhdzia")], self.replay.last_response)
        self.assert_offline_route()

    async def test_r2_submit_risk_confirmation_keeps_original_remainder_binding(self):
        await self.candidates("正手刀", "反手刀", "削刀")
        self.replay.store.bind_origin_message_ids(self.replay.store.get_record(self.replay.author), [701])
        self.replay.submit_extra_items = [{"action": "Create", "word": "旧草稿", "code": "jczg", "type": "Phrase"}]
        with patch.object(self.replay.commands, "_submit_preview_is_authors_own", AsyncMock(return_value=False)):
            await self.turn("正手刀 3，反手刀 3，加入并提交", "Original prompt rewritten", "701")
            record = self.replay.store.get_record(self.replay.author)
            self.assertIsNotNone(record, self.replay.last_response)
            self.assertEqual(record.state.function_name, "keytao_submit_batch")
            self.assertEqual(self.created(), [("正手刀", "qsdzia"), ("反手刀", "fsdzia")])
            self.assertFalse(any(name == "keytao_submit_batch" for name, _ in self.replay.writes))
            await self.turn("确认", self.replay.last_response)
        record = self.replay.store.get_record(self.replay.author)
        self.assertIsNotNone(record, self.replay.last_response)
        self.assertEqual([scope["word"] for scope in record.state.args["_candidate_scopes"]], ["削刀"])
        self.assertIn("701", record.origin_message_ids)
        await self.turn("削刀 3，加入", "Original prompt rewritten", "701")
        self.assertEqual(self.created(), [("正手刀", "qsdzia"), ("反手刀", "fsdzia"), ("削刀", "xhdzia")])
        self.assert_offline_route()

    async def test_r2_failed_remainder_persistence_keeps_written_ticket_locked(self):
        prompt = await self.candidates("正手刀", "反手刀", "削刀")
        record = self.replay.store.get_record(self.replay.author)
        with patch.object(self.replay.store, "_persist_record", return_value=False):
            await self.turn("正手刀 3，反手刀 3，加入", prompt)
        self.assertEqual(self.created(), [("正手刀", "qsdzia"), ("反手刀", "fsdzia")])
        self.assertIs(self.replay.store.get_record(self.replay.author), record)
        self.assertTrue(record.execution_id)
        writes = copy.deepcopy(self.replay.writes)
        await self.turn("正手刀 3，反手刀 3，加入", prompt)
        self.assertEqual(self.replay.writes, writes)
        self.assertTrue(record.execution_id)

    async def test_r2_failed_submit_remainder_persistence_locks_replacement_ticket(self):
        await self.candidates("正手刀", "反手刀", "削刀")
        original = self.replay.store.get_record(self.replay.author)
        self.replay.store.bind_origin_message_ids(original, [701])
        self.replay.submit_extra_items = [{"action": "Create", "word": "旧草稿", "code": "jczg", "type": "Phrase"}]
        with patch.object(self.replay.commands, "_submit_preview_is_authors_own", AsyncMock(return_value=False)), patch.object(
            self.replay.store, "put_back", return_value=False,
        ):
            await self.turn("正手刀 3，反手刀 3，加入并提交", "Original prompt rewritten", "701")
        current = self.replay.store.get_record(self.replay.author)
        self.assertIsNotNone(current, self.replay.last_response)
        self.assertIsNot(current, original)
        self.assertEqual(current.state.function_name, "keytao_submit_batch")
        self.assertTrue(current.execution_id)
        writes = copy.deepcopy(self.replay.writes)
        await self.turn("确认", self.replay.last_response)
        self.assertEqual(self.replay.writes, writes)
        self.assertIs(self.replay.store.get_record(self.replay.author), current)
        self.assertTrue(current.execution_id)

    async def test_r2_cancel_submit_risk_cancels_remainder(self):
        await self.candidates("正手刀", "反手刀", "削刀")
        self.replay.store.bind_origin_message_ids(self.replay.store.get_record(self.replay.author), [701])
        self.replay.submit_extra_items = [{"action": "Create", "word": "旧草稿", "code": "jczg", "type": "Phrase"}]
        with patch.object(self.replay.commands, "_submit_preview_is_authors_own", AsyncMock(return_value=False)):
            await self.turn("正手刀 3，反手刀 3，加入并提交", "Original prompt rewritten", "701")
        self.assertEqual(self.replay.store.get_record(self.replay.author).state.function_name, "keytao_submit_batch")
        writes = copy.deepcopy(self.replay.writes)
        await self.turn("取消", self.replay.last_response)
        self.assertIsNone(self.replay.store.get_record(self.replay.author))
        await self.turn("削刀 3", "Original prompt rewritten", "701")
        self.assertEqual(self.replay.writes, writes)
        self.assertIsInstance(self.replay.store.get_record(self.replay.author).state, harness.PendingAddWord)
        self.assert_offline_route()

    async def test_r2_cancel_cancels_remainder(self):
        prompt = await self.candidates("正手刀", "反手刀", "削刀")
        await self.turn("正手刀 3，反手刀 3，加入", prompt)
        self.assertIsNotNone(self.replay.store.get_record(self.replay.author))
        count = len(self.replay.writes)
        await self.turn("取消", prompt)
        self.assertIsNone(self.replay.store.get_record(self.replay.author))
        self.assertEqual(len(self.replay.writes), count)
        self.assert_offline_route()

    async def test_r2_number_without_ticket_refreshes_word_candidates(self):
        await self.turn("削刀 3，加入并提交")
        record = self.replay.store.get_record(self.replay.author)
        self.assertIsNotNone(record, self.replay.last_response)
        self.assertIsInstance(record.state, harness.PendingAddWord)
        self.assertEqual(record.state.word, "削刀")
        self.assertIn("xhdzia", self.replay.last_response)
        self.assertEqual(self.replay.writes, [])
        self.assert_offline_route()

    async def test_r2_stale_quote_number_refreshes_without_writing(self):
        await self.turn("削刀 3", "Old candidate prompt listing 削刀 xhdz、xhdzi、xhdzia", "701")
        record = self.replay.store.get_record(self.replay.author)
        self.assertIsNotNone(record, self.replay.last_response)
        self.assertIsInstance(record.state, harness.PendingAddWord)
        self.assertEqual(record.state.word, "削刀")
        self.assertEqual(record.state.server_candidates, [("xhdz", True), ("xhdzi", True), ("xhdzia", False)])
        self.assertTrue(any(name == "keytao_prepare_reviewed_add" and args["word"] == "削刀" for name, args in self.replay.calls))
        self.assertEqual(self.replay.writes, [])
        self.assert_offline_route()

    async def test_r3_explicit_code_grammar_uses_normal_add_path(self):
        for message in ("“削刀”加入编码xhdzia", "削刀 加入编码 xhdzia", "给削刀加入编码xhdzia"):
            with self.subTest(message=message):
                self.replay.writes.clear()
                self.replay.draft.clear()
                self.replay.history.clear()
                self.replay.store.delete(self.replay.author)
                await self.turn(message)
                self.assertEqual(self.created(), [("削刀", "xhdzia")], self.replay.last_response)
                self.assert_offline_route()

    async def test_r3_explicit_code_occupied_uses_existing_ticket(self):
        await self.turn("“荡刀”加入编码dpdzia")
        record = self.replay.store.get_record(self.replay.author)
        self.assertIsNotNone(record, self.replay.last_response)
        self.assertIsInstance(record.state, harness.PendingAddWord)
        self.assertEqual(record.state.word, "荡刀")
        self.assertEqual(self.replay.writes, [])
        self.assertIn("挡刀", self.replay.last_response)
        self.assert_offline_route()

    async def test_r3_operand_binding_consumes_complete_grammar(self):
        for message in ("他说“削刀”加入编码xhdzia", "不要给削刀加入编码xhdzia", "给削刀加入编码xhdzia好吗", "给削刀加入编码xhdzia然后删除挡刀"):
            with self.subTest(message=message):
                self.assertIsNone(parse_explicit_entry_code_request(message))

    def test_r3_explicit_literals_are_bound_by_grammar_not_word_denylists(self):
        for word in ("怎么", "我送给", "要领", "可能性", "哪吒", "哪里"):
            for message in (f"“{word}”加入编码xhdzia", f"{word} 加入编码 xhdzia", f"给{word}加入编码xhdzia"):
                with self.subTest(message=message):
                    request = parse_explicit_entry_code_request(message)
                    self.assertIsNotNone(request)
                    self.assertEqual((request.word, request.code), (word, "xhdzia"))

    async def test_r4_occupied_ticket_controls_keep_concise_instruction(self):
        for answer in ("加入", "加入并提交", "加入编码dpdzia"):
            with self.subTest(answer=answer):
                self.replay.store.delete(self.replay.author)
                await self.turn("加词：荡刀 dpdzia")
                record = self.replay.store.get_record(self.replay.author)
                self.assertIsInstance(record.state, harness.PendingAddWord)
                prompt = self.replay.last_response
                self.replay.client.create.reset_mock()
                await self.turn(answer, prompt)
                self.assertEqual(self.replay.last_response, "dpdzia 已有「挡刀」；回复「加入，顶替 挡刀」顶替，或换一个编码。")
                self.assertEqual(self.replay.store.get_record(self.replay.author).state, record.state)
                self.assertEqual(self.replay.writes, [])
                self.assert_offline_route()

    async def test_r5_receipt_names_selected_occupied_candidate(self):
        prompt = await self.candidates("打泡", "荡刀")
        await self.turn("荡刀 3，打泡 2，加入并提交", prompt)
        self.assertEqual(self.created(), [("打泡", "dspzi")], self.replay.last_response)
        self.assertIn("荡刀 3（dpdzia）已有「挡刀」，未加入。", self.replay.last_response.splitlines())
        self.assert_offline_route()

    async def test_r6_carried_occupied_word_renders_once_inside_count(self):
        await self.turn("加词：荡刀 dpdzia")
        await self.candidates("正手刀", "反手刀", "点刀", "削刀")
        response = self.replay.last_response
        self.assertEqual(response.count("「荡刀」候选编码："), 1, response)
        items = re.findall(r"(?m)^\d+[.、．)]\s*「", response)
        count = re.search(r"(\d+) 个词的候选", response)
        self.assertIsNotNone(count, response)
        self.assertEqual(int(count.group(1)), len(items), response)
        for word in ("点刀", "荡刀"):
            self.assertRegex(response, rf"(?m)^\d+[.、．)]\s*「{word}」")
        self.assert_offline_route()

    async def test_r6_empty_recommendation_keeps_occupied_words_and_remainder(self):
        for word in ("荡刀", "点刀"):
            self.replay.candidate_facts[word]["recommended"] = ""
        await self.turn("加词：荡刀 dpdzia")
        await self.candidates("正手刀", "反手刀", "点刀", "削刀")
        response = self.replay.last_response
        items = re.findall(r"(?m)^\d+[.、．)]\s*「([^」]+)」", response)
        self.assertEqual(set(items), {"荡刀", "正手刀", "反手刀", "点刀", "削刀"}, response)
        self.assertIn(f"{len(items)} 个词的候选", response)
        for word in ("荡刀", "点刀"):
            self.assertEqual(response.count(f"「{word}」候选编码："), 1, response)
        record = self.replay.store.get_record(self.replay.author)
        self.replay.store.bind_origin_message_ids(record, [702])
        await self.turn("正手刀 3，反手刀 3，加入并提交", "Original candidate prompt rewritten", "702")
        record = self.replay.store.get_record(self.replay.author)
        self.assertIsNotNone(record, self.replay.last_response)
        self.assertEqual({scope["word"] for scope in record.state.args["_candidate_scopes"]}, {"荡刀", "点刀", "削刀"})
        self.assertIn("702", record.origin_message_ids)
        self.assertEqual(self.created(), [("正手刀", "qsdzia"), ("反手刀", "fsdzia")])
        self.assert_offline_route()

    async def test_r6_no_recommendation_never_authorizes_bare_batch_add(self):
        self.replay.candidate_facts["点刀"]["recommended"] = ""
        for action in ("加入", "加入并提交"):
            with self.subTest(action=action):
                self.replay.store.delete(self.replay.author)
                self.replay.writes.clear()
                self.replay.draft.clear()
                self.replay.history.clear()
                prompt = await self.candidates("正手刀", "点刀")
                self.assertIn("推荐编码：暂无（现有候选均需点名顶替）", prompt)
                state = self.replay.store.get_record(self.replay.author).state
                self.assertEqual(state.args["_candidate_scopes"][1]["reviewedState"]["recommendedCode"], "")
                self.assertEqual([item["word"] for item in state.args["items"]], ["正手刀"])
                await self.turn(action, prompt)
                self.assertEqual(self.created(), [("正手刀", "qsdzi")], self.replay.last_response)
                writes = copy.deepcopy(self.replay.writes)
                await self.turn(action, prompt)
                self.assertEqual(self.replay.writes, writes)
                self.assert_offline_route()

    async def test_r6_no_recommendation_stays_non_actionable_in_remainder(self):
        self.replay.candidate_facts["点刀"]["recommended"] = ""
        for action in ("加入", "加入并提交"):
            with self.subTest(action=action):
                self.replay.store.delete(self.replay.author)
                self.replay.writes.clear()
                self.replay.draft.clear()
                self.replay.history.clear()
                prompt = await self.candidates("正手刀", "点刀", "削刀")
                await self.turn("正手刀 3，加入", prompt)
                self.assertEqual(self.created(), [("正手刀", "qsdzia")], self.replay.last_response)
                await self.turn(action, prompt)
                self.assertEqual(self.created(), [("正手刀", "qsdzia"), ("削刀", "xhdzi")], self.replay.last_response)
                self.assert_offline_route()

    async def test_r6_only_unbound_occupied_candidates_cannot_write(self):
        for word in ("点刀", "荡刀"):
            self.replay.candidate_facts[word]["recommended"] = ""
        prompt = await self.candidates("点刀", "荡刀")
        for action in ("加入", "加入并提交", "点刀 1，加入"):
            with self.subTest(action=action):
                await self.turn(action, prompt)
                self.assertEqual(self.replay.writes, [], self.replay.last_response)
                self.assertIsNotNone(self.replay.store.get_record(self.replay.author))
                self.general.assert_not_awaited()
        await self.turn("取消", prompt)
        self.assertIsNone(self.replay.store.get_record(self.replay.author))
        self.assertEqual(self.replay.writes, [])

    async def test_r6_merging_keeps_unbound_occupied_candidates_without_write_items(self):
        self.replay.candidate_facts["点刀"]["recommended"] = ""
        await self.candidates("正手刀", "点刀")
        prompt = await self.candidates("削刀")
        self.assertRegex(prompt, r"(?m)^\d+[.、．)]\s*「点刀」")
        state = self.replay.store.get_record(self.replay.author).state
        self.assertEqual({scope["word"] for scope in state.args["_candidate_scopes"]}, {"正手刀", "点刀", "削刀"})
        self.assertEqual([item["word"] for item in state.args["items"]], ["正手刀", "削刀"])
        self.assertEqual(self.replay.writes, [])
        self.assert_offline_route()

    async def test_r7_direct_shift_ignores_only_own_source_slot_warning(self):
        replay = ShiftReplay()
        warning = {"warningType": "skipped_candidate_slot", "item": {"word": SHIFT_WORD, "code": "xlblu"},
                   "skippedCodes": ["xlbl"], "message": '编码链中更短候选 "xlbl" 仍是空位'}

        async def tool(name, args, *actor, **kwargs):
            if name != "keytao_shift_phrase_code":
                return await ShiftReplay.tool(replay, name, args, *actor, **kwargs)
            replay.calls.append((name, copy.deepcopy(args)))
            if args.get("confirmed_plan_digest"):
                replay.writes.append((name, copy.deepcopy(args)))
                return json.dumps(replay.result(shiftPlan={"shifted": [], "items": []}))
            return json.dumps(replay.result(
                success=False, requiresConfirmation=True, confirmationKind="shiftPlan",
                planDigest="b" * 64, warningDigest="c" * 64, warnings=[warning],
                shiftPlan={"word": SHIFT_WORD, "sourceCode": "xlbl", "targetCode": "xlblu",
                           "sourcePhrase": {"word": SHIFT_WORD, "code": "xlbl"}, "shifted": [],
                           "items": [{"action": "Delete", "word": SHIFT_WORD, "code": "xlbl"},
                                     {"action": "Create", "word": SHIFT_WORD, "code": "xlblu"}]},
            ))

        with ExitStack() as stack:
            for context in replay.patches():
                stack.enter_context(context)
            stack.enter_context(patch.object(replay.chat, "call_tool_function", AsyncMock(side_effect=tool)))
            await replay.turn({"message": f"顺延“{SHIFT_WORD}”", "to_me": True}, raw_operation())
        self.assertEqual(len(replay.writes), 1, replay.last_response)
        self.assertEqual(replay.writes[0][1]["target_code"], "xlblu")
        self.assertNotIn('"xlbl" 仍是空位', replay.last_response)
        replay.client.create.assert_not_awaited()

    def test_r7_shift_ticket_suppresses_source_but_keeps_real_skipped_slot(self):
        source = {"warningType": "skipped_candidate_slot", "item": {"word": SHIFT_WORD, "code": "xlbluv"},
                  "skippedCodes": ["xlbl"], "message": '编码链中更短候选 "xlbl" 仍是空位'}
        genuine = {**source, "skippedCodes": ["xlblu"], "message": '编码链中更短候选 "xlblu" 仍是空位'}
        data = self.replay.result(
            success=False, requiresConfirmation=True, confirmationKind="shiftPlan",
            planDigest="b" * 64, warningDigest="c" * 64, warnings=[source, genuine],
            shiftPlan={"word": SHIFT_WORD, "sourceCode": "xlbl", "targetCode": "xlbluv",
                       "sourcePhrase": {"word": SHIFT_WORD, "code": "xlbl"}, "shifted": [],
                       "items": [{"action": "Delete", "word": SHIFT_WORD, "code": "xlbl"},
                                 {"action": "Create", "word": SHIFT_WORD, "code": "xlbluv"}]},
        )
        ticket = self.replay.commands._pending_state_from_server_warning(harness.PendingToolConfirm(
            "keytao_shift_phrase_code", {"word": SHIFT_WORD, "target_code": "xlbluv", "_source_code": "xlbl"},
        ), data)
        original = copy.deepcopy(ticket)
        response = self.replay.commands.render_pending_shift_plan(ticket)
        self.assertTrue(response)
        self.assertNotIn('"xlbl" 仍是空位', response)
        self.assertIn('"xlblu" 仍是空位', response)
        self.assertEqual(ticket, original)

    def test_r7_mixed_source_and_empty_slots_preserve_sealed_data(self):
        warning = {"warningType": "skipped_candidate_slot", "item": {"word": SHIFT_WORD, "code": "xlbluv"},
                   "skippedCode": "xlbl", "skippedCodes": ["xlbl", "xlblu"],
                   "message": '编码链中更短候选 "xlbl"、"xlblu" 仍是空位'}
        data = self.replay.result(
            success=False, requiresConfirmation=True, confirmationKind="shiftPlan",
            planDigest="b" * 64, warningDigest="c" * 64, warnings=[warning],
            shiftPlan={"word": SHIFT_WORD, "sourceCode": "xlbl", "targetCode": "xlbluv",
                       "sourcePhrase": {"word": SHIFT_WORD, "code": "xlbl"}, "shifted": [],
                       "items": [{"action": "Delete", "word": SHIFT_WORD, "code": "xlbl"},
                                 {"action": "Create", "word": SHIFT_WORD, "code": "xlbluv"}]},
        )
        original = copy.deepcopy(data)
        warnings = shift_plan_warnings(data)
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]["skippedCodes"], ["xlblu"])
        self.assertEqual(warnings[0]["skippedCode"], "xlblu")
        ticket = self.replay.commands._pending_state_from_server_warning(harness.PendingToolConfirm(
            "keytao_shift_phrase_code", {"word": SHIFT_WORD, "target_code": "xlbluv", "_source_code": "xlbl"},
        ), data)
        response = self.replay.commands.render_pending_shift_plan(ticket)
        self.assertIn("更短的空位", response)
        self.assertNotIn('"xlbl"', response)
        self.assertEqual(ticket.args["confirmed_plan_digest"], data["planDigest"])
        self.assertEqual(ticket.args["expected_warning_digest"], data["warningDigest"])
        self.assertEqual(data, original)


if __name__ == "__main__":
    unittest.main()
