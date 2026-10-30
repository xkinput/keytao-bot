"""Offline B10 regressions for fly-key additions and completed confirmations."""

import copy
import json
import os
import socket
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from eval.routing.replay_prod_cases import Replay
from keytao_bot.harness.semantic_parser import validate_operation
from keytao_bot.harness.state import PendingAddWord, PendingToolConfirm, create_pending_trusted_word_record
from keytao_bot.utils import completed_draft_undo
from keytao_bot.utils.draft_mutation_store import DraftMutationClaimStore
from test_semantic_parser import FakeClient, raw_operation


WORD = "赵孟頫"
CODES = ("qmf", "qmfv", "qmfva", "fmf", "fmfv", "fmfva")
FLY_CODES = CODES[3:]


class FlyKeyReplay(Replay):
    def __init__(self):
        super().__init__({
            "sender": "b10-owner",
            "dictionary": [
                {"word": "找麻烦", "code": "qmf", "type": "Phrase"},
                {"word": "找麻烦", "code": "fmf", "type": "Phrase"},
                {"word": "殖民者", "code": "fmfv", "type": "Phrase"},
            ],
            "candidate_facts": {
                WORD: {"recommended": "qmfv", "codes": list(CODES), "pinyin": "zhao meng fu"},
            },
        }, harness, FakeClient)
        self.fly_codes = list(FLY_CODES)

    async def turn(self, turn, raw):
        """Exercise the real serialized message handler with an offline model."""
        message = turn["message"]
        self.client.create.return_value.choices[0].message.content = json.dumps(raw, ensure_ascii=False)
        event = SimpleNamespace(
            to_me=turn.get("to_me", False), message=[],
            get_plaintext=lambda: message,
        )
        reply = self.harness.ReplyReferenceInfo()
        if turn.get("quote_target"):
            reply = self.harness.ReplyReferenceInfo(
                is_reply=True, is_to_bot=True, text=turn["quote_target"],
            )
        self.trace.clear()
        captured = []

        def traced(stage):
            async def run(ctx):
                captured[:] = [ctx]
                self.trace.append(stage.__name__)
                return await stage(ctx)
            return run

        tokens = (
            (self.chat._current_turn_message, self.chat._current_turn_message.set(message)),
            (self.chat._current_conversational_turn, self.chat._current_conversational_turn.set(None)),
            (self.chat.current_operation_capture, self.chat.current_operation_capture.set(None)),
            (self.chat.current_draft_delivery_claims, self.chat.current_draft_delivery_claims.set([])),
        )
        try:
            with (
                patch.object(self.chat, "STAGES", tuple(traced(stage) for stage in self.chat.STAGES)),
                patch.object(self.chat, "extract_memory_context", AsyncMock(return_value=self.memory)),
                patch.object(self.chat, "extract_reply_reference_info", AsyncMock(return_value=reply)),
                patch.object(self.chat, "_classify_message_command_intent", AsyncMock(return_value=self.chat.MessageCommandIntent())),
            ):
                await self.chat._handle_ai_chat_serialized(SimpleNamespace(), event, "qq", self.author.actor_id)
                ctx = captured[-1]
                response = ctx.response if ctx.response is not None else ctx.scoped_pending_response
                self.last_response = self.chat._prepare_user_facing_reply(str(response), self.memory)
        finally:
            for variable, token in reversed(tokens):
                variable.reset(token)
        ctx = captured[-1]
        return self.outcome(ctx)

    async def tool(self, name, args, *actor, **kwargs):
        if name == "keytao_encode":
            self.calls.append((name, copy.deepcopy(args)))
            return json.dumps({
                "success": True, "word": WORD, "recommendedCode": "qmfv",
                "candidateCodes": list(CODES), "codes": list(CODES[:3]),
                "flyKeyVariants": [{"baseCode": "fmf", "codes": self.fly_codes, "changes": []}],
                "candidateStatuses": self.statuses(),
                "phrasePinyins": ["zhao", "meng", "fu"],
            }, ensure_ascii=False)
        before = copy.deepcopy(self.draft)
        previous_writes = len(self.writes)
        result = await super().tool(name, args, *actor, **kwargs)
        if len(self.writes) > previous_writes:
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
            payload.update(
                writtenItems=snapshot(self.draft, 1)["items"][len(before):],
                updatedItems=[],
            )
            self.commands._capture_successful_draft_write_delivery(name, payload, "qq", self.author.actor_id)
            result = json.dumps(payload, ensure_ascii=False)
        if name == "keytao_prepare_reviewed_add":
            result = json.loads(result)
            result["pronunciations"][0]["candidateStatuses"] = self.statuses()
            return json.dumps(result, ensure_ascii=False)
        return result

    def statuses(self):
        return [{
            "code": code,
            "occupied": any(item["code"] == code for item in self.entries),
            "words": [item["word"] for item in self.entries if item["code"] == code],
            "label": "已占用" if any(item["code"] == code for item in self.entries) else "空位",
            "flyKey": code in self.fly_codes,
        } for code in CODES]


class B10FlyKeyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.replay = FlyKeyReplay()
        stack = self.enterContext(ExitStack())
        directory = stack.enter_context(tempfile.TemporaryDirectory())
        self.replay.claim_store = DraftMutationClaimStore(directory + "/claims.db")
        stack.enter_context(patch.object(completed_draft_undo, "get_default_draft_mutation_claim_store", return_value=self.replay.claim_store))
        stack.enter_context(patch.object(self.replay.chat, "get_default_draft_mutation_claim_store", return_value=self.replay.claim_store))
        for context in self.replay.patches():
            stack.enter_context(context)
        stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")))
        stack.enter_context(patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Offline only")))
        self.general = stack.enter_context(patch.object(
            self.replay.chat, "get_ai_response_core", AsyncMock(return_value="Unexpected general response"),
        ))

    async def add_fly_key(self, message=f"新增“{WORD}”的飞键编码"):
        return await self.replay.turn({"message": message, "to_me": True}, raw_operation(
            op="add", operands=[WORD], codes=[], code_selector="fly_key",
        ))

    async def test_selector_is_preserved_by_semantic_validation(self):
        request, violations = validate_operation(raw_operation(
            op="add", operands=[WORD], codes=[], code_selector="fly_key",
        ), f"新增“{WORD}”的飞键编码", None)
        self.assertEqual(getattr(request, "code_selector", None), "fly_key")
        self.assertEqual(violations, [])

    async def test_selector_requires_literal_fly_key_operand_and_known_value(self):
        for message, changes in (
            (f"添加{WORD}", {"code_selector": "fly_key"}),
            (f"新增{WORD}的飞键编码", {"code_selector": "invented"}),
            (f"新增{WORD}的飞键编码", {"code_selector": ["fly_key"]}),
            (f"新增{WORD}的飞键编码", {"operands": ["找麻烦"], "code_selector": "fly_key"}),
            (f"新增{WORD}的飞键编码", {"codes": ["fmfva"], "code_selector": "fly_key"}),
        ):
            with self.subTest(message=message, changes=changes):
                payload = raw_operation(op="add", operands=[WORD], codes=[])
                payload.update(changes)
                request, violations = validate_operation(payload, message, None)
                self.assertTrue(violations)
                self.assertFalse(request.op == "add" and getattr(request, "code_selector", "") == "fly_key")

    async def test_fly_key_add_uses_first_empty_marked_candidate(self):
        outcome = await self.add_fly_key()
        self.assertEqual(outcome["added"], [[WORD, "fmfva"]], self.replay.last_response)
        self.assertIn("fmfva", self.replay.last_response)
        self.assertIn("https://", self.replay.last_response)
        self.assertIsNone(self.replay.store.get_record(self.replay.author))
        self.general.assert_not_awaited()

    async def test_disabling_direct_exec_requires_confirmation_and_retains_fly_selector(self):
        with patch.dict(os.environ, {"KEYTAO_SEMANTIC_ROUTE_OPS": "add"}):
            outcome = await self.add_fly_key()
            self.assertEqual(outcome["added"], [])
            self.assertEqual(self.replay.calls, [])
            record = self.replay.store.get_record(self.replay.author)
            self.assertIsInstance(record.state, PendingToolConfirm)
            self.assertEqual(record.state.function_name, "keytao_understood_command")
            self.assertEqual(record.state.args.get("code_selector"), "fly_key")
            outcome = await self.replay.turn({"message": "确认", "to_me": True}, raw_operation(
                op="confirm", operands=[], codes=[], refers_to_pending=True,
            ))
        self.assertEqual(outcome["added"], [[WORD, "fmfva"]], self.replay.last_response)
        self.general.assert_not_awaited()

    async def test_supported_fly_key_phrasings_share_direct_route(self):
        for message in (
            f"新增{WORD}的飞键编码", f"加“{WORD}”飞键码", f"添加「{WORD}」飞键",
            f"给{WORD}加飞键编码", f'给"{WORD}"加飞键码', f"新增'{WORD}'的飞键编码",
        ):
            with self.subTest(message=message):
                self.replay.writes.clear()
                self.replay.draft.clear()
                outcome = await self.add_fly_key(message)
                self.assertEqual(outcome["added"], [[WORD, "fmfva"]], self.replay.last_response)
        self.general.assert_not_awaited()

    async def test_candidate_marking_is_authoritative_instead_of_code_prefix(self):
        self.replay.fly_codes = ["qmfva"]
        outcome = await self.add_fly_key()
        self.assertEqual(outcome["added"], [[WORD, "qmfva"]], self.replay.last_response)
        self.general.assert_not_awaited()

    async def test_existing_dictionary_code_allows_an_additional_fly_key_code(self):
        self.replay.entries.append({"word": WORD, "code": "qmfv", "type": "Phrase"})
        outcome = await self.add_fly_key()
        self.assertEqual(outcome["added"], [[WORD, "fmfva"]], self.replay.last_response)
        self.general.assert_not_awaited()

    async def test_existing_draft_code_allows_an_additional_fly_key_code(self):
        await self.assert_pending_code_allows_add("Draft", "draft")

    async def test_submitted_code_allows_an_additional_fly_key_code(self):
        await self.assert_pending_code_allows_add("Pending", "submitted")

    async def assert_pending_code_allows_add(self, status, source):
        self.replay.draft.append({
            "word": WORD, "code": "qmfv", "type": "Phrase", "action": "Create",
            "source": source, "batchId": "already-submitted", "batchStatus": status,
            "itemStatus": "Pending", "batchUrl": "https://offline.invalid/batch/already-submitted",
        })
        outcome = await self.add_fly_key()
        self.assertEqual(outcome["added"], [[WORD, "fmfva"]], self.replay.last_response)
        self.assertEqual(self.replay.draft[0]["code"], "qmfv")
        self.general.assert_not_awaited()

    async def test_no_fly_key_candidate_explains_and_never_calls_general_agent(self):
        self.replay.fly_codes = []
        outcome = await self.add_fly_key()
        self.assertEqual(outcome["added"], [])
        self.assertIn(WORD, self.replay.last_response)
        self.assertRegex(self.replay.last_response, r"(?:没有|无).*(?:飞键)")
        self.assertIsNone(self.replay.store.get_record(self.replay.author))
        self.general.assert_not_awaited()

    async def test_all_occupied_fly_keys_offer_only_bound_selectable_fly_candidates(self):
        self.replay.entries.append({"word": "非默认候选", "code": "fmfva", "type": "Phrase"})
        outcome = await self.add_fly_key()
        self.assertEqual(outcome["added"], [])
        record = self.replay.store.get_record(self.replay.author)
        self.assertIsNotNone(record)
        self.assertIsInstance(record.state, PendingAddWord)
        self.assertEqual([code for code, _occupied in record.state.candidates], list(FLY_CODES))
        self.assertTrue(self.replay.chat._advertised_reply_matches_live_record(self.replay.last_response, record))
        self.assertTrue(record.origin_prompt_digest)
        for code in CODES[:3]:
            self.assertNotRegex(self.replay.last_response, rf"\d+\. {code}\b")
        self.replay.calls.clear()
        await self.replay.turn({"message": "换 3", "to_me": True}, raw_operation(
            op="select_candidate", operands=[], codes=[], refers_to_pending=True,
        ))
        self.assertIn("非默认候选", self.replay.last_response)
        self.assertNotIn("找麻烦", self.replay.last_response)
        self.assertNotIn("殖民者", self.replay.last_response)
        self.general.assert_not_awaited()

    async def test_confirm_without_live_ticket_finishes_before_models_and_tools(self):
        await self.replay.turn({"message": "请执行，不要复读我的话", "to_me": True}, raw_operation(
            op="confirm", operands=[], codes=[], refers_to_pending=True,
        ))
        self.assertEqual(self.replay.last_response, "当前没有待确认的操作。")
        self.assertEqual(self.replay.trace[-1], "_stage_understand_operation_early")
        self.assertEqual(self.replay.calls, [])
        self.replay.client.create.assert_not_awaited()
        self.general.assert_not_awaited()

    async def test_confirm_variants_without_ticket_skip_model_and_other_actor_ticket(self):
        other = harness.ConversationAddress.group("qq", self.replay.author.space_id, "b10-other")
        self.assertTrue(self.replay.store.set(other, PendingAddWord("云翻", "yjfj", [("yjfj", False)])))
        record = self.replay.store.get_record(other)
        for message in ("确认", "请执行", "执行", "执行吧", "好的执行"):
            with self.subTest(message=message):
                await self.replay.turn({"message": message, "to_me": True}, raw_operation(
                    op="confirm", operands=[], codes=[], refers_to_pending=True,
                ))
                self.assertEqual(self.replay.last_response, "当前没有待确认的操作。")
                self.assertEqual(self.replay.calls, [])
                self.assertIs(self.replay.store.get_record(other), record)
        self.replay.client.create.assert_not_awaited()
        self.general.assert_not_awaited()

    async def test_recent_own_write_summary_is_returned_without_new_model_or_tools(self):
        with patch.object(self.replay.chat.time, "time", return_value=1000):
            outcome = await self.add_fly_key()
            self.assertEqual(outcome["added"], [[WORD, "fmfva"]])
        self.replay.calls.clear()
        self.replay.client.create.reset_mock()
        with patch.object(self.replay.chat.time, "time", return_value=1119):
            await self.replay.turn({"message": "请执行", "to_me": True}, raw_operation(
                op="confirm", operands=[], codes=[], refers_to_pending=True,
            ))
        self.assertIn(f"上一条已执行：{WORD} → fmfva", self.replay.last_response)
        self.assertEqual(self.replay.calls, [])
        self.replay.client.create.assert_not_awaited()
        self.general.assert_not_awaited()

    async def test_completed_receipt_expires_after_two_minutes(self):
        with patch.object(self.replay.chat.time, "time", return_value=1000):
            await self.add_fly_key()
        self.replay.calls.clear()
        self.replay.client.create.reset_mock()
        with patch.object(self.replay.chat.time, "time", return_value=1121):
            await self.replay.turn({"message": "执行吧", "to_me": True}, raw_operation(
                op="confirm", operands=[], codes=[], refers_to_pending=True,
            ))
        self.assertEqual(self.replay.last_response, "当前没有待确认的操作。")
        self.assertEqual(self.replay.calls, [])
        self.replay.client.create.assert_not_awaited()
        self.general.assert_not_awaited()

    async def test_quoted_plain_write_receipt_is_not_an_invitation_to_submit(self):
        with patch.object(self.replay.chat.time, "time", return_value=1000):
            outcome = await self.add_fly_key()
            self.assertEqual(outcome["added"], [[WORD, "fmfva"]])
        receipt = self.replay.last_response
        self.assertFalse(self.replay.chat.advertised_reply_contract(receipt).requires_live_state)
        self.assertIsNone(self.replay.store.get_record(self.replay.author))
        self.replay.calls.clear()
        self.replay.writes.clear()
        self.replay.client.create.reset_mock()
        with patch.object(self.replay.chat.time, "time", return_value=1001):
            await self.replay.turn({
                "message": "确认", "to_me": True, "quote_target": receipt,
            }, raw_operation(op="confirm", operands=[], codes=[], refers_to_pending=True))
        self.assertEqual(self.replay.last_response, f"当前没有待确认的操作。\n上一条已执行：{WORD} → fmfva")
        self.assertEqual(self.replay.calls, [])
        self.assertEqual(self.replay.writes, [])
        self.replay.client.create.assert_not_awaited()
        self.general.assert_not_awaited()

    async def test_completed_receipt_belongs_only_to_its_actor(self):
        with patch.object(self.replay.chat.time, "time", return_value=1000):
            await self.add_fly_key()
        self.replay.author = harness.ConversationAddress.group("qq", self.replay.author.space_id, "b10-other")
        self.replay.memory = replace(self.replay.memory, user_id=self.replay.author.actor_id)
        self.replay.calls.clear()
        self.replay.client.create.reset_mock()
        with patch.object(self.replay.chat.time, "time", return_value=1001):
            await self.replay.turn({"message": "确认", "to_me": True}, raw_operation(
                op="confirm", operands=[], codes=[], refers_to_pending=True,
            ))
        self.assertEqual(self.replay.last_response, "当前没有待确认的操作。")
        self.assertEqual(self.replay.calls, [])
        self.replay.client.create.assert_not_awaited()
        self.general.assert_not_awaited()

    async def test_history_claim_alone_does_not_create_a_recent_receipt(self):
        self.replay.history[:] = [{"role": "assistant", "content": f"已变更：{WORD} → fmfva"}]
        await self.replay.turn({"message": "请执行", "to_me": True}, raw_operation(
            op="confirm", operands=[], codes=[], refers_to_pending=True,
        ))
        self.assertEqual(self.replay.last_response, "当前没有待确认的操作。")
        self.assertEqual(self.replay.calls, [])
        self.replay.client.create.assert_not_awaited()
        self.general.assert_not_awaited()

    async def test_context_records_and_recent_submit_pointer_are_not_confirmation_tickets(self):
        states = (
            create_pending_trusted_word_record(WORD, "qmfv", "Phrase"),
            PendingToolConfirm("keytao_submit_batch", {
                "batch_id": "offline-batch", "_recent_own_write": True,
                "_recent_batch_ids": ["offline-batch"],
            }),
        )
        for state in states:
            with self.subTest(state=type(state).__name__):
                self.assertTrue(self.replay.store.set(self.replay.author, state))
                record = self.replay.store.get_record(self.replay.author)
                await self.replay.turn({"message": "请执行", "to_me": True}, raw_operation(
                    op="confirm", operands=[], codes=[], refers_to_pending=True,
                ))
                self.assertEqual(self.replay.last_response, "当前没有待确认的操作。")
                self.assertIs(self.replay.store.get_record(self.replay.author), record)
        self.assertEqual(self.replay.calls, [])
        self.replay.client.create.assert_not_awaited()
        self.general.assert_not_awaited()

    async def test_live_candidate_confirmation_keeps_its_existing_write_route(self):
        pending = PendingAddWord(
            WORD, "fmfva", [("fmfva", False)], server_candidates=[("fmfva", False)],
            pronunciation_codes={"fmfva": "zhao meng fu"}, needs_manual_review=False,
        )
        self.assertTrue(self.replay.store.set(self.replay.author, pending))
        record = self.replay.store.get_record(self.replay.author)
        prompt = self.replay.chat._render_live_single_candidate_record(record)
        self.assertTrue(self.replay.store.bind_origin_prompt_digest(
            self.replay.author, self.replay.chat._prompt_capability_digest(prompt),
        ))
        self.replay.history[:] = [{"role": "assistant", "content": prompt}]
        outcome = await self.replay.turn({"message": "确认", "to_me": True}, raw_operation(
            op="confirm", operands=[WORD], codes=[], refers_to_pending=True,
        ))
        self.assertEqual(outcome["added"], [[WORD, "fmfva"]], self.replay.last_response)
        self.general.assert_not_awaited()

    async def test_semantically_understood_confirmation_has_no_further_model_call(self):
        await self.replay.turn({"message": "帮我执行刚才那项操作", "to_me": True}, raw_operation(
            op="confirm", operands=[], codes=[], refers_to_pending=True,
        ))
        self.assertEqual(self.replay.last_response, "当前没有待确认的操作。")
        self.assertEqual(self.replay.calls, [])
        self.replay.client.create.assert_awaited_once()
        self.general.assert_not_awaited()

    async def test_completed_receipt_is_not_replayed_into_another_group(self):
        with patch.object(self.replay.chat.time, "time", return_value=1000):
            await self.add_fly_key()
        self.replay.author = harness.ConversationAddress.group("qq", "b10-other-group", self.replay.author.actor_id)
        self.replay.memory = replace(self.replay.memory, space_id=self.replay.author.space_id)
        self.replay.calls.clear()
        self.replay.client.create.reset_mock()
        with patch.object(self.replay.chat.time, "time", return_value=1001):
            await self.replay.turn({"message": "确认", "to_me": True}, raw_operation(
                op="confirm", operands=[], codes=[], refers_to_pending=True,
            ))
        self.assertEqual(self.replay.last_response, "当前没有待确认的操作。")
        self.assertEqual(self.replay.calls, [])
        self.replay.client.create.assert_not_awaited()
        self.general.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
