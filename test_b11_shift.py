"""Offline regressions for B11 shifts, receipts and literal bot mentions."""

import copy
import json
import socket
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace

import test_state_machine as harness
import test_b6_policy as b6_policy
from eval.routing.replay_prod_cases import Replay
from keytao_bot.harness.semantic_parser import validate_operation
from keytao_bot.harness.orchestrator import AgentRequestContext
from keytao_bot.harness.tools import ToolContext, ToolExecutor
from keytao_bot.harness.authorization_grammar import parse_standalone_shift, parse_reviewed_shift_words, parse_dictionary_recode_command
from keytao_bot.utils import completed_draft_undo
from keytao_bot.utils.draft_mutation_store import DraftMutationClaimStore
from test_b10_flykey import FlyKeyReplay
from test_semantic_parser import FakeClient, raw_operation


WORD = "爱理不理"
AMBIGUOUS_SHIFT_OPERANDS = (
    "怎么", "为什么", "什么", "哪个", "哪里", "谁", "我", "你", "我们", "这个", "它",
    "帮我", "要", "已经", "可以", "能", "会", "需要", "吗", "呢", "一下", "了",
    "怎么操作", "什么词", "哪个词", "已经完成", "帮我把爱理不理", "爱理不理吗",
    "为啥", "咋", "何时", "帮忙", "还要", "一下吧", "到哪", "请问", "想要", "必须要",
    "什么时候", "什么情况下", "哪天", "哪一条", "哪辆车", "向谁", "有多少",
    "能够", "你的", "一下呗", "即将", "将要", "需不需要",
)


class ShiftReplay(Replay):
    turn = FlyKeyReplay.turn

    def __init__(self):
        super().__init__({
            "sender": "b11-owner",
            "dictionary": [{"word": WORD, "code": "xlbl", "type": "Phrase", "weight": 100}],
            "candidate_facts": {
                WORD: {"recommended": "xlbl", "codes": ["xlbl", "xlblu", "xlbluv", "xbluv"],
                       "pinyin": "ai li bu li"},
            },
        }, harness, FakeClient)
        self.shifted = []

    async def tool(self, name, args, *actor, **kwargs):
        if name == "keytao_encode":
            self.calls.append((name, copy.deepcopy(args)))
            codes = {"取消": ["qxxl", "qxxlu"], "确认": ["qrrf", "qrrfu"]}.get(
                args["word"], ["xlbl", "xlblu", "xlbluv"],
            )
            return json.dumps({"success": True, "word": args["word"],
                               "codes": codes, "candidateCodes": codes})
        if name == "keytao_shift_phrase_code":
            self.calls.append((name, copy.deepcopy(args)))
            result = self.result(shiftPlan={"shifted": self.shifted, "items": []})
            if args.get("confirmed_plan_digest"):
                self.writes.append((name, copy.deepcopy(args)))
            else:
                result.update(success=False, requiresConfirmation=True, confirmationKind="shiftPlan",
                              planDigest="b" * 64, warningDigest="c" * 64, warnings=[])
            return json.dumps(result, ensure_ascii=False)
        return await super().tool(name, args, *actor, **kwargs)


class B11ShiftTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.replay = ShiftReplay()
        stack = self.enterContext(ExitStack())
        directory = stack.enter_context(tempfile.TemporaryDirectory())
        self.claim_store = DraftMutationClaimStore(directory + "/claims.db")
        stack.enter_context(patch.object(completed_draft_undo, "get_default_draft_mutation_claim_store", return_value=self.claim_store))
        stack.enter_context(patch.object(self.replay.chat, "get_default_draft_mutation_claim_store", return_value=self.claim_store))
        for context in self.replay.patches():
            stack.enter_context(context)
        self.replay.chat._finish_ai_chat_response.side_effect = lambda *args: completed_draft_undo.note_completed_operation_reply(
            self.replay.chat.current_draft_delivery_claims.get() or [], args[3],
        )
        stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")))
        stack.enter_context(patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Offline only")))
        self.general = stack.enter_context(patch.object(self.replay.chat, "get_ai_response_core", AsyncMock(return_value="Unexpected general response")))

    async def turn(self, message, op="other", codes=(), **changes):
        return await self.replay.turn({"message": message, "to_me": True}, raw_operation(
            op=op, operands=[WORD], codes=list(codes), **changes,
        ))

    def assert_shift(self, code, word=WORD):
        self.assertEqual(len(self.replay.writes), 1, self.replay.last_response)
        name, args = self.replay.writes[0]
        self.assertEqual((name, args["word"], args["target_code"]), ("keytao_shift_phrase_code", word, code))
        self.assertTrue(args["confirmed_plan_digest"])
        self.assertIn("https://", self.replay.last_response)
        self.general.assert_not_awaited()

    async def test_standalone_shift_variants_use_next_chain_code(self):
        for message in (f"顺延“{WORD}”", f"把「{WORD}」顺延", f'"{WORD}"顺延', f"顺延『{WORD}』一位"):
            with self.subTest(message=message):
                self.replay.writes.clear()
                await self.turn(message, frame="question")
                self.assert_shift("xlblu")
                self.assertIsNone(self.replay.store.get_record(self.replay.author))

    async def test_explicit_shift_code_uses_move_route(self):
        await self.turn(f"顺延“{WORD}”到xbluv", "move", ("xbluv",))
        self.assert_shift("xbluv")

    async def test_unbound_conversational_shifts_never_write_or_override_cancel(self):
        self.replay.entries[:] = [
            {"word": word, "code": code, "type": "Phrase", "weight": 100}
            for word, code in (("取消", "qxxl"), ("确认", "qrrf"), (WORD, "xlbl"))
        ]
        for message in (
            "确认顺延", "取消顺延", "撤销顺延", "直接顺延", "确定顺延", "执行顺延",
            "同意顺延", "继续顺延", "全部顺延", "都顺延", "好的顺延",
            "顺延就行", "顺延好了", "顺延试试", "顺延成功", "先别顺延", "算了不顺延",
        ):
            with self.subTest(message=message):
                self.replay.writes.clear()
                self.replay.calls.clear()
                self.replay.store.delete(self.replay.author)
                self.replay.history.clear()
                raw = raw_operation(op="cancel", operands=[], codes=[])
                await self.replay.turn({"message": message, "to_me": True}, raw)
                self.assertEqual(self.replay.writes, [])
                self.assertFalse(any(name == "keytao_shift_phrase_code" for name, args in self.replay.calls))
                request, violations = validate_operation(raw, message, None)
                self.assertEqual((request.op, request.operands), ("cancel", ()))
                self.assertEqual(violations, [])

    async def test_unquoted_shift_quote_reply_binds_bot_named_word(self):
        await self.replay.turn({
            "message": f"{WORD}顺延", "to_me": True,
            "quote_target": f"「{WORD}」目前在 xlbl。",
        }, raw_operation(op="other", operands=[], codes=[], frame="question"))
        self.assert_shift("xlblu")

    async def test_unquoted_shift_without_reference_never_writes_even_with_keeper(self):
        for has_keeper in (False, True):
            self.replay.entries[:] = [{"word": WORD, "code": "xlbl", "type": "Phrase", "weight": 100}]
            if has_keeper:
                self.replay.entries.append({"word": "保留词", "code": "xlbl", "type": "Phrase", "weight": 100})
            for message in (f"顺延{WORD}", f"{WORD}顺延", f"把{WORD}顺延", f"顺延{WORD}一位"):
                with self.subTest(message=message, has_keeper=has_keeper):
                    self.replay.history[:] = [{"role": "assistant", "content": f"「{WORD}」目前在 xlbl。"}]
                    self.replay.calls.clear()
                    await self.replay.turn({"message": message, "to_me": True}, raw_operation(
                        op="other", operands=[], codes=[], frame="question",
                    ))
                    self.assertEqual(self.replay.writes, [])
                    self.assertFalse(any(name == "keytao_shift_phrase_code" for name, args in self.replay.calls))

    async def test_bound_shift_rejects_extra_text_or_different_word(self):
        for message in (
            f"确认{WORD}顺延", f"先别{WORD}顺延", f"怎么顺延{WORD}", f"顺延{WORD}就行",
            f"{WORD}顺延好了", f"顺延“{WORD}”试试", f"确认顺延“{WORD}”", "取消顺延",
        ):
            with self.subTest(message=message):
                self.replay.calls.clear()
                await self.replay.turn({
                    "message": message, "to_me": True, "quote_target": f"「{WORD}」目前在 xlbl。",
                }, raw_operation(op="other", operands=[], codes=[], frame="question"))
                self.assertEqual(self.replay.writes, [])
                self.assertFalse(any(name == "keytao_shift_phrase_code" for name, args in self.replay.calls))

    async def test_unquoted_shift_binds_only_own_live_ticket(self):
        state = harness.PendingAddWord(WORD, "xlbl", [("xlbl", False)], needs_manual_review=False)
        self.assertTrue(self.replay.store.set(self.replay.author, state))
        await self.turn(f"{WORD}顺延", frame="question")
        self.assert_shift("xlblu")

    async def test_unrelated_pair_ticket_cannot_authorize_a_conversational_operand(self):
        self.replay.entries[:] = [
            {"word": word, "code": "qrrf", "type": "Phrase", "weight": 100}
            for word in ("确认", "保留词")
        ]
        self.replay.store.set(self.replay.author, harness.PendingAddWord(
            "其他待加词", "xlbl", [("xlbl", True)], server_candidates=[("xlbl", True)],
            server_occupied_words={"xlbl": ["其他占用词"]}, needs_manual_review=False,
        ))
        await self.replay.turn({"message": "确认顺延", "to_me": True}, raw_operation(
            op="cancel", operands=[], codes=[],
        ))
        self.assertEqual(self.replay.writes, [])
        self.assertFalse(any(name == "keytao_shift_phrase_code" for name, args in self.replay.calls))

    async def test_quoted_bot_controls_are_not_word_references(self):
        self.replay.entries.append({"word": "取消", "code": "qxxl", "type": "Phrase", "weight": 100})
        for quote in ("回复「确认」执行，或「取消」。", f"「{WORD}」目前在 xlbl。\n回复「确认」执行，或「取消」。"):
            with self.subTest(quote=quote):
                self.replay.calls.clear()
                await self.replay.turn({"message": "取消顺延", "to_me": True, "quote_target": quote}, raw_operation(
                    op="cancel", operands=[], codes=[],
                ))
                self.assertEqual(self.replay.writes, [])
                self.assertFalse(any(name == "keytao_shift_phrase_code" for name, args in self.replay.calls))
        self.general.reset_mock()
        await self.replay.turn({"message": "顺延“取消”", "to_me": True}, raw_operation(
            op="cancel", operands=[], codes=[],
        ))
        self.assert_shift("qxxlu", "取消")

    def test_standalone_shift_reference_requires_bot_reply_and_live_owner(self):
        address = self.replay.author
        foreign = self.replay.chat.normalize_conversation_key(("qq", "other-owner"), address.space_key)
        for owner in (address, foreign):
            self.assertTrue(self.replay.store.set(owner, harness.PendingAddWord(WORD, "xlbl", [])))
            record = self.replay.store.get_record(owner)
            self.assertEqual(self.replay.commands._standalone_shift_bound_words(record, address),
                             (WORD,) if owner == address else ())
        record = self.replay.store.get_record(address)
        record.execution_id = "in-flight"
        self.assertEqual(self.replay.commands._standalone_shift_bound_words(record, address), ())
        record.execution_id = ""
        record.expires_at = 0
        self.assertIsNone(self.replay.store.get_record(address))
        for is_reply, is_to_bot in ((False, True), (True, False), (True, True)):
            reply = harness.ReplyReferenceInfo(is_reply=is_reply, is_to_bot=is_to_bot, text=f"「{WORD}」目前在 xlbl。")
            self.assertEqual(self.replay.commands._standalone_shift_bound_words(None, address, reply),
                             (WORD,) if is_reply and is_to_bot else ())

    async def test_ambiguous_shift_operands_never_write_existing_dictionary_words(self):
        for word in AMBIGUOUS_SHIFT_OPERANDS:
            for has_keeper in (False, True):
                for message in (f"{word}顺延", f"顺延{word}"):
                    with self.subTest(message=message, has_keeper=has_keeper):
                        self.replay.writes.clear()
                        self.replay.calls.clear()
                        self.replay.store.delete(self.replay.author)
                        self.replay.history[:] = [{"role": "assistant", "content": "要顺延到哪个编码？"}]
                        self.replay.entries[:] = [{"word": word, "code": "xlbl", "type": "Phrase", "weight": 100}]
                        if has_keeper:
                            self.replay.entries.append({"word": "保留词", "code": "xlbl", "type": "Phrase", "weight": 100})
                        await self.replay.turn({"message": message, "to_me": True}, raw_operation(
                            op="other", operands=[], codes=[], frame="question",
                        ))
                        self.assertEqual(self.replay.writes, [])
                        self.assertFalse(any(name == "keytao_shift_phrase_code" for name, args in self.replay.calls))

    def test_ambiguous_shift_operands_do_not_override_question_frame(self):
        for word in AMBIGUOUS_SHIFT_OPERANDS:
            for message in (f"{word}顺延", f"把{word}顺延", f"顺延{word}",
                            f"顺延{word}一位", f"顺延{word}到xlblu"):
                with self.subTest(message=message):
                    self.assertIsNone(parse_standalone_shift(message))
                    request, violations = validate_operation(raw_operation(
                        op="other", operands=[], codes=[], frame="question",
                    ), message, None)
                    self.assertEqual((request.op, request.frame, request.operands), ("other", "question", ()))
                    self.assertEqual(violations, [])

    async def test_quoted_question_word_remains_an_explicit_shift_operand(self):
        self.replay.entries[:] = [{"word": "怎么", "code": "xlbl", "type": "Phrase", "weight": 100}]
        await self.replay.turn({"message": "顺延“怎么”", "to_me": True}, raw_operation(
            op="question", operands=[], codes=[], frame="question",
        ))
        self.assertEqual([(name, args["word"], args["target_code"]) for name, args in self.replay.writes],
                         [("keytao_shift_phrase_code", "怎么", "xlblu")])

    async def test_unquoted_ordinary_words_require_a_bot_quote_reply(self):
        for word in ("哪吒", "哪怕", "几何", "寥寥无几", "我送给", "要领", "可能性"):
            self.replay.entries[:] = [{"word": word, "code": "xlbl", "type": "Phrase", "weight": 100}]
            for message in (f"顺延{word}", f"{word}顺延"):
                with self.subTest(message=message):
                    self.replay.writes.clear()
                    await self.replay.turn({
                        "message": message, "to_me": True,
                        "quote_target": f"「{word}」目前在 xlbl。",
                    }, raw_operation(
                        op="other", operands=[], codes=[], frame="question",
                    ))
                    self.assert_shift("xlblu", word)

    def test_shift_operand_guards_preserve_explicit_literals_and_ordinary_words(self):
        for word in AMBIGUOUS_SHIFT_OPERANDS:
            with self.subTest(word=word):
                self.assertIsNone(parse_reviewed_shift_words(f"{word}顺延"))
                self.assertEqual(parse_reviewed_shift_words(f"{word}顺延", known_words=(word,)), (word,))
        for word in (*AMBIGUOUS_SHIFT_OPERANDS, "我送给", "要领", "可能性", "哪吒", "哪怕", "几何", "寥寥无几", WORD):
            for message in (f"顺延“{word}”", f"把「{word}」顺延", f'"{word}"顺延'):
                with self.subTest(message=message):
                    self.assertEqual(parse_standalone_shift(message).word, word)
        for word in ("我送给", "要领", "可能性", "哪吒", "哪怕", "几何", "寥寥无几", WORD):
            for message in (f"顺延{word}", f"把{word}顺延", f"{word}顺延", f"顺延{word}一位"):
                with self.subTest(message=message):
                    self.assertIsNone(parse_standalone_shift(message))
                    self.assertEqual(parse_standalone_shift(message, known_words=(word,)).word, word)

    def test_shift_binding_consumes_only_the_complete_message(self):
        for message in (f"确认顺延“{WORD}”", f"先别{WORD}顺延", f"顺延{WORD}就行",
                        f"顺延“{WORD}” @喵喵", f"顺延喵喵 “{WORD}”"):
            with self.subTest(message=message):
                self.assertIsNone(parse_standalone_shift(message, known_words=(WORD,)))
        self.assertEqual(parse_standalone_shift(f"＠喵喵 把『{WORD}』顺延一位到xlblu！").word, WORD)
        self.assertEqual(parse_standalone_shift("顺延前一位", known_words=("前一位",)).word, "前一位")

    async def test_draft_only_word_can_shift(self):
        self.replay.entries.clear()
        self.replay.draft = [{"id": 1, "action": "Create", "word": WORD, "code": "xlbl", "type": "Phrase", "weight": 100}]
        await self.turn(f"顺延“{WORD}”")
        self.assert_shift("xlblu")

    async def test_multi_code_shift_has_bound_numbered_options_and_cancel(self):
        self.replay.entries.append({"word": WORD, "code": "xlblu", "type": "Phrase", "weight": 100})
        await self.turn(f"顺延“{WORD}”")
        response = self.replay.last_response
        self.assertRegex(response, r"(?m)^1[.、．)]")
        self.assertRegex(response, r"(?m)^2[.、．)]")
        self.assertIn("取消", response)
        record = self.replay.store.get_record(self.replay.author)
        self.assertIsNotNone(record)
        self.assertEqual(record.owner_key, self.replay.author)
        self.assertEqual(self.replay.writes, [])
        await self.turn("取消", "cancel", refers_to_pending=True)
        self.assertIsNone(self.replay.store.get_record(self.replay.author))
        self.general.assert_not_awaited()

    async def test_number_without_ticket_asks_for_literal_code(self):
        await self.turn(f"顺延“{WORD}”到1号", "move", ("1",))
        self.assertIn("编码", self.replay.last_response)
        self.assertEqual(self.replay.writes, [])
        self.general.assert_not_awaited()
        self.replay.client.create.assert_not_awaited()

    async def test_multi_code_selection_keeps_shift_route_and_source(self):
        self.replay.entries.append({"word": WORD, "code": "xlblu", "type": "Phrase", "weight": 100})
        for answer in ("2", "到2号", f"顺延{WORD}到2号"):
            with self.subTest(answer=answer):
                self.replay.writes.clear()
                await self.turn(f"顺延“{WORD}”")
                self.assertEqual(self.replay.writes, [])
                await self.turn(answer, "select_candidate", refers_to_pending=True)
                self.assert_shift("xlbluv")
                self.assertEqual(self.replay.writes[0][1]["_source_code"], "xlblu")

    async def test_multi_code_explicit_target_can_select_existing_target_without_write(self):
        self.replay.entries.append({"word": WORD, "code": "xlblu", "type": "Phrase", "weight": 100})
        await self.turn(f"顺延“{WORD}”到xlblu", "move", ("xlblu",))
        self.assertRegex(self.replay.last_response, r"(?m)^2[.、．)]")
        self.assertIsNotNone(self.replay.store.get_record(self.replay.author))
        await self.turn("2", "select_candidate", refers_to_pending=True)
        self.assertEqual(self.replay.writes, [])
        self.assertIn("已在 xlblu", self.replay.last_response)
        self.general.assert_not_awaited()

    async def test_occupied_next_code_keeps_full_chain_confirmation(self):
        self.replay.entries.append({"word": "别的词", "code": "xlblu", "type": "Phrase", "weight": 100})
        self.replay.shifted = [{"word": "别的词", "fromCode": "xlblu", "toCode": "bdco"}]
        await self.turn(f"顺延“{WORD}”")
        self.assertEqual(self.replay.writes, [])
        self.assertIsNotNone(self.replay.store.get_record(self.replay.author))
        self.assertIn("确认", self.replay.last_response)
        self.general.assert_not_awaited()

    async def test_ordinal_selects_only_live_bound_numbered_candidates(self):
        state = harness.PendingAddWord(
            WORD, "xlblu", [("xlblu", False), ("xbluv", False)],
            server_candidates=[("xlblu", False), ("xbluv", False)],
            pronunciation_codes={"xlblu": "ai li bu li", "xbluv": "ai li bu li"},
            needs_manual_review=False,
        )
        self.assertTrue(self.replay.store.set(self.replay.author, state))
        record = self.replay.store.get_record(self.replay.author)
        prompt = self.replay.chat._render_live_single_candidate_record(record)
        self.replay.store.bind_origin_prompt_digest(self.replay.author, self.replay.chat._prompt_capability_digest(prompt))
        self.replay.history[:] = [{"role": "assistant", "content": prompt}]
        await self.turn("到2号", "select_candidate", refers_to_pending=True)
        self.assertTrue(any(args.get("code") == "xbluv" for name, args in self.replay.calls if name == "keytao_create_phrase"), self.replay.last_response)
        self.general.assert_not_awaited()

    async def test_configured_nickname_prefix_recode_executes(self):
        with patch.object(self.replay.chat.config, "nickname", {"小键"}, create=True):
            await self.turn(f"＠ 小键 修改编码“{WORD}”为xlblu", "recode", ("xlblu",))
        self.assertEqual(len(self.replay.writes), 1, self.replay.last_response)
        self.general.assert_not_awaited()

    async def test_mentions_are_removed_before_recode_validation(self):
        for prefix in ("＠喵喵 ", "@喵喵 ", "＠ 喵喵", "@喵喵"):
            with self.subTest(prefix=prefix):
                self.replay.writes.clear()
                await self.turn(prefix + f"修改编码“{WORD}”为xlblu", "recode", ("xlblu",))
                self.assertEqual(len(self.replay.writes), 1, self.replay.last_response)
                self.assertEqual([(row["action"], row["word"], row["code"])
                                  for row in self.replay.writes[0][1]["items"]],
                                 [("Delete", WORD, "xlbl"), ("Create", WORD, "xlblu")])
        self.general.assert_not_awaited()

    async def test_bare_word_code_offers_recode_or_add_and_executes_choice(self):
        for answer in ("改码", "加入"):
            with self.subTest(answer=answer):
                self.replay.writes.clear()
                self.replay.draft.clear()
                await self.turn(f"{WORD} xlblu", "query", ("xlblu",))
                self.assertEqual(self.replay.writes, [])
                self.assertIn("目前在 xlbl", self.replay.last_response)
                for control in ("改码", "加入", "取消"):
                    self.assertIn(f"「{control}」", self.replay.last_response)
                record = self.replay.store.get_record(self.replay.author)
                self.assertIsNotNone(record)
                self.assertEqual(record.owner_key, self.replay.author)
                await self.turn(answer, "confirm", refers_to_pending=True)
                actions = [row["action"] for name, args in self.replay.writes
                           for row in args.get("items", [{"action": "Create"}])]
                self.assertEqual(actions, ["Delete", "Create"] if answer == "改码" else ["Create"], self.replay.last_response)
                self.assertIsNone(self.replay.store.get_record(self.replay.author))
        self.general.assert_not_awaited()

    async def test_bare_word_code_absent_word_keeps_explicit_add_route(self):
        self.replay.entries.clear()
        await self.turn(f"{WORD} xlblu", "query", ("xlblu",), frame="none")
        self.assertEqual(len(self.replay.writes), 1, self.replay.last_response)
        self.assertIn(("keytao_prepare_reviewed_add", {"word": WORD}), self.replay.calls)
        self.general.assert_not_awaited()

    async def test_literal_word_code_ignores_model_frame_and_operation(self):
        for op in ("query", "other", "help", "recode", "move", "add"):
            with self.subTest(op=op):
                self.replay.store.delete(self.replay.author)
                await self.turn(f"{WORD} xlblu", op, ("xlblu",), frame="question")
                self.assertIn("回复「改码」", self.replay.last_response)
                self.assertEqual(self.replay.writes, [])
        self.general.assert_not_awaited()

    async def test_unnumbered_recode_offer_rejects_ordinal_without_consuming(self):
        await self.turn(f"{WORD} xlblu", "query", ("xlblu",), frame="none")
        record = self.replay.store.get_record(self.replay.author)
        await self.turn("到1号", "select_candidate", refers_to_pending=True)
        self.assertEqual(self.replay.writes, [])
        self.assertIs(self.replay.store.get_record(self.replay.author), record)
        self.assertIn("编码", self.replay.last_response)

    async def test_bare_word_code_occupied_code_keeps_explicit_add_route(self):
        self.replay.entries.append({"word": "别的词", "code": "xlblu", "type": "Phrase", "weight": 100})
        await self.turn(f"{WORD} xlblu", "query", ("xlblu",), frame="none")
        self.assertIn(("keytao_prepare_reviewed_add", {"word": WORD}), self.replay.calls)
        self.assertNotIn("目前在 xlbl", self.replay.last_response)
        self.general.assert_not_awaited()

    async def test_bare_draft_word_recode_keeps_bound_source(self):
        self.replay.entries.clear()
        self.replay.draft = [{"id": 1, "action": "Create", "word": WORD, "code": "xlbl", "type": "Phrase", "weight": 100}]
        await self.turn(f"{WORD} xlblu", "query", ("xlblu",), frame="none")
        await self.turn("改码", "confirm", refers_to_pending=True)
        self.assert_shift("xlblu")
        self.assertEqual(self.replay.writes[0][1]["_source_code"], "xlbl")

    def seed_receipt(self):
        capture = completed_draft_undo.start_operation_turn(self.replay.author)
        self.claim_store.save_completed_operation(capture.scope, "qq", self.replay.author.actor_id, {
            "operationId": capture.operation_id, "completedAt": self.replay.chat.time.time(),
            "lastReplyOperationId": capture.operation_id,
            "before": {"items": []},
            "after": {"items": [{"id": 1, "action": "Create", "word": "电脑端", "code": "dndi"}]},
        })
        self.replay.history[:] = [{"role": "assistant", "content": "✅ 已加入草稿：电脑端 → dndi"}]

    async def test_intervening_reply_prevents_unrelated_completed_summary(self):
        self.seed_receipt()
        await self.turn(WORD, "query", frame="none")
        self.assertIn("xlbl", self.replay.last_response)
        await self.turn("确认", "confirm", refers_to_pending=True)
        self.assertEqual(self.replay.last_response, "当前没有待确认的操作。")
        self.general.assert_not_awaited()

    async def test_last_receipt_summary_is_next_reply_only(self):
        self.seed_receipt()
        await self.turn("确认", "confirm", refers_to_pending=True)
        self.assertEqual(self.replay.last_response, "当前没有待确认的操作。\n上一条已执行：电脑端 → dndi")
        await self.turn("确认", "confirm", refers_to_pending=True)
        self.assertEqual(self.replay.last_response, "当前没有待确认的操作。")

    async def test_turn_without_reply_does_not_hide_last_receipt(self):
        self.seed_receipt()
        completed_draft_undo.start_operation_turn(self.replay.author)
        await self.turn("确认", "confirm", refers_to_pending=True)
        self.assertIn("电脑端 → dndi", self.replay.last_response)

    async def test_image_reply_without_operation_capture_clears_receipt(self):
        self.seed_receipt()
        token = completed_draft_undo.current_operation_capture.set(None)
        self.addCleanup(completed_draft_undo.current_operation_capture.reset, token)
        ctx = self.replay.chat.TurnContext(
            bot=SimpleNamespace(), event=SimpleNamespace(), platform="qq", user_id=self.replay.author.actor_id,
            image_attachments=(object(),), vision_error=self.replay.chat.ImageInputError("Offline invalid image"),
        )
        with patch.object(self.replay.chat, "extract_memory_context", AsyncMock(return_value=self.replay.memory)), patch.object(
            self.replay.chat, "remember_visual_conversation_marker",
        ):
            self.assertTrue(await self.replay.chat._stage_handle_image_turn(ctx))
        await self.turn("确认", "confirm", refers_to_pending=True)
        self.assertEqual(self.replay.last_response, "当前没有待确认的操作。")

    async def test_progress_reply_clears_receipt_without_acknowledging_writes(self):
        self.seed_receipt()
        bot = SimpleNamespace(send=AsyncMock())
        self.assertTrue(await self.replay.chat._send_event_response(
            bot, SimpleNamespace(), self.replay.author.actor_id, self.replay.memory, "处理中。", emit_metrics=False,
        ))
        await self.turn("确认", "confirm", refers_to_pending=True)
        self.assertEqual(self.replay.last_response, "当前没有待确认的操作。")

    def test_negated_and_reported_shift_or_recode_never_authorize(self):
        for message in (f"不要顺延{WORD}", f"他说{WORD}顺延", f"顺延{WORD}？", f'"顺延{WORD}"'):
            self.assertIsNone(parse_standalone_shift(message))
        self.assertIsNotNone(parse_dictionary_recode_command(f"删除 {WORD} xlbl；添加 {WORD} xlblu"))
        self.assertIsNone(parse_dictionary_recode_command(f"不要删除 {WORD} xlbl；添加 {WORD} xlblu"))

    def test_semantic_validator_recognizes_shift_even_if_model_says_other(self):
        request, violations = validate_operation(raw_operation(op="other", operands=[WORD], codes=[], frame="question"), f"顺延“{WORD}”", None)
        self.assertEqual((request.op, request.frame, request.operands), ("move", "direct", (WORD,)))
        self.assertEqual(violations, [])

    def test_only_internal_routes_can_bind_shift_source(self):
        executor = ToolExecutor(lambda name: None, frozenset())
        args = {"word": WORD, "target_code": "xlblu", "_source_code": "xlbl"}
        model_args = executor.canonicalize_arguments(
            "keytao_shift_phrase_code", args, ToolContext(current_message=f"顺延{WORD}到xlblu"),
        )
        self.assertNotIn("_source_code", model_args)
        internal_args = executor.canonicalize_arguments("keytao_shift_phrase_code", args, ToolContext())
        self.assertEqual(internal_args["_source_code"], "xlbl")


class B11BlockedToolLoopTests(unittest.IsolatedAsyncioTestCase):
    policy_message = b6_policy.B7BlockedToolLoopTests.policy_message
    orchestrator = b6_policy.B7BlockedToolLoopTests.orchestrator

    async def test_duplicate_blocked_shift_stops_on_second_attempt(self):
        agent, create, execute = self.orchestrator([
            ("keytao_shift_phrase_code", {"word": WORD, "target_code": "xlblu"}),
        ] * 4)
        termination = {}
        reply = await agent._run_loop(
            f"顺延“{WORD}”",
            AgentRequestContext(platform="qq", user_id="b11-loop", mutations_allowed=True),
            termination_state=termination,
        )
        self.assertEqual(reply, self.policy_message)
        self.assertEqual(termination.get("reason"), "policy_tool_runaway")
        self.assertEqual((create.await_count, execute.await_count), (2, 1))

    async def test_text_follow_up_cannot_reopen_tool_loop(self):
        agent, create, execute = self.orchestrator([
            ("keytao_shift_phrase_code", {"word": WORD, "target_code": "xlblu"}),
        ] * 3, [{"success": False, "policyBlocked": True, "requiresTextFollowUp": True,
                 "blockReason": "binding_incomplete", "message": self.policy_message}])
        termination = {}
        reply = await agent._run_loop(
            f"顺延“{WORD}”", AgentRequestContext(platform="qq", user_id="b11-loop", mutations_allowed=True),
            termination_state=termination,
        )
        self.assertEqual(reply, self.policy_message)
        self.assertEqual(termination["reason"], "policy_tool_runaway")
        self.assertEqual((create.await_count, execute.await_count), (3, 1))
        request = create.await_args_list[-1].kwargs
        self.assertEqual(request["tool_choice"], "none")
        self.assertFalse(request.get("tools"))


draft_tools = harness._draft_tools
real_batch_remove = draft_tools.keytao_batch_remove_draft_items


class B11SourceShiftTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.live = []
        self.snapshot = {
            "success": True, "batchId": "b11-draft", "contentVersion": 7,
            "status": "Draft", "items": [{
                "id": 42, "action": "Create", "word": WORD, "code": "xlbl",
                "type": "Phrase", "weight": 100, "remark": "Original review",
                "needsManualReview": True,
            }],
        }
        self.cleanup_result = None
        self.after_append = None
        stack = self.enterContext(ExitStack())
        stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")))
        stack.enter_context(patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Offline only")))
        stack.enter_context(patch.object(draft_tools, "_fetch_encode_candidates", AsyncMock(return_value={
            "success": True, "candidateCodes": ["xlbl", "xlblu", "xlbluv"],
        })))
        stack.enter_context(patch.object(draft_tools, "_lookup_words_raw", AsyncMock(side_effect=lambda words: {
            "success": True, "results": [{"word": word, "phrases": copy.deepcopy([
                row for row in self.live if row["word"] == word
            ])} for word in words],
        })))
        stack.enter_context(patch.object(draft_tools, "_lookup_codes_raw", AsyncMock(side_effect=lambda codes: {
            "success": True, "results": [{"code": code, "phrases": copy.deepcopy([
                row for row in self.live if row["code"] == code
            ])} for code in codes],
        })))
        stack.enter_context(patch.object(draft_tools, "keytao_list_draft_items", AsyncMock(
            side_effect=lambda *args, **kwargs: copy.deepcopy(self.snapshot),
        )))
        self.append = stack.enter_context(patch.object(draft_tools, "_keytao_strict_batch_add_to_draft",
                                                    AsyncMock(side_effect=self.strict)))
        self.remove = stack.enter_context(patch.object(draft_tools, "keytao_batch_remove_draft_items",
                                                    AsyncMock(side_effect=self.cleanup)))

    async def strict(self, platform, actor, items, **kwargs):
        if not kwargs.get("confirmed"):
            return {"success": False, "requiresConfirmation": True,
                    "warningDigest": "a" * 64, "warnings": []}
        self.assertEqual(kwargs["expected_content_version"], self.snapshot["contentVersion"])
        for item in items:
            self.snapshot["items"].append({"id": len(self.snapshot["items"]) + 100,
                                           "weight": 100, **copy.deepcopy(item)})
        self.snapshot["contentVersion"] += 1
        result = {"success": True, "batchId": self.snapshot["batchId"],
                  "contentVersion": self.snapshot["contentVersion"],
                  "successCount": len(items), "message": "Appended"}
        if self.after_append:
            self.after_append(self.snapshot)
        return result

    async def cleanup(self, platform, actor, ids, **kwargs):
        if self.cleanup_result is not None:
            return self.cleanup_result
        self.assertEqual(ids, [42])
        self.assertEqual(kwargs["expected_content_version"], 8)
        self.assertEqual(kwargs["expected_targets"], [{
            "id": 42, "word": WORD, "code": "xlbl", "action": "Create", "type": "Phrase",
        }])
        self.assertEqual(len(kwargs["expected_target_digest"]), 64)
        self.snapshot["items"] = [row for row in self.snapshot["items"] if row["id"] not in ids]
        self.snapshot["contentVersion"] += 1
        return {"success": True, "batchId": self.snapshot["batchId"],
                "contentVersion": self.snapshot["contentVersion"], "successCount": len(ids),
                "failedCount": 0, "draftItems": copy.deepcopy(self.snapshot["items"])}

    async def preview(self, source="xlbl", target="xlblu", **kwargs):
        return await draft_tools.keytao_shift_phrase_code("qq", "b11-owner", WORD, target,
                                                    _source_code=source, target_type="Phrase", **kwargs)

    async def confirm(self, preview, source="xlbl", target="xlblu"):
        return await self.preview(source, target, confirmed_plan_digest=preview["planDigest"],
                                  batch_id=preview["batchId"],
                                  expected_content_version=preview["contentVersion"],
                                  expected_warning_digest=preview["warningDigest"])

    async def test_draft_only_moves_after_append_with_exact_cleanup(self):
        preview = await self.preview()
        self.assertTrue(preview.get("requiresConfirmation"), preview)
        self.assertEqual([item["action"] for item in preview["shiftPlan"]["items"]], ["Create"])
        self.assertEqual(preview["shiftPlan"]["sourceDraftItem"]["id"], 42)
        result = await self.confirm(preview)
        self.assertTrue(result.get("success"), result)
        self.assertEqual([(row["word"], row["code"]) for row in self.snapshot["items"]], [(WORD, "xlblu")])
        self.assertTrue(self.snapshot["items"][0]["needsManualReview"])
        self.assertEqual(self.snapshot["items"][0]["remark"], "Original review")
        self.assertEqual(result["contentVersion"], 9)
        self.assertEqual(self.remove.await_count, 1)

    async def test_live_multi_code_uses_selected_source(self):
        self.snapshot["items"] = []
        self.live = [{"word": WORD, "code": code, "type": "Phrase", "weight": 100}
                     for code in ("xlbl", "xlblu")]
        preview = await self.preview("xlblu", "xlbluv")
        self.assertTrue(preview.get("requiresConfirmation"), preview)
        self.assertEqual([item["code"] for item in preview["shiftPlan"]["items"]
                          if item["action"] == "Delete"], ["xlblu"])
        self.remove.assert_not_awaited()

    async def test_source_drift_invalidates_replay_before_append(self):
        preview = await self.preview()
        self.snapshot["items"][0]["remark"] = "Changed review"
        result = await self.confirm(preview)
        self.assertFalse(result.get("success"), result)
        self.assertTrue(result.get("staleConfirmation"), result)
        self.assertEqual(self.append.await_count, 1)
        self.remove.assert_not_awaited()

    async def test_post_append_version_drift_keeps_old_source(self):
        preview = await self.preview()
        self.after_append = lambda snapshot: snapshot.update(contentVersion=9)
        result = await self.confirm(preview)
        self.assertFalse(result.get("success"), result)
        self.assertTrue(result.get("partialWrite"), result)
        self.remove.assert_not_awaited()
        self.assertEqual([row["code"] for row in self.snapshot["items"]], ["xlbl", "xlblu"])

    async def test_uncertain_cleanup_returns_partial_without_retry(self):
        preview = await self.preview()
        self.cleanup_result = {"success": False, "uncertain": True, "message": "Timed out"}
        result = await self.confirm(preview)
        self.assertFalse(result.get("success"), result)
        self.assertTrue(result.get("partialWrite"), result)
        self.assertTrue(result.get("uncertain"), result)
        self.assertEqual(self.remove.await_count, 1)
        self.assertEqual([row["code"] for row in self.snapshot["items"]], ["xlbl", "xlblu"])

    async def test_selected_draft_code_preserves_other_codes_and_types(self):
        self.snapshot["items"].extend([
            {"id": 43, "action": "Create", "word": WORD, "code": "other", "type": "Phrase", "weight": 100},
            {"id": 44, "action": "Create", "word": WORD, "code": "xlbl", "type": "CSS", "weight": 100},
        ])
        result = await self.confirm(await self.preview())
        self.assertTrue(result.get("success"), result)
        self.assertEqual([(row["code"], row["type"]) for row in self.snapshot["items"]],
                         [("other", "Phrase"), ("xlbl", "CSS"), ("xlblu", "Phrase")])

    async def test_source_draft_id_may_be_numeric_text(self):
        self.snapshot["items"][0]["id"] = "42"
        self.assertTrue((await self.preview()).get("requiresConfirmation"))

    async def test_deleted_live_source_is_not_effective_source(self):
        self.live = [{"word": WORD, "code": "xlbl", "type": "Phrase", "weight": 100}]
        self.snapshot["items"][0]["action"] = "Delete"
        result = await self.preview()
        self.assertFalse(result.get("success"), result)
        self.append.assert_not_awaited()
        self.remove.assert_not_awaited()

    async def test_source_live_type_is_bound(self):
        self.live = [{"word": WORD, "code": "xlbl", "type": "CSS", "weight": 100}]
        self.snapshot["items"] = []
        result = await self.preview()
        self.assertFalse(result.get("success"), result)
        self.append.assert_not_awaited()

    async def test_live_source_weight_drift_invalidates_digest(self):
        self.snapshot["items"] = []
        self.live = [{"word": WORD, "code": "xlbl", "type": "Phrase", "weight": 100}]
        preview = await self.preview()
        self.live[0]["weight"] = 101
        result = await self.confirm(preview)
        self.assertTrue(result.get("staleConfirmation"), result)
        self.assertEqual(self.append.await_count, 1)

    async def test_append_uncertain_never_deletes_source(self):
        preview = await self.preview()
        self.append.side_effect = None
        self.append.return_value = {"success": False, "uncertain": True, "message": "Timed out"}
        result = await self.confirm(preview)
        self.assertFalse(result.get("success"), result)
        self.assertTrue(result.get("uncertain"), result)
        self.assertEqual(result["sourceCleanup"], "not_attempted")
        self.remove.assert_not_awaited()
        self.assertEqual([row["code"] for row in self.snapshot["items"]], ["xlbl"])

    async def test_changed_destination_review_never_deletes_source(self):
        preview = await self.preview()
        self.after_append = lambda snapshot: snapshot["items"][-1].update(needsManualReview=False)
        result = await self.confirm(preview)
        self.assertTrue(result.get("partialWrite"), result)
        self.remove.assert_not_awaited()

    async def test_success_does_not_change_bound_plan(self):
        preview = await self.preview()
        result = await self.confirm(preview)
        self.assertTrue(result.get("success"), result)
        self.assertEqual(result["shiftPlan"], preview["shiftPlan"])

    async def test_real_cleanup_tool_sends_exact_cas_contract(self):
        from keytao_bot.utils.draft_mutation_store import DraftMutationClaimStore

        owner = self
        requests = []

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def request(self, method, url, **kwargs):
                import json
                payload = json.loads(kwargs["content"])
                requests.append((method, url, payload))
                result = await owner.cleanup("qq", "b11-owner", payload["ids"],
                                             expected_content_version=payload["expectedContentVersion"],
                                             expected_targets=payload["expectedTargets"],
                                             expected_target_digest="x" * 64)
                return SimpleNamespace(status_code=200, json=lambda: result)

        async def request_once(callback, **kwargs):
            return await callback()

        with tempfile.TemporaryDirectory() as directory:
            claims = DraftMutationClaimStore(directory + "/claims.db")
            with (
                patch.object(draft_tools, "keytao_batch_remove_draft_items", real_batch_remove),
                patch.object(draft_tools, "_draft_mutation_claims", return_value=claims),
                patch.object(draft_tools, "get_bot_token", return_value="offline"),
                patch.object(draft_tools, "get_bot_headers", return_value={}),
                patch.object(draft_tools, "get_keytao_url", return_value="https://offline.invalid"),
                patch.object(draft_tools.httpx, "AsyncClient", side_effect=lambda **kwargs: Client(), create=True),
                patch.object(draft_tools.http_client, "request_with_retries", side_effect=request_once),
            ):
                preview = await self.preview()
                result = await self.confirm(preview)
            self.assertTrue(result.get("success"), result)
            self.assertEqual(len(requests), 1)
            self.assertEqual(requests[0][0], "DELETE")
            self.assertEqual(requests[0][2]["expectedContentVersion"], 8)
            self.assertEqual(requests[0][2]["expectedTargets"][0]["id"], 42)



class B11ChangeSourceTests(unittest.IsolatedAsyncioTestCase):
    strict = B11SourceShiftTests.strict
    preview = B11SourceShiftTests.preview
    confirm = B11SourceShiftTests.confirm

    def setUp(self):
        B11SourceShiftTests.setUp(self)
        self.snapshot["items"][0].update(action="Change", oldWord="旧词")
        self.live = [{"word": "旧词", "code": "xlbl", "type": "Phrase", "weight": 100,
                      "remark": "Original live remark"}]

    async def cleanup(self, platform, actor, ids, **kwargs):
        self.assertEqual(ids, [42])
        self.assertEqual(kwargs["expected_content_version"], 8)
        self.assertEqual(kwargs["expected_targets"], [{
            "id": 42, "word": WORD, "code": "xlbl", "action": "Change", "type": "Phrase",
        }])
        self.snapshot["items"] = [row for row in self.snapshot["items"] if row["id"] not in ids]
        self.snapshot["contentVersion"] += 1
        return {"success": True, "batchId": self.snapshot["batchId"],
                "contentVersion": self.snapshot["contentVersion"], "successCount": 1,
                "failedCount": 0, "draftItems": copy.deepcopy(self.snapshot["items"])}

    async def test_change_source_moves_replacement_and_deletes_original_live_word(self):
        preview = await self.preview()
        self.assertTrue(preview.get("requiresConfirmation"), preview)
        self.assertEqual(preview["shiftPlan"]["sourceOriginalPhrase"], self.live[0])
        self.assertEqual([(row["action"], row["word"], row["code"])
                          for row in preview["shiftPlan"]["items"]],
                         [("Delete", "旧词", "xlbl"), ("Create", WORD, "xlblu")])
        result = await self.confirm(preview)
        self.assertTrue(result.get("success"), result)
        self.assertEqual([(row["action"], row["word"], row["code"])
                          for row in self.snapshot["items"]],
                         [("Delete", "旧词", "xlbl"), ("Create", WORD, "xlblu")])
        self.assertEqual(self.remove.await_count, 1)

    async def test_change_source_requires_old_word(self):
        self.snapshot["items"][0].pop("oldWord")
        result = await self.preview()
        self.assertFalse(result.get("success"), result)
        self.assertFalse(result.get("requiresConfirmation"), result)
        self.append.assert_not_awaited()
        self.remove.assert_not_awaited()

    async def test_change_source_requires_unique_original_live_record(self):
        original = copy.deepcopy(self.live[0])
        for rows in ([], [original, dict(original)], [{**original, "type": "CSS"}]):
            with self.subTest(rows=rows):
                self.live = copy.deepcopy(rows)
                result = await self.preview()
                self.assertFalse(result.get("success"), result)
                self.assertFalse(result.get("requiresConfirmation"), result)
        self.append.assert_not_awaited()
        self.remove.assert_not_awaited()

    async def test_change_original_live_snapshot_drift_invalidates_replay(self):
        preview = await self.preview()
        self.assertTrue(preview.get("requiresConfirmation"), preview)
        self.live[0]["remark"] = "Changed live remark"
        result = await self.confirm(preview)
        self.assertTrue(result.get("staleConfirmation"), result)
        self.assertEqual(self.append.await_count, 1)
        self.remove.assert_not_awaited()

    async def test_change_original_live_disappearance_invalidates_replay(self):
        preview = await self.preview()
        self.assertTrue(preview.get("requiresConfirmation"), preview)
        self.live.clear()
        result = await self.confirm(preview)
        self.assertFalse(result.get("success"), result)
        self.assertEqual(self.append.await_count, 1)
        self.remove.assert_not_awaited()

    async def test_change_source_occupied_target_reuses_freed_original_slot(self):
        self.live.append({"word": "占位词", "code": "xlblu", "type": "Phrase", "weight": 100})

        async def encode(word, requested_code=None):
            self.assertIn(word, (WORD, "占位词"))
            return {"success": True, "candidateCodes": ["xlbl", "xlblu", "xlbluv"]
                    if word == WORD else ["xlblu", "xlbl", "xlbluv"]}

        with patch.object(draft_tools, "_fetch_encode_candidates", side_effect=encode):
            preview = await self.preview()
            self.assertTrue(preview.get("requiresConfirmation"), preview)
            self.assertEqual([(row["word"], row["fromCode"], row["toCode"])
                              for row in preview["shiftPlan"]["shifted"]], [("占位词", "xlblu", "xlbl")])
            result = await self.confirm(preview)
        self.assertTrue(result.get("success"), result)
        self.assertEqual([(row["action"], row["word"], row["code"])
                          for row in self.snapshot["items"] if row["action"] == "Create"],
                         [("Create", WORD, "xlblu"), ("Create", "占位词", "xlbl")])

    async def test_change_source_does_not_globally_ignore_old_word(self):
        self.live.append({"word": "旧词", "code": "xlblu", "type": "Phrase", "weight": 100})

        async def encode(word, requested_code=None):
            return {"success": True, "candidateCodes": ["xlbl", "xlblu", "xlbluv"]}

        with patch.object(draft_tools, "_fetch_encode_candidates", side_effect=encode):
            preview = await self.preview()
        self.assertTrue(preview.get("requiresConfirmation"), preview)
        self.assertEqual([(row["word"], row["fromCode"], row["toCode"])
                          for row in preview["shiftPlan"]["shifted"]], [("旧词", "xlblu", "xlbluv")])


if __name__ == "__main__":
    unittest.main()
