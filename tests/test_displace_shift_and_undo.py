"""Topic regressions: displace shift and undo."""

import unittest as _stage_t_unittest
from tests.helpers import (
    AMBIGUOUS_SHIFT_OPERANDS,
    AgentOrchestrator,
    AgentRequestContext,
    AgentRuntimeConfig,
    AsyncMock,
    B11ChangeSourceTests_cleanup,
    B11ChangeSourceTests_setUp,
    B11ShiftTests_assert_shift,
    B11ShiftTests_seed_receipt,
    B11ShiftTests_setUp,
    B11ShiftTests_turn,
    B11SourceShiftTests_cleanup,
    B11SourceShiftTests_confirm,
    B11SourceShiftTests_preview,
    B11SourceShiftTests_setUp,
    B11SourceShiftTests_strict,
    B19PartialReplyBindingTests_created,
    B19PartialReplyBindingTests_matcher_finish,
    B19PartialReplyBindingTests_matcher_send,
    B19PartialReplyBindingTests_turn,
    B24DisplaceTests_finish_shift,
    B24DisplaceTests_seed,
    B24DisplaceTests_setUp,
    B24DisplaceTests_tearDown,
    B25SessionTests_finish_replacement,
    B25SessionTests_greeting,
    B25SessionTests_history_for,
    B25SessionTests_partial_session,
    B25SessionTests_remember,
    B25SessionTests_seed,
    B25SessionTests_setUp,
    B25SessionTests_tearDown,
    B25SessionTests_turn,
    B7BlockedToolLoopTests_orchestrator,
    B7BlockedToolLoopTests_policy_message,
    BridgeTests_call,
    BridgeTests_setUp,
    CHAIN,
    CompletedUndoTests_cancel,
    CompletedUndoTests_setUp,
    CompletedUndoTests_tearDown,
    CompletedUndoTests_write,
    DraftMutationClaimStore,
    DraftServer,
    ExplicitDestinationTests_encode,
    ExplicitDestinationTests_invoke,
    ExplicitDestinationTests_lookup_codes,
    ExplicitDestinationTests_lookup_words,
    ExplicitDestinationTests_setUp,
    ExplicitDestinationTests_strict,
    GenericEvictionTests_run_command,
    GenericProposalBridgeTests_assert_ticket,
    GenericProposalBridgeTests_call,
    GenericProposalBridgeTests_setUp,
    MemoryConversationStateStore,
    MoveToCodeGrammarTests_assert_bound_move,
    MoveToCodeGrammarTests_context,
    PendingToolConfirm,
    REAL_FINISH,
    SimpleNamespace,
    ToolContext,
    ToolExecutor,
    asyncio,
    b11_shift_WORD,
    b11_shift_draft_tools,
    b17_rest_recommended_commands,
    b19_partial_reply_binding_chat,
    b24_displace_CODES,
    b24_displace_OCCUPANT,
    b24_displace_WORD,
    b25_session_CODES,
    b25_session_FACTS,
    b25_session_OCCUPANT,
    b25_session_WORD,
    b25_session_WORDS,
    b25_session_commands,
    completed_draft_undo,
    copy,
    explicit_complete_add_item,
    grammar,
    harness,
    hashlib,
    json,
    parse_dictionary_recode_command,
    parse_eviction_modified_add,
    parse_explicit_entry_code_request,
    parse_reviewed_shift_words,
    parse_standalone_shift,
    patch,
    preview,
    raw_operation,
    real_batch_remove,
    render_missing_option_ticket,
    replace,
    review,
    reviewed,
    reviewed_destination,
    s56_undo_chat,
    s56_undo_row,
    s57_reorder_chat,
    s57_reorder_commands,
    s57_reorder_draft_tools,
    server_warning_pending_state,
    server_warning_ticket_is_complete,
    snapshot,
    tempfile,
    ticket,
    time,
    understand_message,
    unittest,
    validate_operation,
)


# Source: test_b11_shift.py
class B11ShiftTests(unittest.IsolatedAsyncioTestCase):
    setUp = B11ShiftTests_setUp

    turn = B11ShiftTests_turn

    assert_shift = B11ShiftTests_assert_shift

    async def test_standalone_shift_variants_use_next_chain_code(self):
        for message in (f"顺延“{b11_shift_WORD}”", f"把「{b11_shift_WORD}」顺延", f'"{b11_shift_WORD}"顺延', f"顺延『{b11_shift_WORD}』一位"):
            with self.subTest(message=message):
                self.replay.writes.clear()
                await self.turn(message, frame="question")
                self.assert_shift("xlblu")
                self.assertIsNone(self.replay.store.get_record(self.replay.author))

    async def test_explicit_shift_code_uses_move_route(self):
        await self.turn(f"顺延“{b11_shift_WORD}”到xbluv", "move", ("xbluv",))
        self.assert_shift("xbluv")

    async def test_unbound_conversational_shifts_never_write_or_override_cancel(self):
        self.replay.entries[:] = [
            {"word": word, "code": code, "type": "Phrase", "weight": 100}
            for word, code in (("取消", "qxxl"), ("确认", "qrrf"), (b11_shift_WORD, "xlbl"))
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
            "message": f"{b11_shift_WORD}顺延", "to_me": True,
            "quote_target": f"「{b11_shift_WORD}」目前在 xlbl。",
        }, raw_operation(op="other", operands=[], codes=[], frame="question"))
        self.assert_shift("xlblu")

    async def test_unquoted_shift_without_reference_never_writes_even_with_keeper(self):
        for has_keeper in (False, True):
            self.replay.entries[:] = [{"word": b11_shift_WORD, "code": "xlbl", "type": "Phrase", "weight": 100}]
            if has_keeper:
                self.replay.entries.append({"word": "保留词", "code": "xlbl", "type": "Phrase", "weight": 100})
            for message in (f"顺延{b11_shift_WORD}", f"{b11_shift_WORD}顺延", f"把{b11_shift_WORD}顺延", f"顺延{b11_shift_WORD}一位"):
                with self.subTest(message=message, has_keeper=has_keeper):
                    self.replay.store.delete(self.replay.author)
                    self.replay.history[:] = [{"role": "assistant", "content": f"「{b11_shift_WORD}」目前在 xlbl。"}]
                    self.replay.calls.clear()
                    await self.replay.turn({"message": message, "to_me": True}, raw_operation(
                        op="other", operands=[], codes=[], frame="question",
                    ))
                    self.assertEqual(self.replay.writes, [])
                    previews = [args for name, args in self.replay.calls if name == "keytao_shift_phrase_code"]
                    self.assertEqual(bool(previews), has_keeper and message.endswith("顺延"))
                    self.assertFalse(any(args.get("confirmed_plan_digest") for args in previews))

    async def test_bound_shift_rejects_extra_text_or_different_word(self):
        for message in (
            f"确认{b11_shift_WORD}顺延", f"先别{b11_shift_WORD}顺延", f"怎么顺延{b11_shift_WORD}", f"顺延{b11_shift_WORD}就行",
            f"{b11_shift_WORD}顺延好了", f"顺延“{b11_shift_WORD}”试试", f"确认顺延“{b11_shift_WORD}”", "取消顺延",
        ):
            with self.subTest(message=message):
                self.replay.calls.clear()
                await self.replay.turn({
                    "message": message, "to_me": True, "quote_target": f"「{b11_shift_WORD}」目前在 xlbl。",
                }, raw_operation(op="other", operands=[], codes=[], frame="question"))
                self.assertEqual(self.replay.writes, [])
                self.assertFalse(any(name == "keytao_shift_phrase_code" for name, args in self.replay.calls))

    async def test_unquoted_shift_binds_only_own_live_ticket(self):
        state = harness.PendingAddWord(b11_shift_WORD, "xlbl", [("xlbl", False)], needs_manual_review=False)
        self.assertTrue(self.replay.store.set(self.replay.author, state))
        await self.turn(f"{b11_shift_WORD}顺延", frame="question")
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
        for quote in ("回复「确认」执行，或「取消」。", f"「{b11_shift_WORD}」目前在 xlbl。\n回复「确认」执行，或「取消」。"):
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
            self.assertTrue(self.replay.store.set(owner, harness.PendingAddWord(b11_shift_WORD, "xlbl", [])))
            record = self.replay.store.get_record(owner)
            self.assertEqual(self.replay.commands._standalone_shift_bound_words(record, address),
                             (b11_shift_WORD,) if owner == address else ())
        record = self.replay.store.get_record(address)
        record.execution_id = "in-flight"
        self.assertEqual(self.replay.commands._standalone_shift_bound_words(record, address), ())
        record.execution_id = ""
        record.expires_at = 0
        self.assertIsNone(self.replay.store.get_record(address))
        for is_reply, is_to_bot in ((False, True), (True, False), (True, True)):
            reply = harness.ReplyReferenceInfo(is_reply=is_reply, is_to_bot=is_to_bot, text=f"「{b11_shift_WORD}」目前在 xlbl。")
            self.assertEqual(self.replay.commands._standalone_shift_bound_words(None, address, reply),
                             (b11_shift_WORD,) if is_reply and is_to_bot else ())

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
        for word in (*AMBIGUOUS_SHIFT_OPERANDS, "我送给", "要领", "可能性", "哪吒", "哪怕", "几何", "寥寥无几", b11_shift_WORD):
            for message in (f"顺延“{word}”", f"把「{word}」顺延", f'"{word}"顺延'):
                with self.subTest(message=message):
                    self.assertEqual(parse_standalone_shift(message).word, word)
        for word in ("我送给", "要领", "可能性", "哪吒", "哪怕", "几何", "寥寥无几", b11_shift_WORD):
            for message in (f"顺延{word}", f"把{word}顺延", f"{word}顺延", f"顺延{word}一位"):
                with self.subTest(message=message):
                    self.assertIsNone(parse_standalone_shift(message))
                    self.assertEqual(parse_standalone_shift(message, known_words=(word,)).word, word)

    def test_shift_binding_consumes_only_the_complete_message(self):
        for message in (f"确认顺延“{b11_shift_WORD}”", f"先别{b11_shift_WORD}顺延", f"顺延{b11_shift_WORD}就行",
                        f"顺延“{b11_shift_WORD}” @喵喵", f"顺延喵喵 “{b11_shift_WORD}”"):
            with self.subTest(message=message):
                self.assertIsNone(parse_standalone_shift(message, known_words=(b11_shift_WORD,)))
        self.assertEqual(parse_standalone_shift(f"＠喵喵 把『{b11_shift_WORD}』顺延一位到xlblu！").word, b11_shift_WORD)
        self.assertEqual(parse_standalone_shift("顺延前一位", known_words=("前一位",)).word, "前一位")

    async def test_draft_only_word_can_shift(self):
        self.replay.entries.clear()
        self.replay.draft = [{"id": 1, "action": "Create", "word": b11_shift_WORD, "code": "xlbl", "type": "Phrase", "weight": 100}]
        await self.turn(f"顺延“{b11_shift_WORD}”")
        self.assert_shift("xlblu")

    async def test_multi_code_shift_has_bound_numbered_options_and_cancel(self):
        self.replay.entries.append({"word": b11_shift_WORD, "code": "xlblu", "type": "Phrase", "weight": 100})
        await self.turn(f"顺延“{b11_shift_WORD}”")
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
        await self.turn(f"顺延“{b11_shift_WORD}”到1号", "move", ("1",))
        self.assertIn("编码", self.replay.last_response)
        self.assertEqual(self.replay.writes, [])
        self.general.assert_not_awaited()
        self.replay.client.create.assert_not_awaited()

    async def test_multi_code_selection_keeps_shift_route_and_source(self):
        self.replay.entries.append({"word": b11_shift_WORD, "code": "xlblu", "type": "Phrase", "weight": 100})
        for answer in ("2", "到2号", f"顺延{b11_shift_WORD}到2号"):
            with self.subTest(answer=answer):
                self.replay.writes.clear()
                await self.turn(f"顺延“{b11_shift_WORD}”")
                self.assertEqual(self.replay.writes, [])
                await self.turn(answer, "select_candidate", refers_to_pending=True)
                self.assert_shift("xlbluv")
                self.assertEqual(self.replay.writes[0][1]["_source_code"], "xlblu")

    async def test_multi_code_explicit_target_can_select_existing_target_without_write(self):
        self.replay.entries.append({"word": b11_shift_WORD, "code": "xlblu", "type": "Phrase", "weight": 100})
        await self.turn(f"顺延“{b11_shift_WORD}”到xlblu", "move", ("xlblu",))
        self.assertRegex(self.replay.last_response, r"(?m)^2[.、．)]")
        self.assertIsNotNone(self.replay.store.get_record(self.replay.author))
        await self.turn("2", "select_candidate", refers_to_pending=True)
        self.assertEqual(self.replay.writes, [])
        self.assertIn("已在 xlblu", self.replay.last_response)
        self.general.assert_not_awaited()

    async def test_occupied_next_code_keeps_full_chain_confirmation(self):
        self.replay.entries.append({"word": "别的词", "code": "xlblu", "type": "Phrase", "weight": 100})
        self.replay.shifted = [{"word": "别的词", "fromCode": "xlblu", "toCode": "bdco"}]
        await self.turn(f"顺延“{b11_shift_WORD}”")
        self.assertEqual(self.replay.writes, [])
        self.assertIsNotNone(self.replay.store.get_record(self.replay.author))
        self.assertIn("确认", self.replay.last_response)
        self.general.assert_not_awaited()

    async def test_ordinal_selects_only_live_bound_numbered_candidates(self):
        state = harness.PendingAddWord(
            b11_shift_WORD, "xlblu", [("xlblu", False), ("xbluv", False)],
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
            await self.turn(f"＠ 小键 修改编码“{b11_shift_WORD}”为xlblu", "recode", ("xlblu",))
        self.assertEqual(len(self.replay.writes), 1, self.replay.last_response)
        self.general.assert_not_awaited()

    async def test_mentions_are_removed_before_recode_validation(self):
        for prefix in ("＠喵喵 ", "@喵喵 ", "＠ 喵喵", "@喵喵"):
            with self.subTest(prefix=prefix):
                self.replay.writes.clear()
                await self.turn(prefix + f"修改编码“{b11_shift_WORD}”为xlblu", "recode", ("xlblu",))
                self.assertEqual(len(self.replay.writes), 1, self.replay.last_response)
                self.assertEqual([(row["action"], row["word"], row["code"])
                                  for row in self.replay.writes[0][1]["items"]],
                                 [("Delete", b11_shift_WORD, "xlbl"), ("Create", b11_shift_WORD, "xlblu")])
        self.general.assert_not_awaited()

    async def test_bare_word_code_offers_recode_or_add_and_executes_choice(self):
        for answer in ("改码", "加入"):
            with self.subTest(answer=answer):
                self.replay.writes.clear()
                self.replay.draft.clear()
                await self.turn(f"{b11_shift_WORD} xlblu", "query", ("xlblu",))
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
        await self.turn(f"{b11_shift_WORD} xlblu", "query", ("xlblu",), frame="none")
        self.assertEqual(len(self.replay.writes), 1, self.replay.last_response)
        self.assertIn(("keytao_prepare_reviewed_add", {"word": b11_shift_WORD, "requested_code": "xlblu"}), self.replay.calls)
        self.general.assert_not_awaited()

    async def test_literal_word_code_ignores_model_frame_and_operation(self):
        for op in ("query", "other", "help", "recode", "move", "add"):
            with self.subTest(op=op):
                self.replay.store.delete(self.replay.author)
                await self.turn(f"{b11_shift_WORD} xlblu", op, ("xlblu",), frame="question")
                self.assertIn("回复「改码」", self.replay.last_response)
                self.assertEqual(self.replay.writes, [])
        self.general.assert_not_awaited()

    async def test_unnumbered_recode_offer_rejects_ordinal_without_consuming(self):
        await self.turn(f"{b11_shift_WORD} xlblu", "query", ("xlblu",), frame="none")
        record = self.replay.store.get_record(self.replay.author)
        await self.turn("到1号", "select_candidate", refers_to_pending=True)
        self.assertEqual(self.replay.writes, [])
        self.assertIs(self.replay.store.get_record(self.replay.author), record)
        self.assertIn("编码", self.replay.last_response)

    async def test_bare_word_code_occupied_code_keeps_explicit_add_route(self):
        self.replay.entries.append({"word": "别的词", "code": "xlblu", "type": "Phrase", "weight": 100})
        await self.turn(f"{b11_shift_WORD} xlblu", "query", ("xlblu",), frame="none")
        self.assertIn(("keytao_prepare_reviewed_add", {"word": b11_shift_WORD, "requested_code": "xlblu"}), self.replay.calls)
        self.assertNotIn("目前在 xlbl", self.replay.last_response)
        self.general.assert_not_awaited()

    async def test_bare_draft_word_recode_keeps_bound_source(self):
        self.replay.entries.clear()
        self.replay.draft = [{"id": 1, "action": "Create", "word": b11_shift_WORD, "code": "xlbl", "type": "Phrase", "weight": 100}]
        await self.turn(f"{b11_shift_WORD} xlblu", "query", ("xlblu",), frame="none")
        await self.turn("改码", "confirm", refers_to_pending=True)
        self.assert_shift("xlblu")
        self.assertEqual(self.replay.writes[0][1]["_source_code"], "xlbl")

    seed_receipt = B11ShiftTests_seed_receipt

    async def test_intervening_reply_prevents_unrelated_completed_summary(self):
        self.seed_receipt()
        await self.turn(b11_shift_WORD, "query", frame="none")
        self.assertIn("xlbl", self.replay.last_response)
        await self.turn("确认", "confirm", refers_to_pending=True)
        self.assertEqual(self.replay.last_response, "当前没有待确认的操作。")
        self.general.assert_not_awaited()

    async def test_last_receipt_summary_is_next_reply_only(self):
        self.seed_receipt()
        await self.turn("确认", "confirm", refers_to_pending=True)
        self.assertEqual(self.replay.last_response, "当前没有待确认的操作（上一条已执行）。")
        await self.turn("确认", "confirm", refers_to_pending=True)
        self.assertEqual(self.replay.last_response, "当前没有待确认的操作。")

    async def test_turn_without_reply_does_not_hide_last_receipt(self):
        self.seed_receipt()
        completed_draft_undo.start_operation_turn(self.replay.author)
        await self.turn("确认", "confirm", refers_to_pending=True)
        self.assertEqual(self.replay.last_response, "当前没有待确认的操作（上一条已执行）。")

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
        for message in (f"不要顺延{b11_shift_WORD}", f"他说{b11_shift_WORD}顺延", f"顺延{b11_shift_WORD}？", f'"顺延{b11_shift_WORD}"'):
            self.assertIsNone(parse_standalone_shift(message))
        self.assertIsNotNone(parse_dictionary_recode_command(f"删除 {b11_shift_WORD} xlbl；添加 {b11_shift_WORD} xlblu"))
        self.assertIsNone(parse_dictionary_recode_command(f"不要删除 {b11_shift_WORD} xlbl；添加 {b11_shift_WORD} xlblu"))

    def test_semantic_validator_recognizes_shift_even_if_model_says_other(self):
        request, violations = validate_operation(raw_operation(op="other", operands=[b11_shift_WORD], codes=[], frame="question"), f"顺延“{b11_shift_WORD}”", None)
        self.assertEqual((request.op, request.frame, request.operands), ("move", "direct", (b11_shift_WORD,)))
        self.assertEqual(violations, [])

    def test_only_internal_routes_can_bind_shift_source(self):
        executor = ToolExecutor(lambda name: None, frozenset())
        args = {"word": b11_shift_WORD, "target_code": "xlblu", "_source_code": "xlbl"}
        model_args = executor.canonicalize_arguments(
            "keytao_shift_phrase_code", args, ToolContext(current_message=f"顺延{b11_shift_WORD}到xlblu"),
        )
        self.assertNotIn("_source_code", model_args)
        internal_args = executor.canonicalize_arguments("keytao_shift_phrase_code", args, ToolContext())
        self.assertEqual(internal_args["_source_code"], "xlbl")

def set_up_change_header_swap_http(test):
    from urllib.parse import urlsplit
    from tests.helpers import (
        B18RealToolTests_setUp, OfflineResponse, REAL_REVIEW_READS, TOOLS,
        b19_partial_reply_binding_DraftHTTP,
    )

    B18RealToolTests_setUp(test)

    class SwapHTTP(b19_partial_reply_binding_DraftHTTP):
        def _response(self, method, url, kwargs):
            if urlsplit(url).path == "/api/phrases/encode":
                word = kwargs["params"]["word"]
                self.requests.append((method, "/api/phrases/encode", {"word": word}))
                return OfflineResponse({
                    "input": word, "type": "二字词", "codes": ["dsqt", "dsqtv", "dsqtvv"],
                    "altCodes": [], "flyKeyVariants": [], "phrasePinyins": ["da", "quan"],
                    "pronunciationSource": "zdic-phrase", "standardPronunciationStatus": "found",
                    "chars": [{"char": char, "pinyin": pinyin, "pinyins": [pinyin],
                               "phoneticCode": code, "shapeCode": "v"}
                              for char, pinyin, code in zip(word, ("da", "quan"), ("ds", "qt"))],
                })
            return super()._response(method, url, kwargs)

    test.http = SwapHTTP()
    test.http.entries = [
        {"word": "大权", "code": "dsqt", "type": "Phrase", "weight": 100},
        {"word": "大全", "code": "dsqtv", "type": "Phrase", "weight": 100},
    ]
    test.enterContext(patch.object(TOOLS.http_client, "_keytao_client", test.http))
    test.enterContext(patch.multiple(review, **REAL_REVIEW_READS))
    test.enterContext(patch.object(review.http_client, "guarded_fetch", AsyncMock(return_value=OfflineResponse({}, 404))))


# Source: test_b11_shift.py
class B11BlockedToolLoopTests(unittest.IsolatedAsyncioTestCase):
    policy_message = B7BlockedToolLoopTests_policy_message
    orchestrator = B7BlockedToolLoopTests_orchestrator

    async def assert_change_header_swap_writes_real_tools(self, message):
        from tests.helpers import TOOLS

        set_up_change_header_swap_http(self)
        token = completed_draft_undo.current_operation_capture.set(None)
        self.addCleanup(completed_draft_undo.current_operation_capture.reset, token)
        completed_draft_undo.start_operation_turn(self.replay.author)
        functions = {**harness._lookup_tools.TOOL_FUNCTIONS, **harness._review_tools.TOOL_FUNCTIONS,
                     **TOOLS.TOOL_FUNCTIONS}
        items = [
            {"action": "Change", "code": "dsqt", "old_word": "大权", "word": "大全", "type": "Phrase"},
            {"action": "Change", "code": "dsqtv", "old_word": "大全", "word": "大权", "type": "Phrase"},
        ]
        calls = [("keytao_lookup_by_codes_batch", {"codes": ["dsqt", "dsqtv"]}),
                 ("keytao_batch_add_to_draft", {"items": items})]
        client = harness._FakeClient([
            harness._FakeAIResponse("tool_calls", tool_calls=[SimpleNamespace(
                id=str(index), type="function", function=SimpleNamespace(
                    name=name, arguments=json.dumps(args, ensure_ascii=False),
                ),
            )]) for index, (name, args) in enumerate(calls)
        ] + [harness._FakeAIResponse("stop", content="已完成。")])

        class Skills(harness._FakeToolSkillsManager):
            def get_tools(self):
                return [tool for tool in (*harness._lookup_tools.TOOLS, *TOOLS.TOOLS)
                        if tool["function"]["name"] in dict(calls)]

        executor = ToolExecutor(functions.get, self.replay.chat._INJECT_PLATFORM_TOOLS)
        results = []
        real_call = executor.call

        async def capture_call(*args, **kwargs):
            result = await real_call(*args, **kwargs)
            results.append((args[0], result))
            return result

        executor.call = capture_call
        agent = AgentOrchestrator(
            client_factory=lambda: client, runtime=AgentRuntimeConfig("offline-model", 1000, 0.0, 1.0),
            skills_manager=Skills(), tool_executor=executor,
            state_store=self.replay.store, bind_help_text="Offline binding help", system_prompt_core="Offline prompt",
        )
        reply = await agent.run(message, AgentRequestContext(
            platform="qq", user_id=self.replay.author.actor_id, space_type="group", space_id="prod-replay",
            mutations_allowed=True, actor_is_bound=True,
        ))
        self.assertEqual([(row["action"], row.get("oldWord"), row["word"], row["code"])
                          for row in self.http.rows],
                         [("Change", "大权", "大全", "dsqt"), ("Change", "大全", "大权", "dsqtv")],
                         (reply, self.http.requests, results))
        self.assertEqual(len(self.http.writes), 1, reply)
        self.assertIn("已加入草稿", reply)
        for value in ("大权", "大全", "dsqt", "dsqtv", "offline-batch"):
            self.assertIn(value, reply, results)
        self.assertEqual(self.http.errors, [])

    async def test_change_header_swap_writes_both_items_with_real_tools(self):
        await self.assert_change_header_swap_writes_real_tools("喵喵 修改\ndsqt 大全\ndsqtv 大权")

    async def test_inline_change_header_swap_writes_both_items_with_real_tools(self):
        await self.assert_change_header_swap_writes_real_tools("喵喵，修改：dsqt 大全 dsqtv 大权")

    async def test_duplicate_blocked_shift_stops_on_second_attempt(self):
        agent, create, execute = self.orchestrator([
            ("keytao_shift_phrase_code", {"word": b11_shift_WORD, "target_code": "xlblu"}),
        ] * 4)
        termination = {}
        reply = await agent._run_loop(
            f"顺延“{b11_shift_WORD}”",
            AgentRequestContext(platform="qq", user_id="b11-loop", mutations_allowed=True),
            termination_state=termination,
        )
        self.assertEqual(reply, self.policy_message)
        self.assertEqual(termination.get("reason"), "policy_tool_runaway")
        self.assertEqual((create.await_count, execute.await_count), (2, 1))

    async def test_text_follow_up_cannot_reopen_tool_loop(self):
        agent, create, execute = self.orchestrator([
            ("keytao_shift_phrase_code", {"word": b11_shift_WORD, "target_code": "xlblu"}),
        ] * 3, [{"success": False, "policyBlocked": True, "requiresTextFollowUp": True,
                 "blockReason": "binding_incomplete", "message": self.policy_message}])
        termination = {}
        reply = await agent._run_loop(
            f"顺延“{b11_shift_WORD}”", AgentRequestContext(platform="qq", user_id="b11-loop", mutations_allowed=True),
            termination_state=termination,
        )
        self.assertEqual(reply, self.policy_message)
        self.assertEqual(termination["reason"], "policy_tool_runaway")
        self.assertEqual((create.await_count, execute.await_count), (3, 1))
        request = create.await_args_list[-1].kwargs
        self.assertEqual(request["tool_choice"], "none")
        self.assertFalse(request.get("tools"))

# Source: test_b11_shift.py
class B11SourceShiftTests(unittest.IsolatedAsyncioTestCase):
    setUp = B11SourceShiftTests_setUp

    strict = B11SourceShiftTests_strict

    cleanup = B11SourceShiftTests_cleanup

    preview = B11SourceShiftTests_preview

    confirm = B11SourceShiftTests_confirm

    async def test_draft_only_moves_after_append_with_exact_cleanup(self):
        preview = await self.preview()
        self.assertTrue(preview.get("requiresConfirmation"), preview)
        self.assertEqual([item["action"] for item in preview["shiftPlan"]["items"]], ["Create"])
        self.assertEqual(preview["shiftPlan"]["sourceDraftItem"]["id"], 42)
        result = await self.confirm(preview)
        self.assertTrue(result.get("success"), result)
        self.assertEqual([(row["word"], row["code"]) for row in self.snapshot["items"]], [(b11_shift_WORD, "xlblu")])
        self.assertTrue(self.snapshot["items"][0]["needsManualReview"])
        self.assertEqual(self.snapshot["items"][0]["remark"], "Original review")
        self.assertEqual(result["contentVersion"], 9)
        self.assertEqual(self.remove.await_count, 1)

    async def test_live_multi_code_uses_selected_source(self):
        self.snapshot["items"] = []
        self.live = [{"word": b11_shift_WORD, "code": code, "type": "Phrase", "weight": 100}
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
            {"id": 43, "action": "Create", "word": b11_shift_WORD, "code": "other", "type": "Phrase", "weight": 100},
            {"id": 44, "action": "Create", "word": b11_shift_WORD, "code": "xlbl", "type": "CSS", "weight": 100},
        ])
        result = await self.confirm(await self.preview())
        self.assertTrue(result.get("success"), result)
        self.assertEqual([(row["code"], row["type"]) for row in self.snapshot["items"]],
                         [("other", "Phrase"), ("xlbl", "CSS"), ("xlblu", "Phrase")])

    async def test_source_draft_id_may_be_numeric_text(self):
        self.snapshot["items"][0]["id"] = "42"
        self.assertTrue((await self.preview()).get("requiresConfirmation"))

    async def test_deleted_live_source_is_not_effective_source(self):
        self.live = [{"word": b11_shift_WORD, "code": "xlbl", "type": "Phrase", "weight": 100}]
        self.snapshot["items"][0]["action"] = "Delete"
        result = await self.preview()
        self.assertFalse(result.get("success"), result)
        self.append.assert_not_awaited()
        self.remove.assert_not_awaited()

    async def test_source_live_type_is_bound(self):
        self.live = [{"word": b11_shift_WORD, "code": "xlbl", "type": "CSS", "weight": 100}]
        self.snapshot["items"] = []
        result = await self.preview()
        self.assertFalse(result.get("success"), result)
        self.append.assert_not_awaited()

    async def test_live_source_weight_drift_invalidates_digest(self):
        self.snapshot["items"] = []
        self.live = [{"word": b11_shift_WORD, "code": "xlbl", "type": "Phrase", "weight": 100}]
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
                patch.object(b11_shift_draft_tools, "keytao_batch_remove_draft_items", real_batch_remove),
                patch.object(b11_shift_draft_tools, "_draft_mutation_claims", return_value=claims),
                patch.object(b11_shift_draft_tools, "get_bot_token", return_value="offline"),
                patch.object(b11_shift_draft_tools, "get_bot_headers", return_value={}),
                patch.object(b11_shift_draft_tools, "get_keytao_url", return_value="https://offline.invalid"),
                patch.object(b11_shift_draft_tools.httpx, "AsyncClient", side_effect=lambda **kwargs: Client(), create=True),
                patch.object(b11_shift_draft_tools.http_client, "request_with_retries", side_effect=request_once),
            ):
                preview = await self.preview()
                result = await self.confirm(preview)
            self.assertTrue(result.get("success"), result)
            self.assertEqual(len(requests), 1)
            self.assertEqual(requests[0][0], "DELETE")
            self.assertEqual(requests[0][2]["expectedContentVersion"], 8)
            self.assertEqual(requests[0][2]["expectedTargets"][0]["id"], 42)

# Source: test_b11_shift.py
class B11ChangeSourceTests(unittest.IsolatedAsyncioTestCase):
    strict = B11SourceShiftTests_strict
    preview = B11SourceShiftTests_preview
    confirm = B11SourceShiftTests_confirm

    setUp = B11ChangeSourceTests_setUp

    cleanup = B11ChangeSourceTests_cleanup

    async def test_change_source_moves_replacement_and_deletes_original_live_word(self):
        preview = await self.preview()
        self.assertTrue(preview.get("requiresConfirmation"), preview)
        self.assertEqual(preview["shiftPlan"]["sourceOriginalPhrase"], self.live[0])
        self.assertEqual([(row["action"], row["word"], row["code"])
                          for row in preview["shiftPlan"]["items"]],
                         [("Delete", "旧词", "xlbl"), ("Create", b11_shift_WORD, "xlblu")])
        result = await self.confirm(preview)
        self.assertTrue(result.get("success"), result)
        self.assertEqual([(row["action"], row["word"], row["code"])
                          for row in self.snapshot["items"]],
                         [("Delete", "旧词", "xlbl"), ("Create", b11_shift_WORD, "xlblu")])
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
            self.assertIn(word, (b11_shift_WORD, "占位词"))
            return {"success": True, "candidateCodes": ["xlbl", "xlblu", "xlbluv"]
                    if word == b11_shift_WORD else ["xlblu", "xlbl", "xlbluv"]}

        with patch.object(b11_shift_draft_tools, "_fetch_encode_candidates", side_effect=encode):
            preview = await self.preview()
            self.assertTrue(preview.get("requiresConfirmation"), preview)
            self.assertEqual([(row["word"], row["fromCode"], row["toCode"])
                              for row in preview["shiftPlan"]["shifted"]], [("占位词", "xlblu", "xlbl")])
            result = await self.confirm(preview)
        self.assertTrue(result.get("success"), result)
        self.assertEqual([(row["action"], row["word"], row["code"])
                          for row in self.snapshot["items"] if row["action"] == "Create"],
                         [("Create", b11_shift_WORD, "xlblu"), ("Create", "占位词", "xlbl")])

    async def test_change_source_does_not_globally_ignore_old_word(self):
        self.live.append({"word": "旧词", "code": "xlblu", "type": "Phrase", "weight": 100})

        async def encode(word, requested_code=None):
            return {"success": True, "candidateCodes": ["xlbl", "xlblu", "xlbluv"]}

        with patch.object(b11_shift_draft_tools, "_fetch_encode_candidates", side_effect=encode):
            preview = await self.preview()
        self.assertTrue(preview.get("requiresConfirmation"), preview)
        self.assertEqual([(row["word"], row["fromCode"], row["toCode"])
                          for row in preview["shiftPlan"]["shifted"]], [("旧词", "xlblu", "xlbluv")])

# Source: test_b24_displace.py
class B24DisplaceTests(unittest.IsolatedAsyncioTestCase):
    matcher_send = B19PartialReplyBindingTests_matcher_send
    matcher_finish = B19PartialReplyBindingTests_matcher_finish
    turn = B19PartialReplyBindingTests_turn
    created = B19PartialReplyBindingTests_created

    setUp = B24DisplaceTests_setUp

    tearDown = B24DisplaceTests_tearDown

    seed = B24DisplaceTests_seed

    finish_shift = B24DisplaceTests_finish_shift

    async def seed_recommended_displacement(self):
        state = await self.seed()
        state.recommended_code = b24_displace_CODES[0]
        state.server_ordering_assessments = [{
            "newWord": b24_displace_WORD, "occupantWord": b24_displace_OCCUPANT,
            "occupantCode": b24_displace_CODES[0], "freeCode": b24_displace_CODES[1],
            "newCode": b24_displace_CODES[0], "verdict": "front_more_common",
            "summary": "本轮已核验常用度建议调序",
        }]
        self.assertTrue(self.replay.store.set(self.replay.author, state))
        prompt = b19_partial_reply_binding_chat._chat_commands.render_server_backed_single_word_lookup(
            state.word, state.recommended_code, state.server_candidates, state.server_occupied_words,
            ordering_assessments=state.server_ordering_assessments, actionable_controls=True,
        )
        self.assertIn("「豪横」占 hzhr、「毫亨」顺延", prompt)
        await REAL_FINISH(self.bot, SimpleNamespace(message_id=1), self.replay.author.actor_id,
                          self.replay.memory, prompt)
        self.original_id = self.bot.last_id
        return state

    async def test_m1_06_recommended_displacement_add_and_submit_is_consent(self):
        await self.seed_recommended_displacement()
        await self.turn("加入并提交", self.original_id)
        self.assertEqual(self.http.submissions, 1, self.replay.last_response)
        self.assertCountEqual(self.created(), [(b24_displace_WORD, b24_displace_CODES[0]),
                                               (b24_displace_OCCUPANT, b24_displace_CODES[1])])
        self.assertNotIn("回复「确认」", self.replay.last_response)

    async def test_m2_2_recommended_displacement_add_is_consent(self):
        await self.seed_recommended_displacement()
        await self.turn("加入", self.original_id)
        self.assertEqual(len(self.http.writes), 1, self.replay.last_response)
        self.assertCountEqual(self.created(), [(b24_displace_WORD, b24_displace_CODES[0]),
                                               (b24_displace_OCCUPANT, b24_displace_CODES[1])])
        self.assertNotIn("回复「确认」", self.replay.last_response)

    async def test_m1_07_shown_pending_duplicate_add_and_submit_is_consent(self):
        from tests.helpers import DisplaceHTTP, OfflineResponse
        from urllib.parse import urlsplit

        class PendingDisplaceHTTP(DisplaceHTTP):
            def _response(self, method, url, kwargs):
                path = urlsplit(url).path
                if method == "GET" and path == "/api/batches":
                    return OfflineResponse({"batches": [{
                        "id": "other-pending-batch", "status": "Submitted", "creator": {"id": 43},
                    }], "pagination": {"page": 1, "totalPages": 1}})
                if method == "GET" and path == "/api/batches/other-pending-batch":
                    return OfflineResponse({"batch": {
                        "id": "other-pending-batch", "status": "Submitted", "creatorId": 43,
                        "pullRequests": [{"id": 51, "action": "Create", "word": b24_displace_WORD,
                                          "code": b24_displace_CODES[1], "type": "Phrase", "status": "Pending"}],
                    }})
                if method == "POST" and path.endswith("/auto-approve"):
                    return OfflineResponse({"success": False, "message": "本轮交管理员审核"})
                return super()._response(method, url, kwargs)

        self.http = PendingDisplaceHTTP()
        self.enterContext(patch.object(b11_shift_draft_tools.http_client, "_keytao_client", self.http))
        state = await self.seed()
        self.assertEqual([(item["word"], item["code"]) for item in state.shown_others_submitted],
                         [(b24_displace_WORD, b24_displace_CODES[1])])
        for changes in ({"word": "其他词"}, {"code": b24_displace_CODES[2]},
                        {"batchId": "another-pending-batch"}):
            with self.subTest(changes=changes):
                guard, _items = await b11_shift_draft_tools._pending_create_guard(
                    "qq", self.replay.author.actor_id,
                    [{"action": "Create", "word": state.word, "code": state.recommended_code}],
                    duplicate_confirmed=False,
                    shown_others_submitted=[{**state.shown_others_submitted[0], **changes}],
                )
                self.assertTrue(guard["pendingDuplicateConfirmation"])
                self.assertEqual(self.http.writes, [])
        await self.turn("加入并提交", self.original_id)
        self.assertEqual(self.http.submissions, 1, self.replay.last_response)
        self.assertEqual(self.created(), [(b24_displace_WORD, b24_displace_CODES[1])])
        self.assertNotIn("确认再提交", self.replay.last_response)

    def test_f1_displacement_verbs_are_never_explicit_word_code_adds(self):
        for prefix in ("", "加入，", "加入并提交，"):
            for verb in ("顶替", "顶掉", "挤掉", "替换"):
                with self.subTest(prefix=prefix, verb=verb):
                    self.assertIsNone(parse_explicit_entry_code_request(f"{prefix}{verb} {b24_displace_OCCUPANT}"))

    def test_f1_code_operand_requires_an_ascii_letter_in_every_entry_form(self):
        for message in ("加入，豪横 毫亨", "豪横 毫亨，加入", "添加 豪横 毫亨", "给 豪横 加码 毫亨",
                        "豪横 也放到 毫亨", "加入，豪横 123", "加入，豪横 １２３"):
            with self.subTest(message=message):
                self.assertIsNone(parse_explicit_entry_code_request(message))
        for code in ("hzhr", "HZHR", "hz1", "hz毫"):
            with self.subTest(code=code):
                self.assertEqual(parse_explicit_entry_code_request(f"加入，豪横 {code}").code, code)

    async def test_f1_ascii_occupants_do_not_turn_verbs_into_add_targets(self):
        await self.seed()
        self.replay.store.delete(self.replay.author)
        for verb in ("顶替", "顶掉", "挤掉", "替换"):
            for operand in ("AI", "hello"):
                with self.subTest(verb=verb, operand=operand):
                    for prefix in ("加入，", "加入并提交，", "加入 "):
                        message = f"{prefix}{verb} {operand}"
                        self.assertIsNone(parse_explicit_entry_code_request(message))
                        self.assertIsNone(explicit_complete_add_item(message))
                        self.assertNotEqual(understand_message(message).op, "add")
                        await self.turn(message, self.original_id)
                        self.assertEqual(self.replay.last_response, "当前没有待确认的操作。")
                    self.assertIsNone(b19_partial_reply_binding_chat._chat_commands.fresh_entry_code_selection(f"{verb} {operand}"))
                    self.assertIsNone(await b19_partial_reply_binding_chat._chat_commands.try_handle_bare_word_code(
                        f"{verb} {operand.lower()}", "qq", self.replay.author.actor_id,
                        self.replay.author, self.replay.author.space_key, "Offline owner",
                    ))
        self.assertEqual(self.http.writes, [])
        self.assertEqual(parse_explicit_entry_code_request("加入，「顶替」 dgti").word, "顶替")

    async def test_complete_code_target_replacement_keeps_existing_route(self):
        await self.seed()
        self.replay.store.delete(self.replay.author)
        await self.turn("顶替 hzhr为豪横", self.original_id, raw=raw_operation(
            op="move", operands=[b24_displace_WORD], codes=[b24_displace_CODES[0]], relation="replace", refers_to_pending=False,
        ))
        self.assertNotEqual(self.replay.last_response, "当前没有待确认的操作。")
        self.assertEqual(len(self.http.writes), 1, self.replay.last_response)

    async def test_f3_numbered_shift_is_explicit_displacement_consent(self):
        await self.seed()
        await self.finish_shift("1 顺延")

    async def test_f3_change_numbered_shift_is_explicit_displacement_consent(self):
        await self.seed()
        await self.finish_shift("换 1 顺延")

    async def test_f3_comma_numbered_shift_is_explicit_displacement_consent(self):
        await self.seed()
        await self.finish_shift("1，顺延")

    async def test_f3_bare_occupied_number_keeps_commonness_gate(self):
        state = await self.seed()
        await self.turn("1", self.original_id)
        self.assertEqual(self.http.writes, [])
        self.assertIn("hzhr 保留原位，未写入。", self.replay.last_response)
        self.assertIs(self.replay.store.get_record(self.replay.author).state, state)

    async def test_f2_retry_after_consumed_ticket_reminds_without_writing(self):
        await self.seed()
        await self.turn("1", self.original_id)
        refusal_id = self.bot.last_id
        await self.finish_shift("加入，顶替 毫亨")
        writes = copy.deepcopy(self.http.writes)
        await self.turn("加入，顶替 毫亨", refusal_id)
        self.assertEqual(self.http.writes, writes)
        self.assertIn("「豪横」已在当前草稿中（→ hzhr）：", self.replay.last_response)
        self.assertIn("/batch/" + self.http.batch_id, self.replay.last_response)

    async def test_f2_no_completed_shift_keeps_existing_no_ticket_reply(self):
        await self.seed()
        self.replay.store.delete(self.replay.author)
        await self.turn("加入，顶替 毫亨", self.original_id)
        self.assertEqual(self.http.writes, [])
        self.assertEqual(self.replay.last_response, "当前没有待确认的操作。")

    async def test_f2_retry_without_any_live_ticket_reuses_original_candidate_quote(self):
        await self.seed()
        await self.finish_shift("加入，顶替 毫亨")
        self.replay.store.delete(self.replay.author)
        writes = copy.deepcopy(self.http.writes)
        await self.turn("加入，顶替 毫亨", self.original_id)
        self.assertEqual(self.http.writes, writes)
        self.assertEqual(self.replay.last_response,
                         f"「豪横」已在当前草稿中（→ hzhr）：https://keytao.rea.ink/batch/{self.http.batch_id}")

    async def test_f2_partial_shift_facts_do_not_claim_completion(self):
        await self.seed()
        self.replay.store.delete(self.replay.author)
        complete = [
            {"id": 1, "action": "Create", "word": b24_displace_WORD, "code": b24_displace_CODES[0], "type": "Phrase"},
            {"id": 2, "action": "Delete", "word": b24_displace_OCCUPANT, "code": b24_displace_CODES[0], "type": "Phrase"},
            {"id": 3, "action": "Create", "word": b24_displace_OCCUPANT, "code": b24_displace_CODES[1], "type": "Phrase"},
        ]
        for rows in (complete[:1], complete[:2], complete[::2], complete[1:]):
            with self.subTest(rows=rows):
                self.http.rows = copy.deepcopy(rows)
                await self.turn("加入，顶替 毫亨", self.original_id)
                self.assertEqual(self.replay.last_response, "当前没有待确认的操作。")
                self.assertEqual(self.http.writes, [])

    async def test_f2_shift_at_another_code_does_not_match_the_quoted_occupied_code(self):
        await self.seed()
        self.replay.store.delete(self.replay.author)
        self.http.rows = [
            {"id": 1, "action": "Create", "word": b24_displace_WORD, "code": "abcd", "type": "Phrase"},
            {"id": 2, "action": "Delete", "word": b24_displace_OCCUPANT, "code": "abcd", "type": "Phrase"},
            {"id": 3, "action": "Create", "word": b24_displace_OCCUPANT, "code": "abcde", "type": "Phrase"},
        ]
        await self.turn("加入，顶替 毫亨", self.original_id)
        self.assertEqual(self.replay.last_response, "当前没有待确认的操作。")
        self.assertEqual(self.http.writes, [])

    async def test_f2_unrelated_bot_quote_does_not_reuse_completed_shift(self):
        await self.seed()
        await self.finish_shift("加入，顶替 毫亨")
        self.replay.store.delete(self.replay.author)
        unrelated = await self.bot.deliver(event=None, message="喵喵在。")
        writes = copy.deepcopy(self.http.writes)
        await self.turn("加入，顶替 毫亨", str(unrelated["message_id"]))
        self.assertEqual(self.replay.last_response, "当前没有待确认的操作。")
        self.assertEqual(self.http.writes, writes)

    async def test_f1_stale_verb_variants_do_not_look_up_the_verb(self):
        await self.seed()
        self.replay.store.delete(self.replay.author)
        for prefix in ("", "加入，", "加入并提交，"):
            for verb in ("顶替", "顶掉", "挤掉", "替换"):
                with self.subTest(prefix=prefix, verb=verb):
                    await self.turn(f"{prefix}{verb} {b24_displace_OCCUPANT}", self.original_id)
                    self.assertEqual(self.replay.last_response, "当前没有待确认的操作。")
                    self.assertEqual(self.http.writes, [])

    async def test_f3_empty_number_keeps_existing_refusal(self):
        await self.seed()
        await self.turn("2 顺延", self.original_id)
        self.assertEqual(self.http.writes, [])
        self.assertIn("hzhr 保留原位，未写入。", self.replay.last_response)

    async def test_f3_multiple_occupants_keep_commonness_refusal(self):
        state = await self.seed()
        state.server_occupied_words[b24_displace_CODES[0]].append("耗横")
        record = self.replay.store.get_record(self.replay.author)
        await REAL_FINISH(self.bot, None, self.replay.author.actor_id, self.replay.memory,
                              b19_partial_reply_binding_chat._render_live_single_candidate_record(record))
        await self.turn("1 顺延", self.bot.last_id)
        self.assertEqual(self.http.writes, [])
        self.assertEqual(self.replay.last_response, "没找到要执行的操作，本次未写入。")

    async def test_f4_other_speaker_cannot_displace_on_owners_ticket(self):
        state = await self.seed()
        for message in ("1 顺延", "换 1 顺延", "1，顺延", "加入，顶替 毫亨"):
            with self.subTest(message=message):
                await self.turn(message, self.original_id, actor="b24-other-speaker")
                self.assertEqual(self.replay.last_response, "这条提议不是发给你的，本次未执行。")
                self.assertIs(self.replay.store.get_record(self.replay.author).state, state)
                self.assertEqual(self.http.writes, [])

# Source: test_b25_session.py
class B25SessionTests(unittest.IsolatedAsyncioTestCase):
    matcher_send = B19PartialReplyBindingTests_matcher_send
    matcher_finish = B19PartialReplyBindingTests_matcher_finish
    created = B19PartialReplyBindingTests_created

    setUp = B25SessionTests_setUp

    tearDown = B25SessionTests_tearDown

    history_for = B25SessionTests_history_for

    remember = B25SessionTests_remember

    turn = B25SessionTests_turn

    greeting = B25SessionTests_greeting

    seed = B25SessionTests_seed

    partial_session = B25SessionTests_partial_session

    finish_replacement = B25SessionTests_finish_replacement

    def configure_audit_move(self, facts, entries):
        self.enterContext(patch.dict(b25_session_FACTS, facts))
        self.http.entries = [{"word": word, "code": code, "type": "Phrase", "weight": 100}
                             for word, code in entries]
        from tests.helpers import closing, sqlite3, urlsplit

        original = self.http._response

        def response(method, url, kwargs):
            if urlsplit(url).path == "/api/bot/phrases/encode":
                url = url.replace("/api/bot/phrases/encode", "/api/phrases/encode")
            return original(method, url, kwargs)

        self.enterContext(patch.object(self.http, "_response", side_effect=response))

        with closing(sqlite3.connect(review.reference_db_path())) as connection, connection:
            connection.executemany("INSERT OR REPLACE INTO word_commonness VALUES (?, 3, NULL, 2)",
                                   [(word,) for word in facts])

    async def assert_recent_codeless_shift(self, message, word, codes, displayed, *, draft_keeper=False):
        self.configure_audit_move({word: (codes, "cha bao")}, [(word, codes[0])])
        keeper = {"word": "同码词", "code": codes[0], "type": "Phrase", "weight": 100}
        if draft_keeper:
            self.http.rows.append({"id": 1, "action": "Create", **keeper})
        else:
            self.http.entries.append(keeper)
        self.replay.history[:] = [{"role": "assistant", "content": displayed}]
        await self.turn(message, await self.greeting())
        self.assertEqual(self.http.writes, [], self.replay.last_response)
        record = self.replay.store.get_record(self.replay.author)
        self.assertIsNotNone(record, self.replay.last_response)
        self.assertIsInstance(record.state, PendingToolConfirm, self.replay.last_response)
        self.assertEqual(record.state.function_name, "keytao_shift_phrase_code", self.replay.last_response)
        self.assertEqual(record.state.args["word"], word)
        self.assertEqual(record.state.args["target_code"], codes[1])
        self.assertIn("确认", self.replay.last_response)
        await self.turn("确认", self.bot.last_id)
        self.assertEqual(len(self.http.writes), 1, self.replay.last_response)
        self.assertIn((word, codes[1]), self.created())
        self.assertIn((word, codes[0]), [(row["word"], row["code"]) for row in self.http.rows if row["action"] == "Delete"])

    async def test_a1_dalv_codeless_shift_after_display_with_real_tools(self):
        # Production rows 10540/10543; the export has no display before 10540.
        await self.assert_recent_codeless_shift("大绿后移一码", "大绿", ["dsllv", "dsllva", "dsllvai"],
                                                "「大绿」已在词库（dsllv）。")

    async def test_c3_chabao_postpose_after_lookup_with_real_tools(self):
        # Production rows 10743/10744.
        await self.assert_recent_codeless_shift("把查报后置", "查报", ["wsbz", "wsbzi", "wsbziv"],
                                                "已在词库：查词 wsck、查报 wsbz、茶包 wsbz")

    async def test_c4_chabao_spaced_shift_after_lookup_with_real_tools(self):
        # Production row 10746, replayed after its lookup rather than the failed card.
        await self.assert_recent_codeless_shift("把 查报 顺延", "查报", ["wsbz", "wsbzi", "wsbziv"],
                                                "已在词库：查词 wsck、查报 wsbz、茶包 wsbz")

    async def test_d3_chaocuo_shift_after_draft_receipt_with_real_tools(self):
        # Production receipt and command in audit2/new-cases.md, 22:51:54.
        await self.assert_recent_codeless_shift("晁错顺延", "晁错", ["jzcl", "jzcli", "jzcliv"],
                                                "✅ 已加入草稿：抄错 jzcl、抄错 wzcl\n"
                                                "⚠️ jzcl 与「晁错」同码，确认后「抄错」将作为重码写入\n"
                                                "⚠️ wzcl 与「晁错」同码，确认后「抄错」将作为重码写入",
                                                draft_keeper=True)

    async def test_d3_chaocuo_two_codes_select_then_preview_and_confirm(self):
        self.configure_audit_move({"晁错": (["jzcl", "jzcli", "jzcliv", "wzcl", "wzcli", "wzcliv"], "chao cuo")},
                                  [("晁错", "jzcl"), ("晁错", "wzcl")])
        self.http.rows[:] = [{"id": index, "action": "Create", "word": "抄错", "code": code,
                             "type": "Phrase", "weight": 100} for index, code in enumerate(("jzcl", "wzcl"), 1)]
        self.replay.history[:] = [{"role": "assistant", "content":
            "✅ 已加入草稿：抄错 jzcl、抄错 wzcl\n⚠️ jzcl 与「晁错」同码\n⚠️ wzcl 与「晁错」同码"}]
        await self.turn("晁错顺延", await self.greeting())
        self.assertEqual(self.http.writes, [])
        self.assertIn("1.", self.replay.last_response)
        self.assertIn("2.", self.replay.last_response)
        await self.turn("2", self.bot.last_id)
        record = self.replay.store.get_record(self.replay.author)
        self.assertIsNotNone(record, self.replay.last_response)
        self.assertEqual(record.state.function_name, "keytao_shift_phrase_code", self.replay.last_response)
        self.assertEqual(record.state.args["target_code"], "wzcli")
        self.assertEqual(self.http.writes, [])
        await self.turn("确认", self.bot.last_id)
        self.assertEqual(len(self.http.writes), 1, self.replay.last_response)
        self.assertIn(("晁错", "wzcli"), self.created())
        self.assertNotIn(("晁错", "jzcli"), self.created())

    async def test_recent_shift_skips_dictionary_and_draft_occupied_codes(self):
        self.configure_audit_move({"查报": (["wsbz", "wsbzi", "wsbziv", "wsbziva"], "cha bao")},
                                  [("查报", "wsbz"), ("茶包", "wsbz"), ("占位", "wsbzi")])
        self.http.rows[:] = [{"id": 1, "action": "Create", "word": "草稿占位", "code": "wsbziv",
                             "type": "Phrase", "weight": 100}]
        self.replay.history[:] = [{"role": "assistant", "content": "「查报」已在当前草稿中（→ wsbz）。"}]
        await self.turn("查报往后移", await self.greeting())
        record = self.replay.store.get_record(self.replay.author)
        self.assertIsNotNone(record, self.replay.last_response)
        self.assertEqual(record.state.function_name, "keytao_shift_phrase_code")
        self.assertEqual(record.state.args["target_code"], "wsbziva")
        self.assertEqual(self.http.writes, [])
        await self.turn("确认", self.bot.last_id)
        self.assertIn(("查报", "wsbziva"), self.created())
        self.assertNotIn(("占位", "wsbziv"), self.created())

    async def test_codeless_shift_requires_latest_exact_display_and_current_duplicate(self):
        self.configure_audit_move({"查报": (["wsbz", "wsbzi"], "cha bao")}, [("查报", "wsbz"), ("茶包", "wsbz")])
        shown = {"role": "assistant", "content": "已在词库：查报 wsbz、茶包 wsbz"}
        cases = (
            ([], True),
            ([{"role": "user", "content": "查报 wsbz"}], True),
            ([shown, {"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好。"}], True),
            ([{"role": "assistant", "content": "「把查报后置」未收录：bǎ chá bào hòu zhì\n1. bwbf — ✅推荐"}], True),
            ([shown], False),
        )
        for history, duplicate in cases:
            with self.subTest(history=history, duplicate=duplicate):
                self.http.entries[:] = self.http.entries[:1] + ([{"word": "茶包", "code": "wsbz", "type": "Phrase", "weight": 100}] if duplicate else [])
                response = await b25_session_commands.try_handle_advertised_shift(
                    "查报后移", "qq", self.replay.author.actor_id, self.replay.author,
                    self.replay.author.space_key, "", history=history, require_bound_operand=True,
                )
                self.assertIsNone(response)
                self.assertIsNone(self.replay.store.get_record(self.replay.author))
                self.assertEqual(self.http.writes, [])

    async def test_recent_shift_history_does_not_bind_other_user(self):
        self.configure_audit_move({"查报": (["wsbz", "wsbzi"], "cha bao"), "查报后移": ([], "a")},
                                  [("查报", "wsbz"), ("茶包", "wsbz")])
        self.replay.history[:] = [{"role": "assistant", "content": "已在词库：查报 wsbz、茶包 wsbz"}]
        await self.turn("查报后移", await self.greeting(), actor="other-user")
        await self.turn("确认", self.bot.last_id, actor="other-user")
        self.assertEqual(self.http.writes, [])

    async def test_recent_shift_does_not_bind_controls_or_chitchat(self):
        self.configure_audit_move({"确认": (["xlbl", "xlblu"], "que ren"), "是的": (["xlbl", "xlblu"], "shi de")},
                                  [("确认", "xlbl"), ("是的", "xlbl"), ("保留词", "xlbl")])
        for message, displayed in (
            ("确认后移", "「查报」目前在 wsbz，回复「确认」执行，或「取消」。"),
            ("是的后移", "是的"),
        ):
            with self.subTest(message=message):
                response = await b25_session_commands.try_handle_advertised_shift(
                    message, "qq", self.replay.author.actor_id, self.replay.author,
                    self.replay.author.space_key, "", history=[{"role": "assistant", "content": displayed}],
                    require_bound_operand=True,
                )
                self.assertIsNone(response)
                self.assertIsNone(self.replay.store.get_record(self.replay.author))
                self.assertEqual(self.http.writes, [])

    async def test_recent_shift_uses_display_structure_instead_of_operand_stopwords(self):
        await self.assert_recent_codeless_shift("怎么后移", "怎么", ["xlbl", "xlblu", "xlbluv"],
                                                "「怎么」已在词库（xlbl）。")

    async def test_codeless_shift_fragments_never_preview_or_write(self):
        words = ("是的", "还是", "嗯", "好的", "那就", "它", "把它")
        messages = ("是的后移", "还是 顺延", "嗯后移", "好的后移", "那就顺延", "把它顺延")
        self.configure_audit_move({**{word: (["xlbl", "xlblu"], "a") for word in words},
                                   **{message: ([], "a") for message in messages}},
                                  [(word, "xlbl") for word in (*words, "保留词")])
        for message in messages:
            with self.subTest(message=message):
                self.replay.store.delete(self.replay.author)
                self.replay.history[:] = [{"role": "assistant", "content": "已在词库：查报 wsbz、茶包 wsbz"}]
                self.http.requests.clear()
                await self.turn(message, await self.greeting())
                record = self.replay.store.get_record(self.replay.author)
                self.assertFalse(record is not None and isinstance(record.state, PendingToolConfirm)
                                 and record.state.function_name == "keytao_shift_phrase_code", self.replay.last_response)
                await self.turn("确认", self.bot.last_id)
                self.assertEqual(self.http.writes, [], self.replay.last_response)
                self.assertFalse(any(path.endswith(("/pull-requests/batch", "/pull-requests/batch-draft"))
                                     for _, path, _ in self.http.requests))

    async def test_c1_merged_recommendation_shifts_named_occupant_with_real_tools(self):
        self.configure_audit_move({
            "交托": (["jctl", "jctlo", "jctloo"], "jiao tuo"),
            "较妥": (["jctl", "jctlo", "jctloo"], "jiao tuo"),
        }, [("较妥", "jctl")])
        from tests.helpers import closing, sqlite3

        with closing(sqlite3.connect(review.reference_db_path())) as connection, connection:
            connection.execute("UPDATE word_commonness SET corpus_frequency = 1000 WHERE word = '交托'")
            connection.execute("UPDATE word_commonness SET dictionary_presence_count = 0 WHERE word = '较妥'")
            connection.executemany("INSERT INTO bcc_frequency VALUES (?, '交托', 17, 17, 1)",
                                   [("多领域",), ("文学",)])
        await self.seed("交托、公义")
        record = self.replay.store.get_record(self.replay.author)
        self.assertIsInstance(record.state, harness.PendingToolConfirm, self.replay.last_response)
        scopes = record.state.args["_candidate_scopes"]
        scope = next(scope for scope in scopes if scope["word"] == "交托")
        self.assertIsNotNone(b25_session_commands.validated_front_insert_recommendation(
            "交托", scope["candidates"], scope["occupiedWords"], scope["orderingAssessments"],
        ), self.replay.last_response)
        await self.turn('交托 1，顺延“较妥”', self.original_id)
        record = self.replay.store.get_record(self.replay.author)
        self.assertEqual(record.state.function_name, "keytao_shift_phrase_code", self.replay.last_response)
        self.assertEqual(self.http.writes, [])
        self.assertIn("「较妥」jctl → jctlo", self.replay.last_response)
        await self.turn("确认", self.bot.last_id)
        self.assertCountEqual([(row["action"], row["word"], row["code"]) for row in self.http.rows],
                              [("Create", "交托", "jctl"), ("Delete", "较妥", "jctl"),
                               ("Create", "较妥", "jctlo")])
        record = self.replay.store.get_record(self.replay.author)
        self.assertIn("公义", str(record.state.args))
        self.assertEqual(self.http.errors, [])

    async def test_g1_internal_submit_metadata_never_reaches_real_tool(self):
        result = json.loads(await b25_session_commands.call_tool_function("keytao_submit_batch", {
            "batch_id": self.http.batch_id, "_recent_own_write": True,
            "_recent_batch_ids": [self.http.batch_id], "_recent_written_items": [],
            "_recent_content_version": 0,
        }, "qq", self.replay.author.actor_id))
        errors = " ".join(str(call) for call in self.tool_errors.call_args_list)
        self.assertTrue(result.get("requiresConfirmation"), (result, errors))

    async def test_g1_t7_t8_receipt_confirm_submits_exact_batch(self):
        await self.seed()
        await self.finish_replacement("添加 守时 edekoo，顶替 首诗")
        record = self.replay.store.get_record(self.replay.author)
        self.assertEqual(record.state.function_name, "keytao_submit_batch", self.replay.last_response)
        self.assertTrue(record.state.args["_recent_own_write"])
        await self.turn("确认", self.bot.last_id)
        errors = " ".join(str(call) for call in self.tool_errors.call_args_list)
        self.assertEqual(self.http.submissions, 1, (self.replay.last_response, errors))
        self.assertIn("已提交审核", self.replay.last_response)
        self.assertTrue(self.http.writes[-1][0].endswith("/40c69d7a/submit"))

    async def test_g1_t7_t8_receipt_submit_submits_exact_batch(self):
        await self.seed()
        await self.finish_replacement("添加 守时 edekoo，顶替 首诗")
        await self.turn("提交", self.bot.last_id)
        self.assertEqual(self.http.submissions, 1, self.replay.last_response)

    async def test_g1_declared_private_source_code_is_preserved(self):
        self.http.entries = [{"word": "周内", "code": code, "type": "Phrase", "weight": 100}
                             for code in ("fdnsu", "fdn")]
        with patch.dict(b25_session_FACTS, {"周内": (["fdn", "fdnw", "fdnsu"], "zhou nei")}):
            result = json.loads(await b25_session_commands.call_tool_function("keytao_shift_phrase_code", {
                "word": "周内", "target_code": "fdnw", "_source_code": "fdnsu",
            }, "qq", self.replay.author.actor_id))
        self.assertEqual([(row["word"], row["code"]) for row in result["shiftPlan"]["items"]
                          if row["action"] == "Delete"], [("周内", "fdnsu")])

    def test_g2_fillers_are_not_explicit_add_words(self):
        for phrase in ("再加一个", "再加", "加入编码", "加编码", "添加这个", "添加它", "添加一下"):
            with self.subTest(phrase=phrase):
                message = phrase + " fdnw"
                self.assertIsNone(explicit_complete_add_item(message))
                self.assertIsNone(parse_explicit_entry_code_request(message))

    async def test_g2_quoted_additional_code_binds_one_existing_word(self):
        for phrase in ("再加一个", "再加", "加入编码", "加编码"):
            with self.subTest(phrase=phrase):
                self.replay.store.delete(self.replay.author)
                self.http.rows.clear()
                self.http.writes.clear()
                await self.seed("周内")
                await self.turn(phrase + " fdnw", self.original_id)
                self.assertEqual(self.created(), [("周内", "fdnw")], self.replay.last_response)

    async def test_g2_without_single_quoted_word_never_looks_up_fillers(self):
        for phrase in ("再加一个", "再加", "加入编码", "加编码"):
            with self.subTest(phrase=phrase):
                await self.turn(phrase + " fdnw", await self.greeting())
                self.assertEqual(self.http.writes, [])
                self.assertIn("词", self.replay.last_response)

    async def test_g2_quoted_word_still_validates_explicit_code(self):
        await self.seed("周内")
        await self.turn("再加一个 abcd", self.original_id)
        self.assertEqual(self.http.writes, [])
        self.assertIn("周内", self.replay.last_response)
        self.assertIn("abcd", self.replay.last_response)

    async def test_g3_t1_t2_partial_then_t3_named_number_displacement(self):
        reply_id = await self.partial_session()
        self.http.writes.clear()
        rows = copy.deepcopy(self.http.rows)
        await self.turn('守时 3，顺延“首诗”', reply_id)
        self.assertEqual(self.http.writes, [])
        self.assertEqual(self.http.rows, rows)
        self.assertIn("推荐顺延方案", self.replay.last_response)

    async def test_g3_t4_unquoted_replacement_is_not_a_new_word(self):
        await self.seed()
        await self.finish_replacement("守时替代首诗")

    async def test_g3_t6_quoted_replacement_uses_reviewed_occupant(self):
        await self.seed()
        await self.finish_replacement('“守时”替代“首诗”')

    async def test_g3_add_words_containing_displacement_verbs_keeps_candidate_ticket(self):
        facts = {
            "替代品": (["tdpk", "tdpko", "tdpkoo"], "ti dai pin"),
            "取代者": (["qdfv", "qdfvo", "qdfvoo"], "qu dai zhe"),
        }
        with patch.dict(b25_session_FACTS, facts):
            for message, word in (("喵喵 加词 替代品", "替代品"), ("加词 取代者", "取代者")):
                with self.subTest(message=message):
                    self.replay.store.delete(self.replay.author)
                    await self.turn(message, await self.greeting())
                    record = self.replay.store.get_record(self.replay.author)
                    self.assertIsNotNone(record, self.replay.last_response)
                    self.assertIsInstance(record.state, harness.PendingAddWord)
                    self.assertEqual(record.state.word, word)
                    self.assertIn(facts[word][0][0], self.replay.last_response)
                    self.assertEqual(self.http.writes, [])

    async def test_g3_non_displacement_turns_keep_language_route_without_http(self):
        for message in ("加入 替代品", "能用拼音替代编码吗"):
            with self.subTest(message=message):
                await self.turn(message, await self.greeting())
                self.assertEqual(self.replay.last_response, "这是离线解释。")
                self.assertEqual(self.http.requests, [])
                self.assertIsNone(self.replay.store.get_record(self.replay.author))

    async def test_g3_separated_words_containing_displacement_verbs_keep_original_routes(self):
        facts = {
            "替代品": (["tdpk", "tdpko", "tdpkoo"], "ti dai pin"),
            "查一下": (["wyxx", "wyxxo", "wyxxoo"], "cha yi xia"),
            "新词": (["xbck", "xbcko", "xbckoo"], "xin ci"),
        }
        cases = (
            ("喵喵 查词 替代品", ("替代品",)),
            ("查一下 替代品", ("查一下", "替代品")),
            ("喵喵 守时 替代品", (b25_session_WORD, "替代品")),
            ("新词 替代品", ("新词", "替代品")),
            ("加 替代品", ()),
            ("帮我加 替代品", ()),
        )
        with patch.dict(b25_session_FACTS, facts):
            for message, words in cases:
                with self.subTest(message=message):
                    self.replay.store.delete(self.replay.author)
                    self.http.requests.clear()
                    await self.turn(message, await self.greeting())
                    record = self.replay.store.get_record(self.replay.author)
                    if words:
                        self.assertIsNotNone(record, self.replay.last_response)
                        if len(words) == 1:
                            self.assertIsInstance(record.state, harness.PendingAddWord)
                            self.assertEqual(record.state.word, words[0])
                        else:
                            self.assertIsInstance(record.state, harness.PendingToolConfirm)
                            self.assertEqual(record.state.args["_query_words"], list(words))
                        for word in words:
                            self.assertIn(b25_session_FACTS[word][0][0], self.replay.last_response)
                    else:
                        self.assertEqual(self.replay.last_response, "这是离线解释。")
                        self.assertEqual(self.http.requests, [])
                        self.assertIsNone(record)
                    self.assertEqual(self.http.writes, [])

    def test_g3_questions_and_command_operands_are_not_displacements(self):
        for message in ("喵喵 加词 替代品", "加入 替代品", "加词 取代者", "添加 替代品",
                        "能用拼音替代编码吗", "可以用拼音替代编码", "拼音能替代编码吗",
                        "拼音是否替代编码", "加词替代品", "守时替代加入",
                        "用户替代首诗", "用途替代首诗", "用户名替代首诗", "用守时替代首诗"):
            with self.subTest(message=message):
                self.assertIsNone(b25_session_commands.parse_reviewed_displacement(message))

    def test_g3_use_prefix_cannot_split_a_word(self):
        for message in ("「用户」替代首诗", "用「用户」替代首诗", "用 用户 替代 首诗"):
            with self.subTest(message=message):
                self.assertEqual(b25_session_commands.parse_reviewed_displacement(message), ("用户", b25_session_OCCUPANT, ""))

    async def test_g3_unquoted_use_prefix_binds_exact_reviewed_word(self):
        await self.seed()
        await self.finish_replacement("用守时替代首诗")

    def test_g3_lexical_words_are_not_modal_questions(self):
        for word in ("能量", "能力"):
            with self.subTest(word=word):
                self.assertEqual(b25_session_commands.parse_reviewed_displacement(word + "替代首诗"), (word, b25_session_OCCUPANT, ""))

    def test_g3_glued_use_prefix_keeps_unquoted_operand_validation(self):
        for word in ("加入", "加词", "编码"):
            with self.subTest(word=word):
                self.assertIsNone(b25_session_commands.parse_reviewed_displacement("用" + word + "替代首诗", known_words=(word,)))

    async def test_g3_t7_exact_quoted_add_replacement(self):
        await self.seed()
        await self.finish_replacement('添加“守时”edekoo，顶替“首诗”')

    async def test_g3_replacement_verbs_and_optional_prefix(self):
        for verb in ("替代", "取代", "顶替", "替换", "换掉"):
            with self.subTest(verb=verb):
                self.replay.store.delete(self.replay.author)
                self.http.rows.clear()
                self.http.writes.clear()
                await self.seed()
                await self.finish_replacement("用「守时」" + verb + "「首诗」")

    async def test_g3_named_number_displacement_variants(self):
        for tail in ("，顶替首诗", "，替代首诗"):
            with self.subTest(tail=tail):
                self.replay.store.delete(self.replay.author)
                self.http.rows.clear()
                self.http.writes.clear()
                await self.seed("、".join(b25_session_WORDS))
                await self.finish_replacement("守时 3" + tail)
                record = self.replay.store.get_record(self.replay.author)
                if record is not None and record.state.function_name == "keytao_batch_add_to_draft":
                    self.assertNotIn(b25_session_WORD, record.state.args["_query_words"])
        self.replay.store.delete(self.replay.author)
        self.http.rows.clear()
        self.http.writes.clear()
        await self.seed("、".join(b25_session_WORDS))
        await self.turn("守时 3 顺延 首诗", self.original_id)
        self.assertEqual(self.http.writes, [])
        self.assertIn("推荐顺延方案", self.replay.last_response)

    async def test_g3_wrong_occupant_refuses_without_concatenated_lookup(self):
        await self.seed()
        for message in ("守时替代周内", "守时 3，顺延首饰"):
            with self.subTest(message=message):
                await self.turn(message, self.original_id)
                self.assertEqual(self.http.writes, [])
                self.assertIn("未", self.replay.last_response)

    async def test_g3_changed_candidates_preserve_original_batch_ticket(self):
        original = await self.seed("、".join(b25_session_WORDS))
        snapshot = copy.deepcopy(original)
        with patch.dict(b25_session_FACTS, {b25_session_WORD: (b25_session_CODES[:2], "shou shi")}):
            await self.turn("守时 3，顺延首诗", self.original_id)
        self.assertEqual(self.http.writes, [])
        self.assertEqual(self.replay.store.get_record(self.replay.author), snapshot)

    async def test_g3_cold_wrong_occupant_cannot_arm_an_add_confirmation(self):
        await self.turn("公义替代首诗", await self.greeting())
        self.assertIsNone(self.replay.store.get_record(self.replay.author))
        await self.turn("确认", self.bot.last_id)
        self.assertEqual(self.http.writes, [])

    async def test_g2_one_word_with_multiple_existing_codes_can_add_code(self):
        self.http.entries.append({"word": "周内", "code": "fdnwu", "type": "Phrase", "weight": 100})
        await self.seed("周内")
        await self.turn("再加一个 fdnw", self.original_id)
        self.assertEqual(self.created(), [("周内", "fdnw")], self.replay.last_response)

    async def test_g2_single_bot_word_quote_is_rechecked_without_live_state(self):
        await self.seed("周内")
        self.replay.store.delete(self.replay.author)
        await self.turn("加入编码 fdnw", self.original_id)
        self.assertEqual(self.created(), [("周内", "fdnw")], self.replay.last_response)

    async def test_g5_cold_quote_cannot_be_used_by_another_speaker(self):
        await self.seed("周内")
        self.replay.store.delete(self.replay.author)
        await self.turn("加入编码 fdnw", self.original_id, actor="b25-other-speaker")
        self.assertEqual(self.http.writes, [])

    async def test_g2_multiple_quoted_words_do_not_choose_a_target(self):
        original = await self.seed("公义、礼义廉耻")
        snapshot = copy.deepcopy(original)
        await self.turn("再加一个 fdnw", self.original_id)
        self.assertEqual(self.http.writes, [])
        self.assertEqual(self.replay.store.get_record(self.replay.author), snapshot)

    async def test_g2_unrelated_quote_does_not_use_live_word(self):
        await self.seed("周内")
        await self.turn("再加一个 fdnw", await self.greeting())
        self.assertEqual(self.http.writes, [])

    async def test_g5_other_speaker_cannot_add_code_on_owners_quote(self):
        original = await self.seed("周内")
        snapshot = copy.deepcopy(original)
        await self.turn("再加一个 fdnw", self.original_id, actor="b25-other-speaker")
        self.assertEqual(self.http.writes, [])
        self.assertEqual(self.replay.store.get_record(self.replay.author), snapshot)

    async def test_g4_t5_delete_single_word_uses_deterministic_real_tool(self):
        for message in ("删除：首诗", "删除 首诗", "删掉 首诗"):
            with self.subTest(message=message):
                self.replay.store.delete(self.replay.author)
                self.http.rows.clear()
                self.http.writes.clear()
                await self.turn(message, await self.greeting())
                record = self.replay.store.get_record(self.replay.author)
                if not self.http.writes and record is not None:
                    self.assertIsInstance(record.state, harness.PendingToolConfirm)
                    await self.turn("确认", self.bot.last_id)
                self.assertEqual([(row["word"], row["code"], row["action"]) for row in self.http.rows],
                                 [(b25_session_OCCUPANT, b25_session_CODES[2], "Delete")], self.replay.last_response)
                self.assertNotIn("安全拦截", self.replay.last_response)

    async def test_g4_multiple_codes_require_existing_disambiguation(self):
        self.http.entries.append({"word": b25_session_OCCUPANT, "code": "edekov", "type": "Phrase", "weight": 100})
        await self.turn("删除：首诗", await self.greeting())
        self.assertEqual(self.http.writes, [])
        self.assertIn("edekoo", self.replay.last_response)
        self.assertIn("edekov", self.replay.last_response)

    async def test_g5_other_speaker_cannot_act_on_owners_ticket(self):
        record = await self.seed()
        before = copy.deepcopy(record)
        for message in ('守时 3，顺延“首诗”', "守时替代首诗", '“守时”替代“首诗”'):
            with self.subTest(message=message):
                await self.turn(message, self.original_id, actor="b25-other-speaker")
                self.assertEqual(self.http.writes, [])
                self.assertEqual(self.replay.store.get_record(self.replay.author), before)
                self.assertIn("不是发给你的", self.replay.last_response)

# Source: test_s56_generic_eviction.py
class GenericEvictionTests(unittest.TestCase):
    run_command = GenericEvictionTests_run_command

    def test_generic_tail_without_record_cannot_create_protected_shift_ticket(self):
        async def run():
            reply, shift = await self.run_command("加词 耙耙柑 ppg，顺延其他词条")
            self.assertEqual(0, shift.await_count, reply)
            self.assertIn("琵琶骨", reply)
            self.assertIn("本次未写入", reply)
        asyncio.run(run())

    def test_generic_tail_with_live_record_cannot_create_protected_shift_ticket(self):
        async def run():
            reply, shift = await self.run_command("加词 耙耙柑 ppg，顺延其他词条", preexisting=True)
            self.assertEqual(0, shift.await_count, reply)
            self.assertIn("琵琶骨", reply)
            self.assertIn("本次未写入", reply)
        asyncio.run(run())

    def test_lookup_inferred_occupant_is_not_user_named_authorization(self):
        async def run():
            reply, shift = await self.run_command("加词 耙耙柑 ppg 重新编码", verdict="")
            self.assertEqual(0, shift.await_count, reply)
            self.assertIn("琵琶骨", reply)
        asyncio.run(run())

    def test_user_named_occupant_keeps_the_complete_shift_path(self):
        async def run():
            reply, shift = await self.run_command("加词 耙耙柑 ppg，顺延琵琶骨", preexisting=True)
            self.assertEqual("planned fixture", reply)
            self.assertEqual(1, shift.await_count)
            self.assertTrue(shift.call_args.kwargs["auto_confirm_shift_plan"])
            self.assertEqual(("琵琶骨",), shift.call_args.kwargs["auto_confirm_expected_shifted_words"])
        asyncio.run(run())

    def test_synthetic_weaker_assessment_preserves_generic_shift(self):
        async def run():
            # This bound assessment is synthetic; it does not assert corpus frequencies.
            reply, shift = await self.run_command(
                "加词 耙耙柑 ppg，顺延其他词条", verdict="front_more_common",
            )
            self.assertEqual("planned fixture", reply)
            self.assertEqual(1, shift.await_count)
            self.assertFalse(shift.call_args.kwargs["auto_confirm_shift_plan"])
        asyncio.run(run())

# Source: test_s56_undo.py
class CompletedUndoTests(unittest.TestCase):
    setUp = CompletedUndoTests_setUp

    tearDown = CompletedUndoTests_tearDown

    write = CompletedUndoTests_write

    cancel = CompletedUndoTests_cancel

    def test_cancel_without_pending_never_falls_through_to_model(self):
        async def run():
            key = harness.ConversationAddress.private("qq", "s56-undo")
            ctx = s56_undo_chat.TurnContext(SimpleNamespace(), SimpleNamespace(), "qq", "s56-undo")
            ctx.conv_key = key
            ctx.normalized_message_text = "取消"
            ctx.generic_command_intent = await s56_undo_chat._classify_message_command_intent("取消", None)
            with patch.object(s56_undo_chat, "_try_handle_draft_management_command", AsyncMock(return_value=None)):
                for stage in s56_undo_chat.STAGES:
                    if stage.__name__ in {"_stage_handle_completed_draft_undo", "_stage_handle_draft_management"}:
                        await stage(ctx)
            self.assertIsNotNone(ctx.response, "completed-write cancellation reached model fallback")
        asyncio.run(run())

    def test_all_four_forms_restore_whole_chain_and_preserve_old_draft(self):
        async def run():
            for message in ("取消", "撤销", "撤回", "回滚"):
                self.server = DraftServer()
                await self.write()
                receipt = await self.cancel(message)
                self.assertEqual(self.server.items, [s56_undo_row(1, "旧", "jq")])
                for expected in ("撤销整笔", "鎗@qx", "强@qxa", "戕@qxai", self.server.url):
                    self.assertIn(expected, receipt)
                writes = [args for name, args in self.server.calls if name == "keytao_batch_remove_draft_items" and args.get("expected_target_digest")]
                self.assertEqual(len(writes), 1)
                self.assertEqual(writes[0]["ids"], [2, 3, 4, 5, 6])
        asyncio.run(run())

    def test_journal_survives_restart_and_repeated_cancel_is_idempotent(self):
        async def run():
            await self.write()
            self.store = DraftMutationClaimStore(self.directory.name + "/journal.db")
            self.store_factory.return_value = self.store
            receipt = await self.cancel()
            calls = len(self.server.calls)
            self.assertEqual(await self.cancel(), receipt)
            self.assertEqual(len(self.server.calls), calls)
        asyncio.run(run())

    def test_expired_operation_has_one_line_native_quote_confirmation(self):
        async def run():
            await self.write()
            scope = completed_draft_undo.conversation_scope(self.key)
            record = self.store.completed_operation(scope)
            record["completedAt"] = time.time() - 601
            self.store.update_completed_operation(scope, record["operationId"], "ready", "ready", record)
            prompt = await self.cancel()
            self.assertEqual(len(prompt.splitlines()), 1)
            self.assertIn("鎗@qx", prompt)
            self.assertIn(self.server.url, prompt)
            self.assertIn("是」或「否", prompt)
            for command in ("是", "否"):
                self.assertIsNone(await self.cancel(command, SimpleNamespace(is_to_bot=False, text=prompt)))
            receipt = await self.cancel("是", SimpleNamespace(is_to_bot=True, text=prompt))
            self.assertIn("撤销整笔", receipt)
            self.assertEqual(self.server.items, [s56_undo_row(1, "旧", "jq")])
        asyncio.run(run())

    def test_intervening_write_prompts_then_removes_only_original_rows(self):
        async def run():
            await self.write()
            extra = s56_undo_row(7, "新", "xb")
            self.server.items.append(extra)
            self.server.version += 1
            prompt = await self.cancel()
            self.assertIn("草稿又有变化", prompt)
            self.assertEqual(len(self.server.items), 7)
            receipt = await self.cancel("确认", SimpleNamespace(is_to_bot=True, text=prompt))
            self.assertIn("撤销整笔", receipt)
            self.assertEqual(self.server.items, [s56_undo_row(1, "旧", "jq"), extra])
        asyncio.run(run())

    def test_quote_no_and_untrusted_yes_never_write(self):
        async def run():
            await self.write()
            completed_draft_undo.start_operation_turn(self.key)
            prompt = await self.cancel()
            self.assertIsNone(await self.cancel("是", SimpleNamespace(is_to_bot=False, text=prompt)))
            self.assertIn("已保留", await self.cancel("否", SimpleNamespace(is_to_bot=True, text=prompt)))
            self.assertEqual(len(self.server.items), 6)
        asyncio.run(run())

    def test_actor_and_conversation_isolation(self):
        async def run():
            await self.write()
            for key in (harness.ConversationAddress.group("qq", "s56-group", "other"),
                        harness.ConversationAddress.group("qq", "other-group", "s56-undo")):
                self.assertIn("没有可核验", await self.cancel(key=key))
            self.assertEqual(len(self.server.items), 6)
        asyncio.run(run())

    def test_submitted_operation_recalls_before_exact_group_delete(self):
        async def run():
            await self.write(submitted=True)
            receipt = await self.cancel()
            self.assertIn("先撤回提审", receipt)
            self.assertFalse(self.server.submitted)
            self.assertEqual(self.server.items, [s56_undo_row(1, "旧", "jq")])
            names = [name for name, args in self.server.calls]
            self.assertLess(names.index("keytao_recall_batch"), names.index("keytao_batch_remove_draft_items"))
        asyncio.run(run())

    def test_failed_delete_has_truthful_receipt_and_durable_write_fence(self):
        async def run():
            await self.write()
            self.server.fail_delete = True
            receipt = await self.cancel()
            self.assertIn("尚未确认全部完成", receipt)
            self.assertNotIn("✅", receipt)
            self.assertTrue(self.store.actor_has_running_undo("qq", "s56-undo"))
            name = "keytao_shift_phrase_code"
            result = await completed_draft_undo.invoke_with_operation_journal(name, {"confirmed_plan_digest": "a" * 64}, self.server.get(name), self.server.get)
            self.assertFalse(result["success"])
            self.server.fail_delete = False
            self.assertIn("撤销整笔", await self.cancel())
            self.assertFalse(self.store.actor_has_running_undo("qq", "s56-undo"))
        asyncio.run(run())

    def test_changed_original_row_restores_original_weight(self):
        async def run():
            self.server.modify_old_on_shift = True
            await self.write()
            receipt = await self.cancel()
            self.assertIn("撤销整笔", receipt)
            self.assertEqual(completed_draft_undo._semantic_rows(self.server.items), completed_draft_undo._semantic_rows([s56_undo_row(1, "旧", "jq")]))
            self.assertEqual(len(self.server.restore_calls), 2)
        asyncio.run(run())

    def test_real_stage_handles_success_before_classifier_and_model(self):
        async def run():
            await self.write()
            completed_draft_undo.start_operation_turn(self.key)
            ctx = s56_undo_chat.TurnContext(SimpleNamespace(), SimpleNamespace(), "qq", "s56-undo")
            ctx.conv_key = self.key
            ctx.normalized_message_text = "取消"
            with patch.object(s56_undo_chat.conversation_state_store, "get_record", return_value=None), patch.object(
                s56_undo_chat.skills_manager, "get_tool_function", side_effect=self.server.get,
            ), patch.object(s56_undo_chat, "_classify_message_command_intent", AsyncMock(side_effect=AssertionError("model classifier"))):
                await s56_undo_chat._stage_handle_completed_draft_undo(ctx)
            self.assertIn("撤销整笔", ctx.response)
            stages = [stage.__name__ for stage in s56_undo_chat.STAGES]
            self.assertLess(stages.index("_stage_handle_completed_draft_undo"), stages.index("_stage_resolve_current_pending_scope"))
        asyncio.run(run())

    def test_submit_only_cancel_reverts_status_and_keeps_rows(self):
        async def run():
            completed_draft_undo.start_operation_turn(self.key)
            name = "keytao_submit_batch"
            await completed_draft_undo.invoke_with_operation_journal(name, {"confirmed": True, "batch_id": self.server.batch_id}, self.server.get(name), self.server.get)
            receipt = await self.cancel()
            self.assertIn("已撤销本次提审", receipt)
            self.assertEqual(self.server.items, [s56_undo_row(1, "旧", "jq")])
            self.assertFalse(any(name == "keytao_batch_remove_draft_items" for name, _ in self.server.calls))
        asyncio.run(run())

    def test_missing_rows_without_actual_receipt_cannot_claim_success(self):
        async def run():
            await self.write()
            self.server.fail_delete = True
            await self.cancel()
            self.server.items = [s56_undo_row(1, "旧", "jq")]
            receipt = await self.cancel()
            self.assertIn("没有本次撤销的成功回执", receipt)
            self.assertNotIn("✅", receipt)
        asyncio.run(run())

    def test_version_changes_after_quote_requires_fresh_confirmation(self):
        async def run():
            await self.write()
            self.server.version += 1
            prompt = await self.cancel()
            self.server.version += 1
            receipt = await self.cancel("是", SimpleNamespace(is_to_bot=True, text=prompt))
            self.assertIn("草稿又有变化", receipt)
            self.assertNotIn("✅", receipt)
            self.assertEqual(len(self.server.items), 6)
        asyncio.run(run())

    def test_questions_negation_and_reported_text_do_not_undo(self):
        async def run():
            await self.write()
            for message in ("取消？", "撤销?", "不要取消", "别撤销", "他说取消", "「回滚」", "`撤销`", "如果取消", "是否撤回"):
                receipt = await self.cancel(message)
                self.assertNotIn("✅", receipt or "")
                self.assertEqual(len(self.server.items), 6, message)
            self.assertFalse(any(name == "keytao_batch_remove_draft_items" for name, _ in self.server.calls))
        asyncio.run(run())

    def test_removed_draft_restores_nullable_weight_seal_and_remark(self):
        async def run():
            original = {**s56_undo_row(1, "旧", "jq"), "weight": None, "remark": "Original review annotation"}
            self.server.items = [copy.deepcopy(original)]
            self.server.remove_old_on_shift = True
            await self.write()
            receipt = await self.cancel()
            self.assertIn("撤销整笔", receipt)
            self.assertEqual(completed_draft_undo._semantic_rows(self.server.items), completed_draft_undo._semantic_rows([original]))
            self.assertNotEqual(self.server.items[0]["id"], original["id"])
            self.assertNotIn("weight", self.server.restore_calls[-1][0][0])
        asyncio.run(run())

    def test_restore_failure_preserves_partial_state_and_retries_only_definite_rejection(self):
        async def run():
            self.server.remove_old_on_shift = True
            await self.write()
            self.server.restore_failure = "rejected"
            receipt = await self.cancel()
            self.assertIn("原草稿恢复尚未全部确认", receipt)
            self.assertNotIn("✅", receipt)
            self.assertTrue(self.store.actor_has_running_undo("qq", "s56-undo"))
            self.server.restore_failure = ""
            receipt = await self.cancel()
            self.assertIn("撤销整笔", receipt)
            self.assertEqual(len(self.server.items), 1)
        asyncio.run(run())

    def test_restore_lost_reply_is_reconciled_without_duplicate_write(self):
        async def run():
            self.server.remove_old_on_shift = True
            await self.write()
            self.server.restore_failure = "lost_reply"
            receipt = await self.cancel()
            self.assertIn("结果不确定", receipt)
            self.assertNotIn("✅", receipt)
            writes = len(self.server.restore_calls)
            receipt = await self.cancel()
            self.assertIn("撤销整笔", receipt)
            self.assertEqual(len(self.server.restore_calls), writes)
            self.assertEqual(len(self.server.items), 1)
        asyncio.run(run())

    def test_absence_cas_accepts_only_empty_version_zero_before_first_write(self):
        async def run():
            identity = {"platform": "qq", "platform_id": "s56-undo"}
            async def absent(**_args):
                return {"success": True, "batchId": None, "contentVersion": 0, "items": []}
            self.assertIsNone(await completed_draft_undo._read_snapshot(absent, identity, "preview-only-uuid"))
            snapshot = await completed_draft_undo._read_snapshot(absent, identity, "preview-only-uuid", allow_absence=True)
            self.assertEqual(snapshot["batchId"], "")
            for version in (True, 1):
                async def invalid(**_args):
                    return {"success": True, "batchId": None, "contentVersion": version, "items": []}
                self.assertIsNone(await completed_draft_undo._read_snapshot(invalid, identity, "preview-only-uuid", allow_absence=True))
            completed_draft_undo.start_operation_turn(self.key)
            self.server.items = []
            self.server.version = 0
            async def first_write(**_args):
                self.server.items = copy.deepcopy(CHAIN)
                self.server.version = 1
                return {"success": True, "batchId": self.server.batch_id, "batchUrl": self.server.url}
            def get_tool(name):
                if name == "keytao_list_draft_items":
                    async def listed(**args):
                        if self.server.version == 0:
                            return await absent(**args)
                        return await self.server.get(name)(**args)
                    return listed
                return self.server.get(name)
            result = await completed_draft_undo.invoke_with_operation_journal(
                "keytao_create_phrase", {"confirmed": True, "batch_id": "preview-only-uuid", "expected_content_version": 0}, first_write, get_tool,
            )
            self.assertTrue(result["success"])
            self.assertIn("撤销整笔", await self.cancel())
            self.assertEqual(self.server.items, [])
        asyncio.run(run())

    def test_pure_draft_removal_undo_restores_removed_rows(self):
        async def run():
            completed_draft_undo.start_operation_turn(self.key)
            name = "keytao_batch_remove_draft_items"
            preview = await self.server.get(name)(ids=[1], batch_id=self.server.batch_id)
            result = await completed_draft_undo.invoke_with_operation_journal(name, {
                "ids": [1], "batch_id": self.server.batch_id,
                "expected_target_digest": preview["targetDigest"],
                "expected_content_version": self.server.version, "expected_targets": preview["targets"],
            }, self.server.get(name), self.server.get)
            self.assertTrue(result["success"])
            self.assertEqual(self.server.items, [])
            receipt = await self.cancel()
            self.assertIn("撤销整笔", receipt)
            self.assertEqual(completed_draft_undo._semantic_rows(self.server.items), completed_draft_undo._semantic_rows([s56_undo_row(1, "旧", "jq")]))
        asyncio.run(run())

    def test_raw_metadata_mismatch_blocks_write_without_guessing_original_state(self):
        async def run():
            completed_draft_undo.start_operation_turn(self.key)
            with patch.object(completed_draft_undo, "_fetch_raw_item", AsyncMock(return_value={**s56_undo_row(1, "旧", "jq"), "batchId": "other-batch"})):
                name = "keytao_shift_phrase_code"
                result = await completed_draft_undo.invoke_with_operation_journal(name, {"confirmed_plan_digest": "a" * 64}, self.server.get(name), self.server.get)
            self.assertFalse(result["success"])
            self.assertEqual(self.server.items, [s56_undo_row(1, "旧", "jq")])
            self.assertFalse(any(name == "keytao_shift_phrase_code" for name, _ in self.server.calls))
        asyncio.run(run())

    def test_submitted_intervening_version_uses_one_line_quote_confirmation(self):
        async def run():
            await self.write(submitted=True)
            self.server.version += 1
            prompt = await self.cancel()
            self.assertIn("提审后又有变化", prompt)
            self.assertEqual(len(prompt.splitlines()), 1)
            self.assertTrue(self.server.submitted)
            receipt = await self.cancel("是", SimpleNamespace(is_to_bot=True, text=prompt))
            self.assertIn("先撤回提审", receipt)
            self.assertEqual(self.server.items, [s56_undo_row(1, "旧", "jq")])
        asyncio.run(run())

    def test_restore_change_delete_keeps_the_original_live_target_binding(self):
        async def run():
            target = {"id": 123, "word": "强", "code": "qx", "type": "Single", "weight": 10,
                      "remark": None, "status": "Finish", "userId": 3}
            fingerprint = hashlib.sha256(json.dumps(target, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            original = {**s56_undo_row(1, "强", "qx", "Delete"), "_targetPhraseId": 123, "_targetFingerprint": fingerprint}
            phrase = {**target, "user": {"id": 3}}
            with patch.object(completed_draft_undo.http_client, "keytao_json", AsyncMock(return_value={"phrases": [phrase], "pagination": {"totalPages": 1}})):
                self.assertTrue(await completed_draft_undo._restore_targets_unchanged([original]))
                changed = {**original, "_targetFingerprint": "0" * 64}
                self.assertFalse(await completed_draft_undo._restore_targets_unchanged([changed]))
                self.assertFalse(await completed_draft_undo._restore_targets_unchanged([{**original, "_targetPhraseId": None}]))
        asyncio.run(run())

    def test_recent_write_receipt_capability_is_consumed_by_undo_only(self):
        async def run():
            await self.write()
            completed_draft_undo.start_operation_turn(self.key)
            ctx = s56_undo_chat.TurnContext(SimpleNamespace(), SimpleNamespace(), "qq", "s56-undo")
            ctx.conv_key, ctx.normalized_message_text = self.key, "取消"
            state = harness.PendingToolConfirm("keytao_submit_batch", {"_recent_own_write": True, "batch_id": self.server.batch_id, "_recent_batch_ids": [self.server.batch_id]})
            with patch.object(s56_undo_chat.conversation_state_store, "get_record", return_value=SimpleNamespace(state=state)), patch.object(
                s56_undo_chat.conversation_state_store, "delete",
            ) as delete, patch.object(s56_undo_chat.skills_manager, "get_tool_function", side_effect=self.server.get):
                await s56_undo_chat._stage_handle_completed_draft_undo(ctx)
                self.assertIn("撤销整笔", ctx.response)
                delete.assert_called_once_with(self.key)
                delete.reset_mock()
                ctx.normalized_message_text, ctx.response = "提交", None
                await s56_undo_chat._stage_handle_completed_draft_undo(ctx)
                self.assertIsNone(ctx.response)
                delete.assert_not_called()
        asyncio.run(run())

    def test_inflight_recall_uses_raw_batch_status_when_list_omits_status(self):
        async def run():
            await self.write(submitted=True)
            actual = self.server.get
            draft_tools = harness._draft_tools
            async def lost(**args):
                if "expected_content_version" not in args:
                    return await actual("keytao_recall_batch")(**args)
                self.store.begin("qq", "s56-undo", "recall", {"batchId": self.server.batch_id, "contentVersion": args["expected_content_version"]})
                await actual("keytao_recall_batch")(**args)
                return {"success": False, "uncertain": True, "batchId": self.server.batch_id}
            completed_draft_undo.start_operation_turn(self.key)
            first = await completed_draft_undo.handle_completed_undo(self.key, "取消", SimpleNamespace(), lambda name: lost if name == "keytao_recall_batch" else actual(name))
            self.assertNotIn("✅", first)
            async def raw(item_id):
                item = next(item for item in self.server.items if item["id"] == item_id)
                return {**item, "batchId": self.server.batch_id, "batch": {"id": self.server.batch_id, "status": "Draft"}}
            with patch.object(completed_draft_undo, "_fetch_raw_item", side_effect=raw), patch.object(
                draft_tools, "_draft_mutation_claims", return_value=self.store,
            ), patch.object(draft_tools, "get_bot_token", return_value="fixture-token"), patch.object(
                draft_tools, "keytao_list_draft_items", side_effect=actual("keytao_list_draft_items"),
            ), patch.object(draft_tools.httpx, "AsyncClient", side_effect=AssertionError("unexpected network"), create=True):
                completed_draft_undo.start_operation_turn(self.key)
                receipt = await completed_draft_undo.handle_completed_undo(self.key, "取消", SimpleNamespace(), lambda name: draft_tools.keytao_recall_batch if name == "keytao_recall_batch" else actual(name))
            self.assertIn("撤销整笔", receipt)
            self.assertEqual(self.server.items, [s56_undo_row(1, "旧", "jq")])
            self.assertEqual(self.store.get("qq", "s56-undo")["status"], "resolved")
        asyncio.run(run())

    def test_server_review_marker_refuses_undo_before_recall_or_delete(self):
        async def run():
            for marker in ("--- miao-review:start ---", "--- miao-review:end ---"):
                for submitted in (False, True):
                    with self.subTest(marker=marker, submitted=submitted):
                        self.server = DraftServer()
                        self.server.items[0]["remark"] = "Persisted review " + marker
                        self.server.modify_old_on_shift = True
                        await self.write(submitted=submitted)
                        original = copy.deepcopy(self.server.items)
                        calls_before = len(self.server.calls)
                        receipt = await self.cancel()
                        self.assertNotIn("✅", receipt)
                        self.assertIn("审核记录", receipt)
                        self.assertEqual(self.server.items, original)
                        self.assertEqual(self.server.submitted, submitted)
                        self.assertEqual(len(self.server.calls), calls_before)
                        self.assertEqual(self.server.restore_calls, [])
                        self.assertFalse(self.store.actor_has_running_undo("qq", "s56-undo"))
        asyncio.run(run())

# Source: test_s56_undo_review.py
class CompletedUndoReviewTests(unittest.TestCase):
    setUp = CompletedUndoTests_setUp
    tearDown = CompletedUndoTests_tearDown
    write = CompletedUndoTests_write
    cancel = CompletedUndoTests_cancel

    def test_definite_pre_delete_preview_rejection_does_not_lock_actor(self):
        async def run():
            await self.write()
            actual = self.server.get

            def get_tool(name):
                if name == "keytao_batch_remove_draft_items":
                    async def rejected(**_args):
                        return {"success": False, "batchId": self.server.batch_id,
                                "contentVersion": self.server.version, "message": "fixture denied preview"}
                    return rejected
                return actual(name)

            completed_draft_undo.start_operation_turn(self.key)
            reply = await completed_draft_undo.handle_completed_undo(self.key, "取消", SimpleNamespace(), get_tool)
            self.assertNotIn("✅", reply)
            self.assertEqual(6, len(self.server.items))
            self.assertFalse(self.store.actor_has_running_undo("qq", "s56-undo"), reply)
        asyncio.run(run())

    def test_lost_recall_reply_resumes_from_actual_tool_claim_reconciliation(self):
        async def run():
            await self.write(submitted=True)
            actual = self.server.get
            draft_tools = harness._draft_tools
            reconcile = False

            def get_tool(name):
                if name != "keytao_recall_batch":
                    return actual(name)
                if reconcile:
                    return draft_tools.keytao_recall_batch

                async def lost_reply(**args):
                    if "expected_content_version" not in args:
                        return await actual(name)(**args)
                    fingerprint = self.store.begin("qq", "s56-undo", "recall", {
                        "batchId": self.server.batch_id,
                        "contentVersion": args["expected_content_version"],
                    })
                    self.assertIsNotNone(fingerprint)
                    await actual(name)(**args)
                    return {"success": False, "uncertain": True, "batchId": self.server.batch_id,
                            "message": "fixture lost recall reply"}
                return lost_reply

            completed_draft_undo.start_operation_turn(self.key)
            reply = await completed_draft_undo.handle_completed_undo(self.key, "取消", SimpleNamespace(), get_tool)
            self.assertNotIn("✅", reply)
            self.assertFalse(self.server.submitted)
            self.assertTrue(self.store.actor_has_running_undo("qq", "s56-undo"))
            reconcile = True

            async def listed(*_identity, **args):
                return {**await actual("keytao_list_draft_items")(**args), "status": "Draft"}

            with (
                patch.object(draft_tools, "_draft_mutation_claims", return_value=self.store),
                patch.object(draft_tools, "get_bot_token", return_value="fixture-token"),
                patch.object(draft_tools, "keytao_list_draft_items", side_effect=listed),
                patch.object(draft_tools.httpx, "AsyncClient", side_effect=AssertionError("unexpected network"), create=True),
            ):
                completed_draft_undo.start_operation_turn(self.key)
                reply = await completed_draft_undo.handle_completed_undo(self.key, "取消", SimpleNamespace(), get_tool)
            self.assertIn("撤销整笔", reply)
            self.assertEqual(self.server.items, [s56_undo_row(1, "旧", "jq")])
            self.assertFalse(self.store.actor_has_running_undo("qq", "s56-undo"))
        asyncio.run(run())

    def test_restore_does_not_rebind_original_delete_after_target_changes(self):
        async def run():
            target = {"id": 90, "word": "旧", "code": "jq", "type": "Single", "weight": 10,
                      "remark": None, "status": "Finish", "userId": 4}

            def fingerprint():
                return hashlib.sha256(json.dumps(target, ensure_ascii=False, sort_keys=True,
                                                 separators=(",", ":")).encode()).hexdigest()

            self.server.items = [{**s56_undo_row(1, "旧", "jq", "Delete"),
                                  "targetPhraseId": target["id"], "targetFingerprint": fingerprint()}]
            self.server.remove_old_on_shift = True
            await self.write()
            original_restore = completed_draft_undo._restore_request

            async def phrases(_method, _path, **_kwargs):
                self.assertEqual(_path, "/api/phrases/by-word")
                return {"phrases": [{**target, "user": {"id": target["userId"]}}],
                        "pagination": {"totalPages": 1}}

            async def changed_target_restore(identity, batch_id, items, **ticket):
                if ticket.get("confirmed") is not True:
                    target["weight"] = 11
                result = await original_restore(identity, batch_id, items, **ticket)
                if ticket.get("confirmed") is True and result.get("success") is True:
                    for item in self.server.items:
                        if item["id"] >= 100:
                            item["targetPhraseId"] = target["id"]
                            item["targetFingerprint"] = fingerprint()
                return result

            with (patch.object(completed_draft_undo.http_client, "keytao_json", side_effect=phrases),
                  patch.object(completed_draft_undo, "_restore_request", side_effect=changed_target_restore)):
                reply = await self.cancel()
            self.assertNotIn("✅", reply)
            confirmed = [ticket for _items, ticket in self.server.restore_calls if ticket.get("confirmed")]
            self.assertEqual([], confirmed, "undo re-bound a stale Delete onto the changed live phrase")
        asyncio.run(run())

# Source: test_s57_reorder.py
class SameCodeReorderTests(unittest.TestCase):
    def test_non_same_code_relative_request_remains_available_to_existing_route(self):
        from keytao_bot.harness.authorization_grammar import parse_eviction_modified_add
        from keytao_bot.utils.same_code_reorder import try_handle_same_code_reorder
        message = "把小像放到销项前面"
        existing_grammar = parse_eviction_modified_add(message)
        self.assertIsNotNone(existing_grammar)
        self.assertEqual(
            (existing_grammar.word, existing_grammar.named_occupant), ("小像", "销项"),
        )

        async def run():
            for first_rows in ([], [{"word": "小像", "code": "abcdo", "type": "Phrase", "weight": 100}]):
                async def call(name, args, *_args, **_kwargs):
                    if name == "keytao_lookup_by_words_batch":
                        return json.dumps({"success": True, "results": [
                            {"word": "小像", "phrases": first_rows},
                            {"word": "销项", "phrases": [{"word": "销项", "code": "abcd", "type": "Phrase", "weight": 100}]},
                        ]})
                    if name == "keytao_list_draft_items":
                        return json.dumps({"success": True, "batchId": "", "contentVersion": 0, "items": []})
                    raise AssertionError("non-same-code request entered the same-code planner")
                with patch.object(s57_reorder_commands, "call_tool_function", side_effect=call):
                    self.assertIsNone(await try_handle_same_code_reorder(message, "qq", "s57-relative"))
                    explicit = await try_handle_same_code_reorder(
                        "把「小像 abcd」放到「销项 abcd」前面", "qq", "s57-relative",
                    )
                    self.assertIn("未能锁定", explicit)
        asyncio.run(run())

    def test_applied_single_weight_receipt_lists_both_verified_changes(self):
        # Minimized successful tool result captured from S57 target run 3.
        result = {
            "success": True, "batchId": "s57-receipt", "successCount": 2,
            "receipts": [{"step": "dictionary", "status": "applied", "changes": [], "itemCount": 2}],
            "shiftPlan": {
                "word": "嘢", "targetCode": "yeoiav", "scope": "same_code",
                "currentState": [
                    {"word": "嘢", "code": "yeoiav", "type": "Single", "weight": 11, "source": "live"},
                    {"word": "咽", "code": "yeoiav", "type": "Single", "weight": 10, "source": "live"},
                ],
                "proposedState": [
                    {"word": "嘢", "code": "yeoiav", "type": "Single", "weight": 10},
                    {"word": "咽", "code": "yeoiav", "type": "Single", "weight": 11},
                ],
                "shifted": [], "draftUpdates": [],
            },
            "draft_snapshot": {
                "success": True, "batchId": "s57-receipt", "contentVersion": 1,
                "items": [
                    {"action": "Change", "word": "嘢", "oldWord": "嘢", "code": "yeoiav", "type": "Single", "weight": 10},
                    {"action": "Change", "word": "咽", "oldWord": "咽", "code": "yeoiav", "type": "Single", "weight": 11},
                ],
            },
        }
        reply = s57_reorder_commands._format_ranked_shift_success(result)
        self.assertIn("嘢 yeoiav 权重 11→10", reply)
        self.assertIn("咽 yeoiav 权重 10→11", reply)
        self.assertNotIn("嘢 → yeoiav", reply)
        # A proposed state alone cannot prove that an individual row changed.
        result["draft_snapshot"]["items"].pop()
        self.assertNotIn("咽 yeoiav 权重 10→11", s57_reorder_commands._format_ranked_shift_success(result))

    def test_closed_parser_accepts_requested_forms_without_claiming_reported_text(self):
        from keytao_bot.utils.same_code_reorder import parse_same_code_reorder
        forms = (
            '提高"嘢    yeoiav"的优先级，使之位于"咽    yeoiav"之前',
            '提高＂嘢 yeoiav＂的优先级，使之位于＂咽 yeoiav＂之前',
            "提高「嘢 yeoiav」的优先级，使之位于「咽 yeoiav」之前",
            "把 嘢 排到 咽 前面", "把财宝排到财报之前",
            "嘢 和 咽 调换顺序", "嘢和咽对换", "嘢优先于咽",
            "对换财宝和财报的优先级",
        )
        for text in forms:
            with self.subTest(text=text):
                self.assertIsNotNone(parse_same_code_reorder(text))
        for text in (
            "不要把嘢排到咽前面", "他说把嘢排到咽前面", "把嘢排到咽前面？",
            "把嘢排到咽前面 删除咽", "嘢和嘢对换", "强制调序", "四字词语",
            '提高"嘢 yeoiav」的优先级，使之位于"咽 yeoiav"之前',
        ):
            with self.subTest(text=text):
                self.assertIsNone(parse_same_code_reorder(text))

    def test_full_chain_preserves_unmentioned_entries_and_typed_base(self):
        from keytao_bot.utils.same_code_reorder import try_handle_same_code_reorder
        async def run():
            for phrase_type, base, words in (("Single", 10, ("甲", "乙", "丙")), ("Phrase", 100, ("甲词", "乙词", "丙词"))):
                for tied in (False, True):
                    with self.subTest(phrase_type=phrase_type, tied=tied):
                        rows = [
                            {"word": word, "code": "abcd", "type": phrase_type, "weight": base if tied else base + index}
                            for index, word in enumerate(words)
                        ]
                        async def call(name, args, *_args, **_kwargs):
                            if name == "keytao_lookup_by_words_batch":
                                result = {"success": True, "results": [
                                    {"word": word, "phrases": [row for row in rows if row["word"] == word]}
                                    for word in args["words"]
                                ]}
                            elif name == "keytao_lookup_by_code":
                                result = {"success": True, "phrases": rows}
                            elif name == "keytao_list_draft_items":
                                result = {"success": True, "batchId": "", "contentVersion": 0, "items": []}
                            else:
                                raise AssertionError(name)
                            return json.dumps(result)
                        shift = AsyncMock(return_value="preview")
                        with (
                            patch.object(s57_reorder_commands, "call_tool_function", side_effect=call),
                            patch.object(s57_reorder_commands, "_execute_shift_to_code", shift),
                        ):
                            await try_handle_same_code_reorder(f"把{words[2]}排到{words[0]}前面", "qq", "s57-chain")
                        args = shift.call_args.kwargs
                        self.assertEqual(set(args["ordered_words"]), set(words))
                        self.assertLess(args["ordered_words"].index(words[2]), args["ordered_words"].index(words[0]))
                        self.assertEqual(args["expected_codes"], ["abcd"] * 3)
                        self.assertEqual(args["expected_weights"], None if tied else [base, base + 1, base + 2])
                        self.assertEqual(sum(line.startswith("常用度提示：") for line in args["evidence_lines"]), 1)
        asyncio.run(run())

    def test_explicit_same_code_scope_keeps_other_codes_untouched(self):
        async def run():
            rows = [
                {"word": "嘢", "code": "yeoiav", "type": "Single", "weight": 11},
                {"word": "嘢", "code": "yeoia", "type": "Single", "weight": 10},
                {"word": "咽", "code": "yeoiav", "type": "Single", "weight": 10},
                {"word": "其他词", "code": "yeoiav", "type": "Phrase", "weight": 100},
            ]
            async def words(requested):
                return {"success": True, "results": [
                    {"word": word, "phrases": [row for row in rows if row["word"] == word]}
                    for word in requested
                ]}
            async def codes(requested):
                return {"success": True, "results": [
                    {"code": code, "phrases": [row for row in rows if row["code"] == code]}
                    for code in requested
                ]}
            with (
                patch.object(s57_reorder_draft_tools, "_lookup_words_raw", side_effect=words),
                patch.object(s57_reorder_draft_tools, "_lookup_codes_raw", side_effect=codes),
                patch.object(s57_reorder_draft_tools, "_fetch_encode_candidates", AsyncMock(side_effect=AssertionError(
                    "an exact existing code weight change must not re-encode the item"
                ))),
                patch.object(s57_reorder_draft_tools, "keytao_list_draft_items", AsyncMock(return_value={
                    "success": True, "batchId": "", "contentVersion": 0, "items": [],
                })),
            ):
                plan = await s57_reorder_draft_tools._prepare_ranked_reorder_plan(
                    "qq", "s57-scope", ["嘢", "咽"], "yeoiav",
                    expected_codes=["yeoiav", "yeoiav"], expected_weights=[10, 11],
                )
            self.assertTrue(plan.get("success"), plan)
            self.assertEqual(
                [(item["word"], item["code"], item["weight"]) for item in plan["items"]],
                [("嘢", "yeoiav", 10), ("咽", "yeoiav", 11)],
            )
        asyncio.run(run())

    def test_incident_request_creates_real_single_weight_ticket(self):
        async def run():
            user = "s57-reorder"
            rows = [
                {"word": "咽", "code": "yeoiav", "type": "Single", "weight": 10},
                {"word": "嘢", "code": "yeoiav", "type": "Single", "weight": 11},
            ]
            snapshot = {"success": True, "batchId": "", "contentVersion": 0, "items": []}

            async def words(requested):
                return {"success": True, "results": [
                    {"word": word, "phrases": [row for row in rows if row["word"] == word]}
                    for word in requested
                ]}

            async def codes(requested):
                return {"success": True, "results": [
                    {"code": code, "phrases": [row for row in rows if row["code"] == code]}
                    for code in requested
                ]}

            async def call(name, args, *_args, **_kwargs):
                self.assertIn(name, {
                    **harness._lookup_tools.TOOL_FUNCTIONS,
                    **s57_reorder_draft_tools.TOOL_FUNCTIONS,
                }, "route dispatched an unregistered production tool")
                if name == "keytao_lookup_by_words_batch":
                    result = await words(args["words"])
                elif name == "keytao_lookup_by_code":
                    result = {"success": True, "phrases": rows}
                elif name == "keytao_list_draft_items":
                    result = snapshot
                elif name == "keytao_shift_phrase_code":
                    result = await s57_reorder_draft_tools.keytao_shift_phrase_code("qq", user, **args)
                else:
                    raise AssertionError(name)
                return json.dumps(result, ensure_ascii=False)

            with (
                patch.object(s57_reorder_commands, "call_tool_function", side_effect=call),
                patch.object(s57_reorder_draft_tools, "_lookup_words_raw", side_effect=words),
                patch.object(s57_reorder_draft_tools, "_lookup_codes_raw", side_effect=codes),
                patch.object(s57_reorder_draft_tools, "keytao_list_draft_items", AsyncMock(return_value=snapshot)),
                patch.object(s57_reorder_draft_tools, "_fetch_encode_candidates", AsyncMock(return_value={
                    "success": True, "candidateCodes": ["ye"],
                })),
                patch.object(s57_reorder_draft_tools, "_keytao_strict_batch_add_to_draft", AsyncMock(return_value={
                    "success": False, "requiresConfirmation": True,
                    "warningDigest": "d" * 64, "warnings": [],
                })),
                patch("keytao_bot.utils.keytao_review._query_commonness_reference", side_effect=lambda word: {
                    "available": True, "attested": True, "corpusFrequency": 1000 if word == "咽" else 10,
                    "dictionaryPresenceCount": 2,
                }),
            ):
                reply = await s57_reorder_commands._try_handle_draft_management_command(
                    '提高"嘢    yeoiav"的优先级，使之位于"咽    yeoiav"之前',
                    "qq", user, command_intent=s57_reorder_commands.MessageCommandIntent(),
                )
            self.assertIsNotNone(reply, "explicit priority request must create a deterministic preview")
            record = s57_reorder_chat.conversation_state_store.get_record(("qq", user))
            self.assertIsNotNone(record, reply)
            self.assertTrue(s57_reorder_commands.server_warning_ticket_is_complete(record.state), reply)
            plan = record.state.args["_pending_display"]["shiftPlan"]
            self.assertEqual(
                [(item["word"], item["weight"], item["type"]) for item in plan["proposedState"]],
                [("嘢", 10, "Single"), ("咽", 11, "Single")],
            )
            self.assertIn("常用度提示：「咽」较「嘢」更常用：语料频次 1000 vs 10", reply)
            self.assertTrue(s57_reorder_chat._advertised_reply_matches_live_record(reply, record), reply)
            delivered = s57_reorder_chat._enforce_advertised_reply_contract(reply, ("qq", user))
            self.assertIn("常用度提示：「咽」较「嘢」更常用：语料频次 1000 vs 10", delivered)

        asyncio.run(run())

# Source: test_s58_bridge.py
class BridgeTests(unittest.IsolatedAsyncioTestCase):
    setUp = BridgeTests_setUp

    call = BridgeTests_call

    async def test_real_assent_stage_preserves_stale_evidence_without_inventing_expiry(self):
        chat = harness.openai_chat_module
        from keytao_bot.utils.offered_options import render_missing_option_ticket

        key = harness.ConversationAddress.private("qq", "s58-expired")
        now = [100.0]
        store = MemoryConversationStateStore(pending_ttl_seconds=1, clock=lambda: now[0])
        state = server_warning_pending_state(PendingToolConfirm(
            "keytao_shift_phrase_code", {"word": "奈飞", "target_code": "nhfw"},
        ), preview())
        store.set(key, state)
        self.assertEqual(store.get_record(key).expires_at, 101.0)
        proposal = chat._chat_render._format_server_bound_confirmation_prompt(state)
        now[0] = 102.0
        self.assertIsNone(store.get_record(key))
        cases = (
            ([{"role": "assistant", "content": proposal}], True, "奈飞"),
            ([{"role": "assistant", "content": "此前操作曾等待确认，现在是否继续？如需执行请回复「确认」。"}], True, "上一条提议"),
            ([], False, "当前没有待执行的操作"),
            ([{"role": "assistant", "content": render_missing_option_ticket([])}], False, "当前没有待执行的操作"),
        )
        for history, stale, expected in cases:
            with self.subTest(stale=stale, history=history):
                ctx = chat.TurnContext(None, None, "qq", "s58-expired")
                ctx.conv_key = key
                ctx.normalized_message_text = "确认"
                ctx.memory_context = harness.ChatMemoryContext(
                    platform="qq", user_id="s58-expired", space_type="private", space_id="s58-expired",
                )
                with patch.object(chat, "conversation_state_store", store), patch.object(
                    chat, "draft_operation_coordinator", harness.DraftOperationCoordinator(),
                ), patch.object(chat, "get_history", return_value=history), patch.object(
                    chat, "remember_conversation",
                ), patch.object(chat, "_finish_ai_chat_response", AsyncMock()) as finish, patch.object(
                    chat, "get_ai_response_core", AsyncMock(side_effect=AssertionError("assent used model")),
                ) as model, patch.object(chat, "call_tool_function", AsyncMock(side_effect=AssertionError("assent wrote"))) as tool:
                    self.assertTrue(await chat._stage_claim_offered_answer(ctx))
                    self.assertEqual(chat._prepare_user_facing_reply(ctx.response, ctx.memory_context), ctx.response)
                    self.assertEqual(finish.await_count, 1)
                    model.assert_not_awaited()
                    tool.assert_not_awaited()
                self.assertEqual(len(ctx.response.splitlines()), 1)
                self.assertIn(expected, ctx.response)
                self.assertEqual("已过期或记录已不存在" in ctx.response, stale)
                self.assertIn("未写入", ctx.response)
                self.assertIsNone(store.get_record(key))
                self.assertEqual(render_missing_option_ticket([{"role": "assistant", "content": ctx.response}]), ctx.response)

    async def test_early_assent_stage_defers_live_referenced_and_scoped_context(self):
        chat = harness.openai_chat_module
        key = harness.ConversationAddress.group("qq", "s58-scope", "owner")
        state = server_warning_pending_state(PendingToolConfirm(
            "keytao_shift_phrase_code", {"word": "奈飞", "target_code": "nhfw"},
        ), preview())
        for kind in ("live", "reply", "active", "other_owner"):
            with self.subTest(kind=kind):
                store = MemoryConversationStateStore()
                coordinator = harness.DraftOperationCoordinator()
                ctx = chat.TurnContext(None, None, "qq", "owner")
                ctx.conv_key, ctx.space_key = key, key.space_key
                ctx.memory_context = SimpleNamespace(space_type="group")
                ctx.normalized_message_text = "确认"
                if kind == "live":
                    store.set(key, state, space_key=key.space_key)
                elif kind == "reply":
                    ctx.reply_reference = chat.ReplyReferenceInfo(is_reply=True, is_to_bot=True, text="quoted proposal")
                elif kind == "active":
                    self.assertIsNotNone(coordinator.begin(key, "add"))
                else:
                    foreign = harness.ConversationAddress.group("qq", "s58-scope", "other")
                    store.set(foreign, state, space_key=key.space_key)
                with patch.object(chat, "conversation_state_store", store), patch.object(
                    chat, "draft_operation_coordinator", coordinator,
                ), patch.object(chat, "get_history", return_value=[]), patch.object(
                    chat, "_finish_ai_chat_response", AsyncMock(),
                ) as finish:
                    self.assertFalse(await chat._stage_claim_offered_answer(ctx))
                    finish.assert_not_awaited()

    async def test_unknown_verb_builds_only_a_complete_server_proposal(self):
        result = await self.call()
        self.assertTrue(result.get("grammar_gap_bridged"), result)
        self.assertEqual([name for name, _ in self.calls], ["keytao_lookup_by_words_batch", "keytao_shift_phrase_code"])
        state = server_warning_pending_state(PendingToolConfirm(
            function_name="keytao_shift_phrase_code", args=result["grammarGapArguments"],
        ), result)
        self.assertTrue(server_warning_ticket_is_complete(state))
        self.assertEqual(state.args["target_type"], "Phrase")
        self.assertFalse(result["success"])

    async def test_unverifiable_arguments_never_reach_the_planner(self):
        for args in (
            {"word": "外部词", "target_code": "nhfw"},
            {"word": "奈飞", "target_code": "nhfwa"},
            {"word": "奈飞", "target_code": "nhfw", "confirmed_plan_digest": "a" * 64},
            {"word": "奈飞", "target_code": "nhfw", "additional_items": []},
            {"word": "奈飞", "target_code": "NHFW"},
        ):
            with self.subTest(args=args):
                self.calls.clear()
                self.assertFalse((await self.call(args)).get("grammar_gap_bridged"))
                self.assertEqual(self.calls, [])
        self.entries = []
        result = await self.call()
        self.assertFalse(result.get("grammar_gap_bridged"))
        self.assertEqual([name for name, _ in self.calls], ["keytao_lookup_by_words_batch"])

    async def test_questions_negation_reported_text_and_attachments_cannot_bridge(self):
        for message in ("把 奈飞 迁到 nhfw？", "不要把 奈飞 迁到 nhfw", "他说把 奈飞 迁到 nhfw", "例句：把 奈飞 迁到 nhfw", "把 奈飞 迁到 nhfw，但保持浓厚氛围"):
            with self.subTest(message=message):
                self.assertFalse((await self.call(context=replace(self.context, current_message=message))).get("grammar_gap_bridged"))
                self.assertEqual(self.calls, [])
        self.assertFalse((await self.call(context=replace(self.context, attachment_context=True))).get("grammar_gap_bridged"))
        self.assertEqual(self.calls, [])

    async def test_missing_seal_and_failed_plan_do_not_bridge(self):
        for key in ("planDigest", "warningDigest", "shiftPlan"):
            with self.subTest(key=key):
                self.proposal = preview()
                self.proposal.pop(key)
                self.assertFalse((await self.call()).get("grammar_gap_bridged"))
        self.proposal = {"success": False, "message": "invalid code"}
        self.assertFalse((await self.call()).get("grammar_gap_bridged"))

    async def test_real_orchestrator_saves_before_render_and_stops_the_model_loop(self):
        from keytao_bot.harness.orchestrator import AgentOrchestrator, AgentRequestContext, AgentRuntimeConfig

        store = MemoryConversationStateStore()
        client = harness._FakeClient([harness._FakeAIResponse(
            "tool_calls", content="浓厚氛围会去错误的编码 nhfwz",
            tool_calls=[SimpleNamespace(id="s58-move", type="function", function=SimpleNamespace(
                name="keytao_shift_phrase_code", arguments=json.dumps({"word": "奈飞", "target_code": "nhfw"}),
            ))],
        )])
        class ShiftSkills(harness._FakeToolSkillsManager):
            def get_tools(self):
                return [{"type": "function", "function": {
                    "name": "keytao_shift_phrase_code", "description": "Preview a shift",
                    "parameters": {"type": "object", "properties": {
                        "word": {"type": "string"}, "target_code": {"type": "string"},
                    }, "required": ["word", "target_code"]},
                }}]
        orchestrator = AgentOrchestrator(
            client_factory=lambda: client,
            runtime=AgentRuntimeConfig("fake-model", 1000, 0.0, 10.0),
            skills_manager=ShiftSkills(), tool_executor=self.executor,
            state_store=store, bind_help_text="bind help", system_prompt_core="system",
        )
        context = AgentRequestContext(platform="qq", user_id="s58-bridge", mutations_allowed=False)
        reply = await orchestrator.run(self.context.current_message, context)
        record = store.get_record(context.conversation_address)
        self.assertIsNotNone(record, reply)
        self.assertTrue(server_warning_ticket_is_complete(record.state))
        self.assertFalse(record.execution_id)
        self.assertEqual(len(client.completions.calls), 1)
        self.assertNotIn("nhfwz", reply)
        for value in ("奈飞", "浓厚氛围", "nhfwv", "nhfw", "nhfwa", "确认"):
            self.assertIn(value, reply)
        self.assertEqual(len(self.calls), 2)
        foreign = replace(context, user_id="s58-foreign")
        self.assertIsNone(store.get_record(foreign.conversation_address))

    def test_lost_ticket_copy_names_displayed_plan_without_resurrecting_it(self):
        from keytao_bot.utils.offered_options import render_missing_option_ticket
        reply = render_missing_option_ticket([{
            "role": "assistant", "content": "🔁 调整计划：\n• 奈飞：nhfwv→nhfw\n• 浓厚氛围：nhfw→nhfwa\n回复「确认」执行。",
        }])
        self.assertEqual(len(reply.splitlines()), 1)
        self.assertIn("奈飞", reply)
        self.assertIn("nhfw", reply)
        self.assertIn("没有可执行", reply)

    def test_recordless_assents_do_not_turn_previous_refusals_into_proposals(self):
        from keytao_bot.utils.offered_options import render_missing_option_ticket
        reply = render_missing_option_ticket([])
        self.assertNotIn("上一条提议", reply)
        for _ in range(5):
            repeated = render_missing_option_ticket([{"role": "assistant", "content": reply}])
            self.assertEqual(repeated, reply)
        self.assertEqual(render_missing_option_ticket([
            {"role": "assistant", "content": "你好。"},
        ]), reply)
        prior = render_missing_option_ticket([{
            "role": "assistant", "content": "🔁 调整计划：\n• 奈飞：nhfwv→nhfw\n回复「确认」执行。",
        }])
        self.assertEqual(render_missing_option_ticket([
            {"role": "assistant", "content": prior},
        ]), prior)

# Source: test_s58_explicit_destination.py
class ExplicitDestinationTests(unittest.IsolatedAsyncioTestCase):
    setUp = ExplicitDestinationTests_setUp

    encode = ExplicitDestinationTests_encode

    lookup_words = ExplicitDestinationTests_lookup_words

    lookup_codes = ExplicitDestinationTests_lookup_codes

    strict = ExplicitDestinationTests_strict

    invoke = ExplicitDestinationTests_invoke

    async def test_noncandidate_suffix_previews_and_replays_the_same_sealed_plan(self):
        preview = await self.invoke("nhfwzz")
        self.assertTrue(preview.get("requiresConfirmation"), preview)
        self.assertEqual(self.draft["items"], [])
        items = preview["shiftPlan"]["items"]
        custom = next(item for item in items if item["word"] == "浓厚氛围" and item["action"] == "Create")
        self.assertEqual(custom["code"], "nhfwzz")
        self.assertEqual(custom["_reviewed_pinyin"], "nóng hòu fēn wéi")
        self.assertTrue(custom["needsManualReview"])
        self.assertIn("形码 zz 未能核验", custom["remark"])
        self.assertIn("「浓厚氛围」：形码 zz 未能核验，需管理员复核", preview["shiftPlan"]["evidenceLines"])
        from keytao_bot.plugins.chat_commands import _format_server_warning_confirmation
        preview["shiftPlan"]["evidenceLines"].insert(0, "目标编码均来自本轮指令，并由服务端候选链重新核验")
        rendered = _format_server_warning_confirmation("keytao_shift_phrase_code", preview)
        self.assertIn("「浓厚氛围」@ nhfwzz：形码 zz 未能核验，需管理员复核", rendered)
        receipt = await self.invoke(
            "nhfwzz", confirmed_plan_digest=preview["planDigest"], batch_id="",
            expected_content_version=0, expected_warning_digest=preview["warningDigest"],
        )
        self.assertTrue(receipt.get("success"), receipt)
        self.assertEqual(receipt["planDigest"], preview["planDigest"])
        written = next(item for item in self.draft["items"] if item["word"] == "浓厚氛围" and item["action"] == "Create")
        self.assertTrue(written["needsManualReview"])
        self.assertIn("形码 zz 未能核验", written["remark"])
        self.assertNotIn("_reviewed_pinyin", written)
        self.assertEqual([call[1].get("confirmed", False) for call in self.strict_calls], [False, True])

    async def test_invalid_prefix_missing_and_ambiguous_readings_never_preview(self):
        for code, review in (
            ("zzzzzz", reviewed_destination()),
            ("nhfwzzz", reviewed_destination()),
            ("nhfwzz", reviewed_destination("")),
            ("nhfwzz", {**reviewed_destination(), "pronunciationUnresolved": True}),
            ("nhfwzz", {**reviewed_destination(), "reviewDisposition": "BLOCK"}),
            ("nhfwzz", {**reviewed_destination(), "pronunciations": [*reviewed_destination()["pronunciations"], *reviewed_destination("nóng hòu fèn wéi")["pronunciations"]]}),
        ):
            with self.subTest(code=code, review=review):
                self.review = review
                result = await self.invoke(code)
                self.assertFalse(result.get("requiresConfirmation"), result)
                self.assertFalse(result.get("success"), result)
                self.assertEqual(self.strict_calls, [])

    async def test_changed_reading_invalidates_confirmation_before_any_write(self):
        preview = await self.invoke("nhfwzz")
        self.review = reviewed_destination("nóng hòu fèn wéi")
        result = await self.invoke(
            "nhfwzz", confirmed_plan_digest=preview["planDigest"], batch_id="",
            expected_content_version=0, expected_warning_digest=preview["warningDigest"],
        )
        self.assertTrue(result.get("staleConfirmation"), result)
        self.assertEqual(self.draft["items"], [])
        self.assertEqual(len(self.strict_calls), 1)

    async def test_occupied_explicit_destination_uses_the_existing_eviction_chain(self):
        self.rows.append({"word": "浓厚风味", "code": "nhfwa", "type": "Phrase", "weight": 100})
        preview = await self.invoke("nhfwa")
        self.assertTrue(preview.get("requiresConfirmation"), preview)
        from keytao_bot.plugins.chat_commands import _format_server_warning_confirmation
        rendered = _format_server_warning_confirmation("keytao_shift_phrase_code", preview)
        self.assertIn("浓厚风味：nhfwa→nhfwao（顺延）", rendered)
        self.assertEqual(sorted((item["action"], item["word"], item["code"]) for item in preview["shiftPlan"]["items"]), sorted([
            ("Delete", "奈飞", "nhfwv"), ("Create", "奈飞", "nhfw"),
            ("Delete", "浓厚氛围", "nhfw"), ("Create", "浓厚氛围", "nhfwa"),
            ("Delete", "浓厚风味", "nhfwa"), ("Create", "浓厚风味", "nhfwao"),
        ]))

    async def test_command_route_passes_explicit_destination_to_the_shared_planner(self):
        with (
            patch.object(b17_rest_recommended_commands, "_load_merged_word_entries", AsyncMock(return_value={row["word"]: row for row in self.rows})),
            patch.object(b17_rest_recommended_commands, "_execute_shift_to_code", AsyncMock(return_value="sealed preview")) as planner,
            patch.object(b17_rest_recommended_commands, "call_tool_function", AsyncMock(side_effect=AssertionError("route must let the shared planner validate codes"))),
        ):
            result = await b17_rest_recommended_commands.try_handle_move_to_code_command(
                "把 奈飞 调到 nhfw，把 浓厚氛围 挪到 nhfwzz", "qq", "s58-review",
            )
        self.assertEqual(result, "sealed preview")
        self.assertEqual(planner.await_args.kwargs["expected_codes"], ["nhfw", "nhfwzz"])

# Source: test_s58_generic_bridge.py
class GenericProposalBridgeTests(unittest.IsolatedAsyncioTestCase):
    setUp = GenericProposalBridgeTests_setUp

    call = GenericProposalBridgeTests_call

    assert_ticket = GenericProposalBridgeTests_assert_ticket

    async def test_reviewed_create_preserves_server_type_reading_and_manual_seal(self):
        result = await self.call()
        state = self.assert_ticket(result, "keytao_create_phrase")
        self.assertEqual(state.args["_reviewed_pinyin"], "nài fēi")
        self.assertIs(state.args["needs_manual_review"], True)
        self.assertEqual([name for name, _ in self.calls], ["keytao_lookup_by_words_batch", "keytao_create_phrase"])

    async def test_batch_preview_requires_every_reviewed_literal_pair(self):
        args = {"items": [{"word": "奈飞", "code": "nhfw"}, {"word": "浓厚氛围", "code": "nhfwa"}]}
        message = "安排添加 奈飞 nhfw；安排添加 浓厚氛围 nhfwa"
        result = await self.call("keytao_batch_add_to_draft", args, message)
        state = self.assert_ticket(result, "keytao_batch_add_to_draft")
        self.assertEqual([(item["word"], item["code"]) for item in state.args["items"]], [("奈飞", "nhfw"), ("浓厚氛围", "nhfwa")])
        self.assertTrue(all(item["needsManualReview"] for item in state.args["items"]))
        self.calls.clear()
        args["items"][1]["code"] = "nhfw"
        rejected = await self.call("keytao_batch_add_to_draft", args, message)
        self.assertTrue(rejected.get("grammar_gap_rejected"))
        self.assertEqual(self.calls, [])

    async def test_unreviewed_or_nonliteral_create_does_not_invoke_any_tool(self):
        contexts = [
            replace(self.context, trusted_reviewed_items_by_key={}),
            replace(self.context, trusted_reviewed_items_by_key={("奈飞", "nhfw"): {"type": "Phrase"}}),
        ]
        for context in contexts:
            self.assertTrue((await self.call(context=context)).get("grammar_gap_rejected"))
            self.assertEqual(self.calls, [])
        for args in ({"word": "奈飞", "code": "nhfwa"}, {"word": "不存在", "code": "nhfw"}):
            self.assertTrue((await self.call(args=args)).get("grammar_gap_rejected"))
            self.assertEqual(self.calls, [])

    async def test_raw_model_internal_fields_are_rejected_before_canonicalization_can_hide_them(self):
        for field, value in (
            ("_reviewed_pinyin", "forged"), ("_reviewed_candidate_codes", ["nhfw"]),
            ("needs_manual_review", False), ("remark", "trusted"),
            ("confirmed", True), ("preview_only", False), ("batch_id", "foreign"),
            ("expected_warning_digest", "a" * 64),
        ):
            with self.subTest(field=field):
                result = await self.call(args={"word": "奈飞", "code": "nhfw", field: value})
                self.assertFalse(result.get("grammar_gap_bridged"))
                self.assertEqual(self.calls, [])
        result = await self.call("keytao_batch_add_to_draft", {
            "items": [{"word": "奈飞", "code": "nhfw", "needsManualReview": False}],
        })
        self.assertTrue(result.get("grammar_gap_rejected"))
        self.assertEqual(self.calls, [])

    async def test_live_duplicate_and_wrong_type_never_reach_preview(self):
        self.entries["奈飞"] = [{"word": "奈飞", "code": "nhfw", "type": "Phrase"}]
        result = await self.call()
        self.assertTrue(result.get("grammar_gap_rejected"))
        self.assertEqual([name for name, _ in self.calls], ["keytao_lookup_by_words_batch"])
        self.calls.clear()
        result = await self.call(args={"word": "奈飞", "code": "nhfw", "type": "Single"})
        self.assertTrue(result.get("grammar_gap_rejected"))
        self.assertEqual(self.calls, [])

    async def test_partial_replanned_or_unsealed_server_preview_cannot_mint_ticket(self):
        for override in (
            {"failedCount": 1}, {"skippedCount": 1}, {"collisionReplanned": True},
            {"warningDigest": ""}, {"success": True}, {"requiresConfirmation": False},
        ):
            with self.subTest(override=override):
                self.preview_override = override
                result = await self.call()
                self.assertTrue(result.get("grammar_gap_rejected"))
                self.assertFalse(result.get("grammar_gap_bridged"))

    async def test_dictionary_delete_requires_one_live_typed_identity(self):
        self.entries["奈飞"] = [{"word": "奈飞", "code": "nhfw", "type": "Phrase"}]
        result = await self.call(args={"word": "奈飞", "code": "nhfw", "action": "Delete"}, message="安排删除 奈飞 nhfw")
        self.assert_ticket(result, "keytao_create_phrase")
        self.assertEqual(result["grammarGapArguments"]["type"], "Phrase")
        self.assertNotIn("_reviewed_pinyin", result["grammarGapArguments"])
        self.entries["奈飞"].append({"word": "奈飞", "code": "nhfw", "type": "Single"})
        result = await self.call(args={"word": "奈飞", "code": "nhfw", "action": "Delete"}, message="安排删除 奈飞 nhfw")
        self.assertTrue(result.get("grammar_gap_rejected"))

    async def test_batch_change_accepts_real_model_schema_and_seals_canonical_old_word(self):
        from tests import test_state_machine as harness
        from keytao_bot.harness.tools import _validate_arguments

        schema = next(item for item in harness._draft_tools.TOOLS
                      if item["function"]["name"] == "keytao_batch_add_to_draft")
        properties = schema["function"]["parameters"]["properties"]["items"]["items"]["properties"]
        self.assertIn("old_word", properties)
        self.assertNotIn("oldWord", properties)
        args = {"items": [{
            "word": "浓厚氛围", "code": "nhfw", "action": "Change", "old_word": "奈飞",
        }]}
        self.assertIsNone(_validate_arguments("keytao_batch_add_to_draft", args, schema))
        self.entries["奈飞"] = [{"word": "奈飞", "code": "nhfw", "type": "Phrase"}]
        context = replace(self.context, trusted_reviewed_items_by_key={
            ("浓厚氛围", "nhfw"): reviewed("浓厚氛围", "nhfw"),
        })
        result = await self.call("keytao_batch_add_to_draft", args,
                                 "安排修改 奈飞 为 浓厚氛围 nhfw", context)
        state = self.assert_ticket(result, "keytao_batch_add_to_draft")
        self.assertEqual(state.args["items"][0]["oldWord"], "奈飞")
        self.assertNotIn("old_word", state.args["items"][0])
        self.calls.clear()
        args["items"][0]["oldWord"] = args["items"][0].pop("old_word")
        result = await self.call("keytao_batch_add_to_draft", args,
                                 "安排修改 奈飞 为 浓厚氛围 nhfw", context)
        self.assertTrue(result.get("grammar_gap_rejected"))
        self.assertEqual(self.calls, [])

    async def test_single_and_batch_delete_preserve_exact_live_ordered_targets(self):
        for name, args in (
            ("keytao_remove_draft_item", {"pr_id": 123}),
            ("keytao_batch_remove_draft_items", {"ids": [124, 123]}),
        ):
            self.calls.clear()
            result = await self.call(name, args, "安排删除 123 124 奈飞 nhfw")
            state = self.assert_ticket(result, name)
            self.assertEqual(state.args["batch_id"], "actor-draft")
            self.assertEqual([name for name, _ in self.calls], ["keytao_list_draft_items", name])
            self.assertTrue(all(call["platform_id"] == "s58-generic" for _, call in self.calls))

    async def test_delete_nonliteral_ids_and_cas_fields_never_reach_snapshot(self):
        for args in ({"pr_id": 12}, {"pr_id": True}, {"pr_id": 123, "expected_target_digest": "b" * 64}):
            result = await self.call("keytao_remove_draft_item", args, "安排删除 123 奈飞 nhfw")
            self.assertFalse(result.get("grammar_gap_bridged"))
            self.assertEqual(self.calls, [])
        self.preview_override = {"contentVersion": 8}
        result = await self.call("keytao_remove_draft_item", {"pr_id": 123}, "安排删除 123 奈飞 nhfw")
        self.assertTrue(result.get("grammar_gap_rejected"))

    async def test_nonpreview_mutations_and_protection_never_dispatch(self):
        for name in ("keytao_update_draft_item_weight", "keytao_submit_batch", "keytao_recall_batch"):
            result = await self.call(name, {"word": "奈飞", "code": "nhfw", "weight": 102}, "安排调整权重 奈飞 nhfw 102")
            self.assertTrue(result.get("grammar_gap_rejected"))
            self.assertEqual(self.calls, [])
        for message in ("安排添加 奈飞 nhfw，但保持浓厚氛围", "不要安排添加 奈飞 nhfw", "他说安排添加 奈飞 nhfw", "安排添加 奈飞 nhfw？"):
            result = await self.call(message=message)
            self.assertFalse(result.get("grammar_gap_bridged"))
            self.assertEqual(self.calls, [])

    async def test_sealed_batch_confirmation_replays_exact_reviewed_create_change_and_delete(self):
        from tests import test_state_machine as test_state_machine
        from keytao_bot.plugins import chat_commands

        cases = (
            ([{"word": "奈飞", "code": "nhfw"}], "安排添加 奈飞 nhfw", {}),
            ([{"word": "浓厚氛围", "code": "nhfw", "action": "Change", "old_word": "奈飞"}],
             "安排修改 奈飞 为 浓厚氛围 nhfw", {"奈飞": [{"word": "奈飞", "code": "nhfw", "type": "Phrase"}]}),
            ([{"word": "奈飞", "code": "nhfw"}, {"word": "浓厚氛围", "code": "nhfwa", "action": "Delete"}],
             "安排添加 奈飞 nhfw；安排删除 浓厚氛围 nhfwa",
             {"浓厚氛围": [{"word": "浓厚氛围", "code": "nhfwa", "type": "Phrase"}]}),
        )
        for items, message, entries in cases:
            with self.subTest(actions=[item.get("action", "Create") for item in items]):
                self.setUp()
                self.entries.update(entries)
                capabilities = {
                    (item["word"], item["code"]): {
                        **reviewed(item["word"], item["code"]),
                        "needs_manual_review": item.get("action") != "Change",
                    } for item in items if item.get("action") != "Delete"
                }
                context = replace(self.context, trusted_reviewed_items_by_key=capabilities)
                result = await self.call("keytao_batch_add_to_draft", {"items": items}, message, context)
                state = self.assert_ticket(result, "keytao_batch_add_to_draft")
                self.assertTrue(all(args.get("confirmed") is not True for _, args in self.calls))
                expected = copy.deepcopy(state.args["items"])
                replayed = []

                async def strict_sink(**kwargs):
                    self.assertIs(kwargs["confirmed"], True)
                    self.assertNotIn("preview_only", kwargs)
                    self.assertEqual(kwargs["batch_id"], "actor-draft")
                    self.assertEqual(kwargs["expected_content_version"], 7)
                    self.assertEqual(kwargs["expected_warning_digest"], "a" * 64)
                    self.assertEqual(kwargs["items"], expected)
                    replayed.append(copy.deepcopy(kwargs))
                    return {"success": True, "message": "Confirmed fixture", "batchId": "actor-draft"}

                async def replay(name, args, platform, user_id, **kwargs):
                    self.assertEqual(name, "keytao_batch_add_to_draft")
                    return await self.executor.call(name, args, ToolContext(
                        platform=platform, user_id=user_id,
                        trusted_reviewed_items_by_key=kwargs.get("trusted_reviewed_items_by_key"),
                    ))

                self.executor._get_tool_function = lambda name: strict_sink if name == "keytao_batch_add_to_draft" else None
                with patch.object(chat_commands, "call_tool_function", side_effect=replay), patch.object(
                    chat_commands, "_format_draft_response", return_value="Confirmed fixture",
                ):
                    await chat_commands._execute_confirmed_tool(state, "qq", "s58-generic")
                self.assertEqual(len(replayed), 1)
                restored = chat_commands._reviewed_multi_word_capabilities(state)
                self.assertEqual(set(restored), set(capabilities))
                for item in expected:
                    if item["action"] == "Delete":
                        self.assertNotIn("_reviewed_pinyin", item)
                    else:
                        capability = capabilities[(item["word"], item["code"])]
                        self.assertEqual(item["_reviewed_pinyin"], capability["pinyin"])
                        self.assertEqual(item["_reviewed_candidate_codes"], capability["candidate_codes"])
                        self.assertEqual(item["needsManualReview"], capability["needs_manual_review"])

    async def test_unsealed_batch_items_cannot_restore_review_or_execute(self):
        from tests import test_state_machine as test_state_machine
        from keytao_bot.plugins import chat_commands

        result = await self.call("keytao_batch_add_to_draft", {"items": [{"word": "奈飞", "code": "nhfw"}]})
        state = self.assert_ticket(result, "keytao_batch_add_to_draft")
        for field in ("batch_id", "expected_content_version", "expected_warning_digest"):
            with self.subTest(field=field):
                invalid = copy.deepcopy(state)
                invalid.args.pop(field)
                self.assertEqual(chat_commands._reviewed_multi_word_capabilities(invalid), {})
                with patch.object(chat_commands, "call_tool_function") as replay:
                    await chat_commands._execute_confirmed_tool(invalid, "qq", "s58-generic")
                replay.assert_not_called()
        local = replace(state, confirmation_source="local_preview")
        self.assertEqual(chat_commands._reviewed_multi_word_capabilities(local), {})

    async def test_sealed_batch_review_capabilities_are_keyed_by_word_and_code(self):
        from tests import test_state_machine as test_state_machine
        from keytao_bot.plugins import chat_commands

        result = await self.call("keytao_batch_add_to_draft", {"items": [{"word": "奈飞", "code": "nhfw"}]})
        state = self.assert_ticket(result, "keytao_batch_add_to_draft")
        second = {**state.args["items"][0], "code": "nhfwa", "_reviewed_pinyin": "nài fēi",
                  "_reviewed_candidate_codes": ["nhfwa"], "needsManualReview": False}
        state.args["items"].append(second)
        capabilities = chat_commands._reviewed_multi_word_capabilities(state)
        self.assertEqual(set(capabilities), {("奈飞", "nhfw"), ("奈飞", "nhfwa")})
        self.assertEqual(capabilities[("奈飞", "nhfwa")]["candidate_codes"], ("nhfwa",))
        self.assertIs(capabilities[("奈飞", "nhfwa")]["needs_manual_review"], False)

    async def test_real_orchestrator_keeps_raw_args_and_saves_generic_ticket_before_reply(self):
        from tests import test_state_machine as harness
        from keytao_bot.harness.orchestrator import AgentOrchestrator, AgentRequestContext, AgentRuntimeConfig

        store = MemoryConversationStateStore()
        reviewed_result = {
            "success": True, "word": "奈飞", "type": "Phrase", "recommendedCode": "nhfw",
            "preSubmitAudit": {"autoApprove": False, "issues": ["manual review"]},
            "pronunciations": [{"pinyin": "nài fēi", "candidateStatuses": [{"code": "nhfw", "occupied": False}]}],
        }
        tool_calls = [
            SimpleNamespace(id="review", type="function", function=SimpleNamespace(
                name="keytao_prepare_reviewed_add", arguments=json.dumps({"word": "奈飞"}),
            )),
        ]
        client = harness._FakeClient([
            harness._FakeAIResponse("tool_calls", tool_calls=tool_calls),
            harness._FakeAIResponse("tool_calls", content="Added without confirmation", tool_calls=[
                SimpleNamespace(id="create", type="function", function=SimpleNamespace(
                    name="keytao_create_phrase", arguments=json.dumps({"word": "奈飞", "code": "nhfw"}),
                )),
            ]),
        ])

        class Skills(harness._FakeToolSkillsManager):
            def get_tools(self):
                return [{"type": "function", "function": {
                    "name": name, "description": "Read or preview",
                    "parameters": {"type": "object", "properties": {
                        "word": {"type": "string"}, **({"code": {"type": "string"}} if name == "keytao_create_phrase" else {}),
                    }},
                }} for name in ("keytao_prepare_reviewed_add", "keytao_create_phrase")]

        original = self.executor._get_tool_function
        async def review(**_kwargs):
            return copy.deepcopy(reviewed_result)
        self.executor._get_tool_function = lambda name: review if name == "keytao_prepare_reviewed_add" else original(name)
        orchestrator = AgentOrchestrator(
            client_factory=lambda: client, runtime=AgentRuntimeConfig("fake-model", 1000, 0.0, 10.0),
            skills_manager=Skills(), tool_executor=self.executor, state_store=store,
            bind_help_text="bind help", system_prompt_core="system",
        )
        context = AgentRequestContext(platform="qq", user_id="s58-generic", mutations_allowed=False)
        reply = await orchestrator.run("安排添加 奈飞 nhfw", context)
        record = store.get_record(context.conversation_address)
        self.assertIsNotNone(record, reply)
        self.assertTrue(server_warning_ticket_is_complete(record.state))
        self.assertFalse(record.execution_id)
        self.assertEqual(record.state.function_name, "keytao_create_phrase")
        self.assertIn("确认", reply)
        self.assertNotIn("Added without confirmation", reply)
        self.assertEqual(len(client.completions.calls), 2)

# Source: test_s58_grammar.py
class MoveToCodeGrammarTests(unittest.TestCase):
    context = MoveToCodeGrammarTests_context

    assert_bound_move = MoveToCodeGrammarTests_assert_bound_move

    def test_inline_change_header_binds_exact_pairs_with_separators_and_addresses(self):
        items = [
            {"action": "Change", "code": "dsqt", "old_word": "大权", "word": "大全", "type": "Phrase"},
            {"action": "Change", "code": "dsqtv", "old_word": "大全", "word": "大权", "type": "Phrase"},
        ]
        for prefix in ("修改 ", "改 ", "喵喵，修改 ", "喵喵:改：", "@owner 修改：", "@owner，修改：",
                       "修改，", "修改 ：", "修改 ，"):
            for tokens in (("dsqt", "大全", "dsqtv", "大权"), ("大全", "dsqt", "大权", "dsqtv")):
                for separators in ((" ", " ", " "), (" ", "，", " "), (" ", "、", " "),
                                   (" ", "；", " "), ("\n", "；\n", "\t")):
                    message = prefix + "".join(token + separator for token, separator in
                                               zip(tokens, (*separators, "")))
                    with self.subTest(message=message):
                        self.assertEqual(grammar._parse_code_word_changes(message),
                                         (("dsqt", "大全"), ("dsqtv", "大权")))
                        context = ToolContext(current_message=message, trusted_phrase_types_by_key={
                            ("大权", "dsqt"): frozenset({"Phrase"}),
                            ("大全", "dsqtv"): frozenset({"Phrase"}),
                        })
                        self.assertTrue(grammar.message_authorizes_mutation(message))
                        self.assertIsNone(grammar._validate_current_message_binding(
                            "keytao_batch_add_to_draft", {"items": items}, context,
                        ))

    def test_inline_change_header_single_pair_binds_both_tool_entry_points(self):
        item = {"action": "Change", "code": "dsqt", "old_word": "大权", "word": "大全"}
        for message in ("修改：dsqt 大全", "改，大全 dsqt", "喵喵，修改 dsqt 大全", "@owner 改：大全 dsqt"):
            for tool, arguments in (("keytao_create_phrase", item),
                                    ("keytao_batch_add_to_draft", {"items": [item]})):
                with self.subTest(message=message, tool=tool):
                    context = ToolContext(current_message=message, trusted_phrase_types_by_key={
                        ("大权", "dsqt"): frozenset({"Phrase"}),
                    })
                    self.assertIsNone(grammar._validate_current_message_binding(tool, arguments, context))

    def test_inline_change_header_rejects_ambiguous_or_untrusted_tokens(self):
        for message in (
            "修改 dsqt 大全 dsqtv", "修改 dsqt dsqtv 大全 大权",
            "修改 大全 大权 dsqt dsqtv", "修改 dsqt 大全 大权 dsqtv",
            "修改 大全 dsqt dsqtv 大权", "修改 dsqt 大全 dsqt 大权",
            "修改 dsqt 大全 DSQT 大权", "修改 dsqt 大全 dsqtv 大全",
            "修改 dsqt1 大全 dsqtv 大权", "修改 ｄｓｑｔ 大全 dsqtv 大权",
            "修改 dsqt 大全- dsqtv 大权", "修改 dsqt 大全🙂 dsqtv 大权",
            "修改 dsqt 大全 dsqtv 大权？", "不要修改 dsqt 大全 dsqtv 大权",
            "例如：修改 dsqt 大全 dsqtv 大权", "他说「修改 dsqt 大全 dsqtv 大权」",
            "以下是命令：「修改 dsqt 大全 dsqtv 大权」",
        ):
            with self.subTest(message=message):
                self.assertEqual(grammar._parse_code_word_changes(message), ())

    def test_inline_change_header_requires_complete_exact_trusted_change_set(self):
        items = [
            {"action": "Change", "code": "dsqt", "old_word": "大权", "word": "大全"},
            {"action": "Change", "code": "dsqtv", "old_word": "大全", "word": "大权"},
        ]
        message = "喵喵，修改：dsqt 大全 dsqtv 大权"
        context = ToolContext(current_message=message, trusted_phrase_types_by_key={
            ("大权", "dsqt"): frozenset({"Phrase"}),
            ("大全", "dsqtv"): frozenset({"Phrase"}),
        })
        invalid = [items[:1], [*items, items[0]],
                   [{**items[0], "code": "dsqtv"}, {**items[1], "code": "dsqt"}]]
        for field, value in (("word", "大地"), ("old_word", "大地"), ("action", "Create"),
                             ("code", "dsqtvv")):
            invalid.append([{**items[0], field: value}, items[1]])
        for changed in invalid:
            with self.subTest(items=changed):
                self.assertIsNotNone(grammar._validate_current_message_binding(
                    "keytao_batch_add_to_draft", {"items": changed}, context,
                ))
        self.assertIsNotNone(grammar._validate_current_message_binding(
            "keytao_batch_add_to_draft", {"items": items}, ToolContext(current_message=message),
        ))

    def test_change_header_binds_exact_code_word_pairs_in_both_orders(self):
        items = [
            {"action": "Change", "code": "dsqt", "old_word": "大权", "word": "大全", "type": "Phrase"},
            {"action": "Change", "code": "dsqtv", "old_word": "大全", "word": "大权", "type": "Phrase"},
        ]
        for header in ("修改", "改"):
            for pairs in ("dsqt 大全\ndsqtv 大权", "大全 dsqt\n大权 dsqtv"):
                message = f"喵喵 {header}\n{pairs}"
                with self.subTest(message=message):
                    context = ToolContext(
                        current_message=message,
                        trusted_phrase_types_by_key={
                            ("大权", "dsqt"): frozenset({"Phrase"}),
                            ("大全", "dsqtv"): frozenset({"Phrase"}),
                        },
                    )
                    self.assertIsNone(grammar._validate_current_message_binding(
                        "keytao_batch_add_to_draft", {"items": items}, context,
                    ))
                    self.assertTrue(grammar.message_authorizes_mutation(message))

    def test_change_header_single_pair_binds_both_tool_entry_points(self):
        item = {"action": "Change", "code": "dsqt", "old_word": "大权", "word": "大全"}
        for message in ("修改 dsqt 大全", "改 大全 dsqt", "修改\ndsqt 大全"):
            for tool, args in (("keytao_create_phrase", item),
                               ("keytao_batch_add_to_draft", {"items": [item]})):
                with self.subTest(message=message, tool=tool):
                    context = ToolContext(current_message=message, trusted_phrase_types_by_key={
                        ("大权", "dsqt"): frozenset({"Phrase"}),
                    })
                    self.assertTrue(grammar.message_authorizes_mutation(message))
                    self.assertIsNone(grammar._validate_current_message_binding(tool, args, context))

    def test_change_header_rejects_absent_crossed_or_untrusted_targets(self):
        items = [
            {"action": "Change", "code": "dsqt", "old_word": "大权", "word": "大全"},
            {"action": "Change", "code": "dsqtv", "old_word": "大全", "word": "大权"},
        ]
        context = ToolContext(current_message="修改\ndsqt 大全\ndsqtv 大权",
                              trusted_phrase_types_by_key={
                                  ("大权", "dsqt"): frozenset({"Phrase"}),
                                  ("大全", "dsqtv"): frozenset({"Phrase"}),
                              })
        invalid = [items[:1], [*items, items[0]]]
        for field, value in (("code", "dsqtvv"), ("word", "大地"),
                             ("old_word", "大地"), ("action", "Create")):
            invalid.append([{**items[0], field: value}, items[1]])
        invalid.append([{**items[0], "code": "dsqtv"}, {**items[1], "code": "dsqt"}])
        for changed in invalid:
            with self.subTest(items=changed):
                self.assertIsNotNone(grammar._validate_current_message_binding(
                    "keytao_batch_add_to_draft", {"items": changed}, context,
                ))
        self.assertIsNotNone(grammar._validate_current_message_binding(
            "keytao_batch_add_to_draft", {"items": items},
            ToolContext(current_message=context.current_message),
        ))
        for message in ("不要修改\ndsqt 大全", "例如：修改 dsqt 大全",
                        "修改 dsqt 大全？", "改\ndsqt 大全\ndsqt 大权"):
            with self.subTest(message=message):
                self.assertFalse(grammar._parse_code_word_changes(message))


    def test_incident_binds_live_existing_word_without_candidate_inventory(self):
        self.assert_bound_move("喵喵 把 奈飞 调到 nhfw")

    def test_move_verbs_and_subject_shapes(self):
        for verb in ("调到", "调至", "挪到", "挪至", "移到", "移至", "换到",
                     "放到", "放在", "改到", "调整到", "换成", "用"):
            for subject in ("把 奈飞", "将 奈飞", "奈飞"):
                with self.subTest(verb=verb, subject=subject):
                    self.assert_bound_move(f"{subject} {verb} nhfw")

    def test_imperative_wrappers_and_operand_quotes(self):
        for wrapper in ("", "喵喵 ", "@喵喵 ", "执行：", "执行 ", "请执行 ",
                        "帮我 ", "麻烦 ", "喵喵 请执行："):
            for left, right in (("「", "」"), ("“", "”"), ("‘", "’"),
                                ("『", "』"), ('"', '"'), ("'", "'")):
                with self.subTest(wrapper=wrapper, quotes=left + right):
                    self.assert_bound_move(
                        f"{wrapper}把 {left} 奈飞 {right} 调到 {left} nhfw {right}。"
                    )

    def test_optional_eviction_clause_remains_exact(self):
        for tail in ("，浓厚氛围顺延", "；并顺延 浓厚氛围", "、挤掉「浓厚氛围」",
                     ",并且顶替 '浓厚氛围'", " 并顺延 浓厚氛围", " 并顺延"):
            with self.subTest(tail=tail):
                self.assert_bound_move("把 奈飞 调到 nhfw" + tail,
                                       "" if tail == " 并顺延" else "浓厚氛围")
        message = "把 奈飞 调到 nhfw，顶替 不存在"
        self.assertIsNotNone(grammar._validate_current_message_binding(
            "keytao_shift_phrase_code", {"word": "奈飞", "target_code": "nhfw"},
            self.context(message),
        ))

    def test_two_clauses_remain_one_exact_ordered_plan(self):
        for separator in ("，", ",", "；", ";", "、", " 和 ", " 并且 ", "，并且 "):
            for destination in ("nhfwa", "nhfwai"):
                message = f"执行：把「奈飞」调到 'nhfw'{separator}把 浓厚氛围 挪到 {destination}"
                with self.subTest(message=message):
                    plan = grammar.parse_entry_move_plan(message)
                    self.assertIsNotNone(plan)
                    self.assertEqual(tuple((move.word, move.target_code) for move in plan.moves),
                                     (("奈飞", "nhfw"), ("浓厚氛围", destination)))
                    self.assertTrue(grammar.message_authorizes_mutation(message))
                    arguments = {
                        "word": "奈飞", "target_code": "nhfw",
                        "ordered_words": ["奈飞", "浓厚氛围"],
                        "listed_words": ["奈飞", "浓厚氛围"],
                        "expected_codes": ["nhfw", destination],
                    }
                    self.assertIsNone(grammar._validate_current_message_binding(
                        "keytao_shift_phrase_code", arguments, self.context(message),
                    ))
                    for invalid in (
                        {"word": "奈飞", "target_code": "nhfw"},
                        {**arguments, "expected_codes": ["nhfw", "zzzz"]},
                        {**arguments, "ordered_words": ["浓厚氛围", "奈飞"]},
                    ):
                        self.assertIsNotNone(grammar._validate_current_message_binding(
                            "keytao_shift_phrase_code", invalid, self.context(message),
                        ))

    def test_untrusted_or_incomplete_commands_do_not_authorize(self):
        for message in (
            "不要把 奈飞 调到 nhfw", "他说把 奈飞 调到 nhfw", "例如：把 奈飞 调到 nhfw",
            "不要把奈飞调到nhfw", "媒体称把奈飞调到nhfw", "请不要把奈飞调到nhfw",
            "并非把奈飞调到nhfw", "怎么奈飞放到nhfw",
            "执行：不要把 奈飞 调到 nhfw", "执行：把 奈飞 调到 nhfw？",
            "帮我 翻译：把 奈飞 调到 nhfw", "把 奈飞 调到 nhfw 删除 浓厚氛围",
            "把 奈飞 调到 nhfw123", "把 奈飞 调到 nhfwaaa",
            "把 奈飞 调到 nhfw，把 浓厚氛围 挪到 nhfwa，删除 别的词",
        ):
            with self.subTest(message=message):
                self.assertIsNone(grammar.parse_existing_entry_move(message))
                self.assertIsNone(grammar.parse_entry_move_plan(message))

    def test_unknown_verb_is_gap_without_write_authority(self):
        for message in ("喵喵 把 奈飞 迁到 nhfw", "把奈飞迁到nhfw"):
            self.assertFalse(grammar.message_authorizes_mutation(message))
            self.assertTrue(grammar.looks_like_mutation_grammar_gap(message))

    def test_normalizing_keyword_does_not_drop_multiclause_question_guard(self):
        self.assertFalse(grammar.message_authorizes_mutation(
            "喵喵\n加词 王中王 wfw？\n加词 微服务 wfwu"
        ))


class F3RecodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_f3_m1_11_same_code_aborts_multiclause_without_writes(self):
        from tests.helpers import B18RealToolTests_setUp, TOOLS

        B18RealToolTests_setUp(self)
        self.http.entries = [
            {"word": "甲", "code": "jxa", "type": "Phrase", "weight": 100},
            {"word": "蛋白质", "code": "dbfa", "type": "Phrase", "weight": 100},
        ]
        before = copy.deepcopy(self.http.entries)
        functions = {**harness._lookup_tools.TOOL_FUNCTIONS, **TOOLS.TOOL_FUNCTIONS}
        with patch.object(self.replay.chat.tool_executor, "_get_tool_function", functions.get):
            for words, codes, message in (
                (["甲", "蛋白质"], ["aaa", "dbfa"], "改码 甲 aaa，蛋白质 dbfa"),
                (["蛋白质", "甲"], ["dbfa", "aaa"], "改码 蛋白质 dbfa，甲 aaa"),
            ):
                with self.subTest(message=message):
                    await self.replay.turn({"message": message, "to_me": True}, raw_operation(
                        op="recode", operands=words, codes=codes,
                    ))
                    self.assertIn("「蛋白质」已经是 dbfa，无需修改。", self.replay.last_response)
                    self.assertIn("本次未写入", self.replay.last_response)
                    self.assertEqual(self.http.writes, [])
                    self.assertEqual(self.http.rows, [])
                    self.assertEqual(self.http.entries, before)
        self.assertEqual(self.http.errors, [])

    async def test_f3_m1_11_same_code_and_duplicate_recodes_have_specific_reasons(self):
        from keytao_bot.harness.operation_request import OperationRequest
        from tests.helpers import B18RealToolTests_setUp, TOOLS

        B18RealToolTests_setUp(self)
        self.http.entries = [{"word": "蛋白质", "code": "dbfa", "type": "Phrase", "weight": 100}]
        functions = {**harness._lookup_tools.TOOL_FUNCTIONS, **TOOLS.TOOL_FUNCTIONS}
        with patch.object(self.replay.chat.tool_executor, "_get_tool_function", functions.get):
            for operands, codes, message, expected in (
                (("蛋白质",), ("dbfa",), "蛋白质 编码改为 dbfa", "「蛋白质」已经是 dbfa，无需修改。"),
                (("蛋白质", "蛋白质"), ("dbfaa", "dbfai"),
                 "改码 蛋白质 dbfaa，蛋白质 dbfai", "「蛋白质」重复指定改码；本次未写入。"),
            ):
                with self.subTest(message=message):
                    request = OperationRequest(op="recode", operands=operands, codes=codes)
                    resolved, _notes, error = await self.replay.commands._resolve_target_recodes(
                        request, message, (), "qq", self.replay.author.actor_id,
                    )
                    self.assertIsNone(resolved)
                    self.assertEqual(error, expected)
        self.assertEqual(self.http.writes, [])
        self.assertEqual(self.http.errors, [])


def load_tests(loader, standard_tests, pattern):
    return _stage_t_unittest.TestSuite(
        loader.loadTestsFromTestCase(case) for case in (
            F3RecodeTests,
            B11ShiftTests,
            B11BlockedToolLoopTests,
            B11SourceShiftTests,
            B11ChangeSourceTests,
            B24DisplaceTests,
            B25SessionTests,
            GenericEvictionTests,
            CompletedUndoTests,
            CompletedUndoReviewTests,
            SameCodeReorderTests,
            BridgeTests,
            ExplicitDestinationTests,
            GenericProposalBridgeTests,
            MoveToCodeGrammarTests,
        )
    )


if __name__ == "__main__":
    _stage_t_unittest.main()
