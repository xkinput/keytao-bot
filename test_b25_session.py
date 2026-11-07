"""B25 production-shaped QQ replays with real tools and offline boundaries."""

import copy
import json
import sqlite3
import unittest
from contextlib import closing
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from urllib.parse import urlsplit

import test_b24_displace as b24


b19 = b24.b19
chat = b19.chat
commands = chat._chat_commands
WORD, OCCUPANT = "守时", "首诗"
CODES = ["edek", "edeko", "edekoo"]
WORDS = ("公义", "礼义廉耻", "德智体群美", WORD, "随时待命")
FACTS = {
    "公义": (["gjyk", "gjyko", "gjykoo"], "gong yi"),
    "礼义廉耻": (["lyli", "lylio", "lylioo"], "li yi lian chi"),
    "德智体群美": (["dftm", "dftmo", "dftmoo"], "de zhi ti qun mei"),
    WORD: (CODES, "shou shi"),
    "收拾": (CODES, "shou shi"),
    "首饰": (CODES, "shou shi"),
    OCCUPANT: (CODES, "shou shi"),
    "随时待命": (["sudm", "sudmo", "sudmoo"], "sui shi dai ming"),
    "周内": (["fdnw", "fdnsu"], "zhou nei"),
}


class SessionHTTP(b19.DraftHTTP):
    def __init__(self):
        super().__init__()
        self.batch_id = "40c69d7a"
        self.entries = [
            {"word": word, "code": code, "type": "Phrase", "weight": 100}
            for word, code in (("收拾", CODES[0]), ("首饰", CODES[1]), (OCCUPANT, CODES[2]),
                               ("周内", "fdnsu"))
        ]

    def _response(self, method, url, kwargs):
        if urlsplit(url).path == "/api/bot/evidence/fetch":
            return b19.b18.OfflineResponse({}, 404)
        if urlsplit(url).path == "/api/phrases/encode":
            word = (kwargs.get("params") or {})["word"]
            self.requests.append((method, "/api/phrases/encode", {"word": word}))
            if word not in FACTS:
                self.errors.append(("unexpected word", word))
                raise AssertionError("Unexpected concatenated/filler lookup: " + word)
            codes, reading = FACTS[word]
            return b19.b18.OfflineResponse({
                "input": word, "type": "二字词" if len(word) == 2 else "多字词",
                "codes": codes, "altCodes": [], "flyKeyVariants": [],
                "phrasePinyins": [reading], "pronunciationSource": "zdic-phrase",
                "standardPronunciationStatus": "found",
                "chars": [{"char": char, "pinyin": syllable, "pinyins": [syllable],
                           "pronunciationLookupStatus": "found", "phoneticCode": "ed",
                           "shapeCode": "oo"} for char, syllable in zip(word, reading.split())],
            })
        return super()._response(method, url, kwargs)


class B25SessionTests(unittest.IsolatedAsyncioTestCase):
    matcher_send = b19.B19PartialReplyBindingTests.matcher_send
    matcher_finish = b19.B19PartialReplyBindingTests.matcher_finish
    created = b19.B19PartialReplyBindingTests.created

    def setUp(self):
        b24.B24DisplaceTests.setUp(self)
        self.enterContext(patch.dict("os.environ", {"KEYTAO_COMMUNITY_VOTE": "0"}))
        self.http = SessionHTTP()
        self.enterContext(patch.object(b19.b18.TOOLS.http_client, "_keytao_client", self.http))
        self.replay.client.create.side_effect = TimeoutError("B25 semantic parser timeout")
        self.enterContext(patch.object(chat._chat_routing, "AsyncOpenAI", return_value=self.replay.client))
        self.histories = {self.replay.author.actor_id: self.replay.history}
        self.enterContext(patch.object(chat, "get_history", side_effect=self.history_for))
        self.enterContext(patch.object(chat, "remember_conversation", side_effect=self.remember))
        self.tool_errors = self.enterContext(patch.object(commands.logger, "error", wraps=commands.logger.error))
        with closing(sqlite3.connect(b24.review.reference_db_path())) as connection, connection:
            connection.executemany("INSERT OR REPLACE INTO word_commonness VALUES (?, 3, NULL, 2)",
                                   [(word,) for word in FACTS])

    def tearDown(self):
        b19.B19PartialReplyBindingTests.tearDown(self)

    def history_for(self, key):
        actor = key.actor_id if hasattr(key, "actor_id") else key[1]
        return self.histories.setdefault(actor, [])

    def remember(self, key, memory, message, response):
        self.history_for(key).extend([{"role": "user", "content": message},
                                     {"role": "assistant", "content": response}])

    async def turn(self, message, reply_id, actor=None):
        actor = actor or self.replay.author.actor_id
        memory = replace(self.replay.memory, user_id=actor)
        full_message = f"[reply:id={reply_id}][at:qq={self.bot.self_id}] {message}"
        event = SimpleNamespace(
            to_me=True, group_id=self.replay.author.space_id, user_id=actor, message_id=2,
            raw_message=full_message,
            message=[SimpleNamespace(type="reply", data={"id": reply_id}),
                     SimpleNamespace(type="at", data={"qq": self.bot.self_id}),
                     SimpleNamespace(type="text", data={"text": message})],
            get_plaintext=lambda: message,
        )
        tokens = [(variable, variable.set(value)) for variable, value in (
            (chat._current_turn_message, full_message), (chat._current_conversational_turn, None),
            (chat.current_operation_capture, None), (chat.current_draft_delivery_claims, []),
        )]
        before = self.bot.send.await_count
        try:
            with patch.object(chat, "extract_memory_context", AsyncMock(return_value=memory)):
                await chat._handle_ai_chat_serialized(self.bot, event, "qq", actor)
        finally:
            for variable, token in reversed(tokens):
                variable.reset(token)
        self.assertEqual(self.bot.send.await_count, before + 1)
        self.bot.get_msg.assert_awaited_with(message_id=int(reply_id))
        self.replay.last_response = self.bot.messages[self.bot.last_id]["message"]
        return self.bot.last_id

    async def greeting(self, text="喵喵在。"):
        result = await self.bot.deliver(event=None, message=text)
        return str(result["message_id"])

    async def seed(self, word=WORD):
        await self.turn("喵喵 加词 " + word, await self.greeting())
        self.original_id = self.bot.last_id
        record = self.replay.store.get_record(self.replay.author)
        self.assertIsNotNone(record, self.replay.last_response)
        return record

    async def partial_session(self):
        await self.seed("、".join(WORDS))
        await self.turn("公义 3，礼义廉耻 2，德智体群美 2，守时 3，随时待命 1，加入并提交",
                        self.original_id)
        self.assertCountEqual(self.created(), [(word, FACTS[word][0][index]) for word, index in (
            ("公义", 2), ("礼义廉耻", 1), ("德智体群美", 1), ("随时待命", 0),
        )], self.replay.last_response)
        self.assertEqual(self.http.submissions, 1, self.replay.last_response)
        self.assertIn("首诗", self.replay.last_response)
        self.assertIn("未加入", self.replay.last_response)
        return self.bot.last_id

    async def finish_replacement(self, message, reply_id=None):
        await self.turn(message, reply_id or self.original_id)
        if not self.http.writes:
            record = self.replay.store.get_record(self.replay.author)
            self.assertIsNotNone(record, self.replay.last_response)
            self.assertIsInstance(record.state, b19.harness.PendingToolConfirm, self.replay.last_response)
            await self.turn("确认", self.bot.last_id)
        self.assertIn((WORD, CODES[2]), self.created(), self.replay.last_response)
        self.assertIn((OCCUPANT, CODES[2]), [(row["word"], row["code"])
                      for row in self.http.rows if row["action"] == "Delete"], self.replay.last_response)

    async def test_g1_internal_submit_metadata_never_reaches_real_tool(self):
        result = json.loads(await commands.call_tool_function("keytao_submit_batch", {
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
        with patch.dict(FACTS, {"周内": (["fdn", "fdnw", "fdnsu"], "zhou nei")}):
            result = json.loads(await commands.call_tool_function("keytao_shift_phrase_code", {
                "word": "周内", "target_code": "fdnw", "_source_code": "fdnsu",
            }, "qq", self.replay.author.actor_id))
        self.assertEqual([(row["word"], row["code"]) for row in result["shiftPlan"]["items"]
                          if row["action"] == "Delete"], [("周内", "fdnsu")])

    def test_g2_fillers_are_not_explicit_add_words(self):
        for phrase in ("再加一个", "再加", "加入编码", "加编码", "添加这个", "添加它", "添加一下"):
            with self.subTest(phrase=phrase):
                message = phrase + " fdnw"
                self.assertIsNone(b24.explicit_complete_add_item(message))
                self.assertIsNone(b24.parse_explicit_entry_code_request(message))

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
        await self.finish_replacement('守时 3，顺延“首诗”', reply_id)

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
        with patch.dict(FACTS, facts):
            for message, word in (("喵喵 加词 替代品", "替代品"), ("加词 取代者", "取代者")):
                with self.subTest(message=message):
                    self.replay.store.delete(self.replay.author)
                    await self.turn(message, await self.greeting())
                    record = self.replay.store.get_record(self.replay.author)
                    self.assertIsNotNone(record, self.replay.last_response)
                    self.assertIsInstance(record.state, b19.harness.PendingAddWord)
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
            ("喵喵 守时 替代品", (WORD, "替代品")),
            ("新词 替代品", ("新词", "替代品")),
            ("加 替代品", ()),
            ("帮我加 替代品", ()),
        )
        with patch.dict(FACTS, facts):
            for message, words in cases:
                with self.subTest(message=message):
                    self.replay.store.delete(self.replay.author)
                    self.http.requests.clear()
                    await self.turn(message, await self.greeting())
                    record = self.replay.store.get_record(self.replay.author)
                    if words:
                        self.assertIsNotNone(record, self.replay.last_response)
                        if len(words) == 1:
                            self.assertIsInstance(record.state, b19.harness.PendingAddWord)
                            self.assertEqual(record.state.word, words[0])
                        else:
                            self.assertIsInstance(record.state, b19.harness.PendingToolConfirm)
                            self.assertEqual(record.state.args["_query_words"], list(words))
                        for word in words:
                            self.assertIn(FACTS[word][0][0], self.replay.last_response)
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
                self.assertIsNone(commands.parse_reviewed_displacement(message))

    def test_g3_use_prefix_cannot_split_a_word(self):
        for message in ("「用户」替代首诗", "用「用户」替代首诗", "用 用户 替代 首诗"):
            with self.subTest(message=message):
                self.assertEqual(commands.parse_reviewed_displacement(message), ("用户", OCCUPANT, ""))

    async def test_g3_unquoted_use_prefix_binds_exact_reviewed_word(self):
        await self.seed()
        await self.finish_replacement("用守时替代首诗")

    def test_g3_lexical_words_are_not_modal_questions(self):
        for word in ("能量", "能力"):
            with self.subTest(word=word):
                self.assertEqual(commands.parse_reviewed_displacement(word + "替代首诗"), (word, OCCUPANT, ""))

    def test_g3_glued_use_prefix_keeps_unquoted_operand_validation(self):
        for word in ("加入", "加词", "编码"):
            with self.subTest(word=word):
                self.assertIsNone(commands.parse_reviewed_displacement("用" + word + "替代首诗", known_words=(word,)))

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
        for tail in ("，顶替首诗", "，替代首诗", " 顺延 首诗"):
            with self.subTest(tail=tail):
                self.replay.store.delete(self.replay.author)
                self.http.rows.clear()
                self.http.writes.clear()
                await self.seed("、".join(WORDS))
                await self.finish_replacement("守时 3" + tail)
                record = self.replay.store.get_record(self.replay.author)
                if record is not None and record.state.function_name == "keytao_batch_add_to_draft":
                    self.assertNotIn(WORD, record.state.args["_query_words"])

    async def test_g3_wrong_occupant_refuses_without_concatenated_lookup(self):
        await self.seed()
        for message in ("守时替代周内", "守时 3，顺延首饰"):
            with self.subTest(message=message):
                await self.turn(message, self.original_id)
                self.assertEqual(self.http.writes, [])
                self.assertIn("未", self.replay.last_response)

    async def test_g3_changed_candidates_preserve_original_batch_ticket(self):
        original = await self.seed("、".join(WORDS))
        snapshot = copy.deepcopy(original)
        with patch.dict(FACTS, {WORD: (CODES[:2], "shou shi")}):
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
                    self.assertIsInstance(record.state, b19.harness.PendingToolConfirm)
                    await self.turn("确认", self.bot.last_id)
                self.assertEqual([(row["word"], row["code"], row["action"]) for row in self.http.rows],
                                 [(OCCUPANT, CODES[2], "Delete")], self.replay.last_response)
                self.assertNotIn("安全拦截", self.replay.last_response)

    async def test_g4_multiple_codes_require_existing_disambiguation(self):
        self.http.entries.append({"word": OCCUPANT, "code": "edekov", "type": "Phrase", "weight": 100})
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


if __name__ == "__main__":
    unittest.main()
