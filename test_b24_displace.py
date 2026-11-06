"""B24 displacement replays through real tools and offline QQ/HTTP boundaries."""

import copy
import sqlite3
import tempfile
import unittest
from collections import OrderedDict
from contextlib import closing
from pathlib import Path
from unittest.mock import AsyncMock, patch
from urllib.parse import urlsplit

import test_b19_partial_reply_binding as b19
from keytao_bot.utils import keytao_review as review
from keytao_bot.utils.bcc_reference import CHANNELS, HISTORICAL_CHANNELS, create_schema
from keytao_bot.utils.explicit_code import parse_explicit_entry_code_request
from keytao_bot.harness.authorization_grammar import explicit_complete_add_item
from keytao_bot.harness.operation_request import understand_message


WORD, OCCUPANT = "豪横", "毫亨"
CODES = ["hzhr", "hzhro", "hzhrov"]
REAL_REVIEW_READS = {name: getattr(review, name) for name in (
    "collect_pronunciation_evidence_limited", "fetch_keytao_encode", "lookup_words",
)}


class DisplaceHTTP(b19.DraftHTTP):
    def __init__(self):
        super().__init__()
        self.entries = [{"word": OCCUPANT, "code": CODES[0], "type": "Phrase", "weight": 100}]

    def _response(self, method, url, kwargs):
        if urlsplit(url).path == "/api/phrases/encode":
            word = (kwargs.get("params") or {})["word"]
            self.requests.append((method, "/api/phrases/encode", {"word": word}))
            if word not in {WORD, OCCUPANT}:
                self.errors.append(("unexpected word", word))
                raise AssertionError(word)
            return b19.b18.OfflineResponse({
                "input": word, "type": "二字词", "codes": CODES, "altCodes": [],
                "flyKeyVariants": [], "phrasePinyins": ["hao heng"],
                "pronunciationSource": "zdic-phrase", "standardPronunciationStatus": "found",
                "chars": [{"char": char, "pinyin": pinyin, "pinyins": [pinyin],
                           "pronunciationLookupStatus": "found", "phoneticCode": code, "shapeCode": shape}
                          for char, pinyin, code, shape in zip(word, ("hao", "heng"), ("hz", "hr"), ("o", "ov"))],
            })
        return super()._response(method, url, kwargs)


class B24DisplaceTests(unittest.IsolatedAsyncioTestCase):
    matcher_send = b19.B19PartialReplyBindingTests.matcher_send
    matcher_finish = b19.B19PartialReplyBindingTests.matcher_finish
    turn = b19.B19PartialReplyBindingTests.turn
    created = b19.B19PartialReplyBindingTests.created

    def setUp(self):
        b19.B19PartialReplyBindingTests.setUp(self)
        # Earlier concurrent fixtures can leave patched readers behind.
        self.enterContext(patch.multiple(review, **REAL_REVIEW_READS))
        self.http = DisplaceHTTP()
        self.enterContext(patch.object(b19.b18.TOOLS.http_client, "_keytao_client", self.http))
        functions = {**b19.harness._lookup_tools.TOOL_FUNCTIONS,
                     **b19.harness._review_tools.TOOL_FUNCTIONS, **b19.b18.TOOLS.TOOL_FUNCTIONS}
        self.enterContext(patch.object(b19.chat.tool_executor, "_get_tool_function", functions.get))
        self.web = self.enterContext(patch.object(
            review.http_client, "guarded_fetch", AsyncMock(return_value=b19.b18.OfflineResponse({}, 404)),
        ))
        self.enterContext(patch.object(review, "_review_cache", OrderedDict()))
        self.enterContext(patch.object(review, "_evidence_host_failures", {}))
        directory = self.enterContext(tempfile.TemporaryDirectory())
        db = Path(directory) / "reference.db"
        with closing(sqlite3.connect(db)) as connection, connection:
            create_schema(connection)
            connection.execute("CREATE TABLE word_commonness (word TEXT PRIMARY KEY, corpus_frequency INTEGER, "
                               "part_of_speech TEXT, dictionary_presence_count INTEGER)")
            connection.executemany("INSERT INTO word_commonness VALUES (?, 3, NULL, ?)",
                                   [(WORD, 2), (OCCUPANT, 0)])
            for channel in (*CHANNELS, *HISTORICAL_CHANNELS):
                connection.execute("INSERT INTO bcc_dataset VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                   (channel, channel, "word", "offline", "offline", "offline", 1, 1, 100, 1000000))
            connection.executemany("INSERT INTO bcc_frequency VALUES (?, ?, 17, 17, 1)",
                                   [("多领域", WORD), ("文学", WORD)])
        self.enterContext(patch.object(review, "reference_db_path", return_value=db))

    def tearDown(self):
        b19.B19PartialReplyBindingTests.tearDown(self)

    async def seed(self):
        greeting = await self.bot.deliver(event=None, message="喵喵在。")
        await self.turn("喵喵 豪横", str(greeting["message_id"]), raw=b19.b18.raw_operation(
            op="query", operands=[WORD], codes=[], refers_to_pending=False,
        ))
        self.original_id = self.bot.last_id
        record = self.replay.store.get_record(self.replay.author)
        self.assertIsNotNone(record, self.replay.last_response)
        state = record.state
        self.assertIsInstance(state, b19.harness.PendingAddWord)
        self.assertEqual(state.recommended_code, CODES[1], self.replay.last_response)
        self.assertEqual(state.server_occupied_words[CODES[0]], [OCCUPANT])
        self.assertIn("1. hzhr — 已有「毫亨」", self.replay.last_response)
        self.assertIn("2. hzhro — ✅ 推荐（空位）", self.replay.last_response)
        return state

    async def finish_shift(self, message):
        await self.turn(message, self.original_id)
        record = self.replay.store.get_record(self.replay.author)
        if not self.http.writes:
            self.assertIsNotNone(record, self.replay.last_response)
            self.assertIsInstance(record.state, b19.harness.PendingToolConfirm, self.replay.last_response)
            self.assertEqual(record.state.function_name, "keytao_shift_phrase_code", self.replay.last_response)
            self.assertTrue(record.state.args["confirmed_plan_digest"])
            await self.turn("确认", self.bot.last_id)
        self.assertCountEqual(self.created(), [(WORD, CODES[0]), (OCCUPANT, CODES[1])],
                              self.replay.last_response)
        self.assertEqual([(row["word"], row["code"]) for row in self.http.rows if row["action"] == "Delete"],
                         [(OCCUPANT, CODES[0])])
        self.assertEqual(len(self.http.writes), 1)
        record = self.replay.store.get_record(self.replay.author)
        self.assertTrue(record is None or record.state.function_name == "keytao_submit_batch")

    def test_f1_displacement_verbs_are_never_explicit_word_code_adds(self):
        for prefix in ("", "加入，", "加入并提交，"):
            for verb in ("顶替", "顶掉", "挤掉", "替换"):
                with self.subTest(prefix=prefix, verb=verb):
                    self.assertIsNone(parse_explicit_entry_code_request(f"{prefix}{verb} {OCCUPANT}"))

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
                    self.assertIsNone(b19.chat._chat_commands.fresh_entry_code_selection(f"{verb} {operand}"))
                    self.assertIsNone(await b19.chat._chat_commands.try_handle_bare_word_code(
                        f"{verb} {operand.lower()}", "qq", self.replay.author.actor_id,
                        self.replay.author, self.replay.author.space_key, "Offline owner",
                    ))
        self.assertEqual(self.http.writes, [])
        self.assertEqual(parse_explicit_entry_code_request("加入，「顶替」 dgti").word, "顶替")

    async def test_complete_code_target_replacement_keeps_existing_route(self):
        await self.seed()
        self.replay.store.delete(self.replay.author)
        await self.turn("顶替 hzhr为豪横", self.original_id, raw=b19.b18.raw_operation(
            op="move", operands=[WORD], codes=[CODES[0]], relation="replace", refers_to_pending=False,
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
        self.assertIn("保留现有位置；本次未写入", self.replay.last_response)
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
            {"id": 1, "action": "Create", "word": WORD, "code": CODES[0], "type": "Phrase"},
            {"id": 2, "action": "Delete", "word": OCCUPANT, "code": CODES[0], "type": "Phrase"},
            {"id": 3, "action": "Create", "word": OCCUPANT, "code": CODES[1], "type": "Phrase"},
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
            {"id": 1, "action": "Create", "word": WORD, "code": "abcd", "type": "Phrase"},
            {"id": 2, "action": "Delete", "word": OCCUPANT, "code": "abcd", "type": "Phrase"},
            {"id": 3, "action": "Create", "word": OCCUPANT, "code": "abcde", "type": "Phrase"},
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
                    await self.turn(f"{prefix}{verb} {OCCUPANT}", self.original_id)
                    self.assertEqual(self.replay.last_response, "当前没有待确认的操作。")
                    self.assertEqual(self.http.writes, [])

    async def test_f3_empty_number_keeps_existing_refusal(self):
        await self.seed()
        await self.turn("2 顺延", self.original_id)
        self.assertEqual(self.http.writes, [])
        self.assertIn("保留现有位置；本次未写入", self.replay.last_response)

    async def test_f3_multiple_occupants_keep_commonness_refusal(self):
        state = await self.seed()
        state.server_occupied_words[CODES[0]].append("耗横")
        record = self.replay.store.get_record(self.replay.author)
        await b19.REAL_FINISH(self.bot, None, self.replay.author.actor_id, self.replay.memory,
                              b19.chat._render_live_single_candidate_record(record))
        await self.turn("1 顺延", self.bot.last_id)
        self.assertEqual(self.http.writes, [])
        self.assertEqual(self.replay.last_response, "当前没有可验证的可执行操作，本次未写入。")

    async def test_f4_other_speaker_cannot_displace_on_owners_ticket(self):
        state = await self.seed()
        for message in ("1 顺延", "换 1 顺延", "1，顺延", "加入，顶替 毫亨"):
            with self.subTest(message=message):
                await self.turn(message, self.original_id, actor="b24-other-speaker")
                self.assertEqual(self.replay.last_response, "这条提议不是发给你的，本次未执行。")
                self.assertIs(self.replay.store.get_record(self.replay.author).state, state)
                self.assertEqual(self.http.writes, [])


if __name__ == "__main__":
    unittest.main()
