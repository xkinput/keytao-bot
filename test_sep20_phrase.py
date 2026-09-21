"""Offline regressions for literal quoted phrase lookup and reviewed add routing."""

import copy
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from keytao_bot.utils.literal_phrase import (
    parse_explicit_single_phrase_add,
    parse_literal_phrase_query,
)
from keytao_bot.utils import keytao_review as review


chat = harness.openai_chat_module
commands = harness.chat_commands_module
REAL_TOOL_DISPATCH = commands.call_tool_function
PHRASE = "头痛医头，脚痛医脚"
PINYINS = ["tóu", "tòng", "yī", "tóu", "jiǎo", "tòng", "yī", "jiǎo"]
NORMALIZED = ["tou", "tong", "yi", "tou", "jiao", "tong", "yi", "jiao"]
# Exact row read from the repository's built, vendored CEDICT reference database.
CEDICT_PHRASE_ROW = (
    PHRASE, "tou tong yi tou , jiao tong yi jiao",
    "tóu tòng yī tóu , jiǎo tòng yī jiǎo",
    "tou2 tong4 yi1 tou2 , jiao3 tong4 yi1 jiao3", "cedict",
)


def phrase_encode_fixture():
    encoded_readings = [*PINYINS[:4], "，", *PINYINS[4:]]
    return {
        "input": PHRASE, "type": "四字词及以上", "codes": ["ttyj"],
        "candidateCodes": ["ttyj"], "phrasePinyins": encoded_readings,
        "contextPhrasePinyins": encoded_readings,
        "pronunciationSource": "zdic-unavailable",
        "standardPronunciationStatus": "found",
        "semanticPronunciationNeeded": False,
        "chars": [
            {"char": char, "pinyin": pinyin, "pinyins": [pinyin],
             "pronunciationLookupStatus": "absent" if char == "，" else "found"}
            for char, pinyin in zip(PHRASE, encoded_readings)
        ],
    }


class LiteralPhraseRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def route(self, message, *, executor=False, review_callback=None, continue_with_add=False):
        calls = []
        store = harness.MemoryConversationStateStore()
        key = harness.ConversationAddress.private("qq", "sep20-phrase")

        async def tool(name, args, _platform, _user_id, **_kwargs):
            calls.append((name, copy.deepcopy(args)))
            if name == "keytao_lookup_by_word":
                return json.dumps({"success": True, "phrases": []})
            if name == "keytao_pending_items_by_words":
                return json.dumps({"success": True, "complete": True, "items": []})
            if name == "keytao_prepare_reviewed_add":
                if review_callback is not None:
                    return json.dumps(await review_callback(args["word"]))
                return json.dumps({"success": False, "message": "Fixture review reached"})
            if name == "keytao_create_phrase" and continue_with_add:
                self.assertEqual(args["word"], PHRASE)
                self.assertEqual(args["code"], "ttyj")
                self.assertIs(args["needs_manual_review"], True)
                item = {"action": "Create", "word": args["word"], "code": args["code"],
                        "type": "Phrase", "needsManualReview": args["needs_manual_review"]}
                validation = await harness._draft_tools._validate_draft_item_code(
                    item, reviewed_pinyin=args.get("_reviewed_pinyin", ""),
                    reviewed_candidate_codes=args.get("_reviewed_candidate_codes"),
                )
                self.assertIs(validation["success"], True, validation)
                harness._draft_tools._stamp_item_review_flag(item, validation)
                self.assertIs(item["needsManualReview"], True)
                return json.dumps({
                    "success": True, "batchId": "sep20-phrase-fixture",
                    "batchUrl": "https://example.invalid/batch/sep20-phrase-fixture",
                    "writtenItems": [{"id": 1, **item}],
                })
            raise AssertionError("Unexpected external tool: " + name)

        async def classifier(_message, structural_words):
            return harness.SimpleWordQueryIntent(True, structural_words, "word_lookup", 1.0)

        def resolve_tool(name):
            async def dispatch(**arguments):
                if name in commands._INJECT_PLATFORM_TOOLS:
                    self.assertEqual(arguments.pop("platform"), "qq")
                    self.assertEqual(arguments.pop("platform_id"), key.actor_id)
                return json.loads(await tool(name, arguments, "qq", key.actor_id))
            return dispatch

        ctx = chat.TurnContext(
            bot=object(), event=object(), platform="qq", user_id=key.actor_id,
            message_text=message, normalized_message_text=message,
            conv_key=key, space_key=key.space_key, history=[],
        )
        with ExitStack() as stack:
            for patcher in (
                patch.object(chat, "conversation_state_store", store),
                patch.object(chat, "_classify_simple_word_query_intent", AsyncMock(side_effect=classifier)),
                patch.object(chat, "_classify_message_command_intent", AsyncMock(return_value=harness.MessageCommandIntent())),
                patch.object(chat, "remember_conversation"),
                patch.object(chat, "schedule_memory_compaction"),
                patch.object(commands.user_resolver, "resolve_actor_binding", AsyncMock(return_value=True)),
            ):
                stack.enter_context(patcher)
            if executor:
                for name in ("keytao_lookup_by_word", "keytao_prepare_reviewed_add"):
                    self.assertIsNotNone(commands.tool_executor._resolve_schema(name), name)
                stack.enter_context(patch.object(chat, "call_tool_function", REAL_TOOL_DISPATCH))
                stack.enter_context(patch.object(commands.tool_executor, "_get_tool_function", side_effect=resolve_tool))
                stack.enter_context(patch.object(commands.memory_store, "record_tool_receipt"))
            else:
                stack.enter_context(patch.object(chat, "call_tool_function", side_effect=tool))
            await chat._stage_handle_simple_word_query(ctx)
            if continue_with_add:
                bottom_encode = AsyncMock(return_value={"success": True, "candidateCodes": ["ttyj"]})
                stack.enter_context(patch.object(harness._draft_tools, "_fetch_encode_candidates", bottom_encode))
                ctx.query_record = copy.deepcopy(store.get_record(key))
                ctx.add_response = await commands.handle_pending_message_core(
                    "加入", "qq", key.actor_id, key, allow_intent_model=False,
                )
                bottom_encode.assert_awaited_once_with(PHRASE, "ttyj")
        return ctx, calls, store.get_record(key)

    async def assert_review_targets(self, message, expected, *, executor=False):
        ctx, calls, record = await self.route(message, executor=executor)
        self.assertEqual(
            [args["word"] for name, args in calls if name == "keytao_lookup_by_word"], expected,
            "The real word-query stage must preserve the user's lexical operand",
        )
        self.assertEqual(
            [args["word"] for name, args in calls if name == "keytao_prepare_reviewed_add"], expected,
        )
        self.assertIsNotNone(ctx.response)
        self.assertIsNone(record, "A failed review must not mint an executable ticket")
        return ctx.response

    async def test_bare_list_still_reviews_two_words(self):
        await self.assert_review_targets("头痛医头、脚痛医脚", ["头痛医头", "脚痛医脚"])

    async def test_quoted_phrase_is_one_exact_lookup_target(self):
        await self.assert_review_targets("“" + PHRASE + "”", [PHRASE])

    async def test_explicit_single_entry_request_reviews_exact_literal(self):
        await self.assert_review_targets("将“" + PHRASE + "”作为单个词加入词库", [PHRASE])

    async def test_real_executor_preserves_literal_targets_and_failed_review(self):
        for message in ("“" + PHRASE + "”", "将“" + PHRASE + "”作为单个词加入词库"):
            with self.subTest(message=message):
                response = await self.assert_review_targets(message, [PHRASE], executor=True)
                self.assertEqual(response, "Fixture review reached")

    async def test_reported_and_nested_quotes_do_not_become_lookup_operands(self):
        for message in (
            "他说“" + PHRASE + "”",
            "转发：将“" + PHRASE + "”作为单个词加入词库",
            "“将“" + PHRASE + "”作为单个词加入词库”",
        ):
            with self.subTest(message=message):
                ctx, calls, record = await self.route(message)
                self.assertIsNone(ctx.response)
                self.assertEqual(calls, [])
                self.assertIsNone(record)


class LiteralPhraseParserTests(unittest.TestCase):
    def test_query_preserves_exact_internal_punctuation(self):
        for opening, closing in (("“", "”"), ("「", "」"), ("『", "』"), ('"', '"')):
            for punctuation in ("，", ",", "、", "；", ";"):
                phrase = "头痛医头" + punctuation + "脚痛医脚"
                with self.subTest(opening=opening, punctuation=punctuation):
                    self.assertEqual(parse_literal_phrase_query(opening + phrase + closing), phrase)
        self.assertEqual(parse_literal_phrase_query("  “头痛医头”  "), "头痛医头")

    def test_explicit_single_entry_variants_preserve_exact_target(self):
        for message in (
            "将“" + PHRASE + "”作为单个词加入词库",
            "请把「" + PHRASE + "」当作一个词条添加到草稿。",
            "麻烦帮我将『" + PHRASE + "』作为单个词条加到词库",
        ):
            with self.subTest(message=message):
                self.assertEqual(parse_explicit_single_phrase_add(message), PHRASE)

    def test_query_rejects_unclosed_nested_narrative_and_action_shapes(self):
        for message in (
            "", "“”", "“" + PHRASE, "“" + PHRASE + "」",
            "“「" + PHRASE + "」”", "“头痛医头，，脚痛医脚”",
            "“头痛医头，脚痛医脚。”", "“头痛医头，\n脚痛医脚”",
            "\t“" + PHRASE + "”", "“" + "词" * 21 + "”",
            "他说“" + PHRASE + "”", "“头痛医头，提交草稿”",
            "“将亮面加入词库”", "“头痛医头，脚痛医脚吗”",
            "“头痛医头”；再提交草稿", "“头痛医头” “脚痛医脚”",
            "“他说头痛医头”", "“转发头痛医头”", "“记录头痛医头”",
            "“头痛医头，不要提交”", "“头痛医头，别删除旧词”",
        ):
            with self.subTest(message=message):
                self.assertIsNone(parse_literal_phrase_query(message))

    def test_single_entry_rejects_reported_negated_question_and_extra_actions(self):
        command = "将“" + PHRASE + "”作为单个词加入词库"
        for message in (
            "转发：" + command, "他说" + command, "不要" + command,
            command + "吗", command + "？", command + "并提交",
            "如果可以，" + command, command + "，然后删除旧词",
            "“" + command + "”", command.replace("单个词", "两个词"),
            command.replace("“", "「", 1),
            "将“头痛医头，提交草稿”作为单个词加入词库",
        ):
            with self.subTest(message=message):
                self.assertIsNone(parse_explicit_single_phrase_add(message))


class LiteralPhraseReviewTests(unittest.IsolatedAsyncioTestCase):
    @contextmanager
    def reference_runtime(self, *, encode=None, row=CEDICT_PHRASE_ROW):
        with tempfile.TemporaryDirectory(prefix="sep20-phrase-reference-") as directory, ExitStack() as stack:
            db_path = directory + "/reference.db"
            connection = sqlite3.connect(db_path)
            connection.execute("CREATE TABLE readings (word TEXT, normalized TEXT, display TEXT, source_reading TEXT, dataset TEXT)")
            if row is not None:
                connection.execute("INSERT INTO readings VALUES (?, ?, ?, ?, ?)", row)
            connection.commit()
            connection.close()
            stack.enter_context(patch.dict(os.environ, {"PINYIN_REFERENCE_DB": db_path}))
            stack.enter_context(patch.object(review, "AUTHORITATIVE_SOURCES", []))
            for name, replacement in {
                "fetch_keytao_encode": AsyncMock(return_value=encode or phrase_encode_fixture()),
                "lookup_words": AsyncMock(return_value={PHRASE: []}),
                "lookup_codes": AsyncMock(return_value={}),
                "_infer_entity_knowledge": AsyncMock(side_effect=AssertionError("No model")),
                "_infer_semantic_pronunciation_for_review": AsyncMock(side_effect=AssertionError("No model")),
            }.items():
                stack.enter_context(patch.object(review, name, replacement))
            review._clear_review_caches()
            try:
                yield
            finally:
                review._clear_review_caches()

    async def test_real_review_maps_spoken_syllables_without_changing_literal_word(self):
        evidence = {
            "success": True, "lookupComplete": True, "sources": [],
            "groups": [{
                "pinyin": " ".join(PINYINS), "normalized": NORMALIZED,
                "sources": [{"source": "Offline whole-phrase dictionary fixture",
                             "category": "dictionary", "trust": 5}],
                "sourceIds": ["zdic_cibs"], "score": 5, "fallback": False,
            }],
        }
        encode_call = AsyncMock(return_value=phrase_encode_fixture())
        with ExitStack() as stack:
            for name, replacement in {
                "collect_pronunciation_evidence_limited": AsyncMock(return_value=evidence),
                "fetch_keytao_encode": encode_call,
                "lookup_words": AsyncMock(return_value={PHRASE: []}),
                "lookup_codes": AsyncMock(return_value={}),
                "_infer_entity_knowledge": AsyncMock(side_effect=AssertionError("No model")),
                "_infer_semantic_pronunciation_for_review": AsyncMock(side_effect=AssertionError("No model")),
            }.items():
                stack.enter_context(patch.object(review, name, replacement))
            result = await review.prepare_reviewed_word(None, PHRASE)
        encode_call.assert_awaited_once_with(None, PHRASE)
        self.assertEqual(result.get("word"), PHRASE)
        self.assertEqual(result.get("recommendedCode"), "ttyj", result)
        self.assertEqual(result["pronunciations"][0]["normalized"], NORMALIZED)
        self.assertIs(result["needsManualReview"], True)

    async def test_real_local_collector_review_pending_and_add_keep_literal_identity(self):
        async def real_review(word):
            return await harness._review_tools.keytao_prepare_reviewed_add(word, "qq", "sep20-phrase")

        with self.reference_runtime(), patch.object(
            harness._review_tools, "_build_pre_submit_audit", AsyncMock(return_value={
                "success": True, "needsManualReview": True, "autoApprove": False,
                "summary": "Offline audit retains manual review", "issues": [],
            }),
        ):
            collected = await review.collect_pronunciation_evidence(PHRASE)
            self.assertEqual(collected["groups"][0]["normalized"], CEDICT_PHRASE_ROW[1].split())
            self.assertEqual(collected["groups"][0]["sourceIds"], ["cedict"])
            prepared = await review.prepare_reviewed_word(None, PHRASE)
            self.assertEqual(prepared["recommendedCode"], "ttyj", prepared)
            self.assertIs(prepared["needsManualReview"], True)
            self.assertEqual(prepared["pronunciationCharacterIndexes"], [0, 1, 2, 3, 5, 6, 7, 8])
            ctx, calls, _record = await LiteralPhraseRoutingTests.route(
                self, "将“" + PHRASE + "”作为单个词加入词库", executor=True,
                review_callback=real_review, continue_with_add=True,
            )
        self.assertEqual(ctx.query_record.state.word, PHRASE)
        self.assertIs(ctx.query_record.state.needs_manual_review, True)
        self.assertIn(PHRASE, ctx.add_response)
        writes = [args for name, args in calls if name == "keytao_create_phrase"]
        self.assertEqual(len(writes), 1, calls)
        self.assertEqual(writes[0]["word"], PHRASE)
        self.assertIs(writes[0]["needs_manual_review"], True)

    async def test_poisoned_character_payloads_and_unbound_local_evidence_stay_blocked(self):
        for poison in ("wrong_char", "missing_char", "wrong_reading", "missing_reading",
                       "wrong_punctuation", "punctuation_syllable", "malformed_punctuation_readings",
                       "wrong_identity", "missing_identity"):
            encode = phrase_encode_fixture()
            if poison == "wrong_char":
                encode["chars"][5]["char"] = "角"
            elif poison == "missing_char":
                del encode["chars"][4]
            elif poison == "wrong_reading":
                encode["chars"][5]["pinyins"] = ["jué"]
            elif poison == "missing_reading":
                encode["chars"][5]["pronunciationLookupStatus"] = "absent"
            elif poison == "wrong_punctuation":
                encode["chars"][4]["char"] = "、"
            elif poison == "punctuation_syllable":
                encode["chars"][4]["pinyin"] = "x"
            elif poison == "malformed_punctuation_readings":
                encode["chars"][4]["pinyins"] = 5
            elif poison == "wrong_identity":
                encode["input"] = PHRASE.replace("，", "")
            else:
                del encode["input"]
            with self.subTest(poison=poison), self.reference_runtime(encode=encode):
                result = await review.prepare_reviewed_word(None, PHRASE)
                self.assertEqual(result.get("recommendedCode"), "", result)
                self.assertEqual(result["reviewDisposition"], "BLOCK")
        for row in (None, (PHRASE.replace("，", ""), *CEDICT_PHRASE_ROW[1:]),
                    (PHRASE, CEDICT_PHRASE_ROW[1].replace(" , ", " tou "), *CEDICT_PHRASE_ROW[2:])):
            with self.subTest(row=row), self.reference_runtime(row=row):
                result = await review.prepare_reviewed_word(None, PHRASE)
                self.assertEqual(result.get("recommendedCode"), "", result)
                self.assertEqual(result["reviewDisposition"], "BLOCK")

    async def test_write_validation_rechecks_original_word_against_server_candidates(self):
        for codes, accepted in ((["ttyj"], True), (["ttyt"], False)):
            encode = AsyncMock(return_value={"success": True, "candidateCodes": codes})
            with self.subTest(codes=codes), patch.object(harness._draft_tools, "_fetch_encode_candidates", encode):
                result = await harness._draft_tools._validate_draft_item_code(
                    {"action": "Create", "word": PHRASE, "code": "ttyj", "type": "Phrase"},
                    reviewed_pinyin=" ".join(PINYINS), reviewed_candidate_codes=["ttyj"],
                )
            encode.assert_awaited_once_with(PHRASE, "ttyj")
            self.assertIs(result["success"], accepted)

    def test_web_pronunciation_binding_keeps_literal_punctuation(self):
        for text, expected in (
            (PHRASE + " 拼音：" + " ".join(PINYINS), [tuple(NORMALIZED)]),
            (PHRASE.replace("，", "") + " 拼音：" + " ".join(PINYINS), []),
            ("另一条词 拼音：" + " ".join(PINYINS), []),
        ):
            with self.subTest(text=text):
                sequences, _rejections = review._extract_labeled_pinyin_sequences(text, PHRASE)
                self.assertEqual(sequences, expected)

    def test_missing_default_sequence_cannot_crash_alternate_projection(self):
        for word, index in ((PHRASE, 5), ("头痛医头脚痛医脚", 4)):
            encode = phrase_encode_fixture()
            if "，" not in word:
                del encode["chars"][4]
            encode["chars"][0]["pinyin"] = ""
            encode["candidateCodes"] = ["ttyj", "ttaj"]
            encode["alternatePhrasePronunciationCodes"] = [
                {"charIndex": index, "char": "脚", "pinyin": "jiao", "codes": ["ttaj"]},
            ]
            with self.subTest(word=word):
                self.assertEqual(review._returned_pronunciation_groups(word, encode), [])

    def test_alternate_pronunciation_indexes_remain_bound_to_original_chars(self):
        for index, char, expected_count in ((5, "脚", 2), (4, "，", 1), (5, "角", 1)):
            encode = phrase_encode_fixture()
            encode["candidateCodes"].append("ttaj")
            encode["alternatePhrasePronunciationCodes"] = [
                {"charIndex": index, "char": char, "pinyin": "jue", "codes": ["ttaj"]},
            ]
            with self.subTest(index=index, char=char):
                groups = review._returned_pronunciation_groups(PHRASE, encode)
                self.assertEqual(len(groups), expected_count)
                if expected_count == 2:
                    self.assertEqual(groups[1]["normalized"][4], "jue")

    async def test_invalid_full_variant_cannot_fall_back_to_an_index_only_variant(self):
        encode = phrase_encode_fixture()
        encode["candidateCodes"].append("ttay")
        encode["alternatePhrasePronunciationCodes"] = [{
            "charIndex": 5, "char": "脚", "pinyin": "jiao", "codes": ["ttay"],
            "pinyins": [*NORMALIZED[:4], "tou", *NORMALIZED[4:]],
        }]
        with self.reference_runtime(encode=encode):
            result = await review.prepare_reviewed_word(None, PHRASE)
        self.assertEqual(result["recommendedCode"], "ttyj", result)
        self.assertEqual(result["pronunciations"][0]["codes"], ["ttyj"])


if __name__ == "__main__":
    unittest.main()
