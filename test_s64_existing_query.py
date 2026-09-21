"""S64 existing-word delivery: real renderers/gates, fake tools, local BCC DBs."""

import asyncio
from contextlib import closing, contextmanager
import copy
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
import test_s63_bcc as bcc_fixtures
from keytao_bot.utils.explicit_code import parse_explicit_code_request, parse_explicit_entry_code_request
from keytao_bot.utils.pending_confirmation import advertised_batch_binding_pairs, advertised_reply_contract, already_existing_word_copy

chat = harness.openai_chat_module
commands = harness.chat_commands_module
CORPUS_MODES = ("absent", "installed_miss", "modern_hit", "historical_hit")


def lookup_fixture(word="单份", code="dffn", phrase_type="Phrase"):
    row = {"word": word, "code": code, "weight": 100, "type": phrase_type}
    codes = [code, code + "o", code + "i"]
    if word == "单份":
        codes = ["dffn", "dffno", "dffnoi", "effn", "effno", "effnoi"]
    statuses = [{"code": candidate, "occupied": candidate == code,
                 "words": [word] if candidate == code else [],
                 "entries": [(word, 100)] if candidate == code else [],
                 "label": f"已有「{word}」" if candidate == code else "空位"}
                for candidate in codes]
    encoding = {"success": True, "word": word, "type": "二字词" if len(word) == 2 else "fixture",
                "baseCode": code, "recommendedCode": codes[1],
                "candidateCodes": codes, "candidateStatuses": statuses}
    return {
        "keytao_lookup_by_word": {"success": True, "word": word, "phrases": [row]},
        "keytao_list_draft_items": {"success": True, "count": 0, "items": []},
        "keytao_pending_items_by_words": {"success": True, "complete": True, "items": []},
        "keytao_lookup_by_words_batch": {"success": True, "count": 1, "results": [{"word": word, "phrases": [row]}]},
        "keytao_lookup_by_code": {"success": True, "code": code, "count": 1, "phrases": [row]},
        "keytao_encode": encoding,
    }


def pending_candidate(fixture):
    encoding = fixture["keytao_encode"]
    word = encoding["word"]
    return harness.AgentOrchestrator._trusted_single_pending_add(
        {word: tuple((row["code"], row["occupied"]) for row in encoding["candidateStatuses"])},
        {word: tuple(encoding["candidateStatuses"])},
        {word: encoding["recommendedCode"]}, {},
    )


class ExistingQueryTests(unittest.TestCase):
    def setUp(self):
        bcc_fixtures.BccCommonnessTests.setUp(self)

    @contextmanager
    def corpus(self, mode):
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "reference.db"
            shutil.copyfile(self.db, db)
            with closing(sqlite3.connect(db)) as connection, connection:
                if mode == "absent":
                    connection.execute("DROP TABLE bcc_frequency")
                    connection.execute("DROP TABLE bcc_dataset")
                elif mode == "modern_hit":
                    # Synthetic attestation tests installation behavior, not real counts.
                    connection.execute("INSERT INTO bcc_frequency SELECT filename, '单份', 100, 10, 1 FROM bcc_dataset WHERE channel='多领域' AND token_type='word'")
            if mode == "historical_hit":
                bcc_fixtures.seed_historical(db, counts={"单份": (100, 1)})
                bcc_fixtures.seed_historical(db, channel="近代汉语", counts={"单份": (50, 2)})
            with patch.object(commands.keytao_review, "reference_db_path", return_value=db):
                commands.keytao_review._clear_review_caches()
                evidence = commands.keytao_review._query_commonness_reference("单份")["bcc"]
                self.assertEqual(evidence["available"], mode != "absent")
                self.assertEqual(evidence["attested"], mode == "modern_hit")
                if mode == "historical_hit":
                    self.assertEqual([row["count"] for row in evidence["historicalChannels"]], [100, 50])
                yield
            commands.keytao_review._clear_review_caches()

    def test_existing_query_delivery_with_and_without_bcc_and_pending_candidates(self):
        async def run():
            for mode in CORPUS_MODES:
                with self.corpus(mode):
                    for word, code, phrase_type in (("单份", "dffn", "Phrase"), ("乔布斯", "qbsuv", "Phrase"), ("单", "dfo", "Single")):
                        for pending in ("none", "same_word", "other_word"):
                            with self.subTest(corpus=mode, word=word, pending=pending):
                                fixture = lookup_fixture(word, code, phrase_type)
                                store = harness.MemoryConversationStateStore()
                                key = harness.ConversationAddress.group("qq", "865189947", "s64-fixture")
                                previous = pending_candidate(fixture if pending == "same_word" else lookup_fixture("旧词", "jqci"))
                                if pending != "none":
                                    store.set(key, previous)
                                calls = []

                                async def tool(name, args, *_args, **_kwargs):
                                    self.assertIn(name, fixture, "Unexpected tool or mutation")
                                    calls.append(name)
                                    return json.dumps(fixture[name])

                                with (
                                    patch.object(chat, "conversation_state_store", store),
                                    patch.object(chat, "call_tool_function", side_effect=tool),
                                    patch.object(chat, "_get_simple_word_query_words", AsyncMock(return_value=(word,))),
                                    patch.object(chat, "get_ai_response_core", AsyncMock(side_effect=AssertionError("No model calls"))),
                                    patch.object(chat.logger, "warning") as warnings,
                                ):
                                    raw = await commands._try_handle_simple_single_word_query(word, "qq", key.actor_id, key)
                                    self.assertEqual(chat._enforce_advertised_reply_contract(raw, key), raw)
                                    ctx = SimpleNamespace(response=raw, conv_key=key, normalized_message_text=word,
                                                          platform="qq", user_id=key.actor_id, simple_word_query_words=(),
                                                          generic_command_intent=harness.MessageCommandIntent())
                                    await chat._stage_normalize_response(ctx)
                                    await chat._stage_augment_word_query(ctx)
                                    await chat._stage_scope_language_only_response(ctx)
                                    await chat._stage_append_ticket_challenge(ctx)
                                    before_gate = ctx.response
                                    await chat._stage_enforce_advertised_reply_contract(ctx)
                                    self.assertEqual(ctx.response, before_gate)
                                    delivered = chat._prepare_user_facing_reply(ctx.response, SimpleNamespace(conversation_address=key, platform="qq"))
                                    self.assertEqual(delivered.splitlines(), [line for line in before_gate.splitlines() if line])
                                    self.assertIn(f"「{word}」已在词库（{code}）", delivered)
                                    self.assertIn("无需操作", delivered)
                                    self.assertIn("追加一个编码", delivered)
                                    self.assertFalse(any("replace_missing_state" in str(call) for call in warnings.call_args_list))
                                    self.assertEqual(calls, ["keytao_lookup_by_word", "keytao_pending_items_by_words", "keytao_lookup_by_words_batch", "keytao_encode"])
                                    if pending == "none":
                                        self.assertEqual(advertised_reply_contract(raw).command_suggestions, ("换码",))
                                        self.assertTrue(chat._chat_routing._pending_trusted_word_action_matches(store.get(key), "换码"))
                                        self.assertIsNotNone(parse_explicit_code_request("加入编码 " + code + "o", word))
                                    else:
                                        self.assertEqual(store.get(key), previous)
                                        self.assertNotIn("回复「换码」", delivered)
                                        template = f"给 {word} 加一个码 <code>"
                                        self.assertIn(template, delivered)
                                        parsed = parse_explicit_entry_code_request(template.replace("<code>", code + "o"))
                                        self.assertEqual((parsed.word, parsed.code), (word, code + "o"))
        asyncio.run(run())

    def test_invalid_advertisement_is_still_rejected(self):
        store = harness.MemoryConversationStateStore()
        key = harness.ConversationAddress.private("qq", "s64-negative")
        store.set(key, pending_candidate(lookup_fixture()))
        unbound = already_existing_word_copy("单份", ("dffn",), can_choose_other_code=True)
        self.assertEqual(advertised_reply_contract(unbound).command_suggestions, ("换码",))
        self.assertEqual(advertised_batch_binding_pairs(unbound), ())
        with patch.object(chat, "conversation_state_store", store), patch.object(chat.logger, "warning") as warnings:
            self.assertEqual(chat._enforce_advertised_reply_contract(unbound, key), "当前没有可验证的可执行操作，本次未写入。")
            self.assertTrue(any("branch=replace_missing_state state=PendingAddWord bindings=0" in str(call) for call in warnings.call_args_list))

    def test_real_orchestrator_builds_and_delivers_existing_candidate_record(self):
        async def run():
            for mode in CORPUS_MODES:
                with self.subTest(corpus=mode), self.corpus(mode):
                    fixture = lookup_fixture()
                    calls = []
                    def tool_call(name, args):
                        return SimpleNamespace(id=name, type="function", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
                    def tool_function(name):
                        self.assertIn(name, fixture, "Unexpected tool or mutation")
                        async def invoke(**kwargs):
                            calls.append(name)
                            return copy.deepcopy(fixture[name])
                        return invoke
                    model_calls = [tool_call(name, {"words": ["单份"]} if name in {"keytao_pending_items_by_words", "keytao_lookup_by_words_batch"} else {"code": "dffn"} if name == "keytao_lookup_by_code" else {} if name == "keytao_list_draft_items" else {"word": "单份"}) for name in fixture]
                    client = harness._FakeClient([
                        harness._FakeAIResponse("tool_calls", "", model_calls[:3]),
                        harness._FakeAIResponse("tool_calls", "", model_calls[3:]),
                        harness._FakeAIResponse("stop", "「单份」候选编码：\n1. dffn — 已有「单份」\n2. dffno — 空位（推荐）"),
                    ])
                    schemas = [{"type": "function", "function": {"name": name, "description": "Fixture", "parameters": {"type": "object", "properties": {"word": {"type": "string"}, "words": {"type": "array", "items": {"type": "string"}}, "code": {"type": "string"}}}}} for name in fixture]
                    manager = SimpleNamespace(get_skill_instructions=lambda: "", has_tools=lambda: True, get_tools=lambda: schemas)
                    store = harness.MemoryConversationStateStore()
                    context = harness.AgentRequestContext(platform="qq", user_id="s64-fixture", space_type="group", space_id="865189947")
                    agent = harness.AgentOrchestrator(client_factory=lambda: client,
                        runtime=harness.AgentRuntimeConfig(model="fake-model", max_tokens=1000, temperature=0.0, timeout=10),
                        skills_manager=manager, tool_executor=harness.ToolExecutor(tool_function, frozenset()),
                        state_store=store, bind_help_text="fixture", system_prompt_core="fixture")
                    raw = await agent.run("单份", context)
                    record = store.get_record(context.conversation_address)
                    self.assertIsInstance(record.state, harness.PendingAddWord)
                    self.assertEqual(len(client.completions.calls), 3)
                    self.assertEqual(calls, list(fixture))
                    self.assertIn("「单份」已在词库（dffn）", raw)
                    self.assertIn("加入并提交", raw)
                    self.assertIn("回复编号或编码选择", raw)
                    with patch.object(chat, "conversation_state_store", store):
                        self.assertEqual(chat._enforce_advertised_reply_contract(raw, context.conversation_address), raw)
                        async def repeat_lookup(name, args, *_args, **_kwargs):
                            self.assertIn(name, fixture, "Unexpected tool or mutation")
                            return json.dumps(fixture[name])
                        with patch.object(chat, "call_tool_function", side_effect=repeat_lookup), patch.object(
                            chat, "_get_simple_word_query_words", AsyncMock(return_value=("单份",)),
                        ):
                            repeated = await commands._try_handle_simple_single_word_query(
                                "单份", "qq", context.user_id, context.conversation_address,
                            )
                            self.assertIn("「单份」已在词库（dffn）", repeated)
                            self.assertEqual(chat._enforce_advertised_reply_contract(repeated, context.conversation_address), repeated)
                            self.assertEqual(store.get(context.conversation_address), record.state)
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
