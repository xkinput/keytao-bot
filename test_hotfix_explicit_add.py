"""Offline regression for explicit adds with reviewed pronunciation groups."""

import copy
import json
import unittest
from unittest.mock import AsyncMock, patch

import test_state_machine as harness

# Keep the shared fake runtime compatible with the combined safety suite.
harness._fake_nonebot.init = lambda: None
harness._fake_log.logger.exception = harness._fake_log.logger.error

from keytao_bot.harness.operation_request import OperationRequest, understand_message
from keytao_bot.harness.state import DraftOperationCoordinator, MemoryConversationStateStore
from keytao_bot.utils.candidate_inventory import select_candidate_inventory
from keytao_bot.utils.explicit_code import ExplicitEntryCodeRequest
from keytao_bot.utils.keytao_review import _build_statuses_for_codes


WORD = "没头没尾"
READING = "mei tou mei wei"
CODES = ["mtmw", "mtmwa", "mtmwao"]
ALTERNATE_CODES = ["mtmy", "mtmya", "mtmyao"]


def reviewed_word(layout="mixed"):
    groups = []
    for reading, codes in ((READING, CODES), ("mei tou mei yi", ALTERNATE_CODES)):
        group = {
            "pinyin": reading, "normalized": reading.split(), "codes": list(codes),
            "recommendedCode": codes[0], "score": 100, "fallback": False,
            "requiresManualReview": False,
            "sources": [{"source": "汉典（经编码服务）", "url": "https://www.zdic.net/hans/没头没尾"}],
        }
        if layout == "per_group" or (layout == "mixed" and reading == READING):
            group["candidateStatuses"] = _build_statuses_for_codes(codes, {})
        groups.append(group)
    result = {
        "success": True, "word": WORD, "type": "Phrase", "existing": [],
        "pronunciations": groups, "recommendedCode": CODES[0],
        "reviewDisposition": "PASS", "needsManualReview": False,
        "lookupFailed": False, "autoReviewable": True,
        "pronunciationEvidenceStatus": "found", "pronunciationEvidenceComplete": True,
    }
    if layout != "per_group":
        result["candidateStatuses"] = _build_statuses_for_codes(CODES + ALTERNATE_CODES, {})
    return result


class ExplicitAddHotfixTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.chat = harness.openai_chat_module
        self.commands = harness.chat_commands_module
        self.store = MemoryConversationStateStore()
        self.author = harness.ConversationAddress.group("qq", "hotfix-group", "hotfix-owner")
        self.review = reviewed_word()
        self.calls = []
        self.writes = []
        patches = (
            patch.object(self.chat, "conversation_state_store", self.store),
            patch.object(self.chat, "draft_operation_coordinator", DraftOperationCoordinator()),
            patch.object(self.chat, "call_tool_function", side_effect=self.call),
            patch.object(self.chat, "_classify_message_command_intent", AsyncMock(side_effect=AssertionError("No model calls"))),
            patch.object(self.commands.user_resolver, "resolve_actor_binding", AsyncMock(return_value=True)),
            patch.object(self.commands, "_format_draft_response", AsyncMock(return_value="Offline draft receipt")),
        )
        for context in patches:
            context.start()
            self.addCleanup(context.stop)
        token = self.chat._current_conversational_turn.set(None)
        self.addCleanup(self.chat._current_conversational_turn.reset, token)

    async def call(self, name, args, platform, user_id, **kwargs):
        self.assertEqual((platform, user_id), ("qq", self.author.actor_id))
        self.calls.append((name, copy.deepcopy(args)))
        if name == "keytao_prepare_reviewed_add":
            self.assertEqual(args, {"word": WORD})
            audit = {"success": True, "verdict": "pass", "autoApprove": True,
                     "needsManualReview": False, "summary": "Offline reviewed fixture", "issues": []}
            # Run the real tool wrapper; all evidence, audit and ordering IO is replaced.
            with patch.object(harness._review_tools, "prepare_reviewed_word", AsyncMock(return_value=copy.deepcopy(self.review))), patch.object(
                harness._review_tools, "_build_pre_submit_audit", AsyncMock(return_value=audit),
            ), patch.object(harness._review_tools, "assess_candidate_chain_commonness", AsyncMock(return_value=[])):
                result = await harness._review_tools.keytao_prepare_reviewed_add(WORD)
            return json.dumps(result, ensure_ascii=False)
        if name in {"keytao_lookup_by_code", "keytao_lookup_by_word"}:
            return json.dumps({"success": True, "phrases": []})
        if name == "keytao_pending_items_by_words":
            return json.dumps({"success": True, "complete": True, "items": []})
        self.assertEqual(name, "keytao_create_phrase", "Unexpected offline tool")
        result = {"batchId": "offline-hotfix", "contentVersion": 1,
                  "batchUrl": "https://offline.invalid/batch/offline-hotfix"}
        if not args.get("confirmed"):
            return json.dumps({**result, "success": False, "requiresConfirmation": True,
                               "warningDigest": "a" * 64, "warnings": []})
        self.writes.append(copy.deepcopy(args))
        return json.dumps({**result, "success": True, "successCount": 1, "failedCount": 0})

    async def execute(self, route="model", code="mtmw"):
        if route == "model":
            request = OperationRequest(op="add", frame="direct", source="model",
                                       operands=(WORD,), codes=(code,))
        else:
            request = understand_message(f"加词 {WORD} {code}")
            self.assertEqual((request.op, request.operands, request.codes), ("add", (WORD,), (code,)))
            return await self.commands._try_handle_complete_add_command(
                f"加词 {WORD} {code}", "qq", self.author.actor_id,
                self.author, self.author.space_key, "Offline owner",
            )
        command = self.commands.render_understood_command(request)
        self.assertTrue(command)
        return await self.commands.execute_understood_command(
            request, command, "qq", self.author.actor_id, self.author, self.author.space_key, "Offline owner",
        )

    def assert_created(self, reply):
        self.assertEqual(len(self.writes), 1, reply)
        item = self.writes[0]
        self.assertEqual((item.get("action", "Create"), item["word"], item["code"]), ("Create", WORD, "mtmw"))
        self.assertIn("读音 " + READING, item["remark"])
        self.assertEqual(sum(name == "keytao_create_phrase" for name, _ in self.calls), 2)

    def test_reproduction_identifies_unbound_alternate_candidates(self):
        inventory = select_candidate_inventory(reviewed_word())
        self.assertIsNotNone(inventory)
        self.assertEqual(inventory.readings["mtmw"], READING)
        self.assertEqual([code for code, _ in inventory.candidates if not inventory.readings.get(code)], ALTERNATE_CODES)

    async def test_production_model_route_creates_once_with_reviewed_reading(self):
        self.assert_created(await self.execute())

    async def test_plain_grammar_creates_once_with_reviewed_reading(self):
        self.assert_created(await self.execute("grammar"))

    async def test_status_layouts_keep_the_same_reading_binding(self):
        for layout in ("top_level", "per_group"):
            with self.subTest(layout=layout):
                self.review = reviewed_word(layout)
                self.calls.clear()
                self.writes.clear()
                self.store.delete(self.author)
                self.assert_created(await self.execute())

    async def test_unrelated_group_without_reading_does_not_block_requested_code(self):
        self.review = reviewed_word("per_group")
        self.review["pronunciations"][1]["pinyin"] = ""
        self.assert_created(await self.execute())

    async def test_code_outside_reviewed_groups_is_rejected(self):
        # A valid phonetic prefix alone must not authorize an unreviewed suffix.
        for route in ("model", "grammar"):
            with self.subTest(route=route):
                reply = await self.execute(route, code="mtmwi")
                self.assertEqual(self.writes, [], reply)
                self.assertIn("未写入", reply)
                self.assertFalse(any(name == "keytao_lookup_by_code" for name, _ in self.calls))

    async def test_shared_code_across_different_readings_uses_first_reviewed_reading(self):
        alternate = copy.deepcopy(self.review["pronunciations"][0])
        alternate.update(pinyin="mo tou mo wei", normalized=["mo", "tou", "mo", "wei"])
        self.review["pronunciations"].append(alternate)
        for route in ("model", "grammar"):
            with self.subTest(route=route):
                self.calls.clear()
                self.writes.clear()
                self.store.delete(self.author)
                self.assert_created(await self.execute(route))

    async def test_explicit_reading_resolves_a_shared_reviewed_code(self):
        alternate = copy.deepcopy(self.review["pronunciations"][0])
        alternate.update(pinyin="mo tou mo wei", normalized=["mo", "tou", "mo", "wei"])
        self.review["pronunciations"].append(alternate)
        reply = await self.commands._execute_explicit_entry_code_request(
            ExplicitEntryCodeRequest(WORD, "mtmw", pinyin=READING),
            f"添加 {WORD} {READING}:mtmw", "qq", self.author.actor_id,
            self.author, self.author.space_key, "Offline owner",
        )
        self.assert_created(reply)

    async def test_requested_candidate_without_a_reading_is_rejected(self):
        self.review["pronunciations"][0]["pinyin"] = ""
        reply = await self.execute()
        self.assertEqual(self.writes, [], reply)
        self.assertIn("缺少可核验的读音", reply)
        self.assertEqual([name for name, _ in self.calls], ["keytao_prepare_reviewed_add"])

    def mark_resolvable_ambiguity(self):
        self.review.update(
            pronunciationUnresolved=True, reviewDisposition="BLOCK",
            reviewVerdictSite="pronunciation_unresolved",
            multiSenseChoice={"status": "ambiguous"},
        )

    async def test_ambiguous_review_with_unique_code_requires_manual_review(self):
        for route in ("model", "grammar"):
            with self.subTest(route=route):
                self.review = reviewed_word()
                self.mark_resolvable_ambiguity()
                self.calls.clear()
                self.writes.clear()
                self.store.delete(self.author)
                self.assert_created(await self.execute(route))
                self.assertIs(self.writes[0]["needs_manual_review"], True)
                self.assertIn("管理员复核", self.writes[0]["remark"])

    async def test_ambiguous_review_with_shared_code_requires_manual_review(self):
        alternate = copy.deepcopy(self.review["pronunciations"][0])
        alternate["pinyin"] = "mo tou mo wei"
        self.review["pronunciations"].append(alternate)
        self.mark_resolvable_ambiguity()
        for route in ("model", "grammar"):
            with self.subTest(route=route):
                self.calls.clear()
                self.writes.clear()
                self.store.delete(self.author)
                self.assert_created(await self.execute(route))
                self.assertIs(self.writes[0]["needs_manual_review"], True)
                self.assertIn("管理员复核", self.writes[0]["remark"])

    async def test_ambiguous_review_with_explicit_reading_requires_manual_review(self):
        alternate = copy.deepcopy(self.review["pronunciations"][0])
        alternate["pinyin"] = "mo tou mo wei"
        self.review["pronunciations"].append(alternate)
        self.mark_resolvable_ambiguity()
        reply = await self.commands._execute_explicit_entry_code_request(
            ExplicitEntryCodeRequest(WORD, "mtmw", pinyin=READING),
            f"添加 {WORD} {READING}:mtmw", "qq", self.author.actor_id,
            self.author, self.author.space_key, "Offline owner",
        )
        self.assert_created(reply)
        self.assertIs(self.writes[0]["needs_manual_review"], True)
        self.assertIn("管理员复核", self.writes[0]["remark"])

    async def test_unresolved_or_blocked_review_cannot_write(self):
        for change in (
            {"pronunciationUnresolved": True}, {"reviewDisposition": "BLOCK"},
            {"pronunciationUnresolved": True, "reviewDisposition": "BLOCK",
             "reviewVerdictSite": "pronunciation_unresolved", "multiSenseChoice": {"status": "ambiguous"},
             "lookupFailed": True},
        ):
            with self.subTest(change=change):
                self.review = {**reviewed_word(), **change}
                self.calls.clear()
                reply = await self.execute()
                self.assertEqual(self.writes, [], reply)
                self.assertEqual([name for name, _ in self.calls], ["keytao_prepare_reviewed_add"])


if __name__ == "__main__":
    unittest.main()
