"""S58 explicit suffixes retain reviewed reading, occupancy and plan seals."""

import copy
import unittest
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from keytao_bot.plugins import chat_commands as commands


draft_tools = harness._draft_tools


def reviewed_destination(pinyin="nóng hòu fēn wéi"):
    return {
        "success": True, "word": "浓厚氛围", "type": "Phrase", "recommendedCode": "nhfwa",
        "pronunciations": [{
            "pinyin": pinyin, "recommendedCode": "nhfwa",
            "candidateStatuses": [{"code": code, "occupied": code == "nhfw"} for code in ("nhfw", "nhfwa", "nhfwav")],
        }],
    }


class ExplicitDestinationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.rows = [
            {"word": "奈飞", "code": "nhfwv", "type": "Phrase", "weight": 100},
            {"word": "浓厚氛围", "code": "nhfw", "type": "Phrase", "weight": 100},
        ]
        self.candidates = {
            "奈飞": ["nhfw", "nhfwv", "nhfwva"],
            "浓厚氛围": ["nhfw", "nhfwa", "nhfwav"],
            "浓厚风味": ["nhfw", "nhfwa", "nhfwao"],
        }
        self.draft = {"success": True, "batchId": "", "contentVersion": 0, "items": []}
        self.review = reviewed_destination()
        self.strict_calls = []
        self.words = ["奈飞", "浓厚氛围"]

    async def encode(self, word, _requested_code=None):
        return {"success": True, "word": word, "candidateCodes": self.candidates[word]}

    async def lookup_words(self, words):
        return {"success": True, "results": [
            {"word": word, "phrases": [copy.deepcopy(row) for row in self.rows if row["word"] == word]}
            for word in words
        ]}

    async def lookup_codes(self, codes):
        return {"success": True, "results": [
            {"code": code, "phrases": [copy.deepcopy(row) for row in self.rows if row["code"] == code]}
            for code in codes
        ]}

    async def strict(self, platform, user, items, **kwargs):
        self.strict_calls.append((copy.deepcopy(items), dict(kwargs)))
        # Exercise the actual final code-validation sink, including stripping
        # private reading capabilities and preserving the manual-review seal.
        valid, failed = await draft_tools._split_items_by_code_validation(items)
        self.assertEqual(failed, [])
        self.assertEqual(len(valid), len(items))
        if kwargs.get("confirmed"):
            self.draft = {"success": True, "batchId": "s58-local", "contentVersion": 1, "items": valid}
            return {"success": True, "batchId": "s58-local", "contentVersion": 1, "successCount": len(valid)}
        return {"success": False, "requiresConfirmation": True, "warningDigest": "b" * 64, "warnings": []}

    async def invoke(self, destination, **kwargs):
        with (
            patch.object(draft_tools, "_fetch_encode_candidates", self.encode),
            patch.object(draft_tools, "_lookup_words_raw", self.lookup_words),
            patch.object(draft_tools, "_lookup_codes_raw", self.lookup_codes),
            patch.object(draft_tools, "keytao_list_draft_items", AsyncMock(side_effect=lambda *a, **k: copy.deepcopy(self.draft))),
            patch.object(draft_tools, "prepare_reviewed_word", AsyncMock(side_effect=lambda *a, **k: copy.deepcopy(self.review))),
            patch.object(draft_tools, "_keytao_strict_batch_add_to_draft", self.strict),
        ):
            return await draft_tools.keytao_shift_phrase_code(
                "qq", "s58-review", "奈飞", "nhfw",
                ordered_words=self.words, listed_words=self.words,
                expected_codes=["nhfw", destination], **kwargs,
            )

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
            patch.object(commands, "_load_merged_word_entries", AsyncMock(return_value={row["word"]: row for row in self.rows})),
            patch.object(commands, "_execute_shift_to_code", AsyncMock(return_value="sealed preview")) as planner,
            patch.object(commands, "call_tool_function", AsyncMock(side_effect=AssertionError("route must let the shared planner validate codes"))),
        ):
            result = await commands.try_handle_move_to_code_command(
                "把 奈飞 调到 nhfw，把 浓厚氛围 挪到 nhfwzz", "qq", "s58-review",
            )
        self.assertEqual(result, "sealed preview")
        self.assertEqual(planner.await_args.kwargs["expected_codes"], ["nhfw", "nhfwzz"])


if __name__ == "__main__":
    unittest.main()
