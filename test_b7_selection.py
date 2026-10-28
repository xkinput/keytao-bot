"""Offline regressions for advertised B7 add and displacement choices."""

import copy
import unittest
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from keytao_bot.harness import authorization_grammar as grammar
from keytao_bot.harness.operation_request import understand_message
from keytao_bot.harness.state import PendingAddWord, PendingToolConfirm, _pending_add_word_payload
from keytao_bot.plugins import chat_commands as commands, chat_routing as routing


def incident_ticket(words=None):
    inventories = {
        "一夫一妻": ("yfyq", [("yfyq", False)], ""),
        "一夫多妻": ("yfdq", [("yfdq", False)], ""),
        "吴三桂": ("wsg", [("wsg", True), ("wsgo", False)], "我送给"),
        "陈圆圆": ("jyy", [("jyy", True), ("jyya", False)], "纯音乐"),
        "多尔衮": ("deg", [("deg", False), ("dego", False), ("degov", False)], ""),
        "皇太极": ("htju", [("htj", True), ("htju", True), ("htjuv", False), ("htjuvv", False)], "航天局"),
        "福临": ("fjlb", [("fjlb", True), ("fjlbo", False), ("fjlboi", False)], "富临"),
    }
    items, scopes = [], []
    for word in words or inventories:
        code, candidates, occupant = inventories[word]
        occupied = {code: [occupant]} if occupant else {}
        if word == "皇太极":
            occupied["htj"] = ["火腿肠"]
        assessments = [{
            "verdict": "front_more_common", "newWord": word,
            "occupantWord": occupant, "occupantCode": code, "newCode": code,
            "freeCode": next(candidate for candidate, used in candidates if not used),
        }] if occupant else []
        reviewed = PendingAddWord(
            word, code, candidates, server_candidates=candidates,
            server_occupied_words=occupied, server_ordering_assessments=assessments,
            pronunciation_codes={candidate: "offline reading" for candidate, _ in candidates},
            needs_manual_review=False,
        )
        items.append({"action": "Create", "word": word, "code": code})
        scopes.append({
            "word": word, "candidates": candidates, "occupiedWords": occupied,
            "orderingAssessments": assessments, "reviewedState": _pending_add_word_payload(reviewed),
            "reviewedPrompt": "Offline authoritative review",
        })
    return PendingToolConfirm("keytao_batch_add_to_draft", {
        "items": items, "_candidate_scopes": scopes, "_reviewed_multi_word": True,
        "_query_words": [item["word"] for item in items],
    })


class B7SelectionTests(unittest.IsolatedAsyncioTestCase):
    def resolve(self, message, ticket=None):
        ticket = ticket or incident_ticket()
        original = copy.deepcopy(ticket)
        derived, intent, error = routing._resolve_multi_word_pending_candidate_selection(ticket, message)
        self.assertEqual(ticket, original)
        self.assertIsNone(error, error)
        self.assertIsNotNone(derived)
        self.assertEqual(intent.intent, "pending_confirm")
        self.assertTrue(routing.message_authorizes_live_pending_mutation(message, ticket))
        return derived

    def test_number_and_code_keep_advertised_reorder(self):
        for message in ("吴三桂 1，陈圆圆 1", "吴三桂 wsg，陈圆圆 jyy"):
            with self.subTest(message=message):
                derived = self.resolve(message)
                plan = commands._pending_batch_front_insert_plan(derived)
                self.assertIsNotNone(plan)
                self.assertFalse(plan.get("invalid"))
                self.assertEqual(plan["expectedOccupantsByCode"], {"wsg": ["我送给"], "jyy": ["纯音乐"]})
                self.assertFalse(any(item.get("manualReviewReason") == "重码添加需管理员审核" for item in derived.args["items"]))

    def test_alternate_empty_code_stays_plain_add(self):
        derived = self.resolve("吴三桂 2，陈圆圆 jyya")
        self.assertIsNone(commands._pending_batch_front_insert_plan(derived))
        self.assertEqual([(item["word"], item["code"]) for item in derived.args["items"]], [("吴三桂", "wsgo"), ("陈圆圆", "jyya")])

    async def test_unadvertised_occupied_code_stays_duplicate_add(self):
        ticket = incident_ticket(("皇太极",))
        scope = ticket.args["_candidate_scopes"][0]
        scope["orderingAssessments"].append({
            **scope["orderingAssessments"][0],
            "occupantWord": "火腿肠", "occupantCode": "htj", "newCode": "htj",
        })
        scope["reviewedState"]["serverOrderingAssessments"] = copy.deepcopy(scope["orderingAssessments"])
        for message in ("皇太极 1", "皇太极 htj"):
            with self.subTest(message=message):
                store = harness.MemoryConversationStateStore()
                key = harness.ConversationAddress.private("qq", "b7-selection")
                store.set(key, copy.deepcopy(ticket))
                execute = AsyncMock(return_value="Offline duplicate add receipt")
                with patch.object(harness.openai_chat_module, "conversation_state_store", store), patch.object(commands, "_execute_confirmed_tool", execute):
                    response = await commands.handle_pending_message_core(message, "qq", "b7-selection", key, allow_intent_model=False)
                self.assertIn("Offline duplicate add receipt", response)
                execute.assert_awaited_once()
                sent = execute.await_args.args[0]
                self.assertEqual(sent.function_name, "keytao_batch_add_to_draft")
                self.assertFalse(execute.await_args.kwargs.get("auto_confirm_shift_plan"))
                derived = self.resolve(message, ticket)
                item = derived.args["items"][0]
                self.assertEqual(item["code"], "htj")
                self.assertTrue(item["needsManualReview"])
                self.assertEqual(item["manualReviewReason"], "重码添加需管理员审核")
                derived_scope = derived.args["_candidate_scopes"][0]
                self.assertEqual(derived_scope["orderingAssessments"], [])
                self.assertEqual(derived_scope["reviewedState"]["serverOrderingAssessments"], [])
                self.assertIsNone(commands._pending_batch_front_insert_plan(derived))

    def test_exact_t7_keeps_only_selected_reorders(self):
        derived = self.resolve("一夫一妻 1， 一夫多妻 1， 吴三桂 1 ， 陈圆圆 1 ， 多尔衮 3， 皇太极 3， 福临 2")
        plan = commands._pending_batch_front_insert_plan(derived)
        self.assertFalse(plan.get("invalid"))
        self.assertEqual(plan["expectedOccupantsByCode"], {"wsg": ["我送给"], "jyy": ["纯音乐"]})
        self.assertEqual({(item["word"], item["code"]) for item in plan["additionalItems"]}, {
            ("一夫一妻", "yfyq"), ("一夫多妻", "yfdq"), ("多尔衮", "degov"), ("皇太极", "htjuv"), ("福临", "fjlbo"),
        })

    def test_t3_and_whitespace_mixed_shift_accept_same_advertised_pairs(self):
        for message in (
            "一夫一妻 1， 一夫多妻 1， 吴三桂 1， 我送给 顺延， 陈圆圆 1， 纯音乐顺延， 多尔衮 3",
            "吴三桂 1 我送给 顺延，陈圆圆 jyy，「纯音乐」顺延",
            "吴三桂 1 我送给顺延 陈圆圆 1 纯音乐顺延",
            '"我送给"顺延 吴三桂 wsg 陈圆圆 1 「纯音乐」顺延',
            '吴三桂 1，陈圆圆 1，"我送给" 和 「纯音乐」顺延',
            "「我送给」和「纯音乐」顺延",
            "我送给和纯音乐顺延",
            "吴三桂 1 陈圆圆 1 我送给和纯音乐顺延",
        ):
            with self.subTest(message=message):
                request = understand_message(message, incident_ticket())
                self.assertEqual(request.op, "select_candidate")
                self.assertTrue(request.refers_to_pending)
                derived = self.resolve(message)
                self.assertEqual(commands._pending_batch_front_insert_plan(derived)["expectedOccupantsByCode"], {"wsg": ["我送给"], "jyy": ["纯音乐"]})

    def test_shift_conjunction_preserves_quoted_and_bound_literal_words(self):
        self.assertEqual(grammar.parse_reviewed_shift_words("我送给和纯音乐顺延"), ("我送给", "纯音乐"))
        self.assertEqual(grammar.parse_reviewed_shift_words("「共和国」和纯音乐顺延"), ("共和国", "纯音乐"))
        self.assertEqual(grammar.parse_reviewed_shift_words("和平顺延"), ("和平",))
        self.assertEqual(grammar.parse_reviewed_shift_words("共和国顺延", known_words=("共和国",)), ("共和国",))
        self.assertEqual(grammar.parse_reviewed_shift_words("共和国和我送给顺延", known_words=("共和国", "我送给")), ("共和国", "我送给"))
        self.assertEqual(grammar.parse_reviewed_shift_words("「不忘初心」顺延"), ("不忘初心",))
        for message in ("他说我送给顺延", "不要我送给顺延", "我送给和她说纯音乐顺延"):
            self.assertIsNone(grammar.parse_reviewed_shift_words(message))
        ticket = incident_ticket(("吴三桂",))
        scope = ticket.args["_candidate_scopes"][0]
        scope["occupiedWords"]["wsg"] = ["共和国"]
        scope["orderingAssessments"][0]["occupantWord"] = "共和国"
        scope["reviewedState"]["serverOccupiedWords"] = scope["occupiedWords"]
        scope["reviewedState"]["serverOrderingAssessments"] = scope["orderingAssessments"]
        derived = self.resolve("吴三桂 1 共和国顺延", ticket)
        self.assertEqual(commands._pending_batch_front_insert_plan(derived)["expectedOccupantsByCode"], {"wsg": ["共和国"]})

    def test_unknown_occupant_contradictions_and_injections_fail_closed(self):
        for message in (
            "吴三桂 2，我送给顺延", "吴三桂 1，陌生人顺延", "吴三桂 1，我送给和陌生人顺延",
            "吴三桂 1，我送给顺延，然后删除纯音乐", "他说我送给顺延", "不要我送给顺延",
            "「我送给」顺延？", '"我送给顺延"', "吴三桂 1，吴三桂 2",
            "吴三桂 1我送给顺延", "吴三桂 1 我送给顺延删除纯音乐",
            "吴三桂 1 我送给顺延 删除纯音乐 1", '吴三桂 1，"我送给」顺延',
            '吴三桂 1，我送给顺延"', "转述吴三桂 1 我送给顺延",
            "不要吴三桂 1 我送给顺延", "吴三桂 1 我送给顺延？",
        ):
            with self.subTest(message=message):
                derived, intent, _ = routing._resolve_multi_word_pending_candidate_selection(incident_ticket(), message)
                self.assertIsNone(derived)
                self.assertIsNone(intent)
                self.assertFalse(routing.message_authorizes_live_pending_mutation(message, incident_ticket()))

    async def test_selection_dispatches_existing_sealed_shift_plan(self):
        store = harness.MemoryConversationStateStore()
        key = harness.ConversationAddress.private("qq", "b7-selection")
        store.set(key, incident_ticket())
        execute = AsyncMock(return_value="Offline reorder receipt")
        with patch.object(harness.openai_chat_module, "conversation_state_store", store), patch.object(commands, "_execute_confirmed_tool", execute):
            response = await commands.handle_pending_message_core("吴三桂 1，陈圆圆 1", "qq", "b7-selection", key, allow_intent_model=False)
        self.assertIn("Offline reorder receipt", response)
        execute.assert_awaited_once()
        sent = execute.await_args.args[0]
        self.assertEqual(sent.function_name, "keytao_shift_phrase_code")
        self.assertEqual(sent.args["_expected_occupants_by_code"], {"wsg": ["我送给"], "jyy": ["纯音乐"]})
        self.assertTrue(execute.await_args.kwargs["auto_confirm_shift_plan"])


if __name__ == "__main__":
    unittest.main()
