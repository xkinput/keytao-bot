"""S60 two-turn replay against fake reviews and exact created-PR receipts."""

import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import test_state_machine as harness


chat = harness.openai_chat_module
commands = harness.chat_commands_module
BATCH_ID = "s60-e4d21c4d"
PR_ITEMS = [
    {"id": 6001, "action": "Create", "word": "吊杠", "code": "dcgp", "type": "Phrase", "weight": 100},
    {"id": 6002, "action": "Delete", "word": "掉岗", "code": "dcgp", "type": "Phrase", "weight": 100},
    {"id": 6003, "action": "Create", "word": "掉岗", "code": "dcgpi", "type": "Phrase", "weight": 100},
    {"id": 6004, "action": "Create", "word": "沃集鲜", "code": "wjxa", "type": "Phrase", "weight": 100},
]


def review_payload(word):
    """Use the public reviewed-add schema, without forging a pending record."""
    code, reading = {"吊杠": ("dcgp", "diào gàng"), "沃集鲜": ("wjxa", "wò jí xiān")}[word]
    statuses = [{
        "code": code, "occupied": word == "吊杠",
        "words": ["掉岗"] if word == "吊杠" else [],
        "phrases": [{"word": "掉岗", "code": code, "type": "Phrase", "weight": 100}]
        if word == "吊杠" else [],
    }]
    if word == "吊杠":
        statuses.append({"code": "dcgpi", "occupied": False, "words": [], "phrases": []})
    return {
        "success": True, "word": word, "type": "Phrase", "recommendedCode": code,
        "autoReviewable": False, "needsManualReview": True,
        "manualReviewReason": "Fixture requires manual review",
        "pronunciations": [{
            "pinyin": reading, "recommendedCode": code,
            "sources": [{"source": "汉典"}], "candidateStatuses": statuses,
        }],
        "candidateOrderingAssessments": [{
            "newWord": word, "occupantWord": "掉岗", "occupantCode": "dcgp",
            "freeCode": "dcgpi", "newCode": "dcgp", "recommendedCode": "dcgp",
            "verdict": "front_more_common", "summary": "Fixture comparison evidence",
        }] if word == "吊杠" else [],
    }


class S60ScenarioTests(unittest.IsolatedAsyncioTestCase):
    async def replay(self, platform, words=("吊杠", "沃集鲜")):
        actor = "s60-fixture-actor"
        store = harness.MemoryConversationStateStore()
        key = harness.ConversationAddress.group(platform, "s60-fixture-group", actor)
        calls = []
        created = []
        submitted = False
        plan = {
            "word": "吊杠", "targetCode": "dcgp",
            "shifted": [{"word": "掉岗", "fromCode": "dcgp", "toCode": "dcgpi"}],
            "items": [{key: value for key, value in row.items() if key != "id"} for row in PR_ITEMS],
        }

        async def tool(name, args, request_platform, request_actor, **_kwargs):
            nonlocal submitted
            self.assertEqual((request_platform, request_actor), (platform, actor))
            calls.append((name, copy.deepcopy(args)))
            if name == "keytao_lookup_by_word":
                result = {"success": True, "phrases": []}
            elif name == "keytao_pending_items_by_words":
                result = {"success": True, "complete": True, "items": []}
            elif name == "keytao_prepare_reviewed_add":
                result = review_payload(args["word"])
            elif name == "keytao_shift_phrase_code":
                self.assertEqual((args["word"], args["target_code"]), ("吊杠", "dcgp"))
                self.assertEqual([(row["word"], row["code"]) for row in args["additional_items"]],
                                 [("沃集鲜", "wjxa")])
                if not args.get("confirmed_plan_digest"):
                    self.assertFalse(created)
                    result = {
                        "success": False, "requiresConfirmation": True,
                        "confirmationKind": "shiftPlan", "batchId": "", "contentVersion": 0,
                        "planDigest": "a" * 64, "warningDigest": "b" * 64,
                        "warnings": [], "shiftPlan": plan,
                    }
                else:
                    self.assertEqual(args["confirmed_plan_digest"], "a" * 64)
                    self.assertEqual(args["expected_content_version"], 0)
                    self.assertFalse(created, "a confirmed write was replayed twice")
                    created.extend(copy.deepcopy(PR_ITEMS))
                    result = {
                        "success": True, "batchId": BATCH_ID, "contentVersion": 1,
                        "pullRequestCount": len(created), "writtenItems": copy.deepcopy(created),
                        "shiftPlan": plan,
                    }
            elif name == "keytao_submit_batch":
                self.assertEqual(args["batch_id"], BATCH_ID)
                self.assertEqual(created, PR_ITEMS)
                if args.get("preview_only"):
                    result = {
                        "success": False, "requiresConfirmation": True,
                        "batchId": BATCH_ID, "contentVersion": 1,
                        "snapshotDigest": "c" * 64, "warningDigest": "d" * 64,
                        "auditDigest": "e" * 64, "snapshotItems": copy.deepcopy(created),
                        "warnings": [],
                    }
                else:
                    self.assertTrue(args.get("confirmed"))
                    self.assertEqual(args["expected_content_version"], 1)
                    self.assertEqual(args["expected_server_snapshot_digest"], "c" * 64)
                    self.assertEqual(args["expected_warning_digest"], "d" * 64)
                    self.assertEqual(args["expected_audit_digest"], "e" * 64)
                    self.assertFalse(submitted)
                    submitted = True
                    result = {"success": True, "status": "Submitted", "batchId": BATCH_ID,
                              "contentVersion": 2}
            else:
                raise AssertionError(f"Unexpected external tool: {name}")
            return json.dumps(result, ensure_ascii=False)

        render = commands.render_server_backed_batch_candidates

        def record_first_render(*args, **kwargs):
            record = store.get_record(key)
            self.assertIsNotNone(record, "candidate reply was rendered before persistence")
            self.assertEqual(record.state.args["_query_words"], list(words))
            return render(*args, **kwargs)

        with (
            patch.object(chat, "conversation_state_store", store),
            patch.object(chat, "call_tool_function", side_effect=tool),
            patch.object(chat, "_classify_simple_word_query_intent",
                         AsyncMock(side_effect=AssertionError("S60 cannot invoke a model"))),
            patch.object(chat, "_classify_message_command_intent",
                         AsyncMock(side_effect=AssertionError("S60 cannot invoke a model"))),
            patch.object(commands, "render_server_backed_batch_candidates", side_effect=record_first_render),
        ):
            discovery = await commands._try_handle_simple_single_word_query(
                " ".join(words), platform, actor, key, key.space_key, "S60 fixture",
            )
            self.assertEqual(discovery.count("审词："), 2)
            self.assertIn("占 dcgp", discovery)
            self.assertIn("掉岗」顺延", discovery)
            self.assertIn("wjxa", discovery)
            self.assertTrue(chat._advertised_reply_matches_live_record(discovery, store.get_record(key)))
            self.assertEqual(chat._enforce_advertised_reply_contract(discovery, key), discovery)
            self.assertFalse(created, "the discovery turn wrote draft rows")
            self.assertEqual([args["word"] for name, args in calls if name == "keytao_prepare_reviewed_add"],
                             list(words))
            reply = await commands.handle_pending_message_core(
                "加入并提交", platform, actor, key,
                space_key=key.space_key, owner_label="S60 fixture",
                allow_intent_model=False,
            )
            self.assertTrue(submitted, reply)
            self.assertEqual(created, PR_ITEMS)
            self.assertIsNone(store.get_record(key))
            context = SimpleNamespace(response=reply, conv_key=key)
            await chat._stage_normalize_response(context)
            delivered = chat._enforce_advertised_reply_contract(context.response, key)

        self.assertEqual([name for name, _args in calls if name in {
            "keytao_shift_phrase_code", "keytao_submit_batch",
        }], ["keytao_shift_phrase_code", "keytao_shift_phrase_code",
             "keytao_submit_batch", "keytao_submit_batch"])
        self.assertIn("吊杠 → dcgp", delivered)
        self.assertIn("掉岗 dcgp→dcgpi", delivered)
        self.assertIn("沃集鲜 → wjxa", delivered)
        self.assertIn("已提交审核", delivered)
        host = "https://keytao.rea.ink" if platform == "qq" else "https://keytao.vercel.app"
        self.assertEqual(delivered.count(f"{host}/batch/{BATCH_ID}"), 1, delivered)
        self.assertLess(delivered.index(words[0]), delivered.index(words[1]), delivered)
        print(json.dumps({"scenario": "S60", "mode": "fixture/fake-model", "platform": platform,
                          "query": " ".join(words), "confirmation": "加入并提交",
                          "createdPullRequestIds": [row["id"] for row in created],
                          "paidModelCalls": 0, "realProviderCalls": 0,
                          "receipt": delivered}, ensure_ascii=False), flush=True)

    async def test_qq_incident_two_turns(self):
        await self.replay("qq")

    async def test_telegram_incident_two_turns(self):
        await self.replay("telegram")

    async def test_qq_reversed_requested_order(self):
        await self.replay("qq", ("沃集鲜", "吊杠"))

    async def test_telegram_reversed_requested_order(self):
        await self.replay("telegram", ("沃集鲜", "吊杠"))


if __name__ == "__main__":
    unittest.main()
