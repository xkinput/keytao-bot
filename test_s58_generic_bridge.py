"""Grammar gaps use explicit proposal-only tool protocols and live operands."""

import copy
import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from keytao_bot.harness.tools import ToolContext, ToolExecutor
from keytao_bot.harness.state import (
    MemoryConversationStateStore, PendingToolConfirm,
    server_warning_pending_state, server_warning_ticket_is_complete,
)


def reviewed(word, code):
    return {
        "type": "Phrase", "pinyin": "nài fēi" if word == "奈飞" else "nóng hòu fēn wéi",
        "candidate_codes": [code], "needs_manual_review": True,
        "remark": "Server-reviewed reading; manual shape review required",
    }


class GenericProposalBridgeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.calls = []
        self.preview_override = {}
        self.entries = {"奈飞": [], "浓厚氛围": []}
        self.rows = [
            {"id": 123, "word": "奈飞", "code": "nhfw", "type": "Phrase", "action": "Create"},
            {"id": 124, "word": "浓厚氛围", "code": "nhfwa", "type": "Phrase", "action": "Create"},
        ]
        self.context = ToolContext(
            platform="qq", user_id="s58-generic", writes_allowed=False,
            current_message="安排添加 奈飞 nhfw",
            trusted_reviewed_items_by_key={
                ("奈飞", "nhfw"): reviewed("奈飞", "nhfw"),
                ("浓厚氛围", "nhfwa"): reviewed("浓厚氛围", "nhfwa"),
            },
        )

        async def dispatch(name, **kwargs):
            self.calls.append((name, copy.deepcopy(kwargs)))
            if name == "keytao_lookup_by_words_batch":
                return {"success": True, "results": [
                    {"word": word, "phrases": copy.deepcopy(self.entries[word])}
                    for word in kwargs["words"]
                ]}
            if name == "keytao_list_draft_items":
                return {"success": True, "batchId": "actor-draft", "contentVersion": 7, "items": copy.deepcopy(self.rows)}
            if name in {"keytao_create_phrase", "keytao_batch_add_to_draft"}:
                self.assertIs(kwargs.get("preview_only"), True)
                self.assertIs(kwargs.get("confirmed"), False)
                self.assertNotIn("expected_warning_digest", kwargs)
                return {
                    "success": False, "requiresConfirmation": True,
                    "batchId": "actor-draft", "contentVersion": 7,
                    "warningDigest": "a" * 64, "message": "Reviewed entry proposal",
                    **copy.deepcopy(self.preview_override),
                }
            self.assertIn(name, {"keytao_remove_draft_item", "keytao_batch_remove_draft_items"})
            self.assertNotIn("expected_target_digest", kwargs)
            self.assertNotIn("expected_content_version", kwargs)
            ids = kwargs.get("ids", [kwargs.get("pr_id")])
            return {
                "success": False, "requiresConfirmation": True,
                "confirmationKind": "deleteTargets", "batchId": "actor-draft",
                "contentVersion": 7, "targetDigest": "b" * 64,
                "targets": [copy.deepcopy(next(row for row in self.rows if row["id"] == value)) for value in ids],
                **copy.deepcopy(self.preview_override),
            }

        registered = {
            "keytao_lookup_by_words_batch", "keytao_list_draft_items",
            "keytao_create_phrase", "keytao_batch_add_to_draft",
            "keytao_remove_draft_item", "keytao_batch_remove_draft_items",
        }
        self.executor = ToolExecutor(
            lambda name: (lambda **kwargs: dispatch(name, **kwargs)) if name in registered else None,
            frozenset(registered - {"keytao_lookup_by_words_batch"}),
        )

    async def call(self, name="keytao_create_phrase", args=None, message=None, context=None):
        active = context or self.context
        if message is not None:
            active = replace(active, current_message=message)
        return json.loads(await self.executor.call(
            name, args if args is not None else {"word": "奈飞", "code": "nhfw"}, active,
        ))

    def assert_ticket(self, result, name):
        self.assertTrue(result.get("grammar_gap_bridged"), result)
        state = server_warning_pending_state(
            PendingToolConfirm(function_name=name, args=result["grammarGapArguments"]), result,
        )
        self.assertTrue(server_warning_ticket_is_complete(state))
        self.assertNotIn("preview_only", result["grammarGapArguments"])
        self.assertNotIn("confirmed", result["grammarGapArguments"])
        return state

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
        import test_state_machine as harness
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
        import test_state_machine
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
        import test_state_machine
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
        import test_state_machine
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
        import test_state_machine as harness
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


if __name__ == "__main__":
    unittest.main()
