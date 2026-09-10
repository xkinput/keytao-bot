"""S58 structural advertisement detection and sealed-ticket redraw controls."""

import asyncio
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from keytao_bot.utils.pending_confirmation import advertised_command_suggestions


chat = harness.openai_chat_module


def shift_ticket():
    return harness.PendingToolConfirm(
        "keytao_shift_phrase_code",
        {"word": "奈飞", "target_code": "nhfw", "batch_id": "",
         "expected_content_version": 0, "confirmed_plan_digest": "a" * 64,
         "_pending_display": {"shiftPlan": {
             "word": "奈飞", "targetCode": "nhfw",
             "currentState": [{"word": "奈飞", "code": "nhfwv"}, {"word": "浓厚氛围", "code": "nhfw"}],
             "proposedState": [{"word": "奈飞", "code": "nhfw"}, {"word": "浓厚氛围", "code": "nhfwa"}],
             "shifted": [{"word": "浓厚氛围", "fromCode": "nhfw", "toCode": "nhfwa"}],
         }}},
        confirmation_source="server_warning",
    )


class S58AdvertisementTests(unittest.TestCase):
    def test_s45_character_facts_survive_unbound_linguistic_examples(self):
        raw = (
            "「亻」加「巨」是「佢」字，读作 qú（二声）。\n\n"
            "这是方言用字，常见于粤语，意思是「他/她/它」，例如粤语里「佢哋」就是「他们」。\n\n"
            "工具核对确认：该字读 qú，字库中暂未收录，但字符身份和读音都明确。"
        )

        async def run(response):
            calls = []
            store = harness.MemoryConversationStateStore()
            owner = harness.ConversationAddress.private("qq", "s45-character-example")

            class Skills(harness._FakeSkillsManager):
                def has_tools(self):
                    return True

                def get_tools(self):
                    return [{"type": "function", "function": {"name": name, "description": "Offline fixture",
                        "parameters": {"type": "object", "properties": {"word": {"type": "string"}}, "required": ["word"]}}}
                        for name in ("keytao_encode", "keytao_lookup_by_word", "keytao_prepare_reviewed_add")]

            async def dispatch(name, **kwargs):
                calls.append((name, kwargs))
                self.assertEqual(kwargs, {"word": "佢"})
                if name == "keytao_encode":
                    return {"success": True, "word": "佢", "phrasePinyins": ["qú"],
                            "chars": [{"char": "佢", "pinyin": "qú", "pinyins": ["qú"],
                                       "phoneticCode": "ql", "c1": None, "c2": None, "shapeCode": None}]}
                if name == "keytao_lookup_by_word":
                    return {"success": True, "word": "佢", "phrases": []}
                raise AssertionError(name)

            def tool_call(name):
                return SimpleNamespace(id=name, type="function", function=SimpleNamespace(
                    name=name, arguments=json.dumps({"word": "佢"})))

            client = harness._FakeClient([
                harness._FakeAIResponse("tool_calls", tool_calls=[tool_call("keytao_encode")]),
                harness._FakeAIResponse("tool_calls", tool_calls=[tool_call("keytao_lookup_by_word")]),
                harness._FakeAIResponse("stop", response),
            ])
            orchestrator = harness.AgentOrchestrator(
                client_factory=lambda: client,
                runtime=harness.AgentRuntimeConfig(model="fake-model", max_tokens=1000, temperature=0.0, timeout=10.0),
                skills_manager=Skills(), tool_executor=harness.ToolExecutor(
                    lambda name: lambda **kwargs: dispatch(name, **kwargs), frozenset()),
                state_store=store, bind_help_text="bind help", system_prompt_core="system",
            )
            question = "单人旁加个巨字是什么字"
            reply = await orchestrator.run(question, harness.AgentRequestContext(
                platform="qq", user_id=owner.actor_id, mutations_allowed=False))
            token = chat._current_turn_message.set(question)
            try:
                with patch.object(chat, "conversation_state_store", store):
                    delivered = chat._prepare_user_facing_reply(reply, SimpleNamespace(conversation_address=owner, platform="qq"))
            finally:
                chat._current_turn_message.reset(token)
            self.assertIn("「佢」字，读作 qú", delivered)
            self.assertIn("字符身份和读音都明确", delivered)
            self.assertNotIn("没有可确认的计划", delivered)
            self.assertFalse(advertised_command_suggestions(delivered))
            self.assertFalse(chat.advertised_reply_contract(delivered).requires_live_state)
            self.assertEqual([name for name, _args in calls], ["keytao_encode", "keytao_lookup_by_word"])
            self.assertIsNone(store.get_record(owner))
            self.assertEqual({tool["function"]["name"] for tool in client.completions.calls[0]["tools"]},
                             {"keytao_encode", "keytao_lookup_by_word"})

        asyncio.run(run(raw))
        asyncio.run(run(raw.replace(
            "这是方言用字，常见于粤语，意思是「他/她/它」，例如粤语里「佢哋」就是「他们」。",
            "例如「把 奈飞 迁到 nhfw」，请回复「确认」执行。",
        )))

    def test_unbound_sentence_sanitation_rechecks_complete_remaining_contract(self):
        from keytao_bot.harness.orchestrator import _strip_unbound_command_suggestions

        first, last = "「佢」读作 qú。", "字符身份和读音都明确。"
        for advertisement in (
            "例如「把 奈飞 迁到 nhfw」。",
            "例如：\n执行：把 奈飞 迁到 nhfw。",
            "例如：\n任意新动词 奈飞 nhfw。",
            "请回复「确认」或发送「执行」。",
            "请回复「执行：把 奈飞 迁到 nhfw。再提交」。",
        ):
            with self.subTest(advertisement=advertisement):
                raw = first + "\n" + advertisement + "\n" + last
                sanitized = _strip_unbound_command_suggestions(raw, advertised_command_suggestions(raw))
                self.assertIn(first, sanitized)
                self.assertIn(last, sanitized)
                self.assertFalse(chat.advertised_reply_contract(sanitized).requires_live_state)
                self.assertNotIn("奈飞", sanitized)
        raw = first + "请回复「查看草稿」。请回复「确认」。" + last
        sanitized = _strip_unbound_command_suggestions(raw, advertised_command_suggestions(raw))
        self.assertEqual(advertised_command_suggestions(sanitized), ("查看草稿",))
        self.assertIsNotNone(chat._chat_routing.parse_draft_view_command("查看草稿"))
        self.assertFalse(chat.advertised_reply_contract(sanitized).requires_live_state)
        self.assertEqual(_strip_unbound_command_suggestions("请回复「确认」。", ("确认",)), "")

        async def reject_remaining_stateful_copy():
            for response in ("请回复「确认」。", "这些词加入草稿。请回复「确认」。"):
                client = harness._FakeClient([harness._FakeAIResponse("stop", response)])
                store = harness.MemoryConversationStateStore()
                orchestrator = harness.AgentOrchestrator(
                    client_factory=lambda: client,
                    runtime=harness.AgentRuntimeConfig(model="fake-model", max_tokens=1000, temperature=0.0, timeout=10.0),
                    skills_manager=harness._FakeSkillsManager(),
                    tool_executor=harness.ToolExecutor(lambda _name: None, frozenset()),
                    state_store=store, bind_help_text="bind help", system_prompt_core="system",
                )
                result = await orchestrator.run("说明上次查询结果", harness.AgentRequestContext(
                    platform="qq", user_id="s45-sanitize", mutations_allowed=False))
                self.assertTrue(result)
                self.assertNotIn("调整", result)
                self.assertFalse(chat.advertised_reply_contract(result).requires_live_state)
                self.assertNotIn("这些词加入草稿", result)
                self.assertIsNone(store.get_record(harness.ConversationAddress.private("qq", "s45-sanitize")))

        asyncio.run(reject_remaining_stateful_copy())

    def test_sender_identity_explanation_preserves_exact_s27_provider_answer(self):
        question = "你是否会先确认对方有没有绑定账号？"
        raw = (
            "会的。所有草稿、提交这类写操作，执行器都会先校验当前发送者有没有绑定键道平台账号——"
            "你现在的会话身份（S27-unbound）还没有绑定，所以刚才「加入并提交」被拦下，而不是真的写入了草稿。\n\n"
            "绑定是一次性的身份关联：登录 https://keytao.vercel.app ，到 https://keytao.vercel.app/profile "
            "的【机器人账号绑定】里生成绑定码，再用 /bind 命令把它发给我就行。绑定成功后，你加的词、"
            "提交的批次都会归属到你自己的账号下。\n\n查词、问规则这类只读操作不受绑定限制，随时可以问。"
        )

        async def run(raw, expected_delivery):
            store = harness.MemoryConversationStateStore()
            owner = harness.ConversationAddress.private("qq", "s27-meta")
            client = harness._FakeClient([harness._FakeAIResponse("stop", raw)])
            executor = harness.ToolExecutor(lambda _name: None, frozenset())
            orchestrator = harness.AgentOrchestrator(
                client_factory=lambda: client,
                runtime=harness.AgentRuntimeConfig(model="fake-model", max_tokens=1000, temperature=0.0, timeout=10.0),
                skills_manager=harness._FakeSkillsManager(), tool_executor=executor,
                state_store=store, bind_help_text="bind help", system_prompt_core="system",
            )
            with patch.object(executor, "call", AsyncMock(side_effect=AssertionError("unexpected tool call"))) as execute:
                reply = await orchestrator.run(
                    question, harness.AgentRequestContext(platform="qq", user_id=owner.actor_id, mutations_allowed=False),
                )
            self.assertEqual(reply, raw, advertised_command_suggestions(raw))
            token = chat._current_turn_message.set(question)
            try:
                with patch.object(chat, "conversation_state_store", store):
                    delivered = chat._prepare_user_facing_reply(reply, SimpleNamespace(conversation_address=owner, platform="qq"))
            finally:
                chat._current_turn_message.reset(token)
            self.assertEqual(delivered, expected_delivery)
            self.assertNotIn("没有可确认的计划", delivered)
            self.assertFalse(advertised_command_suggestions(raw))
            self.assertEqual(execute.await_count, 0)
            self.assertEqual(len(client.completions.calls), 1)
            self.assertIsNone(store.get_record(owner))
            for introduction in ("例如", "比如", "回复", "发送", "请回复", "可发送"):
                for separator in ("\n", "，"):
                    advertised = raw.rstrip("。") + f"{separator}{introduction}「任意新动词 奈飞 nhfw」"
                    self.assertEqual(advertised_command_suggestions(advertised), ("任意新动词 奈飞 nhfw",))
                    with patch.object(chat, "conversation_state_store", store):
                        rejected = chat._enforce_advertised_reply_contract(advertised, owner)
                    self.assertNotEqual(rejected, advertised)
                    self.assertFalse(advertised_command_suggestions(rejected))
            self.assertEqual(advertised_command_suggestions("当前发送者请发送「确认」"), ("确认",))
            self.assertEqual(advertised_command_suggestions(raw + "\n执行：任意新动词 奈飞 nhfw"), ("执行：任意新动词 奈飞 nhfw",))
            for instruction in (
                "发送了「确认」就会执行。",
                "回复提到的「确认」即可。",
                "例如，你刚才发送了「确认」。",
                "例如，上一条回复提到了「确认」。",
            ):
                self.assertEqual(advertised_command_suggestions(instruction), ("确认",))

        asyncio.run(run(raw, harness.AgentOrchestrator._binding_meta_question_reply(question)))
        for explanation in (
            "你刚才发送了「加入并提交」，请求已经被拦下。",
            "上一条回复提到了「加入并提交」被拦下的原因。",
        ):
            with self.subTest(explanation=explanation):
                asyncio.run(run(explanation, explanation))

    def test_pending_word_query_facts_survive_delivery_without_unbound_actions(self):
        from keytao_bot.utils.pending_confirmation import pending_word_reminder_lines

        async def run():
            commands = harness.chat_commands_module
            store = harness.MemoryConversationStateStore()
            owner = harness.ConversationAddress.private("qq", "s58-pending-query")
            item = {"source": "submitted", "batchId": "pending-batch", "batchUrl": "https://keytao.test/batch/pending-batch",
                    "batchStatus": "Submitted", "itemStatus": "Pending", "action": "Create",
                    "word": "开团", "code": "khtt", "type": "Phrase"}
            calls = []

            async def tool(name, args, *_args, **_kwargs):
                calls.append(name)
                if name == "keytao_lookup_by_word":
                    return json.dumps({"success": True, "phrases": []})
                if name == "keytao_pending_items_by_words":
                    return json.dumps({"success": True, "complete": True, "bound": True, "items": [item]})
                if name == "keytao_prepare_reviewed_add":
                    return json.dumps({"success": True, "word": "开团", "recommendedCode": "khtt",
                        "preSubmitAudit": {"success": True, "verdict": "pass", "autoApprove": True},
                        "pronunciations": [{"pinyin": "kai tuan", "recommendedCode": "khtt", "sources": [],
                            "candidateStatuses": [{"code": code, "occupied": False, "label": "空位"}
                                                  for code in ("khtt", "khtto", "khttoo")]}]})
                raise AssertionError(name)

            with patch.object(chat, "conversation_state_store", store), patch.object(chat, "call_tool_function", side_effect=tool), patch.object(
                chat, "_classify_simple_word_query_intent", AsyncMock(return_value=harness.SimpleWordQueryIntent(True, ("开团",), "word_lookup", 1.0)),
            ):
                for source, status in (("submitted", "Submitted"), ("draft", "Draft")):
                    item.update(source=source, batchStatus=status)
                    raw = await commands._try_handle_simple_single_word_query("开团", "qq", owner.actor_id, owner)
                    expected = chat.render_platform_public_links(pending_word_reminder_lines([item])[0], "qq")
                    delivered = chat._prepare_user_facing_reply(raw, SimpleNamespace(conversation_address=owner, platform="qq"))
                    self.assertTrue(delivered.startswith(expected), delivered)
                    self.assertIn("其他编码", delivered)
                    self.assertNotIn("<编码>", delivered)
                    if source == "submitted":
                        self.assertIn("撤回", delivered)
                    self.assertFalse(advertised_command_suggestions(delivered))
                    self.assertIsNone(store.get_record(owner))
                self.assertEqual(calls, ["keytao_lookup_by_word", "keytao_pending_items_by_words"] * 2)
                item.update(source="submitted", batchStatus="Submitted")
                with patch.object(commands.user_resolver, "_find_user_payload", AsyncMock(return_value={"found": True, "user": {"id": 1}})):
                    raw = await commands._try_handle_simple_single_word_query("加词 开团", "qq", owner.actor_id, owner)
                delivered = chat._prepare_user_facing_reply(raw, SimpleNamespace(conversation_address=owner, platform="qq"))
                expected = chat.render_platform_public_links(pending_word_reminder_lines([item])[0], "qq")
                self.assertTrue(delivered.startswith(expected), delivered)
                self.assertTrue(chat._advertised_reply_matches_live_record(delivered, store.get_record(owner)))
                self.assertIn("khtto", delivered)
        asyncio.run(run())

    def test_selected_reading_and_manual_review_survive_producer_and_canonical_redraw(self):
        async def run(reason):
            commands = harness.chat_commands_module
            store = harness.MemoryConversationStateStore()
            owner = harness.ConversationAddress.private("qq", "s39-selected-reading")
            statuses = [
                {"code": "jjqt", "occupied": True, "label": "已有「除权」", "words": ["除权"],
                 "phrases": [{"word": "除权", "code": "jjqt", "weight": 100, "type": "Phrase"}]},
                *({"code": code, "occupied": False, "label": "空位", "words": [], "phrases": []}
                  for code in ("jjqta", "jjqtai")),
            ]
            review = {
                "success": True, "word": "出圈", "existing": [], "recommendedCode": "jjqta",
                "needsManualReview": True, "manualReviewReason": reason,
                "preSubmitAudit": {"success": True, "verdict": "pass", "autoApprove": True},
                "pronunciations": [{"pinyin": "chū quān", "normalized": ["chu", "quan"],
                    "codes": ["jjqt", "jjqta", "jjqtai"], "sources": [{"source": "开放词典数据（CC-CEDICT）"}],
                    "semanticPronunciation": True, "requiresManualReview": True,
                    "sourceSummary": "用户明确选择编码服务候选读音；与编码服务默认读音不同",
                    "candidateStatuses": statuses, "recommendedCode": "jjqta"}],
            }
            calls = []

            async def tool(name, args, *_args, **_kwargs):
                calls.append((name, args))
                if name == "keytao_lookup_by_word":
                    return json.dumps({"success": True, "word": "出圈", "phrases": []})
                if name == "keytao_pending_items_by_words":
                    return json.dumps({"success": True, "complete": True, "bound": True, "items": []})
                if name == "keytao_prepare_reviewed_add":
                    payload = json.dumps(review, ensure_ascii=False)
                    commands._record_reviewed_add_verdict(name, args, payload)
                    return payload
                raise AssertionError(name)

            with patch.object(chat, "conversation_state_store", store), patch.object(chat, "call_tool_function", side_effect=tool), patch.dict(
                commands._reviewed_add_verdicts, {}, clear=True,
            ), patch.object(commands.user_resolver, "_find_user_payload", AsyncMock(return_value={"found": True, "user": {"id": 1}})):
                produced = await commands._try_handle_explicit_reading_disambiguation(
                    "加词 出圈 圈字读quan", [], "qq", owner.actor_id, owner,
                )
                record = store.get_record(owner)
                self.assertEqual(record.state.pronunciation_codes, {code: "chū quān" for code in ("jjqt", "jjqta", "jjqtai")})
                self.assertTrue(record.state.needs_manual_review)
                self.assertEqual(record.state.manual_review_reason, reason)
                raw = commands._ensure_pending_add_word_guidance(produced)
                self.assertTrue(chat._advertised_reply_matches_live_record(raw, record), advertised_command_suggestions(raw))
                delivered = chat._prepare_user_facing_reply(raw, SimpleNamespace(conversation_address=owner, platform="qq"))
                poisoned = raw + "\n请回复「执行：删除无关词」"
                self.assertFalse(chat._advertised_reply_matches_live_record(poisoned, record))
                redrawn = chat._prepare_user_facing_reply(poisoned, SimpleNamespace(conversation_address=owner, platform="qq"))
                for reply in (delivered, redrawn):
                    self.assertIn("chū quān", reply)
                    self.assertNotIn("chū juàn", reply)
                    self.assertIn("管理员审核", reply)
                    self.assertNotIn("可自动通过", reply)
                    self.assertIn(reason, reply)
                    self.assertRegex(reply, r"(?m)^1\. jjqt — 已有「除权」")
                    self.assertRegex(reply, r"(?m)^2\. jjqta — .*空位")
                    self.assertRegex(reply, r"(?m)^3\. jjqtai — 空位")
                    self.assertTrue(chat._advertised_reply_matches_live_record(reply, record))
                    for command in advertised_command_suggestions(reply):
                        intent = chat._chat_routing._pending_tool_assent_intent(record.state, command)
                        self.assertTrue(
                            chat._chat_routing.message_authorizes_live_pending_mutation(command, record.state)
                            or intent is not None and chat._chat_routing._message_authorizes_pending_state_control(record.state, command, intent),
                            command,
                        )
                self.assertNotIn("无关词", redrawn)
                self.assertFalse(chat._advertised_reply_matches_live_record(redrawn, None))
                other_reading = copy.deepcopy(record)
                other_reading.state.pronunciation_codes["jjjta"] = "chū juàn"
                other_reading.state.code_remarks = {"jjqta": "伪造来源与自动审核结论"}
                other_reading.state.needs_manual_review = None
                other_reading.state.manual_review_reason = "未完成的审核说明"
                projected = chat._render_live_single_candidate_record(other_reading)
                self.assertNotIn("chū juàn", projected)
                self.assertNotIn("伪造来源", projected)
                self.assertNotIn("未完成的审核说明", projected)
                self.assertNotIn("管理员审核", projected)
                claimed = copy.deepcopy(record)
                claimed.execution_id = "claimed"
                self.assertEqual(chat._render_live_single_candidate_record(claimed), "")
            self.assertEqual([name for name, _args in calls], ["keytao_lookup_by_word", "keytao_pending_items_by_words", "keytao_prepare_reviewed_add"])
            self.assertEqual(calls[-1][1]["requested_reading"], "圈=quan")

        for reason in ("用户指定读音 chu quan 与编码服务默认读音不同", "语境读音仍需人工复核"):
            with self.subTest(reason=reason):
                asyncio.run(run(reason))

    def test_pending_duplicate_real_producer_keeps_fact_and_one_bound_confirmation(self):
        from keytao_bot.utils.pending_confirmation import pending_confirmation_copy, pending_word_reminder_lines

        async def run():
            commands = harness.chat_commands_module
            store = harness.MemoryConversationStateStore()
            owner = harness.ConversationAddress.private("qq", "s58-pending-duplicate")
            item = {"source": "submitted", "batchId": "pending-batch", "batchUrl": "https://keytao.test/batch/pending-batch",
                    "batchStatus": "Submitted", "itemStatus": "Pending", "action": "Create",
                    "word": "开团", "code": "khtt", "type": "Phrase"}
            data = {"success": False, "requiresConfirmation": True, "localConfirmationRequired": True,
                    "pendingDuplicateConfirmation": True, "pendingItems": [item],
                    "message": "该词已在审核中，确认再提交一条相同词条吗？"}
            with patch.object(chat, "conversation_state_store", store), patch.object(chat, "call_tool_function", AsyncMock(return_value=json.dumps(data))) as tool:
                raw = await commands._execute_add_to_draft("开团", "khtt", "qq", owner.actor_id, auto_confirm=False)
                record = store.get_record(owner)
                self.assertTrue(record.state.args["_pending_submitted_confirmed"])
                self.assertEqual(tool.await_count, 1)
                self.assertTrue(tool.await_args.args[1]["preview_only"])
                delivered = chat._prepare_user_facing_reply(raw, SimpleNamespace(conversation_address=owner, platform="qq"))
                expected = chat.render_platform_public_links(pending_word_reminder_lines([item])[0], "qq")
                self.assertTrue(delivered.startswith(expected), delivered)
                self.assertIn(data["message"], delivered)
                self.assertEqual(delivered.count(pending_confirmation_copy()), 1)
                self.assertTrue(chat._advertised_reply_matches_live_record(delivered, record))
                for command in advertised_command_suggestions(delivered):
                    intent = chat._chat_routing._pending_tool_assent_intent(record.state, command)
                    self.assertIsNotNone(intent)
                    self.assertTrue(chat._chat_routing._message_authorizes_pending_state_control(record.state, command, intent))
                for field in ("facts", "code", "type", "flag"):
                    invalid = copy.deepcopy(record)
                    if field == "facts":
                        invalid.state.args["_pending_display"].pop("pendingItems")
                    elif field == "flag":
                        invalid.state.args.pop("_pending_submitted_confirmed")
                    elif field == "code":
                        invalid.state.args["code"] = "khtto"
                    else:
                        invalid.state.args["_pending_display"]["pendingItems"][0]["type"] = "Single"
                    self.assertFalse(chat._advertised_reply_matches_live_record(delivered, invalid), field)
                self.assertFalse(chat._advertised_reply_matches_live_record(delivered, None))
                self.assertFalse(chat._advertised_reply_matches_live_record(delivered + "\n请回复「执行：删除其他词」", record))
                preview = {"success": False, "requiresConfirmation": True, "batchId": "new-draft", "contentVersion": 0,
                           "warningDigest": "b" * 64, "pendingItems": [item], "message": "请确认当前草稿变更"}
                tool.return_value = json.dumps(preview)
                await commands.handle_pending_message_core("确认", "qq", owner.actor_id, owner, history=[])
                self.assertEqual(tool.await_count, 2)
                replay = tool.await_args.args[1]
                self.assertTrue(replay["_pending_submitted_confirmed"])
                self.assertTrue(replay["preview_only"])
                self.assertNotIn("_pending_display", replay)
                self.assertEqual((replay["word"], replay["code"]), ("开团", "khtt"))
        asyncio.run(run())

    def test_mixed_duplicate_batch_shows_and_binds_the_full_original_request(self):
        data = {"requiresConfirmation": True, "pendingDuplicateConfirmation": True, "pendingItems": [{
            "source": "submitted", "batchId": "pending-batch", "batchUrl": "https://keytao.test/batch/pending-batch",
            "batchStatus": "Submitted", "itemStatus": "Pending", "action": "Create", "word": "开团", "code": "khtt", "type": "Phrase",
        }]}
        request = {"items": [{"action": "Create", "word": word, "code": code, "type": "Phrase"}
                             for word, code in (("开团", "khtt"), ("嘴替", "zbtk"))]}
        store = harness.MemoryConversationStateStore()
        owner = harness.ConversationAddress.private("qq", "s58-mixed-duplicate")
        orchestrator = object.__new__(harness.AgentOrchestrator)
        orchestrator._state_store = store
        self.assertTrue(orchestrator._save_pending_tool_confirm(owner, owner.space_key, "", "keytao_batch_add_to_draft", request, data))
        record = store.get_record(owner)
        raw = chat._chat_render._format_pending_duplicate_confirmation(record.state)
        with patch.object(chat, "conversation_state_store", store):
            delivered = chat._prepare_user_facing_reply(raw, SimpleNamespace(conversation_address=owner, platform="qq"))
        self.assertIn("「开团」→ khtt", delivered)
        self.assertIn("「嘴替」→ zbtk", delivered)
        self.assertTrue(chat._advertised_reply_matches_live_record(delivered, record))
        changed = copy.deepcopy(record)
        changed.state.args["items"].append({"action": "Delete", "word": "无关词", "code": "wgcu", "type": "Phrase"})
        self.assertFalse(chat._advertised_reply_matches_live_record(delivered, changed))
        self.assertFalse(chat._advertised_reply_matches_live_record(delivered.replace("- 「嘴替」→ zbtk\n", ""), record))
        from keytao_bot.harness.state import pending_duplicate_confirmation_is_complete

        for replacement, valid in (
            ({"old_word": "原词"}, True),
            ({"oldWord": "原词"}, True),
            ({"old_word": "原词", "oldWord": "原词"}, True),
            ({}, False),
            ({"old_word": ""}, False),
            ({"old_word": "原词", "oldWord": "无关词"}, False),
            ({"oldWord": ["原词"]}, False),
        ):
            mixed = copy.deepcopy(request)
            mixed["items"][1].update(action="Change", **replacement)
            orchestrator._save_pending_tool_confirm(owner, owner.space_key, "", "keytao_batch_add_to_draft", mixed, data)
            mixed_record = store.get_record(owner)
            self.assertEqual(pending_duplicate_confirmation_is_complete(mixed_record.state), valid, replacement)
            rendered = chat._chat_render._format_pending_duplicate_confirmation(mixed_record.state)
            if valid:
                self.assertIn("替换「原词」→「嘴替」@ zbtk", rendered)
                self.assertTrue(chat._advertised_reply_matches_live_record(rendered, mixed_record))
            else:
                self.assertFalse(rendered, replacement)

    def test_binding_notice_survives_real_discovery_redraw_without_becoming_command_authority(self):
        from keytao_bot.utils.pending_confirmation import UNBOUND_BINDING_PRECHECK_NOTICE, _BIND_HELP_TEXT

        async def run():
            commands = harness.chat_commands_module
            resolver = commands.user_resolver
            word = "来都来了"
            store = harness.MemoryConversationStateStore()
            owner = harness.ConversationAddress.private("qq", "s58-binding-notice")

            async def tool(name, args, *_args, **_kwargs):
                if name == "keytao_lookup_by_word":
                    return json.dumps({"success": True, "phrases": []})
                if name == "keytao_pending_items_by_words":
                    return json.dumps({"success": True, "complete": True, "items": []})
                if name == "keytao_prepare_reviewed_add":
                    return json.dumps({"success": True, "word": word, "recommendedCode": "ldll",
                        "preSubmitAudit": {"success": True, "verdict": "pass", "autoApprove": True},
                        "pronunciations": [{"pinyin": "lai dou lai le", "recommendedCode": "ldll", "sources": [],
                            "candidateStatuses": [{"code": code, "occupied": False, "label": "空位"}
                                                  for code in ("ldll", "ldllv", "ldllvu")]}]})
                raise AssertionError(name)

            with patch.object(chat, "conversation_state_store", store), patch.object(chat, "call_tool_function", side_effect=tool), patch.object(
                chat, "_classify_simple_word_query_intent", AsyncMock(return_value=harness.SimpleWordQueryIntent(True, (word,), "word_lookup", 1.0)),
            ):
                context = SimpleNamespace(conversation_address=owner, platform="qq")
                notice = chat.render_platform_public_links(UNBOUND_BINDING_PRECHECK_NOTICE, "qq")
                for payload, expected_count in (({"found": False}, 1), (None, 0), ({"found": True, "user": {"id": 1}}, 0)):
                    with patch.object(resolver, "_find_user_payload", AsyncMock(return_value=payload)):
                        raw = await commands._try_handle_simple_single_word_query(word, "qq", owner.actor_id, owner)
                    self.assertIsNotNone(raw)
                    poisoned = str(raw) + "\n例如：执行：把 奈飞 调到 zzzzz"
                    delivered = chat._prepare_user_facing_reply(poisoned, context)
                    self.assertEqual(delivered.count(notice), expected_count)
                    self.assertNotIn("zzzzz", delivered)
                    self.assertNotIn("/bind", advertised_command_suggestions(delivered))
                    self.assertTrue(chat._advertised_reply_matches_live_record(delivered, store.get_record(owner)))
                    self.assertEqual(chat._prepare_user_facing_reply(delivered, context), delivered)
                with patch.object(resolver, "_find_user_payload", AsyncMock(return_value={"found": False})):
                    await resolver.resolve_actor_binding("qq", owner.actor_id)
                foreign = harness.ConversationAddress.private("qq", "foreign-binding-actor")
                store.set(foreign, copy.deepcopy(store.get_record(owner).state))
                foreign_context = SimpleNamespace(conversation_address=foreign, platform="qq")
                foreign_reply = chat._prepare_user_facing_reply(UNBOUND_BINDING_PRECHECK_NOTICE + "\n" + str(raw), foreign_context)
                self.assertNotIn(notice, foreign_reply)
                self.assertTrue(chat._advertised_reply_matches_live_record(foreign_reply, store.get_record(foreign)))
                meta = "会先检查账号绑定情况。"
                self.assertEqual(chat._prepare_user_facing_reply(meta, context), meta)
                store.delete(owner)
                rejected = chat._prepare_user_facing_reply(UNBOUND_BINDING_PRECHECK_NOTICE + "\n" + poisoned, context)
                self.assertNotIn(notice, rejected)
                self.assertFalse(advertised_command_suggestions(rejected))
                for help_text in (_BIND_HELP_TEXT, resolver.get_not_bound_message()):
                    help_reply = chat._prepare_user_facing_reply(help_text, context)
                    self.assertIn("/bind", help_reply)
                    self.assertIn("/profile", help_reply)
                    self.assertFalse(advertised_command_suggestions(help_reply))
        asyncio.run(run())

    def test_account_notice_facts_are_reset_and_isolated_by_actor_and_async_task(self):
        async def run():
            resolver = harness.chat_commands_module.user_resolver
            with patch.object(resolver, "_find_user_payload", AsyncMock(return_value={"found": False})):
                await resolver.resolve_actor_binding("qq", "owner")
            self.assertIs(resolver.resolved_binding_for_notice("qq", "owner"), False)
            self.assertIsNone(resolver.resolved_binding_for_notice("qq", "other"))
            self.assertIsNone(resolver.resolved_binding_for_notice("telegram", "owner"))
            child_ready = asyncio.Event()
            parent_checked = asyncio.Event()

            async def child():
                with patch.object(resolver, "_find_user_payload", AsyncMock(return_value={"found": True, "user": {"id": 1}})):
                    await resolver.resolve_actor_binding("qq", "owner")
                child_ready.set()
                await parent_checked.wait()
                self.assertIs(resolver.resolved_binding_for_notice("qq", "owner"), True)

            task = asyncio.create_task(child())
            await child_ready.wait()
            self.assertIs(resolver.resolved_binding_for_notice("qq", "owner"), False)
            parent_checked.set()
            await task
            with patch.object(resolver, "_find_user_payload", AsyncMock(return_value=None)):
                await resolver.resolve_actor_binding("qq", "owner")
            self.assertIsNone(resolver.resolved_binding_for_notice("qq", "owner"))
            with patch.object(resolver, "_find_user_payload", AsyncMock(return_value={"found": False})):
                await resolver.resolve_actor_binding("qq", "owner")
            with patch.object(resolver, "_find_user_payload", AsyncMock(side_effect=OSError("lookup unavailable"))):
                with self.assertRaises(OSError):
                    await resolver.resolve_actor_binding("qq", "owner")
            self.assertIsNone(resolver.resolved_binding_for_notice("qq", "owner"))
            with patch.object(resolver, "_find_user_payload", AsyncMock(return_value={"found": False})):
                await resolver.resolve_actor_binding("qq", "owner")

            async def first_stage(_ctx):
                self.assertIsNone(resolver.resolved_binding_for_notice("qq", "owner"))
                return True

            with patch.object(chat, "STAGES", (first_stage,)):
                await chat._handle_ai_chat_serialized(None, None, "qq", "owner")
        asyncio.run(run())

    def test_reviewed_candidate_precedence_survives_real_delivery_and_confirms_exact_items(self):
        from keytao_bot.utils.pending_confirmation import pending_confirmation_copy, pending_batch_confirmation_copy
        from test_s54_multiword import MultiwordRouteTests

        async def run():
            pairs = (("显眼包", "xyb"), ("嘴替", "zbtk"))
            store = harness.MemoryConversationStateStore()
            owner = harness.ConversationAddress.private("qq", "s58-precedence")
            calls = [SimpleNamespace(id=f"review-{index}", type="function", function=SimpleNamespace(
                name="keytao_prepare_reviewed_add", arguments=json.dumps({"word": word}),
            )) for index, (word, _code) in enumerate(pairs)]
            client = harness._FakeClient([
                harness._FakeAIResponse("tool_calls", "", calls),
                harness._FakeAIResponse("stop", "\n".join(f"- 「{word}」→ {code}" for word, code in pairs) + "\n" + pending_batch_confirmation_copy()),
            ])

            async def review(word, **_kwargs):
                code = dict(pairs)[word]
                return {"success": True, "word": word, "type": "Phrase", "recommendedCode": code,
                        "candidateCodes": [code, code + "a"], "needsManualReview": False,
                        "candidateStatuses": [{"code": value, "occupied": False} for value in (code, code + "a")],
                        "preSubmitAudit": {"success": True, "verdict": "pass", "autoApprove": True,
                                           "needsManualReview": False, "approvedItems": [word]}}

            skills = harness._FakeToolSkillsManager()
            skills.get_tools = lambda: [{"type": "function", "function": {
                "name": "keytao_prepare_reviewed_add", "description": "Review a word",
                "parameters": {"type": "object", "properties": {"word": {"type": "string"}}, "required": ["word"]},
            }}]
            orchestrator = harness.AgentOrchestrator(
                client_factory=lambda: client,
                runtime=harness.AgentRuntimeConfig(model="fake-model", max_tokens=1000, temperature=0.0, timeout=10.0),
                skills_manager=skills,
                tool_executor=harness.ToolExecutor(lambda name: review if name == "keytao_prepare_reviewed_add" else None,
                                                  frozenset({"keytao_prepare_reviewed_add"})),
                state_store=store, bind_help_text="bind help", system_prompt_core="system",
            )
            initial = await orchestrator.run("加词 显眼包 嘴替", harness.AgentRequestContext(
                platform="qq", user_id="s58-precedence", mutations_allowed=True,
            ))
            modern_reply, modern_store, modern_owner = await MultiwordRouteTests().discover()
            for live_store, live_owner in ((store, owner), (modern_store, modern_owner)):
                record = live_store.get_record(live_owner)
                self.assertIsNotNone(record, initial)
                expected = copy.deepcopy(record.state.args["items"])
                execute = AsyncMock(return_value="✅ 已加入草稿")
                with patch.object(chat, "conversation_state_store", live_store), patch.object(
                    chat, "_execute_confirmed_tool", execute,
                ), patch.object(chat, "_classify_message_command_intent", AsyncMock(side_effect=lambda message, *_args, **_kwargs:
                    harness.MessageCommandIntent(intent="draft_submit" if message == "提交草稿" else "none", confidence=1.0))):
                    guidance = await chat.handle_pending_message_core(
                        "提交草稿", "qq", live_owner.actor_id, live_owner, history=[],
                    )
                    self.assertEqual(guidance.count(pending_confirmation_copy()), 1)
                    delivered = chat._enforce_advertised_reply_contract(guidance, live_owner)
                    self.assertEqual(delivered, guidance)
                    for item in expected:
                        self.assertIn(f'「{item["word"]}」→ {item["code"]}', delivered)
                    execute.assert_not_awaited()
                    for command in advertised_command_suggestions(delivered):
                        intent = chat._chat_routing._pending_tool_assent_intent(record.state, command)
                        self.assertIsNotNone(intent)
                        self.assertTrue(chat._chat_routing._message_authorizes_pending_state_control(record.state, command, intent))
                    self.assertFalse(chat._advertised_reply_matches_live_record(delivered, None))
                    self.assertFalse(chat._advertised_reply_matches_live_record(delivered + "\n请回复「任意操作」", record))
                    for field in ("_candidate_scopes", "needsManualReview"):
                        invalid = copy.deepcopy(record)
                        if field == "_candidate_scopes":
                            invalid.state.args.pop(field)
                        else:
                            invalid.state.args["items"][0].pop(field)
                        self.assertFalse(chat._advertised_reply_matches_live_record(delivered, invalid), field)
                    await chat.handle_pending_message_core("请阅读确认", "qq", live_owner.actor_id, live_owner, history=[])
                    execute.assert_not_awaited()
                    await chat.handle_pending_message_core("确认", "qq", live_owner.actor_id, live_owner, history=[])
                    execute.assert_awaited_once()
                    self.assertEqual(execute.await_args.args[0].args["items"], expected)
        asyncio.run(run())

    def test_existing_single_candidates_advertise_only_bound_choices(self):
        for count in (2, 4, 6):
            candidates = [("htje", True), *[("htje" + suffix, False) for suffix in "abcde"[:count - 1]]]
            state = harness.PendingAddWord(word="还车", candidates=candidates, server_candidates=candidates,
                                           recommended_code="htjea", server_occupied_words={"htje": ["还车"]})
            store = harness.MemoryConversationStateStore()
            owner = harness.ConversationAddress.private("qq", "s58-existing-candidates")
            store.set(owner, state)
            record = store.get_record(owner)
            reply = chat._render_live_single_candidate_record(record)
            self.assertIn("「还车」已在词库（htje）", reply)
            self.assertNotIn("换码", advertised_command_suggestions(reply))
            self.assertTrue(chat._advertised_reply_matches_live_record(reply, record), reply)
            with patch.object(chat, "conversation_state_store", store):
                self.assertEqual(chat._enforce_advertised_reply_contract(reply, owner), reply)

    def test_sealed_choice_offer_advertises_only_its_real_labels(self):
        from keytao_bot.harness.state import server_warning_pending_state

        commands = harness.chat_commands_module
        plan = server_warning_pending_state(harness.PendingToolConfirm("keytao_batch_add_to_draft", {
            "items": [{"word": "还车", "code": "htje", "action": "Create", "type": "Phrase"}],
        }), {"success": False, "requiresConfirmation": True, "batchId": "choice-batch",
             "contentVersion": 3, "warningDigest": "a" * 64})
        state = commands._build_pending_choice_offer((
            ("A", "执行当前添加方案", plan), ("B", "保留现状", None),
        ))
        self.assertIsNotNone(state)
        store = harness.MemoryConversationStateStore()
        owner = harness.ConversationAddress.private("qq", "s58-choice-offer")
        foreign = harness.ConversationAddress.private("qq", "other-choice-owner")
        store.set(owner, state)
        with patch.object(chat, "conversation_state_store", store), patch.object(
            chat, "call_tool_function", AsyncMock(side_effect=AssertionError("preview wrote")),
        ) as sink:
            raw = commands.render_pending_choice_offer(state)
            self.assertIn("回复 A 或 B 即可", raw)
            challenged = chat._append_pending_ticket_challenge(raw, owner)
            self.assertEqual(challenged, raw)
            self.assertEqual(advertised_command_suggestions(raw), ("A", "B"))
            self.assertEqual([commands._parse_pending_choice_label(value) for value in advertised_command_suggestions(raw)], ["A", "B"])
            self.assertEqual(chat._enforce_advertised_reply_contract(raw, owner), raw)
            self.assertFalse(chat._advertised_reply_matches_live_record(raw, None))
            self.assertFalse(advertised_command_suggestions(chat._enforce_advertised_reply_contract(raw, foreign)))
            record = store.get_record(owner)
            self.assertFalse(chat._advertised_reply_matches_live_record(raw + "\n请回复「确认」", record))
            for field in ("options", "warning_digest", "plan", "empty_plans"):
                invalid = copy.deepcopy(record)
                if field == "options":
                    invalid.state.args["options"].pop()
                elif field == "plan":
                    invalid.state.args["options"][0].pop("plan")
                elif field == "empty_plans":
                    invalid.state.args["options"][0]["plan"] = None
                else:
                    invalid.state.args["options"][0]["plan"]["args"].pop("expected_warning_digest")
                self.assertFalse(chat._advertised_reply_matches_live_record(raw, invalid), field)
            record.execution_id = "claimed"
            self.assertFalse(chat._advertised_reply_matches_live_record(raw, record))
            sink.assert_not_awaited()

    def test_real_shift_conflict_producer_seals_its_reason_for_final_choice_delivery(self):
        async def run():
            commands = harness.chat_commands_module
            store = harness.MemoryConversationStateStore()
            owner = harness.ConversationAddress.private("qq", "s58-real-choice")
            target = {"id": 19463, "action": "Create", "word": "哲思", "code": "fesko", "type": "Phrase"}
            results = [
                {"success": False, "policyBlocked": True, "requiresDraftCleanup": True,
                 "batchId": "choice-conflict-batch", "contentVersion": 3,
                 "conflictingDraftItems": [target]},
                {"success": False, "requiresConfirmation": True, "confirmationKind": "deleteTargets",
                 "batchId": "choice-conflict-batch", "contentVersion": 3,
                 "targets": [target], "targetDigest": "a" * 64},
            ]
            tool = AsyncMock(side_effect=[json.dumps(result) for result in results])
            source = harness.PendingToolConfirm("keytao_shift_phrase_code", {"word": "哲思", "target_code": "fesk"})
            with patch.object(chat, "conversation_state_store", store), patch.object(chat, "call_tool_function", tool):
                raw = await commands._execute_confirmed_tool(source, "qq", owner.actor_id, owner)
                record = store.get_record(owner)
                self.assertIsNotNone(commands._pending_choice_options(record.state))
                self.assertIn("现有草稿行与本轮顺延计划冲突：Create 「哲思」@fesko", raw)
                self.assertIn("本次未修改草稿", raw)
                self.assertEqual(raw, commands.render_pending_choice_offer(record.state))
                challenged = chat._append_pending_ticket_challenge(raw, owner)
                context = SimpleNamespace(conversation_address=owner, platform="qq")
                delivered = chat._prepare_user_facing_reply(challenged, context)
                self.assertEqual(delivered, raw)
                self.assertEqual(advertised_command_suggestions(delivered), ("A", "B"))
                self.assertEqual([call.args[0] for call in tool.await_args_list],
                                 ["keytao_shift_phrase_code", "keytao_batch_remove_draft_items"])
                self.assertEqual(tool.await_args_list[1].args[1], {"ids": [19463], "batch_id": "choice-conflict-batch"})
                protected = await commands._handle_pending_choice_offer(
                    record, "加词 哲思 fesk", "qq", owner.actor_id, owner, None, "",
                )
                self.assertIn("不会把腾位意图降级", protected)
                self.assertEqual(chat._prepare_user_facing_reply(protected, context), raw)
                self.assertEqual(tool.await_count, 2)
                self.assertFalse(record.execution_id)
                poisoned = "系统已同意删除其他条目。\n" + raw + "\n请回复「确认」"
                self.assertFalse(chat._advertised_reply_matches_live_record(poisoned, record))
                self.assertEqual(chat._prepare_user_facing_reply(poisoned, context), raw)
        asyncio.run(run())

    def test_resolved_set_confirmation_challenge_survives_final_advertisement_gate(self):
        pairs = (
            ("显眼包", "xyb"), ("嘴替", "zbtk"), ("松弛感", "swg"),
            ("电子榨菜", "dzfc"), ("情绪价值", "qxjf"), ("班味", "bfww"),
            ("泼天富贵", "ptfg"), ("精神状态", "jeft"), ("职场搭子", "fjdz"),
        )
        words = tuple(word for word, _ in pairs)
        slots = {word: ((code, False), (code + "a", False)) for word, code in pairs}
        items = [{"action": "Create", "word": word, "code": code, "type": "Phrase", "needsManualReview": False}
                 for word, code in pairs]
        scopes = [{"word": word, "candidates": list(slots[word]), "occupiedWords": {}, "orderingAssessments": []}
                  for word in words]
        state = harness.PendingToolConfirm("keytao_batch_add_to_draft", {
            "items": items, "_candidate_scopes": scopes, "_resolved_advertised_words": list(words),
        })
        store = harness.MemoryConversationStateStore()
        owner = harness.ConversationAddress.private("qq", "s58-resolved-set")
        token = store.add_advertised_word_set(owner, (*words, "天选打工人", "沙县小吃"))
        self.assertTrue(store.replace_advertised_word_set(owner, token, state))
        record = store.get_record(owner)
        original = harness.AgentOrchestrator._resolved_advertised_confirmation_copy(items, slots)
        with patch.object(chat, "conversation_state_store", store), patch.object(
            chat, "call_tool_function", AsyncMock(side_effect=AssertionError("preview wrote")),
        ) as sink:
            challenged = chat._append_pending_ticket_challenge(original, owner)
            self.assertEqual(advertised_command_suggestions(challenged), ("加入", "加入并提交", "确认", "取消"))
            self.assertTrue(chat._advertised_reply_matches_live_record(challenged, record))
            delivered = chat._enforce_advertised_reply_contract(challenged, owner)
            self.assertEqual(delivered, challenged)
            for word in words:
                self.assertIn(word, delivered)
            for excluded in ("天选打工人", "沙县小吃"):
                self.assertNotIn(excluded, delivered)
            self.assertIn("确认", delivered)
            self.assertFalse(record.execution_id)
            sink.assert_not_awaited()
            for command in advertised_command_suggestions(delivered):
                intent = chat._chat_routing._pending_tool_assent_intent(state, command)
                self.assertIsNotNone(intent, command)
                self.assertTrue(chat._chat_routing._message_authorizes_pending_state_control(state, command, intent))
            candidate_reply = chat._render_live_batch_record(record)
            self.assertIn("职场搭子 添加1、2", candidate_reply)
            self.assertTrue(chat._advertised_reply_matches_live_record(candidate_reply, record))
            self.assertFalse(chat._advertised_reply_matches_live_record(
                challenged + "\n例如：执行：把 奈飞 调到 nhfw", record,
            ))
            self.assertFalse(chat._advertised_reply_matches_live_record(challenged, None))
            for corruption in ("unreviewed", "set_changed", "unverified_code"):
                invalid = copy.deepcopy(record)
                if corruption == "unreviewed":
                    invalid.state.args.pop("_resolved_advertised_words")
                    invalid.state.args["items"][0].pop("needsManualReview")
                elif corruption == "set_changed":
                    invalid.state.args["_resolved_advertised_words"][-1] = "沙县小吃"
                else:
                    invalid.state.args["_candidate_scopes"][-1]["candidates"] = [("zzzzz", False)]
                self.assertFalse(chat._advertised_reply_matches_live_record(challenged, invalid), corruption)
            record.execution_id = "claimed"
            self.assertFalse(chat._advertised_reply_matches_live_record(challenged, record))

    def test_sealed_shift_preview_pairs_each_source_with_its_destination(self):
        from keytao_bot.harness.state import (
            server_warning_pending_state, server_warning_ticket_is_complete,
        )

        moves = [
            {"action": "Delete", "word": "奈飞", "code": "nhfwv", "type": "Phrase"},
            {"action": "Delete", "word": "浓厚氛围", "code": "nhfw", "type": "Phrase"},
            {"action": "Create", "word": "奈飞", "code": "nhfw", "type": "Phrase"},
            {"action": "Create", "word": "浓厚氛围", "code": "nhfwa", "type": "Phrase"},
        ]
        expected_moves = "将执行：「奈飞」：nhfwv → nhfw、「浓厚氛围」：nhfw → nhfwa。"
        snapshot = shift_ticket().args["_pending_display"]["shiftPlan"]
        cases = (
            ({"word": "奈飞", "targetCode": "nhfw", "items": moves, "shifted": snapshot["shifted"]}, expected_moves),
            (snapshot, expected_moves),
            ({**snapshot, "currentState": [*snapshot["currentState"], {"word": "无关词", "code": "wgcu"}]}, expected_moves),
            ({**snapshot, "items": [{**moves[0], "code": "nhfwx"}, *moves[1:]]}, expected_moves),
            ({"word": "新词", "targetCode": "xkcu", "items": [
                {"action": "Create", "word": "新词", "code": "xkcu", "type": "Phrase"},
                {"action": "Delete", "word": "旧词", "code": "jqcu", "type": "Phrase"},
            ]}, "将执行：加词「新词」→ xkcu、删除「旧词」@ jqcu。"),
            ({"word": "奈飞", "targetCode": "nhfw", "shifted": snapshot["shifted"]},
             "将执行：调码「奈飞」→ nhfw、「浓厚氛围」：nhfw → nhfwa。"),
        )
        for plan, expected in cases:
            with self.subTest(expected=expected, fields=tuple(plan)):
                state = server_warning_pending_state(
                    harness.PendingToolConfirm("keytao_shift_phrase_code", {
                        "word": plan["word"], "target_code": plan["targetCode"],
                    }),
                    {"success": False, "requiresConfirmation": True, "shiftPlan": plan,
                     "batchId": "", "contentVersion": 0, "planDigest": "a" * 64, "warningDigest": "b" * 64},
                )
                self.assertTrue(server_warning_ticket_is_complete(state))
                rendered = chat._chat_render._format_server_bound_confirmation_prompt(state)
                self.assertEqual(rendered.splitlines()[0], expected)
                self.assertNotIn("nhfwx", rendered)
                self.assertNotIn("无关词", rendered)
                commands = advertised_command_suggestions(rendered)
                self.assertEqual(commands, ("确认", "取消"))
                store = harness.MemoryConversationStateStore()
                owner = harness.ConversationAddress.private("qq", "s58-shift-directions")
                store.set(owner, state)
                self.assertTrue(chat._advertised_reply_matches_live_record(rendered, store.get_record(owner)), rendered)
                for command in commands:
                    intent = chat._chat_routing._pending_tool_assent_intent(state, command)
                    self.assertIsNotNone(intent)
                    self.assertTrue(chat._chat_routing._message_authorizes_pending_state_control(state, command, intent))

    def test_sealed_entry_tickets_show_actions_and_replacement_source(self):
        from keytao_bot.harness.state import (
            server_warning_pending_state, server_warning_ticket_is_complete,
        )

        create = {"word": "奈飞", "code": "nhfw", "action": "Create", "type": "Phrase"}
        delete = {"word": "浓厚氛围", "code": "nhfwa", "action": "Delete", "type": "Phrase"}
        change = {"word": "奈飞", "oldWord": "奈非", "code": "nhfw", "action": "Change", "type": "Phrase"}
        cases = (
            ("keytao_create_phrase", create, ("加词「奈飞」→ nhfw",)),
            ("keytao_create_phrase", delete, ("删除「浓厚氛围」@ nhfwa",)),
            ("keytao_create_phrase", {**{key: value for key, value in change.items() if key != "oldWord"}, "old_word": change["oldWord"]},
             ("替换「奈非」→「奈飞」@ nhfw",)),
            ("keytao_batch_add_to_draft", {"items": [create]}, ("批量加词：", "「奈飞」→ nhfw")),
            ("keytao_batch_add_to_draft", {"items": [delete]}, ("批量删除：", "删除「浓厚氛围」@ nhfwa")),
            ("keytao_batch_add_to_draft", {"items": [change]}, ("批量替换：", "替换「奈非」→「奈飞」@ nhfw")),
            ("keytao_batch_add_to_draft", {"items": [create, delete, {**change, "word": "奈菲", "code": "nhfwb"}]},
             ("批量修改：", "加词「奈飞」→ nhfw", "删除「浓厚氛围」@ nhfwa", "替换「奈非」→「奈菲」@ nhfwb")),
        )
        for tool_name, args, details in cases:
            with self.subTest(tool_name=tool_name, args=args):
                state = server_warning_pending_state(
                    harness.PendingToolConfirm(tool_name, args),
                    {"success": False, "requiresConfirmation": True,
                     "batchId": "s58-actions", "contentVersion": 3, "warningDigest": "b" * 64},
                )
                self.assertTrue(server_warning_ticket_is_complete(state))
                rendered = chat._chat_render._format_server_bound_confirmation_prompt(state)
                for detail in details:
                    self.assertIn(detail, rendered)
                commands = advertised_command_suggestions(rendered)
                self.assertEqual(commands, ("确认", "取消"))
                store = harness.MemoryConversationStateStore()
                owner = harness.ConversationAddress.private("qq", "s58-entry-actions")
                store.set(owner, state)
                self.assertTrue(chat._advertised_reply_matches_live_record(rendered, store.get_record(owner)), rendered)
                for command in commands:
                    intent = chat._chat_routing._pending_tool_assent_intent(state, command)
                    self.assertIsNotNone(intent, command)
                    self.assertTrue(chat._chat_routing._message_authorizes_pending_state_control(state, command, intent))

    def test_read_only_advertisement_uses_the_real_closed_dispatch_parser(self):
        async def run():
            routing = chat._chat_routing
            with patch.object(routing, "AsyncOpenAI", side_effect=AssertionError("read-only command used an intent model")):
                for command in ("查看草稿", "查询我的草稿条目", "显示当前草稿", "列出草稿内容"):
                    intent = await routing._classify_message_command_intent(command)
                    self.assertEqual(intent.intent, "draft_view")
                    reply = f"草稿内容仍需核对。可发送「{command}」。"
                    self.assertEqual(chat._enforce_advertised_reply_contract(reply, None), reply)
            for command in ("查看草稿？", "查看草稿并提交", "不要查看草稿", "他说查看草稿", "查看草稿 删除奈飞"):
                self.assertIsNone(routing.parse_draft_view_command(command))
                reply = f"可发送「{command}」。"
                self.assertNotEqual(chat._enforce_advertised_reply_contract(reply, None), reply)
        asyncio.run(run())

    def test_retired_authorization_copy_redraws_at_real_final_delivery(self):
        banned = (
            "顺延操作的词条或目标编码未精确绑定",
            "回复中的操作说法没有可验证的服务端绑定记录，已移除",
            "仍缺少明确的执行动词",
            "这条消息没有明确的执行指令",
            "系统将这条位置调整判定为「缺少明确的执行指令」",
        )
        store = harness.MemoryConversationStateStore()
        token = chat._current_turn_message.set("把 奈飞 调到 nhfw")
        try:
            with patch.object(chat, "conversation_state_store", store):
                for raw in banned:
                    with self.subTest(raw=raw):
                        self.assertTrue(chat._chat_render._reply_requires_deterministic_redraw(raw))
                        actual = chat._prepare_user_facing_reply(raw, None)
                        self.assertIn("奈飞", actual)
                        self.assertIn("nhfw", actual)
                        self.assertIn("未写入", actual)
                        self.assertNotIn("\n", actual)
                        self.assertFalse(advertised_command_suggestions(actual))
                        self.assertFalse(chat._chat_render._reply_requires_deterministic_redraw(actual))
        finally:
            chat._current_turn_message.reset(token)

    def test_incident_shapes_and_generic_advertisements_are_detected(self):
        command = "执行：把 奈飞 调到 nhfw，把 浓厚氛围 挪到 nhfwa"
        cases = [
            ("请明确回复一句执行指令（例如「确认调整 / 执行」）", ("确认调整", "执行")),
            ("例如：\n" + command, (command,)),
            (command, (command,)),
            ('请回复 "确认调整"', ("确认调整",)),
            ("可发送『任意新动词 奈飞 nhfw』", ("任意新动词 奈飞 nhfw",)),
            ("回复：\n任意新动词 奈飞 nhfw", ("任意新动词 奈飞 nhfw",)),
            ("比如：任意新动词 奈飞 nhfw", ("任意新动词 奈飞 nhfw",)),
            ("回复 确认调整", ("确认调整",)),
            ("例如：\n「确认调整」", ("确认调整",)),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(advertised_command_suggestions(raw), expected)

    def test_unbacked_incident_examples_cannot_escape_real_delivery(self):
        key = harness.ConversationAddress.private("qq", "s58-advertisement")
        raw_forms = (
            "请明确回复一句执行指令（例如「确认调整 / 执行」）",
            "例如：\n执行：把 奈飞 调到 nhfw，把 浓厚氛围 挪到 nhfwa",
            "请回复『任意新动词 奈飞 nhfw』",
        )
        with patch.object(chat, "conversation_state_store", harness.MemoryConversationStateStore()):
            for raw in raw_forms:
                with self.subTest(raw=raw):
                    actual = chat._enforce_advertised_reply_contract(raw, key)
                    self.assertNotEqual(actual, raw)
                    self.assertIn("未写入", actual)
                    self.assertFalse(advertised_command_suggestions(actual))

    def test_live_assent_aliases_round_trip_real_parser_and_binding(self):
        state = harness.PendingToolConfirm(
            "keytao_shift_phrase_code",
            {"word": "奈飞", "target_code": "nhfw", "batch_id": "",
             "expected_content_version": 0, "confirmed_plan_digest": "a" * 64},
            confirmation_source="server_warning",
        )
        store = harness.MemoryConversationStateStore()
        key = harness.ConversationAddress.private("qq", "s58-aliases")
        store.set(key, state)
        for command in ("确认调整", "确认执行", "确认", "执行", "好", "强制", "取消"):
            with self.subTest(command=command):
                intent = chat._chat_routing._pending_tool_assent_intent(state, command)
                self.assertIsNotNone(intent)
                self.assertTrue(chat._chat_routing._message_authorizes_pending_state_control(state, command, intent))
                raw = f"请回复「{command}」"
                with patch.object(chat, "conversation_state_store", store):
                    self.assertEqual(chat._enforce_advertised_reply_contract(raw, key), raw)
        state.args.pop("confirmed_plan_digest")
        with patch.object(chat, "conversation_state_store", store):
            self.assertNotEqual(chat._enforce_advertised_reply_contract("请回复「确认」", key), "请回复「确认」")

    def test_generic_model_advertisement_is_removed_in_real_orchestrator(self):
        async def run():
            raw = "例如：\n执行：把 奈飞 调到 nhfw，把 浓厚氛围 挪到 nhfwa"
            client = harness._FakeClient([harness._FakeAIResponse("stop", raw)])
            orchestrator = harness.AgentOrchestrator(
                client_factory=lambda: client,
                runtime=harness.AgentRuntimeConfig(model="fake-model", max_tokens=1000, temperature=0.0, timeout=10.0),
                skills_manager=harness._FakeSkillsManager(),
                tool_executor=harness.ToolExecutor(lambda _name: None, frozenset()),
                state_store=harness.MemoryConversationStateStore(),
                bind_help_text="bind help", system_prompt_core="system",
            )
            reply = await orchestrator.run(
                "把 奈飞 迁到 nhfw",
                harness.AgentRequestContext(platform="qq", user_id="s58-advertisement", mutations_allowed=False),
            )
            self.assertFalse(advertised_command_suggestions(reply), reply)
            self.assertNotIn("已移除", reply)
            self.assertIn("未写入", reply)
        asyncio.run(run())

    def test_validated_shift_redraw_uses_only_server_destinations(self):
        store = harness.MemoryConversationStateStore()
        owner = harness.ConversationAddress.group("qq", "s58-redraw", "owner")
        foreign = harness.ConversationAddress.group("qq", "s58-redraw", "other")
        state = shift_ticket()
        store.set(owner, state)
        raw = "例如：\n执行：把 奈飞 调到 nhfw，把 浓厚氛围 挪到 zzzzz"
        with patch.object(chat, "conversation_state_store", store):
            actual = chat._enforce_advertised_reply_contract(raw, owner)
            self.assertIn("nhfwa", actual)
            self.assertNotIn("zzzzz", actual)
            self.assertNotIn("已移除", actual)
            prose = "浓厚氛围会改成 zzzzz，请回复「确认」"
            self.assertEqual(chat._enforce_advertised_reply_contract(prose, owner), actual)
            for command in advertised_command_suggestions(actual):
                intent = chat._chat_routing._pending_tool_assent_intent(state, command)
                self.assertIsNotNone(intent, command)
                self.assertTrue(chat._chat_routing._message_authorizes_pending_state_control(state, command, intent))
            self.assertFalse(advertised_command_suggestions(chat._enforce_advertised_reply_contract(raw, foreign)))
            store.get_record(owner).execution_id = "claimed"
            self.assertFalse(advertised_command_suggestions(chat._enforce_advertised_reply_contract(raw, owner)))

    def test_real_orchestrator_redraws_model_advertisement_from_sealed_plan(self):
        async def run():
            store = harness.MemoryConversationStateStore()
            key = harness.ConversationAddress.private("qq", "s58-redraw")
            store.set(key, shift_ticket())
            raw = "请明确回复一句执行指令（例如「确认调整 / 执行」）。例如：\n执行：把 奈飞 调到 zzzzz"
            client = harness._FakeClient([harness._FakeAIResponse("stop", raw)])
            orchestrator = harness.AgentOrchestrator(
                client_factory=lambda: client,
                runtime=harness.AgentRuntimeConfig(model="fake-model", max_tokens=1000, temperature=0.0, timeout=10.0),
                skills_manager=harness._FakeSkillsManager(),
                tool_executor=harness.ToolExecutor(lambda _name: None, frozenset()),
                state_store=store, bind_help_text="bind help", system_prompt_core="system",
            )
            reply = await orchestrator.run(
                "把 奈飞 迁到 nhfw",
                harness.AgentRequestContext(platform="qq", user_id="s58-redraw", mutations_allowed=False),
            )
            self.assertIn("nhfwa", reply)
            self.assertNotIn("zzzzz", reply)
            self.assertNotIn("已移除", reply)
            self.assertTrue(advertised_command_suggestions(reply))
        asyncio.run(run())

    def test_reviewed_multiword_display_cannot_hide_an_unbound_extra_command(self):
        from test_s54_multiword import MultiwordRouteTests

        async def run():
            reply, store, key = await MultiwordRouteTests().discover()
            record = store.get_record(key)
            self.assertTrue(chat._advertised_reply_matches_live_record(reply, record))
            poisoned = reply + "\n例如：\n执行：把 奈飞 调到 zzzzz"
            self.assertFalse(chat._advertised_reply_matches_live_record(poisoned, record))
            with patch.object(chat, "conversation_state_store", store):
                self.assertEqual(chat._enforce_advertised_reply_contract(poisoned, key), reply)
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
