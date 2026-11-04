"""Offline B15 regressions for sibling codes and other authors' submissions."""

import asyncio
import copy
import json
import socket
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from eval.routing.replay_prod_cases import Replay
from keytao_bot.harness.state import (
    PendingToolConfirm, pending_duplicate_confirmation_is_complete,
    pending_duplicate_confirmation_state,
)
from keytao_bot.utils import completed_draft_undo
from keytao_bot.utils.draft_mutation_store import DraftMutationClaimStore
from keytao_bot.utils.pending_confirmation import trusted_pending_word_items
from test_b12_select import SelectionReplay
from test_semantic_parser import FakeClient, raw_operation


WORDS = ("球坐标系", "柱坐标系")
CODES = ["qzbx", "qzbxv", "qzbxvi"]
OTHER_BATCH = "b1500000-0000-4000-8000-000000000007"
OWN_BATCH = "b1500000-0000-4000-8000-000000000042"


def pending_item(word=WORDS[0], code="qzbxi", source="others_submitted", **changes):
    return {
        "source": source, "batchId": OTHER_BATCH,
        "batchUrl": f"https://keytao.rea.ink/batch/{OTHER_BATCH}",
        "batchStatus": "Submitted", "itemStatus": "Pending", "action": "Create",
        "word": word, "code": code, "type": "Phrase", **changes,
    }


def notice(item):
    return f"「{item['word']}」已有其他用户提交（→ {item['code']}，审核中）：{item['batchUrl']}"


class SiblingReplay(SelectionReplay):
    def __init__(self):
        Replay.__init__(self, {
            "sender": "b15-owner", "dictionary": [],
            "candidate_facts": {
                word: {"recommended": "qzbx", "codes": list(CODES), "pinyin": pinyin}
                for word, pinyin in zip(WORDS, ("qiu zuo biao xi", "zhu zuo biao xi"))
            },
        }, harness, FakeClient)
        self.submit_extra_items = []
        self.pending_items = []
        self.pending_complete = True

    async def tool(self, name, args, *actor, **kwargs):
        if name == "keytao_pending_items_by_words":
            self.calls.append((name, copy.deepcopy(args)))
            return json.dumps({
                "success": self.pending_complete, "complete": self.pending_complete,
                "items": [item for item in self.pending_items
                          if item["word"] in args["words"] or item.get("oldWord") in args["words"]],
            }, ensure_ascii=False)
        if name in {"keytao_create_phrase", "keytao_batch_add_to_draft"}:
            items = args.get("items", [{"action": "Create", **args}])
            with patch.object(harness._draft_tools, "keytao_pending_items_by_words", AsyncMock(return_value={
                "success": True, "complete": True, "items": self.pending_items,
            })):
                guard, _ = await harness._draft_tools._pending_create_guard(
                    "qq", self.author.actor_id, items,
                    duplicate_confirmed=args.get("_pending_submitted_confirmed") is True,
                )
            if guard:
                self.calls.append((name, copy.deepcopy(args)))
                return json.dumps(guard, ensure_ascii=False)
        if name == "keytao_submit_batch" and not args.get("confirmed"):
            self.calls.append((name, copy.deepcopy(args)))
            return json.dumps(self.result(
                success=False, requiresConfirmation=True, confirmationKind="submit",
                snapshotDigest="d" * 64, warningDigest="e" * 64, auditDigest="f" * 64,
                warnings=[], snapshotItems=copy.deepcopy(self.draft),
            ), ensure_ascii=False)
        return await super().tool(name, args, *actor, **kwargs)


class B15ChatTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.replay = SiblingReplay()
        stack = self.enterContext(ExitStack())
        directory = stack.enter_context(tempfile.TemporaryDirectory())
        self.replay.claim_store = DraftMutationClaimStore(directory + "/claims.db")
        stack.enter_context(patch.object(completed_draft_undo, "get_default_draft_mutation_claim_store", return_value=self.replay.claim_store))
        stack.enter_context(patch.object(self.replay.chat, "get_default_draft_mutation_claim_store", return_value=self.replay.claim_store))
        for context in self.replay.patches():
            stack.enter_context(context)
        stack.enter_context(patch.object(self.replay.commands.tool_executor, "call", AsyncMock(side_effect=self.replay.tool)))
        self.replay.chat._finish_ai_chat_response.side_effect = lambda *args: completed_draft_undo.note_completed_operation_reply(
            self.replay.chat.current_draft_delivery_claims.get() or [], args[3],
        )
        stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")))
        stack.enter_context(patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Offline only")))
        self.general = stack.enter_context(patch.object(self.replay.chat, "get_ai_response_core", AsyncMock(return_value="Unexpected general response")))
        stack.enter_context(patch.object(self.replay.commands.keytao_review, "compare_word_commonness", AsyncMock(return_value={})))
        stack.enter_context(patch.object(self.replay.commands.keytao_review, "assess_candidate_chain_commonness", AsyncMock(return_value=[])))

    async def turn(self, message, quote=""):
        raw = raw_operation(op="draft_submit", operands=[], codes=[]) if message == "提交" else raw_operation()
        return await self.replay.turn({"message": message, "to_me": True, "quote_target": quote}, raw)

    async def candidates(self, words=WORDS):
        await self.turn("加词：" + "，".join(words))
        self.assertIsNotNone(self.replay.store.get_record(self.replay.author), self.replay.last_response)
        self.replay.client.create.reset_mock()
        self.general.reset_mock()
        return self.replay.last_response

    def created(self):
        items = []
        for name, args in self.replay.writes:
            if name == "keytao_create_phrase":
                items.append((args["word"], args["code"]))
            elif name == "keytao_batch_add_to_draft":
                items.extend((item["word"], item["code"]) for item in args["items"])
        return items

    def submissions(self):
        return sum(name == "keytao_submit_batch" for name, _ in self.replay.writes)

    def reset_ticket(self):
        self.replay.store.delete(self.replay.author)
        self.replay.history.clear()
        self.replay.writes.clear()
        self.replay.draft.clear()

    async def test_r1_sibling_recommendation_footer_binding_and_bare_add(self):
        for answer, submits in (("加入", 0), ("加入并提交", 1)):
            with self.subTest(answer=answer):
                self.reset_ticket()
                prompt = await self.candidates()
                self.assertIn("1. qzbx — 本次推荐给「球坐标系」", prompt)
                self.assertIn("2. qzbxv — ✅ 推荐（空位）", prompt)
                self.assertIn("推荐编码：qzbxv", prompt)
                self.assertIn("球坐标系 1，柱坐标系 2", prompt)
                record = self.replay.store.get_record(self.replay.author)
                self.assertEqual([(item["word"], item["code"]) for item in record.state.args["items"]],
                                 [(WORDS[0], "qzbx"), (WORDS[1], "qzbxv")])
                await self.turn(answer, prompt)
                self.assertEqual(self.created(), [(WORDS[0], "qzbx"), (WORDS[1], "qzbxv")], self.replay.last_response)
                self.assertEqual(self.submissions(), submits, self.replay.last_response)
                self.general.assert_not_awaited()
                self.replay.client.create.assert_not_awaited()

    async def test_r1_no_empty_sibling_chain_has_no_recommendation(self):
        self.replay.candidate_facts[WORDS[1]]["codes"] = ["qzbx"]
        prompt = await self.candidates()
        record = self.replay.store.get_record(self.replay.author)
        self.assertEqual([(item["word"], item["code"]) for item in record.state.args["items"]], [(WORDS[0], "qzbx")])
        self.assertNotIn("「柱坐标系」→ qzbx", prompt)
        await self.turn("加入", prompt)
        self.assertEqual(self.created(), [(WORDS[0], "qzbx")], self.replay.last_response)

    async def test_r1_same_code_selection_rejects_all_and_keeps_ticket(self):
        for message in ("球坐标系 1，柱坐标系 1", "球坐标系 qzbx，柱坐标系 qzbx"):
            with self.subTest(message=message):
                self.reset_ticket()
                prompt = await self.candidates()
                record = self.replay.store.get_record(self.replay.author)
                before = copy.deepcopy(record)
                await self.turn(message, prompt)
                self.assertEqual(self.replay.writes, [], self.replay.last_response)
                self.assertEqual(self.replay.last_response, "球坐标系、柱坐标系 选了同一个编码 qzbx，请改选；本次未写入。")
                self.assertIs(self.replay.store.get_record(self.replay.author), record)
                self.assertEqual(record, before)
                self.general.assert_not_awaited()

    async def test_r1_selection_tokens_bind_to_ticket_words_and_fail_closed(self):
        for message in (
            "球坐标系 1，未知词 2", "球坐标系 qzbx，未知词 qzbxv",
            "球坐标系 1，柱坐标系 2，garbage", "球坐标系 1，柱坐标系 2和2",
            "球坐标系 qzbxqzbxv", "球坐标系 0，柱坐标系 2",
            "球坐标系 1，球坐标系 2", "球坐标系 1，柱坐标系 9",
        ):
            with self.subTest(message=message):
                self.reset_ticket()
                prompt = await self.candidates()
                record = self.replay.store.get_record(self.replay.author)
                before = copy.deepcopy(record)
                await self.turn(message, prompt)
                self.assertEqual(self.replay.writes, [], self.replay.last_response)
                self.assertIs(self.replay.store.get_record(self.replay.author), record)
                self.assertEqual(record, before)
                self.general.assert_not_awaited()

    async def test_r1_valid_named_selections_write_only_bound_codes(self):
        for message, second_code in (("球坐标系 1，柱坐标系 2", "qzbxv"),
                                     ("球坐标系 qzbx，柱坐标系 qzbxvi", "qzbxvi")):
            with self.subTest(message=message):
                self.reset_ticket()
                prompt = await self.candidates()
                await self.turn(message, prompt)
                self.assertEqual(self.created(), [(WORDS[0], "qzbx"), (WORDS[1], second_code)], self.replay.last_response)
                self.assertEqual(self.submissions(), 0)
                self.general.assert_not_awaited()
                self.replay.client.create.assert_not_awaited()

    async def test_r2_other_author_notice_keeps_single_and_multiword_candidates(self):
        item = pending_item()
        self.replay.pending_items = [item]
        for words in ((WORDS[0],), WORDS):
            for prefix in ("", "加词："):
                with self.subTest(words=words, prefix=prefix):
                    self.reset_ticket()
                    await self.turn(prefix + "，".join(words))
                    prompt = self.replay.last_response
                    self.assertTrue(prompt.startswith(notice(item)), prompt)
                    self.assertIn("1. qzbx — ✅ 推荐（空位）", prompt)
                    self.assertNotIn("若需撤回", prompt)
                    record = self.replay.store.get_record(self.replay.author)
                    self.assertIsNotNone(record)
                    if len(words) == 1:
                        self.assertEqual(record.state.recommended_code, "qzbx")
                    self.assertEqual(self.replay.writes, [])

    async def test_r2_same_pair_requires_duplicate_confirmation(self):
        self.replay.pending_items = [pending_item(code="qzbx")]
        for words in ((WORDS[0],), WORDS):
            for answer, submits in (("加入", 0), ("加入并提交", 1)):
                with self.subTest(words=words, answer=answer):
                    self.reset_ticket()
                    prompt = await self.candidates(words)
                    self.assertIn(notice(self.replay.pending_items[0]), prompt)
                    await self.turn(answer, prompt)
                    self.assertEqual(self.created(), [])
                    self.assertIn("该词已在审核中，确认再提交一条相同词条吗？", self.replay.last_response)
                    state = self.replay.store.get_record(self.replay.author).state
                    self.assertTrue(pending_duplicate_confirmation_is_complete(state))
                    await self.turn("确认", self.replay.last_response)
                    expected = [(WORDS[0], "qzbx")]
                    if len(words) == 2:
                        expected.append((WORDS[1], "qzbxv"))
                    self.assertEqual(self.created(), expected, self.replay.last_response)
                    self.assertEqual(self.submissions(), submits, self.replay.last_response)
                    self.assertIsNone(self.replay.store.get_record(self.replay.author))

    async def test_r2_unseen_overlap_submit_prompts_then_confirm_submits(self):
        self.replay.draft = [{"action": "Create", "word": WORDS[0], "code": "qzbx", "type": "Phrase"}]
        self.replay.pending_items = [pending_item()]
        await self.turn("提交")
        self.assertEqual(self.submissions(), 0, self.replay.last_response)
        self.assertIn(notice(self.replay.pending_items[0]), self.replay.last_response)
        record = self.replay.store.get_record(self.replay.author)
        self.assertIsNotNone(record)
        self.assertEqual(record.state.function_name, "keytao_submit_batch")
        prompt = self.replay.last_response
        await self.turn("确认", prompt)
        self.assertEqual(self.submissions(), 1, self.replay.last_response)

    async def test_r2_unseen_overlap_cancel_never_submits(self):
        self.replay.draft = [{"action": "Create", "word": WORDS[0], "code": "qzbx", "type": "Phrase"}]
        self.replay.pending_items = [pending_item()]
        await self.turn("提交")
        self.assertEqual(self.submissions(), 0, self.replay.last_response)
        await self.turn("取消", self.replay.last_response)
        self.assertEqual(self.submissions(), 0)
        self.assertIsNone(self.replay.store.get_record(self.replay.author))

    async def test_r2_shown_overlap_add_and_submit_has_no_second_prompt(self):
        self.replay.pending_items = [pending_item()]
        for words in ((WORDS[0],), WORDS):
            with self.subTest(words=words):
                self.reset_ticket()
                prompt = await self.candidates(words)
                self.assertIn(notice(self.replay.pending_items[0]), prompt)
                await self.turn("加入并提交", prompt)
                self.assertEqual(self.submissions(), 1, self.replay.last_response)
                self.assertIsNone(self.replay.store.get_record(self.replay.author))

    async def test_r2_new_overlap_after_candidate_ticket_requires_submit_confirmation(self):
        prompt = await self.candidates()
        self.replay.pending_items = [pending_item()]
        await self.turn("加入并提交", prompt)
        self.assertEqual(self.created(), [(WORDS[0], "qzbx"), (WORDS[1], "qzbxv")], self.replay.last_response)
        self.assertEqual(self.submissions(), 0, self.replay.last_response)
        self.assertIn(notice(self.replay.pending_items[0]), self.replay.last_response)
        self.assertEqual(self.replay.store.get_record(self.replay.author).state.function_name, "keytao_submit_batch")
        await self.turn("确认", self.replay.last_response)
        self.assertEqual(self.submissions(), 1, self.replay.last_response)

    async def test_r2_submit_lookup_failure_does_not_block_submission(self):
        self.replay.pending_complete = False
        self.replay.draft = [{"action": "Create", "word": WORDS[0], "code": "qzbx", "type": "Phrase"}]
        await self.turn("提交")
        self.assertEqual(self.submissions(), 1, self.replay.last_response)
        self.assertIsNone(self.replay.store.get_record(self.replay.author))


class B15PendingLookupTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tools = harness._draft_tools
        self.batches = []
        self.details = {}
        self.detail_calls = []
        self.active_details = 0
        self.maximum_active_details = 0
        self.submit_preview = None
        self.submit_status = 400
        self.submit_posts = []
        self.http_posts = []
        stack = self.enterContext(ExitStack())
        stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")))
        stack.enter_context(patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Offline only")))
        stack.enter_context(patch.object(self.tools, "get_bot_token", return_value="offline-token"))
        stack.enter_context(patch.object(self.tools, "get_keytao_url", return_value="https://offline.invalid"))
        stack.enter_context(patch.object(self.tools, "get_bot_headers", return_value={}))
        stack.enter_context(patch.object(self.tools, "make_batch_url", side_effect=lambda batch: f"https://offline.invalid/batch/{batch}"))
        stack.enter_context(patch.object(self.tools, "keytao_list_draft_items", AsyncMock(return_value={"success": True, "items": []})))
        async def request_once(request, **kwargs):
            return await request()

        stack.enter_context(patch.object(self.tools.http_client, "request_with_retries", AsyncMock(side_effect=request_once)))
        owner = self

        class Response:
            def __init__(self, payload, status=200):
                self.status_code = status
                self.payload = payload

            def json(self):
                return self.payload

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, url, **kwargs):
                owner.http_posts.append(url)
                if url.endswith("/submit"):
                    owner.submit_posts.append((url, json.loads(kwargs["content"])))
                    return Response(copy.deepcopy(owner.submit_preview), owner.submit_status)
                return Response({"found": True, "user": {"id": 42}})

            async def get(self, url, **kwargs):
                if url.endswith("/api/batches"):
                    return Response({"batches": copy.deepcopy(owner.batches), "pagination": {"page": 1, "totalPages": 1}})
                batch_id = url.rsplit("/", 1)[1]
                owner.detail_calls.append(batch_id)
                owner.active_details += 1
                owner.maximum_active_details = max(owner.maximum_active_details, owner.active_details)
                try:
                    await asyncio.sleep(0)
                    value = owner.details[batch_id]
                    if isinstance(value, Exception):
                        raise value
                    return Response({"batch": copy.deepcopy(value)})
                finally:
                    owner.active_details -= 1

        stack.enter_context(patch.object(self.tools.httpx, "AsyncClient", side_effect=lambda **kwargs: Client(), create=True))

    def batch(self, batch_id, creator, items):
        self.batches.append({"id": batch_id, "status": "Submitted", "creator": {"id": creator}})
        self.details[batch_id] = {"id": batch_id, "creatorId": creator, "status": "Submitted", "pullRequests": items}

    async def lookup(self, words=WORDS):
        return await self.tools.keytao_pending_items_by_words("qq", "b15-owner", list(words))

    async def test_r2_lookup_discovers_others_all_actions_exact_word_and_old_word(self):
        self.batch(OWN_BATCH, 42, [pending_item(source="submitted", code="qzbx")])
        self.batch(OTHER_BATCH, 7, [
            pending_item(code="qzbx"), pending_item(word="新词", oldWord=WORDS[0], action="Change"),
            pending_item(word=WORDS[1], action="Delete", code="qzbxv"),
            pending_item(word="长球坐标系", code="qzbxvi"), pending_item(itemStatus="Approved", status="Approved"),
        ])
        for item in self.details[OWN_BATCH]["pullRequests"] + self.details[OTHER_BATCH]["pullRequests"]:
            item.setdefault("status", "Pending")
        result = await self.lookup()
        self.assertTrue(result["complete"], result)
        own = [item for item in result["items"] if item["source"] == "submitted"]
        others = [item for item in result["items"] if item["source"] == "others_submitted"]
        self.assertEqual([(item["word"], item["code"]) for item in own], [(WORDS[0], "qzbx")])
        self.assertEqual({item["action"] for item in others}, {"Create", "Change", "Delete"})
        self.assertEqual(len(others), 3, result)
        self.assertEqual(trusted_pending_word_items(others), others)
        result = await self.lookup(("坐标系",))
        self.assertEqual(result["items"], [], result)

    async def test_r2_other_detail_failure_is_logged_advisory_and_complete(self):
        self.batch(OTHER_BATCH, 7, [])
        self.details[OTHER_BATCH] = self.tools.httpx.TransportError("offline detail unavailable")
        with patch.object(self.tools.logger, "warning") as warning:
            result = await self.lookup()
        self.assertEqual(self.detail_calls, [OTHER_BATCH])
        self.assertTrue(result["success"], result)
        self.assertTrue(result["complete"], result)
        self.assertEqual(result["items"], [])
        warning.assert_called()

    async def test_r2_other_detail_budget_is_ten_and_sequential_own_is_preserved(self):
        for number in range(12):
            self.batch(f"b1500000-0000-4000-8000-{number:012d}", 7, [])
        self.batch(OWN_BATCH, 42, [])
        result = await self.lookup()
        self.assertTrue(result["complete"], result)
        self.assertEqual(len([batch for batch in self.detail_calls if batch != OWN_BATCH]), 10)
        self.assertEqual(self.detail_calls.count(OWN_BATCH), 1)
        self.assertEqual(self.maximum_active_details, 1)

    async def test_r2_own_detail_failure_keeps_incomplete_behavior(self):
        self.batch(OWN_BATCH, 42, [])
        self.details[OWN_BATCH] = self.tools.httpx.TransportError("offline own detail unavailable")
        result = await self.lookup()
        self.assertFalse(result["success"], result)
        self.assertFalse(result["complete"], result)

    async def test_r2_submit_tool_preserves_http_sequence_and_single_use_audit_ticket(self):
        snapshot = [{"action": "Create", "word": WORDS[0], "code": "qzbx", "type": "Phrase"}]
        audit = {
            "success": True, "verdict": "manual_review", "autoApprove": False,
            "summary": "Offline audit", "issues": [], "batchId": OWN_BATCH,
            "contentVersion": 7, "snapshotDigest": "c" * 64, "snapshotItems": snapshot,
        }
        self.submit_preview = {
            "success": False, "requiresConfirmation": True, "batchId": OWN_BATCH,
            "contentVersion": 7, "snapshotDigest": "a" * 64,
            "warningDigest": "b" * 64, "warnings": [], "snapshotItems": snapshot,
        }
        submit_url = f"https://offline.invalid/api/bot/batches/{OWN_BATCH}/submit"
        with (
            patch.object(self.tools, "_audit_current_draft_for_auto_approval", AsyncMock(return_value=audit)) as audit_call,
            patch.object(self.tools, "_SUBMIT_AUDIT_TICKETS", self.tools._SubmitAuditTicketStore()),
        ):
            preview = await self.tools.keytao_submit_batch("qq", "b15-owner", batch_id=OWN_BATCH, preview_only=True)
            self.assertEqual(self.http_posts, [submit_url])
            self.assertNotIn("pendingItems", preview)
            self.submit_status = 200
            self.submit_preview = {"success": True, "contentVersion": 8}
            arguments = {
                "confirmed": True, "batch_id": OWN_BATCH,
                "expected_content_version": 7,
                "expected_server_snapshot_digest": preview["snapshotDigest"],
                "expected_warning_digest": preview["warningDigest"],
                "expected_audit_digest": preview["auditDigest"],
            }
            confirmed = await self.tools.keytao_submit_batch("qq", "b15-owner", **arguments)
            replayed = await self.tools.keytao_submit_batch("qq", "b15-owner", **arguments)
        self.assertTrue(confirmed["success"], confirmed)
        self.assertTrue(replayed["staleConfirmation"], replayed)
        self.assertEqual(self.http_posts, [submit_url, submit_url])
        audit_call.assert_awaited_once()

    async def test_r2_submit_preview_caller_enriches_actual_lookup_and_keeps_failures_advisory(self):
        snapshot = [{"action": "Create", "word": WORDS[0], "code": "qzbx", "type": "Phrase"}]
        audit = {
            "success": True, "verdict": "manual_review", "autoApprove": False,
            "summary": "Offline audit", "issues": [], "batchId": OWN_BATCH,
            "contentVersion": 7, "snapshotDigest": "c" * 64, "snapshotItems": snapshot,
        }
        self.submit_preview = {
            "success": False, "requiresConfirmation": True, "batchId": OWN_BATCH,
            "contentVersion": 7, "snapshotDigest": "a" * 64,
            "warningDigest": "b" * 64, "warnings": [], "snapshotItems": snapshot,
        }
        for status in (200, 400):
            for failure in ("none", "other", "own", "lookup", "malformed"):
                with self.subTest(status=status, failure=failure):
                    self.batches.clear()
                    self.details.clear()
                    self.detail_calls.clear()
                    self.submit_posts.clear()
                    self.submit_status = status
                    self.batch(OTHER_BATCH, 7, [{**pending_item(), "status": "Pending"}])
                    if failure == "other":
                        self.details[OTHER_BATCH] = self.tools.httpx.TransportError("offline other detail unavailable")
                    elif failure == "own":
                        self.batch(OWN_BATCH, 42, [])
                        self.details[OWN_BATCH] = self.tools.httpx.TransportError("offline own detail unavailable")

                    async def actual_tool(name, args, platform, user_id):
                        if name == "keytao_pending_items_by_words":
                            if failure == "lookup":
                                raise self.tools.httpx.TransportError("offline lookup unavailable")
                            if failure == "malformed":
                                return "invalid json"
                        return json.dumps(await getattr(self.tools, name)(platform, user_id, **args))

                    async def actual_read(name, args, context):
                        return await actual_tool(name, args, context.platform, context.user_id)

                    with (
                        patch.object(self.tools, "_audit_current_draft_for_auto_approval", AsyncMock(return_value=audit)),
                        patch.object(self.tools, "_SUBMIT_AUDIT_TICKETS", self.tools._SubmitAuditTicketStore()),
                        patch.object(harness.chat_commands_module, "call_tool_function", AsyncMock(side_effect=actual_tool)) as call,
                        patch.object(harness.chat_commands_module.tool_executor, "call", AsyncMock(side_effect=actual_read)) as read,
                    ):
                        result = await harness.chat_commands_module._perform_submit_current_draft(
                            "qq", "b15-owner", batch_id=OWN_BATCH,
                        )
                    self.assertEqual([entry.args[0] for entry in call.await_args_list],
                                     ["keytao_submit_batch"])
                    self.assertEqual([entry.args[0] for entry in read.await_args_list],
                                     ["keytao_pending_items_by_words"])
                    self.assertEqual(read.await_args.args[1], {"words": [WORDS[0]]})
                    self.assertEqual((read.await_args.args[2].platform, read.await_args.args[2].user_id),
                                     ("qq", "b15-owner"))
                    self.assertIsNotNone(result.pending_state, result.text)
                    preview = result.data
                    self.assertTrue(preview["requiresConfirmation"], preview)
                    self.assertNotIn("uncertain", preview)
                    self.assertEqual(preview["snapshotDigest"], "a" * 64)
                    self.assertEqual(preview["warningDigest"], "b" * 64)
                    self.assertEqual(preview["snapshotItems"], snapshot)
                    self.assertEqual(len(self.submit_posts), 1)
                    self.assertTrue(self.submit_posts[0][1]["previewOnly"])
                    self.assertFalse(self.submit_posts[0][1]["confirmed"])
                    if failure == "none":
                        self.assertIn(notice(preview["pendingItems"][0]), result.text)
                        self.assertEqual(self.detail_calls, [OTHER_BATCH])
                        self.assertEqual([(item["source"], item["word"]) for item in preview["pendingItems"]],
                                         [("others_submitted", WORDS[0])])
                    else:
                        self.assertEqual(preview.get("pendingItems", []), [])

    async def test_r2_duplicate_guard_and_ticket_accept_other_author_create_only(self):
        item = pending_item(code="qzbx")
        for code, action, duplicate in (("qzbx", "Create", True), ("qzbxv", "Create", False), ("qzbx", "Change", False)):
            with self.subTest(code=code, action=action):
                with patch.object(self.tools, "keytao_pending_items_by_words", AsyncMock(return_value={
                    "success": True, "complete": True, "items": [{**item, "action": action}],
                })):
                    guard, _ = await self.tools._pending_create_guard("qq", "b15-owner", [{
                        "action": "Create", "word": WORDS[0], "code": code, "type": "Phrase",
                    }], duplicate_confirmed=False)
                if duplicate:
                    self.assertIsNotNone(guard)
                    self.assertTrue(guard["pendingDuplicateConfirmation"])
                    state = pending_duplicate_confirmation_state(PendingToolConfirm("keytao_create_phrase", {
                        "word": WORDS[0], "code": code, "type": "Phrase",
                    }), guard)
                    self.assertTrue(pending_duplicate_confirmation_is_complete(state))
                else:
                    self.assertIsNone(guard)


if __name__ == "__main__":
    unittest.main()
