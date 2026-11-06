"""B18 routing regressions through real draft tools and offline HTTP contracts."""

import copy
import json
import socket
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch
from urllib.parse import urlsplit

import test_state_machine as harness
from eval.routing.replay_prod_cases import Replay
from keytao_bot.harness.state import _pending_add_word_payload
from keytao_bot.plugins import chat_commands as commands
from keytao_bot.utils import completed_draft_undo
from keytao_bot.utils.draft_mutation_store import DraftMutationClaimStore
from keytao_bot.utils.pending_confirmation import render_server_backed_single_word_candidates
from test_b10_flykey import FlyKeyReplay
from test_semantic_parser import FakeClient, raw_operation


TOOLS = harness._draft_tools
REAL_CALL_TOOL = commands.call_tool_function
REAL_FORMAT_DRAFT = commands._format_draft_response
FACTS = {
    "小美": ("xmww", ["xmww", "xmwwi"], "xiao mei"),
    "小帅": ("xkeu", ["xkeu", "xkeui"], "xiao shuai"),
    "腰线": ("yzxm", ["yzxm", "yzxmi"], "yao xian"),
    "招笑": ("qzxc", ["qzxc", "qzxci", "qzxciv", "fzxc", "fzxci", "fzxciv"], "zhao xiao"),
    "宜居": ("ykjlo", ["ykjl", "ykjlo", "ykjloa"], "yi ju"),
    "挥拍": ("hvpdi", ["hvpd", "hvpdi"], "hui pai"),
    "对练": ("dvlmi", ["dvlm", "dvlmi"], "dui lian"),
    "蹭蹭": ("crcri", ["crcr", "crcri"], "ceng ceng"),
    "蹲蹲": ("dwdwi", ["dwdw", "dwdwi"], "dun dun"),
    "停售": ("tgek", ["tgek", "tgeki"], "ting shou"),
    "马哲": ("msqe", ["msqe", "msqea", "msqeai", "msfe", "msfea"], "ma zhe"),
}
TEN_WORDS = tuple(FACTS)[:10]


def reviewed_ticket(words, reorder=True):
    items, scopes = [], []
    for word in words:
        code, codes, pinyin = FACTS[word]
        occupied = {"ykjlo": ["义举"]} if reorder and word == "宜居" else {}
        candidates = [(value, value in occupied) for value in codes]
        assessments = [{"verdict": "front_more_common", "newWord": word,
                        "newCode": code, "occupantWord": "义举", "occupantCode": code,
                        "freeCode": "ykjloa"}] if occupied else []
        reviewed = harness.PendingAddWord(
            word, code, candidates, server_candidates=candidates,
            server_occupied_words=occupied, server_ordering_assessments=assessments,
            pronunciation_codes={value: pinyin for value in codes},
            code_remarks={value: "Offline sealed review" for value in codes},
            needs_manual_review=True,
        )
        items.append({"action": "Create", "word": word, "code": code, "type": "Phrase",
                      "remark": "Offline sealed review", "needsManualReview": True})
        scopes.append({"word": word, "candidates": candidates, "occupiedWords": occupied,
                       "orderingAssessments": assessments, "reviewedState": _pending_add_word_payload(reviewed),
                       "reviewedPrompt": render_server_backed_single_word_candidates(
                           word, code, candidates, occupied, assessments,
                       )})
    return harness.PendingToolConfirm("keytao_batch_add_to_draft", {
        "items": items, "_candidate_scopes": scopes, "_reviewed_multi_word": True,
        "_query_words": list(words),
    })


class OfflineResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status
        self.is_success = 200 <= status < 300
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return copy.deepcopy(self.payload)


class DraftHTTP:
    """Local keytao-next route contracts; only confirmed HTTP writes mutate rows."""

    def __init__(self):
        self.rows, self.requests, self.writes = [], [], []
        self.version = 0
        self.submissions = 0
        self.errors = []
        self.batch_id = "offline-batch"
        self.entries = [{"word": "义举", "code": "ykjlo", "type": "Phrase", "weight": 100}]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def get(self, url, **kwargs):
        return self._response("GET", url, kwargs)

    async def post(self, url, **kwargs):
        return self._response("POST", url, kwargs)

    async def request(self, method, url, **kwargs):
        return self._response(method, url, kwargs)

    def _response(self, method, url, kwargs):
        path = urlsplit(url).path
        body = json.loads(kwargs["content"]) if kwargs.get("content") is not None else kwargs.get("json") or {}
        params = kwargs.get("params") or {}
        self.requests.append((method, path, copy.deepcopy(body or params)))
        common = {"success": True, "batchId": self.batch_id, "contentVersion": self.version}
        if path.endswith("/latest-draft"):
            return OfflineResponse({**common, "exists": True, "pullRequestCount": len(self.rows), "isNew": False})
        if path.endswith("/latest-draft/items"):
            return OfflineResponse({**common, "items": self.rows, "count": len(self.rows)})
        if path.startswith("/api/pull-requests/"):
            row = next(row for row in self.rows if row["id"] == int(path.rsplit("/", 1)[1]))
            return OfflineResponse({"pullRequest": {**row, "batchId": self.batch_id,
                "batch": {"id": self.batch_id, "status": "Submitted" if self.submissions else "Draft"}}})
        if path.endswith("/preview"):
            changes = [{"phraseType": row["type"], "codes": [row["code"]],
                        "diffs": [{"type": "remove" if row["action"] == "Delete" else "add", "phrase": row}],
                        "before": [row] if row["action"] == "Delete" else [],
                        "after": [] if row["action"] == "Delete" else [row], "beforeStartLine": 0}
                       for row in self.rows]
            return OfflineResponse({"preview": {"changes": changes, "rejected": [], "summary": {
                "added": sum(row["action"] == "Create" for row in self.rows), "modified": 0,
                "deleted": sum(row["action"] == "Delete" for row in self.rows), "rejected": 0,
            }}})
        if path.endswith("/user/find"):
            return OfflineResponse({"found": True, "user": {"id": 42}})
        if path == "/api/batches":
            return OfflineResponse({"batches": [], "pagination": {"page": 1, "totalPages": 0}})
        if path.endswith("/by-word/batch") or path.endswith("/by-code/batch"):
            field = "word" if path.endswith("/by-word/batch") else "code"
            values = body[field + "s"]
            return OfflineResponse({"success": True, "count": len(values), "results": [
                {field: value, "phrases": [row for row in self.entries if row[field] == value]} for value in values
            ]})
        if path == "/api/phrases/encode":
            word = params["word"]
            codes = ["ykjl", "ykjlo", "ykjloa"] if word == "义举" else FACTS[word][1]
            return OfflineResponse({"input": word, "type": "二字词", "chars": [], "codes": codes,
                                    "altCodes": [], "flyKeyVariants": [], "phrasePinyins": ["yi ju"],
                                    "pronunciationSource": "zdic-phrase", "standardPronunciationStatus": "found",
                                    "semanticPronunciationNeeded": False, "semanticPronunciationAccepted": False})
        if path.endswith("/pull-requests/batch") or path.endswith("/pull-requests/batch-draft"):
            if not body.get("confirmed"):
                payload = {**common, "success": False, "requiresConfirmation": True,
                           "warningDigest": "a" * 64, "warnings": [], "message": "请确认将修改写入草稿"}
                if path.endswith("batch-draft"):
                    payload.update(successCount=0, failedCount=0, skippedCount=0, warnedCount=0,
                                   failed=[], skipped=[], warned=[], draftItems=[], draftTotal=0)
                return OfflineResponse(payload, 400 if path.endswith("batch-draft") else 200)
            assert body.get("expectedContentVersion") == self.version, body
            assert body.get("expectedWarningDigest") == "a" * 64, body
            keys = [(row.get("type", "Phrase"), row["code"], row["word"]) for row in body["items"]]
            assert len(keys) == len(set(keys)), keys
            written = []
            for item in body["items"]:
                row = {**copy.deepcopy(item), "id": len(self.rows) + 1, "status": "Pending"}
                row.setdefault("needsManualReview", True)
                row.setdefault("weight", 100)
                self.rows.append(row)
                written.append(row)
            self.version += 1
            self.writes.append((path, copy.deepcopy(body)))
            payload = {**common, "contentVersion": self.version, "message": "成功创建批次，包含修改提议"}
            if path.endswith("batch-draft"):
                payload.update(successCount=len(written), failedCount=0, skippedCount=0,
                               warnedCount=0, failed=[], skipped=[], warned=[],
                               draftItems=self.rows, draftTotal=len(self.rows))
            else:
                payload["pullRequestCount"] = len(written)
            return OfflineResponse(payload)
        if path.endswith("/submit"):
            if not body.get("confirmed"):
                return OfflineResponse({**common, "success": False, "requiresConfirmation": True,
                                        "warningDigest": "a" * 64, "snapshotDigest": "b" * 64,
                                        "warnings": [], "message": "批次检查完成，请确认后提交"}, 400)
            assert body.get("expectedContentVersion") == self.version, body
            assert body.get("expectedSnapshotDigest") == "b" * 64, body
            assert body.get("expectedWarningDigest") == "a" * 64, body
            self.submissions += 1
            self.version += 1
            self.writes.append((path, copy.deepcopy(body)))
            return OfflineResponse({"success": True, "message": "批次已提交审核", "batch": {
                "id": self.batch_id, "status": "Submitted", "contentVersion": self.version,
            }})
        self.errors.append((method, path, body, params))
        raise AssertionError((method, path, body, params))


class RealToolReplay(Replay):
    turn = FlyKeyReplay.turn


class B18RealToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.replay = RealToolReplay({"sender": "b18-owner", "dictionary": []}, harness, FakeClient)
        self.http = DraftHTTP()
        stack = self.enterContext(ExitStack())
        directory = stack.enter_context(tempfile.TemporaryDirectory())
        self.claim_store = DraftMutationClaimStore(directory + "/claims.db")
        stack.enter_context(patch.object(completed_draft_undo, "get_default_draft_mutation_claim_store", return_value=self.claim_store))
        stack.enter_context(patch.object(self.replay.chat, "get_default_draft_mutation_claim_store", return_value=self.claim_store))
        stack.enter_context(patch.object(TOOLS, "get_default_draft_mutation_claim_store", return_value=self.claim_store))
        for context in self.replay.patches():
            stack.enter_context(context)
        self.replay.chat._finish_ai_chat_response.side_effect = lambda *args: completed_draft_undo.note_completed_operation_reply(
            self.replay.chat.current_draft_delivery_claims.get() or [], args[3],
        )
        # Dispatch and tool implementation stay real; the transport is the fake boundary.
        stack.enter_context(patch.object(self.replay.chat, "call_tool_function", REAL_CALL_TOOL))
        stack.enter_context(patch.object(commands, "_format_draft_response", REAL_FORMAT_DRAFT))
        stack.enter_context(patch.object(commands.tool_executor, "_get_tool_function", TOOLS.TOOL_FUNCTIONS.get))
        stack.enter_context(patch.object(TOOLS.httpx, "AsyncClient", side_effect=lambda **_kwargs: self.http, create=True))
        stack.enter_context(patch.object(TOOLS.http_client, "_keytao_client", self.http))
        stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")))
        stack.enter_context(patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Offline only")))

    def tearDown(self):
        self.assertEqual(self.http.errors, [], "Incomplete offline HTTP contract")

    def seed(self, words=TEN_WORDS, reorder=True):
        ticket = reviewed_ticket(words, reorder)
        self.assertTrue(self.replay.store.set(self.replay.author, ticket))
        record = self.replay.store.get_record(self.replay.author)
        self.replay.prompt = self.replay.chat._render_live_batch_record(record)
        self.assertTrue(self.replay.prompt)
        self.replay.history[:] = [{"role": "assistant", "content": self.replay.prompt}]
        return record

    async def turn(self, message, raw=None):
        return await self.replay.turn({"message": message, "to_me": True}, raw or raw_operation(
            op="select_candidate", refers_to_pending=True,
        ))

    def created(self):
        return [(row["word"], row["code"]) for row in self.http.rows if row["action"] == "Create"]

    async def test_f2_multicode_named_plus_reorder_rest_submits_once(self):
        self.seed()
        await self.turn("招笑 1 4，其他用推荐的 加入并提交")
        expected = [("招笑", "qzxc"), ("招笑", "fzxc")] + [
            (word, FACTS[word][0]) for word in TEN_WORDS if word != "招笑"
        ] + [("义举", "ykjloa")]
        self.assertCountEqual(self.created(), expected, self.replay.last_response)
        self.assertIn(("义举", "ykjlo"), [(row["word"], row["code"]) for row in self.http.rows if row["action"] == "Delete"])
        self.assertEqual(sum(path.endswith("/pull-requests/batch") for path, _body in self.http.writes), 1)
        self.assertEqual(self.http.submissions, 1, self.replay.last_response)
        self.assertIsNone(self.replay.store.get_record(self.replay.author))

    async def test_r2_batch_multicode_named_plus_plain_rest(self):
        self.seed(("招笑", "小美", "小帅"), reorder=False)
        await self.turn("招笑 1 4，其他用推荐的 加入并提交")
        self.assertCountEqual(self.created(), [("招笑", "qzxc"), ("招笑", "fzxc"),
                                             ("小美", "xmww"), ("小帅", "xkeu")], self.replay.last_response)
        self.assertTrue(any(path.endswith("batch-draft") for path, _body in self.http.writes))
        self.assertEqual(self.http.submissions, 1, self.replay.last_response)

    async def test_r2_named_reorder_one_code_with_plain_rest(self):
        self.seed(("宜居", "小美", "小帅"))
        await self.turn("宜居 2，其他用推荐的 加入并提交")
        self.assertCountEqual(self.created(), [("宜居", "ykjlo"), ("义举", "ykjloa"),
                                             ("小美", "xmww"), ("小帅", "xkeu")], self.replay.last_response)
        self.assertEqual(self.http.submissions, 1, self.replay.last_response)

    async def test_r2_named_reorder_two_codes_with_plain_rest(self):
        self.seed(("宜居", "小美", "小帅"))
        await self.turn("宜居 1 2，其他用推荐的 加入并提交")
        self.assertCountEqual(self.created(), [("宜居", "ykjl"), ("宜居", "ykjlo"), ("义举", "ykjloa"),
                                             ("小美", "xmww"), ("小帅", "xkeu")], self.replay.last_response)
        self.assertEqual(self.http.submissions, 1, self.replay.last_response)

    async def test_r2_named_reorder_two_codes_reversed_with_plain_rest(self):
        self.seed(("宜居", "小美", "小帅"))
        await self.turn("宜居 2 1，其他用推荐的 加入并提交")
        self.assertCountEqual(self.created(), [("宜居", "ykjl"), ("宜居", "ykjlo"), ("义举", "ykjloa"),
                                             ("小美", "xmww"), ("小帅", "xkeu")], self.replay.last_response)
        self.assertEqual(self.http.submissions, 1, self.replay.last_response)

    async def test_r2_named_reorder_word_selects_empty_code_with_plain_rest(self):
        self.seed(("宜居", "小美", "小帅"))
        await self.turn("宜居 1，其他用推荐的 加入并提交")
        self.assertCountEqual(self.created(), [("宜居", "ykjl"), ("小美", "xmww"), ("小帅", "xkeu")],
                              self.replay.last_response)
        self.assertEqual(self.http.submissions, 1, self.replay.last_response)

    async def test_r2_named_reorder_two_codes_cannot_take_occupants_only_destination(self):
        self.seed(("宜居", "小美", "小帅"))
        await self.turn("宜居 2 3，其他用推荐的 加入并提交")
        self.assertEqual(self.created(), [], self.replay.last_response)
        self.assertEqual(self.http.writes, [])

    async def test_r2_shift_multicode_named_without_submit(self):
        self.seed(("招笑", "宜居", "小美"))
        await self.turn("招笑 1 4，其他用推荐的 加入")
        self.assertCountEqual(self.created(), [("招笑", "qzxc"), ("招笑", "fzxc"), ("宜居", "ykjlo"),
                                             ("义举", "ykjloa"), ("小美", "xmww")], self.replay.last_response)
        self.assertEqual(self.http.submissions, 0)

    async def test_r2_batch_without_submit(self):
        self.seed(("招笑", "小美"), reorder=False)
        await self.turn("招笑 1 4，其他用推荐的 加入")
        self.assertCountEqual(self.created(), [("招笑", "qzxc"), ("招笑", "fzxc"), ("小美", "xmww")],
                              self.replay.last_response)
        self.assertEqual(self.http.submissions, 0)

    async def test_r2_single_word_rest_uses_real_create_without_submit(self):
        ticket = reviewed_ticket(("小美",), False)
        payload = ticket.args["_candidate_scopes"][0]["reviewedState"]
        state = harness.PendingAddWord(
            "小美", "xmww", [("xmww", False), ("xmwwi", False)],
            server_candidates=[("xmww", False), ("xmwwi", False)],
            pronunciation_codes=payload["pronunciationCodes"], needs_manual_review=True,
        )
        self.assertTrue(self.replay.store.set(self.replay.author, state))
        await self.turn("小美 1，其他用推荐的 加入")
        self.assertEqual(self.created(), [("小美", "xmww")], self.replay.last_response)
        self.assertEqual(self.http.submissions, 0)
        requests = [body for method, path, body in self.http.requests if path.endswith("/pull-requests/batch")]
        self.assertTrue(requests)
        self.assertTrue(all("previewOnly" in body for body in requests), requests)

    async def test_r2_single_word_rest_uses_real_create_and_submit(self):
        state = harness.PendingAddWord(
            "小美", "xmww", [("xmww", False), ("xmwwi", False)],
            server_candidates=[("xmww", False), ("xmwwi", False)],
            pronunciation_codes={"xmww": "xiao mei", "xmwwi": "xiao mei"}, needs_manual_review=True,
        )
        self.assertTrue(self.replay.store.set(self.replay.author, state))
        record = self.replay.store.get_record(self.replay.author)
        self.replay.store.bind_origin_message_ids(record, [1802])
        prompt = self.replay.chat._render_live_single_candidate_record(record)
        original = harness.ReplyReferenceInfo

        def reference(**kwargs):
            return original(**{**kwargs, "message_id": "1802"})

        with patch.object(harness, "ReplyReferenceInfo", side_effect=reference):
            await self.replay.turn({"message": "小美 1，其他用推荐的 加入并提交", "to_me": True,
                                    "quote_target": prompt}, raw_operation())
        self.assertEqual(self.created(), [("小美", "xmww")], self.replay.last_response)
        self.assertEqual(self.http.submissions, 1, self.replay.last_response)
        self.assertIsNone(self.replay.store.get_record(self.replay.author))

    async def test_r3_partial_selection_keeps_live_remainder_and_reply_is_truthful(self):
        rest = tuple(word for word in TEN_WORDS if word != "宜居")
        record = self.seed(("马哲", *rest), reorder=False)
        self.replay.store.bind_origin_message_ids(record, [1801])
        await self.turn("马哲 2 5")
        self.assertCountEqual(self.created(), [("马哲", "msqea"), ("马哲", "msfea")], self.replay.last_response)
        remaining = self.replay.store.get_record(self.replay.author)
        self.assertIsNotNone(remaining, self.replay.last_response)
        self.assertEqual(remaining.state.args["_query_words"], list(rest))
        self.assertIn("1801", remaining.origin_message_ids)
        self.assertEqual(remaining.created_at, record.created_at)
        self.assertEqual(remaining.expires_at, record.expires_at)
        self.assertEqual(remaining.expires_at - remaining.created_at, 14400)
        self.assertIn("候选仍有效，请逐词处理", self.replay.last_response)
        self.assertIn("未选择：" + "、".join("「" + word + "」" for word in rest), self.replay.last_response)
        await self.turn("其它的用默认编码")
        self.assertCountEqual(self.created(), [("马哲", "msqea"), ("马哲", "msfea")] + [
            (word, FACTS[word][0]) for word in rest
        ], self.replay.last_response)
        self.assertEqual(self.http.submissions, 0)

    async def test_r4_rest_only_plain_words_adds_all_recommendations(self):
        for message in ("其它的用默认编码", "其他用推荐的", "其余用推荐 加入并提交"):
            with self.subTest(message=message):
                self.http.rows.clear()
                self.http.writes.clear()
                self.http.version = self.http.submissions = 0
                self.replay.store.delete(self.replay.author)
                self.seed(("小美", "小帅"), reorder=False)
                await self.turn(message)
                self.assertCountEqual(self.created(), [("小美", "xmww"), ("小帅", "xkeu")], self.replay.last_response)
                self.assertEqual(self.http.submissions, int("提交" in message), self.replay.last_response)

    async def test_r5_rest_only_without_ticket_never_looks_up_sentence(self):
        for message in ("其它的用默认编码", "其他用推荐的", "其余用推荐 加入并提交"):
            with self.subTest(message=message):
                await self.turn(message, raw_operation())
                self.assertEqual(self.replay.last_response, "当前没有待确认的操作。")
                self.assertEqual(self.http.requests, [])

    async def test_r5_semantic_confirm_without_ticket_never_looks_up_sentence(self):
        for frame in ("direct", "unknown", "question", "none"):
            with self.subTest(frame=frame):
                await self.turn("继续刚才那一批", raw_operation(op="confirm", operands=[], codes=[],
                                                          refers_to_pending=True, frame=frame))
                self.assertEqual(self.replay.last_response, "当前没有待确认的操作。")
                self.assertEqual(self.http.requests, [])


class B18ShiftValidationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")))
        self.enterContext(patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Offline only")))

    async def test_r1_invalid_companions_rejected_before_http(self):
        base = {"action": "Create", "word": "招笑", "code": "qzxc", "type": "Phrase", "needsManualReview": True}
        invalid = {
            "duplicate_pair": [base, dict(base)],
            "target_code": [{**base, "code": "ykjlo"}],
            "companion_code_collision": [base, {**base, "word": "小美"}],
            "non_create": [{**base, "action": "Delete"}],
            "too_many": [{**base, "word": "词" + str(index)} for index in range(21)],
            "same_word_outside_seal": [{**base, "word": "宜居", "code": "abcdef"}],
        }
        for case, items in invalid.items():
            with self.subTest(case=case), patch.object(TOOLS.httpx, "AsyncClient", side_effect=AssertionError("No HTTP"), create=True) as http:
                result = await TOOLS.keytao_shift_phrase_code(
                    "qq", "b18-owner", "宜居", "ykjlo", additional_items=items,
                    _reviewed_pinyin="yi ju", _reviewed_candidate_codes=["ykjl", "ykjlo", "ykjloa"],
                )
                self.assertFalse(result.get("success"), result)
                http.assert_not_called()


if __name__ == "__main__":
    unittest.main()
