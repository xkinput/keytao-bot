"""B20 display regressions through real draft tools and offline QQ turns."""

import copy
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

import test_b19_partial_reply_binding as b19


class PendingWordHTTP(b19.DraftHTTP):
    def __init__(self, source):
        super().__init__()
        self.source = source
        self.pending_batch_id = self.batch_id if source == "draft" else source + "-batch"
        self.pending_row = {"id": 1, "word": "名创", "code": "mgwx", "type": "Phrase",
                            "action": "Create", "status": "Pending", "weight": 100}
        if source == "draft":
            self.rows.append(copy.deepcopy(self.pending_row))

    def _response(self, method, url, kwargs):
        path = urlsplit(url).path
        creator = 43 if self.source == "others_submitted" else 42
        if method == "GET" and path == "/api/batches":
            self.requests.append((method, path, copy.deepcopy(kwargs.get("params") or {})))
            batches = [] if self.source == "draft" else [{
                "id": self.pending_batch_id, "status": "Submitted", "creator": {"id": creator},
            }]
            return b19.b18.OfflineResponse({"batches": batches, "pagination": {"page": 1, "totalPages": 1}})
        if (method == "GET" and self.source != "draft"
                and path == f"/api/batches/{self.pending_batch_id}"):
            self.requests.append((method, path, {}))
            return b19.b18.OfflineResponse({"batch": {
                "id": self.pending_batch_id, "status": "Submitted", "creatorId": creator,
                "pullRequests": [copy.deepcopy(self.pending_row)],
            }})
        return super()._response(method, url, kwargs)


class B20DisplayTests(unittest.IsolatedAsyncioTestCase):
    setUp = b19.B19PartialReplyBindingTests.setUp
    tearDown = b19.B19PartialReplyBindingTests.tearDown
    matcher_send = b19.B19PartialReplyBindingTests.matcher_send
    matcher_finish = b19.B19PartialReplyBindingTests.matcher_finish
    created = b19.B19PartialReplyBindingTests.created
    seed = b19.B19PartialReplyBindingTests.seed
    turn = b19.B19PartialReplyBindingTests.turn
    partial = b19.B19PartialReplyBindingTests.partial

    async def test_d1_other_speaker_selection_is_refused_without_changing_owner(self):
        partial_id = await self.partial()
        record = self.replay.store.get_record(self.replay.author)
        snapshot, writes = copy.deepcopy(record), copy.deepcopy(self.http.writes)
        await self.turn("小美 2", partial_id, "b20-other-speaker")
        self.assertEqual(self.http.writes, writes)
        self.assertIs(self.replay.store.get_record(self.replay.author), record)
        self.assertEqual(record, snapshot)
        self.assertEqual(self.replay.last_response, "这条提议不是发给你的，本次未执行。")

    async def assert_pending_link(self, source, label):
        tools = b19.b18.TOOLS
        self.http = PendingWordHTTP(source)
        self.enterContext(patch.object(tools.http_client, "_keytao_client", self.http))
        self.enterContext(patch.object(tools, "get_keytao_url", return_value="https://keytao.vercel.app"))
        self.enterContext(patch.dict(b19.b18.FACTS, {"名创": ("mgwm", ["mgwm", "mgwx"], "ming chuang")}))
        self.enterContext(patch.dict("os.environ", {"KEYTAO_PUBLIC_BASE_QQ": "https://keytao.rea.ink"}))
        await self.seed(("名创",))
        await self.turn("名创 1", self.original_id)
        expected = f"「名创」{label}（→ mgwx" + ("）：" if source == "draft" else "，审核中）：")
        url = f"https://keytao.rea.ink/batch/{self.http.pending_batch_id}"
        reminders = [line for line in self.replay.last_response.splitlines() if line.startswith(expected)]
        self.assertEqual(reminders, [expected + url], self.replay.last_response)
        self.assertIn(("名创", "mgwm"), self.created(), self.replay.last_response)
        self.assertEqual(len(self.http.writes), 1)
        self.assertNotIn("keytao.vercel.app", self.replay.last_response)

    async def test_d2_pending_reminders_keep_batch_urls(self):
        for source, label in (("submitted", "已在待审核批次中"),
                              ("others_submitted", "已有其他用户提交"),
                              ("draft", "已在当前草稿中")):
            with self.subTest(source=source):
                await self.assert_pending_link(source, label)


if __name__ == "__main__":
    unittest.main()
