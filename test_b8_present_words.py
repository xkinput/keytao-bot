"""Offline B8 compact present-word replies for explicit multi-word adds."""

import unittest

import test_semantic_query_route as query
from keytao_bot.harness.state import PendingToolConfirm
from keytao_bot.utils.pending_confirmation import (
    advertised_batch_binding_pairs,
    advertised_reply_contract,
    already_existing_word_copy,
)


class B8PresentWordsTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = query.SemanticQueryRouteTests.asyncSetUp
    asyncTearDown = query.SemanticQueryRouteTests.asyncTearDown
    lookup = query.SemanticQueryRouteTests.lookup

    present = {
        "喜上眉梢": "xsms", "欢天喜地": "htxd", "眉开眼笑": "mkyx",
        "喜气洋洋": "xqyy", "眉飞色舞": "mfsw", "愁眉苦脸": "jmkl",
        "唉声叹气": "astq",
    }

    async def add(self, words):
        self.entries.update(self.present)
        self.candidate_codes.update({"哀声叹气": "astq", "发薪": "fsxb"})
        return await self.commands._try_handle_simple_single_word_query(
            "加词：" + "，".join(words), "qq", self.author.actor_id,
            self.author, self.author.space_key, "Offline owner",
        )

    def assert_present_copy(self, response):
        self.assertEqual(response.splitlines().count(
            "已在词库：" + "、".join(f"{word} {code}" for word, code in self.present.items())
        ), 1)
        self.assertNotIn("可再追加一个编码", response)
        self.assertNotIn("当前没有必须执行的变更", response)

    async def assert_candidates(self, missing):
        words = (*self.present, *missing)
        response = await self.add(words)
        self.assert_present_copy(response)
        record = self.store.get_record(self.author)
        self.assertIsInstance(record.state, PendingToolConfirm)
        self.assertEqual(record.owner_key, self.author)
        self.assertEqual(record.state.args["_query_words"], list(words))
        self.assertEqual([row["word"] for row in record.state.args["items"]], list(missing))
        self.assertEqual([row["word"] for row in record.state.args["_candidate_scopes"]], list(missing))
        candidates = self.commands.render_server_backed_batch_candidates(
            record.state.args["items"], record.state.args["_candidate_scopes"],
        )
        self.assertTrue(response.endswith(candidates))
        self.assertEqual(
            advertised_reply_contract(response).command_suggestions,
            advertised_reply_contract(candidates).command_suggestions,
        )
        self.assertEqual(advertised_batch_binding_pairs(response), advertised_batch_binding_pairs(candidates))
        self.assertTrue(self.chat._advertised_reply_matches_live_record(response, record))
        self.assertEqual(self.chat._render_live_batch_record(record), response)
        self.main.assert_not_awaited()
        self.client.create.assert_not_awaited()

    async def test_seven_present_and_one_candidate_keep_exact_controls_and_bindings(self):
        await self.assert_candidates(("哀声叹气",))

    async def test_seven_present_and_two_candidates_keep_exact_controls_and_bindings(self):
        await self.assert_candidates(("哀声叹气", "发薪"))

    async def test_all_present_share_one_fact_line_without_hint(self):
        response = await self.add(tuple(self.present))
        self.assert_present_copy(response)
        self.assertEqual(len(response.splitlines()), 1)
        self.assertIsNone(self.store.get_record(self.author))
        self.assertFalse(advertised_reply_contract(response).requires_live_state)

    async def test_single_present_word_keeps_original_copy(self):
        response = await self.add(("喜上眉梢",))
        self.assertEqual(response, already_existing_word_copy(
            "喜上眉梢", ("xsms",), can_choose_other_code=True,
        ))


if __name__ == "__main__":
    unittest.main()
