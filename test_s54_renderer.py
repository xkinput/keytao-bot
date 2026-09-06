"""Focused S54 coverage for record-backed shared multi-word rendering."""

import copy
import unittest

from keytao_bot.utils.pending_confirmation import (
    advertised_command_suggestions,
    advertised_batch_binding_pairs,
    front_insert_recommendation_copy,
    render_server_backed_batch_candidates,
    render_server_backed_single_word_lookup,
)


def reviewed_scope(word, codes, recommended, pinyin, occupied):
    candidates = [(code, code in occupied) for code in codes]
    prompt = "\n".join((
        f"词库暂无收录「{word}」：",
        f"审词：读音 {pinyin}；来源 组合推断（{word}模式）；",
        "自动审核：需管理员审核（组合推断仍需人工核验）",
        "候选编码:",
        *(f"{index}. {code} — " + (
            "已有「" + "、".join(occupied[code]) + "」"
            if code in occupied else "空位"
        ) + ("（推荐）" if code == recommended else "")
          for index, code in enumerate(codes, 1)),
        f"• 「{word}」→ {recommended}（推荐）",
        "回复编号或编码选择（可多选，如「添加1、2」）；",
        "回复「加入」写入草稿，或回复「加入并提交」写入并提交。",
    ))
    state = {
        "word": word,
        "recommendedCode": recommended,
        "serverCandidates": candidates,
        "serverOccupiedWords": occupied,
        "serverOrderingAssessments": [],
    }
    return {
        "word": word,
        "candidates": candidates,
        "occupiedWords": occupied,
        "orderingAssessments": [],
        "reviewedState": state,
        "reviewedPrompt": prompt,
    }


class S54RendererTests(unittest.TestCase):
    def setUp(self):
        self.items = [
            {"word": "大端", "code": "dsdtvo", "needsManualReview": True},
            {"word": "小端", "code": "xcdti", "needsManualReview": True},
        ]
        self.scopes = [
            reviewed_scope("大端", ["dsdt", "dsdtv", "dsdtvo"], "dsdtvo",
                           "dà duān", {"dsdt": ["打断"], "dsdtv": ["大段"]}),
            reviewed_scope("小端", ["xcdt", "xcdti", "xcdtio"], "xcdti",
                           "xiǎo duān", {"xcdt": ["小段"]}),
        ]

    def test_batch_preserves_each_shared_review_block(self):
        rendered = render_server_backed_batch_candidates(self.items, self.scopes)
        self.assertIn("审词：读音 dà duān", rendered)
        self.assertIn("审词：读音 xiǎo duān", rendered)
        self.assertEqual(2, rendered.count("组合推断仍需人工核验"))
        self.assertNotIn("dhdt", rendered)
        self.assertNotIn("thdt", rendered)
        self.assertIn("1. dsdt — 已有「打断」", rendered)
        self.assertIn("推荐编码：dsdtvo", rendered)
        self.assertIn("推荐编码：xcdti", rendered)
        self.assertNotIn("添加1、2", rendered)
        self.assertNotIn("小端 添加1", rendered)
        self.assertIn("「大端 3，小端 2」", rendered)
        self.assertIn("未选择的词保持未选", rendered)
        self.assertEqual(
            (("大端", "dsdtvo"), ("小端", "xcdti")),
            advertised_batch_binding_pairs(rendered),
        )

    def test_all_advertised_forms_reach_real_grammar(self):
        from keytao_bot.harness.authorization_grammar import (
            parse_reviewed_multi_word_selection,
        )
        from keytao_bot.utils.pending_confirmation import parse_pending_assent_phrase

        rendered = render_server_backed_batch_candidates(self.items, self.scopes)
        commands = advertised_command_suggestions(rendered)
        self.assertEqual(("加入", "加入并提交", "大端 3，小端 2"), commands)
        for command in commands:
            self.assertTrue(
                parse_pending_assent_phrase(command).matched
                or parse_reviewed_multi_word_selection(command),
                command,
            )

    def test_one_actionable_word_keeps_word_scoped_selection(self):
        rendered = render_server_backed_batch_candidates(self.items[1:], self.scopes[1:])
        self.assertIn("审词：读音 xiǎo duān", rendered)
        self.assertIn("「小端 2」", rendered)
        self.assertNotIn("添加1、2", rendered)

    def test_rich_scope_mismatch_fails_closed(self):
        tampered = copy.deepcopy(self.scopes)
        tampered[0]["reviewedState"]["recommendedCode"] = "dhdtvo"
        self.assertEqual("", render_server_backed_batch_candidates(self.items, tampered))
        tampered = copy.deepcopy(self.scopes)
        tampered[0]["reviewedPrompt"] = tampered[0]["reviewedPrompt"].replace("dsdt", "dhdt")
        self.assertEqual("", render_server_backed_batch_candidates(self.items, tampered))

    def test_reorder_recommendation_keeps_one_scoped_control(self):
        scope = reviewed_scope(
            "大端", ["dsdt", "dsdtv", "dsdtvo"], "dsdt", "dà duān",
            {"dsdt": ["打断"], "dsdtv": ["大段"]},
        )
        assessment = {
            "verdict": "front_more_common",
            "newWord": "大端", "occupantWord": "打断",
            "occupantCode": "dsdt", "freeCode": "dsdtvo", "newCode": "dsdt",
            "summary": "该词在此编码的使用证据更充分",
        }
        scope["orderingAssessments"] = [assessment]
        scope["reviewedState"]["serverOrderingAssessments"] = [assessment]
        scope["reviewedPrompt"] = scope["reviewedPrompt"].replace(
            "• 「大端」→ dsdt（推荐）",
            front_insert_recommendation_copy(assessment, 3),
        )
        rendered = render_server_backed_batch_candidates(
            [{"word": "大端", "code": "dsdt", "needsManualReview": True}], [scope],
        )
        self.assertIn("审词：读音 dà duān", rendered)
        self.assertIn("推荐调序：「大端」占 dsdt，「打断」顺延", rendered)
        self.assertIn("不调序备选编码：dsdtvo", rendered)
        self.assertNotIn("不重排选 3", rendered)
        self.assertEqual(
            ("加入", "加入并提交", "大端 1"), advertised_command_suggestions(rendered),
        )

    def test_single_word_block_without_controls_matches_batch(self):
        scope = self.scopes[0]
        rendered = render_server_backed_single_word_lookup(
            "大端", "dsdtvo", scope["candidates"], scope["occupiedWords"],
            reviewed_prompt=scope["reviewedPrompt"], actionable_controls=True,
            include_controls=False,
        )
        batch = render_server_backed_batch_candidates(self.items, self.scopes)
        self.assertIn(rendered, batch)
        self.assertNotIn("回复", rendered)


if __name__ == "__main__":
    unittest.main()
