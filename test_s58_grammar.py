"""Literal move commands share source normalization and live binding."""

import unittest

from keytao_bot.harness import authorization_grammar as grammar
from keytao_bot.harness.tools import ToolContext


class MoveToCodeGrammarTests(unittest.TestCase):
    def context(self, message):
        return ToolContext(
            current_message=message,
            trusted_word_lookup_codes_by_word={
                "奈飞": frozenset({"nhfwv"}),
                "浓厚氛围": frozenset({"nhfw"}),
            },
            trusted_entries_by_code={"nhfw": (("浓厚氛围", 100),)},
        )

    def assert_bound_move(self, message, occupant=""):
        move = grammar.parse_existing_entry_move(message)
        self.assertIsNotNone(move, message)
        self.assertEqual((move.word, move.target_code, move.named_occupant),
                         ("奈飞", "nhfw", occupant))
        self.assertTrue(grammar.message_authorizes_mutation(message), message)
        self.assertIsNone(grammar._validate_current_message_binding(
            "keytao_shift_phrase_code", {"word": "奈飞", "target_code": "nhfw"},
            self.context(message),
        ), message)

    def test_incident_binds_live_existing_word_without_candidate_inventory(self):
        self.assert_bound_move("喵喵 把 奈飞 调到 nhfw")

    def test_move_verbs_and_subject_shapes(self):
        for verb in ("调到", "调至", "挪到", "挪至", "移到", "移至", "换到",
                     "放到", "放在", "改到", "调整到", "换成", "用"):
            for subject in ("把 奈飞", "将 奈飞", "奈飞"):
                with self.subTest(verb=verb, subject=subject):
                    self.assert_bound_move(f"{subject} {verb} nhfw")

    def test_imperative_wrappers_and_operand_quotes(self):
        for wrapper in ("", "喵喵 ", "@喵喵 ", "执行：", "执行 ", "请执行 ",
                        "帮我 ", "麻烦 ", "喵喵 请执行："):
            for left, right in (("「", "」"), ("“", "”"), ("‘", "’"),
                                ("『", "』"), ('"', '"'), ("'", "'")):
                with self.subTest(wrapper=wrapper, quotes=left + right):
                    self.assert_bound_move(
                        f"{wrapper}把 {left} 奈飞 {right} 调到 {left} nhfw {right}。"
                    )

    def test_optional_eviction_clause_remains_exact(self):
        for tail in ("，浓厚氛围顺延", "；并顺延 浓厚氛围", "、挤掉「浓厚氛围」",
                     ",并且顶替 '浓厚氛围'", " 并顺延 浓厚氛围", " 并顺延"):
            with self.subTest(tail=tail):
                self.assert_bound_move("把 奈飞 调到 nhfw" + tail,
                                       "" if tail == " 并顺延" else "浓厚氛围")
        message = "把 奈飞 调到 nhfw，顶替 不存在"
        self.assertIsNotNone(grammar._validate_current_message_binding(
            "keytao_shift_phrase_code", {"word": "奈飞", "target_code": "nhfw"},
            self.context(message),
        ))

    def test_two_clauses_remain_one_exact_ordered_plan(self):
        for separator in ("，", ",", "；", ";", "、", " 和 ", " 并且 ", "，并且 "):
            for destination in ("nhfwa", "nhfwai"):
                message = f"执行：把「奈飞」调到 'nhfw'{separator}把 浓厚氛围 挪到 {destination}"
                with self.subTest(message=message):
                    plan = grammar.parse_entry_move_plan(message)
                    self.assertIsNotNone(plan)
                    self.assertEqual(tuple((move.word, move.target_code) for move in plan.moves),
                                     (("奈飞", "nhfw"), ("浓厚氛围", destination)))
                    self.assertTrue(grammar.message_authorizes_mutation(message))
                    arguments = {
                        "word": "奈飞", "target_code": "nhfw",
                        "ordered_words": ["奈飞", "浓厚氛围"],
                        "listed_words": ["奈飞", "浓厚氛围"],
                        "expected_codes": ["nhfw", destination],
                    }
                    self.assertIsNone(grammar._validate_current_message_binding(
                        "keytao_shift_phrase_code", arguments, self.context(message),
                    ))
                    for invalid in (
                        {"word": "奈飞", "target_code": "nhfw"},
                        {**arguments, "expected_codes": ["nhfw", "zzzz"]},
                        {**arguments, "ordered_words": ["浓厚氛围", "奈飞"]},
                    ):
                        self.assertIsNotNone(grammar._validate_current_message_binding(
                            "keytao_shift_phrase_code", invalid, self.context(message),
                        ))

    def test_untrusted_or_incomplete_commands_do_not_authorize(self):
        for message in (
            "不要把 奈飞 调到 nhfw", "他说把 奈飞 调到 nhfw", "例如：把 奈飞 调到 nhfw",
            "不要把奈飞调到nhfw", "媒体称把奈飞调到nhfw", "请不要把奈飞调到nhfw",
            "并非把奈飞调到nhfw", "怎么奈飞放到nhfw",
            "执行：不要把 奈飞 调到 nhfw", "执行：把 奈飞 调到 nhfw？",
            "帮我 翻译：把 奈飞 调到 nhfw", "把 奈飞 调到 nhfw 删除 浓厚氛围",
            "把 奈飞 调到 nhfw123", "把 奈飞 调到 nhfwaaa",
            "把 奈飞 调到 nhfw，把 浓厚氛围 挪到 nhfwa，删除 别的词",
        ):
            with self.subTest(message=message):
                self.assertIsNone(grammar.parse_existing_entry_move(message))
                self.assertIsNone(grammar.parse_entry_move_plan(message))

    def test_unknown_verb_is_gap_without_write_authority(self):
        for message in ("喵喵 把 奈飞 迁到 nhfw", "把奈飞迁到nhfw"):
            self.assertFalse(grammar.message_authorizes_mutation(message))
            self.assertTrue(grammar.looks_like_mutation_grammar_gap(message))

    def test_normalizing_keyword_does_not_drop_multiclause_question_guard(self):
        self.assertFalse(grammar.message_authorizes_mutation(
            "喵喵\n加词 王中王 wfw？\n加词 微服务 wfwu"
        ))


if __name__ == "__main__":
    unittest.main()
