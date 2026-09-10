"""S62 explicit frequency comparisons stay on the read-only parser seam."""

import unittest

from keytao_bot.utils.commonness_query import parse_commonness_query


MESSAGE = "加量，加梁，加亮，加辆，价量，脊索裂畸形，排列以上词的使用频率"
WORDS = ("加量", "加梁", "加亮", "加辆", "价量", "脊索裂畸形")


class S62CommonnessParserTests(unittest.TestCase):
    def test_incident_suffix_returns_only_the_six_explicit_words(self):
        self.assertEqual(parse_commonness_query(MESSAGE), WORDS)

    def test_negative_reported_and_mixed_write_requests_are_not_read_only_queries(self):
        for message in (
            "不要" + MESSAGE,
            "别" + MESSAGE,
            MESSAGE.replace("排列", "不要排列"),
            "他说" + MESSAGE,
            "他说：" + MESSAGE,
            "「" + MESSAGE + "」",
            MESSAGE + "，然后删除加辆",
            MESSAGE + "并加入草稿",
            "加量，加梁；删除加辆，排列以上词的使用频率",
            "加量，加梁，删除加辆，排列以上词的使用频率",
            "加量，加梁，加入草稿，排列以上词的使用频率",
            "加量，加梁，修改加辆，排列以上词的使用频率",
            "加量，加梁，并提交，排列以上词的使用频率",
            "加量，加梁，把加辆排到加量前面，排列以上词的使用频率",
        ):
            with self.subTest(message=message):
                self.assertIsNone(parse_commonness_query(message))

    def test_suffix_keeps_literal_word_and_count_bounds(self):
        self.assertEqual(parse_commonness_query(MESSAGE + "？"), WORDS)
        self.assertEqual(parse_commonness_query(MESSAGE.replace("排列", "请排列")), WORDS)
        twelve = ("甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸", "子", "丑")
        suffix = "，排列以上词的使用频率"
        self.assertEqual(parse_commonness_query("、".join(twelve) + suffix), twelve)
        self.assertEqual(parse_commonness_query("甲" * 32 + "，乙" + suffix), ("甲" * 32, "乙"))
        for body in ("加量", "加量，加量", "加量，ASCII", "甲" * 33 + "，乙", "、".join((*twelve, "寅"))):
            with self.subTest(body=body):
                self.assertIsNone(parse_commonness_query(body + suffix))

    def test_existing_comparison_forms_keep_their_literal_words(self):
        for message, words in (
            ("词频排序：夜泊和耶博、伊莎贝拉", ("夜泊", "耶博", "伊莎贝拉")),
            ("按常用度排序：耶博和夜泊", ("耶博", "夜泊")),
            ("耶博 和 夜泊 哪个更常用？", ("耶博", "夜泊")),
            ("哪个更常用：添加 删除", ("添加", "删除")),
            ("常用度对比：和平 和 夜泊", ("和平", "夜泊")),
        ):
            with self.subTest(message=message):
                self.assertEqual(parse_commonness_query(message), words)


if __name__ == "__main__":
    unittest.main()
