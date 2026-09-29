"""Offline S72 regression for whole-clause review verdict stripping."""

import unittest

import test_state_machine as harness
from keytao_bot.plugins import chat_render


BODY = "本次权威来源查询未完成（汉典（经编码服务）、汉典、百度百科、汉语辞典）"
INCIDENT = f"{BODY}，本轮仍需管理员审核"
WRAPPED = f"自动审核：该词需管理员审核（{INCIDENT}）"
PROBES = (
    (INCIDENT, BODY, BODY),
    (f"{BODY}，仍需管理员审核", BODY, BODY),
    (WRAPPED, BODY, WRAPPED),
    ("「田调」加词预审已标记为需管理员审核",) * 3,
    ("重码添加需管理员审核",) * 3,
    ("纯删除需要管理员确认",) * 3,
    (BODY,) * 3,
)


def capture_review(reason):
    reviewed = {}
    harness.AgentOrchestrator._update_trusted_capabilities(
        tool_name="keytao_prepare_reviewed_add",
        arguments={"word": "田调"},
        result={
            "success": True, "word": "田调", "recommendedCode": "tmtn",
            "preSubmitAudit": {"autoApprove": False, "issues": [reason]},
        },
        codes_by_word={}, word_lookup_codes_by_word={}, entries_by_code={},
        draft_words_by_id={}, draft_items_by_id={}, phrase_types_by_key={},
        reviewed_items_by_key=reviewed, candidate_slots_by_word={},
    )
    return reviewed[("田调", "tmtn")]


class ReviewReasonTests(unittest.TestCase):
    def test_trailing_verdict_is_a_whole_clause(self):
        cases = PROBES + tuple(
            (f"{BODY}{separator}{verdict}", BODY, BODY)
            for separator in "，,；;。"
            for verdict in ("仍需要管理员审核。", "该词预计需管理员确认")
        ) + (("本轮该词仍需管理员审核", "", ""),)
        for source, visible, persisted in cases:
            compact = chat_render._compact_review_reason(source)
            captured = capture_review(source)
            for stage, actual, expected in (
                ("display", compact, visible),
                ("persisted", captured["manual_review_reason"], persisted),
            ):
                with self.subTest(source=source, stage=stage):
                    self.assertEqual(actual, expected)
                    self.assertFalse(actual.endswith(("本轮仍", "仍", "为", "添加", "纯删除")))
            with self.subTest(source=source, stage="remark"):
                self.assertEqual(captured["remark"], f"喵喵审词：自动审核：{persisted}，需要管理员审核")
                self.assertIs(captured["needs_manual_review"], True)

        for sealed in (False, True):
            with self.subTest(stage="preview", sealed=sealed):
                review = {"preSubmitAudit": {"autoApprove": False, "issues": [INCIDENT]}}
                if sealed:
                    review.update(needsManualReview=True, manualReviewReason=INCIDENT)
                self.assertEqual(
                    chat_render._format_pre_submit_audit_preview(review, "tmtn"),
                    f"自动审核：{BODY}，需要管理员审核",
                )

        state = harness.AgentOrchestrator._trusted_single_pending_add(
            {"田调": (("tmtn", False), ("tmtno", False))},
            {"田调": ({"code": "tmtn"}, {"code": "tmtno"})},
            {"田调": "tmtn"}, {("田调", "tmtn"): capture_review(INCIDENT)},
        )
        self.assertEqual(state.manual_review_reason, BODY)
        self.assertEqual(state.code_remarks, {"tmtn": f"喵喵审词：自动审核：{BODY}，需要管理员审核"})


if __name__ == "__main__":
    unittest.main()
