#!/usr/bin/env python3
"""Focused regression tests for the KeyTao auto-approval review gate.

Self-contained: stubs nonebot/httpx/openai the same way test_state_machine.py
does, so it runs without a NoneBot runtime.

    uv run python test_review_gate.py
"""
import asyncio
import importlib.util
import json
import os
import sys
import types
from unittest.mock import AsyncMock, patch
from urllib.parse import unquote

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- external module stubs (must come before importing keytao_bot) ----------

sys.modules["httpx"] = types.ModuleType("httpx")

_fake_nonebot = types.ModuleType("nonebot")


class _FakeMatcher:
    def handle(self):
        return lambda f: f

    async def finish(self, *a, **kw):
        pass


_fake_nonebot.on_message = lambda **kw: _FakeMatcher()
_fake_nonebot.on_command = lambda *a, **kw: _FakeMatcher()


class _FakeConfig:
    openai_api_key = "fake"
    openai_base_url = "https://fake"
    openai_model = "fake-model"
    openai_max_tokens = 1000
    openai_temperature = 0.7
    keytao_api_base = "https://fake"
    bot_api_token = "fake"


class _FakeDriver:
    config = _FakeConfig()

    def on_shutdown(self, func):
        return func


_fake_nonebot.get_driver = lambda: _FakeDriver()
sys.modules["nonebot"] = _fake_nonebot

_fake_adapters = types.ModuleType("nonebot.adapters")
_fake_adapters.Bot = type("Bot", (), {})
_fake_adapters.Event = type("Event", (), {})
sys.modules["nonebot.adapters"] = _fake_adapters

_fake_rule = types.ModuleType("nonebot.rule")
_fake_rule.Rule = lambda f: f
_fake_rule.to_me = lambda: lambda: None
sys.modules["nonebot.rule"] = _fake_rule

_fake_log = types.ModuleType("nonebot.log")


class _FakeLogger:
    def info(self, *a, **kw):
        pass

    def debug(self, *a, **kw):
        pass

    def warning(self, *a, **kw):
        pass

    def error(self, *a, **kw):
        pass


_fake_log.logger = _FakeLogger()
sys.modules["nonebot.log"] = _fake_log

_fake_exception = types.ModuleType("nonebot.exception")


class FinishedException(Exception):
    pass


_fake_exception.FinishedException = FinishedException
sys.modules["nonebot.exception"] = _fake_exception

_fake_openai = types.ModuleType("openai")
_fake_openai.AsyncOpenAI = None
sys.modules["openai"] = _fake_openai

sys.modules["duckduckgo_search"] = types.ModuleType("duckduckgo_search")

# --- modules under test -----------------------------------------------------

from keytao_bot.utils import keytao_batch_review as batch_review_module  # noqa: E402
from keytao_bot.utils import keytao_review as review_module  # noqa: E402
from keytao_bot.utils.http_client import KeytaoApiError  # noqa: E402
from keytao_bot.utils.keytao_encoding import normalize_contextual_phrase_encoding  # noqa: E402
from keytao_bot.utils.keytao_review import (  # noqa: E402
    ReviewHttpConfig,
    audit_draft_items,
    manual_preaudit_issue_for_item,
    prepare_css_reviewed_item,
    prepare_reviewed_word,
)
from keytao_bot.utils.review_flags import (  # noqa: E402
    MANUAL_REVIEW_FIELD,
    MANUAL_REVIEW_PREFIX,
    MANUAL_REVIEW_REASON_FIELD,
    read_manual_review_flag,
)

_review_tools_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "keytao_bot",
    "skills",
    "keytao-review",
    "tools.py",
)
_review_spec = importlib.util.spec_from_file_location("keytao_review_tools_for_gate_test", _review_tools_path)
_review_tools = importlib.util.module_from_spec(_review_spec)
_review_spec.loader.exec_module(_review_tools)


passed = 0
failed = 0


def check(name: str, result: bool):
    global passed, failed
    if result:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}")


CONFIG = ReviewHttpConfig(api_base="https://fake", bot_token="fake")


def _encode_data():
    return {
        "success": True,
        "codes": ["ceek", "ceeko"],
        "chars": [
            {
                "char": "测",
                "pinyin": "ce",
                "pinyins": ["cè"],
                "pronunciationLookupStatus": "found",
                "phoneticCode": "ce",
                "shapeCode": "k",
            },
            {
                "char": "试",
                "pinyin": "shi",
                "pinyins": ["shì"],
                "pronunciationLookupStatus": "found",
                "phoneticCode": "ek",
                "shapeCode": "o",
            },
        ],
    }


def _authoritative_evidence():
    return {
        "success": True,
        "groups": [{
            "pinyin": "ce shi",
            "normalized": ["ce", "shi"],
            "sources": [{"source": "汉典", "url": "https://example.test", "category": "dictionary", "trust": 5}],
            "score": 5,
            "fallback": False,
        }],
        "sources": [],
    }


def test_s14_wrong_entry_pronunciation_never_reaches_candidates():
    print("\n🧪 S14 wrong-entry pronunciation poisoning")

    async def _run():
        review_module._clear_review_caches()
        encode_data = {
            "success": True,
            "codes": ["lxmm", "lxmmo", "lxmmov"],
            "altCodes": [],
            "pronunciationSource": "pinyin-pro-context",
            "standardPronunciationStatus": "absent",
            "semanticPronunciationNeeded": False,
            "chars": [
                {
                    "char": "亮",
                    "pinyin": "liàng",
                    "pinyins": ["liàng"],
                    "pronunciationLookupStatus": "found",
                    "phoneticCode": "lx",
                    "shapeCode": "o",
                },
                {
                    "char": "面",
                    "pinyin": "miàn",
                    "pinyins": ["miàn"],
                    "pronunciationLookupStatus": "found",
                    "phoneticCode": "mm",
                    "shapeCode": "v",
                },
            ],
        }

        async def poisoned_search(query, max_results=3):
            if "site:zdic.net" not in query:
                return []
            return [{
                "title": "光面_汉典",
                "url": "https://www.zdic.net/hans/%E5%85%89%E9%9D%A2",
                "snippet": "光面 拼音：guāng miàn",
            }]

        async def poisoned_page(url):
            if "光面" not in unquote(url):
                return ""
            return "光面 汉典 拼音：guāng miàn 光滑的表面。"

        with (
            patch.object(review_module, "_search_web", side_effect=poisoned_search),
            patch.object(review_module, "_fetch_text", side_effect=poisoned_page),
            patch.object(
                review_module,
                "fetch_keytao_encode",
                AsyncMock(return_value=encode_data),
            ),
            patch.object(review_module, "lookup_words", AsyncMock(return_value={})),
            patch.object(review_module, "lookup_codes", AsyncMock(return_value={})),
            patch.object(
                review_module,
                "_infer_entity_knowledge",
                AsyncMock(return_value={"recognized": False}),
            ),
            patch.object(
                review_module,
                "_contextual_pronunciation_group",
                AsyncMock(return_value=None),
            ),
        ):
            review = await prepare_reviewed_word(CONFIG, "亮面")

        pronunciations = review.get("pronunciations", [])
        syllables = {
            syllable
            for pronunciation in pronunciations
            for syllable in pronunciation.get("normalized", [])
        }
        codes = {
            code
            for pronunciation in pronunciations
            for code in pronunciation.get("codes", [])
        }
        check("S14 rejects the poisoned guang syllable", "guang" not in syllables)
        check(
            "S14 rejects every poisoned gxmm candidate",
            not any(code.startswith("gxmm") for code in codes),
        )
        check(
            "S14 keeps the verified own-character lxmm chain",
            any(code.startswith("lxmm") for code in codes),
        )

    asyncio.run(_run())


def test_reviewed_add_chi_xi_no_authoritative_entry_or_web_uses_verified_own_characters():
    """吃席 must remain usable when only its own verified character readings exist."""
    print("\n🧪 吃席 reviewed-add fallback without authoritative or web evidence")

    async def _run():
        review_module._clear_review_caches()
        no_web_evidence = {
            "success": True,
            "word": "吃席",
            "groups": [],
            "sources": [],
            "hasEvidence": False,
            "rejections": [],
        }
        own_character_encode = {
            "success": True,
            "word": "吃席",
            "codes": ["wkxk", "wkxko", "wkxkoo"],
            "altCodes": [],
            "pronunciationSource": "zdic-unavailable",
            "standardPronunciationStatus": "unavailable",
            "semanticPronunciationNeeded": False,
            "phrasePinyins": ["chī", "xí"],
            "contextPhrasePinyins": ["chī", "xí"],
            "chars": [
                {
                    "char": "吃",
                    "pinyin": "chī",
                    "pinyins": ["chī"],
                    "phoneticCode": "wk",
                    "shapeCode": "ouva",
                },
                {
                    "char": "席",
                    "pinyin": "xí",
                    "pinyins": ["xí"],
                    "phoneticCode": "xk",
                    "shapeCode": "ovia",
                },
            ],
        }
        entity_knowledge = {
            "recognized": True,
            "word": "吃席",
            "entityType": "common_word",
            "confidence": 0.95,
            "description": "赴宴吃酒席，是现代汉语常见说法。",
            "pinyin": "chī xí",
        }
        pre_submit_audit = {
            "success": True,
            "autoApprove": False,
            "issues": ["无权威整词页，保留人工审核"],
            "previewOnly": True,
        }

        with (
            patch.object(
                review_module,
                "collect_pronunciation_evidence_limited",
                AsyncMock(return_value=no_web_evidence),
            ),
            patch.object(
                review_module,
                "fetch_keytao_encode",
                AsyncMock(return_value=own_character_encode),
            ),
            patch.object(review_module, "lookup_words", AsyncMock(return_value={})),
            patch.object(review_module, "lookup_codes", AsyncMock(return_value={})),
            patch.object(
                review_module,
                "_infer_entity_knowledge",
                AsyncMock(return_value=entity_knowledge),
            ) as infer_entity,
            patch.object(
                review_module,
                "_contextual_pronunciation_group",
                AsyncMock(return_value=None),
            ),
            patch.object(
                _review_tools,
                "_build_pre_submit_audit",
                AsyncMock(return_value=pre_submit_audit),
            ),
            patch.object(
                _review_tools,
                "assess_candidate_chain_commonness",
                AsyncMock(return_value=[]),
            ),
        ):
            review = await _review_tools.keytao_prepare_reviewed_add("吃席")

        check("吃席 entity/context gate still runs", infer_entity.await_count == 1)
        check("吃席 reviewed-add keeps a usable pronunciation", bool(review.get("pronunciations")))
        check("吃席 reviewed-add recommends wkxk", review.get("recommendedCode") == "wkxk")
        check("吃席 is not pronunciation-unresolved", review.get("pronunciationUnresolved") is not True)
        check("吃席 fallback remains sealed for manual review", read_manual_review_flag(review) is True)
        first_pronunciation = next(iter(review.get("pronunciations") or []), {})
        check(
            "吃席 fallback keeps the historical entity/context label",
            first_pronunciation.get("sourceSummary")
            == "本喵实体语境判断（常见词，暂无权威页）",
        )
        check(
            "吃席 fallback is structurally marked as own-character evidence",
            first_pronunciation.get("readingEvidenceKind")
            == "own_character_entity_context",
        )

    asyncio.run(_run())


def test_pronunciation_word_binding_window_and_exact_direct_entry():
    print("\n🧪 pronunciation evidence word-binding boundaries")

    near_text = "亮面" + ("甲" * 70) + " 拼音：liàng miàn"
    near, near_rejections = review_module._extract_labeled_pinyin_sequences(
        near_text,
        "亮面",
    )
    check("word inside the 80-character vicinity is accepted", near == [("liang", "mian")])
    check("nearby match records no binding rejection", near_rejections == 0)

    far_text = "亮面" + ("甲" * 90) + " 拼音：liàng miàn"
    far, far_rejections = review_module._extract_labeled_pinyin_sequences(
        far_text,
        "亮面",
    )
    check("word outside the 80-character vicinity is rejected", far == [])
    check("far match records one binding rejection", far_rejections == 1)

    direct, direct_rejections = review_module._extract_labeled_pinyin_sequences(
        "拼音：liàng miàn",
        "亮面",
        exact_entry_direct_url=review_module._direct_url_entry_matches_word(
            "https://www.zdic.net/hans/%E4%BA%AE%E9%9D%A2",
            "亮面",
        ),
    )
    check("exact-entry direct URL may bind without nearby title text", direct == [("liang", "mian")])
    check("exact-entry direct URL records no rejection", direct_rejections == 0)
    check(
        "different-entry URL never receives the direct-entry exemption",
        review_module._direct_url_entry_matches_word(
            "https://www.zdic.net/hans/%E5%85%89%E9%9D%A2",
            "亮面",
        ) is False,
    )


def test_pronunciation_groups_require_known_character_readings():
    print("\n🧪 pronunciation groups require known per-character readings")

    async def review_with(first_char):
        review_module._clear_review_caches()
        evidence = {
            "success": True,
            "groups": [{
                "pinyin": "guang mian",
                "normalized": ["guang", "mian"],
                "sources": [{
                    "source": "汉典",
                    "url": "https://www.zdic.net/hans/%E4%BA%AE%E9%9D%A2",
                    "category": "dictionary",
                    "trust": 5,
                }],
                "score": 5,
            }],
            "sources": [],
        }
        encode_data = {
            "success": True,
            "codes": ["lxmm", "lxmmo", "lxmmov"],
            "pronunciationSource": "pinyin-pro-context",
            "standardPronunciationStatus": "absent",
            "semanticPronunciationNeeded": False,
            "chars": [
                first_char,
                {
                    "char": "面",
                    "pinyin": "miàn",
                    "pinyins": ["miàn"],
                    "pronunciationLookupStatus": "found",
                    "phoneticCode": "mm",
                    "shapeCode": "v",
                },
            ],
        }
        with (
            patch.object(
                review_module,
                "collect_pronunciation_evidence_limited",
                AsyncMock(return_value=evidence),
            ),
            patch.object(
                review_module,
                "fetch_keytao_encode",
                AsyncMock(return_value=encode_data),
            ),
            patch.object(review_module, "lookup_words", AsyncMock(return_value={})),
            patch.object(review_module, "lookup_codes", AsyncMock(return_value={})),
            patch.object(review_module.logger, "warning") as warning_mock,
        ):
            result = await prepare_reviewed_word(CONFIG, "亮面")
        return result, warning_mock

    async def _run():
        mismatch, mismatch_log = await review_with({
            "char": "亮",
            "pinyin": "liàng",
            "pinyins": ["liàng"],
            "pronunciationLookupStatus": "found",
            "phoneticCode": "lx",
            "shapeCode": "o",
        })
        mismatch_sequences = [
            item.get("normalized")
            for item in mismatch.get("pronunciations", [])
            if isinstance(item, dict)
        ]
        check("mismatched guang group is removed", ["guang", "mian"] not in mismatch_sequences)
        check("verified own-character liang group remains", ["liang", "mian"] in mismatch_sequences)
        check("mismatch keeps the verified lxmm recommendation", mismatch.get("recommendedCode") == "lxmm")
        check("mismatch rejection is logged", mismatch_log.call_count >= 1)

        unavailable, unavailable_log = await review_with({
            "char": "亮",
            "pinyin": "guāng",
            "pinyins": [],
            "pronunciationLookupStatus": "unavailable",
            "phoneticCode": "gx",
            "shapeCode": "o",
        })
        check("unavailable lookup does not authorize guang", unavailable.get("pronunciations") == [])
        check("unavailable lookup fails closed", unavailable.get("pronunciationUnresolved") is True)
        check("unavailable rejection is logged", unavailable_log.call_count >= 1)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 13. A failed dictionary lookup must never read as "that code is free"
# ---------------------------------------------------------------------------

def test_lookup_failure_forces_manual_review():
    print("\n🧪 lookup failure forces manual review (13)")

    async def _run():
        review_module._clear_review_caches()

        async def failing_lookup_codes(config, codes):
            raise KeytaoApiError("词库编码批量查询失败")

        with patch.object(review_module, "collect_pronunciation_evidence_limited",
                          AsyncMock(return_value=_authoritative_evidence())):
            with patch.object(review_module, "fetch_keytao_encode", AsyncMock(return_value=_encode_data())):
                with patch.object(review_module, "lookup_words", AsyncMock(return_value={})):
                    with patch.object(review_module, "lookup_codes", side_effect=failing_lookup_codes):
                        with patch.object(review_module, "_infer_entity_knowledge", AsyncMock(return_value={})):
                            review = await prepare_reviewed_word(CONFIG, "测试")

        check("failed lookup is reported explicitly", review.get("lookupFailed") is True)
        check("failed lookup is not auto reviewable", review.get("autoReviewable") is False)
        check("failed lookup sets structured manual flag", read_manual_review_flag(review) is True)
        check("failed lookup reason is recorded", "词库占位查询失败" in str(review.get(MANUAL_REVIEW_REASON_FIELD)))
        check("failed lookup emits no recommended code", review.get("recommendedCode") == "")
        statuses = review.get("pronunciations", [{}])[0].get("candidateStatuses", [])
        check("failed lookup marks occupancy unknown, not free",
              bool(statuses) and all(status.get("occupied") is None for status in statuses))

        # An item built on that review must not auto-pass.
        async def fake_prepare(config, word):
            return review

        with patch.object(review_module, "prepare_reviewed_word", side_effect=fake_prepare):
            audit = await audit_draft_items(CONFIG, [
                {"action": "Create", "word": "测试", "code": "ceek", "type": "Phrase"},
            ])

        check("lookup failure blocks auto approval", audit.get("autoApprove") is False)
        check("lookup failure produces an admin issue",
              any("词库占位查询失败" in issue for issue in audit.get("issues", [])))
        check("lookup failure yields no approved item", audit.get("approvedItems") == [])
        check("audit carries structured manual flag", read_manual_review_flag(audit) is True)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 14. exact_existing means DUPLICATE, not auto-approvable
# ---------------------------------------------------------------------------

def test_exact_existing_is_duplicate_not_approval():
    print("\n🧪 exact existing row is a duplicate (14)")

    async def _run():
        review_module._clear_review_caches()
        existing = {"否则": [{"word": "否则", "code": "fao", "type": "CSS"}]}

        async def fake_commonness(word):
            # Deliberately strong: even a very common word must not auto-pass
            # once it is already in the dictionary at this exact code.
            return {
                "success": True,
                "word": word,
                "score": 0.95,
                "signals": {"corpus": 0.9, "search": 0.9, "dictionary": 0.9, "encyclopedia": 0.9},
                "evidence": {"dictionary": ["汉典"]},
                "entityKnowledge": {"accepted": False},
            }

        with patch.object(review_module, "lookup_words", AsyncMock(return_value=existing)):
            with patch.object(review_module, "lookup_codes", AsyncMock(return_value={})):
                with patch.object(review_module, "estimate_word_commonness", side_effect=fake_commonness):
                    css_review = await prepare_css_reviewed_item(CONFIG, {
                        "action": "Create", "word": "否则", "code": "fao", "type": "CSS",
                    })

        check("exact existing row is flagged duplicate", css_review.get("duplicate") is True)
        check("duplicate is not auto reviewable", css_review.get("autoReviewable") is False)
        check("duplicate keeps exactExisting evidence",
              bool(css_review.get("cssShortCodeReview", {}).get("exactExisting")))
        check("duplicate sets structured manual flag", read_manual_review_flag(css_review) is True)

        async def fake_css_review(config, item):
            return css_review

        with patch.object(review_module, "prepare_css_reviewed_item", side_effect=fake_css_review):
            audit = await audit_draft_items(CONFIG, [
                {"action": "Create", "word": "否则", "code": "fao", "type": "CSS"},
            ])

        check("duplicate blocks auto approval", audit.get("autoApprove") is False)
        check("duplicate is skipped, not approved", audit.get("approvedItems") == [])
        check("duplicate is recorded as skipped",
              any("词库已有" in item for item in audit.get("skippedItems", [])))
        check("duplicate message says 词库已有（跳过）",
              any("词库已有（跳过）" in issue for issue in audit.get("issues", [])))

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 22. _compact_json must always emit valid JSON
# ---------------------------------------------------------------------------

def test_compact_json_always_parses():
    print("\n🧪 _compact_json always emits valid JSON (22)")

    payload = {
        "batch": {"id": "b-1", "description": "描述" * 400},
        "pullRequests": [
            {
                "id": index,
                "word": f"测试词{index}",
                "code": "abcdef",
                "remark": "备注" * 200,
                "conflictInfo": {"impact": "冲突" * 100},
            }
            for index in range(40)
        ],
        "deterministicAudit": {
            "reviewedWords": {
                f"测试词{index}@Phrase": {
                    "pronunciations": [{
                        "codes": ["abc", "abcd"],
                        "sources": [{"source": "汉典", "url": "https://example.test/" + "x" * 200}] * 6,
                        "candidateStatuses": [
                            {"code": "abc", "phrases": [{"word": "占位词", "code": "abc"}] * 5}
                        ] * 8,
                    }],
                    "chars": [{"char": "测", "shapeCode": "vvvv"}] * 8,
                }
                for index in range(20)
            },
            "issues": ["问题" * 60] * 20,
        },
    }

    full = batch_review_module._compact_json(payload, max_chars=10 ** 9)
    check("untrimmed payload round-trips", json.loads(full) == payload)

    for limit in (28000, 8000, 2000, 500, 200, 80, 40, 20, 5):
        text = batch_review_module._compact_json(payload, max_chars=limit)
        try:
            json.loads(text)
            parsed = True
        except Exception:
            parsed = False
        check(f"max_chars={limit} still parses as JSON", parsed)
        check(f"max_chars={limit} respects the budget", len(text) <= limit or limit < 5)

    trimmed = batch_review_module._compact_json(payload, max_chars=2000)
    check("structural trim keeps a JSON object", isinstance(json.loads(trimmed), dict))


# ---------------------------------------------------------------------------
# 16. Structured flag drives manual pre-audit, LLM prose cannot flip it
# ---------------------------------------------------------------------------

def test_manual_preaudit_uses_structured_flag():
    print("\n🧪 manual pre-audit reads the structured flag (16)")

    structured_manual = {
        "word": "追速",
        "code": "fbsjuv",
        MANUAL_REVIEW_FIELD: True,
        MANUAL_REVIEW_REASON_FIELD: "常用词信号不足",
        # LLM-authored prose that merely *contains* approving Chinese text.
        "remark": "喵喵审词：本喵建议通过，该词可自动通过，读音和编码一致",
    }
    check("structured True blocks the item", "需管理员审核" in manual_preaudit_issue_for_item(structured_manual))
    check("structured True carries the reason", "常用词信号不足" in manual_preaudit_issue_for_item(structured_manual))

    structured_pass = {
        "word": "摆件",
        "code": "bhjmi",
        MANUAL_REVIEW_FIELD: False,
        # Legacy manual-review remark must lose to the structured False.
        "remark": f"喵喵审词：{MANUAL_REVIEW_PREFIX}（常用词信号不足）",
    }
    check("structured False clears the item", manual_preaudit_issue_for_item(structured_pass) == "")

    legacy_only = {
        "word": "追速",
        "code": "fbsjuv",
        "remark": f"喵喵审词：读音 zhui su；{MANUAL_REVIEW_PREFIX}（常用词信号不足）",
    }
    check("legacy remark still blocks when no structured field",
          "需管理员审核" in manual_preaudit_issue_for_item(legacy_only))

    approving_prose_only = {
        "word": "追速",
        "code": "fbsjuv",
        "remark": "本喵建议通过：读音和编码一致，可自动通过",
    }
    check("approving prose alone does not block", manual_preaudit_issue_for_item(approving_prose_only) == "")

    async def _run():
        review_module._clear_review_caches()
        prepare_mock = AsyncMock(return_value={"success": True, "autoReviewable": True, "pronunciations": []})
        with patch.object(review_module, "prepare_reviewed_word", prepare_mock):
            audit = await audit_draft_items(CONFIG, [structured_manual])
        check("structured manual item cannot auto approve", audit.get("autoApprove") is False)
        check("structured manual item skips source lookup", prepare_mock.await_count == 0)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 19. Contextual pronunciation correction rewrites derived code lists
# ---------------------------------------------------------------------------

def test_contextual_correction_drops_stale_code_lists():
    print("\n🧪 contextual correction drops stale derived codes (19)")

    response = {
        "input": "复购率",
        "codes": ["fge", "fgeu", "fgeua", "fgeuao"],
        "altCodes": ["fge", "fgeu", "fgex"],
        "candidateCodes": ["fge", "fglu"],
        "requestedCandidateCodes": ["fgeua"],
        "chars": [
            {"char": "复", "pinyin": "fù", "pinyins": ["fù"], "phoneticCode": "fj", "shapeCode": "uvoi"},
            {"char": "购", "pinyin": "gòu", "pinyins": ["gòu"], "phoneticCode": "gd", "shapeCode": "aoua"},
            {"char": "率", "pinyin": "shuài", "pinyins": ["shuài", "lǜ"], "phoneticCode": "eg", "shapeCode": "ovaa"},
        ],
    }

    result = normalize_contextual_phrase_encoding("复购率", response)
    corrected = result["codes"]

    check("codes are rebuilt from the corrected reading", corrected == ["fgl", "fglu", "fglua", "fgluao"])
    check("stale altCodes are dropped", all(code in corrected for code in result["altCodes"]))
    check("stale shuai altCodes are gone", "fge" not in result["altCodes"] and "fgex" not in result["altCodes"])
    check("candidateCodes keep only derivable codes", result["candidateCodes"] == ["fglu"])
    check("requestedCandidateCodes empties rather than keeping stale values",
          result["requestedCandidateCodes"] == [])
    check("original payload is not mutated", response["altCodes"] == ["fge", "fgeu", "fgex"])


# ---------------------------------------------------------------------------
# 21. reviewedWords is keyed by (word, type)
# ---------------------------------------------------------------------------

def test_reviewed_words_key_includes_type():
    print("\n🧪 reviewedWords key includes the type (21)")

    async def _run():
        review_module._clear_review_caches()

        async def fake_prepare(config, word):
            return {
                "success": True,
                "word": word,
                "autoReviewable": True,
                "lookupFailed": False,
                "pronunciations": [{
                    "pinyin": "xi",
                    "sources": [{"source": "汉典", "url": "https://example.test"}],
                    "codes": ["xk", "xko"],
                }],
            }

        async def fake_priority(item, review):
            return {"word": item.get("word"), "code": item.get("code"), "hasRecommendation": False, "commonness": {}}

        with patch.object(review_module, "prepare_reviewed_word", side_effect=fake_prepare):
            with patch.object(review_module, "_review_code_chain_priority", side_effect=fake_priority):
                audit = await audit_draft_items(CONFIG, [
                    {"action": "Create", "word": "喜", "code": "xk", "type": "Single"},
                    {"action": "Create", "word": "喜", "code": "xko", "type": "Phrase"},
                ])

        reviewed = audit.get("reviewedWords", {})
        check("same word with two types does not collide", len(reviewed) == 2)
        check("single key is word@Single", "喜@Single" in reviewed)
        check("phrase key is word@Phrase", "喜@Phrase" in reviewed)
        check("keys are JSON-encodable strings", all(isinstance(key, str) for key in reviewed))
        check("serialised audit survives JSON round trip",
              json.loads(json.dumps(audit, ensure_ascii=False, default=str)).get("reviewedWords", {}).keys()
              == reviewed.keys())

        # The batch-review consumer must resolve both key formats.
        entry = batch_review_module._reviewed_word_entry(audit, "喜", "Phrase")
        check("consumer resolves word@type key", isinstance(entry, dict))
        legacy = batch_review_module._reviewed_word_entry({"reviewedWords": {"喜": {"ok": True}}}, "喜", "Phrase")
        check("consumer still resolves legacy word key", legacy == {"ok": True})

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 17. The pre-submit preview item must carry an id
# ---------------------------------------------------------------------------

def test_preview_item_has_usable_id():
    print("\n🧪 preview item carries a usable id (17)")

    item = _review_tools._preview_create_item("百岁山", "bsev")
    check("preview item has an int id", isinstance(item.get("id"), int))
    check("preview id cannot collide with a real PR id", item["id"] < 0)

    extracted = batch_review_module._extract_items({"pullRequests": [item]})
    check("batch review accepts the preview item", len(extracted) == 1)
    check("batch review keeps the preview id", extracted[0]["id"] == item["id"])
    check("preview ids stay unique", _review_tools._preview_create_item("测试", "abc")["id"] != item["id"])


# ---------------------------------------------------------------------------
# 15. Degraded fallback: no fabricated sources, never auto-approvable
# ---------------------------------------------------------------------------

def test_degraded_fallback_is_never_auto_approvable():
    print("\n🧪 degraded fallback is never auto approvable (15)")

    async def _run():
        review_module._clear_review_caches()
        items = [{"id": 1, "action": "Create", "word": "测试", "code": "ceek", "type": "Phrase"}]

        async def failing_review(config, word):
            return {"success": False, "message": "读音审查失败"}

        with patch.object(batch_review_module, "prepare_reviewed_word", side_effect=failing_review):
            with patch.object(batch_review_module, "fetch_keytao_encode",
                              AsyncMock(return_value={"success": True, "codes": ["ceek", "ceeko"],
                                                      "chars": [{"char": "测", "pinyin": "ce"}]})):
                with patch.object(batch_review_module, "lookup_codes", AsyncMock(return_value={})):
                    audit = await batch_review_module._fallback_audit_with_encode(CONFIG, items, "确定性来源审查超时")

        reviewed = audit.get("reviewedWords", {}).get("测试", {})
        pronunciations = reviewed.get("pronunciations") or [{}]
        check("fallback never auto approves", audit.get("autoApprove") is False)
        check("fallback sets structured manual flag", read_manual_review_flag(audit) is True)
        check("fallback reviewed word is not auto reviewable", reviewed.get("autoReviewable") is False)
        check("fallback reviewed word is flagged manual", read_manual_review_flag(reviewed) is True)
        check("fallback emits no fabricated source", pronunciations[0].get("sources") == [])

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 20. Reorder suggestions are advisory only
# ---------------------------------------------------------------------------

def test_code_chain_reorder_is_advisory():
    print("\n🧪 code chain reorder is advisory only (20)")

    async def _run():
        review_module._clear_review_caches()
        review = {
            "success": True,
            "word": "直播间",
            "autoReviewable": True,
            "pronunciations": [{
                "codes": ["fbjui", "fbjuio"],
                "sources": [{"source": "汉典", "url": "https://example.test"}],
                "candidateStatuses": [
                    {"code": "fbjui", "occupied": True, "label": "已有「质保金」",
                     "phrases": [{"word": "质保金", "code": "fbjui", "type": "Phrase"}]},
                    {"code": "fbjuio", "occupied": False, "label": "空位", "phrases": []},
                ],
            }],
        }

        def commonness(score):
            return {
                "success": True,
                "score": score,
                "signals": {"corpus": score, "search": score, "dictionary": 0.25, "encyclopedia": 0.25},
                "evidence": {"search": ["https://example.test"]},
                "entityKnowledge": {"accepted": False},
            }

        async def wide_gap(word):
            return commonness({"直播间": 0.92, "质保金": 0.35}.get(word, 0.5))

        async def narrow_gap(word):
            # 0.42 vs 0.31 -> rounds to a 0.11 pairwise gap, below the 0.20 margin.
            return commonness({"直播间": 0.42, "质保金": 0.31}.get(word, 0.5))

        item = {"action": "Create", "word": "直播间", "code": "fbjuio", "type": "Phrase"}

        with patch.object(review_module, "estimate_word_commonness", side_effect=wide_gap):
            wide = await review_module._review_code_chain_priority(item, review)
        review_module._clear_review_caches()
        with patch.object(review_module, "estimate_word_commonness", side_effect=narrow_gap):
            narrow = await review_module._review_code_chain_priority(item, review)

        moves = wide.get("recommendedMoves", [])
        target_codes = [move.get("toCode") for move in wide.get("recommendedOrder", [])]
        check("wide commonness gap yields a recommendation", wide.get("hasRecommendation") is True)
        check("recommendation is marked advisory", wide.get("advisory") is True)
        check("each move is marked advisory", all(move.get("advisory") is True for move in moves))
        check("target codes are deduped", len(target_codes) == len(set(target_codes)))
        check("scores are rounded to 2 decimals",
              all(round(move.get("score", 0), 2) == move.get("score") for move in moves))
        check("narrow pairwise gap suppresses the recommendation", narrow.get("hasRecommendation") is False)

    asyncio.run(_run())



def test_structured_flag_survives_whitelist_wording():
    """A sealed verdict must not be reopened just because it *reads* overridable.

    The LLM-override whitelist matches Chinese substrings such as
    "没有权威读音来源" / "常用词信号不足". Those are exactly the phrases a
    structured manual-review verdict puts into its own reason text, so before
    this fix a sealed item could smuggle itself back into LLM review and be
    flipped to pass.
    """
    print("\n🧪 structured manual-review flag is terminal")
    from keytao_bot.utils.keytao_review import can_llm_override_audit_issues
    from keytao_bot.utils.keytao_batch_review import _normalize_llm_review
    from keytao_bot.utils import review_flags as rf

    # An issue phrased with whitelist wording, but sealed by a structured flag.
    sealed_issue = "「追速」加词预审已标记为需管理员审核：没有权威读音来源，且常用词信号不足"
    sealed_audit = {
        "issues": [sealed_issue],
        "structuredManualReviewIssues": [sealed_issue],
    }
    check(
        "sealed issue is not LLM-overridable despite whitelist wording",
        can_llm_override_audit_issues(sealed_audit) is False,
    )

    # The same wording, NOT sealed, stays overridable (feature preserved).
    open_audit = {"issues": ["「白鹣」没有权威读音来源，且常用词信号不足，需要管理员审核"]}
    check(
        "unsealed whitelist issue remains overridable",
        can_llm_override_audit_issues(open_audit) is True,
    )
    check(
        "blocked wording is still refused",
        can_llm_override_audit_issues({"issues": ["纯删除「x」@ab 必须由管理员审核"]}) is False,
    )

    # An LLM verdict of "pass" cannot downgrade a structurally sealed item.
    item = rf.apply_manual_review_flag(
        {"id": 7, "action": "Create", "word": "追速", "code": "vusi", "type": "Phrase"},
        True,
        "常用词信号不足",
    )
    raw = {
        "verdict": "pass",
        "items": [{
            "prId": 7,
            "status": "pass",
            "title": "本喵建议通过",
            "reasons": ["常见词，编码在候选链中"],
        }],
    }
    normalized = _normalize_llm_review(raw, [item], None, None)
    result_item = normalized["items"][0]
    check("LLM pass is clamped to manual_review", result_item["status"] == "manual_review")
    check(
        "clamped item keeps the structured flag",
        rf.read_manual_review_flag(result_item) is True,
    )
    check(
        "clamped item explains why it cannot be overridden",
        any("结构化审核" in reason for reason in result_item["reasons"]),
    )

    # An item with no structured flag is left alone.
    plain = {"id": 8, "action": "Create", "word": "白鹣", "code": "rjab", "type": "Phrase"}
    raw_plain = {
        "verdict": "pass",
        "items": [{"prId": 8, "status": "pass", "title": "本喵建议通过", "reasons": ["常见词"]}],
    }
    plain_result = _normalize_llm_review(raw_plain, [plain], None, None)["items"][0]
    check("unflagged item may still pass", plain_result["status"] == "pass")

    # An item explicitly flagged False is not clamped either.
    cleared = rf.apply_manual_review_flag(dict(plain), False, "")
    cleared_result = _normalize_llm_review(raw_plain, [cleared], None, None)["items"][0]
    check("explicitly cleared item may pass", cleared_result["status"] == "pass")



def test_manual_review_flag_survives_extract_items_round_trip():
    """The structured flag must survive every dict rebuild, not just end-to-end.

    _extract_items rebuilds each PR into a fresh dict. Dropping the flag there
    silently disarms the clamp in _normalize_llm_review, and an end-to-end test
    that happens to exercise another path would not notice.
    """
    print("\n🧪 structured flag survives _extract_items round trip")
    from keytao_bot.utils.keytao_batch_review import _extract_items, _normalize_llm_review
    from keytao_bot.utils import review_flags as rf

    batch = {
        "pullRequests": [
            {
                "id": 11, "action": "Create", "word": "追速", "code": "vusi",
                "type": "Phrase", "remark": "喵喵审词：自动审核：该词需管理员审核（常用词信号不足）",
                "needsManualReview": True, "manualReviewReason": "常用词信号不足",
            },
            {
                "id": 12, "action": "Create", "word": "白鹣", "code": "rjab",
                "type": "Phrase", "needsManualReview": False,
            },
            {   # legacy row: no structured field, only the code-generated remark
                "id": 13, "action": "Create", "word": "旧条目", "code": "abcd",
                "type": "Phrase", "remark": "喵喵审词：自动审核：该词需管理员审核（历史记录）",
            },
            {   # no verdict at all
                "id": 14, "action": "Create", "word": "普通", "code": "efgh", "type": "Phrase",
            },
        ]
    }

    items = _extract_items(batch)
    by_id = {item["id"]: item for item in items}
    check("all four items survive extraction", len(items) == 4)
    check(
        "flagged item keeps needsManualReview=True after rebuild",
        rf.read_manual_review_flag(by_id[11]) is True,
    )
    check(
        "flagged item keeps its reason after rebuild",
        rf.manual_review_reason(by_id[11]) == "常用词信号不足",
    )
    check(
        "explicit False survives as False (not dropped to None)",
        rf.read_manual_review_flag(by_id[12]) is False,
    )
    check("legacy remark survives the rebuild", "需管理员审核" in by_id[13]["remark"])
    check("unflagged item stays unflagged", rf.read_manual_review_flag(by_id[14]) is None)

    # The combined helper is what the clamp consults.
    check("structured True requires manual review", rf.item_requires_manual_review(by_id[11]) is True)
    check("structured False clears it", rf.item_requires_manual_review(by_id[12]) is False)
    check("legacy remark still seals the item", rf.item_requires_manual_review(by_id[13]) is True)
    check("no verdict means no seal", rf.item_requires_manual_review(by_id[14]) is False)

    # End-to-end through the extracted items: the LLM says pass for everything.
    raw = {
        "verdict": "pass",
        "items": [
            {"prId": pid, "status": "pass", "title": "本喵建议通过", "reasons": ["常见词"]}
            for pid in (11, 12, 13, 14)
        ],
    }
    normalized = _normalize_llm_review(raw, items, None, None)
    status_by_id = {item["prId"]: item["status"] for item in normalized["items"]}
    check("sealed item is clamped after round trip", status_by_id[11] == "manual_review")
    check("explicitly cleared item may pass", status_by_id[12] == "pass")
    check("legacy-remark item is clamped after round trip", status_by_id[13] == "manual_review")
    check("unflagged item may pass", status_by_id[14] == "pass")



def test_phrase_branch_detects_duplicates():
    """Duplicate detection must cover ordinary Phrase items, not only CSS."""
    print("\n🧪 duplicate detection covers the Phrase branch")
    from keytao_bot.utils.keytao_review import _has_exact_existing_phrase, DUPLICATE_REASON

    existing = [
        {"word": "测试", "code": "abcd", "type": "Phrase"},
        {"word": "测试", "code": "zzzz", "type": "Phrase"},
        {"word": "测试", "code": "abcd", "type": "CSS"},
        # A by-word batch lookup can carry rows for OTHER words.
        {"word": "别的词", "code": "efgh", "type": "Phrase"},
    ]
    check("exact word@code@type match is a duplicate",
          _has_exact_existing_phrase(existing, "测试", "abcd", "Phrase") is True)
    check("code casing is normalised",
          _has_exact_existing_phrase(existing, "测试", "ABCD", "Phrase") is True)
    check("different code is not a duplicate",
          _has_exact_existing_phrase(existing, "测试", "efgh", "Phrase") is False)
    check("same code under another type is not a duplicate",
          _has_exact_existing_phrase(existing, "测试", "zzzz", "CSS") is False)
    # The regression this guards: another word's row must never mark a brand-new
    # entry as a duplicate and get it silently dropped.
    check("another word's row with the same code is not a duplicate",
          _has_exact_existing_phrase(existing, "新词", "efgh", "Phrase") is False)
    check("empty existing list is not a duplicate",
          _has_exact_existing_phrase([], "测试", "abcd", "Phrase") is False)
    check("non-list existing is handled",
          _has_exact_existing_phrase(None, "测试", "abcd", "Phrase") is False)
    check("empty word is not a duplicate",
          _has_exact_existing_phrase(existing, "", "abcd", "Phrase") is False)
    check("duplicate reason is the shared constant", "词库已有" in DUPLICATE_REASON)



def test_audit_seal_survives_shard_compaction():
    """Sealing fields must survive _compact_audit_for_items and clamp the verdict."""
    print("\n🧪 audit seal survives shard compaction")
    from keytao_bot.utils.keytao_batch_review import (
        _compact_audit_for_items, _normalize_llm_review,
    )
    from keytao_bot.utils import review_flags as rf

    audit = rf.apply_manual_review_flag({
        "success": True,
        "verdict": "needs_admin",
        "autoApprove": False,
        "summary": "存在需管理员确认项",
        "issues": ["「追速」词库占位查询失败，无法确认编码空位，需要管理员审核"],
        "structuredManualReviewIssues": ["「追速」词库占位查询失败，无法确认编码空位，需要管理员审核"],
        "lookupFailed": True,
        "reviewedWords": {},
    }, True, "词库占位查询失败")

    items = [{"id": 21, "action": "Create", "word": "追速", "code": "vusi", "type": "Phrase"}]
    compact = _compact_audit_for_items(audit, items)

    check("needsManualReview survives compaction",
          rf.read_manual_review_flag(compact) is True)
    check("manualReviewReason survives compaction",
          rf.manual_review_reason(compact) == "词库占位查询失败")
    check("structuredManualReviewIssues survives compaction",
          bool(compact.get("structuredManualReviewIssues")))
    check("lookupFailed survives compaction", compact.get("lookupFailed") is True)

    # Even with every item unflagged, an audit-level seal blocks auto-approval.
    raw = {
        "verdict": "pass",
        "items": [{"prId": 21, "status": "pass", "title": "本喵建议通过", "reasons": ["常见词"]}],
    }
    normalized = _normalize_llm_review(raw, items, None, compact)
    check("audit-level seal clamps the batch verdict",
          normalized["verdict"] == "manual_review")
    check("audit seal is reported structurally", normalized.get("auditSealed") is True)

    # Without a seal the same input may still pass.
    clean = {"success": True, "verdict": "pass", "autoApprove": True, "issues": []}
    clean_norm = _normalize_llm_review(raw, items, None, clean)
    check("unsealed audit still allows pass", clean_norm["verdict"] == "pass")
    check("unsealed audit reports no seal", clean_norm.get("auditSealed") is False)


def test_pending_add_word_carries_structured_verdict():
    """The pending-add chain must not round-trip the verdict through LLM prose."""
    print("\n🧪 pending-add chain carries the structured verdict")
    import json as _json
    from keytao_bot.harness.state import PendingAddWord
    import keytao_bot.plugins.openai_chat as chat

    state = PendingAddWord(word="追速", recommended_code="vusi", candidates=[("vusi", False)])
    check("verdict defaults to unknown, not False", state.needs_manual_review is None)

    # The tool result -- not the model's wording -- populates the verdict.
    chat._reviewed_add_verdicts.clear()
    chat._record_reviewed_add_verdict(
        "keytao_prepare_reviewed_add",
        {"word": "追速"},
        _json.dumps({"word": "追速", "needsManualReview": True,
                     "manualReviewReason": "常用词信号不足"}),
    )
    flag, reason = chat._take_reviewed_add_verdict("追速")
    check("verdict is recorded from the tool result", flag is True)
    check("reason is recorded from the tool result", reason == "常用词信号不足")
    check("unknown words stay unknown", chat._take_reviewed_add_verdict("没查过")[0] is None)

    # An unrelated tool must not populate anything.
    chat._reviewed_add_verdicts.clear()
    chat._record_reviewed_add_verdict(
        "keytao_encode", {"word": "追速"}, _json.dumps({"word": "追速", "needsManualReview": True}),
    )
    check("only the review tool records verdicts",
          chat._take_reviewed_add_verdict("追速")[0] is None)

    # The parsed pending state picks the verdict up, regardless of the prose.
    chat._reviewed_add_verdicts.clear()
    chat._record_reviewed_add_verdict(
        "keytao_prepare_reviewed_add", {"word": "追速"},
        _json.dumps({"word": "追速", "needsManualReview": True, "manualReviewReason": "常用词信号不足"}),
    )
    # Note the response text claims auto-approval; the structured verdict wins.
    response = (
        "是否以编码 vusi 将「追速」加入草稿？\n"
        "1. vusi - ✅ 空位\n"
        "审词：读音 zhui su；来源 暂无权威页；自动审核：该词可自动通过（常见词）\n"
    )
    pending = chat._parse_pending_add_word(response)
    check("pending state was parsed", pending is not None)
    check("structured verdict overrides approving prose",
          pending.needs_manual_review is True)
    check("structured reason is carried", pending.manual_review_reason == "常用词信号不足")



def test_draft_snapshot_item_keeps_flag_through_enrichment():
    """keytao-next now persists and returns needsManualReview on draft items.

    That makes the cross-repo round trip the PRIMARY path (no longer a legacy
    remark fallback), so the bot-side normalisation of those items must not drop
    the field -- the same class of bug already found in _extract_items and
    _compact_audit_for_items.
    """
    print("\n🧪 draft snapshot items keep the structured flag")
    import importlib.util, os
    from keytao_bot.utils import review_flags as rf
    from keytao_bot.utils.keytao_review import manual_preaudit_issue_for_item

    spec = importlib.util.spec_from_file_location(
        "draft_tools_flag_probe",
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "keytao_bot", "skills", "keytao-draft", "tools.py"),
    )
    draft_tools = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(draft_tools)

    # Exactly what GET /api/bot/batches/latest-draft/items now returns.
    snapshot_item = {
        "id": 41, "action": "Create", "word": "追速", "code": "vusi", "type": "Phrase",
        "needsManualReview": True, "manualReviewReason": "常用词信号不足",
        "remark": "喵喵审词：读音 zhui su",   # no legacy marker in the remark
    }
    enriched = draft_tools.enrich_pr_item_labels(snapshot_item)

    check("enrichment preserves needsManualReview",
          rf.read_manual_review_flag(enriched) is True)
    check("enrichment preserves manualReviewReason",
          rf.manual_review_reason(enriched) == "常用词信号不足")
    check("enrichment still adds its display labels", bool(enriched.get("action_label")))
    check("enrichment does not mutate the source item", "action_label" not in snapshot_item)

    # The gate seals it from the structured field alone, with no marker in the remark.
    issue = manual_preaudit_issue_for_item(enriched)
    check("gate seals the item without any remark marker", bool(issue))
    check("gate surfaces the structured reason", "常用词信号不足" in issue)

    # An item returned with an explicit False is not sealed.
    cleared = draft_tools.enrich_pr_item_labels({
        "id": 42, "action": "Create", "word": "白鹣", "code": "rjab", "type": "Phrase",
        "needsManualReview": False,
    })
    check("explicit False survives enrichment",
          rf.read_manual_review_flag(cleared) is False)
    check("explicitly cleared item is not sealed",
          not manual_preaudit_issue_for_item(cleared))



def test_unknown_verdict_never_becomes_false():
    """unknown != False.

    An earlier revision always stamped the field so it was never absent, which
    turned "we never obtained a verdict" into a positive "reviewed and fine"
    claim that persisted to keytao-next. Every way of losing the verdict (TTL
    expiry, cache eviction, process restart, resuming from history) landed
    there, making the failure mode fail-open.
    """
    print("\n🧪 unknown verdict never becomes False")
    import importlib.util, os
    from keytao_bot.utils import review_flags as rf

    spec = importlib.util.spec_from_file_location(
        "draft_tools_unknown_probe",
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "keytao_bot", "skills", "keytao-draft", "tools.py"),
    )
    draft_tools = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(draft_tools)
    stamp = draft_tools._stamp_item_review_flag

    unknown = {"word": "x", "code": "ccc"}
    stamp(unknown)
    check("unknown stays absent, not False",
          rf.MANUAL_REVIEW_FIELD not in unknown)
    check("unknown reads back as None", rf.read_manual_review_flag(unknown) is None)

    sealed = {"word": "y", "code": "ddd", "needsManualReview": True}
    stamp(sealed)
    check("True is preserved", rf.read_manual_review_flag(sealed) is True)

    cleared = {"word": "z", "code": "eee", "needsManualReview": False}
    stamp(cleared)
    check("an affirmative False is preserved", rf.read_manual_review_flag(cleared) is False)

    legacy = {"word": "w", "code": "fff",
              "remark": "喵喵审词：自动审核：该词需管理员审核（常用词信号不足）"}
    stamp(legacy)
    check("legacy remark marker still seals", rf.read_manual_review_flag(legacy) is True)

    # Deterministic code validation alone must not manufacture a False.
    validated = {"word": "v", "code": "ggg"}
    stamp(validated, {"success": True, "word": "v", "code": "ggg"})
    check("passing code validation does not assert 'no review needed'",
          rf.MANUAL_REVIEW_FIELD not in validated)

    # But a validation that demands review does seal.
    flagged = {"word": "u", "code": "hhh"}
    stamp(flagged, {"success": True, "needsManualReview": True,
                    "manualReviewReason": "CSS 无确定性规则"})
    check("validation-demanded review seals", rf.read_manual_review_flag(flagged) is True)


def test_every_add_branch_carries_the_verdict():
    """All confirmation branches must forward the structured verdict.

    A probe caught the real confirmation payload as {'word','code'} only -- the
    verdict never reached keytao_create_phrase, which then recorded False.
    """
    print("\n🧪 every add branch carries the verdict")
    import inspect
    import keytao_bot.plugins.openai_chat as chat
    from keytao_bot.harness.state import PendingAddWord

    sealed_state = PendingAddWord(
        word="sealed", recommended_code="aaaa", candidates=[("aaaa", True)],
        code_remarks={"aaaa": "喵喵审词：读音 x"},
        needs_manual_review=True, manual_review_reason="常用词信号不足",
    )
    args = chat._create_phrase_args(sealed_state, "aaaa")
    check("confirm args carry the verdict", args.get("needs_manual_review") is True)
    check("confirm args keep the remark", "remark" in args)
    check("confirm args keep word and code",
          args["word"] == "sealed" and args["code"] == "aaaa")

    unknown_state = PendingAddWord(
        word="plain", recommended_code="bbbb", candidates=[("bbbb", False)])
    unknown_args = chat._create_phrase_args(unknown_state, "bbbb")
    check("unknown verdict is omitted, not sent as False",
          "needs_manual_review" not in unknown_args)

    cleared_state = PendingAddWord(
        word="ok", recommended_code="cccc", candidates=[("cccc", False)],
        needs_manual_review=False)
    check("an affirmative False is forwarded",
          chat._create_phrase_args(cleared_state, "cccc").get("needs_manual_review") is False)

    # No confirmation branch may hand-roll the argument dict any more.
    source = inspect.getsource(chat)
    check("no branch rebuilds bare word/code create args",
          'args={"word": state.word, "code": target_code}' not in source)
    # Both add-and-submit entry points accept the verdict.
    for fn in (chat._perform_add_to_draft_and_submit, chat._execute_add_to_draft,
               chat._execute_add_to_draft_and_submit):
        check(f"{fn.__name__} accepts needs_manual_review",
              "needs_manual_review" in inspect.signature(fn).parameters)


def test_batch_add_uses_structured_verdict_not_prose():
    """The multi-word path must read the verdict table, not the model's wording."""
    print("\n🧪 batch add uses the structured verdict")
    import json as _json
    import keytao_bot.plugins.openai_chat as chat

    chat._reviewed_add_verdicts.clear()
    chat._record_reviewed_add_verdict(
        "keytao_prepare_reviewed_add", {"word": "sealed"},
        _json.dumps({"word": "sealed", "needsManualReview": True,
                     "manualReviewReason": "常用词信号不足"}),
    )
    # The prose claims both words can be auto-approved.
    response = (
        "本喵建议把「sealed」→aaaa 和「other」→bbbb 一起加入草稿，好吗？\n"
        "「sealed」\n  审词：读音 x；自动审核：该词可自动通过（常见词）\n"
        "「other」\n  审词：读音 y；自动审核：该词可自动通过（常见词）\n"
    )
    pending = chat._parse_pending_batch_add(response)
    check("batch pending state was parsed", pending is not None)
    items = {item["word"]: item for item in pending.args["items"]}
    check("both words parsed", set(items) == {"sealed", "other"})
    check("recorded verdict wins over approving prose",
          items["sealed"].get("needsManualReview") is True)
    check("its reason is carried",
          items["sealed"].get("manualReviewReason") == "常用词信号不足")
    check("a word with no recorded verdict stays unknown, not False",
          "needsManualReview" not in items["other"])


def test_candidate_commonness_wiring_and_timeout():
    """Candidate preparation compares only the first two occupants and degrades."""
    print("\n🧪 candidate commonness wiring and timeout")

    review = {
        "success": True,
        "word": "射覆",
        "recommendedCode": "eefju",
        "pronunciations": [{
            "recommendedCode": "eefju",
            "candidateStatuses": [
                {
                    "code": "eefj",
                    "occupied": True,
                    "words": ["慑服"],
                    "phrases": [{"word": "慑服", "code": "eefj", "type": "Phrase"}],
                },
                {
                    "code": "eefji",
                    "occupied": True,
                    "words": ["设伏"],
                    "phrases": [{"word": "设伏", "code": "eefji", "type": "Phrase"}],
                },
                {
                    "code": "eefjk",
                    "occupied": True,
                    "words": ["社福"],
                    "phrases": [{"word": "社福", "code": "eefjk", "type": "Phrase"}],
                },
                {"code": "eefju", "occupied": False, "words": [], "phrases": []},
            ],
        }],
    }

    async def _run():
        calls = []

        async def compare(front_word, behind_word):
            calls.append((front_word, behind_word))
            return {
                "success": True,
                "verdict": (
                    "front_more_common"
                    if behind_word == "慑服"
                    else "behind_more_common"
                ),
                "summary": "comparison",
            }

        with patch.object(review_module, "compare_word_commonness", side_effect=compare):
            assessments = await review_module.assess_candidate_chain_commonness(review)
        check(
            "comparator receives the first two new-word/occupant pairs",
            calls == [("射覆", "慑服"), ("射覆", "设伏")],
        )
        check("comparison work is capped at two occupants", len(assessments) == 2)
        check(
            "front verdict recommends the occupied code",
            assessments[0]["newCode"] == "eefj",
        )
        check(
            "behind verdict keeps the free code",
            assessments[1]["newCode"] == "eefju",
        )

        started = asyncio.Event()

        async def never_finishes(*_args):
            started.set()
            await asyncio.sleep(60)

        with patch.object(
            review_module,
            "compare_word_commonness",
            side_effect=never_finishes,
        ):
            timed_out = await review_module.assess_candidate_chain_commonness(
                review,
                timeout=0.01,
            )
        check("timeout branch was exercised", started.is_set())
        check(
            "timeout degrades every pair to insufficient evidence",
            len(timed_out) == 2
            and all(item["verdict"] == "not_enough_evidence" for item in timed_out)
            and all(item["degradation"] == "timeout" for item in timed_out),
        )

        ordering = [{
            "verdict": "front_more_common",
            "newWord": "射覆",
            "occupantWord": "慑服",
            "occupantCode": "eefj",
            "freeCode": "eefju",
            "newCode": "eefj",
        }]
        prepared = {**review, "needsManualReview": False}
        with (
            patch.object(
                _review_tools,
                "prepare_reviewed_word",
                new=AsyncMock(return_value=prepared),
            ),
            patch.object(
                _review_tools,
                "_build_pre_submit_audit",
                new=AsyncMock(return_value={"autoApprove": True, "summary": "pass"}),
            ),
            patch.object(
                _review_tools,
                "assess_candidate_chain_commonness",
                new=AsyncMock(return_value=ordering),
            ) as wired,
        ):
            tool_result = await _review_tools.keytao_prepare_reviewed_add("射覆")
        check("review tool invokes the shared assessment helper", wired.await_count == 1)
        check(
            "review tool returns the structured ordering snapshot",
            tool_result.get("candidateOrderingAssessments") == ordering,
        )

    asyncio.run(_run())


def test_audit_budget_nesting_and_timeout_retains_review():
    print("\n🧪 audit budget nesting and partial-result retention")

    encode_retry_backoff = sum(
        0.5 * (2 ** retry_index)
        for retry_index in range(review_module.KEYTAO_ENCODE_MAX_ATTEMPTS - 1)
    )
    encode_worst_case = (
        review_module.KEYTAO_ENCODE_REQUEST_TIMEOUT
        * review_module.KEYTAO_ENCODE_MAX_ATTEMPTS
        + encode_retry_backoff
    )
    lookup_retry_backoff = sum(
        0.5 * (2 ** retry_index)
        for retry_index in range(review_module.REVIEW_LOOKUP_MAX_ATTEMPTS - 1)
    )
    lookup_worst_case = (
        review_module.REVIEW_LOOKUP_REQUEST_TIMEOUT
        * review_module.REVIEW_LOOKUP_MAX_ATTEMPTS
        + lookup_retry_backoff
    )
    ordinary_review_worst_case = (
        max(
            review_module.PRONUNCIATION_EVIDENCE_TIMEOUT,
            encode_worst_case,
            lookup_worst_case,
        )
        + lookup_worst_case
    )
    check(
        "ordinary review children fit the review-stage budget",
        ordinary_review_worst_case <= review_module.AUDIT_REVIEW_STAGE_TIMEOUT,
    )
    check(
        "worst sequential audit chain fits below its parent budget",
        review_module.AUDIT_WORST_CASE_SEQUENTIAL_SECONDS
        < review_module.AUDIT_ITEM_TIMEOUT,
    )
    check(
        "candidate commonness stays within the commonness-stage budget",
        review_module.CANDIDATE_COMMONNESS_TIMEOUT_SECONDS
        <= review_module.AUDIT_COMMONNESS_STAGE_TIMEOUT,
    )

    async def _run():
        resolved_review = {
            "success": True,
            "word": "石蒜",
            "existing": [],
            "autoReviewable": True,
            "pronunciations": [{
                "pinyin": "shi suan",
                "normalized": ["shi", "suan"],
                "codes": ["ekso"],
                "sources": [{
                    "source": "汉典",
                    "url": "https://example.test/shisuan",
                    "category": "dictionary",
                    "trust": 5,
                }],
                "candidateStatuses": [{
                    "code": "ekso",
                    "occupied": False,
                    "phrases": [],
                }],
            }],
        }

        async def unfinished_priority(_item, _review):
            await asyncio.Event().wait()

        with (
            patch.object(
                review_module,
                "prepare_reviewed_word",
                new=AsyncMock(return_value=resolved_review),
            ),
            patch.object(
                review_module,
                "_review_code_chain_priority",
                side_effect=unfinished_priority,
            ),
            patch.object(review_module, "AUDIT_ITEM_TIMEOUT", 0.02),
            patch.object(review_module.logger, "info") as info_log,
        ):
            audit = await audit_draft_items(CONFIG, [{
                "id": 1,
                "action": "Create",
                "word": "石蒜",
                "code": "ekso",
                "type": "Phrase",
            }])

        retained = audit.get("reviewedWords", {}).get("石蒜@Phrase", {})
        normalized = [
            item.get("normalized")
            for item in retained.get("pronunciations", [])
            if isinstance(item, dict)
        ]
        check(
            "resolved pronunciation survives the item guillotine",
            ["shi", "suan"] in normalized,
        )
        check(
            "timeout names only the unfinished priority stage",
            any(
                "编码链优先级评估" in issue and "超时" in issue
                for issue in audit.get("issues", [])
            )
            and all("审词超过" not in issue for issue in audit.get("issues", [])),
        )
        audit_log_lines = [
            str(call.args[0])
            for call in info_log.call_args_list
            if call.args and str(call.args[0]).startswith("[audit_item]")
        ]
        check(
            "audit emits one bounded INFO line with per-stage timing",
            len(audit_log_lines) == 1
            and "word=石蒜" in audit_log_lines[0]
            and "status=timeout:priority" in audit_log_lines[0]
            and "review=" in audit_log_lines[0]
            and "priority=" in audit_log_lines[0]
            and "\n" not in audit_log_lines[0],
        )

    asyncio.run(_run())


def main():
    test_s14_wrong_entry_pronunciation_never_reaches_candidates()
    test_reviewed_add_chi_xi_no_authoritative_entry_or_web_uses_verified_own_characters()
    test_pronunciation_word_binding_window_and_exact_direct_entry()
    test_pronunciation_groups_require_known_character_readings()
    test_lookup_failure_forces_manual_review()
    test_exact_existing_is_duplicate_not_approval()
    test_compact_json_always_parses()
    test_manual_preaudit_uses_structured_flag()
    test_contextual_correction_drops_stale_code_lists()
    test_reviewed_words_key_includes_type()
    test_preview_item_has_usable_id()
    test_degraded_fallback_is_never_auto_approvable()
    test_code_chain_reorder_is_advisory()
    test_structured_flag_survives_whitelist_wording()
    test_manual_review_flag_survives_extract_items_round_trip()
    test_phrase_branch_detects_duplicates()
    test_audit_seal_survives_shard_compaction()
    test_pending_add_word_carries_structured_verdict()
    test_draft_snapshot_item_keeps_flag_through_enrichment()
    test_unknown_verdict_never_becomes_false()
    test_every_add_branch_carries_the_verdict()
    test_batch_add_uses_structured_verdict_not_prose()
    test_candidate_commonness_wiring_and_timeout()
    test_audit_budget_nesting_and_timeout_retains_review()

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{passed + failed} passed" + (f", {failed} failed" if failed else ""))
    if failed:
        print("❌ SOME TESTS FAILED")
        return 1
    print("✅ ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
