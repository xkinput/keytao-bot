#!/usr/bin/env python3
"""Focused regression tests for the KeyTao auto-approval review gate.

Self-contained: stubs nonebot/httpx/openai the same way test_state_machine.py
does, so it runs without a NoneBot runtime.

    uv run python test_review_gate.py
"""
import asyncio
import gzip
import hashlib
import importlib.util
import json
import os
import sqlite3
import sys
import tempfile
import types
from contextlib import contextmanager
from pathlib import Path
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
from keytao_bot.utils import pinyin_reference as pinyin_reference_module  # noqa: E402
from keytao_bot.utils import pinyin_reference_build as pinyin_reference_build_module  # noqa: E402
from keytao_bot.utils.pinyin_reference_build import (  # noqa: E402
    build_reference_database,
)
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
    REVIEW_VERDICT_SITE_POLICIES,
    ReviewDisposition,
    read_manual_review_flag,
    read_review_disposition,
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


def test_review_disposition_registry():
    expected = {
        "semantic_context_common_word": ReviewDisposition.PASS,
        "empty_word": ReviewDisposition.BLOCK,
        "pronunciation_unresolved": ReviewDisposition.BLOCK,
        "code_unresolved": ReviewDisposition.BLOCK,
        "lookup_unavailable": ReviewDisposition.BLOCK,
        "invalid_code": ReviewDisposition.BLOCK,
        "injection_shaped_input": ReviewDisposition.BLOCK,
        "missing_authoritative_page": ReviewDisposition.SEAL,
        "pronunciation_lookup_incomplete": ReviewDisposition.SEAL,
        "entity_context_reading": ReviewDisposition.SEAL,
        "unvalidated_type": ReviewDisposition.SEAL,
        "pre_submit_judgement": ReviewDisposition.SEAL,
        "pre_submit_review_unavailable": ReviewDisposition.SEAL,
        "duplicate_formation": ReviewDisposition.SEAL,
        "code_chain_priority": ReviewDisposition.SEAL,
    }
    check(
        "every PASS/BLOCK/SEAL review verdict site is declared centrally",
        REVIEW_VERDICT_SITE_POLICIES == expected,
    )


def _semantic_context_review(
    word,
    code,
    *,
    complete=True,
    meaning="表示一个含义和读音都明确的现代汉语组合",
    confidence=0.96,
    common_transparent=True,
    context_method="meaning_backed_semantic_pronunciation",
    chosen_pinyins=None,
    known_readings=None,
    extra_pronunciations=None,
):
    chosen_pinyins = list(chosen_pinyins or ["chan", "ji"])
    known_readings = list(known_readings or [[chosen_pinyins[0]], [chosen_pinyins[1]]])
    pronunciation = {
        "pinyin": " ".join(chosen_pinyins),
        "normalized": list(chosen_pinyins),
        "codes": [code],
        "sources": [],
        "semanticPronunciation": True,
        "requiresManualReview": True,
        "readingEvidenceKind": "own_character_semantic",
        "contextPronunciation": {
            "entityType": "transparent_compound",
            "label": "常用透明组合",
            "confidence": confidence,
            "description": meaning,
            "method": context_method,
            "commonTransparent": common_transparent,
        },
        "characterReadings": [
            {
                "char": char,
                "chosenPinyin": chosen,
                "knownReadings": list(readings),
                "lookupStatus": "found",
            }
            for char, chosen, readings in zip(word, chosen_pinyins, known_readings)
        ],
    }
    return {
        "success": True,
        "word": word,
        "autoReviewable": False,
        "lookupFailed": False,
        "pronunciationEvidenceComplete": complete,
        "requiresManualPronunciationReview": True,
        "existing": [],
        "pronunciations": [pronunciation, *(extra_pronunciations or [])],
        "needsManualReview": True,
        "manualReviewReason": (
            "读音由有明确含义支撑的整词语境判定，"
            "但缺少权威整词读音来源"
            if complete
            else "本次权威来源查询未完成"
        ),
        "reviewDisposition": "SEAL",
        "reviewVerdictSite": (
            "entity_context_reading"
            if complete
            else "pronunciation_lookup_incomplete"
        ),
    }


def _reference_row(
    word,
    *,
    frequency=None,
    dictionary_presence=0,
    available=True,
):
    return {
        "available": available,
        "attested": frequency is not None or dictionary_presence > 0,
        "word": word,
        "corpusFrequency": frequency,
        "partOfSpeech": "n" if frequency is not None else None,
        "dictionaryPresenceCount": dictionary_presence,
    }


def test_semantic_context_auto_pass_corpus_and_mutation_matrix():
    """Every predicate leg has an observable corpus case that kills its mutant."""
    print("\n🧪 semantic context auto-pass corpus and mutation matrix")

    async def audit_case(review, references):
        word = review["word"]

        def query_reference(value):
            return references.get(value, _reference_row(value))

        with (
            patch.object(
                review_module,
                "prepare_reviewed_word",
                AsyncMock(return_value=review),
            ),
            patch.object(
                review_module,
                "_query_commonness_reference",
                side_effect=query_reference,
            ),
            patch.object(
                review_module,
                "_review_code_chain_priority",
                AsyncMock(return_value={
                    "word": word,
                    "code": review["pronunciations"][0]["codes"][0],
                    "hasRecommendation": False,
                    "commonness": {},
                }),
            ),
        ):
            return await audit_draft_items(CONFIG, [{
                "action": "Create",
                "word": word,
                "code": review["pronunciations"][0]["codes"][0],
                "type": "Phrase",
            }])

    async def _run():
        passing_cases = [
            (
                "frequency-only leg",
                _semantic_context_review("粮棉", "llmm"),
                {
                    "粮棉": _reference_row("粮棉", frequency=55),
                    "粮": _reference_row("粮", frequency=1),
                    "棉": _reference_row("棉", frequency=1),
                },
                "corpus_frequency",
                "jieba 词频 55（阈值 10）",
            ),
            (
                "common-chars plus LLM leg",
                _semantic_context_review("产季", "ijjk"),
                {
                    "产季": _reference_row("产季"),
                    "产": _reference_row("产", frequency=6838),
                    "季": _reference_row("季", frequency=1619),
                },
                "common_characters_and_llm",
                "产 6838、季 1619",
            ),
        ]
        for name, review, references, route, copy_marker in passing_cases:
            audit = await audit_case(review, references)
            reviewed = audit.get("reviewedWords", {}).get(
                f"{review['word']}@Phrase",
                {},
            )
            assessment = reviewed.get("semanticContextAutoPass") or {}
            check(
                f"{name} auto-passes",
                audit.get("autoApprove") is True
                and audit.get("verdict") == "pass"
                and read_manual_review_flag(audit) is False,
            )
            check(
                f"{name} declares registered PASS",
                read_review_disposition(audit) is ReviewDisposition.PASS
                and audit.get("reviewVerdictSite")
                == "semantic_context_common_word",
            )
            check(
                f"{name} records its unique non-obscurity route",
                assessment.get("nonObscurity", {}).get("route") == route,
            )
            check(
                f"{name} renders honest evidence copy",
                copy_marker in str(assessment.get("basisLine") or "")
                and "该词可自动通过" in str(assessment.get("basisLine") or ""),
            )

        ambiguous_extra = {
            **_semantic_context_review(
                "重行",
                "isxk",
                chosen_pinyins=["zhong", "xing"],
                known_readings=[["zhong", "chong"], ["xing", "hang"]],
            )["pronunciations"][0],
            "pinyin": "chong hang",
            "normalized": ["chong", "hang"],
            "characterReadings": [
                {
                    "char": "重",
                    "chosenPinyin": "chong",
                    "knownReadings": ["zhong", "chong"],
                    "lookupStatus": "found",
                },
                {
                    "char": "行",
                    "chosenPinyin": "hang",
                    "knownReadings": ["xing", "hang"],
                    "lookupStatus": "found",
                },
            ],
        }
        mutation_cases = [
            (
                "lookup-completed condition",
                _semantic_context_review("产季", "ijjk", complete=False),
                "lookupCompleted",
            ),
            (
                "known-character-reading condition",
                _semantic_context_review(
                    "产季",
                    "ijjk",
                    known_readings=[["chan"], ["qi"]],
                ),
                "knownCharacterReadings",
            ),
            (
                "single-unambiguous-reading condition",
                _semantic_context_review(
                    "重行",
                    "isxk",
                    chosen_pinyins=["zhong", "xing"],
                    known_readings=[["zhong", "chong"], ["xing", "hang"]],
                    extra_pronunciations=[ambiguous_extra],
                ),
                "singleSemanticPronunciation",
            ),
            (
                "concrete-meaning condition",
                _semantic_context_review("产季", "ijjk", meaning="该词的意思"),
                "concreteMeaning",
            ),
            (
                "semantic-confidence condition",
                _semantic_context_review("产季", "ijjk", confidence=0.74),
                "meaningConfidence",
            ),
            (
                "multi-reading meaning-binding condition",
                _semantic_context_review(
                    "重行",
                    "isxk",
                    chosen_pinyins=["zhong", "xing"],
                    known_readings=[["zhong", "chong"], ["xing", "hang"]],
                    context_method="unbacked_context",
                ),
                "multiReadingMeaningBacked",
            ),
        ]
        common_references = {
            "产季": _reference_row("产季", dictionary_presence=1),
            "产": _reference_row("产", frequency=6838),
            "季": _reference_row("季", frequency=1619),
            "重行": _reference_row("重行", dictionary_presence=1),
            "重": _reference_row("重", frequency=2000),
            "行": _reference_row("行", frequency=3000),
        }
        for name, review, failed_check in mutation_cases:
            audit = await audit_case(review, common_references)
            reviewed = audit.get("reviewedWords", {}).get(
                f"{review['word']}@Phrase",
                {},
            )
            assessment = reviewed.get("semanticContextAutoPass") or {}
            issue = next(iter(audit.get("issues") or []), "")
            expected_issue = (
                "本次权威来源查询未完成"
                if failed_check == "lookupCompleted"
                else (
                    f"「{review['word']}」读音由有明确含义支撑的整词语境判定，"
                    "但缺少权威整词读音来源，需要管理员审核"
                )
            )
            check(
                f"{name} mutant is killed by a sealed result",
                audit.get("autoApprove") is False
                and expected_issue in issue
                and any(
                    expected_issue in str(item)
                    for item in (
                        audit.get("structuredManualReviewIssues") or []
                    )
                ),
            )
            check(
                f"{name} has would-have-auto-passed proof",
                assessment.get("failedChecks") == [failed_check]
                and assessment.get("wouldPassWithout") == failed_check,
            )

        non_obscurity_controls = [
            (
                "below-threshold frequency",
                _semantic_context_review("低频", "djpb"),
                {
                    "低频": _reference_row("低频", frequency=9),
                    "低": _reference_row("低", frequency=999),
                    "频": _reference_row("频", frequency=4000),
                },
            ),
            (
                "obscure character",
                _semantic_context_review("龘季", "djjk"),
                {
                    "龘季": _reference_row("龘季"),
                    "龘": _reference_row("龘"),
                    "季": _reference_row("季", frequency=1619),
                },
            ),
            (
                "rare common-character combination",
                _semantic_context_review(
                    "季产",
                    "jkij",
                    common_transparent=False,
                ),
                {
                    "季产": _reference_row("季产"),
                    "季": _reference_row("季", frequency=1619),
                    "产": _reference_row("产", frequency=6838),
                },
            ),
        ]
        for name, review, references in non_obscurity_controls:
            audit = await audit_case(review, references)
            reviewed = audit.get("reviewedWords", {}).get(
                f"{review['word']}@Phrase",
                {},
            )
            assessment = reviewed.get("semanticContextAutoPass") or {}
            check(
                f"{name} remains sealed",
                audit.get("autoApprove") is False
                and read_manual_review_flag(audit) is True
                and bool(audit.get("structuredManualReviewIssues")),
            )
            check(
                f"{name} kills removal of the not-obscure condition",
                assessment.get("failedChecks") == ["notObscure"]
                and assessment.get("wouldPassWithout") == "notObscure",
            )

    asyncio.run(_run())


def test_rejected_offline_whole_word_reading_cannot_auto_pass_by_dictionary_presence():
    """A rejected offline reading must not become positive evidence in this lane."""
    print("\n🧪 rejected offline whole-word reading stays sealed")

    async def _run():
        word = "行长"
        code = "xzab"
        encode_data = {
            "success": True,
            "word": word,
            "codes": [code],
            "pronunciationSource": "zdic-character-default",
            "standardPronunciationStatus": "absent",
            "semanticPronunciationNeeded": False,
            "phrasePinyins": ["xíng", "zhǎng"],
            "contextPhrasePinyins": ["xíng", "zhǎng"],
            "chars": [
                {
                    "char": "行",
                    "pinyin": "xíng",
                    "pinyins": ["xíng"],
                    "pronunciationLookupStatus": "found",
                },
                {
                    "char": "长",
                    "pinyin": "zhǎng",
                    "pinyins": ["zhǎng"],
                    "pronunciationLookupStatus": "found",
                },
            ],
        }
        evidence = {
            "success": True,
            "word": word,
            "groups": [{
                "pinyin": "háng zhǎng",
                "normalized": ["hang", "zhang"],
                "sources": [
                    {
                        "source": "CC-CEDICT",
                        "url": "",
                        "category": "dictionary",
                        "trust": 4,
                        "dataset": "cedict",
                    },
                    {
                        "source": "large_pinyin",
                        "url": "",
                        "category": "dictionary",
                        "trust": 4,
                        "dataset": "large_pinyin",
                    },
                ],
                "sourceIds": ["cedict", "large_pinyin"],
                "score": 8,
                "readingEvidenceKind": "bound_external",
            }],
            "sources": [],
            "hasEvidence": True,
            "rejections": [],
            "lookupComplete": True,
            "lookupStatus": "completed",
            "sourceOutcomes": [
                {
                    "sourceId": "cedict",
                    "source": "CC-CEDICT",
                    "status": "completed",
                    "lookupResult": "found",
                },
                {
                    "sourceId": "handian",
                    "source": "汉典",
                    "status": "completed",
                    "lookupResult": "absent",
                },
            ],
        }
        entity = {
            "recognized": True,
            "entityType": "common_word",
            "confidence": 0.96,
            "description": "指某种事物在特定条件下的具体表现和用途",
            "pinyin": "xing zhang",
            "commonTransparent": True,
            "commonnessReason": "两个字都很常用，组合关系透明",
            "canonicalNames": [],
            "aliases": [],
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
            patch.object(
                review_module,
                "_infer_entity_knowledge",
                AsyncMock(return_value=entity),
            ),
            patch.object(review_module, "_search_web", AsyncMock(return_value=[])),
        ):
            prepared = await prepare_reviewed_word(CONFIG, word)

        check(
            "mismatching offline whole-word reading is explicitly rejected",
            prepared.get("pronunciationRejections", [{}])[0].get("reason")
            == "character_1_reading_mismatch",
        )
        check(
            "contextual replacement begins sealed",
            read_review_disposition(prepared) is ReviewDisposition.SEAL
            and read_manual_review_flag(prepared) is True,
        )

        references = {
            word: _reference_row(word, dictionary_presence=2),
            "行": _reference_row("行", frequency=3, dictionary_presence=1),
            "长": _reference_row("长", frequency=3, dictionary_presence=1),
        }
        with (
            patch.object(
                review_module,
                "prepare_reviewed_word",
                AsyncMock(return_value=prepared),
            ),
            patch.object(
                review_module,
                "_query_commonness_reference",
                side_effect=lambda value: references[value],
            ),
            patch.object(
                review_module,
                "_review_code_chain_priority",
                AsyncMock(return_value={
                    "word": word,
                    "code": code,
                    "hasRecommendation": False,
                    "commonness": {},
                }),
            ),
        ):
            audit = await audit_draft_items(CONFIG, [{
                "action": "Create",
                "word": word,
                "code": code,
                "type": "Phrase",
            }])

        reviewed = audit.get("reviewedWords", {}).get(f"{word}@Phrase", {})
        assessment = reviewed.get("semanticContextAutoPass") or {}
        check(
            "dictionary disagreement cannot clear the seal",
            audit.get("autoApprove") is False
            and read_review_disposition(reviewed) is ReviewDisposition.SEAL
            and read_manual_review_flag(audit) is True,
        )
        check(
            "rejected dictionary presence is absent from auto-pass copy",
            audit.get("semanticContextAutoPassItems") == []
            and assessment.get("basisLine") == ""
            and "离线读音/词典收录" not in str(assessment.get("basisLine") or ""),
        )

    asyncio.run(_run())


def test_semantic_context_full_vendored_corpus_facts():
    """Freeze the production-sized offline facts used by the S17 policy case."""
    print("\n🧪 semantic context full vendored corpus facts")

    with tempfile.TemporaryDirectory() as temporary:
        db_path = Path(temporary) / "pinyin-reference.db"
        result = build_reference_database(
            Path(__file__).parent / "vendor" / "pinyin_reference",
            db_path,
        )
        with patch.dict(os.environ, {"PINYIN_REFERENCE_DB": str(db_path)}):
            facts = {
                value: review_module._query_commonness_reference(value)
                for value in ("产季", "产", "季", "龘", "粮棉")
            }

    check(
        "full vendored corpus was imported instead of a reduced fixture",
        result.commonness_word_count == 634829
        and result.corpus_word_count == 349045
        and result.word_count == 428180,
    )
    check(
        "产季 is absent as a whole word while both characters clear the 1000 threshold",
        facts["产季"]["attested"] is False
        and facts["产"]["corpusFrequency"] == 6838
        and facts["季"]["corpusFrequency"] == 1619,
    )
    check(
        "the full corpus contains a frequency-only control above the word threshold",
        facts["粮棉"]["dictionaryPresenceCount"] == 0
        and facts["粮棉"]["corpusFrequency"] == 55,
    )
    check(
        "the obscure-character control has no single-character corpus frequency",
        facts["龘"]["corpusFrequency"] is None,
    )


def test_semantic_context_pass_clears_prepare_seal_and_enters_autoapprove_chain():
    """The registered PASS must survive preview copy and the canonical batch gate."""
    print("\n🧪 semantic context PASS clears the add seal and enters auto-approval")

    async def _run():
        word = "产季"
        code = "ijjk"
        base_review = _semantic_context_review(word, code)
        base_review["recommendedCode"] = code
        base_review["pronunciations"][0]["recommendedCode"] = code
        base_review["pronunciations"][0]["candidateStatuses"] = [{
            "code": code,
            "occupied": False,
            "label": "空位",
        }]
        basis_line = (
            "该词可自动通过（语境读音与含义明确，常用字组合且语义判断为常用或透明组合；"
            "语料/词典证据：逐字 jieba 词频 产 6838、季 1619（高频字阈值 1000），"
            "语义判断为常用或透明组合）"
        )
        pass_audit = {
            "success": True,
            "verdict": "pass",
            "autoApprove": True,
            "summary": "语境读音、具体含义和非生僻证据一致，允许本喵自动通过",
            "issues": [],
            "approvedItems": [f"Create：{word}@{code}"],
            "semanticContextAutoPassItems": [{
                "word": word,
                "code": code,
                "basisLine": basis_line,
            }],
            "needsManualReview": False,
            "manualReviewReason": basis_line,
            "reviewDisposition": "PASS",
            "reviewVerdictSite": "semantic_context_common_word",
        }
        with (
            patch.object(
                _review_tools,
                "prepare_reviewed_word",
                AsyncMock(return_value=base_review),
            ),
            patch.object(
                _review_tools,
                "_build_pre_submit_audit",
                AsyncMock(return_value=pass_audit),
            ),
            patch.object(
                _review_tools,
                "assess_candidate_chain_commonness",
                AsyncMock(return_value=[]),
            ),
        ):
            prepared = await _review_tools.keytao_prepare_reviewed_add(word)

        check(
            "registered PASS clears the original prepare-stage SEAL",
            read_manual_review_flag(prepared) is False
            and read_review_disposition(prepared) is ReviewDisposition.PASS,
        )
        check(
            "prepare-stage PASS retains the exact registered site",
            prepared.get("reviewVerdictSite") == "semantic_context_common_word",
        )

        import keytao_bot.plugins.openai_chat as chat
        from keytao_bot.utils import review_flags as rf

        prompt = chat._format_reviewed_add_prompt(prepared) or ""
        check(
            "review copy is one honest auto-pass basis line",
            f"审词：读音 chan ji；来源" in prompt
            and "自动审核：语境读音与含义明确" in prompt
            and "语料/词典证据" in prompt
            and prompt.count("可自动通过") == 1
            and "需管理员审核" not in prompt,
        )
        check(
            "canonical submit predicate accepts the semantic PASS audit",
            rf.audit_allows_batch_auto_approve(pass_audit) is True,
        )

        chat._reviewed_add_verdicts.clear()
        chat._record_reviewed_add_verdict(
            "keytao_prepare_reviewed_add",
            {"word": word},
            json.dumps(prepared, ensure_ascii=False),
        )
        response = (
            f"是否以编码 {code} 将「{word}」加入草稿？\n"
            f"1. {code} - ✅ 空位\n"
            f"审词：读音 chan ji；来源 本喵整词语境判断；自动审核：{basis_line}\n"
        )
        pending = chat._parse_pending_add_word(response)
        check(
            "bot submit pending state carries the explicit unsealed flag",
            pending is not None and pending.needs_manual_review is False,
        )
        create_args = chat._create_phrase_args(pending, code) if pending else {}
        check(
            "draft create receives needsManualReview=false",
            create_args.get("needs_manual_review") is False,
        )

        malformed_pass = {
            **pass_audit,
            "reviewVerdictSite": "entity_context_reading",
        }
        malformed_base = _semantic_context_review(word, code)
        malformed_base["recommendedCode"] = code
        with (
            patch.object(
                _review_tools,
                "prepare_reviewed_word",
                AsyncMock(return_value=malformed_base),
            ),
            patch.object(
                _review_tools,
                "_build_pre_submit_audit",
                AsyncMock(return_value=malformed_pass),
            ),
            patch.object(
                _review_tools,
                "assess_candidate_chain_commonness",
                AsyncMock(return_value=[]),
            ),
        ):
            rejected = await _review_tools.keytao_prepare_reviewed_add(word)
        check(
            "PASS value cannot clear SEAL when its registry site declares SEAL",
            read_manual_review_flag(rejected) is True
            and read_review_disposition(rejected) is ReviewDisposition.SEAL,
        )

        explicit_reading_base = _semantic_context_review(word, code)
        explicit_reading_base["recommendedCode"] = code
        explicit_reading_base["reviewVerdictSite"] = "missing_authoritative_page"
        with (
            patch.object(
                _review_tools,
                "prepare_reviewed_word",
                AsyncMock(return_value=explicit_reading_base),
            ),
            patch.object(
                _review_tools,
                "_build_pre_submit_audit",
                AsyncMock(return_value=pass_audit),
            ),
            patch.object(
                _review_tools,
                "assess_candidate_chain_commonness",
                AsyncMock(return_value=[]),
            ),
        ):
            explicit_reading = await _review_tools.keytao_prepare_reviewed_add(word)
        check(
            "generic pre-submit PASS cannot clear an explicit-reading seal",
            read_manual_review_flag(explicit_reading) is True
            and read_review_disposition(explicit_reading) is ReviewDisposition.SEAL,
        )

        blocked_base = _semantic_context_review(word, code)
        blocked_base["recommendedCode"] = code
        blocked_base["reviewDisposition"] = "BLOCK"
        blocked_base["reviewVerdictSite"] = "invalid_code"
        with (
            patch.object(
                _review_tools,
                "prepare_reviewed_word",
                AsyncMock(return_value=blocked_base),
            ),
            patch.object(
                _review_tools,
                "_build_pre_submit_audit",
                AsyncMock(return_value=pass_audit),
            ),
            patch.object(
                _review_tools,
                "assess_candidate_chain_commonness",
                AsyncMock(return_value=[]),
            ),
        ):
            blocked = await _review_tools.keytao_prepare_reviewed_add(word)
        check(
            "registered PASS cannot clear a base BLOCK",
            read_manual_review_flag(blocked) is True
            and read_review_disposition(blocked) is ReviewDisposition.BLOCK,
        )

    asyncio.run(_run())


def test_chanji_semantic_prepare_revalidation_reaches_common_char_pass():
    """A 产季-style semantic re-encode must produce the evidence the new lane reads."""
    print("\n🧪 产季 semantic revalidation reaches common-character PASS")

    async def _run():
        word = "产季"
        code = "ijjk"
        baseline_encode = {
            "success": True,
            "word": word,
            "codes": ["isjk"],
            "candidateCodes": ["isjk", code],
            "alternatePhrasePronunciationCodes": [{
                "char": "产",
                "charIndex": 0,
                "pinyin": "chǎn",
                "codes": [code],
            }],
            "pronunciationSource": "zdic-character-default",
            "standardPronunciationStatus": "absent",
            "semanticPronunciationNeeded": True,
            "semanticPronunciationAccepted": False,
            "phrasePinyins": ["shān", "jì"],
            "contextPhrasePinyins": ["chǎn", "jì"],
            "chars": [
                {
                    "char": "产",
                    "pinyin": "shān",
                    "pinyins": ["shān", "chǎn"],
                    "pronunciationLookupStatus": "found",
                },
                {
                    "char": "季",
                    "pinyin": "jì",
                    "pinyins": ["jì"],
                    "pronunciationLookupStatus": "found",
                },
            ],
        }
        proposal = {
            "accepted": True,
            "word": word,
            "pinyins": ["chan", "ji"],
            "meaning": "指农产品集中生产或上市的季节",
            "confidence": 0.96,
            "usageType": "transparent_compound",
            "commonTransparent": True,
            "commonnessReason": "产与季组合关系透明",
        }
        evidence = {
            "success": True,
            "groups": [],
            "sources": [],
            "lookupComplete": True,
            "sourceOutcomes": [{
                "sourceId": "handian",
                "source": "汉典",
                "status": "completed",
                "lookupResult": "absent",
            }],
        }
        encode_mock = AsyncMock(return_value=baseline_encode)
        with (
            patch.object(
                review_module,
                "collect_pronunciation_evidence_limited",
                AsyncMock(return_value=evidence),
            ),
            patch.object(
                review_module,
                "fetch_keytao_encode",
                encode_mock,
            ),
            patch.object(
                review_module,
                "lookup_words",
                AsyncMock(return_value={}),
            ),
            patch.object(
                review_module,
                "lookup_codes",
                AsyncMock(return_value={}),
            ),
            patch.object(
                review_module,
                "_infer_semantic_pronunciation_for_review",
                AsyncMock(return_value=proposal),
            ),
        ):
            prepared = await prepare_reviewed_word(CONFIG, word)

        pronunciation = prepared.get("pronunciations", [{}])[0]
        check(
            "semantic reading selects the matching chain from one encode result",
            encode_mock.await_count == 1
            and encode_mock.await_args.kwargs == {},
        )
        check(
            "prepared review exposes exact known readings for every character",
            pronunciation.get("characterReadings") == [
                {
                    "char": "产",
                    "chosenPinyin": "chan",
                    "knownReadings": ["shan", "chan"],
                    "lookupStatus": "found",
                },
                {
                    "char": "季",
                    "chosenPinyin": "ji",
                    "knownReadings": ["ji"],
                    "lookupStatus": "found",
                },
            ],
        )
        check(
            "prepare remains sealed until the non-obscurity predicate runs",
            read_review_disposition(prepared) is ReviewDisposition.SEAL
            and read_manual_review_flag(prepared) is True,
        )

        references = {
            word: _reference_row(word),
            "产": _reference_row("产", frequency=6838),
            "季": _reference_row("季", frequency=1619),
        }
        with (
            patch.object(
                review_module,
                "prepare_reviewed_word",
                AsyncMock(return_value=prepared),
            ),
            patch.object(
                review_module,
                "_query_commonness_reference",
                side_effect=lambda value: references[value],
            ),
            patch.object(
                review_module,
                "_review_code_chain_priority",
                AsyncMock(return_value={
                    "word": word,
                    "code": code,
                    "hasRecommendation": False,
                    "commonness": {},
                }),
            ),
        ):
            audit = await audit_draft_items(CONFIG, [{
                "action": "Create",
                "word": word,
                "code": code,
                "type": "Phrase",
            }])
        check(
            "产季-style common characters plus semantic judgment auto-pass",
            audit.get("autoApprove") is True
            and read_review_disposition(audit) is ReviewDisposition.PASS,
        )

    asyncio.run(_run())


def test_chanji_entity_context_reaches_common_char_pass():
    """The ordinary no-conflict 产季 path must retain the same semantic fields."""
    print("\n🧪 产季 entity context reaches common-character PASS")

    async def _run():
        word = "产季"
        code = "jfjk"
        encode_data = {
            "success": True,
            "word": word,
            "codes": [code],
            "pronunciationSource": "zdic-character-default",
            "standardPronunciationStatus": "absent",
            "semanticPronunciationNeeded": False,
            "phrasePinyins": ["chǎn", "jì"],
            "contextPhrasePinyins": ["chǎn", "jì"],
            "chars": [
                {
                    "char": "产",
                    "pinyin": "chǎn",
                    "pinyins": ["chǎn"],
                    "pronunciationLookupStatus": "found",
                },
                {
                    "char": "季",
                    "pinyin": "jì",
                    "pinyins": ["jì"],
                    "pronunciationLookupStatus": "found",
                },
            ],
        }
        evidence = {
            "success": True,
            "groups": [],
            "sources": [],
            "lookupComplete": True,
            "sourceOutcomes": [{
                "sourceId": "handian",
                "source": "汉典",
                "status": "completed",
                "lookupResult": "absent",
            }],
        }
        entity = {
            "recognized": True,
            "entityType": "transparent_compound",
            "confidence": 0.96,
            "description": "指农产品集中生产或上市的季节",
            "pinyin": "chan ji",
            "commonTransparent": True,
            "commonnessReason": "产与季组合关系透明",
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
            patch.object(
                review_module,
                "lookup_words",
                AsyncMock(return_value={}),
            ),
            patch.object(
                review_module,
                "lookup_codes",
                AsyncMock(return_value={}),
            ),
            patch.object(
                review_module,
                "_infer_entity_knowledge",
                AsyncMock(return_value=entity),
            ),
        ):
            prepared = await prepare_reviewed_word(CONFIG, word)

        context = prepared["pronunciations"][0]["contextPronunciation"]
        check(
            "entity context preserves the transparent-compound judgment",
            context.get("method") == "entity_knowledge_context"
            and context.get("commonTransparent") is True
            and context.get("description") == entity["description"],
        )

        references = {
            word: _reference_row(word),
            "产": _reference_row("产", frequency=6838),
            "季": _reference_row("季", frequency=1619),
        }
        with (
            patch.object(
                review_module,
                "prepare_reviewed_word",
                AsyncMock(return_value=prepared),
            ),
            patch.object(
                review_module,
                "_query_commonness_reference",
                side_effect=lambda value: references[value],
            ),
            patch.object(
                review_module,
                "_review_code_chain_priority",
                AsyncMock(return_value={
                    "word": word,
                    "code": code,
                    "hasRecommendation": False,
                    "commonness": {},
                }),
            ),
        ):
            audit = await audit_draft_items(CONFIG, [{
                "action": "Create",
                "word": word,
                "code": code,
                "type": "Phrase",
            }])
        check(
            "ordinary 产季 entity path auto-passes through common characters",
            audit.get("autoApprove") is True
            and read_review_disposition(audit) is ReviewDisposition.PASS,
        )

    asyncio.run(_run())


CONFIG = ReviewHttpConfig(api_base="https://fake", bot_token="fake")
PRONUNCIATION_FIXTURES = (
    Path(__file__).parent / "test_fixtures" / "pronunciation_sources"
)


REFERENCE_FIXTURE_SOURCES = {
    "zdic_cibs": (
        "zdic_cibs.txt.gz",
        "phrase",
        True,
        (
            "诉讼费: sù sòng fèi\n"
            "诉讼法: sù sòng fǎ\n"
            "光面: guāng miàn\n"
            "慑服: shè fú\n"
            "射覆: shè fù\n"
            "双汉典: shuāng hàn diǎn\n"
            "朝阳: zhāo yáng\n"
            "朝阳: cháo yáng\n"
        ),
    ),
    "zdic_cybs": (
        "zdic_cybs.txt.gz",
        "phrase",
        True,
        "一心一意: yī xīn yī yì\n双汉典: shuāng hàn diǎn\n",
    ),
    "large_pinyin": (
        "large_pinyin.txt.gz",
        "phrase",
        True,
        (
            "诉讼费: sù sòng fèi\n"
            "诉讼法: sù sòng fǎ\n"
            "光面: guāng miàn\n"
            "慑服: shè fú\n"
            "射覆: shè fù\n"
            "阿勒泰: ā lè tài\n"
        ),
    ),
    "pinyin": (
        "pinyin.txt.gz",
        "phrase",
        False,
        "不直接导入: bù zhí jiē dǎo rù\n",
    ),
    "cedict": (
        "cedict.txt.gz",
        "cedict",
        True,
        (
            "# CC-CEDICT fixture\n"
            "诉讼法 诉讼法 [su4 song4 fa3] /procedural law/\n"
            "石蒜 石蒜 [shi2 suan4] /red spider lily/\n"
            "傳統 简体 [lu:4 se4] /fixture/\n"
            "吃席 吃席 [chi1 xi2] /owner-governed exclusion/\n"
        ),
    ),
    "jieba": (
        "jieba_dict.txt.gz",
        "jieba",
        True,
        (
            "诉讼法 66 n\n"
            "诉讼费 15 n\n"
            "光面 20 n\n"
            "慑服 26 v\n"
            "石蒜 11 n\n"
            "粮棉 55 n\n"
        ),
    ),
}


def _write_deterministic_gzip(path, text):
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", filename="", mtime=0) as handle:
            handle.write(text.encode("utf-8"))


def _build_reference_fixture(root):
    source_dir = root / "source"
    source_dir.mkdir()
    datasets = []
    for dataset, (filename, source_format, imported, content) in REFERENCE_FIXTURE_SOURCES.items():
        _write_deterministic_gzip(source_dir / filename, content)
        datasets.append({
            "id": dataset,
            "file": filename,
            "format": source_format,
            "import": imported,
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        })
    (source_dir / "manifest.json").write_text(
        json.dumps({"formatVersion": 1, "datasets": datasets}),
        encoding="utf-8",
    )
    (source_dir / "excluded_words.txt").write_text("吃席\n", encoding="utf-8")
    db_path = root / "pinyin_reference.db"
    result = build_reference_database(source_dir, db_path)
    return source_dir, db_path, result


@contextmanager
def _reference_fixture_environment():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source_dir, db_path, result = _build_reference_fixture(root)
        with patch.dict(os.environ, {"PINYIN_REFERENCE_DB": str(db_path)}):
            yield source_dir, db_path, result


def _pronunciation_fixture(name):
    return (PRONUNCIATION_FIXTURES / name).read_text(encoding="utf-8")


async def _collect_hwxnet_fixture(search_fixture, entry_fixture):
    review_module._clear_review_caches()
    source = dict(review_module._source_by_id("hwxnet_cidian"))
    search_url = source["follow_search_url"].format(word="%E8%AF%89%E8%AE%BC%E8%B4%B9")
    calls = []

    async def fixture_fetch(url, **kwargs):
        calls.append((url, dict(kwargs)))
        if url == search_url:
            content = _pronunciation_fixture(search_fixture)
        elif "/view/" in url:
            content = _pronunciation_fixture(entry_fixture)
        else:
            raise AssertionError(f"Unexpected fixture URL: {url}")
        if not kwargs.get("preserve_html"):
            content = review_module._strip_tags(content)
        return review_module._LookupText(content, lookup_status="completed")

    with (
        patch.object(review_module, "AUTHORITATIVE_SOURCES", [source]),
        patch.object(
            review_module,
            "_collect_local_pronunciation_evidence",
            return_value={"entries": [], "outcomes": []},
        ),
        patch.object(review_module, "_fetch_text", side_effect=fixture_fetch),
        patch.object(review_module, "_search_web", AsyncMock(return_value=[])),
    ):
        evidence = await review_module.collect_pronunciation_evidence("诉讼费")
    return evidence, calls


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


def _review_evidence(*, complete=True, handian_status="completed"):
    return {
        "success": True,
        "word": "诉讼费",
        "groups": [],
        "sources": [],
        "hasEvidence": False,
        "rejections": [],
        "lookupStatus": "completed" if complete else "incomplete",
        "lookupComplete": complete,
        "sourceOutcomes": [{
            "sourceId": "handian",
            "source": "汉典",
            "status": handian_status,
        }],
    }


def _su_song_fei_encode(*, status, source, phrase_pinyins=None):
    return {
        "success": True,
        "word": "诉讼费",
        "codes": ["ssfw", "ssfwo", "ssfwov"],
        "altCodes": [],
        "pronunciationSource": source,
        "standardPronunciationStatus": status,
        "semanticPronunciationNeeded": False,
        "phrasePinyins": phrase_pinyins or ["sù", "sòng", "fèi"],
        "contextPhrasePinyins": ["sù", "sòng", "fèi"],
        "chars": [
            {"char": "诉", "pinyin": "sù", "pinyins": ["sù"], "pronunciationLookupStatus": "found"},
            {"char": "讼", "pinyin": "sòng", "pinyins": ["sòng"], "pronunciationLookupStatus": "found"},
            {"char": "费", "pinyin": "fèi", "pinyins": ["fèi"], "pronunciationLookupStatus": "found"},
        ],
    }


async def _review_word_with_encode(word, evidence, encode_data):
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
        return await prepare_reviewed_word(CONFIG, word)


def test_local_reference_import_is_deterministic_and_preserves_readings():
    print("\n🧪 local reference import correctness and deterministic rebuild")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source_dir, db_path, first = _build_reference_fixture(root)
        first_bytes = db_path.read_bytes()
        second = build_reference_database(source_dir, db_path)
        second_bytes = db_path.read_bytes()
        duplicate_db = root / "duplicate.db"
        duplicate = build_reference_database(source_dir, duplicate_db)

        cedict = pinyin_reference_module.query_reference_readings(
            "石蒜", db_path=db_path
        )
        multi_reading = pinyin_reference_module.query_reference_readings(
            "朝阳", db_path=db_path
        )
        simplified = pinyin_reference_module.query_reference_readings(
            "简体", db_path=db_path
        )
        traditional = pinyin_reference_module.query_reference_readings(
            "傳統", db_path=db_path
        )
        excluded = pinyin_reference_module.query_reference_readings(
            "吃席", db_path=db_path
        )
        connection = sqlite3.connect(db_path)
        try:
            commonness_rows = {
                row[0]: row[1:]
                for row in connection.execute(
                    """
                    SELECT word, corpus_frequency, part_of_speech,
                        dictionary_presence_count
                    FROM word_commonness
                    WHERE word IN (
                        '诉讼法', '诉讼费', '射覆', '粮棉', '双汉典'
                    )
                    """
                )
            }
        finally:
            connection.close()

        check("first fixture build writes the DB", first.rebuilt is True)
        check("same source fingerprint skips rebuilding", second.rebuilt is False)
        check("idempotent skip leaves DB bytes unchanged", second_bytes == first_bytes)
        check(
            "independent deterministic builds have identical bytes",
            duplicate.rebuilt is True and duplicate_db.read_bytes() == first_bytes,
        )
        check(
            "CC-CEDICT tone numbers become tone marks and preserve source form",
            len(cedict) == 1
            and cedict[0].normalized == ("shi", "suan")
            and cedict[0].display == "shí suàn"
            and cedict[0].source_reading == "shi2 suan4"
            and cedict[0].dataset == "cedict",
        )
        check(
            "one dataset can retain multiple readings for one word",
            {reading.normalized for reading in multi_reading}
            == {("zhao", "yang"), ("chao", "yang")},
        )
        check(
            "CC-CEDICT lookup uses only the simplified key",
            len(simplified) == 1
            and simplified[0].display == "lǜ sè"
            and traditional == [],
        )
        check("owner-governed absent word is excluded", excluded == [])
        check(
            "jieba frequency and part of speech are stored",
            commonness_rows["诉讼法"] == (66, "n", 3)
            and commonness_rows["粮棉"] == (55, "n", 0),
        )
        check(
            "dictionary presence groups the two zdic variants as one source",
            commonness_rows["诉讼费"] == (15, "n", 2)
            and commonness_rows["射覆"] == (None, None, 2)
            and commonness_rows["双汉典"] == (None, None, 1),
        )
        check(
            "build metadata counts corpus and commonness words",
            first.corpus_word_count == 6
            and first.commonness_word_count >= first.corpus_word_count
            and first.dataset_counts["jieba"].imported == 6,
        )

        connection = pinyin_reference_module._read_only_connection(db_path)
        try:
            try:
                connection.execute(
                    "INSERT INTO metadata (key, value) VALUES ('write_probe', '1')"
                )
                write_blocked = False
            except sqlite3.OperationalError:
                write_blocked = True
        finally:
            connection.close()
        check("runtime connection rejects writes", write_blocked)


def test_reference_version_mismatch_rebuilds_and_missing_schema_warns():
    """Startup rebuild keys include builder/schema versions and missing tables are loud."""
    print("\n🧪 reference version mismatch rebuild and schema warning")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source_dir, db_path, _first = _build_reference_fixture(root)
        connection = sqlite3.connect(db_path)
        try:
            connection.execute(
                "UPDATE metadata SET value = '2' WHERE key = 'builder_version'"
            )
            connection.commit()
        finally:
            connection.close()

        rebuilt = build_reference_database(source_dir, db_path)
        connection = sqlite3.connect(db_path)
        try:
            connection.execute(
                "UPDATE metadata SET value = '1' WHERE key = 'schema_version'"
            )
            connection.commit()
        finally:
            connection.close()
        schema_rebuilt = build_reference_database(source_dir, db_path)
        connection = sqlite3.connect(db_path)
        try:
            metadata = dict(connection.execute(
                "SELECT key, value FROM metadata WHERE key IN ('schema_version', 'builder_version')"
            ))
        finally:
            connection.close()
        check(
            "builder/schema-version mismatch forces an automatic rebuild",
            rebuilt.rebuilt is True
            and schema_rebuilt.rebuilt is True
            and metadata == {
                "builder_version": pinyin_reference_build_module.BUILDER_VERSION,
                "schema_version": pinyin_reference_build_module.SCHEMA_VERSION,
            },
        )

        stale_db = root / "missing-commonness.db"
        connection = sqlite3.connect(stale_db)
        try:
            connection.executescript("""
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE readings (word TEXT NOT NULL);
            """)
            connection.commit()
        finally:
            connection.close()

        with (
            patch.dict(os.environ, {"PINYIN_REFERENCE_DB": str(stale_db)}),
            patch.object(review_module.logger, "warning") as query_warning,
        ):
            unavailable = review_module._query_commonness_reference("产季")
        check(
            "missing commonness query path logs at warning",
            unavailable.get("available") is False
            and query_warning.call_count == 1
            and "word_commonness" in str(query_warning.call_args.args[0]),
        )

        schema_assertion = getattr(
            pinyin_reference_build_module,
            "assert_commonness_reference_schema",
            None,
        )
        check(
            "startup exposes a one-shot commonness schema assertion",
            callable(schema_assertion),
        )
        if callable(schema_assertion):
            with patch.object(
                pinyin_reference_build_module.logger,
                "warning",
            ) as startup_warning:
                schema_ok = schema_assertion(stale_db)
            check(
                "startup assertion warns loudly when word_commonness is absent",
                schema_ok is False
                and startup_warning.call_count == 1
                and "word_commonness" in str(startup_warning.call_args.args[0]),
            )


def test_offline_commonness_verdict_rules_and_copy():
    print("\n🧪 offline commonness verdict rules and evidence copy")

    async def _run():
        with _reference_fixture_environment():
            review_module._clear_review_caches()
            forbidden_fallback = AsyncMock(
                side_effect=AssertionError("offline-attested comparison used web fallback")
            )
            with patch.object(
                review_module,
                "_estimate_word_commonness_web_fallback",
                forbidden_fallback,
            ):
                high_ratio = await review_module.compare_word_commonness(
                    "诉讼法", "诉讼费"
                )
                low_ratio = await review_module.compare_word_commonness(
                    "光面", "诉讼费"
                )
                attested_absent = await review_module.compare_word_commonness(
                    "石蒜", "亮面"
                )
                s9 = await review_module.compare_word_commonness("射覆", "慑服")
                estimate = await review_module.estimate_word_commonness("诉讼法")

            check(
                "frequency ratio above 2.0 yields a definite verdict",
                high_ratio.get("verdict") == "front_more_common"
                and high_ratio.get("decisionReason") == "frequency_ratio",
            )
            check(
                "frequency ratio below 2.0 is close rather than flapping",
                low_ratio.get("verdict") == "close"
                and low_ratio.get("decisionReason") == "frequency_ratio_below_threshold",
            )
            check(
                "corpus and dictionary attestation beats a fully absent word",
                attested_absent.get("verdict") == "front_more_common"
                and attested_absent.get("decisionReason")
                == "corpus_and_dictionary_vs_absent",
            )
            check(
                "S9 fixture has a definite keep-order verdict",
                s9.get("verdict") == "behind_more_common"
                and s9.get("decisionReason")
                == "corpus_attested_with_no_presence_deficit",
            )
            check(
                "local estimator exposes log-scaled corpus and presence signals",
                estimate.get("method") == "offline_reference"
                and 0 < estimate.get("signals", {}).get("corpus", 0) < 1
                and estimate.get("signals", {}).get("dictionary") == 1
                and estimate.get("reference", {}).get("partOfSpeech") == "n",
            )
            check(
                "reference-backed paths never reach the web fallback",
                forbidden_fallback.await_count == 0,
            )
            check(
                "comparison copy cites frequency and dictionary presence on one line",
                high_ratio.get("summary")
                == "「诉讼法」较「诉讼费」更常用：语料频次 66 vs 15，词典收录 3 vs 2"
                and "\n" not in high_ratio.get("summary", "")
                and "语料频次 20 vs 15，词典收录 2 vs 2"
                in low_ratio.get("summary", "")
                and "语料频次 无 vs 26，词典收录 2 vs 2"
                in s9.get("summary", ""),
            )
            assessment = review_module._candidate_commonness_assessment(
                {
                    "newWord": "射覆",
                    "occupantWord": "慑服",
                    "occupantCode": "eefj",
                    "freeCode": "eefju",
                },
                s9,
            )
            check(
                "candidate assessment preserves the evidence-citing copy",
                assessment.get("summary") == s9.get("summary")
                and assessment.get("newCode") == "eefju",
            )

    asyncio.run(_run())


def test_both_absent_commonness_uses_existing_bounded_web_fallback():
    print("\n🧪 both-absent commonness uses the bounded web fallback")

    async def _run():
        with _reference_fixture_environment():
            review_module._clear_review_caches()
            entity_signal = AsyncMock(return_value={
                "accepted": False,
                "word": "",
                "entityType": "unclear",
                "confidence": 0.0,
            })
            pronunciation = AsyncMock(return_value={"success": False, "groups": []})
            search = AsyncMock(return_value=[])
            with (
                patch.object(
                    review_module,
                    "_estimate_entity_knowledge_signal",
                    entity_signal,
                ),
                patch.object(
                    review_module,
                    "collect_pronunciation_evidence_limited",
                    pronunciation,
                ),
                patch.object(review_module, "_search_web", search),
            ):
                result = await review_module.compare_word_commonness(
                    "全无甲", "全无乙"
                )

            check(
                "both absent words use web fallback and remain insufficient",
                result.get("webFallback") is True
                and result.get("verdict") == "not_enough_evidence",
            )
            check(
                "fallback preserves the existing five queries per word",
                entity_signal.await_count == 2
                and pronunciation.await_count == 2
                and search.await_count
                == 2 * len(review_module.COMMONNESS_SEARCH_QUERIES),
            )
            check(
                "fallback copy cites its basis without claiming local evidence",
                result.get("summary")
                == "常用度信号不足：离线均无收录，网页回退得分 0.00 vs 0.00",
            )
            check(
                "candidate and audit fallback budgets stay at five seconds",
                review_module.CANDIDATE_COMMONNESS_TIMEOUT_SECONDS == 5.0
                and review_module.AUDIT_COMMONNESS_STAGE_TIMEOUT == 5.0,
            )

    asyncio.run(_run())


def test_collector_queries_local_reference_first_and_scores_agreement():
    print("\n🧪 collector queries local reference before live corroboration")

    async def _run():
        with _reference_fixture_environment():
            review_module._clear_review_caches()
            events = []
            real_query = review_module.query_reference_readings
            live_source = {
                "id": "live_fixture",
                "label": "Live fixture dictionary",
                "domain": "live.test",
                "category": "dictionary",
                "trust": 3,
                "query": 'site:live.test "{word}" 拼音',
                "direct_urls": [],
            }

            def tracked_local_query(word):
                events.append("local")
                return real_query(word)

            async def live_search(_query, max_results=3):
                events.append("network")
                return [{
                    "title": "诉讼费 pronunciation",
                    "url": "https://live.test/susongfei",
                    "snippet": "诉讼费 拼音：sù sòng fèi",
                }]

            async def live_page(_url, **_kwargs):
                return review_module._LookupText(
                    "诉讼费 拼音：sù sòng fèi",
                    lookup_status="completed",
                )

            with (
                patch.object(review_module, "AUTHORITATIVE_SOURCES", [live_source]),
                patch.object(
                    review_module,
                    "query_reference_readings",
                    side_effect=tracked_local_query,
                ),
                patch.object(review_module, "_search_web", side_effect=live_search),
                patch.object(review_module, "_fetch_text", side_effect=live_page),
            ):
                evidence = await review_module.collect_pronunciation_evidence("诉讼费")

            group = next(iter(evidence.get("groups") or []), {})
            sources = group.get("sources") or []
            check("local indexed lookup happens before network work", events[0] == "local")
            check("live corroboration still runs after the local hit", "network" in events)
            check("toned local reading is preserved for display", group.get("pinyin") == "sù sòng fèi")
            check(
                "local Han Dian provenance is honest",
                any(
                    source.get("source") == "汉典（离线数据集）"
                    and source.get("dataset") == "zdic_cibs"
                    and source.get("trust") == 5
                    for source in sources
                ),
            )
            check(
                "matching local and live source trust scores accumulate",
                group.get("score") == 12
                and set(group.get("sourceIds") or [])
                == {"zdic_cibs", "large_pinyin", "live_fixture"},
            )
            check("a completed local hit survives as complete authority", evidence.get("lookupComplete") is True)

    asyncio.run(_run())


def test_local_reference_miss_falls_through_to_live_sources():
    print("\n🧪 local reference miss preserves live-source fallback")

    async def _run():
        with _reference_fixture_environment():
            review_module._clear_review_caches()
            network_calls = 0
            live_source = {
                "id": "live_fixture",
                "label": "Live fixture dictionary",
                "domain": "live.test",
                "category": "dictionary",
                "trust": 3,
                "query": 'site:live.test "{word}" 拼音',
                "direct_urls": [],
            }

            async def live_search(_query, max_results=3):
                nonlocal network_calls
                network_calls += 1
                return [{
                    "title": "测试词 pronunciation",
                    "url": "https://live.test/test-word",
                    "snippet": "测试词 拼音：cè shì cí",
                }]

            async def live_page(_url, **_kwargs):
                return review_module._LookupText(
                    "测试词 拼音：cè shì cí",
                    lookup_status="completed",
                )

            with (
                patch.object(review_module, "AUTHORITATIVE_SOURCES", [live_source]),
                patch.object(review_module, "_search_web", side_effect=live_search),
                patch.object(review_module, "_fetch_text", side_effect=live_page),
            ):
                evidence = await review_module.collect_pronunciation_evidence("测试词")

            group = next(iter(evidence.get("groups") or []), {})
            local_outcomes = [
                outcome
                for outcome in evidence.get("sourceOutcomes") or []
                if outcome.get("sourceId") in pinyin_reference_module.REFERENCE_DATASET_POLICY_BY_ID
            ]
            check("local miss still invokes the existing live path", network_calls == 1)
            check(
                "local miss is recorded as completed absence",
                len(local_outcomes) == 4
                and all(outcome.get("lookupResult") == "absent" for outcome in local_outcomes),
            )
            check(
                "live evidence is returned unchanged after the local miss",
                group.get("normalized") == ["ce", "shi", "ci"]
                and group.get("sourceIds") == ["live_fixture"]
                and group.get("score") == 3,
            )

    asyncio.run(_run())


def test_poisoned_local_reference_row_fails_per_syllable_validation():
    print("\n🧪 poisoned local reference row still fails per-syllable validation")

    async def _run():
        with _reference_fixture_environment() as (_source_dir, db_path, _result):
            connection = sqlite3.connect(db_path)
            connection.execute(
                """
                INSERT INTO readings
                    (word, normalized, display, source_reading, dataset)
                VALUES (?, ?, ?, ?, ?)
                """,
                ("诉讼费", "su song", "sù sòng", "sù sòng", "zdic_cibs"),
            )
            connection.commit()
            connection.close()
            review_module._clear_review_caches()

            with patch.object(review_module, "AUTHORITATIVE_SOURCES", []):
                evidence = await review_module.collect_pronunciation_evidence("诉讼费")
            review = await _review_word_with_encode(
                "诉讼费",
                evidence,
                _su_song_fei_encode(
                    status="absent",
                    source="pinyin-pro-context",
                ),
            )

            check(
                "corrupt local row reaches collector as a separate group",
                any(
                    group.get("normalized") == ["su", "song"]
                    for group in evidence.get("groups") or []
                ),
            )
            check(
                "downstream validation records the local syllable-count poison",
                any(
                    rejection.get("reason") == "syllable_count_mismatch"
                    and rejection.get("sourceIds") == ["zdic_cibs"]
                    for rejection in review.get("pronunciationRejections") or []
                ),
            )
            check(
                "poisoned local sequence never becomes a reviewed pronunciation",
                not any(
                    pronunciation.get("normalized") == ["su", "song"]
                    for pronunciation in review.get("pronunciations") or []
                ),
            )

    asyncio.run(_run())


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

        async def poisoned_page(url, **_kwargs):
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
            "prepare result drops the constant source policy payload",
            "sourcePolicy" not in review,
        )
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
            "verdict": "pass",
            "autoApprove": True,
            "needsManualReview": False,
            "issues": [],
            "summary": "后续建议认为可自动通过",
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
        check(
            "吃席 unavailable authority verdict explicitly declares incomplete-lookup SEAL",
            read_review_disposition(review) is ReviewDisposition.SEAL
            and review.get("reviewVerdictSite") == "pronunciation_lookup_incomplete",
        )
        check(
            "later preview approval cannot clear the SEAL verdict",
            read_manual_review_flag(review) is True,
        )
        first_pronunciation = next(iter(review.get("pronunciations") or []), {})
        check(
            "吃席 fallback keeps entity context and exposes the authority outage",
            first_pronunciation.get("sourceSummary")
            == (
                "本喵实体语境判断（常见词，暂无权威页）；"
                "本次权威来源查询未完成（汉典（经编码服务））"
            ),
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


def test_hwxnet_real_fixture_extracts_honest_provenance():
    print("\n🧪 hwxnet real fixture and honest provenance")

    async def _run():
        evidence, calls = await _collect_hwxnet_fixture(
            "hwxnet-search-exact.html",
            "hwxnet-susongfei.html",
        )
        group = next(iter(evidence.get("groups") or []), {})
        sources = group.get("sources") or []
        entry_calls = [url for url, _kwargs in calls if "/view/" in url]
        check("real hwxnet fixture yields su song fei", group.get("pinyin") == "su song fei")
        check("hwxnet label is honest and attached", sources == [{
            "source": "汉文学网·汉语词典",
            "url": "https://cd.hwxnet.com/view/pmglmbimchjkneeh.html",
            "category": "dictionary",
            "trust": 4,
        }])
        check("the first exact anchor is the only followed entry", entry_calls == [
            "https://cd.hwxnet.com/view/pmglmbimchjkneeh.html",
        ])
        check("search and entry use one attempt each", [kwargs.get("max_attempts") for _url, kwargs in calls] == [1, 1])
        check("search fetch preserves HTML only for exact-anchor inspection", [
            kwargs.get("preserve_html", False) for _url, kwargs in calls
        ] == [True, False])
        check("word and character carriers are length-scoped", (
            review_module._source_applies_to_word(
                review_module._source_by_id("hwxnet_cidian"),
                "诉讼费",
            )
            and not review_module._source_applies_to_word(
                review_module._source_by_id("hwxnet_cidian"),
                "诉",
            )
            and review_module._source_applies_to_word(
                review_module._source_by_id("hwxnet_xinhua"),
                "诉",
            )
            and not review_module._source_applies_to_word(
                review_module._source_by_id("hwxnet_xinhua"),
                "诉讼费",
            )
        ))

    asyncio.run(_run())


def test_hwxnet_follow_requires_exact_anchor_text():
    print("\n🧪 hwxnet follow requires exact anchor text")

    async def _run():
        evidence, calls = await _collect_hwxnet_fixture(
            "hwxnet-search-wrong-word.html",
            "hwxnet-susongfei-poisoned.html",
        )
        check("wrong-word anchor never triggers a follow", len(calls) == 1)
        check("wrong-word search page yields no pronunciation", evidence.get("groups") == [])
        check("wrong-word search page yields no source entry", evidence.get("sources") == [])
        check("wrong-word search page fails the explicit anchor binding", any(
            rejection.get("reason") == "search_anchor_not_exact_word"
            for rejection in evidence.get("rejections") or []
            if isinstance(rejection, dict)
        ))
        check("the completed non-match remains a clean miss", evidence.get("lookupComplete") is True)

    asyncio.run(_run())


def test_hwxnet_poisoned_fixture_fails_per_syllable_validation():
    print("\n🧪 hwxnet poison still fails per-syllable validation")

    async def _run():
        evidence, _calls = await _collect_hwxnet_fixture(
            "hwxnet-search-exact.html",
            "hwxnet-susongfei-poisoned.html",
        )
        poisoned = next(iter(evidence.get("groups") or []), {})
        review = await _review_word_with_encode(
            "诉讼费",
            evidence,
            _su_song_fei_encode(
                status="absent",
                source="pinyin-pro-context",
            ),
        )
        check("poison fixture reaches the ordinary extraction stage", poisoned.get("pinyin") == "shu song fei")
        check("poison fixture carries the hwxnet source id", poisoned.get("sourceIds") == ["hwxnet_cidian"])
        check("poisoned shu sequence is absent after validation", not any(
            item.get("normalized") == ["shu", "song", "fei"]
            for item in review.get("pronunciations") or []
            if isinstance(item, dict)
        ))
        check("per-syllable mismatch is recorded", any(
            rejection.get("reason") == "character_1_reading_mismatch"
            and rejection.get("sourceIds") == ["hwxnet_cidian"]
            for rejection in review.get("pronunciationRejections") or []
            if isinstance(rejection, dict)
        ))
        check("poisoned hwxnet provenance cannot survive validation", not any(
            source.get("source") == "汉文学网·汉语词典"
            for item in review.get("pronunciations") or []
            if isinstance(item, dict)
            for source in item.get("sources") or []
            if isinstance(source, dict)
        ))

    asyncio.run(_run())


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
        check(
            "unresolved pronunciation explicitly declares BLOCK",
            read_review_disposition(unavailable) is ReviewDisposition.BLOCK,
        )
        check("unavailable rejection is logged", unavailable_log.call_count >= 1)

    asyncio.run(_run())


def test_pronunciation_source_failure_is_not_cached_and_retry_refetches():
    print("\n🧪 failed pronunciation lookup is not cached")

    async def _run():
        review_module._clear_review_caches()
        fetch_calls = 0
        source = {
            "id": "handian",
            "label": "汉典",
            "domain": "zdic.net",
            "category": "dictionary",
            "trust": 5,
            "query": 'site:zdic.net "{word}" 拼音',
            "direct_urls": ["https://www.zdic.net/hans/{word}"],
        }

        async def cold_then_ready(_url):
            nonlocal fetch_calls
            fetch_calls += 1
            if fetch_calls == 1:
                raise RuntimeError("simulated cold-start fetch failure")
            return "诉讼费 拼音：sù sòng fèi 注音"

        with (
            patch.object(review_module, "AUTHORITATIVE_SOURCES", [source]),
            patch.object(
                review_module,
                "_collect_local_pronunciation_evidence",
                return_value={"entries": [], "outcomes": []},
            ),
            patch.object(review_module, "_fetch_text", side_effect=cold_then_ready),
            patch.object(review_module, "_search_web", AsyncMock(return_value=[])),
        ):
            failed_lookup = await review_module.collect_pronunciation_evidence("诉讼费")
            retried_lookup = await review_module.collect_pronunciation_evidence("诉讼费")
            cached_positive = await review_module.collect_pronunciation_evidence("诉讼费")

        check("all-unavailable lookup is marked incomplete", failed_lookup.get("lookupComplete") is False)
        check("failed source outcome is exposed", failed_lookup.get("sourceOutcomes") == [{
            "sourceId": "handian",
            "source": "汉典",
            "status": "errored",
            "lookupResult": "unavailable",
        }])
        check("retry re-fetches after the failure", fetch_calls == 2)
        recovered_group = next(iter(retried_lookup.get("groups") or []), {})
        check("retry recovers authoritative evidence", recovered_group.get("pinyin") == "su song fei")
        check("recovered positive result is cached", cached_positive == retried_lookup)

    asyncio.run(_run())


def test_proxy_http_404_absence_is_a_completed_source_outcome():
    print("\n🧪 proxy HTTP 404 absence is a completed source outcome")

    class ProxyResponse:
        status_code = 404

        @staticmethod
        def json():
            return {"ok": False, "status": 404}

    async def _run():
        review_module._clear_review_caches()
        source = dict(review_module._source_by_id("handian"))
        direct_fetch = AsyncMock(
            side_effect=AssertionError("proxy absence must not use direct fallback")
        )
        search_web = AsyncMock(return_value=review_module._LookupResults([]))

        with (
            patch.object(review_module, "AUTHORITATIVE_SOURCES", [source]),
            patch.object(
                review_module,
                "_collect_local_pronunciation_evidence",
                return_value={"entries": [], "outcomes": []},
            ),
            patch.object(
                review_module.http_client,
                "keytao_request",
                AsyncMock(return_value=ProxyResponse()),
            ),
            patch.object(review_module, "_fetch_text", direct_fetch),
            patch.object(review_module, "_search_web", search_web),
            patch.object(
                review_module,
                "_bot_evidence_proxy_endpoint_available",
                None,
            ),
            patch.object(
                review_module,
                "_bot_evidence_proxy_feature_probe_failed",
                False,
            ),
        ):
            evidence = await review_module.collect_pronunciation_evidence("你还")

        check("proxy absence completes the lookup", evidence.get("lookupComplete") is True)
        check("proxy absence yields no evidence", evidence.get("hasEvidence") is False)
        check("proxy absence is normalized on the source outcome", evidence.get("sourceOutcomes") == [{
            "sourceId": "handian",
            "source": "汉典",
            "status": "completed",
            "lookupResult": "absent",
        }])
        check("proxy absence skips the direct fallback", direct_fetch.await_count == 0)
        check("proxy absence skips optional search", search_web.await_count == 0)

    asyncio.run(_run())


def test_proxy_found_evidence_is_complete_through_review():
    print("\n🧪 proxy found evidence stays complete through review")

    class ProxyResponse:
        status_code = 200

        @staticmethod
        def json():
            return {
                "ok": True,
                "status": 200,
                "text": "诉讼费 拼音：sù sòng fèi 注音",
            }

    async def _run():
        review_module._clear_review_caches()
        source = dict(review_module._source_by_id("handian"))
        direct_fetch = AsyncMock(
            side_effect=AssertionError("proxy found must not use direct fallback")
        )

        with (
            patch.object(review_module, "AUTHORITATIVE_SOURCES", [source]),
            patch.object(
                review_module,
                "_collect_local_pronunciation_evidence",
                return_value={"entries": [], "outcomes": []},
            ),
            patch.object(
                review_module.http_client,
                "keytao_request",
                AsyncMock(return_value=ProxyResponse()),
            ),
            patch.object(review_module, "_fetch_text", direct_fetch),
            patch.object(
                review_module,
                "_search_web",
                AsyncMock(return_value=review_module._LookupResults([])),
            ),
            patch.object(
                review_module,
                "_bot_evidence_proxy_endpoint_available",
                None,
            ),
            patch.object(
                review_module,
                "_bot_evidence_proxy_feature_probe_failed",
                False,
            ),
        ):
            evidence = await review_module.collect_pronunciation_evidence("诉讼费")

        outcome = next(iter(evidence.get("sourceOutcomes") or []), {})
        group = next(iter(evidence.get("groups") or []), {})
        check(
            "proxy found yields complete evidence",
            evidence.get("lookupComplete") is True
            and evidence.get("hasEvidence") is True,
        )
        check(
            "proxy found is normalized on the source outcome",
            outcome.get("status") == "completed"
            and outcome.get("lookupResult") == "found",
        )
        check(
            "proxy text still passes extraction and validation",
            group.get("normalized") == ["su", "song", "fei"]
            and group.get("sourceIds") == ["handian"],
        )
        check("proxy found skips the direct fallback", direct_fetch.await_count == 0)

        review = await _review_word_with_encode(
            "诉讼费",
            evidence,
            _su_song_fei_encode(status="found", source="zdic-phrase"),
        )
        check(
            "found evidence stays complete through review",
            review.get("pronunciationEvidenceComplete") is True
            and review.get("autoReviewable") is True,
        )

    asyncio.run(_run())


def test_unreachable_optional_sources_do_not_block_completed_absence():
    print("\n🧪 unreachable optional sources do not block completed absence")

    class ProxyAbsentResponse:
        status_code = 404

        @staticmethod
        def json():
            return {"ok": False, "status": 404}

    async def _run():
        review_module._clear_review_caches()
        sources = [
            dict(review_module._source_by_id(source_id))
            for source_id in ("handian", "baidu_baike", "cidian")
        ]

        async def proxy_request(_method, _path, **kwargs):
            if kwargs.get("json_body", {}).get("sourceId") == "handian":
                return ProxyAbsentResponse()
            raise RuntimeError("simulated unavailable proxy carrier")

        async def unavailable_direct(_url, **_kwargs):
            return review_module._LookupText("", lookup_status="errored")

        async def source_search(query, max_results=3):
            status = "timed_out" if "cidian.qianp.com" in query else "errored"
            return review_module._LookupResults([], lookup_status=status)

        with (
            patch.object(review_module, "AUTHORITATIVE_SOURCES", sources),
            patch.object(
                review_module,
                "_collect_local_pronunciation_evidence",
                return_value={
                    "entries": [],
                    "outcomes": [{
                        "sourceId": "zdic_cibs",
                        "source": "汉典词典（离线数据集）",
                        "status": "completed",
                        "lookupResult": "absent",
                    }],
                },
            ),
            patch.object(
                review_module.http_client,
                "keytao_request",
                side_effect=proxy_request,
            ),
            patch.object(review_module, "_fetch_text", side_effect=unavailable_direct),
            patch.object(review_module, "_search_web", side_effect=source_search),
            patch.object(
                review_module,
                "_bot_evidence_proxy_endpoint_available",
                None,
            ),
            patch.object(
                review_module,
                "_bot_evidence_proxy_feature_probe_failed",
                False,
            ),
        ):
            evidence = await review_module.collect_pronunciation_evidence("你还")

        outcomes = {
            outcome["sourceId"]: outcome
            for outcome in evidence.get("sourceOutcomes") or []
        }
        check("reachable absences complete the lookup", evidence.get("lookupComplete") is True)
        check("completed reachable lookup still has no evidence", evidence.get("hasEvidence") is False)
        check(
            "proxy absence remains terminal",
            outcomes.get("handian", {}).get("status") == "completed"
            and outcomes.get("handian", {}).get("lookupResult") == "absent",
        )
        check(
            "errored Baidu carrier is retained as unavailable",
            outcomes.get("baidu_baike", {}).get("status") == "errored"
            and outcomes.get("baidu_baike", {}).get("lookupResult") == "unavailable",
        )
        check(
            "timed-out cidian carrier is retained as unavailable",
            outcomes.get("cidian", {}).get("status") == "timed_out"
            and outcomes.get("cidian", {}).get("lookupResult") == "unavailable",
        )

        review = await _review_word_with_encode(
            "你还",
            evidence,
            {
                "success": True,
                "word": "你还",
                "codes": ["nh"],
                "altCodes": [],
                "pronunciationSource": "pinyin-pro-context",
                "standardPronunciationStatus": "absent",
                "semanticPronunciationNeeded": False,
                "phrasePinyins": ["nǐ", "hái"],
                "contextPhrasePinyins": ["nǐ", "hái"],
                "chars": [
                    {
                        "char": "你",
                        "pinyin": "nǐ",
                        "pinyins": ["nǐ"],
                        "pronunciationLookupStatus": "found",
                    },
                    {
                        "char": "还",
                        "pinyin": "hái",
                        "pinyins": ["hái", "huán"],
                        "pronunciationLookupStatus": "found",
                    },
                ],
            },
        )
        pronunciation = next(iter(review.get("pronunciations") or []), {})
        check(
            "completed no-evidence lookup stays complete through review",
            review.get("pronunciationEvidenceComplete") is True,
        )
        check(
            "completed no-evidence lookup seals as a missing authority page",
            read_review_disposition(review) is ReviewDisposition.SEAL
            and review.get("reviewVerdictSite") == "missing_authoritative_page",
        )
        check(
            "completed no-evidence lookup never gets incomplete wording",
            "本次权威来源查询未完成" not in review.get("autoReviewReason", "")
            and "本次权威来源查询未完成"
            not in pronunciation.get("sourceSummary", ""),
        )

    asyncio.run(_run())


def test_review_fetch_retries_transient_dns_within_source_budget():
    print("\n🧪 review fetch retries cold DNS within the source budget")

    class FakeTransportError(Exception):
        pass

    class FakeTimeoutException(FakeTransportError):
        pass

    class FakeConnectTimeout(FakeTimeoutException):
        pass

    class FakeConnectError(FakeTransportError):
        pass

    async def _run():
        attempts = []

        class FetchResponse:
            status_code = 200
            is_success = True
            text = "诉讼费 拼音：sù sòng fèi"

        async def cold_dns_then_success(_url, **kwargs):
            attempts.append(kwargs.get("timeout"))
            if len(attempts) == 1:
                raise review_module.http_client.BlockedUrlError(
                    "域名解析失败：www.zdic.net（temporary failure）",
                    transient=True,
                )
            return FetchResponse()

        fake_httpx = sys.modules["httpx"]
        with (
            patch.object(fake_httpx, "ConnectTimeout", FakeConnectTimeout, create=True),
            patch.object(fake_httpx, "TimeoutException", FakeTimeoutException, create=True),
            patch.object(fake_httpx, "ConnectError", FakeConnectError, create=True),
            patch.object(fake_httpx, "TransportError", FakeTransportError, create=True),
            patch.object(review_module.http_client, "guarded_fetch", side_effect=cold_dns_then_success),
        ):
            text = await review_module._fetch_text(
                "https://www.zdic.net/hans/%E8%AF%89%E8%AE%BC%E8%B4%B9"
            )

        worst_case = (
            review_module.PRONUNCIATION_FETCH_ATTEMPT_TIMEOUT
            * review_module.PRONUNCIATION_FETCH_MAX_ATTEMPTS
            + 0.5
        )
        check("cold DNS retry returns the authoritative page", "sù sòng fèi" in text)
        check("cold DNS failure gets one bounded retry", len(attempts) == 2)
        check("each attempt uses the bounded timeout", attempts == [
            review_module.PRONUNCIATION_FETCH_ATTEMPT_TIMEOUT,
            review_module.PRONUNCIATION_FETCH_ATTEMPT_TIMEOUT,
        ])
        check("two attempts plus backoff fit the source budget", worst_case < review_module.PRONUNCIATION_SOURCE_TIMEOUT)

    asyncio.run(_run())


def test_entity_direct_fetch_keeps_its_single_attempt_budget():
    print("\n🧪 entity direct fetch keeps its original single-attempt budget")

    async def _run():
        fetch_policies = []

        async def empty_page(_url, **kwargs):
            fetch_policies.append(kwargs)
            return ""

        with (
            patch.object(review_module, "_entity_direct_source_urls", return_value=[(
                "汉典",
                "https://www.zdic.net/hans/%E8%AF%89%E8%AE%BC%E8%B4%B9",
            )]),
            patch.object(review_module, "_fetch_text", side_effect=empty_page),
        ):
            hits = await review_module._fetch_entity_direct_hits(
                "诉讼费",
                {"entityType": "common_word", "description": "诉讼案件费用"},
            )

        check("empty entity page yields no hit", hits == [])
        check("entity direct fetch does not inherit the two-attempt pronunciation policy", fetch_policies == [{
            "max_attempts": 1,
            "attempt_timeout": review_module.ENTITY_DIRECT_FETCH_ATTEMPT_TIMEOUT,
        }])
        check(
            "entity attempt keeps nearly all of its outer budget with cancellation margin",
            review_module.ENTITY_DIRECT_FETCH_ATTEMPT_TIMEOUT
            < review_module.ENTITY_DIRECT_FETCH_TIMEOUT
            and review_module.ENTITY_DIRECT_FETCH_TIMEOUT
            - review_module.ENTITY_DIRECT_FETCH_ATTEMPT_TIMEOUT
            >= 0.09,
        )

    asyncio.run(_run())


def test_pronunciation_source_timeout_is_exposed_and_not_cached():
    print("\n🧪 timed-out pronunciation lookup is not cached")

    async def _run():
        review_module._clear_review_caches()
        fetch_calls = 0
        source = {
            "id": "handian",
            "label": "汉典",
            "domain": "zdic.net",
            "category": "dictionary",
            "trust": 5,
            "query": 'site:zdic.net "{word}" 拼音',
            "direct_urls": ["https://www.zdic.net/hans/{word}"],
        }

        async def never_finishes(_url):
            nonlocal fetch_calls
            fetch_calls += 1
            await asyncio.Event().wait()

        with (
            patch.object(review_module, "AUTHORITATIVE_SOURCES", [source]),
            patch.object(
                review_module,
                "_collect_local_pronunciation_evidence",
                return_value={"entries": [], "outcomes": []},
            ),
            patch.object(review_module, "PRONUNCIATION_SOURCE_TIMEOUT", 0.01),
            patch.object(review_module, "_fetch_text", side_effect=never_finishes),
            patch.object(review_module, "_search_web", AsyncMock(return_value=[])),
        ):
            first = await review_module.collect_pronunciation_evidence("超时词")
            second = await review_module.collect_pronunciation_evidence("超时词")

        check("timed-out lookup is marked incomplete", first.get("lookupComplete") is False)
        check("timed-out source outcome is exposed", first.get("sourceOutcomes") == [{
            "sourceId": "handian",
            "source": "汉典",
            "status": "timed_out",
            "lookupResult": "unavailable",
        }])
        check("timed-out lookup is fetched again", fetch_calls == 2 and second.get("lookupComplete") is False)

    asyncio.run(_run())


def test_pronunciation_genuine_no_evidence_is_cached():
    print("\n🧪 completed pronunciation miss is cached")

    async def _run():
        review_module._clear_review_caches()
        fetch_calls = 0
        source = {
            "id": "handian",
            "label": "汉典",
            "domain": "zdic.net",
            "category": "dictionary",
            "trust": 5,
            "query": 'site:zdic.net "{word}" 拼音',
            "direct_urls": ["https://www.zdic.net/hans/{word}"],
        }

        async def no_entry(_url):
            nonlocal fetch_calls
            fetch_calls += 1
            return ""

        with (
            patch.object(review_module, "AUTHORITATIVE_SOURCES", [source]),
            patch.object(
                review_module,
                "_collect_local_pronunciation_evidence",
                return_value={"entries": [], "outcomes": []},
            ),
            patch.object(review_module, "_fetch_text", side_effect=no_entry),
            patch.object(review_module, "_search_web", AsyncMock(return_value=[])),
        ):
            first = await review_module.collect_pronunciation_evidence("不存在词")
            second = await review_module.collect_pronunciation_evidence("不存在词")

        check("completed miss is marked complete", first.get("lookupComplete") is True)
        check("completed source outcome is exposed", first.get("sourceOutcomes") == [{
            "sourceId": "handian",
            "source": "汉典",
            "status": "completed",
            "lookupResult": "absent",
        }])
        check("completed miss has no evidence", first.get("hasEvidence") is False)
        check("completed miss is served from cache", fetch_calls == 1 and second == first)

    asyncio.run(_run())


def test_reviewed_word_distinguishes_incomplete_lookup_from_completed_miss():
    print("\n🧪 reviewed-word payload distinguishes lookup failure from no evidence")

    async def review_with(evidence, *, encode_status):
        encode_data = {
            "success": True,
            "codes": ["ssfw", "ssfwo", "ssfwov"],
            "altCodes": [],
            "pronunciationSource": (
                "zdic-unavailable"
                if encode_status == "unavailable"
                else "pinyin-pro-context"
            ),
            "standardPronunciationStatus": encode_status,
            "semanticPronunciationNeeded": False,
            "chars": [
                {"char": "诉", "pinyin": "sù", "pinyins": ["sù"], "pronunciationLookupStatus": "found"},
                {"char": "讼", "pinyin": "sòng", "pinyins": ["sòng"], "pronunciationLookupStatus": "found"},
                {"char": "费", "pinyin": "fèi", "pinyins": ["fèi"], "pronunciationLookupStatus": "found"},
            ],
        }
        with (
            patch.object(review_module, "collect_pronunciation_evidence_limited", AsyncMock(return_value=evidence)),
            patch.object(review_module, "fetch_keytao_encode", AsyncMock(return_value=encode_data)),
            patch.object(review_module, "lookup_words", AsyncMock(return_value={})),
            patch.object(review_module, "lookup_codes", AsyncMock(return_value={})),
            patch.object(review_module, "_infer_entity_knowledge", AsyncMock(return_value={"recognized": False})),
            patch.object(review_module, "_contextual_pronunciation_group", AsyncMock(return_value=None)),
        ):
            return await prepare_reviewed_word(CONFIG, "诉讼费")

    async def _run():
        incomplete = await review_with({
            "success": True,
            "word": "诉讼费",
            "groups": [],
            "sources": [],
            "hasEvidence": False,
            "lookupStatus": "incomplete",
            "lookupComplete": False,
            "sourceOutcomes": [{
                "sourceId": "moedict",
                "source": "萌典",
                "status": "errored",
            }],
        }, encode_status="unavailable")
        completed = await review_with({
            "success": True,
            "word": "诉讼费",
            "groups": [],
            "sources": [],
            "hasEvidence": False,
            "lookupStatus": "completed",
            "lookupComplete": True,
            "sourceOutcomes": [{
                "sourceId": "handian",
                "source": "汉典",
                "status": "completed",
            }],
        }, encode_status="absent")

        incomplete_summary = incomplete.get("pronunciations", [{}])[0].get("sourceSummary", "")
        completed_summary = completed.get("pronunciations", [{}])[0].get("sourceSummary", "")
        check("failed lookup remains sealed", read_manual_review_flag(incomplete) is True)
        check(
            "failed lookup declares the incomplete-lookup SEAL site",
            read_review_disposition(incomplete) is ReviewDisposition.SEAL
            and incomplete.get("reviewVerdictSite") == "pronunciation_lookup_incomplete",
        )
        check("completed miss remains sealed", read_manual_review_flag(completed) is True)
        check("caller payload keeps incomplete status", incomplete.get("pronunciationEvidenceComplete") is False)
        check("caller payload keeps completed status", completed.get("pronunciationEvidenceComplete") is True)
        check("failed lookup reason says this lookup failed", "本次权威来源查询未完成" in incomplete.get("autoReviewReason", ""))
        check("failed lookup reason names the failed source", "萌典" in incomplete.get("autoReviewReason", ""))
        check("failed lookup source summary is truthful", "本次权威来源查询未完成" in incomplete_summary)
        check("completed miss keeps no-authority wording", completed.get("autoReviewReason") == "未找到权威来源，仅使用编码服务默认读音")
        check("completed miss is not mislabeled as a failed lookup", "本次权威来源查询未完成" not in completed_summary)

    asyncio.run(_run())


def test_encode_found_records_handian_authority_and_reaches_auto_approval():
    print("\n🧪 encode-found whole-word zdic authority reaches auto approval")

    async def _run():
        check(
            "whole-word zdic authority source set is exact",
            review_module.ENCODE_WHOLE_WORD_ZDIC_SOURCES
            == frozenset({"zdic-phrase", "zdic-aabb"}),
        )
        encode_data = _su_song_fei_encode(
            status="found",
            source="zdic-phrase",
        )
        review = await _review_word_with_encode(
            "诉讼费",
            _review_evidence(),
            encode_data,
        )
        pronunciation = next(iter(review.get("pronunciations") or []), {})
        sources = pronunciation.get("sources") or []
        check("encode-found is auto reviewable", review.get("autoReviewable") is True)
        check("encode-found clears the manual-review seal", read_manual_review_flag(review) is False)
        check(
            "encode-found records truthful Handian-via-encode provenance",
            sources == [{
                "source": "汉典（经编码服务）",
                "url": "https://www.zdic.net/hans/%E8%AF%89%E8%AE%BC%E8%B4%B9",
                "category": "dictionary",
                "trust": 5,
                "via": "encode-service",
                "pronunciationSource": "zdic-phrase",
            }],
        )

        with patch.object(
            review_module,
            "prepare_reviewed_word",
            AsyncMock(return_value=review),
        ):
            audit = await audit_draft_items(CONFIG, [{
                "id": 1,
                "action": "Create",
                "word": "诉讼费",
                "code": "ssfw",
                "type": "Phrase",
            }])
        check("encode-found reaches the auto-approval decision", audit.get("autoApprove") is True)
        check("auto-approval summary stays affirmative", "允许本喵自动通过" in audit.get("summary", ""))

        aabb_review = await _review_word_with_encode(
            "匆匆忙忙",
            {**_review_evidence(), "word": "匆匆忙忙"},
            {
                "success": True,
                "word": "匆匆忙忙",
                "codes": ["ccmm"],
                "altCodes": [],
                "pronunciationSource": "zdic-aabb",
                "standardPronunciationStatus": "found",
                "semanticPronunciationNeeded": False,
                "phrasePinyins": ["cōng", "cōng", "máng", "máng"],
                "contextPhrasePinyins": ["cōng", "cōng", "máng", "máng"],
                "chars": [
                    {"char": "匆", "pinyin": "cōng", "pinyins": ["cōng"], "pronunciationLookupStatus": "found"},
                    {"char": "匆", "pinyin": "cōng", "pinyins": ["cōng"], "pronunciationLookupStatus": "found"},
                    {"char": "忙", "pinyin": "máng", "pinyins": ["máng"], "pronunciationLookupStatus": "found"},
                    {"char": "忙", "pinyin": "máng", "pinyins": ["máng"], "pronunciationLookupStatus": "found"},
                ],
            },
        )
        aabb_source = next(iter(aabb_review.get("pronunciations") or []), {}).get("sources", [{}])[0]
        check(
            "zdic-aabb provenance points to the actual base entry",
            aabb_review.get("autoReviewable") is True
            and aabb_source.get("url")
            == "https://www.zdic.net/hans/%E5%8C%86%E5%BF%99",
        )

        mismatched = await _review_word_with_encode(
            "诉讼费",
            _review_evidence(),
            _su_song_fei_encode(
                status="found",
                source="zdic-phrase",
                phrase_pinyins=["shū", "sòng", "fèi"],
            ),
        )
        check(
            "encode authority still requires every syllable to be a known character reading",
            mismatched.get("autoReviewable") is False
            and not any(
                pronunciation.get("sources")
                for pronunciation in mismatched.get("pronunciations") or []
                if isinstance(pronunciation, dict)
            ),
        )

    asyncio.run(_run())


def test_encode_absent_remains_sealed_as_completed_miss():
    print("\n🧪 encode-absent is a completed Handian miss and remains sealed")

    async def _run():
        encode_data = {
            "success": True,
            "word": "阿勒泰",
            "codes": ["altt"],
            "altCodes": [],
            "pronunciationSource": "pinyin-pro-context",
            "standardPronunciationStatus": "absent",
            "semanticPronunciationNeeded": False,
            "phrasePinyins": ["ā", "lè", "tài"],
            "contextPhrasePinyins": ["ā", "lè", "tài"],
            "chars": [
                {"char": "阿", "pinyin": "ā", "pinyins": ["ā"], "pronunciationLookupStatus": "found"},
                {"char": "勒", "pinyin": "lè", "pinyins": ["lè"], "pronunciationLookupStatus": "found"},
                {"char": "泰", "pinyin": "tài", "pinyins": ["tài"], "pronunciationLookupStatus": "found"},
            ],
        }
        evidence = _review_evidence(complete=False, handian_status="timed_out")
        evidence["word"] = "阿勒泰"
        review = await _review_word_with_encode("阿勒泰", evidence, encode_data)
        encode_outcome = next(
            (
                outcome
                for outcome in review.get("pronunciationSourceOutcomes") or []
                if outcome.get("sourceId") == "handian_encode"
            ),
            {},
        )
        check("encode-absent remains sealed", read_manual_review_flag(review) is True)
        check("encode-absent is not auto reviewable", review.get("autoReviewable") is False)
        check(
            "encode-absent is recorded as a completed miss",
            encode_outcome.get("status") == "completed"
            and encode_outcome.get("lookupResult") == "absent",
        )
        check(
            "encode-absent supersedes a failed duplicate Handian scrape",
            review.get("pronunciationEvidenceComplete") is True
            and "本次权威来源查询未完成" not in review.get("autoReviewReason", ""),
        )
        check(
            "encode-absent keeps the missing-authority SEAL site",
            read_review_disposition(review) is ReviewDisposition.SEAL
            and review.get("reviewVerdictSite") == "missing_authoritative_page",
        )

    asyncio.run(_run())


def test_encode_unavailable_preserves_incomplete_lookup_semantics():
    print("\n🧪 encode-unavailable keeps incomplete lookup semantics")

    async def _run():
        review = await _review_word_with_encode(
            "诉讼费",
            _review_evidence(),
            _su_song_fei_encode(
                status="unavailable",
                source="zdic-unavailable",
            ),
        )
        check("encode-unavailable remains sealed", read_manual_review_flag(review) is True)
        check("encode-unavailable marks evidence incomplete", review.get("pronunciationEvidenceComplete") is False)
        check(
            "encode-unavailable declares the incomplete-lookup SEAL site",
            read_review_disposition(review) is ReviewDisposition.SEAL
            and review.get("reviewVerdictSite") == "pronunciation_lookup_incomplete",
        )
        check(
            "encode-unavailable reason reports an unfinished authority lookup",
            "本次权威来源查询未完成" in review.get("autoReviewReason", "")
            and "汉典（经编码服务）" in review.get("autoReviewReason", ""),
        )

    asyncio.run(_run())


def test_scraper_failure_cannot_erase_encode_found_authority():
    print("\n🧪 scraper failure cannot erase encode-found authority")

    async def _run():
        review = await _review_word_with_encode(
            "诉讼费",
            _review_evidence(complete=False, handian_status="timed_out"),
            _su_song_fei_encode(
                status="found",
                source="zdic-phrase",
            ),
        )
        pronunciation = next(iter(review.get("pronunciations") or []), {})
        check("scraper failure retains encode authority", bool(pronunciation.get("sources")))
        check("scraper failure does not seal encode-found", read_manual_review_flag(review) is False)
        check("scraper failure leaves encode-found auto reviewable", review.get("autoReviewable") is True)
        check("primary encode result completes the authority decision", review.get("pronunciationEvidenceComplete") is True)

    asyncio.run(_run())


def test_audit_never_overrides_incomplete_pronunciation_lookup():
    print("\n🧪 audit keeps incomplete pronunciation lookups sealed")

    incomplete_review = {
        "success": True,
        "word": "诉讼费",
        "autoReviewable": False,
        "autoReviewReason": "本次权威来源查询未完成（汉典），本轮仍需管理员审核",
        "needsManualReview": True,
        "manualReviewReason": "本次权威来源查询未完成（汉典），本轮仍需管理员审核",
        "reviewDisposition": "SEAL",
        "reviewVerdictSite": "pronunciation_lookup_incomplete",
        "lookupFailed": False,
        "pronunciationEvidenceComplete": False,
        "pronunciationSourceOutcomes": [{
            "sourceId": "handian",
            "source": "汉典",
            "status": "errored",
        }],
        "existing": [],
        "pronunciations": [{
            "pinyin": "su song fei",
            "codes": ["ssfw", "ssfwo", "ssfwov"],
            "sources": [],
            "fallback": True,
        }],
    }
    authoritative_review = {
        "success": True,
        "word": "新词",
        "autoReviewable": True,
        "needsManualReview": False,
        "lookupFailed": False,
        "pronunciationEvidenceComplete": True,
        "existing": [],
        "pronunciations": [{
            "pinyin": "xin ci",
            "codes": ["xkck"],
            "sources": [{"source": "汉典", "url": "https://example.test"}],
        }],
    }

    async def _run():
        commonness = AsyncMock(return_value={
            "success": True,
            "score": 0.99,
            "signals": {"corpus": 1.0, "search": 1.0, "dictionary": 1.0},
            "entityKnowledge": {"accepted": False},
        })
        with (
            patch.object(review_module, "prepare_reviewed_word", AsyncMock(return_value=incomplete_review)),
            patch.object(review_module, "estimate_word_commonness", commonness),
        ):
            create_audit = await audit_draft_items(CONFIG, [{
                "action": "Create",
                "word": "诉讼费",
                "code": "ssfw",
                "type": "Phrase",
            }])

        async def review_for_change(_config, word):
            return incomplete_review if word == "旧词" else authoritative_review

        with patch.object(review_module, "prepare_reviewed_word", side_effect=review_for_change):
            change_audit = await audit_draft_items(CONFIG, [{
                "action": "Change",
                "old_word": "旧词",
                "word": "新词",
                "code": "xkck",
                "type": "Phrase",
            }])

        check("commonness cannot override an incomplete lookup", commonness.await_count == 0)
        check("incomplete create remains non-approvable", create_audit.get("autoApprove") is False)
        check("incomplete create yields no approved item", create_audit.get("approvedItems") == [])
        check("incomplete create issue names the failed lookup", any(
            "本次权威来源查询未完成" in issue and "汉典" in issue
            for issue in create_audit.get("issues", [])
        ))
        check("incomplete create issue is structurally sealed", any(
            "本次权威来源查询未完成" in issue
            for issue in create_audit.get("structuredManualReviewIssues", [])
        ))
        check("incomplete old side blocks change approval", change_audit.get("autoApprove") is False)
        check("incomplete change yields no approved item", change_audit.get("approvedItems") == [])
        check("incomplete change is not mislabeled as no authoritative evidence", all(
            "旧词未找到权威证据" not in issue
            for issue in change_audit.get("issues", [])
        ) and any(
            "权威来源查询未完成" in issue
            for issue in change_audit.get("issues", [])
        ))

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
        check(
            "review tool promotes the comparator verdict to the one recommendation",
            tool_result.get("recommendedCode") == "eefj"
            and tool_result["pronunciations"][0].get("recommendedCode") == "eefj",
        )

    asyncio.run(_run())


def test_modern_semantic_vs_dictionary_dominated_commonness_matrix():
    """Only the confident-modern/weak-classical asymmetry overrides baseline."""
    print("\n🧪 modern semantic vs dictionary-dominated commonness matrix")

    def review(*, modern=True):
        semantic_items = []
        if modern:
            semantic_items = [{
                "word": "冒菜",
                "code": "mzchi",
                "assessment": {
                    "accepted": True,
                    "confidence": 0.96,
                    "meaning": "现代常用饮食词",
                    "nonObscurity": {
                        "route": "common_characters_and_llm",
                        "characterReferences": [
                            {"char": "冒", "corpusFrequency": 5231},
                            {"char": "菜", "corpusFrequency": 8544},
                        ],
                    },
                },
            }]
        return {
            "success": True,
            "word": "冒菜",
            "recommendedCode": "mzchi",
            "preSubmitAudit": {
                "semanticContextAutoPassItems": semantic_items,
            },
            "pronunciations": [{
                "recommendedCode": "mzchi",
                "candidateStatuses": [
                    {
                        "code": "mzch",
                        "occupied": True,
                        "words": ["茂才"],
                        "phrases": [{
                            "word": "茂才",
                            "code": "mzch",
                            "type": "Phrase",
                        }],
                    },
                    {"code": "mzchi", "occupied": False, "words": []},
                ],
            }],
        }

    def comparison(
        verdict,
        *,
        candidate_frequency=None,
        candidate_presence=0,
        occupant_frequency=None,
        occupant_presence=2,
    ):
        return {
            "success": True,
            "verdict": verdict,
            "summary": "baseline comparison",
            "decisionReason": "baseline",
            "front": {"reference": {
                "available": True,
                "attested": candidate_frequency is not None or candidate_presence > 0,
                "corpusFrequency": candidate_frequency,
                "dictionaryPresenceCount": candidate_presence,
            }},
            "behind": {"reference": {
                "available": True,
                "attested": occupant_frequency is not None or occupant_presence > 0,
                "corpusFrequency": occupant_frequency,
                "dictionaryPresenceCount": occupant_presence,
            }},
        }

    async def assess(subject, baseline):
        with patch.object(
            review_module,
            "compare_word_commonness",
            AsyncMock(return_value=baseline),
        ):
            return (await review_module.assess_candidate_chain_commonness(subject))[0]

    async def _run():
        archaic_asymmetry = await assess(
            review(),
            comparison("behind_more_common", occupant_frequency=None),
        )
        equal_modern = await assess(
            review(),
            comparison(
                "behind_more_common",
                candidate_frequency=40,
                candidate_presence=1,
                occupant_frequency=80,
                occupant_presence=1,
            ),
        )
        equal_classical = await assess(
            review(modern=False),
            comparison(
                "close",
                candidate_presence=2,
                occupant_presence=2,
            ),
        )
        conflicting = await assess(
            review(),
            comparison(
                "behind_more_common",
                occupant_frequency=30,
                occupant_presence=2,
            ),
        )

        check(
            "modern food word outranks dictionary-dominated archaic incumbent",
            archaic_asymmetry.get("verdict") == "front_more_common"
            and archaic_asymmetry.get("newCode") == "mzch"
            and archaic_asymmetry.get("decisionReason")
            == "modern_semantic_vs_dictionary_dominated"
            and archaic_asymmetry.get("summary")
            == "冒菜：现代常用饮食词（语义判断）；茂才：古语，词典收录但语料频次低",
        )
        check(
            "equal-modern comparison keeps baseline thresholds",
            equal_modern.get("verdict") == "behind_more_common"
            and equal_modern.get("decisionReason") == "baseline",
        )
        check(
            "equal-classical comparison keeps baseline thresholds",
            equal_classical.get("verdict") == "close"
            and equal_classical.get("decisionReason") == "baseline",
        )
        check(
            "genuine modern corpus conflict stays conservative",
            conflicting.get("verdict") == "behind_more_common"
            and conflicting.get("decisionReason") == "baseline",
        )

        import keytao_bot.plugins.openai_chat as chat

        rendered = chat._format_candidate_ordering_assessment(
            archaic_asymmetry,
            {"mzch": 1, "mzchi": 2},
        )
        check(
            "ordering copy cites both semantic and dictionary/corpus evidence",
            "冒菜：现代常用饮食词（语义判断）" in rendered
            and "茂才：古语，词典收录但语料频次低" in rendered
            and "推荐：「冒菜」占 mzch、「茂才」顺延" in rendered
            and "不重排选 2（mzchi）" in rendered,
        )

    asyncio.run(_run())


def test_existing_code_chain_commonness_ranking_uses_modern_override_and_asks_on_unknown():
    """Existing-chain ranking shares the comparator and fails closed on no evidence."""
    print("\n🧪 existing code-chain commonness ranking")

    entries = [
        {"word": "茂才", "code": "mkdr", "type": "Phrase", "weight": 100},
        {"word": "冒菜", "code": "mkdr", "type": "Phrase", "weight": 101},
    ]
    baseline = {
        "success": True,
        "verdict": "front_more_common",
        "summary": "「茂才」较「冒菜」更常用：词典信号",
        "decisionReason": "dictionary_presence_margin",
        "front": {"reference": {
            "available": True,
            "attested": True,
            "corpusFrequency": None,
            "dictionaryPresenceCount": 2,
        }},
        "behind": {"reference": {
            "available": True,
            "attested": False,
            "corpusFrequency": None,
            "dictionaryPresenceCount": 0,
        }},
    }
    modern_review = {
        "preSubmitAudit": {
            "semanticContextAutoPassItems": [{
                "word": "冒菜",
                "assessment": {
                    "accepted": True,
                    "confidence": 0.96,
                    "meaning": "现代常用饮食词",
                    "nonObscurity": {
                        "route": "common_characters_and_llm",
                        "characterReferences": [
                            {"corpusFrequency": 5231},
                            {"corpusFrequency": 8544},
                        ],
                    },
                },
            }],
        },
    }

    async def _run():
        semantic_loader = AsyncMock(return_value=modern_review)
        with patch.object(
            review_module,
            "compare_word_commonness",
            AsyncMock(return_value=baseline),
        ):
            ranked = await review_module.rank_code_chain_by_commonness(
                entries,
                semantic_review_loader=semantic_loader,
            )
        check(
            "modern word moves ahead of dictionary-dominated archaic word",
            ranked.get("status") == "reorder"
            and [item["word"] for item in ranked.get("proposedOrder", [])]
            == ["冒菜", "茂才"],
        )
        check(
            "modern override evidence is retained",
            ranked.get("comparisons", [{}])[0].get("decisionReason")
            == "modern_semantic_vs_dictionary_dominated",
        )
        check(
            "semantic evidence is loaded only for the possible modern side",
            semantic_loader.await_count == 1
            and semantic_loader.await_args.args == ("冒菜",),
        )

        with patch.object(
            review_module,
            "compare_word_commonness",
            AsyncMock(return_value={
                "success": True,
                "verdict": "not_enough_evidence",
                "summary": "常用度信号不足",
                "front": {},
                "behind": {},
            }),
        ):
            unknown = await review_module.rank_code_chain_by_commonness(entries)
        check(
            "no-evidence tie produces deterministic ASK",
            unknown.get("status") == "ask"
            and unknown.get("reason") == "not_enough_evidence",
        )

        with patch.object(
            review_module,
            "compare_word_commonness",
            AsyncMock(return_value={
                "success": True,
                "verdict": "close",
                "summary": "两词接近",
                "front": {},
                "behind": {},
            }),
        ):
            unevidenced_close = await review_module.rank_code_chain_by_commonness(entries)
        check(
            "close result without any evidence also produces deterministic ASK",
            unevidenced_close.get("status") == "ask"
            and unevidenced_close.get("reason") == "not_enough_evidence",
        )

        with patch.object(
            review_module,
            "compare_word_commonness",
            AsyncMock(return_value={
                "success": True,
                "verdict": "unexpected",
                "summary": "未知比较结论",
                "front": baseline["front"],
                "behind": baseline["behind"],
            }),
        ):
            unexpected = await review_module.rank_code_chain_by_commonness(entries)
        check(
            "unknown comparator verdict fails closed",
            unexpected.get("status") == "ask"
            and unexpected.get("reason") == "not_enough_evidence",
        )

    asyncio.run(_run())


def test_audit_budget_nesting_and_timeout_retains_review():
    print("\n🧪 audit budget nesting and partial-result retention")

    check(
        "audit stages declare gating or advisory behavior centrally",
        {
            stage: policy.get("classification")
            for stage, policy in review_module.AUDIT_STAGE_POLICIES.items()
        } == {
            "review": "gating",
            "css_review": "gating",
            "commonness": "advisory",
            "change_commonness": "advisory",
            "priority": "advisory",
        },
    )

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

        async def recommended_priority(item, _review):
            return {
                "word": item["word"],
                "code": item["code"],
                "hasRecommendation": True,
                "advisory": True,
                "commonness": {},
                "recommendedMoves": [{"word": item["word"], "toCode": "eksoa"}],
            }

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
            patch.object(review_module, "AUDIT_PRIORITY_STAGE_TIMEOUT", 0.01),
            patch.object(review_module, "AUDIT_ITEM_TIMEOUT", 0.10),
            patch.object(review_module, "current_turn_id", return_value="a1b2c3d4"),
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
            "advisory timeout preserves the word-review verdict",
            audit.get("verdict") == "pass"
            and audit.get("autoApprove") is True
            and audit.get("issues") == []
            and all("需管理员审核" not in issue for issue in audit.get("issues", [])),
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
            and audit_log_lines[0].endswith("turn_id=a1b2c3d4")
            and "\n" not in audit_log_lines[0],
        )

        with (
            patch.object(
                review_module,
                "prepare_reviewed_word",
                new=AsyncMock(return_value=resolved_review),
            ),
            patch.object(
                review_module,
                "_review_code_chain_priority",
                side_effect=recommended_priority,
            ),
        ):
            recommendation_audit = await audit_draft_items(CONFIG, [{
                "id": 2,
                "action": "Create",
                "word": "石蒜",
                "code": "ekso",
                "type": "Phrase",
            }])
        check(
            "advisory recommendation cannot downgrade the verdict",
            recommendation_audit.get("verdict") == "pass"
            and recommendation_audit.get("issues") == []
            and recommendation_audit.get("codeChainPriorityReviews", [{}])[0].get("hasRecommendation") is True,
        )

    asyncio.run(_run())


def test_multi_sense_agreeing_evidence_recommends_authoritative_reading():
    print("\n🧪 multi-sense agreeing evidence recommends authoritative reading")

    async def _run():
        evidence = {
            "success": True,
            "groups": [{
                "pinyin": "hái chē",
                "normalized": ["hai", "che"],
                "sources": [{
                    "source": "汉典（离线数据集）",
                    "url": "",
                    "category": "dictionary",
                    "trust": 5,
                }],
                "sourceIds": ["zdic_cibs"],
                "score": 5,
                "fallback": False,
            }],
            "sources": [],
            "lookupComplete": True,
            "sourceOutcomes": [{
                "sourceId": "zdic_cibs",
                "source": "汉典（离线数据集）",
                "status": "completed",
                "lookupResult": "found",
            }],
        }
        encode_data = {
            "success": True,
            "word": "还车",
            "codes": ["htje", "htjev", "htjevv"],
            "altCodes": ["htwe", "htwev", "htwevv"],
            "candidateCodes": [
                "htje", "htjev", "htjevv", "htwe", "htwev", "htwevv",
            ],
            "alternatePhrasePronunciationCodes": [{
                "char": "还",
                "charIndex": 0,
                "pinyin": "hái",
                "codes": ["htwe", "htwev", "htwevv"],
            }],
            "pronunciationSource": "zdic-phrase",
            "standardPronunciationStatus": "found",
            "phrasePinyins": ["huán", "chē"],
            "contextPhrasePinyins": ["huán", "chē"],
            "chars": [
                {
                    "char": "还",
                    "pinyin": "huán",
                    "pinyins": ["huán", "hái"],
                    "pronunciationLookupStatus": "found",
                    "phoneticCode": "h",
                    "shapeCode": "t",
                },
                {
                    "char": "车",
                    "pinyin": "chē",
                    "pinyins": ["chē"],
                    "pronunciationLookupStatus": "found",
                    "phoneticCode": "j",
                    "shapeCode": "e",
                },
            ],
        }
        agreeing_proposal = {
            "accepted": True,
            "word": "还车",
            "pinyins": ["huan", "che"],
            "meaning": "把租用或借用的车辆归还给原主或服务方",
            "confidence": 0.98,
            "commonTransparent": True,
            "commonnessReason": "归还车辆是明确且常见的现代汉语用法",
            "usageType": "transparent_compound",
        }

        async def prepare(proposal, prepared_encode=encode_data):
            async def encode_for_reading(_config, _word):
                return prepared_encode

            with (
                patch.object(
                    review_module,
                    "collect_pronunciation_evidence_limited",
                    AsyncMock(return_value=evidence),
                ),
                patch.object(
                    review_module,
                    "fetch_keytao_encode",
                    AsyncMock(side_effect=encode_for_reading),
                ),
                patch.object(review_module, "lookup_words", AsyncMock(return_value={})),
                patch.object(review_module, "lookup_codes", AsyncMock(return_value={})),
                patch.object(
                    review_module,
                    "_infer_semantic_pronunciation_for_review",
                    AsyncMock(return_value=proposal),
                ),
            ):
                return await prepare_reviewed_word(CONFIG, "还车")

        agreed = await prepare(agreeing_proposal)
        agreed_groups = agreed.get("pronunciations", [])
        check(
            "whole-word authority and meaning agreement recommends huan che",
            agreed.get("multiSenseChoice", {}).get("status") == "resolved"
            and agreed_groups[0].get("normalized") == ["huan", "che"]
            and agreed.get("recommendedCode") == "htje"
            and agreed.get("autoReviewable") is True,
        )
        check(
            "both agreeing-case reading groups remain visible",
            {tuple(group.get("normalized", [])) for group in agreed_groups}
            == {("huan", "che"), ("hai", "che")},
        )
        codes_by_reading = {
            tuple(group.get("normalized", [])): group.get("codes", [])
            for group in agreed_groups
        }
        check(
            "service-returned reading groups keep their own chains",
            codes_by_reading.get(("huan", "che")) == ["htje", "htjev", "htjevv"]
            and codes_by_reading.get(("hai", "che")) == ["htwe", "htwev", "htwevv"],
        )

        unscoped_encode = dict(encode_data)
        unscoped_encode.pop("alternatePhrasePronunciationCodes")
        unscoped_encode["candidateCodes"] = None
        unscoped = await prepare(agreeing_proposal, unscoped_encode)
        unscoped_codes = {
            tuple(group.get("normalized", [])): group.get("codes", [])
            for group in unscoped.get("pronunciations", [])
        }
        check(
            "sole reviewed alternate binds the ordinary response altCodes chain",
            unscoped_codes.get(("hai", "che")) == ["htwe", "htwev", "htwevv"],
        )

        authority_only = await prepare({"accepted": False, "word": "还车"})
        check(
            "sole whole-word authority recommends when no decisive source contradicts it",
            authority_only.get("multiSenseChoice", {}).get("status") == "resolved"
            and authority_only.get("recommendedCode") == "htje"
            and authority_only.get("pronunciationUnresolved") is not True,
        )

    asyncio.run(_run())


def test_multi_sense_conflicting_evidence_asks_for_clarification():
    print("\n🧪 multi-sense conflicting evidence asks for clarification")

    async def _run():
        evidence = {
            "success": True,
            "groups": [{
                "pinyin": "chū quān",
                "normalized": ["chu", "quan"],
                "sources": [{
                    "source": "现代用法证据",
                    "url": "https://example.test/chuquan",
                    "category": "dictionary",
                    "trust": 4,
                }],
                "sourceIds": ["modern-usage"],
                "score": 4,
                "fallback": False,
            }],
            "sources": [],
            "lookupComplete": True,
            "sourceOutcomes": [{
                "sourceId": "handian",
                "source": "汉典",
                "status": "completed",
                "lookupResult": "found",
            }],
        }
        encode_data = {
            "success": True,
            "word": "出圈",
            "codes": ["jjjt", "jjjto", "jjjtou"],
            "altCodes": [],
            "pronunciationSource": "zdic-phrase",
            "standardPronunciationStatus": "found",
            "phrasePinyins": ["chū", "juàn"],
            "contextPhrasePinyins": ["chū", "juàn"],
            "chars": [
                {
                    "char": "出",
                    "pinyin": "chū",
                    "pinyins": ["chū"],
                    "pronunciationLookupStatus": "found",
                    "phoneticCode": "j",
                    "shapeCode": "t",
                },
                {
                    "char": "圈",
                    "pinyin": "juàn",
                    "pinyins": ["juàn", "quān"],
                    "pronunciationLookupStatus": "found",
                    "phoneticCode": "j",
                    "shapeCode": "t",
                },
            ],
        }
        resolved_proposal = {
            "accepted": True,
            "word": "出圈",
            "pinyins": ["chu", "quan"],
            "meaning": "指作品、人物或话题突破原有圈层并获得更广泛传播",
            "confidence": 0.97,
            "commonTransparent": True,
            "commonnessReason": "现代网络语境中的常见用法",
            "usageType": "modern_word",
        }

        async def encode_for_reading(_config, _word):
            return encode_data

        async def prepare(proposal):
            with (
                patch.object(
                    review_module,
                    "collect_pronunciation_evidence_limited",
                    AsyncMock(return_value=evidence),
                ),
                patch.object(
                    review_module,
                    "fetch_keytao_encode",
                    AsyncMock(side_effect=encode_for_reading),
                ),
                patch.object(review_module, "lookup_words", AsyncMock(return_value={})),
                patch.object(review_module, "lookup_codes", AsyncMock(return_value={})),
                patch.object(
                    review_module,
                    "_infer_semantic_pronunciation_for_review",
                    AsyncMock(return_value=proposal),
                ),
            ):
                return await prepare_reviewed_word(CONFIG, "出圈")

        conflicted = await prepare(resolved_proposal)
        conflict = conflicted.get("multiSenseChoice", {})
        conflicted_groups = conflicted.get("pronunciations", [])
        assessment = review_module._assess_semantic_context_auto_pass(
            "出圈",
            "jjjt",
            conflicted,
        )
        check(
            "whole-word authority conflicting with meaning and modern usage asks",
            conflict.get("status") == "ambiguous"
            and conflicted.get("pronunciationUnresolved") is True
            and conflicted.get("recommendedCode") == ""
            and "chu juan" in conflicted.get("message", "")
            and "chū quān" in conflicted.get("message", ""),
        )
        check(
            "both conflicting reading groups remain visible",
            {tuple(group.get("normalized", [])) for group in conflicted_groups}
            == {("chu", "juan"), ("chu", "quan")},
        )
        check(
            "conflicting multi-sense choice cannot enter semantic auto-pass",
            conflicted.get("autoReviewable") is False
            and assessment.get("accepted") is False
            and "multiSenseResolved" in assessment.get("failedChecks", []),
        )

    asyncio.run(_run())


def test_explicit_reading_selects_one_group_from_the_single_encode_result():
    print("\n🧪 explicit reading selects one group from the single encode result")

    async def _run():
        evidence = {
            "success": True,
            "groups": [{
                "pinyin": "chū quān",
                "normalized": ["chu", "quan"],
                "sources": [{
                    "source": "现代用法证据",
                    "url": "https://example.test/chuquan",
                    "category": "dictionary",
                    "trust": 4,
                }],
                "sourceIds": ["modern-usage"],
                "score": 4,
                "fallback": False,
            }],
            "sources": [],
            "lookupComplete": True,
            "sourceOutcomes": [{
                "sourceId": "handian",
                "source": "汉典",
                "status": "completed",
                "lookupResult": "found",
            }],
        }
        baseline = {
            "success": True,
            "word": "出圈",
            "codes": ["jjjt", "jjjto", "jjjtou"],
            "altCodes": [],
            "candidateCodes": [
                "jjjt", "jjjto", "jjjtou", "jjqt", "jjqta", "jjqtai",
            ],
            "alternatePhrasePronunciationCodes": [{
                "char": "圈",
                "charIndex": 1,
                "pinyin": "quān",
                "codes": ["jjqt", "jjqta", "jjqtai"],
            }],
            "pronunciationSource": "zdic-phrase",
            "standardPronunciationStatus": "found",
            "phrasePinyins": ["chū", "juàn"],
            "contextPhrasePinyins": ["chū", "juàn"],
            "chars": [
                {
                    "char": "出",
                    "pinyin": "chū",
                    "pinyins": ["chū"],
                    "pronunciationLookupStatus": "found",
                    "phoneticCode": "j",
                    "shapeCode": "t",
                },
                {
                    "char": "圈",
                    "pinyin": "juàn",
                    "pinyins": ["juàn", "quān"],
                    "pronunciationLookupStatus": "found",
                    "phoneticCode": "j",
                    "shapeCode": "t",
                },
            ],
        }
        encode_mock = AsyncMock(return_value=baseline)
        with (
            patch.object(
                review_module,
                "collect_pronunciation_evidence_limited",
                AsyncMock(return_value=evidence),
            ),
            patch.object(review_module, "fetch_keytao_encode", encode_mock),
            patch.object(review_module, "lookup_words", AsyncMock(return_value={})),
            patch.object(review_module, "lookup_codes", AsyncMock(return_value={})),
            patch.object(
                review_module,
                "_infer_semantic_pronunciation_for_review",
                AsyncMock(side_effect=AssertionError("explicit reading must bypass semantic guess")),
            ),
        ):
            reviewed = await prepare_reviewed_word(
                CONFIG,
                "出圈",
                requested_reading="chū quān",
            )

        check(
            "requested reading reuses the one ordinary encode call",
            encode_mock.await_count == 1
            and encode_mock.await_args.kwargs == {},
        )
        check(
            "only the explicitly selected reading group is rendered",
            reviewed.get("pronunciationUnresolved") is not True
            and reviewed.get("multiSenseChoice", {}).get("status") == "resolved"
            and [group.get("normalized") for group in reviewed.get("pronunciations", [])]
            == [["chu", "quan"]]
            and reviewed.get("recommendedCode") == "jjqt",
        )
        check(
            "a choice differing from the authoritative whole-word reading stays sealed",
            reviewed.get("needsManualReview") is True
            and reviewed.get("requiresManualPronunciationReview") is True,
        )
        import keytao_bot.plugins.openai_chat as chat

        sealed_preview = {
            **reviewed,
            "preSubmitAudit": {
                "success": True,
                "verdict": "pass",
                "autoApprove": True,
                "summary": "权威来源、编码和常用度证据一致",
                "issues": [],
                "approvedItems": ["出圈@jjqt"],
            },
        }
        sealed_prompt = chat._format_reviewed_add_prompt(sealed_preview) or ""
        check(
            "rendering cannot advertise auto-pass for an explicit-reading seal",
            "需要管理员审核" in sealed_prompt
            and "可自动通过" not in sealed_prompt,
        )

        absent_baseline = {
            **baseline,
            "pronunciationSource": "zdic-character-default",
            "standardPronunciationStatus": "absent",
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
                AsyncMock(return_value=absent_baseline),
            ),
            patch.object(review_module, "lookup_words", AsyncMock(return_value={})),
            patch.object(review_module, "lookup_codes", AsyncMock(return_value={})),
        ):
            selected_without_whole_word_page = await prepare_reviewed_word(
                CONFIG,
                "出圈",
                requested_reading="圈=quan",
            )
        check(
            "a non-default explicit reading stays sealed without a whole-word page",
            selected_without_whole_word_page.get("needsManualReview") is True
            and selected_without_whole_word_page.get(
                "requiresManualPronunciationReview"
            ) is True,
        )

        for requested in ("圈=quan", "chū quān"):
            one_call_encode = AsyncMock(return_value=baseline)
            with (
                patch.object(
                    review_module,
                    "collect_pronunciation_evidence_limited",
                    AsyncMock(return_value=evidence),
                ),
                patch.object(
                    review_module,
                    "fetch_keytao_encode",
                    one_call_encode,
                ),
                patch.object(review_module, "lookup_words", AsyncMock(return_value={})),
                patch.object(review_module, "lookup_codes", AsyncMock(return_value={})),
            ):
                selected_by_form = await prepare_reviewed_word(
                    CONFIG,
                    "出圈",
                    requested_reading=requested,
                )
            check(
                f"reading selector form chooses the returned quan group: {requested}",
                one_call_encode.await_count == 1
                and selected_by_form.get("recommendedCode") == "jjqt",
            )

        unmatched_encode = AsyncMock(return_value=baseline)
        with (
            patch.object(
                review_module,
                "collect_pronunciation_evidence_limited",
                AsyncMock(return_value=evidence),
            ),
            patch.object(
                review_module,
                "fetch_keytao_encode",
                unmatched_encode,
            ),
            patch.object(review_module, "lookup_words", AsyncMock(return_value={})),
        ):
            unmatched = await prepare_reviewed_word(
                CONFIG,
                "出圈",
                requested_reading="chū qióng",
            )
        unmatched_message = unmatched.get("message", "")
        check(
            "a missing reading lists only the readings returned by encode",
            unmatched.get("pronunciationUnresolved") is True
            and "都不匹配" in unmatched_message
            and "chū juàn" in unmatched_message
            and "chū quān" in unmatched_message
            and "管理员" not in unmatched_message
            and "复算" not in unmatched_message,
        )

        meaning_encode = AsyncMock(return_value=baseline)
        with (
            patch.object(
                review_module,
                "collect_pronunciation_evidence_limited",
                AsyncMock(return_value=evidence),
            ),
            patch.object(review_module, "fetch_keytao_encode", meaning_encode),
            patch.object(review_module, "lookup_words", AsyncMock(return_value={})),
            patch.object(review_module, "lookup_codes", AsyncMock(return_value={})),
            patch.object(
                review_module,
                "_infer_requested_meaning_pronunciation_for_review",
                AsyncMock(return_value={
                    "accepted": True,
                    "pinyins": ["chu", "quan"],
                    "meaning": "指作品走红并突破原有圈层",
                    "confidence": 0.98,
                }),
            ) as meaning_mapper,
        ):
            meaning_selected = await prepare_reviewed_word(
                CONFIG,
                "出圈",
                requested_meaning="作品走红、突破原有圈层的用法",
            )
        check(
            "a concrete sense maps to one returned reading group",
            meaning_mapper.await_count == 1
            and meaning_encode.await_count == 1
            and meaning_selected.get("recommendedCode") == "jjqt"
            and meaning_selected.get("multiSenseChoice", {}).get("method")
            == "user_meaning_selected_encode_group",
        )

    asyncio.run(_run())


def main():
    test_review_disposition_registry()
    test_semantic_context_auto_pass_corpus_and_mutation_matrix()
    test_rejected_offline_whole_word_reading_cannot_auto_pass_by_dictionary_presence()
    test_semantic_context_full_vendored_corpus_facts()
    test_semantic_context_pass_clears_prepare_seal_and_enters_autoapprove_chain()
    test_chanji_semantic_prepare_revalidation_reaches_common_char_pass()
    test_chanji_entity_context_reaches_common_char_pass()
    test_local_reference_import_is_deterministic_and_preserves_readings()
    test_reference_version_mismatch_rebuilds_and_missing_schema_warns()
    test_offline_commonness_verdict_rules_and_copy()
    test_both_absent_commonness_uses_existing_bounded_web_fallback()
    test_collector_queries_local_reference_first_and_scores_agreement()
    test_local_reference_miss_falls_through_to_live_sources()
    test_poisoned_local_reference_row_fails_per_syllable_validation()
    test_s14_wrong_entry_pronunciation_never_reaches_candidates()
    test_reviewed_add_chi_xi_no_authoritative_entry_or_web_uses_verified_own_characters()
    test_pronunciation_word_binding_window_and_exact_direct_entry()
    test_hwxnet_real_fixture_extracts_honest_provenance()
    test_hwxnet_follow_requires_exact_anchor_text()
    test_hwxnet_poisoned_fixture_fails_per_syllable_validation()
    test_pronunciation_groups_require_known_character_readings()
    test_pronunciation_source_failure_is_not_cached_and_retry_refetches()
    test_proxy_http_404_absence_is_a_completed_source_outcome()
    test_proxy_found_evidence_is_complete_through_review()
    test_unreachable_optional_sources_do_not_block_completed_absence()
    test_review_fetch_retries_transient_dns_within_source_budget()
    test_entity_direct_fetch_keeps_its_single_attempt_budget()
    test_pronunciation_source_timeout_is_exposed_and_not_cached()
    test_pronunciation_genuine_no_evidence_is_cached()
    test_reviewed_word_distinguishes_incomplete_lookup_from_completed_miss()
    test_encode_found_records_handian_authority_and_reaches_auto_approval()
    test_encode_absent_remains_sealed_as_completed_miss()
    test_encode_unavailable_preserves_incomplete_lookup_semantics()
    test_scraper_failure_cannot_erase_encode_found_authority()
    test_audit_never_overrides_incomplete_pronunciation_lookup()
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
    test_modern_semantic_vs_dictionary_dominated_commonness_matrix()
    test_existing_code_chain_commonness_ranking_uses_modern_override_and_asks_on_unknown()
    test_audit_budget_nesting_and_timeout_retains_review()
    test_multi_sense_agreeing_evidence_recommends_authoritative_reading()
    test_multi_sense_conflicting_evidence_asks_for_clarification()
    test_explicit_reading_selects_one_group_from_the_single_encode_result()

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{passed + failed} passed" + (f", {failed} failed" if failed else ""))
    if failed:
        print("❌ SOME TESTS FAILED")
        return 1
    print("✅ ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
