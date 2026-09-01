"""Scenario pack v1 with end-state and reply-marker assertions."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit

from keytao_bot.utils.pending_confirmation import (
    SYSTEM_REPLY_TEMPLATE_MARKERS,
    UNBOUND_BINDING_PRECHECK_NOTICE,
    advertised_batch_binding_pairs,
    advertised_reply_contract,
    pending_confirmation_copy,
)
from keytao_bot.plugins.chat_render import (
    _reply_has_internal_fragment,
    public_base_for_platform,
    render_platform_public_links,
)

from .recording import ArtifactRecorder
from .runtime import E2EBotHarness, LocalNextClient
from .safety import (
    EncodeDelayController,
    PronunciationPoisonController,
    SafetyViolation,
)


PHRASE_TYPE_BASE_WEIGHTS = {
    "Single": 10,
    "Phrase": 100,
    "Supplement": 100,
    "Symbol": 10,
    "Link": 10000,
    "CSS": 100,
    "CSSSingle": 10,
    "English": 100,
}


class ScenarioAssertionError(AssertionError):
    """A fact or stable reply marker did not match the scenario contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ScenarioAssertionError(message)


def item_key(item: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(item.get("action") or ""),
        str(item.get("word") or ""),
        str(item.get("code") or ""),
    )


def same_unique_item_set(actual: object, expected: object) -> bool:
    """Compare complete item collections without treating insertion order as data."""
    if not isinstance(actual, (list, tuple)) or not isinstance(expected, (list, tuple)):
        return False
    actual_items = tuple(actual)
    expected_items = tuple(expected)
    return bool(
        len(actual_items) == len(expected_items)
        and len(actual_items) == len(set(actual_items))
        and len(expected_items) == len(set(expected_items))
        and set(actual_items) == set(expected_items)
    )


def ordered_candidate_codes(payload: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for key in ("codes", "altCodes"):
        values = payload.get(key)
        if not isinstance(values, list):
            continue
        for value in values:
            code = str(value or "").strip().lower()
            if code and "?" not in code and code not in result:
                result.append(code)
    for item in payload.get("candidateStatuses", []):
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or "").strip().lower()
        if code and "?" not in code and code not in result:
            result.append(code)
    return result


def assert_no_code_request(reply: str) -> None:
    blocked = (
        r"请.{0,8}(?:提供|告诉).{0,6}编码",
        r"(?:提供|告诉).{0,8}(?:编码|代码)",
        r"哪个编码",
        r"选择.{0,6}编码",
    )
    require(
        not any(re.search(pattern, reply) for pattern in blocked),
        f"reply delegated code discovery to the user: {reply}",
    )


def assert_reply_mentions(reply: str, *markers: str) -> None:
    missing = [marker for marker in markers if marker not in reply]
    require(not missing, f"reply omitted stable markers {missing}: {reply}")


def duplicate_visible_lines(reply: str) -> tuple[str, ...]:
    """Return normalized non-empty lines that were displayed more than once."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for raw_line in reply.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        if line in seen and line not in duplicates:
            duplicates.append(line)
        seen.add(line)
    return tuple(duplicates)


def batch_link_ids(reply: str) -> set[str]:
    return set(re.findall(r"/batch/([A-Za-z0-9_-]+)", reply))


def assert_batch_link_hosts(reply: str, platform: str) -> None:
    """Reject any KeyTao public link rendered for the wrong chat platform."""
    expected_base = public_base_for_platform(platform)
    expected_host = urlsplit(expected_base).netloc.lower()
    public_hosts = {
        "keytao.rea.ink",
        "www.keytao.rea.ink",
        "keytao.vercel.app",
        "www.keytao.vercel.app",
    }
    mismatched = sorted({
        url
        for url in re.findall(r"https?://[^\s)\]]+", reply)
        if (
            "/batch/" in urlsplit(url).path
            or (urlsplit(url).hostname or "").lower() in public_hosts
        )
        and urlsplit(url).netloc.lower() != expected_host
    })
    require(
        bool(expected_host) and not mismatched,
        f"reply used the wrong public host for {platform}: "
        f"configured={expected_base!r}, mismatched={mismatched}",
    )


def assert_only_materialized_batch_links(
    replies: list[str],
    draft: dict[str, Any],
) -> None:
    materialized_id = str(draft.get("batchId") or "").strip()
    linked_ids = set().union(*(batch_link_ids(reply) for reply in replies))
    require(
        not linked_ids or linked_ids == {materialized_id},
        "reply exposed a provisional or mismatched batch URL: "
        f"linked={sorted(linked_ids)}, materialized={materialized_id!r}",
    )


def assert_no_below_base_weight(draft: dict[str, Any]) -> None:
    invalid = []
    for item in draft.get("items", []):
        phrase_type = str(item.get("type") or "Phrase")
        weight = item.get("weight")
        base = PHRASE_TYPE_BASE_WEIGHTS.get(phrase_type)
        if (
            base is not None
            and isinstance(weight, int)
            and not isinstance(weight, bool)
            and weight < base
        ):
            invalid.append({"item": item, "base": base})
    require(not invalid, f"draft contains weights below the type base: {invalid}")


def assert_no_same_turn_resend(reply: str, message: str) -> None:
    resend_markers = (
        "重新发送",
        "再次发送",
        "再发送",
        "重发",
        "原样发送",
        "原样转述",
        "下一条消息",
        "重复这条",
    )
    compact_reply = re.sub(r"[\s`'\"「」“”‘’]+", "", reply)
    compact_message = re.sub(r"[\s`'\"「」“”‘’]+", "", message)
    require(
        not any(marker in reply for marker in resend_markers),
        f"reply asked for a same-turn resend: {reply}",
    )
    require(
        compact_message not in compact_reply,
        f"reply repeated the failed instruction as remediation: {reply}",
    )


@dataclass
class ScenarioContext:
    scenario_id: str
    attempt: int
    identity: dict[str, str]
    next_client: LocalNextClient
    bot: E2EBotHarness
    recorder: ArtifactRecorder
    encode_delay: EncodeDelayController
    pronunciation_poison: PronunciationPoisonController
    fixture_facts: dict[str, Any]
    admin_identity: dict[str, str]
    admin_user: dict[str, Any]
    admin_token: str

    @property
    def platform_id(self) -> str:
        return self.identity["platform_id"]

    @property
    def sender_name(self) -> str:
        return self.identity["name"]

    @property
    def last_reply_message_id(self) -> int | None:
        return self.bot.last_reply_message_id

    def inject_bot_message(self, text: str) -> int:
        return self.bot.inject_bot_message(
            platform_id=self.platform_id,
            text=text,
        )

    async def send(self, text: str) -> str:
        reply = await self.bot.send(
            platform_id=self.platform_id,
            sender_name=self.sender_name,
            text=text,
        )
        assert_batch_link_hosts(reply, "qq")
        return reply

    async def send_group(self, text: str, *, to_me: bool) -> str:
        reply = await self.bot.send_group(
            platform_id=self.platform_id,
            sender_name=self.sender_name,
            text=text,
            to_me=to_me,
        )
        assert_batch_link_hosts(reply, "qq")
        return reply

    async def send_group_reply(
        self,
        text: str,
        *,
        reply_message_id: int,
        to_me: bool,
    ) -> str:
        reply = await self.bot.send_group_reply(
            platform_id=self.platform_id,
            sender_name=self.sender_name,
            text=text,
            reply_message_id=reply_message_id,
            to_me=to_me,
        )
        assert_batch_link_hosts(reply, "qq")
        return reply

    async def draft(self) -> dict[str, Any]:
        return await self.next_client.get_draft(self.platform_id)

    def attempt_events(self) -> list[dict[str, Any]]:
        return self.recorder.events_for(self.scenario_id, self.attempt)


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    name: str
    execute: Callable[[ScenarioContext], Awaitable[dict[str, Any]]]


async def scenario_s1(ctx: ScenarioContext) -> dict[str, Any]:
    reply = await ctx.send("把吃席放在赤溪前面")
    draft = await ctx.draft()
    next_code = ctx.fixture_facts["chixi_next_code"]
    expected = {
        ("Delete", "赤溪", "wkxk"),
        ("Create", "吃席", "wkxk"),
        ("Create", "赤溪", next_code),
    }
    actual = {item_key(item) for item in draft["items"]}
    require(len(draft["items"]) == 3 and actual == expected, f"unexpected S1 draft: {draft}")
    assert_no_code_request(reply)
    require("安全拦截" not in reply, f"S1 was policy-blocked: {reply}")
    assert_reply_mentions(reply, "吃席", "赤溪", "wkxk", next_code)
    return {
        "messages": ["把吃席放在赤溪前面"],
        "replies": [reply],
        "draft": draft,
        "facts": {"expectedItems": sorted(expected), "actualItems": sorted(actual)},
    }


async def scenario_s2(ctx: ScenarioContext) -> dict[str, Any]:
    message = "添加「吃席」 wkxk，同码即可"
    reply = await ctx.send(message)
    draft = await ctx.draft()
    require(len(draft["items"]) == 1, f"S2 did not create exactly one draft item: {draft}")
    item = draft["items"][0]
    require(item_key(item) == ("Create", "吃席", "wkxk"), f"unexpected S2 item: {item}")
    weight = item.get("weight")
    # The message asks for a duplicate but states no ordering relation, so the
    # newcomer appends behind the occupant at base + 1. Only an explicit
    # 前面/之前 request (see S12) may take the base slot and bump the occupant.
    require(
        isinstance(weight, int) and not isinstance(weight, bool) and weight == 101,
        f"S2 duplicate weight is not the appended slot: {weight}",
    )
    occupants = await ctx.next_client.phrases_by_code("wkxk")
    require(
        any(
            phrase.get("word") == "赤溪"
            and phrase.get("code") == "wkxk"
            and phrase.get("weight") == 100
            for phrase in occupants
        ),
        f"the existing 赤溪 entry no longer coexists: {occupants}",
    )
    assert_no_code_request(reply)
    assert_reply_mentions(reply, "吃席", "wkxk")
    require(
        "同码" in reply or "排在" in reply or "重码" in reply,
        f"S2 reply omitted the duplicate ordering fact: {reply}",
    )
    return {
        "messages": [message],
        "replies": [reply],
        "draft": draft,
        "facts": {"createWeight": weight, "dictionaryOccupants": occupants},
    }


async def scenario_s3(ctx: ScenarioContext) -> dict[str, Any]:
    reply = await ctx.send("把吃席放在赤溪后面")
    draft = await ctx.draft()
    expected_code = ctx.fixture_facts["chixi_subject_next_free_code"]
    require(len(draft["items"]) == 1, f"S3 moved existing entries: {draft}")
    require(
        item_key(draft["items"][0]) == ("Create", "吃席", expected_code),
        f"S3 did not use the next served free candidate {expected_code}: {draft}",
    )
    assert_no_code_request(reply)
    assert_reply_mentions(reply, "吃席", expected_code)
    return {
        "messages": ["把吃席放在赤溪后面"],
        "replies": [reply],
        "draft": draft,
        "facts": {"expectedNextFreeCode": expected_code, "itemCount": 1},
    }


async def scenario_s4(ctx: ScenarioContext) -> dict[str, Any]:
    ctx.bot.seed_expired_confirmation_advertisement(
        platform_id=ctx.platform_id,
    )
    stale_reply = await ctx.send("确认")
    draft = await ctx.draft()
    require(not draft["items"], f"stale confirmation changed the draft: {draft}")
    require(
        "已经过期" in stale_reply or "已过期" in stale_reply,
        f"S4 omitted expiry marker: {stale_reply}",
    )
    require(
        not re.search(
            r"(?:重新|再次).{0,8}(?:发送|回复).{0,4}[「“]?确认",
            stale_reply,
        ),
        f"S4 advised resending confirmation: {stale_reply}",
    )

    fixture_word = "呵呵呵"
    fixture_code = "hhhooo"
    await ctx.next_client.add_draft_items(
        platform_id=ctx.platform_id,
        items=[{
            "action": "Create",
            "word": fixture_word,
            "code": fixture_code,
            "type": "Phrase",
            "weight": 100,
            "needsManualReview": True,
            "remark": "S4 single-delete confirmation fixture",
        }],
    )
    seeded_draft = await ctx.draft()
    require(
        len(seeded_draft["items"]) == 1
        and item_key(seeded_draft["items"][0])
        == ("Create", fixture_word, fixture_code),
        f"S4 could not seed its delete fixture: {seeded_draft}",
    )

    delete_message = f"删除草稿里的「{fixture_word}」"
    delete_prompt = await ctx.send(delete_message)
    require(
        delete_prompt.count(pending_confirmation_copy()) == 1,
        f"S4 delete did not expose exactly one server-bound prompt: {delete_prompt}",
    )
    assert_reply_mentions(delete_prompt, fixture_word, fixture_code, "删除草稿条目：")
    require("PR#" not in delete_prompt, f"S4 delete leaked an internal id: {delete_prompt}")
    require(
        len((await ctx.draft())["items"]) == 1,
        "S4 delete executed before its one required confirmation",
    )

    delete_completion = await ctx.send("确认")
    final_draft = await ctx.draft()
    require(not final_draft["items"], f"S4 confirmed delete did not execute: {final_draft}")
    require(
        pending_confirmation_copy() not in delete_completion,
        f"S4 delete asked for a second confirmation: {delete_completion}",
    )
    require("PR#" not in delete_completion, f"S4 completion leaked an internal id: {delete_completion}")
    return {
        "messages": ["确认", delete_message, "确认"],
        "replies": [stale_reply, delete_prompt, delete_completion],
        "draft": final_draft,
        "facts": {
            "staleDraftChanges": 0,
            "expiryMarker": True,
            "deleteConfirmationSteps": 1,
            "deleteFixture": [fixture_word, fixture_code],
            "internalIdLeak": False,
        },
    }


async def scenario_s5(ctx: ScenarioContext) -> dict[str, Any]:
    message = "把吃席放在赤溪前面，目标编码请你自己查清楚后直接完成，不要问我"
    reply = await ctx.send(message)
    draft = await ctx.draft()
    next_code = ctx.fixture_facts["chixi_next_code"]
    expected = {
        ("Delete", "赤溪", "wkxk"),
        ("Create", "吃席", "wkxk"),
        ("Create", "赤溪", next_code),
    }
    actual = {item_key(item) for item in draft["items"]}
    require(actual == expected and len(draft["items"]) == 3, f"S5 did not complete add: {draft}")
    events = ctx.attempt_events()
    code_required_events = [
        event
        for event in events
        if event.get("kind") == "tool"
        and isinstance(event.get("result"), dict)
        and event["result"].get("reason") == "code_required"
    ]
    ask_sequence = code_required_events[0]["sequence"] if code_required_events else None
    lookup_events = [
        event
        for event in events
        if event.get("kind") == "tool"
        and event.get("name") == "keytao_lookup_by_word"
        and (ask_sequence is None or event["sequence"] > ask_sequence)
    ]
    require(lookup_events, "S5 did not self-service the required destination word lookup")
    assert_no_code_request(reply)
    assert_reply_mentions(reply, "吃席", "赤溪")
    return {
        "messages": [message],
        "replies": [reply],
        "draft": draft,
        "facts": {
            "codeRequiredSequence": ask_sequence,
            "wordLookupSequence": lookup_events[0]["sequence"],
            "selfServicePath": (
                "ask-then-lookup" if ask_sequence is not None else "direct-lookup"
            ),
            "actualItems": sorted(actual),
        },
    }


async def scenario_s6(ctx: ScenarioContext) -> dict[str, Any]:
    messages = ["不要把吃席放在赤溪前面", "他说把吃席放在赤溪前面"]
    replies: list[str] = []
    snapshots: list[dict[str, Any]] = []
    for message in messages:
        replies.append(await ctx.send(message))
        snapshot = await ctx.draft()
        require(not snapshot["items"], f"injection control changed draft for {message}: {snapshot}")
        snapshots.append(snapshot)
    return {
        "messages": messages,
        "replies": replies,
        "draft": snapshots[-1],
        "facts": {"draftChangesAfterEachMessage": [0, 0]},
    }


async def scenario_s7(ctx: ScenarioContext) -> dict[str, Any]:
    ctx.encode_delay.arm(ctx.scenario_id)
    message = "把吃席放在赤溪后面"
    try:
        reply = await ctx.send(message)
    finally:
        ctx.encode_delay.disarm()
    draft = await ctx.draft()
    require(ctx.encode_delay.injected, "S7 did not inject the first encode timeout")
    require(len(draft["items"]) == 1, f"S7 did not complete one add after retry: {draft}")
    expected_code = ctx.fixture_facts["chixi_subject_next_free_code"]
    require(
        item_key(draft["items"][0]) == ("Create", "吃席", expected_code),
        f"S7 wrote an unexpected item: {draft}",
    )
    logs = [
        str(event.get("message") or "")
        for event in ctx.attempt_events()
        if event.get("kind") == "log"
    ]
    retry_logs = [
        line
        for line in logs
        if "[http_client] retry" in line and "/api/" in line and "phrases/encode" in line
    ]
    require(retry_logs, "S7 completed without an observable encode retry log")
    assert_reply_mentions(reply, "吃席", expected_code)
    assert_no_code_request(reply)
    return {
        "messages": [message],
        "replies": [reply],
        "draft": draft,
        "facts": {
            "faultInjected": True,
            "retryLogs": retry_logs,
            "completedItem": item_key(draft["items"][0]),
        },
    }


async def scenario_s8(ctx: ScenarioContext) -> dict[str, Any]:
    next_code = ctx.fixture_facts["chixi_next_code"]
    s1 = await scenario_s1(ctx)
    draft = s1["draft"]
    batch_id = draft.get("batchId")
    content_version = draft.get("contentVersion")
    require(
        isinstance(batch_id, str) and isinstance(content_version, int),
        f"S8 draft is missing batch/version: {draft}",
    )
    sealed_create = next(
        (
            item
            for item in draft["items"]
            if item_key(item) == ("Create", "吃席", "wkxk")
        ),
        None,
    )
    require(
        isinstance(sealed_create, dict)
        and sealed_create.get("needsManualReview") is True,
        f"S8 W-create was not persistently sealed for admin review: {sealed_create}",
    )

    submit = await ctx.next_client.submit_batch(
        platform_id=ctx.platform_id,
        batch_id=batch_id,
        content_version=content_version,
    )
    submitted_detail = await ctx.next_client.get_admin_batch(
        batch_id=batch_id,
        admin_token=ctx.admin_token,
    )
    require(
        submitted_detail.get("status") == "Submitted",
        f"S8 batch did not reach Submitted: {submitted_detail}",
    )
    submitted_rows = submitted_detail["pullRequests"]
    require(len(submitted_rows) == 3, f"S8 admin review saw wrong PR count: {submitted_rows}")
    submitted_sealed = next(
        (
            item
            for item in submitted_rows
            if item_key(item) == ("Create", "吃席", "wkxk")
        ),
        None,
    )
    require(
        isinstance(submitted_sealed, dict)
        and submitted_sealed.get("needsManualReview") is True,
        f"S8 admin path lost the sealed review flag: {submitted_sealed}",
    )

    approval = await ctx.next_client.approve_admin_batch(
        batch_id=batch_id,
        admin_token=ctx.admin_token,
        review_note="E2E S8 human-admin approval of sealed positional move",
    )
    approved_detail = await ctx.next_client.get_admin_batch(
        batch_id=batch_id,
        admin_token=ctx.admin_token,
    )
    require(
        approved_detail.get("status") == "Approved",
        f"S8 batch is not Approved after admin approval: {approved_detail}",
    )
    require(
        approved_detail.get("reviewerId") == ctx.admin_user.get("id"),
        "S8 approval was not attributed to the reserved local admin identity",
    )
    approved_rows = approved_detail["pullRequests"]
    require(
        len(approved_rows) == 3
        and all(item.get("status") == "Approved" for item in approved_rows),
        f"S8 PR rows are not all Approved: {approved_rows}",
    )
    approved_sealed = next(
        (
            item
            for item in approved_rows
            if item_key(item) == ("Create", "吃席", "wkxk")
        ),
        None,
    )
    require(
        isinstance(approved_sealed, dict)
        and approved_sealed.get("needsManualReview") is True
        and approved_sealed.get("status") == "Approved",
        f"S8 sealed W-create did not pass the real admin gate: {approved_sealed}",
    )

    wkxk = await ctx.next_client.phrases_by_code("wkxk")
    shifted = await ctx.next_client.phrases_by_code(next_code)
    chixi = await ctx.next_client.phrases_by_word("赤溪")
    exact_wkxk = [row for row in wkxk if row.get("code") == "wkxk"]
    require(
        len(exact_wkxk) == 1
        and exact_wkxk[0].get("word") == "吃席"
        and exact_wkxk[0].get("type") == "Phrase",
        f"S8 dictionary lacks 吃席@wkxk: {wkxk}",
    )
    require(
        len(shifted) == 1
        and shifted[0].get("word") == "赤溪"
        and shifted[0].get("type") == "Phrase",
        f"S8 dictionary lacks shifted 赤溪@{next_code}: {shifted}",
    )
    require(
        len(chixi) == 1
        and chixi[0].get("code") == next_code
        and not any(item.get("code") == "wkxk" for item in chixi),
        f"S8 left 赤溪@wkxk or an unexpected 赤溪 row: {chixi}",
    )

    return {
        "messages": s1["messages"],
        "replies": s1["replies"],
        "draft": draft,
        "facts": {
            "batchId": batch_id,
            "submitStatus": submit["submitted"]["batch"]["status"],
            "approvalStatus": approval["batch"]["status"],
            "targetChangedRegressionAbsent": True,
            "adminReviewerId": approved_detail.get("reviewerId"),
            "adminRoleGate": "R:MANAGER JWT",
            "sealedCreateNeedsManualReview": True,
            "sealedCreateApproved": True,
            "approvedPullRequestCount": len(approved_rows),
            "dictionaryBeforeCleanup": {
                "wkxk": wkxk,
                next_code: shifted,
                "chixi": chixi,
            },
        },
    }


async def scenario_s9(ctx: ScenarioContext) -> dict[str, Any]:
    fixture = ctx.fixture_facts["s9"]
    before = await ctx.draft()
    message = "喵喵 射覆"
    reply = await ctx.send(message)
    after = await ctx.draft()
    assert_reply_mentions(reply, "常用度评估", "射覆", "慑服", "eefj")

    free_code = str(fixture["recommendedFreeCode"])
    keep_shape = (
        "「慑服」不弱于「射覆」" in reply
        and "维持现有排序，推荐空位" in reply
        and free_code in reply
    )
    require(
        keep_shape,
        f"S9 offline reference did not keep the corpus-attested occupant first: {reply}",
    )
    require(
        before.get("batchId") == after.get("batchId")
        and before.get("contentVersion") == after.get("contentVersion")
        and before.get("items") == after.get("items") == [],
        f"S9 presentation wrote to the draft: before={before}, after={after}",
    )
    return {
        "messages": [message],
        "replies": [reply],
        "draft": after,
        "facts": {
            "fixtureOccupant": fixture["occupantWord"],
            "fixtureCode": fixture["occupiedCode"],
            "freeCode": free_code,
            "recommendationShape": "keep-order",
            "commonnessReferenceVerdict": "慑服保持在射覆前",
            "draftUnchanged": True,
        },
    }


async def scenario_s10(ctx: ScenarioContext) -> dict[str, Any]:
    require(
        ctx.fixture_facts.get("multiAdd", {}).get("bothFree") is True,
        "S10 multi-add fixture did not prove both exact codes free",
    )
    message = "喵喵\n加词 王中王 wfw\n加词 微服务 wfwu"
    messages = [message]
    replies = [await ctx.send_group(message, to_me=False)]
    draft = await ctx.draft()
    confirmation_turns = 0

    if not draft["items"]:
        require(
            not batch_link_ids(replies[0]),
            f"S10 preview exposed a provisional batch URL: {replies[0]}",
        )
        confirmation = "确认"
        messages.append(confirmation)
        replies.append(await ctx.send_group(confirmation, to_me=True))
        confirmation_turns = 1
        draft = await ctx.draft()

    expected = {
        ("Create", "王中王", "wfw"),
        ("Create", "微服务", "wfwu"),
    }
    actual = {item_key(item) for item in draft["items"]}
    require(
        len(draft["items"]) == 2 and actual == expected,
        f"S10 did not write both exact adds: {draft}",
    )
    require(confirmation_turns <= 1, "S10 required more than one confirmation")
    resend_markers = (
        "请把下面这条指令原样转述",
        "重新发送完整操作指令",
        "重新发送包含词条和编码",
    )
    require(
        not any(marker in reply for marker in resend_markers for reply in replies),
        f"S10 asked the user to resend authorization wording: {replies}",
    )
    assert_only_materialized_batch_links(replies, draft)
    return {
        "messages": messages,
        "replies": replies,
        "draft": draft,
        "facts": {
            "confirmationTurns": confirmation_turns,
            "expectedItems": sorted(expected),
            "actualItems": sorted(actual),
            "provisionalBatchLinks": [],
        },
    }


async def scenario_s11(ctx: ScenarioContext) -> dict[str, Any]:
    initial = "把吃席同码放在赤溪前面"
    first_reply = await ctx.send(initial)
    before_confirmation = await ctx.draft()
    expected = {
        ("Create", "吃席", "wkxk"),
        ("Change", "赤溪", "wkxk"),
    }
    if before_confirmation["items"]:
        actual = {item_key(item) for item in before_confirmation["items"]}
        require(
            expected.issubset(actual),
            f"S11 auto-confirmed an unexpected front insert: {before_confirmation}",
        )
        assert_only_materialized_batch_links([first_reply], before_confirmation)
        return {
            "messages": [initial],
            "replies": [first_reply],
            "draft": before_confirmation,
            "facts": {
                "autoConfirmed": True,
                "expectedItems": sorted(expected),
                "actualItems": sorted(actual),
                "provisionalBatchLinks": [],
            },
        }
    require(
        not before_confirmation["items"],
        f"S11 did not pause on the noninformational warning ticket: {before_confirmation}",
    )
    require(
        "确认" in first_reply,
        f"S11 did not expose a live confirmation path: {first_reply}",
    )
    require(
        not batch_link_ids(first_reply),
        f"S11 preview exposed a provisional batch URL: {first_reply}",
    )

    combined = "确认并提交"
    second_reply = await ctx.send_group(combined, to_me=True)
    draft = await ctx.draft()
    write_results = [
        event.get("result")
        for event in ctx.attempt_events()
        if event.get("kind") == "tool"
        and event.get("name") == "keytao_batch_add_to_draft"
        and isinstance(event.get("result"), dict)
        and event["result"].get("success") is True
    ]
    require(write_results, "S11 never completed the pending batch write")
    write_result = write_results[-1]
    written_snapshot = (
        write_result.get("draft_snapshot")
        if isinstance(write_result.get("draft_snapshot"), dict)
        else {}
    )
    write_proof = (
        draft
        if draft["items"]
        else {
            "batchId": write_result.get("batchId"),
            "items": written_snapshot.get("items", []),
            "submittedAfterWrite": True,
        }
    )
    actual = {item_key(item) for item in write_proof["items"]}
    require(
        expected.issubset(actual),
        f"S11 confirmation lost the live write ticket: {write_proof}",
    )
    require(
        "没有找到待提交的草稿批次" not in second_reply,
        f"S11 submitted before consuming the write ticket: {second_reply}",
    )
    assert_only_materialized_batch_links(
        [first_reply, second_reply],
        write_proof,
    )
    return {
        "messages": [initial, combined],
        "replies": [first_reply, second_reply],
        "draft": write_proof,
        "facts": {
            "autoConfirmed": False,
            "liveTicketReached": True,
            "expectedItems": sorted(expected),
            "actualItems": sorted(actual),
            "emptyDraftClaimAbsent": True,
        },
    }


async def scenario_s12(ctx: ScenarioContext) -> dict[str, Any]:
    message = "把吃席同码放在赤溪前面"
    reply = await ctx.send(message)
    draft = await ctx.draft()
    items = {
        (str(item.get("action")), str(item.get("word"))): item
        for item in draft["items"]
    }
    require(
        len(draft["items"]) == 2
        and items.get(("Create", "吃席"), {}).get("code") == "wkxk"
        and items[("Create", "吃席")].get("weight") == 100
        and items.get(("Change", "赤溪"), {}).get("code") == "wkxk"
        and items[("Change", "赤溪")].get("weight") == 101,
        f"S12 did not materialize the legal front-insert plan: {draft}",
    )
    assert_no_below_base_weight(draft)
    assert_only_materialized_batch_links([reply], draft)
    return {
        "messages": [message],
        "replies": [reply],
        "draft": draft,
        "facts": {
            "newcomerWeight": items[("Create", "吃席")]["weight"],
            "incumbentWeight": items[("Change", "赤溪")]["weight"],
            "allWeightsAtOrAboveBase": True,
        },
    }


async def scenario_s13(ctx: ScenarioContext) -> dict[str, Any]:
    message = "将草稿中「亮面」lxmmov 的权重调整为 101"
    reply = await ctx.send(message)
    draft = await ctx.draft()
    require(not draft["items"], f"S13 unexpectedly mutated an empty draft: {draft}")
    assert_no_same_turn_resend(reply, message)
    require(
        "无法" in reply or "不能" in reply or "没有" in reply or "未找到" in reply,
        f"S13 omitted a deterministic failure explanation: {reply}",
    )
    return {
        "messages": [message],
        "replies": [reply],
        "draft": draft,
        "facts": {
            "sameTurnResendAbsent": True,
            "draftUnchanged": True,
        },
    }


async def scenario_s14(ctx: ScenarioContext) -> dict[str, Any]:
    from unittest.mock import patch

    from keytao_bot.utils import keytao_review as review_module

    review_module._clear_review_caches()
    before = await ctx.draft()
    ctx.pronunciation_poison.arm(ctx.scenario_id)
    message = "喵喵 亮面"
    try:
        with (
            patch.object(
                review_module,
                "_search_web",
                side_effect=ctx.pronunciation_poison.search_web,
            ),
            patch.object(
                review_module,
                "_fetch_text",
                side_effect=ctx.pronunciation_poison.fetch_text,
            ),
        ):
            reply = await ctx.send(message)
    finally:
        ctx.pronunciation_poison.disarm()
    after = await ctx.draft()

    require(
        ctx.pronunciation_poison.injected,
        "S14 collector did not observe the injected wrong-entry search hit and page",
    )
    review_events = [
        event
        for event in ctx.attempt_events()
        if event.get("kind") == "tool"
        and event.get("name") == "keytao_prepare_reviewed_add"
        and isinstance(event.get("result"), dict)
        and event.get("arguments", {}).get("word") == "亮面"
    ]
    require(review_events, "S14 did not review 亮面 through the public add-review tool")
    review = review_events[-1]["result"]
    pronunciations = [
        item
        for item in review.get("pronunciations", [])
        if isinstance(item, dict)
    ]
    syllables = {
        str(syllable or "").strip().lower()
        for item in pronunciations
        for syllable in item.get("normalized", [])
    }
    candidate_codes = {
        str(code or "").strip().lower()
        for item in pronunciations
        for code in item.get("codes", [])
    }
    wrong_codes = sorted(code for code in candidate_codes if code.startswith("gxmm"))
    correct_codes = sorted(code for code in candidate_codes if code.startswith("lxmm"))
    manual_review = bool(
        review.get("pronunciationUnresolved")
        or review.get("requiresManualPronunciationReview")
    )
    require("guang" not in syllables, f"S14 accepted the poisoned guang reading: {review}")
    require(not wrong_codes, f"S14 exposed poisoned gxmm candidates: {wrong_codes}")
    require(
        bool(correct_codes) or manual_review,
        f"S14 neither offered lxmm candidates nor failed closed: {review}",
    )
    require(
        before.get("batchId") == after.get("batchId")
        and before.get("contentVersion") == after.get("contentVersion")
        and before.get("items") == after.get("items") == [],
        f"S14 pronunciation review mutated the draft: before={before}, after={after}",
    )
    return {
        "messages": [message],
        "replies": [reply],
        "draft": after,
        "facts": {
            "faultInjected": True,
            "poisonedEntry": "光面",
            "poisonedPinyin": "guang mian",
            "acceptedSyllables": sorted(syllables),
            "candidateCodes": sorted(candidate_codes),
            "manualReview": manual_review,
            "draftUnchanged": True,
        },
    }


def _successful_submit_batch_id(
    events: list[dict[str, Any]],
    *,
    after_sequence: int,
) -> str:
    for event in reversed(events):
        if (
            int(event.get("sequence") or 0) <= after_sequence
            or event.get("kind") != "tool"
            or event.get("name") != "keytao_submit_batch"
        ):
            continue
        result = event.get("result")
        if not isinstance(result, dict) or result.get("success") is not True:
            continue
        batch_id = str(result.get("batchId") or "").strip()
        if batch_id:
            return batch_id
    return ""


def _submitted_item(
    batch: dict[str, Any],
    *,
    word: str,
    code: str,
) -> dict[str, Any] | None:
    return next(
        (
            item
            for item in batch.get("pullRequests", [])
            if isinstance(item, dict)
            and item_key(item) == ("Create", word, code)
        ),
        None,
    )


async def scenario_s15(ctx: ScenarioContext) -> dict[str, Any]:
    fixture = ctx.fixture_facts["s15"]
    candidate_codes = list(fixture["candidateCodes"])
    require(
        len(candidate_codes) >= 2,
        f"S15 fixture lacks candidate 2: {fixture}",
    )
    selected_code = str(candidate_codes[1])
    messages = ["喵喵 射覆"]
    replies = [await ctx.send(messages[-1])]
    require(
        re.search(rf"(?m)^2\.\s*{re.escape(selected_code)}\b", replies[-1])
        is not None,
        f"S15 discovery did not publish candidate 2={selected_code}: {replies[-1]}",
    )

    first_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append("2 添加并提交")
    replies.append(await ctx.send(messages[-1]))
    require(
        "没有匹配到" not in replies[-1]
        and "执行动词" not in replies[-1]
        and "本轮没有可执行的已绑定写操作" not in replies[-1],
        f"S15 numbered reply hit the old authorization rejection: {replies[-1]}",
    )
    first_batch_id = _successful_submit_batch_id(
        ctx.attempt_events(),
        after_sequence=first_cutoff,
    )
    require(first_batch_id, "S15 numbered reply never completed submit")
    first_batch = await ctx.next_client.get_admin_batch(
        batch_id=first_batch_id,
        admin_token=ctx.admin_token,
    )
    require(
        first_batch.get("status") in {"Submitted", "Approved"},
        f"S15 numbered batch did not pass through submission: {first_batch}",
    )
    require(
        _submitted_item(first_batch, word="射覆", code=selected_code) is not None,
        f"S15 numbered batch lacks 射覆@{selected_code}: {first_batch}",
    )

    messages.append("喵喵 亮面")
    discovery_reply = await ctx.send(messages[-1])
    replies.append(discovery_reply)
    second_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    completion_reply_start = len(replies)
    messages.append("添加并提交")
    guidance = await ctx.send(messages[-1])
    replies.append(guidance)
    second_batch_id = _successful_submit_batch_id(
        ctx.attempt_events(),
        after_sequence=second_cutoff,
    )
    additional_confirmation_steps = 0
    if not second_batch_id and pending_confirmation_copy() in guidance:
        messages.append("确认")
        replies.append(await ctx.send(messages[-1]))
        additional_confirmation_steps = 1
        second_batch_id = _successful_submit_batch_id(
            ctx.attempt_events(),
            after_sequence=second_cutoff,
        )
        require(
            second_batch_id,
            "S15 one legitimate confirmation did not complete submit",
        )

    if second_batch_id:
        expected_code_match = re.search(
            r"(?:推荐编码[：:]|是否以编码)\s*"
            r"(?P<code>[a-z]{2,12})"
            r"(?:（本次仅查询）|\s+将「亮面」加入草稿)",
            discovery_reply,
        )
        require(
            expected_code_match is not None,
            f"S15 direct completion lacked a bound expected code: {discovery_reply}",
        )
        expected_code = expected_code_match.group("code")
        for reply in replies[completion_reply_start:]:
            require(
                "没有匹配到" not in reply
                and "执行动词" not in reply
                and "本轮没有可执行的已绑定写操作" not in reply,
                f"S15 direct completion hit another correction: {reply}",
            )
        second_batch = await ctx.next_client.get_admin_batch(
            batch_id=second_batch_id,
            admin_token=ctx.admin_token,
        )
        require(
            second_batch.get("status") in {"Submitted", "Approved"},
            f"S15 direct-completion batch did not pass through submission: {second_batch}",
        )
        direct_item = _submitted_item(
            second_batch,
            word="亮面",
            code=expected_code,
        )
        require(
            direct_item is not None,
            f"S15 direct-completion batch lacks 亮面@{expected_code}: {second_batch}",
        )
        require(
            direct_item.get("needsManualReview") is True,
            f"S15 direct completion lost the 亮面 manual-review seal: {direct_item}",
        )
        suggestion_facts = {
            "suggestionSubcase": "direct-completion",
            "directCompletionCode": expected_code,
            "directCompletionBatchId": second_batch_id,
            "directCompletionBatchStatus": second_batch.get("status"),
            "directCompletionSealed": True,
            "additionalConfirmationSteps": additional_confirmation_steps,
        }
    else:
        suggestion_match = re.search(
            r"(?m)^(?P<suggestion>-\s*[「“『]添加\s+亮面\s+"
            r"(?P<code>[a-z]{2,12})\s+并提交[」”』](?:（亮面）)?)$",
            guidance,
        )
        require(
            suggestion_match is not None,
            f"S15 did not render a copy-paste add-and-submit suggestion: {guidance}",
        )
        quoted_suggestion = suggestion_match.group("suggestion")
        suggested_code = suggestion_match.group("code")
        quoted_cutoff = max(
            (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
            default=0,
        )
        messages.append(quoted_suggestion)
        replies.append(await ctx.send(messages[-1]))
        require(
            "没有匹配到" not in replies[-1]
            and "执行动词" not in replies[-1]
            and "本轮没有可执行的已绑定写操作" not in replies[-1],
            f"S15 literal quoted suggestion hit another correction: {replies[-1]}",
        )
        second_batch_id = _successful_submit_batch_id(
            ctx.attempt_events(),
            after_sequence=quoted_cutoff,
        )
        require(second_batch_id, "S15 quoted suggestion never completed submit")
        second_batch = await ctx.next_client.get_admin_batch(
            batch_id=second_batch_id,
            admin_token=ctx.admin_token,
        )
        require(
            second_batch.get("status") in {"Submitted", "Approved"},
            f"S15 quoted-suggestion batch did not pass through submission: {second_batch}",
        )
        require(
            _submitted_item(
                second_batch,
                word="亮面",
                code=suggested_code,
            )
            is not None,
            f"S15 quoted-suggestion batch lacks 亮面@{suggested_code}: {second_batch}",
        )
        suggestion_facts = {
            "suggestionSubcase": "quoted-suggestion",
            "quotedSuggestion": quoted_suggestion,
            "quotedSuggestionCode": suggested_code,
            "quotedSuggestionBatchId": second_batch_id,
            "quotedSuggestionBatchStatus": second_batch.get("status"),
            "additionalConfirmationSteps": 0,
        }
    return {
        "messages": messages,
        "replies": replies,
        "draft": await ctx.draft(),
        "facts": {
            "numberedCandidateIndex": 2,
            "numberedCandidateCode": selected_code,
            "numberedBatchId": first_batch_id,
            "numberedBatchStatus": first_batch.get("status"),
            "additionalCorrectionRequired": False,
            **suggestion_facts,
        },
    }


async def scenario_s16(ctx: ScenarioContext) -> dict[str, Any]:
    fixture = ctx.fixture_facts["s16"]
    occupant_word = str(fixture["occupantWord"])
    occupied_code = str(fixture["occupiedCode"])
    shifted_code = str(fixture["shiftedCode"])
    messages = ["喵喵 加词 载流 载流子"]
    replies = [await ctx.send(messages[-1])]
    discovery_reply = replies[-1]
    expected_bindings = {("载流", "zhlq"), ("载流子", occupied_code)}
    review_events = [
        event
        for event in ctx.attempt_events()
        if event.get("kind") == "tool"
        and event.get("name") == "keytao_prepare_reviewed_add"
        and isinstance(event.get("result"), dict)
    ]
    structured_bindings = {
        (
            str(event["result"].get("word") or "").strip(),
            str(event["result"].get("recommendedCode") or "").strip().lower(),
        )
        for event in review_events
    }
    require(
        expected_bindings <= structured_bindings,
        f"S16 structured review did not bind both exact items: {review_events}",
    )
    displayed_bindings = set(advertised_batch_binding_pairs(discovery_reply))
    require(
        displayed_bindings == expected_bindings,
        f"S16 discovery rendering did not advertise both exact bindings: {discovery_reply}",
    )
    require(
        f"「载流子」占 {occupied_code}、「{occupant_word}」顺延" in discovery_reply
        and "推荐：\n- " in discovery_reply
        and "不重排选 2（zlzu）。" in discovery_reply,
        f"S16 discovery did not render the comparator default coherently: {discovery_reply}",
    )
    saved_ticket_events = [
        event
        for event in ctx.attempt_events()
        if event.get("kind") == "log"
        and "Saved advertised reviewed batch candidate" in str(
            event.get("message") or ""
        )
        and "items=2" in str(event.get("message") or "")
    ]
    require(saved_ticket_events, "S16 discovery did not persist one trusted two-item ticket")
    require(
        "加入并提交" in discovery_reply,
        f"S16 discovery did not advertise bare add-and-submit: {discovery_reply}",
    )

    cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append("加入并提交")
    replies.append(await ctx.send(messages[-1]))
    rejected_markers = (
        "没有引用机器人给出的候选消息",
        "需要把词条和编码写完整",
        "执行动词",
        "executionVerb",
        "提交草稿",
        "本轮没有可执行的已绑定写操作",
        "重新发送完整操作指令",
    )
    batch_id = _successful_submit_batch_id(
        ctx.attempt_events(),
        after_sequence=cutoff,
    )
    if not batch_id:
        require(
            pending_confirmation_copy() in replies[-1],
            f"S16 bare assent neither submitted nor reached one bound confirmation: {replies[-1]}",
        )
        messages.append("确认")
        replies.append(await ctx.send(messages[-1]))
        batch_id = _successful_submit_batch_id(
            ctx.attempt_events(),
            after_sequence=cutoff,
        )
    for reply in replies[1:]:
        require(
            not any(marker in reply for marker in rejected_markers),
            f"S16 advertised assent hit old remediation: {reply}",
        )
    require(batch_id, "S16 bare assent never completed submit")
    require(len(messages) - 2 <= 1, "S16 used more than one server-bound confirmation")
    batch = await ctx.next_client.get_admin_batch(
        batch_id=batch_id,
        admin_token=ctx.admin_token,
    )
    require(
        batch.get("status") in {"Submitted", "Approved"},
        f"S16 batch did not pass through submission: {batch}",
    )
    expected_items = {
        ("Create", "载流", "zhlq"),
        ("Delete", occupant_word, occupied_code),
        ("Create", occupant_word, shifted_code),
        ("Create", "载流子", occupied_code),
    }
    actual_items = {
        item_key(item)
        for item in batch.get("pullRequests", [])
        if isinstance(item, dict)
    }
    require(
        actual_items == expected_items,
        f"S16 submitted batch did not atomically combine the free add and front insert: {batch}",
    )
    return {
        "messages": messages,
        "replies": replies,
        "draft": await ctx.draft(),
        "facts": {
            "submittedWords": ["载流", "载流子"],
            "submittedCodes": ["zhlq", occupied_code],
            "shiftedOccupant": occupant_word,
            "shiftedToCode": shifted_code,
            "batchId": batch_id,
            "batchStatus": batch.get("status"),
            "quoteRequired": False,
            "additionalConfirmationSteps": len(messages) - 2,
        },
    }


def _recommended_empty_code(reply: str, *, word: str) -> str:
    match = re.search(
        rf"(?:是否以|仍以)编码\s+(?P<code>[a-z]{{2,12}})\s+将「{re.escape(word)}」加入草稿",
        reply,
    )
    if match is None:
        match = re.search(
            rf"(?m)^[•*-]\s*「{re.escape(word)}」\s*→\s*(?P<code>[a-z]{{2,12}})\s*（推荐）\s*$",
            reply,
        )
    if match is None:
        match = re.search(
            r"(?m)^不重排选\s+\d+\s*[（(]"
            r"(?P<code>[a-z]{2,12})[）)]。?\s*$",
            reply,
        )
    require(match is not None, f"{word} discovery omitted a recommended empty code: {reply}")
    code = match.group("code")
    require(
        re.search(
            rf"(?m)^\d+\.\s*{re.escape(code)}\s+—\s+.*空位.*$",
            reply,
        )
        is not None,
        f"{word} recommended code was not rendered as an empty slot: {reply}",
    )
    return code


async def scenario_s17(ctx: ScenarioContext) -> dict[str, Any]:
    messages = ["喵喵 加词 产季"]
    replies = [await ctx.send(messages[-1])]
    semantic_reply = replies[-1]
    assert_reply_mentions(
        semantic_reply,
        "可自动通过",
        "语境读音与含义明确",
        "语料/词典证据",
        "逐字 jieba 词频 产 6838、季 1619",
        "高频字阈值 1000",
    )
    semantic_code = _recommended_empty_code(semantic_reply, word="产季")
    semantic_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append("添加并提交")
    replies.append(await ctx.send(messages[-1]))
    semantic_batch_id = _successful_submit_batch_id(
        ctx.attempt_events(),
        after_sequence=semantic_cutoff,
    )
    if not semantic_batch_id and pending_confirmation_copy() in replies[-1]:
        messages.append("确认")
        replies.append(await ctx.send(messages[-1]))
        semantic_batch_id = _successful_submit_batch_id(
            ctx.attempt_events(),
            after_sequence=semantic_cutoff,
        )
    require(semantic_batch_id, "S17 产季 add-and-submit never completed")
    semantic_batch = await ctx.next_client.get_admin_batch(
        batch_id=semantic_batch_id,
        admin_token=ctx.admin_token,
    )
    semantic_item = _submitted_item(
        semantic_batch,
        word="产季",
        code=semantic_code,
    )
    require(
        semantic_batch.get("status") == "Approved",
        f"S17 产季 batch was not auto-approved: {semantic_batch}",
    )
    require(
        isinstance(semantic_item, dict)
        and semantic_item.get("needsManualReview") is False,
        f"S17 产季 persisted with a manual-review seal: {semantic_item}",
    )

    messages.append("喵喵 加词 龘季")
    replies.append(await ctx.send(messages[-1]))
    obscure_reply = replies[-1]
    assert_reply_mentions(obscure_reply, "龘季", "需要管理员审核")
    obscure_code = _recommended_empty_code(obscure_reply, word="龘季")
    obscure_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append("添加并提交")
    replies.append(await ctx.send(messages[-1]))
    obscure_batch_id = _successful_submit_batch_id(
        ctx.attempt_events(),
        after_sequence=obscure_cutoff,
    )
    if not obscure_batch_id and pending_confirmation_copy() in replies[-1]:
        messages.append("确认")
        replies.append(await ctx.send(messages[-1]))
        obscure_batch_id = _successful_submit_batch_id(
            ctx.attempt_events(),
            after_sequence=obscure_cutoff,
        )
    require(obscure_batch_id, "S17 龘季 control add-and-submit never completed")
    obscure_batch = await ctx.next_client.get_admin_batch(
        batch_id=obscure_batch_id,
        admin_token=ctx.admin_token,
    )
    obscure_item = _submitted_item(
        obscure_batch,
        word="龘季",
        code=obscure_code,
    )
    require(
        obscure_batch.get("status") == "Submitted",
        f"S17 obscure-character control did not remain submitted: {obscure_batch}",
    )
    require(
        isinstance(obscure_item, dict)
        and obscure_item.get("needsManualReview") is True,
        f"S17 obscure-character control lost its seal: {obscure_item}",
    )
    return {
        "messages": messages,
        "replies": replies,
        "draft": await ctx.draft(),
        "facts": {
            "semanticWord": "产季",
            "semanticCode": semantic_code,
            "semanticBatchId": semantic_batch_id,
            "semanticBatchStatus": semantic_batch.get("status"),
            "semanticNeedsManualReview": semantic_item.get("needsManualReview"),
            "nonObscurityRoute": "common_characters_and_llm",
            "obscureWord": "龘季",
            "obscureCode": obscure_code,
            "obscureBatchId": obscure_batch_id,
            "obscureBatchStatus": obscure_batch.get("status"),
            "obscureNeedsManualReview": obscure_item.get("needsManualReview"),
        },
    }


def _rendered_candidate_index(reply: str, code: str) -> int:
    match = re.search(
        rf"(?m)^(?P<index>\d+)\.\s*{re.escape(code)}\b.*$",
        reply,
    )
    require(match is not None, f"candidate list omitted {code}: {reply}")
    return int(match.group("index"))


def _rendered_candidate_reading_rows(reply: str) -> list[tuple[str, str]]:
    return [
        (match.group("pinyin").strip(), match.group("sources").strip())
        for match in re.finditer(
            r"(?m)^\d+\.\s*(?P<pinyin>[^；\n]+)；来源\s+(?P<sources>.+)$",
            reply,
        )
    ]


async def scenario_s18(ctx: ScenarioContext) -> dict[str, Any]:
    messages = ["喵喵 还车"]
    replies = [await ctx.send(messages[-1])]
    discovery = replies[-1]
    candidate_reading_rows = _rendered_candidate_reading_rows(discovery)
    candidate_readings = [pinyin for pinyin, _sources in candidate_reading_rows]
    reading_sections = re.findall(
        r"(?m)^候选编码（读音 (?P<index>\d+)）:$",
        discovery,
    )
    require(
        candidate_readings == ["huan che", "hái chē"]
        and "汉典（经编码服务）" in candidate_reading_rows[0][1]
        and any(
            label in candidate_reading_rows[1][1]
            for label in (
                "开放拼音数据（large_pinyin）",
                "汉典（离线数据集）",
            )
        )
        and reading_sections == ["1", "2"],
        f"S18 discovery did not render both incident readings: {discovery}",
    )
    empty_code = "htjev"
    occupied_code = "htwe"
    empty_index = _rendered_candidate_index(discovery, empty_code)
    occupied_index = _rendered_candidate_index(discovery, occupied_code)
    require(
        re.search(
            rf"(?m)^{empty_index}\.\s*{empty_code}\b.*空位.*$",
            discovery,
        )
        is not None,
        f"S18 {empty_code} was not rendered as an empty slot: {discovery}",
    )
    require(
        re.search(
            rf"(?m)^{occupied_index}\.\s*{occupied_code}\b.*已有.*换车.*$",
            discovery,
        )
        is not None,
        f"S18 {occupied_code} was not occupied by 换车: {discovery}",
    )
    require(
        "可多选，如「添加2、4」" in discovery,
        f"S18 discovery omitted the advertised multi-select form: {discovery}",
    )
    if "可自动通过" in discovery:
        empty_needs_manual_review = False
    else:
        require(
            "需要管理员审核" in discovery,
            f"S18 discovery omitted its review verdict: {discovery}",
        )
        empty_needs_manual_review = True

    control = f"添加{empty_index}、99"
    messages.append(control)
    replies.append(await ctx.send(control))
    control_draft = await ctx.draft()
    require(
        not control_draft.get("items"),
        f"S18 out-of-range control wrote a partial batch: {control_draft}",
    )
    internal_markers = (
        "boundTarget",
        "blockReason",
        "binding_incomplete",
        "exactAuthorizedItemSet",
    )
    require(
        not any(marker in replies[-1] for marker in internal_markers),
        f"S18 control leaked an internal policy field: {replies[-1]}",
    )

    selected = f"添加{empty_index}、{occupied_index}"
    selected_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append(selected)
    replies.append(await ctx.send(selected))
    draft = await ctx.draft()
    confirmation_steps = 0
    if not draft.get("items"):
        require(
            "确认" in replies[-1],
            f"S18 selection neither wrote nor offered one bound confirmation: {replies[-1]}",
        )
        messages.append("确认")
        replies.append(await ctx.send(messages[-1]))
        confirmation_steps = 1
        draft = await ctx.draft()

    expected = {
        ("Create", "还车", empty_code),
        ("Create", "还车", occupied_code),
    }
    actual = {item_key(item) for item in draft.get("items", [])}
    require(
        len(draft.get("items", [])) == 2 and actual == expected,
        f"S18 did not write the exact selected pair in one batch: {draft}",
    )
    require(
        bool(str(draft.get("batchId") or "")),
        f"S18 exact pair lacks one materialized batch: {draft}",
    )
    empty_item = next(
        item for item in draft["items"] if item_key(item) == ("Create", "还车", empty_code)
    )
    occupied_item = next(
        item for item in draft["items"] if item_key(item) == ("Create", "还车", occupied_code)
    )
    require(
        empty_item.get("needsManualReview") is empty_needs_manual_review,
        f"S18 empty slot did not preserve its own review verdict: {empty_item}",
    )
    require(
        occupied_item.get("needsManualReview") is True,
        f"S18 occupied slot lost the duplicate-code review seal: {occupied_item}",
    )
    duplicate_warnings = [
        warning
        for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > selected_cutoff
        and event.get("kind") == "tool"
        and event.get("name") == "keytao_batch_add_to_draft"
        and isinstance(event.get("result"), dict)
        for warning in (event["result"].get("warnings") or [])
        if isinstance(warning, dict)
        and warning.get("warningType") == "duplicate_code"
        and isinstance(warning.get("item"), dict)
        and item_key(warning["item"]) == ("Create", "还车", occupied_code)
    ]
    require(
        duplicate_warnings,
        "S18 occupied selection lacked an exact duplicate-code warning ticket",
    )

    submit_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append("提交")
    replies.append(await ctx.send(messages[-1]))
    batch_id = _successful_submit_batch_id(
        ctx.attempt_events(),
        after_sequence=submit_cutoff,
    )
    if not batch_id and "确认" in replies[-1]:
        require(
            confirmation_steps == 0,
            f"S18 required more than one server-bound confirmation: {replies[-1]}",
        )
        messages.append("确认")
        replies.append(await ctx.send(messages[-1]))
        confirmation_steps += 1
        batch_id = _successful_submit_batch_id(
            ctx.attempt_events(),
            after_sequence=submit_cutoff,
        )
    require(batch_id, "S18 exact multi-select batch did not submit")
    require(confirmation_steps <= 1, "S18 used more than one confirmation step")
    batch = await ctx.next_client.get_admin_batch(
        batch_id=batch_id,
        admin_token=ctx.admin_token,
    )
    submitted = {
        item_key(item)
        for item in batch.get("pullRequests", [])
        if isinstance(item, dict)
    }
    require(
        batch.get("status") in {"Submitted", "Approved"}
        and submitted == expected
        and len(batch.get("pullRequests", [])) == 2,
        f"S18 submitted batch does not equal the parsed selection set: {batch}",
    )
    return {
        "messages": messages,
        "replies": replies,
        "draft": draft,
        "facts": {
            "selectedIndexes": [empty_index, occupied_index],
            "selectedCodes": [empty_code, occupied_code],
            "candidateReadings": candidate_readings,
            "controlIndex": 99,
            "exactItemSet": sorted(expected),
            "batchId": batch_id,
            "batchStatus": batch.get("status"),
            "duplicateWarningSealed": True,
            "emptyNeedsManualReview": empty_needs_manual_review,
            "additionalConfirmationSteps": confirmation_steps,
        },
    }


S19_ADVERTISED_WORDS = (
    "显眼包",
    "嘴替",
    "松弛感",
    "电子榨菜",
    "情绪价值",
    "班味",
    "泼天富贵",
    "精神状态",
    "职场搭子",
    "天选打工人",
    "沙县小吃",
)


async def scenario_s19(ctx: ScenarioContext) -> dict[str, Any]:
    scan_message = (
        "喵喵，请批量检查这些常用词是否已收录；只列出未收录词，"
        "并说明可以把列表中的词加入草稿："
        + "、".join(S19_ADVERTISED_WORDS)
    )
    messages = [scan_message]
    replies = [await ctx.send(scan_message)]
    discovery = replies[-1]
    require(
        all(word in discovery for word in S19_ADVERTISED_WORDS),
        f"S19 scan did not render the complete absent-word set: {discovery}",
    )
    require(
        "草稿" in discovery and any(marker in discovery for marker in ("加入", "添加", "加到")),
        f"S19 scan did not advertise the rendered list as addable: {discovery}",
    )

    control = "火星词先不要，其他都加"
    messages.append(control)
    replies.append(await ctx.send(control))
    require(
        "火星词" in replies[-1]
        and any(marker in replies[-1] for marker in ("不在", "候选", "选择")),
        f"S19 out-of-snapshot exclusion did not deterministically ASK: {replies[-1]}",
    )
    control_draft = await ctx.draft()
    require(
        not control_draft.get("items"),
        f"S19 out-of-snapshot exclusion wrote draft items: {control_draft}",
    )

    selection = "天选打工人先不要，其他可以加，沙县小吃也不要"
    expected_words = S19_ADVERTISED_WORDS[:-2]
    cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append(selection)
    replies.append(await ctx.send(selection))
    selection_events = [
        event
        for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > cutoff
    ]
    progress_lines = [
        str(event.get("text") or "")
        for event in selection_events
        if event.get("kind") == "message"
        and event.get("direction") == "reply"
        and "正在处理" in str(event.get("text") or "")
    ]
    require(
        any(
            "已完成 8/9" in line and "预计还剩 1 轮" in line
            for line in progress_lines
        ),
        f"S19 chunked review did not report 8/9 progress: {progress_lines}",
    )
    require(
        "确认" in replies[-1]
        and all(word in replies[-1] for word in expected_words)
        and "天选打工人" not in replies[-1]
        and "沙县小吃" not in replies[-1],
        f"S19 final confirmation did not show only the resolved set: {replies[-1]}",
    )
    require(
        not (await ctx.draft()).get("items"),
        "S19 inferred set wrote before its one required confirmation",
    )

    confirmation_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append("确认")
    replies.append(await ctx.send(messages[-1]))
    draft = await ctx.draft()
    actual_words = tuple(
        str(item.get("word") or "")
        for item in draft.get("items", [])
        if isinstance(item, dict)
    )
    require(
        same_unique_item_set(actual_words, expected_words)
        and all(
            item.get("action") == "Create"
            for item in draft.get("items", [])
            if isinstance(item, dict)
        ),
        f"S19 draft does not equal snapshot minus exclusions: {draft}",
    )
    require(
        bool(str(draft.get("batchId") or "")),
        f"S19 resolved items did not reach one materialized draft batch: {draft}",
    )
    write_calls = [
        event
        for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > confirmation_cutoff
        and event.get("kind") == "tool"
        and event.get("name") == "keytao_batch_add_to_draft"
    ]
    require(write_calls, "S19 confirmation did not invoke the batch draft tool")
    for event in write_calls:
        tool_items = event.get("arguments", {}).get("items", [])
        tool_words = tuple(
            str(item.get("word") or "")
            for item in tool_items
            if isinstance(item, dict)
        )
        require(
            isinstance(tool_items, list)
            and same_unique_item_set(tool_words, expected_words),
            f"S19 tool call was not bound to the complete resolved set: {event}",
        )
    all_reply_text = [
        str(event.get("text") or "")
        for event in ctx.attempt_events()
        if event.get("kind") == "message"
        and event.get("direction") == "reply"
    ] + replies
    require(
        not any("参数格式错误" in reply for reply in all_reply_text),
        f"S19 surfaced the obsolete argument-format diagnosis: {all_reply_text}",
    )
    return {
        "messages": messages,
        "replies": replies,
        "draft": draft,
        "facts": {
            "advertisedWords": list(S19_ADVERTISED_WORDS),
            "excludedWords": ["天选打工人", "沙县小吃"],
            "resolvedWords": list(expected_words),
            "progressLines": progress_lines,
            "batchId": draft.get("batchId"),
            "confirmationSteps": 1,
            "outOfSnapshotControl": "ASK-without-write",
        },
    }


S20_BATCH_WORDS = S19_ADVERTISED_WORDS[:3]


async def scenario_s20(ctx: ScenarioContext) -> dict[str, Any]:
    initial = "喵喵 加词 " + " ".join(S20_BATCH_WORDS)
    messages = [initial]
    replies = [await ctx.send_group(initial, to_me=True)]
    advertised_pairs = advertised_batch_binding_pairs(replies[-1])
    require(
        len(advertised_pairs) == len(S20_BATCH_WORDS)
        and tuple(word for word, _code in advertised_pairs) == S20_BATCH_WORDS,
        f"S20 did not advertise the exact three-word batch: {replies[-1]}",
    )
    require(
        len(set(advertised_pairs)) == len(advertised_pairs),
        f"S20 advertised duplicate word/code bindings: {advertised_pairs}",
    )
    saved_ticket_events = [
        event
        for event in ctx.attempt_events()
        if event.get("kind") == "log"
        and "Saved advertised reviewed batch candidate" in str(
            event.get("message") or ""
        )
        and f"items={len(S20_BATCH_WORDS)}" in str(event.get("message") or "")
    ]
    require(saved_ticket_events, "S20 discovery did not persist one trusted batch ticket")

    quote_message_id = ctx.last_reply_message_id
    require(
        isinstance(quote_message_id, int) and quote_message_id > 0,
        "S20 harness did not expose the bot advertisement message id",
    )
    cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append("都加")
    replies.append(await ctx.send_group_reply(
        "都加",
        reply_message_id=quote_message_id,
        to_me=True,
    ))

    additional_confirmation_steps = 0
    draft = await ctx.draft()
    if not draft.get("items"):
        require(
            pending_confirmation_copy() in replies[-1]
            or "确认" in replies[-1],
            f"S20 quoted assent neither wrote nor reached one bound confirmation: {replies[-1]}",
        )
        additional_confirmation_steps = 1
        messages.append("确认")
        replies.append(await ctx.send_group("确认", to_me=True))
        draft = await ctx.draft()

    actual_items = [
        item
        for item in draft.get("items", [])
        if isinstance(item, dict)
    ]
    actual_pairs = tuple(
        (
            str(item.get("word") or "").strip(),
            str(item.get("code") or "").strip().lower(),
        )
        for item in actual_items
    )
    require(
        same_unique_item_set(actual_pairs, advertised_pairs)
        and all(item.get("action") == "Create" for item in actual_items),
        f"S20 draft differs from the native-quoted advertised batch: {draft}",
    )
    require(
        bool(str(draft.get("batchId") or "")),
        f"S20 exact items did not reach one materialized draft batch: {draft}",
    )
    write_calls = [
        event
        for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > cutoff
        and event.get("kind") == "tool"
        and event.get("name") == "keytao_batch_add_to_draft"
    ]
    require(write_calls, "S20 quoted assent did not invoke the batch draft tool")
    for event in write_calls:
        tool_items = event.get("arguments", {}).get("items", [])
        tool_pairs = tuple(
            (
                str(item.get("word") or "").strip(),
                str(item.get("code") or "").strip().lower(),
            )
            for item in tool_items
            if isinstance(item, dict)
        )
        require(
            isinstance(tool_items, list)
            and same_unique_item_set(tool_pairs, advertised_pairs),
            f"S20 tool call was not bound to the complete displayed set: {event}",
        )

    all_reply_text = [
        str(event.get("text") or "")
        for event in ctx.attempt_events()
        if event.get("kind") == "message"
        and event.get("direction") == "reply"
    ] + replies
    require(
        not any(
            "引用文字不能创建或恢复确认权限" in reply
            for reply in all_reply_text
        ),
        f"S20 hit the quoted-ticket refusal: {all_reply_text}",
    )
    return {
        "messages": messages,
        "replies": replies,
        "draft": draft,
        "facts": {
            "advertisedPairs": [list(pair) for pair in advertised_pairs],
            "nativeQuoteMessageId": quote_message_id,
            "additionalConfirmationSteps": additional_confirmation_steps,
            "batchId": draft.get("batchId"),
        },
    }


S21_BATCH_WORDS = (
    "显眼包",
    "嘴替",
)


async def scenario_s21(ctx: ScenarioContext) -> dict[str, Any]:
    async def discover() -> tuple[str, tuple[tuple[str, str], ...]]:
        message = "喵喵 加词 " + " ".join(S21_BATCH_WORDS)
        messages.append(message)
        reply = await ctx.send_group(message, to_me=True)
        replies.append(reply)
        pairs = advertised_batch_binding_pairs(reply)
        require(
            len(pairs) == len(S21_BATCH_WORDS)
            and tuple(word for word, _code in pairs) == S21_BATCH_WORDS,
            f"S21 did not persist/render the exact live multi-word ticket: {reply}",
        )
        return reply, pairs

    async def establish_clean_ticket(label: str) -> tuple[
        tuple[tuple[str, str], ...],
        dict[str, Any],
    ]:
        cleanup = await ctx.next_client.clean_draft(ctx.platform_id)
        require(
            cleanup.get("success") is True,
            f"S21 {label} draft cleanup failed: {cleanup}",
        )
        clean_snapshot = await ctx.draft()
        require(
            not clean_snapshot.get("items"),
            f"S21 {label} requires an empty actor draft before ticket creation: "
            f"{clean_snapshot}",
        )
        await ctx.bot.reset_conversation(platform_id=ctx.platform_id)
        _reply, pairs = await discover()
        return pairs, cleanup

    def batch_write_events(after_sequence: int) -> list[dict[str, Any]]:
        return [
            event
            for event in ctx.attempt_events()
            if int(event.get("sequence") or 0) > after_sequence
            and event.get("kind") == "tool"
            and event.get("name") == "keytao_batch_add_to_draft"
        ]

    messages: list[str] = []
    replies: list[str] = []
    first_pairs, first_precondition_cleanup = await establish_clean_ticket("first")
    excluded_word = S21_BATCH_WORDS[-1]
    expected_remaining = S21_BATCH_WORDS[:-1]

    outside_control = "都加 跳过火星词"
    messages.append(outside_control)
    replies.append(await ctx.send_group(outside_control, to_me=True))
    require(
        "火星词" in replies[-1]
        and all(word in replies[-1] for word in S21_BATCH_WORDS),
        f"S21 out-of-ticket exclusion did not ASK with the live words: {replies[-1]}",
    )
    require(
        not (await ctx.draft()).get("items"),
        "S21 out-of-ticket exclusion wrote before a valid selection",
    )

    modifier_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    modifier = f"都加 跳过{excluded_word}"
    messages.append(modifier)
    replies.append(await ctx.send_group(modifier, to_me=True))
    first_draft = await ctx.draft()
    modifier_confirmation_steps = 0
    if not first_draft.get("items"):
        require(
            "确认" in replies[-1],
            f"S21 modifier neither wrote nor reached one bound confirmation: {replies[-1]}",
        )
        modifier_confirmation_steps = 1
        messages.append("确认")
        replies.append(await ctx.send_group("确认", to_me=True))
        first_draft = await ctx.draft()
    first_items = [
        item for item in first_draft.get("items", []) if isinstance(item, dict)
    ]
    first_words = tuple(str(item.get("word") or "") for item in first_items)
    require(
        same_unique_item_set(first_words, expected_remaining),
        f"S21 modifier did not write ticket minus {excluded_word}: {first_draft}",
    )
    require(
        excluded_word not in {str(item.get("word") or "") for item in first_items},
        f"S21 excluded word reached the draft: {first_draft}",
    )
    modifier_writes = batch_write_events(modifier_cutoff)
    require(modifier_writes, "S21 modifier never reached the batch tool")
    for event in modifier_writes:
        tool_words = [
            str(item.get("word") or "")
            for item in event.get("arguments", {}).get("items", [])
            if isinstance(item, dict)
        ]
        require(
            same_unique_item_set(tool_words, expected_remaining),
            f"S21 modifier tool call escaped the resolved exact set: {event}",
        )
    first_batch_id = str(first_draft.get("batchId") or "")
    require(first_batch_id, f"S21 modifier did not materialize one batch: {first_draft}")

    second_pairs, second_precondition_cleanup = await establish_clean_ticket("second")
    require(second_pairs == first_pairs, "S21 second live ticket changed advertised bindings")

    messages.append("提交草稿")
    guidance = await ctx.send_group("提交草稿", to_me=True)
    replies.append(guidance)
    require(
        guidance.count(pending_confirmation_copy()) == 1
        and "请引用本条消息回复" not in guidance,
        f"S21 precedence refusal did not render one coherent option block: {guidance}",
    )
    require(
        all(
            f"「{word}」→ {code}" in guidance
            for word, code in second_pairs
        ),
        f"S21 precedence refusal did not enumerate the live record: {guidance}",
    )
    require(
        "或取消当前票据" not in guidance,
        f"S21 precedence refusal retained the broken cancel-only block: {guidance}",
    )
    rendered_line = "确认"

    unrelated = "请阅读" + rendered_line
    unrelated_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append(unrelated)
    replies.append(await ctx.send_group(unrelated, to_me=True))
    require(
        not batch_write_events(unrelated_cutoff),
        "S21 unrelated text outside the quote authorized a batch write",
    )
    require(
        not (await ctx.draft()).get("items"),
        "S21 unrelated quote envelope changed the draft",
    )

    rendered_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append(rendered_line)
    replies.append(await ctx.send_group(rendered_line, to_me=True))
    rendered_draft = await ctx.draft()
    rendered_confirmation_steps = 0
    if not rendered_draft.get("items"):
        require(
            "确认" in replies[-1],
            f"S21 rendered line neither wrote nor reached confirmation: {replies[-1]}",
        )
        rendered_confirmation_steps = 1
        messages.append("确认")
        replies.append(await ctx.send_group("确认", to_me=True))
        rendered_draft = await ctx.draft()
    rendered_items = [
        item for item in rendered_draft.get("items", []) if isinstance(item, dict)
    ]
    rendered_pairs = tuple(
        (
            str(item.get("word") or "").strip(),
            str(item.get("code") or "").strip().lower(),
        )
        for item in rendered_items
    )
    require(
        same_unique_item_set(rendered_pairs, second_pairs),
        f"S21 rendered remediation did not execute the record exact set: {rendered_draft}",
    )
    rendered_writes = batch_write_events(rendered_cutoff)
    require(rendered_writes, "S21 rendered remediation never reached the batch tool")
    for event in rendered_writes:
        tool_pairs = tuple(
            (
                str(item.get("word") or "").strip(),
                str(item.get("code") or "").strip().lower(),
            )
            for item in event.get("arguments", {}).get("items", [])
            if isinstance(item, dict)
        )
        require(
            same_unique_item_set(tool_pairs, second_pairs),
            f"S21 rendered remediation tool call escaped the live ticket: {event}",
        )

    return {
        "messages": messages,
        "replies": replies,
        "draft": rendered_draft,
        "facts": {
            "advertisedWords": list(S21_BATCH_WORDS),
            "excludedWord": excluded_word,
            "resolvedWords": list(expected_remaining),
            "modifierBatchId": first_batch_id,
            "modifierConfirmationSteps": modifier_confirmation_steps,
            "renderedRemediationLine": rendered_line,
            "renderedConfirmationSteps": rendered_confirmation_steps,
            "outOfTicketControl": "ASK-without-write",
            "unrelatedQuoteControl": "blocked-without-write",
            "renderedBatchId": rendered_draft.get("batchId"),
            "preconditionCleanups": {
                "first": first_precondition_cleanup,
                "second": second_precondition_cleanup,
            },
        },
    }


S22_BATCH_WORDS = S19_ADVERTISED_WORDS[:2]


async def scenario_s22(ctx: ScenarioContext) -> dict[str, Any]:
    """Re-review after state loss must couple its advertisement to a live ticket."""
    messages: list[str] = []
    replies: list[str] = []

    cleanup = await ctx.next_client.clean_draft(ctx.platform_id)
    require(
        cleanup.get("success") is True,
        f"S22 draft cleanup failed: {cleanup}",
    )
    require(
        not (await ctx.draft()).get("items"),
        "S22 requires an empty actor draft before candidate discovery",
    )
    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)

    discovery_message = "喵喵 加词 " + " ".join(S22_BATCH_WORDS)
    messages.append(discovery_message)
    discovery = await ctx.send_group(discovery_message, to_me=True)
    replies.append(discovery)
    discovery_message_id = ctx.last_reply_message_id
    discovery_pairs = advertised_batch_binding_pairs(discovery)
    require(
        discovery_message_id is not None,
        "S22 discovery did not expose a bot message id",
    )
    require(
        len(discovery_pairs) == len(S22_BATCH_WORDS)
        and same_unique_item_set(
            tuple(word for word, _code in discovery_pairs),
            S22_BATCH_WORDS,
        ),
        f"S22 discovery did not render the exact word set: {discovery}",
    )
    require(
        not (await ctx.draft()).get("items"),
        "S22 discovery wrote before assent",
    )

    # Reproduce the incident's missing-state precondition, then ask a later
    # read-only turn to re-resolve every word from server review results.
    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)
    word_count = len(S22_BATCH_WORDS)
    rereview_message = (
        f"喵喵 请只重新复核以下 {word_count} 个词的读音、编码和占用状态。"
        "逐项重列，明确标出每个词当前的推荐编码，不得省略或增加词条；"
        "末尾说明可直接回复「加入并提交」："
        + "、".join(S22_BATCH_WORDS)
    )
    messages.append(rereview_message)
    rereview = await ctx.send_group_reply(
        rereview_message,
        reply_message_id=discovery_message_id,
        to_me=True,
    )
    replies.append(rereview)
    rereview_message_id = ctx.last_reply_message_id
    rereview_pairs = advertised_batch_binding_pairs(rereview)
    require(
        rereview_message_id is not None
        and rereview_message_id != discovery_message_id,
        "S22 re-review did not expose a distinct bot message id",
    )
    require(
        len(rereview_pairs) == word_count
        and same_unique_item_set(
            tuple(word for word, _code in rereview_pairs),
            S22_BATCH_WORDS,
        ),
        f"S22 re-review invented, dropped, or omitted a displayed binding: {rereview}",
    )
    require(
        "加入并提交" in rereview,
        f"S22 re-review did not advertise the incident assent path: {rereview}",
    )
    require(
        not any(
            marker in rereview
            for marker in (
                "添加 词条 编码",
                "加入 词条 编码",
                "添加 XX",
                "加入 XX",
            )
        ),
        f"S22 re-review advertised a placeholder command: {rereview}",
    )
    require(
        not (await ctx.draft()).get("items"),
        "S22 re-review wrote before the bare assent",
    )
    establishment_events = [
        event
        for event in ctx.attempt_events()
        if event.get("kind") == "log"
        and "branch=establish_from_server_records"
        in str(event.get("message") or "")
        and f"items={word_count}" in str(event.get("message") or "")
    ]
    require(
        establishment_events,
        "S22 re-review advertised executable forms without establishing the "
        f"exact {word_count}-item server-record ticket: {rereview}",
    )

    write_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append("加入并提交")
    assent_reply = await ctx.send_group("加入并提交", to_me=True)
    replies.append(assent_reply)
    draft = await ctx.draft()
    completed_batch_id = _successful_submit_batch_id(
        ctx.attempt_events(),
        after_sequence=write_cutoff,
    )
    confirmation_steps = 0
    if not draft.get("items") and not completed_batch_id:
        require(
            "确认" in assent_reply,
            f"S22 bare assent neither wrote nor reached confirmation: {assent_reply}",
        )
        confirmation_steps = 1
        messages.append("确认")
        replies.append(await ctx.send_group("确认", to_me=True))
        draft = await ctx.draft()
        completed_batch_id = _successful_submit_batch_id(
            ctx.attempt_events(),
            after_sequence=write_cutoff,
        )

    batch_status = "Draft"
    auto_approved = False
    expected_item_keys = tuple(
        ("Create", word, code)
        for word, code in rereview_pairs
    )
    if completed_batch_id:
        batch_id = completed_batch_id
        completed_batch = await ctx.next_client.get_admin_batch(
            batch_id=batch_id,
            admin_token=ctx.admin_token,
        )
        actual_item_keys = tuple(
            item_key(item)
            for item in completed_batch.get("pullRequests", [])
            if isinstance(item, dict)
        )
        actual_pairs = tuple((word, code) for _action, word, code in actual_item_keys)
        batch_status = str(completed_batch.get("status") or "")
        require(
            batch_status in {"Submitted", "Approved"},
            f"S22 completed batch did not pass through submission: {completed_batch}",
        )
        successful_submit_results = [
            event.get("result")
            for event in ctx.attempt_events()
            if int(event.get("sequence") or 0) > write_cutoff
            and event.get("kind") == "tool"
            and event.get("name") == "keytao_submit_batch"
            and isinstance(event.get("result"), dict)
            and event["result"].get("success") is True
            and str(event["result"].get("batchId") or "") == batch_id
        ]
        auto_approved = any(
            result.get("autoApproved") is True
            for result in successful_submit_results
        )
        if auto_approved or "批次已加入词库" in replies[-1]:
            require(
                batch_status == "Approved",
                f"S22 claimed completed approval without an approved batch: {completed_batch}",
            )
        require(
            same_unique_item_set(
                advertised_batch_binding_pairs(replies[-1]),
                rereview_pairs,
            ),
            f"S22 completion copy did not honestly render the exact batch: {replies[-1]}",
        )
    else:
        batch_id = str(draft.get("batchId") or "")
        actual_item_keys = tuple(
            item_key(item)
            for item in draft.get("items", [])
            if isinstance(item, dict)
        )
        actual_pairs = tuple((word, code) for _action, word, code in actual_item_keys)
        require(batch_id, f"S22 did not materialize a draft batch: {draft}")

    require(
        same_unique_item_set(actual_item_keys, expected_item_keys)
        and same_unique_item_set(actual_pairs, rereview_pairs),
        "S22 advertised path did not land the exact displayed bindings in one batch: "
        f"status={batch_status}, actual={actual_item_keys}",
    )
    linked_batch_ids = batch_link_ids(replies[-1])
    require(
        not linked_batch_ids or linked_batch_ids == {batch_id},
        "S22 completion copy exposed a mismatched batch URL: "
        f"linked={sorted(linked_batch_ids)}, batch={batch_id}",
    )

    write_events = [
        event
        for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > write_cutoff
        and event.get("kind") == "tool"
        and event.get("name") == "keytao_batch_add_to_draft"
    ]
    require(write_events, "S22 advertised path never reached the batch tool")
    require(
        any(
            isinstance(event.get("result"), dict)
            and event["result"].get("success") is True
            and str(event["result"].get("batchId") or "") == batch_id
            for event in write_events
        ),
        f"S22 batch tool never materialized the completed batch: {write_events}",
    )
    for event in write_events:
        event_pairs = tuple(
            (
                str(item.get("word") or "").strip(),
                str(item.get("code") or "").strip().lower(),
            )
            for item in event.get("arguments", {}).get("items", [])
            if isinstance(item, dict)
        )
        require(
            same_unique_item_set(event_pairs, rereview_pairs),
            f"S22 batch tool escaped the re-established ticket: {event}",
        )

    require(
        not any("没有引用机器人给出的候选消息" in reply for reply in replies),
        f"S22 falsely diagnosed a missing quote: {replies}",
    )
    require(
        not any(
            marker in reply
            for reply in replies
            for marker in (
                "添加 词条 编码",
                "加入 词条 编码",
                "添加 XX",
                "加入 XX",
            )
        ),
        f"S22 exposed a placeholder remediation: {replies}",
    )
    return {
        "messages": messages,
        "replies": replies,
        "draft": draft,
        "facts": {
            "advertisedPairs": [list(pair) for pair in rereview_pairs],
            "discoveryPairs": [list(pair) for pair in discovery_pairs],
            "discoveryMessageId": discovery_message_id,
            "rereviewMessageId": rereview_message_id,
            "forcedStateLoss": True,
            "confirmationSteps": confirmation_steps,
            "batchId": batch_id,
            "batchStatus": batch_status,
            "autoApproved": auto_approved,
        },
    }


S23_BATCH_WORDS = S19_ADVERTISED_WORDS[:9]


async def scenario_s23(ctx: ScenarioContext) -> dict[str, Any]:
    """A stale advertised assent re-reviews and consumes that assent immediately."""
    messages: list[str] = []
    replies: list[str] = []

    cleanup = await ctx.next_client.clean_draft(ctx.platform_id)
    require(cleanup.get("success") is True, f"S23 draft cleanup failed: {cleanup}")
    require(
        not (await ctx.draft()).get("items"),
        "S23 requires an empty actor draft before candidate discovery",
    )
    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)

    discovery_message = "喵喵 加词 " + " ".join(S23_BATCH_WORDS)
    messages.append(discovery_message)
    discovery = await ctx.send_group(discovery_message, to_me=True)
    replies.append(discovery)
    stale_message_id = ctx.last_reply_message_id
    stale_pairs = advertised_batch_binding_pairs(discovery)
    require(stale_message_id is not None, "S23 discovery exposed no bot message id")
    require(
        len(stale_pairs) == len(S23_BATCH_WORDS)
        and tuple(word for word, _code in stale_pairs) == S23_BATCH_WORDS,
        f"S23 discovery did not render the exact stale candidate: {discovery}",
    )
    require(
        not (await ctx.draft()).get("items"),
        "S23 discovery wrote before assent",
    )

    # Force the production incident precondition: the bot message remains
    # quotable while every live candidate/ticket for this actor is gone.
    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)
    write_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append("加入并提交")
    recovered = await ctx.send_group_reply(
        "加入并提交",
        reply_message_id=stale_message_id,
        to_me=True,
    )
    replies.append(recovered)
    recovery_message_id = ctx.last_reply_message_id
    require(
        recovery_message_id is not None and recovery_message_id != stale_message_id,
        "S23 same-turn recovery exposed no distinct result message",
    )
    require(
        "加入并提交」则加入后提交" not in recovered
        and "回复「加入并提交」" not in recovered,
        f"S23 recovery re-prompted for the assent it had already received: {recovered}",
    )
    require(
        not any(
            marker in recovered
            for marker in (
                "可执行候选状态不存在",
                "没有匹配的可执行候选状态",
                "请重新发起",
            )
        ),
        f"S23 stale assent still dead-ended: {recovered}",
    )
    recovery_writes = [
        event
        for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > write_cutoff
        and event.get("kind") == "tool"
        and event.get("name") == "keytao_batch_add_to_draft"
    ]
    draft = await ctx.draft()
    completed_batch_id = _successful_submit_batch_id(
        ctx.attempt_events(),
        after_sequence=write_cutoff,
    )
    confirmation_steps = 0
    if not draft.get("items") and not completed_batch_id:
        require(
            pending_confirmation_copy() in recovered or "确认" in recovered,
            f"S23 recovered assent neither executed nor reached policy confirmation: {recovered}",
        )
        confirmation_steps = 1
        messages.append("确认")
        replies.append(await ctx.send_group("确认", to_me=True))
        draft = await ctx.draft()
        completed_batch_id = _successful_submit_batch_id(
            ctx.attempt_events(),
            after_sequence=write_cutoff,
        )

    batch_status = "Draft"
    expected_keys = tuple(("Create", word, code) for word, code in stale_pairs)
    if completed_batch_id:
        batch_id = completed_batch_id
        completed_batch = await ctx.next_client.get_admin_batch(
            batch_id=batch_id,
            admin_token=ctx.admin_token,
        )
        actual_keys = tuple(
            item_key(item)
            for item in completed_batch.get("pullRequests", [])
            if isinstance(item, dict)
        )
        batch_status = str(completed_batch.get("status") or "")
        require(
            batch_status in {"Submitted", "Approved"},
            f"S23 completed batch never reached submission: {completed_batch}",
        )
    else:
        batch_id = str(draft.get("batchId") or "")
        actual_keys = tuple(
            item_key(item)
            for item in draft.get("items", [])
            if isinstance(item, dict)
        )
        require(batch_id, f"S23 did not materialize one draft batch: {draft}")

    require(
        same_unique_item_set(actual_keys, expected_keys),
        "S23 recovered assent did not write exactly its displayed set: "
        f"expected={expected_keys}, actual={actual_keys}",
    )
    write_events = [
        event
        for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > write_cutoff
        and event.get("kind") == "tool"
        and event.get("name") == "keytao_batch_add_to_draft"
    ]
    require(write_events, "S23 recovered assent never reached the batch tool")
    for event in write_events:
        event_pairs = tuple(
            (
                str(item.get("word") or "").strip(),
                str(item.get("code") or "").strip().lower(),
            )
            for item in event.get("arguments", {}).get("items", [])
            if isinstance(item, dict)
        )
        require(
            same_unique_item_set(event_pairs, stale_pairs),
            f"S23 write escaped the recovered ticket: {event}",
        )

    return {
        "messages": messages,
        "replies": replies,
        "draft": draft,
        "facts": {
            "staleAdvertisedPairs": [list(pair) for pair in stale_pairs],
            "recoveredAppliedPairs": [list(pair) for pair in stale_pairs],
            "staleMessageId": stale_message_id,
            "recoveryMessageId": recovery_message_id,
            "forcedStateLoss": True,
            "recoveryWrites": len(recovery_writes),
            "confirmationSteps": confirmation_steps,
            "batchId": batch_id,
            "batchStatus": batch_status,
        },
    }


S24_WORD = "还车"
S24_RECOMMENDED_CODE = "htje"
S24_NATURAL_ASSENT = "加入草稿，然后就提交。"


async def scenario_s24(ctx: ScenarioContext) -> dict[str, Any]:
    """A native quote plus natural assent consumes one exact live candidate."""
    messages: list[str] = []
    replies: list[str] = []

    cleanup = await ctx.next_client.clean_draft(ctx.platform_id)
    require(cleanup.get("success") is True, f"S24 draft cleanup failed: {cleanup}")
    require(
        not (await ctx.draft()).get("items"),
        "S24 requires an empty actor draft before candidate discovery",
    )
    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)

    discovery_message = f"喵喵 {S24_WORD}"
    messages.append(discovery_message)
    discovery = await ctx.send_group(discovery_message, to_me=True)
    replies.append(discovery)
    discovery_message_id = ctx.last_reply_message_id
    require(
        isinstance(discovery_message_id, int) and discovery_message_id > 0,
        "S24 discovery exposed no bot message id",
    )
    binding = re.search(
        rf"(?:推荐编码[：:]|是否以编码)\s*"
        rf"(?P<code>[a-z]{{1,12}})"
        rf"(?:（本次仅查询）|\s+将「{re.escape(S24_WORD)}」加入草稿)",
        discovery,
    )
    require(binding is not None, f"S24 discovery omitted its exact candidate: {discovery}")
    recommended_code = binding.group("code")
    require(
        recommended_code == S24_RECOMMENDED_CODE,
        f"S24 fixture drifted from {S24_RECOMMENDED_CODE}: {discovery}",
    )
    advertised_forms = advertised_reply_contract(discovery).batch_assent_forms
    require(
        advertised_forms == ("加入", "加入并提交"),
        f"S24 single-word copy advertised the wrong forms: {advertised_forms}; {discovery}",
    )
    require(
        not (await ctx.draft()).get("items"),
        "S24 discovery wrote before assent",
    )

    cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append(S24_NATURAL_ASSENT)
    assent_reply = await ctx.send_group_reply(
        S24_NATURAL_ASSENT,
        reply_message_id=discovery_message_id,
        to_me=True,
    )
    replies.append(assent_reply)

    completed_batch_id = _successful_submit_batch_id(
        ctx.attempt_events(),
        after_sequence=cutoff,
    )
    confirmation_steps = 0
    if not completed_batch_id:
        confirmation_command = "确认" if "确认" in assent_reply else ""
        require(
            bool(confirmation_command),
            f"S24 natural assent neither submitted nor reached one bound confirmation: {assent_reply}",
        )
        confirmation_steps = 1
        messages.append(confirmation_command)
        replies.append(await ctx.send_group(confirmation_command, to_me=True))
        completed_batch_id = _successful_submit_batch_id(
            ctx.attempt_events(),
            after_sequence=cutoff,
        )

    require(completed_batch_id, "S24 natural assent never completed submission")
    require(confirmation_steps <= 1, "S24 used more than one server-bound confirmation")
    completed_batch = await ctx.next_client.get_admin_batch(
        batch_id=completed_batch_id,
        admin_token=ctx.admin_token,
    )
    batch_status = str(completed_batch.get("status") or "")
    require(
        batch_status in {"Submitted", "Approved"},
        f"S24 batch never reached submission: {completed_batch}",
    )
    submitted_items = [
        item
        for item in completed_batch.get("pullRequests", [])
        if isinstance(item, dict)
    ]
    require(
        len(submitted_items) == 1
        and item_key(submitted_items[0])
        == ("Create", S24_WORD, S24_RECOMMENDED_CODE),
        f"S24 submitted a target outside the live candidate: {completed_batch}",
    )

    post_assent_replies = replies[1:]
    forbidden_copy = (
        "把词条和编码写完整",
        "完整指令",
        "未能匹配",
        "没有匹配当前可执行候选",
    )
    require(
        not any(
            marker in reply
            for reply in post_assent_replies
            for marker in forbidden_copy
        ),
        f"S24 natural assent hit dishonest full-operand remediation: {post_assent_replies}",
    )
    write_events = [
        event
        for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > cutoff
        and event.get("kind") == "tool"
        and event.get("name") in {"keytao_create_phrase", "keytao_batch_add_to_draft"}
    ]
    require(write_events, "S24 natural assent never reached a draft write tool")
    for event in write_events:
        arguments = event.get("arguments", {})
        event_pairs = (
            tuple(
                (
                    str(item.get("word") or "").strip(),
                    str(item.get("code") or "").strip().lower(),
                )
                for item in arguments.get("items", [])
                if isinstance(item, dict)
            )
            if event.get("name") == "keytao_batch_add_to_draft"
            else ((
                str(arguments.get("word") or "").strip(),
                str(arguments.get("code") or "").strip().lower(),
            ),)
        )
        require(
            event_pairs == ((S24_WORD, S24_RECOMMENDED_CODE),),
            f"S24 write escaped the live candidate state: {event}",
        )

    return {
        "messages": messages,
        "replies": replies,
        "draft": await ctx.draft(),
        "facts": {
            "word": S24_WORD,
            "code": S24_RECOMMENDED_CODE,
            "naturalAssent": S24_NATURAL_ASSENT,
            "advertisedForms": list(advertised_forms),
            "discoveryMessageId": discovery_message_id,
            "confirmationSteps": confirmation_steps,
            "batchId": completed_batch_id,
            "batchStatus": batch_status,
        },
    }


S25_WORD = "炒冷饭"
S25_PREFIX_CODE = "wlf"
S25_REVIEWED_PREFIX_CODE = "jlf"
S25_SELECTED_CODE = S25_REVIEWED_PREFIX_CODE
S25_NATURAL_ADD = f"补上{S25_WORD}的 {S25_PREFIX_CODE} 编码"
S25_COMBINED_COMMAND = f"添加 {S25_WORD} {S25_SELECTED_CODE} 并提交"


async def scenario_s25(ctx: ScenarioContext) -> dict[str, Any]:
    """Replay the natural-add, record-backed number, and combined-submit incident."""
    messages: list[str] = []
    replies: list[str] = []

    cleanup = await ctx.next_client.clean_draft(ctx.platform_id)
    require(cleanup.get("success") is True, f"S25 initial cleanup failed: {cleanup}")
    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)

    natural_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append(S25_NATURAL_ADD)
    replies.append(await ctx.send_group(S25_NATURAL_ADD, to_me=True))
    natural_calls = [
        event
        for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > natural_cutoff
        and event.get("kind") == "tool"
        and event.get("name") == "keytao_create_phrase"
        and str(event.get("arguments", {}).get("word") or "").strip() == S25_WORD
        and str(event.get("arguments", {}).get("code") or "").strip().lower()
        == S25_PREFIX_CODE
    ]
    require(
        natural_calls,
        "S25 natural add verb never reached the duplicate-code write gate",
    )

    # The first request may have materialized a duplicate after its server
    # warning. Start candidate selection from a clean actor state so that the
    # numbered and combined-command subcases cannot pass via stale draft data.
    cleanup = await ctx.next_client.clean_draft(ctx.platform_id)
    require(cleanup.get("success") is True, f"S25 post-natural cleanup failed: {cleanup}")
    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)

    discovery_message = f"喵喵 {S25_WORD}"
    messages.append(discovery_message)
    discovery = await ctx.send_group(discovery_message, to_me=True)
    replies.append(discovery)
    selected_index = _rendered_candidate_index(discovery, S25_SELECTED_CODE)
    require(
        re.search(
            rf"(?m)^\d+\.\s*{re.escape(S25_REVIEWED_PREFIX_CODE)}\b.*空位.*$",
            discovery,
        )
        is not None,
        f"S25 did not render the encode-service series prefix: {discovery}",
    )
    require(
        re.search(r"(?m)^\d+\.\s*wlf(?:o+)?\b", discovery) is None,
        f"S25 leaked the explicit-code fixture into the reviewed chain: {discovery}",
    )
    require(
        re.search(
            rf"(?m)^{selected_index}\.\s*{re.escape(S25_SELECTED_CODE)}\b.*空位.*$",
            discovery,
        )
        is not None,
        f"S25 did not render {S25_SELECTED_CODE} as the selectable empty slot: {discovery}",
    )

    number_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    number_message = str(selected_index)
    messages.append(number_message)
    replies.append(await ctx.send_group(number_message, to_me=True))
    selected_draft = await ctx.draft()
    require(
        len(selected_draft.get("items", [])) == 1
        and item_key(selected_draft["items"][0])
        == ("Create", S25_WORD, S25_SELECTED_CODE),
        f"S25 bare number did not execute the record-selected item: {selected_draft}",
    )
    number_writes = [
        event
        for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > number_cutoff
        and event.get("kind") == "tool"
        and event.get("name") == "keytao_create_phrase"
        and isinstance(event.get("result"), dict)
        and event["result"].get("success") is True
    ]
    require(number_writes, "S25 bare number never reached a successful write receipt")
    require(
        all(
            (
                str(event.get("arguments", {}).get("word") or "").strip(),
                str(event.get("arguments", {}).get("code") or "").strip().lower(),
            )
            == (S25_WORD, S25_SELECTED_CODE)
            for event in number_writes
        ),
        f"S25 bare number escaped its trusted candidate record: {number_writes}",
    )

    cleanup = await ctx.next_client.clean_draft(ctx.platform_id)
    require(cleanup.get("success") is True, f"S25 pre-combined cleanup failed: {cleanup}")
    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)

    combined_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append(S25_COMBINED_COMMAND)
    combined_reply = await ctx.send_group(S25_COMBINED_COMMAND, to_me=True)
    replies.append(combined_reply)
    completed_batch_id = _successful_submit_batch_id(
        ctx.attempt_events(),
        after_sequence=combined_cutoff,
    )
    confirmation_steps = 0
    if not completed_batch_id:
        confirmation_command = "确认" if "确认" in combined_reply else ""
        require(
            bool(confirmation_command),
            f"S25 combined command neither submitted nor reached confirmation: {combined_reply}",
        )
        confirmation_steps = 1
        messages.append(confirmation_command)
        replies.append(await ctx.send_group(confirmation_command, to_me=True))
        completed_batch_id = _successful_submit_batch_id(
            ctx.attempt_events(),
            after_sequence=combined_cutoff,
        )

    require(completed_batch_id, "S25 combined add-and-submit never completed")
    require(confirmation_steps <= 1, "S25 used more than one server-bound confirmation")
    completion_exchange = "\n".join(replies[-(confirmation_steps + 1):])
    escaped_word = re.escape(S25_WORD)
    escaped_code = re.escape(S25_SELECTED_CODE)
    require(
        re.search(
            rf"已将[「\"]?{escaped_word}[」\"]?\s*→\s*{escaped_code}"
            r"\s*写入草稿",
            completion_exchange,
        ) is not None
        and re.search(
            r"已提交审核",
            completion_exchange,
        ) is not None
        and completed_batch_id not in completion_exchange.replace(
            f"/batch/{completed_batch_id}",
            "",
        )
        and re.search(r"未写入|未提交|提交未完成|无法执行", completion_exchange)
        is None,
        "S25 completion replies did not truthfully report both completed steps: "
        f"{completion_exchange}",
    )
    combined_events = [
        event
        for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > combined_cutoff
        and event.get("kind") == "tool"
    ]
    successful_creates = [
        event
        for event in combined_events
        if event.get("name") == "keytao_create_phrase"
        and isinstance(event.get("result"), dict)
        and event["result"].get("success") is True
    ]
    require(
        successful_creates
        and all(
            (
                str(event.get("arguments", {}).get("word") or "").strip(),
                str(event.get("arguments", {}).get("code") or "").strip().lower(),
            )
            == (S25_WORD, S25_SELECTED_CODE)
            for event in successful_creates
        ),
        f"S25 combined command did not write its exact same-turn item: {combined_events}",
    )

    completed_batch = await ctx.next_client.get_admin_batch(
        batch_id=completed_batch_id,
        admin_token=ctx.admin_token,
    )
    batch_status = str(completed_batch.get("status") or "")
    submitted_items = [
        item
        for item in completed_batch.get("pullRequests", [])
        if isinstance(item, dict)
    ]
    require(
        batch_status in {"Submitted", "Approved"}
        and len(submitted_items) == 1
        and item_key(submitted_items[0])
        == ("Create", S25_WORD, S25_SELECTED_CODE),
        f"S25 submitted batch differs from the combined command: {completed_batch}",
    )

    refusal_copy = (
        "安全层拦截",
        "只读轮",
        "无法执行",
        "本次未写入",
        "当前没有可安全执行" + "的后续命令",
        "请把下面这条指令原样转述给用户",
    )
    require(
        not any(marker in reply for reply in replies for marker in refusal_copy),
        f"S25 surfaced refusal copy on an executable path: {replies}",
    )
    return {
        "messages": messages,
        "replies": replies,
        "draft": await ctx.draft(),
        "facts": {
            "word": S25_WORD,
            "prefixCode": S25_PREFIX_CODE,
            "selectedCode": S25_SELECTED_CODE,
            "selectedIndex": selected_index,
            "naturalVerbReachedWriteGate": True,
            "bareNumberWroteFromRecord": True,
            "combinedCommand": S25_COMBINED_COMMAND,
            "confirmationSteps": confirmation_steps,
            "batchId": completed_batch_id,
            "batchStatus": batch_status,
        },
    }


S26_WORD = "吃席"
S26_CODE = "wkxk"
S26_OCCUPANT = "赤溪"
S26_COMMAND = f"添加 {S26_WORD} {S26_CODE}，{S26_OCCUPANT}顺延"


async def scenario_s26(ctx: ScenarioContext) -> dict[str, Any]:
    """Replay one server-resolved add plus named-occupant eviction."""
    cleanup = await ctx.next_client.clean_draft(ctx.platform_id)
    require(cleanup.get("success") is True, f"S26 cleanup failed: {cleanup}")
    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)
    cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages = [S26_COMMAND]
    replies = [await ctx.send_group(S26_COMMAND, to_me=True)]
    next_code = ctx.fixture_facts["chixi_next_code"]
    expected = {
        ("Delete", S26_OCCUPANT, S26_CODE),
        ("Create", S26_WORD, S26_CODE),
        ("Create", S26_OCCUPANT, next_code),
    }
    draft = await ctx.draft()
    confirmation_steps = 0
    if {item_key(item) for item in draft.get("items", [])} != expected:
        confirmation_command = "确认" if "确认" in replies[-1] else ""
        require(
            bool(confirmation_command),
            f"S26 neither completed nor returned one confirmation: {replies[-1]}",
        )
        confirmation_steps = 1
        messages.append(confirmation_command)
        replies.append(await ctx.send_group(confirmation_command, to_me=True))
        draft = await ctx.draft()

    actual = {item_key(item) for item in draft.get("items", [])}
    require(
        len(draft.get("items", [])) == 3 and actual == expected,
        f"S26 did not atomically add and evict: {draft}",
    )
    require(
        confirmation_steps <= 1,
        "S26 used more than one confirmation",
    )
    completion = "\n".join(replies[-(confirmation_steps + 1):])
    assert_reply_mentions(
        completion,
        S26_WORD,
        S26_CODE,
        S26_OCCUPANT,
        next_code,
    )
    require(
        re.search(r"未执行任何新写入|没有成功写入|本次未写入", completion)
        is None,
        f"S26 denied its completed write: {completion}",
    )
    events = [
        event
        for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > cutoff
        and event.get("kind") == "tool"
    ]
    materialized_batches = {
        str(event.get("result", {}).get("batchId") or "").strip()
        for event in events
        if isinstance(event.get("result"), dict)
        and event["result"].get("success") is True
        and str(event["result"].get("batchId") or "").strip()
    }
    require(
        len(materialized_batches) == 1,
        f"S26 did not materialize exactly one batch: {events}",
    )
    duplicate_auto_confirms = [
        event
        for event in events
        if isinstance(event.get("result"), dict)
        and event["result"].get("autoConfirmedWarnings") is True
        and any(
            isinstance(warning, dict)
            and warning.get("warningType") == "duplicate_code"
            for warning in event["result"].get("warnings") or []
        )
    ]
    require(
        not duplicate_auto_confirms,
        f"S26 auto-confirmed duplicate creation: {duplicate_auto_confirms}",
    )
    return {
        "messages": messages,
        "replies": replies,
        "draft": draft,
        "facts": {
            "command": S26_COMMAND,
            "nextCode": next_code,
            "confirmationSteps": confirmation_steps,
            "batchId": next(iter(materialized_batches)),
            "actualItems": sorted(actual),
        },
    }


S37_WORD = "耙耙柑"
S37_OCCUPANT = "琵琶骨"
S37_TARGET_CODE = "ppg"
S37_COMMAND = "加词 耙耙柑 把琵琶骨顶掉"
S37_SELECTION = "1 重新编码"


async def scenario_s37(ctx: ScenarioContext) -> dict[str, Any]:
    """Close occupant-derived eviction and stale selected-slot recovery."""
    messages: list[str] = []
    replies: list[str] = []

    cleanup = await ctx.next_client.clean_draft(ctx.platform_id)
    require(cleanup.get("success") is True, f"S37 cleanup failed: {cleanup}")
    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)
    occupant_rows = [
        row
        for row in await ctx.next_client.phrases_by_word(S37_OCCUPANT)
        if row.get("word") == S37_OCCUPANT
        and row.get("code") == S37_TARGET_CODE
    ]
    require(
        len(occupant_rows) == 1,
        f"S37 requires {S37_OCCUPANT}@{S37_TARGET_CODE}: {occupant_rows}",
    )
    shifted_code = str(
        (ctx.fixture_facts.get("s37") or {}).get("shiftedCode") or ""
    ).strip()
    require(
        shifted_code and shifted_code != S37_TARGET_CODE,
        f"S37 fixture omitted the occupant's next free slot: {ctx.fixture_facts}",
    )

    messages.append(S37_COMMAND)
    eviction_reply = await ctx.send_group(S37_COMMAND, to_me=True)
    replies.append(eviction_reply)
    draft = await ctx.draft()
    require(
        bool(draft.get("items")),
        f"S37 named-occupant eviction did not execute directly: {eviction_reply}",
    )
    require(
        pending_confirmation_copy() not in eviction_reply,
        f"S37 named-occupant eviction requested redundant confirmation: {eviction_reply}",
    )
    expected = {
        ("Delete", S37_OCCUPANT, S37_TARGET_CODE),
        ("Create", S37_WORD, S37_TARGET_CODE),
        ("Create", S37_OCCUPANT, shifted_code),
    }
    actual = {item_key(item) for item in draft.get("items", [])}
    require(
        actual == expected and len(draft.get("items", [])) == 3,
        f"S37 did not materialize one front-insert plan: {draft}",
    )
    batch_id = str(draft.get("batchId") or "").strip()
    require(
        bool(batch_id) and batch_link_ids(eviction_reply) == {batch_id},
        f"S37 receipt omitted its materialized batch link: {eviction_reply}",
    )
    assert_only_materialized_batch_links([eviction_reply], draft)
    assert_reply_mentions(
        eviction_reply,
        f"「{S37_WORD}」 → {S37_TARGET_CODE}",
        f"「{S37_OCCUPANT}」 {S37_TARGET_CODE} → {shifted_code}",
    )

    cleanup = await ctx.next_client.clean_draft(ctx.platform_id)
    require(cleanup.get("success") is True, f"S37 recovery cleanup failed: {cleanup}")
    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)
    messages.append(f"喵喵 {S37_WORD}")
    discovery = await ctx.send_group(messages[-1], to_me=True)
    replies.append(discovery)
    require(
        S37_WORD in discovery and S37_TARGET_CODE in discovery,
        f"S37 current discovery omitted its target: {discovery}",
    )

    stale_display = (
        f"「{S37_WORD}」候选编码：\n"
        "1. zzzz — 已有「旧占位」\n"
        f"2. {S37_TARGET_CODE} — 已有「{S37_OCCUPANT}」\n"
        f"• 「{S37_WORD}」→ zzzz（推荐）\n"
        "若要挪开已有词，回复“1 重新编码”。"
    )
    stale_message_id = ctx.inject_bot_message(stale_display)
    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)
    messages.append(S37_SELECTION)
    refreshed = await ctx.send_group_reply(
        S37_SELECTION,
        reply_message_id=stale_message_id,
        to_me=True,
    )
    replies.append(refreshed)
    require(
        "已刷新为当前候选" in refreshed
        and S37_TARGET_CODE in refreshed
        and "候选编码集合已变化" not in refreshed
        and "移除 " not in refreshed,
        f"S37 stale selection did not recover with a fresh list: {refreshed}",
    )

    messages.append(S37_SELECTION)
    second = await ctx.send_group(S37_SELECTION, to_me=True)
    replies.append(second)
    require(
        second != refreshed
        and "候选编码集合已变化" not in second
        and "没有执行添加" not in second,
        f"S37 repeated selection returned the same dead-end: {second}",
    )
    final_cleanup = await ctx.next_client.clean_draft(ctx.platform_id)
    require(
        final_cleanup.get("success") is True,
        f"S37 final cleanup failed: {final_cleanup}",
    )
    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)
    return {
        "messages": messages,
        "replies": replies,
        "draft": draft,
        "facts": {
            "verbatimEviction": S37_COMMAND,
            "targetCode": S37_TARGET_CODE,
            "shiftedCode": shifted_code,
            "batchId": batch_id,
            "confirmationSteps": 0,
            "freshListRecovery": True,
            "identicalRefusalRepeated": False,
        },
    }


S27_WORD = "来都来了"
S27_ASSENT = "加入并提交"
S27_META_QUESTION = "你是否会先确认对方有没有绑定账号？"


async def scenario_s27(ctx: ScenarioContext) -> dict[str, Any]:
    """Warn an unbound actor at discovery, then keep meta chat conversational."""
    unbound_platform_id = "8" + ctx.platform_id[1:]
    require(
        await ctx.next_client.find_user(unbound_platform_id) is None,
        "S27 synthetic unbound actor unexpectedly resolved to a local user",
    )
    require(
        await ctx.next_client.find_user(ctx.platform_id) is not None,
        "S27 bound control actor did not resolve to its provisioned local user",
    )

    async def send_as(platform_id: str, sender_name: str, text: str) -> str:
        reply = await ctx.bot.send_group(
            platform_id=platform_id,
            sender_name=sender_name,
            text=text,
            to_me=True,
        )
        assert_batch_link_hosts(reply, "qq")
        return reply

    await ctx.bot.reset_conversation(platform_id=unbound_platform_id)
    qq_unbound_notice = render_platform_public_links(
        UNBOUND_BINDING_PRECHECK_NOTICE,
        "qq",
    )
    first_reply = await send_as(unbound_platform_id, "S27-unbound", S27_WORD)
    require(S27_WORD in first_reply, f"S27 unbound review omitted the word: {first_reply}")
    unbound_contract = advertised_reply_contract(first_reply)
    require(
        unbound_contract.read_only_single_word_lookup
        or unbound_contract.requires_live_state,
        f"S27 unbound review did not render a candidate contract: {first_reply}",
    )
    require(
        first_reply.count(qq_unbound_notice) == 1,
        f"S27 unbound first candidate reply did not carry exactly one notice: {first_reply}",
    )

    bind_reply = await send_as(unbound_platform_id, "S27-unbound", S27_ASSENT)
    assert_reply_mentions(bind_reply, "你还未绑定键道账号", "/bind", "/profile")

    cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    meta_reply = await send_as(
        unbound_platform_id,
        "S27-unbound",
        S27_META_QUESTION,
    )
    stance = re.search(
        r"(?:^|[\s，,。；;：:！!？?～~])"
        r"(?:会(?:的|先|在|于|检查|确认|校验)?|不会|是的|不是|有的|没有|否)",
        meta_reply,
    )
    require(
        "绑定" in meta_reply and stance is not None,
        f"S27 meta turn was not answered directly: {meta_reply}",
    )
    require(
        not any(marker in meta_reply for marker in SYSTEM_REPLY_TEMPLATE_MARKERS),
        f"S27 meta answer impersonated a system template: {meta_reply}",
    )
    require(
        "可执行命令：" not in meta_reply,
        f"S27 meta answer was a remediation reply: {meta_reply}",
    )
    meta_tools = [
        event
        for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > cutoff
        and event.get("kind") == "tool"
    ]
    require(not meta_tools, f"S27 pure meta question called tools: {meta_tools}")

    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)
    bound_reply = await send_as(ctx.platform_id, ctx.sender_name, S27_WORD)
    bound_contract = advertised_reply_contract(bound_reply)
    require(
        bound_contract.read_only_single_word_lookup
        or bound_contract.requires_live_state,
        f"S27 bound control did not render candidates: {bound_reply}",
    )
    require(
        qq_unbound_notice not in bound_reply,
        f"S27 bound control received the unbound notice: {bound_reply}",
    )
    draft = await ctx.draft()
    require(not draft.get("items"), f"S27 read-only flow changed the bound draft: {draft}")

    return {
        "messages": [S27_WORD, S27_ASSENT, S27_META_QUESTION, S27_WORD],
        "replies": [first_reply, bind_reply, meta_reply, bound_reply],
        "draft": draft,
        "facts": {
            "unboundPlatformId": unbound_platform_id,
            "bindingNoticeCount": first_reply.count(qq_unbound_notice),
            "bindingGuidanceAfterAssent": True,
            "metaQuestionToolCalls": len(meta_tools),
            "systemTemplateMarkersAbsent": True,
            "boundControlNoticeAbsent": True,
        },
    }


S28_WORD = "还车"
S28_DISCOVERY = f"喵喵 {S28_WORD}"
S28_INVALID_CODE = "zzzzzz"


def _rendered_candidate_rows(reply: str) -> list[tuple[int, str]]:
    return [
        (int(match.group("index")), match.group("code").lower())
        for match in re.finditer(
            r"(?m)^(?P<index>\d+)\.\s*(?P<code>[a-z]{1,12})\s+(?:—|–|-).*$",
            reply,
            re.IGNORECASE,
        )
    ]


def _rendered_recommended_code(reply: str) -> str:
    actionable = re.search(
        r"是否以编码\s+(?P<code>[a-z]{1,12})\s+将",
        reply,
        re.IGNORECASE,
    )
    read_only = re.search(
        r"(?m)^推荐编码[：:]\s*(?P<code>[a-z]{1,12})"
        r"(?:（本次仅查询）|\(本次仅查询\))\s*$",
        reply,
        re.IGNORECASE,
    )
    match = actionable or read_only
    require(match is not None, f"candidate reply omitted its recommendation: {reply}")
    return match.group("code").lower()


async def scenario_s28(ctx: ScenarioContext) -> dict[str, Any]:
    """Replay reviewed-write parity, live replacement, fresh number, and failure copy."""
    messages: list[str] = []
    replies: list[str] = []
    dictionary_cleanups: list[dict[str, Any]] = []

    async def reset_case(label: str) -> None:
        dictionary_cleanup = await ctx.next_client.remove_rig_owned_dictionary_words(
            platform_id=ctx.platform_id,
            admin_token=ctx.admin_token,
            scenario_id="S28",
            fixture_words=(S28_WORD,),
        )
        require(
            dictionary_cleanup.get("verified") is True,
            f"S28 {label} dictionary cleanup was not verified: {dictionary_cleanup}",
        )
        dictionary_cleanups.append(dictionary_cleanup)
        cleanup = await ctx.next_client.clean_draft(ctx.platform_id)
        require(cleanup.get("success") is True, f"S28 {label} cleanup failed: {cleanup}")
        await ctx.bot.reset_conversation(platform_id=ctx.platform_id)

    async def discover(label: str) -> tuple[str, list[tuple[int, str]], str]:
        messages.append(S28_DISCOVERY)
        reply = await ctx.send_group(S28_DISCOVERY, to_me=True)
        replies.append(reply)
        readings = _rendered_candidate_reading_rows(reply)
        rows = _rendered_candidate_rows(reply)
        recommended = _rendered_recommended_code(reply)
        require(len(readings) >= 2, f"S28 {label} did not render a multi-reading list: {reply}")
        require(len(rows) >= 4, f"S28 {label} rendered fewer than four candidates: {reply}")
        require(recommended in {code for _index, code in rows}, f"S28 {label} recommendation was not listed: {reply}")
        return reply, rows, recommended

    async def finish_submission(
        *,
        cutoff: int,
        first_reply: str,
        label: str,
        word: str,
        code: str,
    ) -> tuple[str, int, dict[str, Any]]:
        batch_id = _successful_submit_batch_id(
            ctx.attempt_events(),
            after_sequence=cutoff,
        )
        confirmation_steps = 0
        if not batch_id:
            command = "确认" if "确认" in first_reply else ""
            require(command, f"S28 {label} neither submitted nor returned a confirmation: {first_reply}")
            confirmation_steps = 1
            messages.append(command)
            confirmation_reply = await ctx.send_group(command, to_me=True)
            replies.append(confirmation_reply)
            batch_id = _successful_submit_batch_id(
                ctx.attempt_events(),
                after_sequence=cutoff,
            )
        require(batch_id, f"S28 {label} never completed submission")
        batch = await ctx.next_client.get_admin_batch(
            batch_id=batch_id,
            admin_token=ctx.admin_token,
        )
        require(
            str(batch.get("status") or "") in {"Submitted", "Approved"}
            and _submitted_item(batch, word=word, code=code) is not None,
            f"S28 {label} submitted the wrong item: {batch}",
        )
        return batch_id, confirmation_steps, batch

    # R1: the exact reviewed recommendation must remain writable.
    await reset_case("reviewed-assent")
    _first, _rows, advertised_code = await discover("reviewed-assent")
    assent_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append("加入")
    assent_reply = await ctx.send_group("加入", to_me=True)
    replies.append(assent_reply)
    assent_draft = await ctx.draft()
    require(
        len(assent_draft.get("items", [])) == 1
        and item_key(assent_draft["items"][0])
        == ("Create", S28_WORD, advertised_code),
        f"S28 reviewed recommendation was not write-valid: {assent_draft}; reply={assent_reply}",
    )
    require(
        any(
            int(event.get("sequence") or 0) > assent_cutoff
            and event.get("kind") == "tool"
            and event.get("name") == "keytao_create_phrase"
            and isinstance(event.get("result"), dict)
            and event["result"].get("success") is True
            for event in ctx.attempt_events()
        ),
        "S28 reviewed assent had no successful create receipt",
    )

    # R2: a complete same-word command replaces, rather than mutates, live state.
    await reset_case("explicit-replacement")
    _replacement_discovery, replacement_rows, replacement_recommended = await discover("explicit-replacement")
    replacement_code = next(code for index, code in replacement_rows if index == 4)
    replacement_command = f"添加 {S28_WORD} {replacement_code} 并提交"
    replacement_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append(replacement_command)
    replacement_reply = await ctx.send_group(replacement_command, to_me=True)
    replies.append(replacement_reply)
    replacement_batch_id, replacement_confirmations, _replacement_batch = await finish_submission(
        cutoff=replacement_cutoff,
        first_reply=replacement_reply,
        label="explicit replacement",
        word=S28_WORD,
        code=replacement_code,
    )
    require(
        "可执行命令：\n加入并提交" not in replacement_reply,
        f"S28 replacement contradicted the explicit code choice: {replacement_reply}",
    )

    # R3: a freshly rendered multi-reading state binds number 4 to its reading.
    await reset_case("fresh-number")
    _number_discovery, number_rows, _number_recommended = await discover("fresh-number")
    number_code = next(code for index, code in number_rows if index == 4)
    number_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    number_command = "添加4并提交"
    messages.append(number_command)
    number_reply = await ctx.send_group(number_command, to_me=True)
    replies.append(number_reply)
    number_batch_id, number_confirmations, _number_batch = await finish_submission(
        cutoff=number_cutoff,
        first_reply=number_reply,
        label="fresh number",
        word=S28_WORD,
        code=number_code,
    )
    require(
        "没有保留编码" not in "\n".join(replies[-(number_confirmations + 1):]),
        f"S28 fresh number lost its reading binding: {replies[-(number_confirmations + 1):]}",
    )

    # Control: invalid literal code is compact, honest, and side-effect free.
    await reset_case("invalid-control")
    invalid_discovery, invalid_rows, _invalid_recommended = await discover("invalid-control")
    invalid_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    invalid_command = f"添加 {S28_WORD} {S28_INVALID_CODE} 并提交"
    messages.append(invalid_command)
    invalid_reply = await ctx.send_group(invalid_command, to_me=True)
    replies.append(invalid_reply)
    invalid_draft = await ctx.draft()
    require(not invalid_draft.get("items"), f"S28 invalid control wrote a draft: {invalid_draft}")
    assert_reply_mentions(invalid_reply, S28_INVALID_CODE, "可选读音链", "没有执行添加")
    require(
        "草稿地址：https://keytao.vercel.app" not in invalid_reply
        and "草稿地址：http://localhost" not in invalid_reply,
        f"S28 invalid control exposed a bogus draft URL: {invalid_reply}",
    )
    mentioned_candidates = {
        code
        for _index, code in invalid_rows
        if re.search(rf"\b{re.escape(code)}\b", invalid_reply)
    }
    require(
        len(mentioned_candidates) < len(invalid_rows),
        f"S28 invalid control dumped the raw candidate list: {invalid_reply}",
    )
    invalid_writes = [
        event
        for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > invalid_cutoff
        and event.get("kind") == "tool"
        and event.get("name") in {"keytao_create_phrase", "keytao_submit_batch"}
    ]
    require(not invalid_writes, f"S28 invalid control reached a write tool: {invalid_writes}")

    return {
        "messages": messages,
        "replies": replies,
        "draft": invalid_draft,
        "facts": {
            "word": S28_WORD,
            "advertisedWriteCode": advertised_code,
            "replacementCode": replacement_code,
            "replacementBatchId": replacement_batch_id,
            "replacementConfirmationSteps": replacement_confirmations,
            "freshNumber": 4,
            "freshNumberCode": number_code,
            "freshNumberBatchId": number_batch_id,
            "freshNumberConfirmationSteps": number_confirmations,
            "invalidCode": S28_INVALID_CODE,
            "invalidWriteCalls": len(invalid_writes),
            "verifiedDictionaryCleanups": len(dictionary_cleanups),
        },
    }


S29_CODE = "mkdr"
S29_CURRENT = (("火锅", 100), ("电脑", 101))
S29_PROPOSED = (("电脑", 100), ("火锅", 101))
S29_INCIDENT_COMMAND = "mkdr 按优先级排一下"
S29_PRESENCE_CONTROL = "这两个词现在词库都有吗"


async def scenario_s29(ctx: ScenarioContext) -> dict[str, Any]:
    messages: list[str] = []
    replies: list[str] = []

    # Register the bot-authored operation-summary context directly at the
    # OneBot boundary. The scenario under test starts with the user's native
    # quote and therefore must not depend on an unrelated model-generated turn.
    summary_reply = (
        "✅ 操作已完成\n"
        "草稿变更：\n"
        "• Change：「火锅」mkdr，权重 101 → 102\n"
        "请检查以上变更，确认后提交。"
    )
    replies.append(summary_reply)
    assert_reply_mentions(summary_reply, "操作已完成", "mkdr", "提交")
    summary_message_id = ctx.inject_bot_message(summary_reply)
    require(
        isinstance(summary_message_id, int) and summary_message_id > 0,
        "S29 harness did not retain the operation-summary message id",
    )

    reorder_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append(S29_INCIDENT_COMMAND)
    plan_reply = await ctx.send_group_reply(
        S29_INCIDENT_COMMAND,
        reply_message_id=summary_message_id,
        to_me=True,
    )
    replies.append(plan_reply)
    assert_reply_mentions(
        plan_reply,
        "编码 mkdr 调整计划",
        "「电脑」：权重 101 → 100",
        "「火锅」：权重 100 → 101",
        "依据：",
    )
    require(
        plan_reply.count(pending_confirmation_copy()) == 1
        and len(plan_reply.splitlines()) <= 8,
        f"S29 plan was not compact with one shared confirmation: {plan_reply}",
    )
    draft_before_confirm = await ctx.draft()
    require(
        not draft_before_confirm.get("items"),
        f"S29 wrote before confirmation: {draft_before_confirm}",
    )
    reorder_events = [
        event
        for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > reorder_cutoff
        and event.get("kind") == "tool"
    ]
    require(
        any(event.get("name") == "keytao_lookup_by_code" for event in reorder_events)
        and not any(event.get("name") == "keytao_lookup_by_words_batch" for event in reorder_events),
        f"S29 incident turn did not stay on the code-chain route: {reorder_events}",
    )

    messages.append("确认")
    confirm_reply = await ctx.send_group("确认", to_me=True)
    replies.append(confirm_reply)
    assert_reply_mentions(confirm_reply, "已加入草稿", "火锅", "电脑", "mkdr")
    draft = await ctx.draft()
    actual = tuple(sorted(
        (
            str(item.get("word") or ""),
            item.get("weight"),
        )
        for item in draft.get("items", [])
        if isinstance(item, dict)
        and item.get("action") == "Change"
        and item.get("code") == S29_CODE
        and item.get("type") == "Phrase"
        and item.get("oldWord") == item.get("word")
    ))
    require(
        len(draft.get("items", [])) == 2
        and actual == tuple(sorted(S29_PROPOSED)),
        f"S29 confirmation did not write the exact proposed weights: {draft}",
    )
    batch_calls = [
        event
        for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > reorder_cutoff
        and event.get("kind") == "tool"
        and event.get("name") == "keytao_batch_add_to_draft"
    ]
    confirmed_calls = [
        event
        for event in batch_calls
        if event.get("arguments", {}).get("confirmed") is True
    ]
    require(
        len(confirmed_calls) == 1,
        f"S29 expected one confirmed batch replay: {batch_calls}",
    )

    presence_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append(S29_PRESENCE_CONTROL)
    presence_reply = await ctx.send_group_reply(
        S29_PRESENCE_CONTROL,
        reply_message_id=summary_message_id,
        to_me=True,
    )
    replies.append(presence_reply)
    assert_reply_mentions(
        presence_reply,
        "「操作已完成」：未收录",
        "「火锅」：已收录",
    )
    presence_events = [
        event
        for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > presence_cutoff
        and event.get("kind") == "tool"
    ]
    require(
        any(event.get("name") == "keytao_lookup_by_words_batch" for event in presence_events)
        and not any(
            event.get("name") == "keytao_batch_add_to_draft"
            for event in presence_events
        ),
        f"S29 presence control did not stay on the quoted lookup route: {presence_events}",
    )
    draft_after_control = await ctx.draft()
    require(
        tuple(sorted(
            (str(item.get("word") or ""), item.get("weight"))
            for item in draft_after_control.get("items", [])
            if isinstance(item, dict)
        )) == tuple(sorted(S29_PROPOSED)),
        f"S29 presence control changed the sealed draft: {draft_after_control}",
    )

    return {
        "messages": messages,
        "replies": replies,
        "draft": draft_after_control,
        "facts": {
            "code": S29_CODE,
            "quotedSummaryMessageId": summary_message_id,
            "currentOrder": [word for word, _weight in S29_CURRENT],
            "proposedOrder": [word for word, _weight in S29_PROPOSED],
            "confirmationSteps": 1,
            "confirmedBatchCalls": len(confirmed_calls),
            "presenceControlTool": "keytao_lookup_by_words_batch",
        },
    }


S30_WORD = "吃席"
S30_CANCEL = "先别加"
S30_NATURAL_ASSENT = "那就加入并提交吧"


async def scenario_s30(ctx: ScenarioContext) -> dict[str, Any]:
    """Pin read-only lookup, cancel-negation, and tolerant natural assent."""
    messages: list[str] = []
    replies: list[str] = []
    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)

    read_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append(S30_WORD)
    read_reply = await ctx.send(S30_WORD)
    replies.append(read_reply)
    assert_reply_mentions(read_reply, S30_WORD)
    require(
        not (await ctx.draft()).get("items"),
        f"S30 bare lookup wrote a draft item: {await ctx.draft()}",
    )

    messages.append("好")
    stale_reply = await ctx.send("好")
    replies.append(stale_reply)
    read_events = [
        event
        for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > read_cutoff
        and event.get("kind") == "tool"
        and event.get("name") in {
            "keytao_create_phrase",
            "keytao_batch_add_to_draft",
            "keytao_submit_batch",
            "keytao_shift_phrase_code",
        }
    ]
    require(not read_events, f"S30 read lookup armed a later write: {read_events}")
    require(
        not (await ctx.draft()).get("items"),
        f"S30 bare 好 after lookup wrote a draft item: {await ctx.draft()}",
    )

    messages.append(f"加词 {S30_WORD}")
    cancel_candidate = await ctx.send(f"加词 {S30_WORD}")
    replies.append(cancel_candidate)
    assert_reply_mentions(cancel_candidate, S30_WORD, "候选编码")
    cancel_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append(S30_CANCEL)
    cancel_reply = await ctx.send(S30_CANCEL)
    replies.append(cancel_reply)
    require(
        "取消" in cancel_reply or "放弃" in cancel_reply,
        f"S30 negated add did not cancel: {cancel_reply}",
    )
    require(
        "可执行命令" not in cancel_reply and "「加入" not in cancel_reply,
        f"S30 negation advertised an add command: {cancel_reply}",
    )
    cancel_writes = [
        event
        for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > cancel_cutoff
        and event.get("kind") == "tool"
        and event.get("name") in {
            "keytao_create_phrase",
            "keytao_batch_add_to_draft",
            "keytao_submit_batch",
            "keytao_shift_phrase_code",
        }
    ]
    require(not cancel_writes, f"S30 cancellation wrote: {cancel_writes}")
    require(
        not (await ctx.draft()).get("items"),
        f"S30 cancellation changed the draft: {await ctx.draft()}",
    )

    messages.append(f"加词 {S30_WORD}")
    natural_candidate = await ctx.send(f"加词 {S30_WORD}")
    replies.append(natural_candidate)
    assert_reply_mentions(natural_candidate, S30_WORD, "候选编码")
    recommended_code = _recommended_empty_code(natural_candidate, word=S30_WORD).lower()
    submit_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append(S30_NATURAL_ASSENT)
    assent_reply = await ctx.send(S30_NATURAL_ASSENT)
    replies.append(assent_reply)
    completed_batch_id = _successful_submit_batch_id(
        ctx.attempt_events(),
        after_sequence=submit_cutoff,
    )
    confirmation_steps = 0
    while not completed_batch_id and confirmation_steps < 3:
        require(
            "确认" in assent_reply,
            f"S30 natural assent neither progressed nor requested confirmation: {assent_reply}",
        )
        confirmation_steps += 1
        messages.append("确认")
        assent_reply = await ctx.send("确认")
        replies.append(assent_reply)
        completed_batch_id = _successful_submit_batch_id(
            ctx.attempt_events(),
            after_sequence=submit_cutoff,
        )
    require(completed_batch_id, f"S30 natural assent never submitted: {replies}")
    completed_batch = await ctx.next_client.get_admin_batch(
        batch_id=completed_batch_id,
        admin_token=ctx.admin_token,
    )
    actual_items = [
        item_key(item)
        for item in completed_batch.get("pullRequests", [])
        if isinstance(item, dict)
    ]
    require(
        same_unique_item_set(
            actual_items,
            [("Create", S30_WORD, recommended_code)],
        ),
        f"S30 natural assent submitted a different item set: {completed_batch}",
    )
    draft = await ctx.draft()
    return {
        "messages": messages,
        "replies": replies,
        "draft": draft,
        "facts": {
            "readQueryMutationCalls": len(read_events),
            "cancelMutationCalls": len(cancel_writes),
            "naturalAssent": S30_NATURAL_ASSENT,
            "recommendedCode": recommended_code,
            "submittedBatchId": completed_batch_id,
            "confirmationSteps": confirmation_steps,
        },
    }


S31_COMMAND = "把 幂等 放到 米等 前面，米等顺延到下一个空位"


async def scenario_s31(ctx: ScenarioContext) -> dict[str, Any]:
    """Execute the verbatim positional-plus-顺延 incident through the real path."""
    target_code = "mkdr"
    target_rows = await ctx.next_client.phrases_by_code(target_code)
    require(not target_rows, f"S31 requires an empty {target_code} slot: {target_rows}")
    await ctx.next_client.seed_phrase(
        platform_id=ctx.platform_id,
        word="米等",
        code=target_code,
        weight=100,
    )
    subject_encoding = await ctx.next_client.encode("幂等")
    occupant_encoding = await ctx.next_client.encode("米等")
    subject_codes = ordered_candidate_codes(subject_encoding)
    occupant_codes = ordered_candidate_codes(occupant_encoding)
    require(
        target_code in subject_codes and target_code in occupant_codes,
        f"S31 fixture encodings do not share {target_code}: "
        f"subject={subject_encoding}, occupant={occupant_encoding}",
    )
    target_index = occupant_codes.index(target_code)
    require(
        target_index + 1 < len(occupant_codes),
        f"S31 米等 has no served successor after {target_code}: {occupant_encoding}",
    )
    next_code = occupant_codes[target_index + 1]
    require(
        not await ctx.next_client.phrases_by_code(next_code),
        f"S31 successor {next_code} is occupied",
    )
    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)
    cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    reply = await ctx.send(S31_COMMAND)
    draft = await ctx.draft()
    expected = {
        ("Delete", "米等", target_code),
        ("Create", "幂等", target_code),
        ("Create", "米等", next_code),
    }
    actual = {item_key(item) for item in draft.get("items", [])}
    require(actual == expected, f"S31 verbatim command produced {draft}")
    assert_reply_mentions(reply, "幂等", "米等", target_code, next_code)
    assert_no_code_request(reply)
    events = [
        event
        for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > cutoff
        and event.get("kind") == "tool"
    ]
    shift_plan_events = [
        event
        for event in events
        if isinstance(event.get("result"), dict)
        and isinstance(event["result"].get("shiftPlan"), dict)
        and {
            item_key(item)
            for item in event["result"]["shiftPlan"].get("items", [])
            if isinstance(item, dict)
        }
        == expected
    ]
    require(
        any(event.get("name") == "keytao_lookup_by_word" for event in events)
        and bool(shift_plan_events),
        f"S31 did not traverse lookup plus shift planning: {events}",
    )
    return {
        "messages": [S31_COMMAND],
        "replies": [reply],
        "draft": draft,
        "facts": {
            "targetCode": target_code,
            "successorCode": next_code,
            "expectedItems": sorted(expected),
            "actualItems": sorted(actual),
        },
    }


S32_CODE = "mkdr"
S32_PREFIX_CODE = "mkdro"
S32_DRAFT_WORD = "幂等"
S32_CHAIN_COMMAND = "重新排序下mkdr 编码链这几个词按优先级"
S32_WORD_LIST_COMMAND = "重新排序下\n米等\n幂等\n迷瞪"


async def scenario_s32(ctx: ScenarioContext) -> dict[str, Any]:
    """Replay both chain-scope incidents against one live+draft merged view."""
    messages: list[str] = []
    replies: list[str] = []

    for code in (S32_CODE, S32_PREFIX_CODE):
        exact_rows = [
            row
            for row in await ctx.next_client.phrases_by_code(code)
            if str(row.get("code") or "") == code
        ]
        require(not exact_rows, f"S32 requires an empty {code} fixture slot: {exact_rows}")

    await ctx.next_client.seed_phrase(
        platform_id=ctx.platform_id,
        word="米等",
        code=S32_CODE,
        weight=100,
    )
    await ctx.next_client.seed_phrase(
        platform_id=ctx.platform_id,
        word="迷瞪",
        code=S32_PREFIX_CODE,
        weight=100,
    )
    await ctx.next_client.add_draft_items(
        platform_id=ctx.platform_id,
        items=[{
            "action": "Create",
            "word": S32_DRAFT_WORD,
            "code": S32_CODE,
            "type": "Phrase",
            "weight": 101,
            "needsManualReview": False,
            "remark": "S32 draft-aware chain fixture",
        }],
    )
    original_draft = await ctx.draft()
    require(
        len(original_draft.get("items", [])) == 1
        and item_key(original_draft["items"][0])
        == ("Create", S32_DRAFT_WORD, S32_CODE)
        and original_draft["items"][0].get("weight") == 101,
        f"S32 could not establish its current draft fixture: {original_draft}",
    )

    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)
    chain_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append(S32_CHAIN_COMMAND)
    chain_reply = await ctx.send(S32_CHAIN_COMMAND)
    replies.append(chain_reply)
    assert_reply_mentions(
        chain_reply,
        "幂等：mkdr / 101（草稿）→ mkdr / 100",
        "米等：mkdr / 100（词库）→ mkdr / 101",
        "依据：",
    )
    require(
        chain_reply.count(pending_confirmation_copy()) == 1
        and len(chain_reply.splitlines()) <= 8
        and sum(line.startswith("依据：") for line in chain_reply.splitlines()) == 1,
        f"S32 merged-view plan was not compact with one confirmation: {chain_reply}",
    )
    require(
        (await ctx.draft()).get("items") == original_draft.get("items"),
        "S32 merged-view plan wrote before confirmation",
    )
    chain_events = [
        event
        for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > chain_cutoff
        and event.get("kind") == "tool"
    ]
    require(
        any(event.get("name") == "keytao_lookup_by_code" for event in chain_events)
        and any(event.get("name") == "keytao_list_draft_items" for event in chain_events)
        and any(event.get("name") == "keytao_shift_phrase_code" for event in chain_events),
        f"S32 merged-view turn did not traverse both views and the shared planner: {chain_events}",
    )

    messages.append("取消")
    cancel_reply = await ctx.send("取消")
    replies.append(cancel_reply)
    require(
        "取消" in cancel_reply or "放弃" in cancel_reply,
        f"S32 could not cancel the first sealed plan: {cancel_reply}",
    )
    require(
        (await ctx.draft()).get("items") == original_draft.get("items"),
        "S32 cancellation changed the original draft fixture",
    )

    word_list_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append(S32_WORD_LIST_COMMAND)
    word_list_reply = await ctx.send(S32_WORD_LIST_COMMAND)
    replies.append(word_list_reply)
    assert_reply_mentions(
        word_list_reply,
        "幂等：mkdr / 101（草稿）→ mkdr / 100",
        "米等：mkdr / 100（词库）→",
        "与你列的不一致",
        "依据：",
    )
    word_list_lines = word_list_reply.splitlines()
    evidence_lines = [
        line for line in word_list_lines
        if line.startswith("依据：")
    ]
    require(
        "没有可用的服务端候选记录" not in word_list_reply
        and word_list_reply.count(pending_confirmation_copy()) == 1
        and len(word_list_lines) <= 8
        and len(evidence_lines) == 1
        and "迷瞪" in evidence_lines[0]
        and not any(
            line.startswith("• 迷瞪：") and "→" in line
            for line in word_list_lines
        )
        and "当前状态" not in word_list_reply
        and "建议状态" not in word_list_reply
        and "服务端校验：" not in word_list_reply,
        f"S32 word-list turn did not produce one compact executable plan: {word_list_reply}",
    )
    require(
        (await ctx.draft()).get("items") == original_draft.get("items"),
        "S32 word-list plan wrote before confirmation",
    )
    word_list_events = [
        event
        for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > word_list_cutoff
        and event.get("kind") == "tool"
    ]
    plan_event = next(
        (
            event
            for event in word_list_events
            if event.get("name") == "keytao_shift_phrase_code"
            and isinstance(event.get("result"), dict)
            and isinstance(event["result"].get("shiftPlan"), dict)
            and event["result"]["shiftPlan"].get("scope") == "prefix_chain"
        ),
        None,
    )
    require(plan_event is not None, f"S32 did not use the generalized prefix planner: {word_list_events}")
    shift_plan = plan_event["result"]["shiftPlan"]
    current_state = shift_plan.get("currentState") or []
    proposed_state = shift_plan.get("proposedState") or []
    require(
        {str(entry.get("word") or "") for entry in current_state}
        == {"米等", "幂等", "迷瞪"}
        and {str(entry.get("word") or "") for entry in proposed_state}
        == {"米等", "幂等", "迷瞪"}
        and next(
            entry for entry in current_state if entry.get("word") == "迷瞪"
        ).get("code") == S32_PREFIX_CODE,
        f"S32 prefix plan omitted an incident word: {shift_plan}",
    )
    proposed_by_word = {
        str(entry.get("word") or ""): str(entry.get("code") or "")
        for entry in proposed_state
        if isinstance(entry, dict)
    }
    require(
        proposed_by_word.get(S32_DRAFT_WORD) == S32_CODE
        and len(set(proposed_by_word.values())) == 3
        and all(code.startswith(S32_CODE) for code in proposed_by_word.values()),
        f"S32 proposed codes are not one ordered mkdr prefix chain: {proposed_by_word}",
    )
    require(
        shift_plan.get("items") == [
            {"action": "Delete", "word": "米等", "code": S32_CODE, "type": "Phrase"},
            {
                "action": "Create",
                "word": "米等",
                "code": proposed_by_word["米等"],
                "type": "Phrase",
                "weight": 100,
            },
        ]
        and len(shift_plan.get("draftUpdates") or []) == 1
        and (shift_plan.get("draftUpdates") or [])[0].get("word") == S32_DRAFT_WORD
        and (shift_plan.get("draftUpdates") or [])[0].get("fromWeight") == 101
        and (shift_plan.get("draftUpdates") or [])[0].get("toWeight") == 100,
        f"S32 did not seal the exact incident dictionary plus weight shape: {shift_plan}",
    )

    messages.append("确认")
    completion_reply = await ctx.send("确认")
    replies.append(completion_reply)
    assert_reply_mentions(
        completion_reply,
        "操作已完成",
        "米等",
        "幂等",
        "草稿地址：",
    )
    require(
        len(completion_reply.splitlines()) <= 3
        and "服务端校验：" not in completion_reply,
        f"S32 completion was not compact: {completion_reply}",
    )
    final_draft = await ctx.draft()
    proposed_mi_code = proposed_by_word["米等"]
    expected_items = {
        ("Create", "幂等", S32_CODE),
        ("Delete", "米等", S32_CODE),
        ("Create", "米等", proposed_mi_code),
    }
    actual_items = {item_key(item) for item in final_draft.get("items", [])}
    draft_idempotent = next(
        (item for item in final_draft.get("items", []) if item.get("word") == "幂等"),
        {},
    )
    require(
        actual_items == expected_items and draft_idempotent.get("weight") == 100,
        f"S32 confirmation did not materialize the sealed live+draft plan: {final_draft}",
    )
    confirmed_shift_calls = [
        event
        for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > word_list_cutoff
        and event.get("kind") == "tool"
        and event.get("name") == "keytao_shift_phrase_code"
        and event.get("arguments", {}).get("confirmed_plan_digest")
    ]
    require(
        len(confirmed_shift_calls) == 1,
        f"S32 expected one confirmed generalized-plan replay: {confirmed_shift_calls}",
    )
    confirmed_result = confirmed_shift_calls[0].get("result") or {}
    receipts = confirmed_result.get("receipts") or []
    require(
        confirmed_result.get("success") is True
        and confirmed_result.get("successCount") == 3
        and [receipt.get("step") for receipt in receipts]
        == ["dictionary", "draftWeight"]
        and all(receipt.get("status") in {"applied", "alreadyApplied"} for receipt in receipts),
        f"S32 one confirmation did not return complete step receipts: {confirmed_result}",
    )

    return {
        "messages": messages,
        "replies": replies,
        "draft": final_draft,
        "facts": {
            "mergedViewWords": ["米等", "幂等"],
            "wordListWords": ["米等", "幂等", "迷瞪"],
            "listedOrder": ["米等", "幂等", "迷瞪"],
            "proposedOrder": [entry.get("word") for entry in proposed_state],
            "proposedCodes": proposed_by_word,
            "chainPlanCancelled": True,
            "wordListConfirmationSteps": 1,
            "wordListPlanLines": len(word_list_reply.splitlines()),
            "completionLines": len(completion_reply.splitlines()),
            "receiptSteps": [receipt.get("step") for receipt in receipts],
        },
    }


S33_WORDS = ("洒漏", "撒漏")
S33_SIX_WORDS = ("洒漏", "洒溇")
S33_DISCOVERY = "喵喵 加词 洒漏 撒漏"
S33_EXTERNAL_WORDS = ("缩手", "所售")
S33_EXTERNAL_OCCUPANT = ("所受", "sled")
S33_EXTERNAL_QUERY = "缩手 所售"
S33_EXTERNAL_EXPECTED = (("缩手", "sleda"), ("所售", "sledu"))


def _s33_external_query_pairs(reply: str) -> tuple[tuple[str, str], ...]:
    """Read either the sealed batch layout or the rich read-only dual list."""
    batch_pairs = advertised_batch_binding_pairs(reply)
    if batch_pairs:
        return batch_pairs
    headings: list[tuple[str, int]] = []
    for word in S33_EXTERNAL_WORDS:
        heading = re.search(
            rf"(?m)^(?:\d+\.\s*)?(?:"
            rf"[「【]{re.escape(word)}[」】][^\r\n]{{0,80}}"
            rf"|{re.escape(word)}(?:（[^\r\n]{{0,80}}|\s*|的键道编码[^\r\n]{{0,80}})"
            r")$",
            reply,
        )
        if heading is None:
            return ()
        headings.append((word, heading.start()))
    if tuple(word for word, _start in sorted(headings, key=lambda item: item[1])) != (
        S33_EXTERNAL_WORDS
    ):
        return ()
    pairs: list[tuple[str, str]] = []
    for index, (word, start) in enumerate(headings):
        end = headings[index + 1][1] if index + 1 < len(headings) else len(reply)
        block = reply[start:end]
        expected_code = dict(S33_EXTERNAL_EXPECTED)[word]
        recommendation = re.search(
            rf"(?m)^推荐编码[：:]\s*{re.escape(expected_code)}\s*$",
            block,
        ) or re.search(
            rf"(?m)^\s*\d+\.\s*{re.escape(expected_code)}\s+—\s+"
            r".*空位.*(?:✅|推荐).*$",
            block,
        )
        if recommendation is None:
            return ()
        pairs.append((word, expected_code))
    return tuple(pairs)


async def scenario_s33(ctx: ScenarioContext) -> dict[str, Any]:
    """Pin batch-aware homophone allocation and the sink collision controls."""
    messages: list[str] = []
    replies: list[str] = []
    dictionary_cleanups: list[dict[str, Any]] = []
    external_replies: list[str] = []

    async def reset_words() -> None:
        dictionary_cleanup = await ctx.next_client.remove_rig_owned_dictionary_words(
            platform_id=ctx.platform_id,
            admin_token=ctx.admin_token,
            scenario_id="S33",
            fixture_words=(*S33_WORDS, "洒溇"),
        )
        require(
            dictionary_cleanup.get("verified") is True,
            f"S33 dictionary cleanup was not verified: {dictionary_cleanup}",
        )
        dictionary_cleanups.append(dictionary_cleanup)
        cleanup = await ctx.next_client.clean_draft(ctx.platform_id)
        require(cleanup.get("success") is True, f"S33 draft cleanup failed: {cleanup}")
        await ctx.bot.reset_conversation(platform_id=ctx.platform_id)

    async def submit_advertised(label: str, discovery: str) -> tuple[dict[str, Any], dict[str, str]]:
        displayed = dict(advertised_batch_binding_pairs(discovery))
        require(
            set(displayed) == set(S33_WORDS),
            f"S33 {label} discovery did not bind both homophones: {discovery}",
        )
        cutoff = max(
            (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
            default=0,
        )
        messages.append("加入并提交")
        reply = await ctx.send("加入并提交")
        replies.append(reply)
        batch_id = _successful_submit_batch_id(
            ctx.attempt_events(),
            after_sequence=cutoff,
        )
        if not batch_id:
            require(
                pending_confirmation_copy() in reply,
                f"S33 {label} assent neither submitted nor exposed one ticket: {reply}",
            )
            messages.append("确认")
            confirm_reply = await ctx.send("确认")
            replies.append(confirm_reply)
            batch_id = _successful_submit_batch_id(
                ctx.attempt_events(),
                after_sequence=cutoff,
            )
        require(batch_id, f"S33 {label} did not submit its advertised set")
        batch = await ctx.next_client.get_admin_batch(
            batch_id=batch_id,
            admin_token=ctx.admin_token,
        )
        require(
            batch.get("status") in {"Submitted", "Approved"},
            f"S33 {label} batch was not submitted: {batch}",
        )
        for word, code in displayed.items():
            require(
                _submitted_item(batch, word=word, code=code) is not None,
                f"S33 {label} batch lacks {word}@{code}: {batch}",
            )
        return batch, displayed

    external_fixture_words = (*S33_EXTERNAL_WORDS, S33_EXTERNAL_OCCUPANT[0])
    external_cleanup = await ctx.next_client.remove_rig_owned_dictionary_words(
        platform_id=ctx.platform_id,
        admin_token=ctx.admin_token,
        scenario_id="S33",
        fixture_words=external_fixture_words,
    )
    require(
        external_cleanup.get("verified") is True,
        f"S33 external-occupant cleanup was not verified: {external_cleanup}",
    )
    dictionary_cleanups.append(external_cleanup)
    external_seed = await ctx.next_client.seed_phrase(
        platform_id=ctx.platform_id,
        word=S33_EXTERNAL_OCCUPANT[0],
        code=S33_EXTERNAL_OCCUPANT[1],
    )
    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)
    external_reply = await ctx.send(S33_EXTERNAL_QUERY)
    external_replies.append(external_reply)
    external_pairs = _s33_external_query_pairs(external_reply)
    require(
        external_pairs == S33_EXTERNAL_EXPECTED,
        f"S33 external occupant was not allocated by priority: {external_reply}",
    )
    require(
        S33_EXTERNAL_OCCUPANT[0] in external_reply
        and S33_EXTERNAL_OCCUPANT[1] in external_reply,
        f"S33 external occupant was not shown: {external_reply}",
    )
    require(
        "无法唯一绑定" not in external_reply
        and "本次不会写入" not in external_reply
        and "查看草稿" not in external_reply,
        f"S33 pure query used write-flow failure copy: {external_reply}",
    )
    require(
        all(
            marker not in external_reply
            for marker in ("日常语感", "直觉比较", "频率可能", "可能略高")
        ),
        f"S33 pure query leaked speculative commonness copy: {external_reply}",
    )
    external_draft = await ctx.draft()
    require(
        not external_draft.get("items"),
        f"S33 pure query unexpectedly wrote a draft: {external_draft}",
    )
    external_post_cleanup = await ctx.next_client.remove_rig_owned_dictionary_words(
        platform_id=ctx.platform_id,
        admin_token=ctx.admin_token,
        scenario_id="S33",
        fixture_words=external_fixture_words,
    )
    require(
        external_post_cleanup.get("verified") is True,
        f"S33 external fixture post-cleanup was not verified: {external_post_cleanup}",
    )
    dictionary_cleanups.append(external_post_cleanup)

    await reset_words()
    messages.append(S33_DISCOVERY)
    discovery = await ctx.send(S33_DISCOVERY)
    replies.append(discovery)
    main_pairs = dict(advertised_batch_binding_pairs(discovery))
    require(
        set(main_pairs) == set(S33_WORDS)
        and len(set(main_pairs.values())) == 2,
        f"S33 candidate display repeated a short recommendation: {discovery}",
    )
    review_events = [
        event
        for event in ctx.attempt_events()
        if event.get("kind") == "tool"
        and event.get("name") == "keytao_prepare_reviewed_add"
        and str((event.get("result") or {}).get("word") or "") in S33_WORDS
    ]

    def reviewed_candidate_chain(event: dict[str, Any]) -> tuple[str, ...]:
        result = event.get("result") or {}
        direct = ordered_candidate_codes(result)
        if direct:
            return tuple(direct)
        pronunciations = result.get("pronunciations") or []
        recommended = str(result.get("recommendedCode") or "").lower()
        pronunciation = next(
            (
                item for item in pronunciations
                if isinstance(item, dict)
                and str(item.get("recommendedCode") or "").lower() == recommended
            ),
            pronunciations[0] if pronunciations else {},
        )
        return tuple(ordered_candidate_codes(pronunciation))

    review_chains = {
        str((event.get("result") or {}).get("word") or ""):
            reviewed_candidate_chain(event)
        for event in review_events
    }
    require(
        len(review_chains) == 2
        and all(review_chains.values())
        and len({chain[0] for chain in review_chains.values()}) == 1,
        f"S33 reviews did not expose one common candidate family: {review_events}",
    )
    candidate_family = next(iter(review_chains.values()))[0]
    main_batch, main_displayed = await submit_advertised("distinct", discovery)
    require(
        len(messages) == 2,
        f"S33 distinct reviewed set required an extra confirmation: {replies}",
    )

    await reset_words()
    explicit_request = (
        f"喵喵\n加词 {S33_WORDS[0]} {candidate_family}\n"
        f"加词 {S33_WORDS[1]} {candidate_family}\n同码"
    )
    explicit_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append(explicit_request)
    explicit_reply = await ctx.send(explicit_request)
    replies.append(explicit_reply)
    explicit_draft = await ctx.draft()
    if len(explicit_draft.get("items") or []) != 2:
        require(
            pending_confirmation_copy() in explicit_reply,
            f"S33 explicit same-code request neither wrote nor exposed one ticket: {explicit_reply}",
        )
        messages.append("确认")
        replies.append(await ctx.send("确认"))
        explicit_draft = await ctx.draft()
    explicit_items = [
        item_key(item) for item in explicit_draft.get("items") or []
    ]
    require(
        len(explicit_items) == 2
        and {
            (action, word, code) for action, word, code in explicit_items
        } == {
            ("Create", word, candidate_family) for word in S33_WORDS
        },
        f"S33 explicit same-code request did not preserve the duplicate: {explicit_draft}",
    )
    explicit_events = [
        event
        for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > explicit_cutoff
        and event.get("kind") == "tool"
        and event.get("name") == "keytao_batch_add_to_draft"
    ]
    require(
        explicit_events
        and all(
            (event.get("result") or {}).get("collisionBlocked") is not True
            and (event.get("result") or {}).get("collisionReplanned") is not True
            for event in explicit_events
        ),
        f"S33 explicit same-code opt-in did not pass the sink: {explicit_events}",
    )

    await reset_words()
    six_code = "ssldaa"
    six_command = (
        f"喵喵\n加词 {S33_SIX_WORDS[0]} {six_code}\n"
        f"加词 {S33_SIX_WORDS[1]} {six_code}"
    )
    six_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append(six_command)
    six_reply = await ctx.send(six_command)
    replies.append(six_reply)
    six_events = [
        event
        for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > six_cutoff
        and event.get("kind") == "tool"
        and event.get("name") == "keytao_batch_add_to_draft"
    ]
    require(
        six_events
        and all(
            (event.get("result") or {}).get("collisionBlocked") is not True
            and (event.get("result") or {}).get("collisionReplanned") is not True
            for event in six_events
        ),
        f"S33 six-code duplicate did not pass the sink gate: {six_events}",
    )

    await reset_words()
    short_code = candidate_family
    collision_command = (
        f"喵喵\n加词 {S33_WORDS[0]} {short_code}\n"
        f"加词 {S33_WORDS[1]} {short_code}"
    )
    collision_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append(collision_command)
    collision_reply = await ctx.send(collision_command)
    replies.append(collision_reply)
    assert_reply_mentions(collision_reply, "已调整：", f"{short_code}→")
    require(
        collision_reply.count(pending_confirmation_copy()) == 1
        and not (await ctx.draft()).get("items"),
        f"S33 sink replan did not pause on one changed-set confirmation: {collision_reply}",
    )
    collision_events = [
        event
        for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > collision_cutoff
        and event.get("kind") == "tool"
        and event.get("name") == "keytao_batch_add_to_draft"
    ]
    require(
        len(collision_events) == 1
        and (collision_events[0].get("result") or {}).get("collisionReplanned") is True,
        f"S33 model-composed collision did not replan exactly once at the sink: {collision_events}",
    )
    messages.append("取消")
    replies.append(await ctx.send("取消"))

    return {
        "messages": [S33_EXTERNAL_QUERY, *messages],
        "replies": [*external_replies, *replies],
        "draft": await ctx.draft(),
        "facts": {
            "words": list(S33_WORDS),
            "externalOccupant": {
                "words": list(S33_EXTERNAL_WORDS),
                "occupant": list(S33_EXTERNAL_OCCUPANT),
                "allocated": [list(pair) for pair in S33_EXTERNAL_EXPECTED],
                "seedBatchId": external_seed.get("batchId"),
                "readOnly": True,
            },
            "candidateChains": {
                word: list(chain) for word, chain in review_chains.items()
            },
            "candidateFamily": candidate_family,
            "priorityOrder": list(main_displayed),
            "distinctCodes": list(main_displayed.values()),
            "mainBatchId": main_batch.get("id"),
            "explicitSameCode": candidate_family,
            "sixCodeControl": six_code,
            "sinkReplanned": True,
            "dictionaryCleanups": dictionary_cleanups,
        },
    }


S34_WORD = "开团"
S34_PENDING_CODE = "khtt"


async def scenario_s34(ctx: ScenarioContext) -> dict[str, Any]:
    """Re-query one actor-owned Submitted item and gate exact duplicate assent."""
    messages: list[str] = []
    replies: list[str] = []

    await ctx.next_client.clean_draft(ctx.platform_id)
    submitted_cleanup = await ctx.next_client.clean_submitted_batches(
        ctx.platform_id
    )
    dictionary_cleanup = await ctx.next_client.remove_rig_owned_dictionary_words(
        platform_id=ctx.platform_id,
        admin_token=ctx.admin_token,
        scenario_id="S34",
        fixture_words=(S34_WORD,),
    )
    require(
        dictionary_cleanup.get("verified") is True,
        f"S34 dictionary cleanup was not verified: {dictionary_cleanup}",
    )
    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)

    added = await ctx.next_client.add_draft_items(
        platform_id=ctx.platform_id,
        items=[{
            "action": "Create",
            "word": S34_WORD,
            "code": S34_PENDING_CODE,
            "type": "Phrase",
            "needsManualReview": True,
            "remark": "S34 pending-batch awareness fixture",
        }],
    )
    draft_before_submit = await ctx.draft()
    batch_id = str(draft_before_submit.get("batchId") or "").strip()
    content_version = draft_before_submit.get("contentVersion")
    require(
        added.get("success") is True
        and batch_id
        and isinstance(content_version, int),
        f"S34 could not create its pending fixture: {added}; {draft_before_submit}",
    )
    submitted = await ctx.next_client.submit_batch(
        platform_id=ctx.platform_id,
        batch_id=batch_id,
        content_version=content_version,
    )
    require(
        ((submitted.get("submitted") or {}).get("batch") or {}).get("status")
        == "Submitted",
        f"S34 fixture did not stay Submitted: {submitted}",
    )

    public_batch_url = (
        f"{public_base_for_platform('qq').rstrip('/')}/batch/{batch_id}"
    )
    reminder = (
        f"「{S34_WORD}」已在待审核批次中"
        f"（→ {S34_PENDING_CODE}，审核中）：{public_batch_url}"
    )

    query_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append(f"喵喵 {S34_WORD}")
    query_reply = await ctx.send(messages[-1])
    replies.append(query_reply)
    require(
        query_reply.startswith(reminder),
        f"S34 repeated query did not lead with the pending fact: {query_reply}",
    )
    require(
        "其他编码" in query_reply
        and "撤回" in query_reply
        and pending_confirmation_copy() not in query_reply,
        f"S34 repeated query restarted or omitted pending actions: {query_reply}",
    )
    require(
        not any(
            int(event.get("sequence") or 0) > query_cutoff
            and event.get("kind") == "tool"
            and event.get("name") == "keytao_prepare_reviewed_add"
            for event in ctx.attempt_events()
        ),
        "S34 repeated query restarted the reviewed-add ritual",
    )

    messages.append(f"喵喵 加词 {S34_WORD}")
    discovery = await ctx.send(messages[-1])
    replies.append(discovery)
    require(
        discovery.startswith(reminder) and S34_PENDING_CODE in discovery,
        f"S34 explicit add omitted the pending lead or reviewed code: {discovery}",
    )

    duplicate_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append("加入")
    duplicate_reply = await ctx.send(messages[-1])
    replies.append(duplicate_reply)
    require(
        duplicate_reply.startswith(reminder)
        and "该词已在审核中，确认再提交一条相同词条吗？" in duplicate_reply
        and duplicate_reply.count(pending_confirmation_copy()) == 1,
        f"S34 exact duplicate assent did not warn once: {duplicate_reply}",
    )
    duplicate_events = [
        event
        for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > duplicate_cutoff
        and event.get("kind") == "tool"
        and event.get("name") == "keytao_create_phrase"
    ]
    require(
        len(duplicate_events) == 1
        and (duplicate_events[0].get("result") or {}).get(
            "pendingDuplicateConfirmation"
        ) is True
        and (duplicate_events[0].get("result") or {}).get("success") is not True,
        f"S34 exact duplicate reached a write or missed the sink gate: {duplicate_events}",
    )
    messages.append("取消")
    replies.append(await ctx.send(messages[-1]))

    messages.append(f"喵喵 加词 {S34_WORD}")
    different_discovery = await ctx.send(messages[-1])
    replies.append(different_discovery)
    candidate_rows = [
        (int(index), code.lower())
        for index, code in re.findall(
            r"(?m)^\s*(\d+)[.、]\s*([a-z]{1,6})\b",
            different_discovery,
        )
    ]
    different_row = next(
        ((index, code) for index, code in candidate_rows
         if code != S34_PENDING_CODE),
        None,
    )
    require(
        different_discovery.startswith(reminder) and different_row is not None,
        f"S34 did not expose a different reviewed code with the reminder: {different_discovery}",
    )
    different_index, different_code = different_row
    different_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append(str(different_index))
    different_reply = await ctx.send(messages[-1])
    replies.append(different_reply)
    require(
        different_reply.startswith(reminder),
        f"S34 different-code add did not keep the pending lead: {different_reply}",
    )
    current_draft = await ctx.draft()
    if not any(
        item_key(item) == ("Create", S34_WORD, different_code)
        for item in current_draft.get("items", [])
        if isinstance(item, dict)
    ):
        require(
            pending_confirmation_copy() in different_reply,
            f"S34 different-code add neither wrote nor returned its normal confirmation: {different_reply}",
        )
        messages.append("确认")
        confirmation_reply = await ctx.send(messages[-1])
        replies.append(confirmation_reply)
        require(
            confirmation_reply.startswith(reminder),
            f"S34 different-code confirmation dropped the pending lead: {confirmation_reply}",
        )
        current_draft = await ctx.draft()
    require(
        any(
            item_key(item) == ("Create", S34_WORD, different_code)
            for item in current_draft.get("items", [])
            if isinstance(item, dict)
        ),
        f"S34 different-code add did not proceed: {current_draft}",
    )
    require(
        any(
            int(event.get("sequence") or 0) > different_cutoff
            and event.get("kind") == "tool"
            and event.get("name") == "keytao_create_phrase"
            and (event.get("result") or {}).get("success") is True
            for event in ctx.attempt_events()
        ),
        "S34 different-code path has no successful sink receipt",
    )

    return {
        "messages": messages,
        "replies": replies,
        "draft": current_draft,
        "facts": {
            "word": S34_WORD,
            "pendingCode": S34_PENDING_CODE,
            "differentCode": different_code,
            "pendingBatchId": batch_id,
            "reminder": reminder,
            "submittedCleanup": submitted_cleanup,
            "dictionaryCleanup": dictionary_cleanup,
            "queryRestartedReview": False,
            "exactDuplicateBlocked": True,
            "differentCodeProceeded": True,
        },
    }


S35_FRONT_CASES = (
    ("发布会", "重病号", "fbh"),
    ("计算机", "建三江", "jsj"),
)
S35_FREE_CONTROL = ("无事忙", "wem")


async def scenario_s35(ctx: ScenarioContext) -> dict[str, Any]:
    """Pin comparator-driven copy, default front insert, and both controls."""
    fixture = ctx.fixture_facts["s35"]
    messages: list[str] = []
    replies: list[str] = []

    default_case = fixture["frontCases"][0]
    default_word = str(default_case["newcomerWord"])
    default_occupant = str(default_case["occupantWord"])
    default_code = str(default_case["occupiedCode"])
    default_free = str(default_case["freeCode"])
    default_shifted = str(default_case["shiftedCode"])
    messages.append(f"喵喵 {default_word}")
    recommendation = await ctx.send(messages[-1])
    replies.append(recommendation)
    recommendation_lines = [
        line
        for line in recommendation.splitlines()
        if line.startswith("推荐：") or line.startswith("不重排选 ")
    ]
    require(
        len(recommendation_lines) == 2
        and recommendation_lines[0] == "推荐："
        and recommendation_lines[1] == f"不重排选 2（{default_free}）。"
        and (
            f'- “「{default_word}」占 {default_code}、'
            f'「{default_occupant}」顺延”'
        ) in recommendation
        and "依据：" in recommendation
        and "推荐编码：" not in recommendation
        and "当前建议不调整现有排序" not in recommendation
        and f"{default_word} → {default_free}（推荐）" not in recommendation,
        f"S35 default recommendation was contradictory or longer than two lines: "
        f"{recommendation}",
    )
    require(
        not (await ctx.draft()).get("items"),
        f"S35 recommendation wrote before assent: {await ctx.draft()}",
    )

    submit_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append("加入并提交")
    assent_reply = await ctx.send(messages[-1])
    replies.append(assent_reply)
    confirmation_steps = 0
    default_batch_id = _successful_submit_batch_id(
        ctx.attempt_events(),
        after_sequence=submit_cutoff,
    )
    if not default_batch_id:
        require(
            pending_confirmation_copy() in assent_reply,
            f"S35 default assent neither submitted nor exposed the sealed plan: {assent_reply}",
        )
        messages.append("确认")
        replies.append(await ctx.send(messages[-1]))
        confirmation_steps = 1
        default_batch_id = _successful_submit_batch_id(
            ctx.attempt_events(),
            after_sequence=submit_cutoff,
        )
    require(default_batch_id, "S35 one front-insert confirmation did not submit")
    default_batch = await ctx.next_client.get_admin_batch(
        batch_id=default_batch_id,
        admin_token=ctx.admin_token,
    )
    default_items = [
        item_key(item)
        for item in default_batch.get("pullRequests", [])
        if isinstance(item, dict)
    ]
    expected_default_items = (
        ("Delete", default_occupant, default_code),
        ("Create", default_occupant, default_shifted),
        ("Create", default_word, default_code),
    )
    newcomer_item = next(
        (
            item
            for item in default_batch.get("pullRequests", [])
            if isinstance(item, dict)
            and item_key(item) == ("Create", default_word, default_code)
        ),
        None,
    )
    require(
        default_batch.get("status") in {"Submitted", "Approved"}
        and same_unique_item_set(default_items, expected_default_items)
        and isinstance(newcomer_item, dict)
        and newcomer_item.get("needsManualReview") is True,
        f"S35 default assent did not submit the sealed front insert: {default_batch}",
    )
    confirmed_shift_calls = [
        event
        for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > submit_cutoff
        and event.get("kind") == "tool"
        and event.get("name") == "keytao_shift_phrase_code"
        and event.get("arguments", {}).get("confirmed_plan_digest")
    ]
    require(
        confirmation_steps <= 1 and len(confirmed_shift_calls) == 1,
        f"S35 default reorder did not preserve one-confirmation semantics: "
        f"steps={confirmation_steps}, calls={confirmed_shift_calls}",
    )

    await ctx.next_client.clean_draft(ctx.platform_id)
    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)
    opt_out_case = fixture["frontCases"][1]
    opt_out_word = str(opt_out_case["newcomerWord"])
    opt_out_occupant = str(opt_out_case["occupantWord"])
    opt_out_code = str(opt_out_case["occupiedCode"])
    opt_out_free = str(opt_out_case["freeCode"])
    messages.append(f"喵喵 {opt_out_word}")
    opt_out_discovery = await ctx.send(messages[-1])
    replies.append(opt_out_discovery)
    require(
        f"「{opt_out_word}」占 {opt_out_code}、「{opt_out_occupant}」顺延"
        in opt_out_discovery
        and "推荐：\n- " in opt_out_discovery
        and f"不重排选 2（{opt_out_free}）。" in opt_out_discovery,
        f"S35 opt-out discovery lacked the bound fallback: {opt_out_discovery}",
    )
    opt_out_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append("2 添加并提交")
    opt_out_reply = await ctx.send(messages[-1])
    replies.append(opt_out_reply)
    opt_out_confirmations = 0
    opt_out_batch_id = _successful_submit_batch_id(
        ctx.attempt_events(),
        after_sequence=opt_out_cutoff,
    )
    if not opt_out_batch_id and pending_confirmation_copy() in opt_out_reply:
        messages.append("确认")
        replies.append(await ctx.send(messages[-1]))
        opt_out_confirmations = 1
        opt_out_batch_id = _successful_submit_batch_id(
            ctx.attempt_events(),
            after_sequence=opt_out_cutoff,
        )
    require(opt_out_batch_id, "S35 numbered opt-out never submitted")
    opt_out_batch = await ctx.next_client.get_admin_batch(
        batch_id=opt_out_batch_id,
        admin_token=ctx.admin_token,
    )
    require(
        opt_out_batch.get("status") in {"Submitted", "Approved"}
        and [
            item_key(item)
            for item in opt_out_batch.get("pullRequests", [])
            if isinstance(item, dict)
        ]
        == [("Create", opt_out_word, opt_out_free)],
        f"S35 numbered opt-out did not land on the free slot: {opt_out_batch}",
    )

    await ctx.next_client.clean_draft(ctx.platform_id)
    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)
    free_control = fixture["freeControl"]
    free_word = str(free_control["word"])
    free_code = str(free_control["recommendedCode"])
    messages.append(f"喵喵 {free_word}")
    free_discovery = await ctx.send(messages[-1])
    replies.append(free_discovery)
    require(
        free_word in free_discovery
        and free_code in free_discovery
        and "不重排选" not in free_discovery,
        f"S35 no-recommendation control did not advertise the free default: "
        f"{free_discovery}",
    )
    free_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append("加入并提交")
    free_reply = await ctx.send(messages[-1])
    replies.append(free_reply)
    free_confirmations = 0
    free_batch_id = _successful_submit_batch_id(
        ctx.attempt_events(),
        after_sequence=free_cutoff,
    )
    if not free_batch_id and pending_confirmation_copy() in free_reply:
        messages.append("确认")
        replies.append(await ctx.send(messages[-1]))
        free_confirmations = 1
        free_batch_id = _successful_submit_batch_id(
            ctx.attempt_events(),
            after_sequence=free_cutoff,
        )
    require(free_batch_id, "S35 no-recommendation assent never submitted")
    free_batch = await ctx.next_client.get_admin_batch(
        batch_id=free_batch_id,
        admin_token=ctx.admin_token,
    )
    require(
        free_batch.get("status") in {"Submitted", "Approved"}
        and [
            item_key(item)
            for item in free_batch.get("pullRequests", [])
            if isinstance(item, dict)
        ]
        == [("Create", free_word, free_code)],
        f"S35 no-recommendation assent changed the current free-slot default: {free_batch}",
    )

    final_draft = await ctx.draft()

    return {
        "messages": messages,
        "replies": replies,
        "draft": final_draft,
        "facts": {
            "defaultRecommendation": {
                "newcomer": default_word,
                "occupiedCode": default_code,
                "shiftedOccupant": default_occupant,
                "shiftedToCode": default_shifted,
                "batchId": default_batch_id,
                "batchStatus": default_batch.get("status"),
                "confirmationSteps": confirmation_steps,
                "sealedCreate": True,
            },
            "numberedOptOut": {
                "newcomer": opt_out_word,
                "selectedNumber": 2,
                "code": opt_out_free,
                "batchId": opt_out_batch_id,
                "batchStatus": opt_out_batch.get("status"),
                "confirmationSteps": opt_out_confirmations,
            },
            "noRecommendationControl": {
                "word": free_word,
                "code": free_code,
                "batchId": free_batch_id,
                "batchStatus": free_batch.get("status"),
                "confirmationSteps": free_confirmations,
            },
        },
    }


def _s36_assert_no_internal_reply_fragments(replies: list[str]) -> None:
    leaked = [reply for reply in replies if _reply_has_internal_fragment(reply)]
    require(not leaked, f"S36 exposed internal model/tool fragments: {leaked}")


async def scenario_s36(ctx: ScenarioContext) -> dict[str, Any]:
    """Delete, exact move/swap, named follow-up, and leak-guard incident round."""
    fixture = ctx.fixture_facts["s36"]
    messages: list[str] = []
    replies: list[str] = []

    delete_word = str(fixture["delete"]["word"])
    delete_code = str(fixture["delete"]["code"])
    delete_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append(f"删词 {delete_word}")
    delete_prompt = await ctx.send(messages[-1])
    replies.append(delete_prompt)
    require(
        pending_confirmation_copy() in delete_prompt
        and delete_word in delete_prompt
        and delete_code in delete_prompt
        and not (await ctx.draft()).get("items"),
        f"S36 dictionary delete was not one locked preview: {delete_prompt}",
    )
    messages.append("确认")
    replies.append(await ctx.send(messages[-1]))
    delete_draft = await ctx.draft()
    require(
        [item_key(item) for item in delete_draft.get("items", [])]
        == [("Delete", delete_word, delete_code)],
        f"S36 delete confirmation did not stage the exact Delete: {delete_draft}",
    )
    messages.append("提交")
    submit_reply = await ctx.send(messages[-1])
    replies.append(submit_reply)
    delete_batch_id = _successful_submit_batch_id(
        ctx.attempt_events(), after_sequence=delete_cutoff,
    )
    if not delete_batch_id:
        require(
            pending_confirmation_copy() in submit_reply,
            f"S36 delete submit neither completed nor requested confirmation: {submit_reply}",
        )
        messages.append("确认")
        replies.append(await ctx.send(messages[-1]))
        delete_batch_id = _successful_submit_batch_id(
            ctx.attempt_events(), after_sequence=delete_cutoff,
        )
    require(delete_batch_id, "S36 delete flow never submitted")
    delete_batch = await ctx.next_client.get_admin_batch(
        batch_id=delete_batch_id,
        admin_token=ctx.admin_token,
    )
    require(
        delete_batch.get("status") in {"Submitted", "Approved"}
        and [
            item_key(item)
            for item in delete_batch.get("pullRequests", [])
            if isinstance(item, dict)
        ] == [("Delete", delete_word, delete_code)],
        f"S36 submitted delete batch drifted: {delete_batch}",
    )

    await ctx.next_client.clean_draft(ctx.platform_id)
    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)
    messages.append("把 jmtdu 的箭头删掉")
    qualified_prompt = await ctx.send(messages[-1])
    replies.append(qualified_prompt)
    require(
        pending_confirmation_copy() in qualified_prompt
        and "箭头" in qualified_prompt
        and "jmtdu" in qualified_prompt
        and not (await ctx.draft()).get("items"),
        f"S36 code-qualified delete was not bound: {qualified_prompt}",
    )
    messages.append("取消")
    replies.append(await ctx.send(messages[-1]))

    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)
    move_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append("把「箭头」换到jmtd，把「剪贴」换到jmtdoa")
    move_prompt = await ctx.send(messages[-1])
    replies.append(move_prompt)
    require(
        pending_confirmation_copy() in move_prompt
        and all(marker in move_prompt for marker in ("箭头", "剪贴", "jmtd", "jmtdoa"))
        and not (await ctx.draft()).get("items"),
        f"S36 two-move request was not one complete plan: {move_prompt}",
    )
    messages.append("确认")
    replies.append(await ctx.send(messages[-1]))
    move_draft = await ctx.draft()
    expected_move_items = {
        ("Delete", "箭头", "jmtdu"),
        ("Create", "箭头", "jmtd"),
        ("Delete", "剪贴", "jmtd"),
        ("Create", "剪贴", "jmtdoa"),
    }
    require(
        {item_key(item) for item in move_draft.get("items", [])}
        == expected_move_items,
        f"S36 two-move plan did not land atomically in one draft: {move_draft}",
    )
    confirmed_move_calls = [
        event for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > move_cutoff
        and event.get("kind") == "tool"
        and event.get("name") == "keytao_shift_phrase_code"
        and event.get("arguments", {}).get("confirmed_plan_digest")
    ]
    require(
        len(confirmed_move_calls) == 1,
        f"S36 two moves did not execute as one confirmed plan: {confirmed_move_calls}",
    )

    await ctx.next_client.clean_draft(ctx.platform_id)
    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)
    priority_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append("把火锅和电脑调换优先级")
    priority_prompt = await ctx.send(messages[-1])
    replies.append(priority_prompt)
    require(
        pending_confirmation_copy() in priority_prompt
        and all(marker in priority_prompt for marker in ("火锅", "电脑", "100", "101")),
        f"S36 priority swap did not render the exact ring plan: {priority_prompt}",
    )
    messages.append("确认")
    replies.append(await ctx.send(messages[-1]))
    priority_draft = await ctx.draft()
    priority_items = {
        (str(item.get("word") or ""), item.get("weight"))
        for item in priority_draft.get("items", [])
        if item_key(item)[0] == "Change"
        and item_key(item)[2] == "mkdr"
    }
    require(
        priority_items == {("火锅", 101), ("电脑", 100)},
        f"S36 priority swap did not exchange exact weights: {priority_draft}",
    )
    confirmed_priority_calls = [
        event for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > priority_cutoff
        and event.get("kind") == "tool"
        and event.get("name") == "keytao_shift_phrase_code"
        and event.get("arguments", {}).get("confirmed_plan_digest")
    ]
    require(
        len(confirmed_priority_calls) == 1,
        f"S36 priority swap used more than one confirmed plan: {confirmed_priority_calls}",
    )

    await ctx.next_client.clean_draft(ctx.platform_id)
    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)
    messages.append("查询词库里的箭头，不要执行写入")
    lookup_reply = await ctx.send(messages[-1])
    replies.append(lookup_reply)
    require(
        "箭头" in lookup_reply
        and "jmtdu" in lookup_reply
        and not (await ctx.draft()).get("items"),
        f"S36 lookup turn did not expose the trusted dictionary row: {lookup_reply}",
    )
    messages.append("换码")
    named_action_prompt = await ctx.send(messages[-1])
    replies.append(named_action_prompt)
    require(
        pending_confirmation_copy() in named_action_prompt
        and "箭头" in named_action_prompt
        and "jmtdu" in named_action_prompt
        and not (await ctx.draft()).get("items"),
        f"S36 bare advertised action fell through to word query: {named_action_prompt}",
    )
    messages.append("取消")
    replies.append(await ctx.send(messages[-1]))

    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)
    no_records_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append("换码")
    plain_query = await ctx.send(messages[-1])
    replies.append(plain_query)
    no_records_writes = [
        event for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > no_records_cutoff
        and event.get("kind") == "tool"
        and event.get("name") in {
            "keytao_batch_add_to_draft",
            "keytao_create_phrase",
            "keytao_shift_phrase_code",
        }
    ]
    require(
        "换码" in plain_query
        and pending_confirmation_copy() not in plain_query
        and not no_records_writes
        and not (await ctx.draft()).get("items"),
        f"S36 recordless 换码 did not stay a word query: {plain_query}; {no_records_writes}",
    )
    _s36_assert_no_internal_reply_fragments(replies)
    return {
        "messages": messages,
        "replies": replies,
        "draft": priority_draft,
        "facts": {
            "deleteBatchId": delete_batch_id,
            "deleteItem": ["Delete", delete_word, delete_code],
            "qualifiedDeleteBound": True,
            "twoMoveItems": sorted(expected_move_items),
            "twoMoveConfirmedPlans": len(confirmed_move_calls),
            "namedActionFollowUp": True,
            "bareQueryControl": True,
            "priorityWeights": sorted(priority_items),
            "priorityConfirmedPlans": len(confirmed_priority_calls),
            "leakGuard": True,
        },
    }


S38_EXPLICIT_READING_MESSAGE = "加词 出圈，读音是 chū quān"
S38_EXPLANATION_MESSAGE = "耙耙柑为pá pá gān，因此这三个字的声母分别为p, p, g"
S38_NEGATIVE_MODIFIER_MESSAGE = "加词 耙耙柑 ppg，不要顺延其他相关的词条"
S38_POSITIVE_MODIFIER_MESSAGE = "加词 耙耙柑 ppg，顺延其他词条"
S38_QUERY_CONTROLS = ("1", "回复1", "加入")


async def scenario_s38(ctx: ScenarioContext) -> dict[str, Any]:
    """Close explicit-reading, query recovery, modifier, and suggestion incidents."""
    messages: list[str] = []
    replies: list[str] = []

    async def clean_and_reset(label: str) -> None:
        cleaned = await ctx.next_client.clean_draft(ctx.platform_id)
        require(cleaned.get("success") is True, f"S38 {label} cleanup failed: {cleaned}")
        await ctx.bot.reset_conversation(platform_id=ctx.platform_id)

    async def finish_one_control(reply: str, *, label: str) -> tuple[str, dict[str, Any]]:
        current = reply
        draft = await ctx.draft()
        if not draft.get("items") and pending_confirmation_copy() in current:
            messages.append("确认")
            current = await ctx.send_group(messages[-1], to_me=True)
            replies.append(current)
            draft = await ctx.draft()
        require(bool(draft.get("items")), f"S38 {label} did not reach a draft: {current}")
        return current, draft

    await clean_and_reset("explicit reading")
    reading_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append(S38_EXPLICIT_READING_MESSAGE)
    reading_reply = await ctx.send_group(messages[-1], to_me=True)
    replies.append(reading_reply)
    require(
        "出圈" in reading_reply
        and "候选" in reading_reply
        and "请明确要采用的读音或具体含义" not in reading_reply
        and "这条单词候选列表无法唯一绑定到本轮服务端编码记录" not in reading_reply,
        f"S38 explicit reading did not produce one candidate group: {reading_reply}",
    )
    reading_review_calls = [
        event for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > reading_cutoff
        and event.get("kind") == "tool"
        and event.get("name") == "keytao_prepare_reviewed_add"
        and event.get("arguments", {}).get("word") == "出圈"
        and event.get("arguments", {}).get("requested_reading") == "chu quan"
    ]
    require(
        bool(reading_review_calls),
        f"S38 explicit reading was not supplied to server review: {ctx.attempt_events()}",
    )
    messages.append("加入")
    add_reply = await ctx.send_group(messages[-1], to_me=True)
    replies.append(add_reply)
    _completion, reading_draft = await finish_one_control(
        add_reply,
        label="explicit-reading add",
    )
    require(
        any(item_key(item)[0:2] == ("Create", "出圈") for item in reading_draft.get("items", [])),
        f"S38 explicit-reading add did not land 出圈: {reading_draft}",
    )

    await clean_and_reset("explanation")
    messages.append(S38_EXPLANATION_MESSAGE)
    explanation_reply = await ctx.send_group(messages[-1], to_me=True)
    replies.append(explanation_reply)
    require(
        S37_WORD in explanation_reply
        and S37_TARGET_CODE in explanation_reply
        and "连续两次没有生成可见回复或工具调用" not in explanation_reply,
        f"S38 explanatory reading burned the reasoning budget: {explanation_reply}",
    )
    messages.append("取消")
    replies.append(await ctx.send_group(messages[-1], to_me=True))

    recovered_controls: list[str] = []
    for control in S38_QUERY_CONTROLS:
        await clean_and_reset(f"query recovery {control}")
        messages.append(f"喵喵 {S37_WORD}")
        query_reply = await ctx.send_group(messages[-1], to_me=True)
        replies.append(query_reply)
        require(
            S37_WORD in query_reply
            and S37_TARGET_CODE in query_reply
            and "本次仅查询" in query_reply,
            f"S38 query did not render a read-only candidate record: {query_reply}",
        )
        messages.append(control)
        control_reply = await ctx.send_group(messages[-1], to_me=True)
        replies.append(control_reply)
        control_reply, control_draft = await finish_one_control(
            control_reply,
            label=f"query recovery {control}",
        )
        require(
            any(item_key(item)[1] == S37_WORD for item in control_draft.get("items", []))
            and "这条消息没有可用的服务端候选记录" not in control_reply
            and "引用候选没有保留编码" not in control_reply
            and "没有执行添加" not in control_reply,
            f"S38 query control {control} dead-ended: {control_reply}; {control_draft}",
        )
        recovered_controls.append(control)

    await clean_and_reset("negative modifier")
    messages.append(S38_NEGATIVE_MODIFIER_MESSAGE)
    negative_reply = await ctx.send_group(messages[-1], to_me=True)
    replies.append(negative_reply)
    negative_reply, negative_draft = await finish_one_control(
        negative_reply,
        label="negative modifier",
    )
    negative_items = [item_key(item) for item in negative_draft.get("items", [])]
    require(
        negative_items == [("Create", S37_WORD, S37_TARGET_CODE)]
        and "执行动词" not in negative_reply,
        f"S38 negative modifier shifted or lost the add verb: {negative_reply}; {negative_draft}",
    )

    await clean_and_reset("positive modifier")
    positive_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append(S38_POSITIVE_MODIFIER_MESSAGE)
    positive_reply = await ctx.send_group(messages[-1], to_me=True)
    replies.append(positive_reply)
    require(
        "执行动词" not in positive_reply
        and "缺少明确" not in positive_reply
        and "连续两次没有生成可见回复或工具调用" not in positive_reply
        and "调整计划" in positive_reply
        and f"{S37_WORD}：Create {S37_TARGET_CODE}" in positive_reply
        and f"{S37_OCCUPANT}：Delete {S37_TARGET_CODE}" in positive_reply,
        f"S38 positive modifier lost its explicit add verb: {positive_reply}",
    )
    advertised = re.search(
        rf"顺延「{re.escape(S37_WORD)}」到\s*{re.escape(S37_TARGET_CODE)}",
        positive_reply,
    )
    if advertised is not None:
        messages.append(advertised.group(0))
        replay = await ctx.send_group(messages[-1], to_me=True)
        replies.append(replay)
        require(
            "不是「耙耙柑」的有效候选编码" not in replay
            and "不在它的候选链里" not in replay,
            f"S38 advertised shift failed its own validator: {replay}",
        )
    shift_calls = [
        event for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > positive_cutoff
        and event.get("kind") == "tool"
        and event.get("name") == "keytao_shift_phrase_code"
        and event.get("arguments", {}).get("word") == S37_WORD
        and event.get("arguments", {}).get("target_code") == S37_TARGET_CODE
    ]
    require(
        bool(shift_calls),
        f"S38 positive modifier did not reach its same-record shift validator: {positive_reply}",
    )
    await clean_and_reset("final")
    return {
        "messages": messages,
        "replies": replies,
        "draft": negative_draft,
        "facts": {
            "explicitReadingMessage": S38_EXPLICIT_READING_MESSAGE,
            "explanationMessage": S38_EXPLANATION_MESSAGE,
            "readingReviewBound": True,
            "completedExplicitReadingAdd": True,
            "recoveredQueryControls": recovered_controls,
            "negativeModifierDuplicate": negative_items,
            "advertisedShiftCount": 1 if advertised is not None else 0,
            "advertisedShiftValidated": advertised is None or bool(shift_calls),
        },
    }


S39_WORD = "出圈"
S39_OCCUPANT = "除权"
S39_TARGET_CODE = "jjqt"
S39_COMMAND = "加词 出圈 圈字读quan"
S39_SELECTION = "1 重新编码"
S39_OCCUPANT_COMMAND = '重新编码 "除权" jjqt'


async def scenario_s39(ctx: ScenarioContext) -> dict[str, Any]:
    """Collapse reading selection and occupant eviction into two user turns."""
    messages: list[str] = []
    replies: list[str] = []
    fixture = ctx.fixture_facts["s39"]
    shifted_code = str(fixture.get("shiftedCode") or "").strip()
    require(
        shifted_code and shifted_code != S39_TARGET_CODE,
        f"S39 fixture omitted the occupant's next free slot: {fixture}",
    )

    async def clean_and_reset(label: str) -> None:
        cleaned = await ctx.next_client.clean_draft(ctx.platform_id)
        require(cleaned.get("success") is True, f"S39 {label} cleanup failed: {cleaned}")
        await ctx.bot.reset_conversation(platform_id=ctx.platform_id)

    async def require_live_occupant(label: str) -> None:
        rows = [
            row
            for row in await ctx.next_client.phrases_by_word(S39_OCCUPANT)
            if row.get("word") == S39_OCCUPANT
            and row.get("code") == S39_TARGET_CODE
        ]
        require(
            len(rows) == 1,
            f"S39 {label} requires {S39_OCCUPANT}@{S39_TARGET_CODE}: {rows}",
        )

    await clean_and_reset("happy path")
    await require_live_occupant("happy path")
    happy_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append(S39_COMMAND)
    reading_reply = await ctx.send_group(messages[-1], to_me=True)
    replies.append(reading_reply)
    require(
        S39_WORD in reading_reply
        and "chū quān" in reading_reply
        and "chū juàn" not in reading_reply
        and re.search(
            rf"(?m)^1\.\s*{S39_TARGET_CODE}\s*—\s*已有「{S39_OCCUPANT}」",
            reading_reply,
        )
        and re.search(r"(?m)^2\.\s*jjqta\s*—\s*.*空位", reading_reply)
        and re.search(r"(?m)^3\.\s*jjqtai\s*—\s*.*空位", reading_reply)
        and "管理员审核" in reading_reply
        and "回复编号或编码选择" in reading_reply
        and S39_SELECTION in reading_reply
        and "复算" not in reading_reply
        and "网页端人工处理" not in reading_reply,
        f"S39 first turn did not render the selected reading group: {reading_reply}",
    )
    review_calls = [
        event
        for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > happy_cutoff
        and event.get("kind") == "tool"
        and event.get("name") == "keytao_prepare_reviewed_add"
        and event.get("arguments", {}).get("word") == S39_WORD
        and event.get("arguments", {}).get("requested_reading") == "圈=quan"
    ]
    require(
        len(review_calls) == 1,
        f"S39 reading selection did not use one reviewed-add request: {review_calls}",
    )

    messages.append(S39_SELECTION)
    selection_reply = await ctx.send_group(messages[-1], to_me=True)
    replies.append(selection_reply)
    happy_draft = await ctx.draft()
    expected_items = {
        ("Delete", S39_OCCUPANT, S39_TARGET_CODE),
        ("Create", S39_WORD, S39_TARGET_CODE),
        ("Create", S39_OCCUPANT, shifted_code),
    }
    actual_items = {item_key(item) for item in happy_draft.get("items", [])}
    newcomer_item = next(
        (
            item
            for item in happy_draft.get("items", [])
            if item_key(item) == ("Create", S39_WORD, S39_TARGET_CODE)
        ),
        None,
    )
    confirmed_shift_calls = [
        event
        for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > happy_cutoff
        and event.get("kind") == "tool"
        and event.get("name") == "keytao_shift_phrase_code"
        and event.get("arguments", {}).get("word") == S39_WORD
        and event.get("arguments", {}).get("target_code") == S39_TARGET_CODE
        and event.get("arguments", {}).get("confirmed_plan_digest")
    ]
    require(
        actual_items == expected_items
        and len(happy_draft.get("items", [])) == 3
        and isinstance(newcomer_item, dict)
        and newcomer_item.get("needsManualReview") is True
        and len(confirmed_shift_calls) == 1
        and pending_confirmation_copy() not in selection_reply,
        f"S39 second turn did not finish the sealed eviction: "
        f"reply={selection_reply}; draft={happy_draft}; calls={confirmed_shift_calls}",
    )

    await clean_and_reset("unmatched reading")
    messages.append("加词 出圈 圈字读xing")
    unmatched_reply = await ctx.send_group(messages[-1], to_me=True)
    replies.append(unmatched_reply)
    require(
        "可用读音" in unmatched_reply
        and "chū juàn" in unmatched_reply
        and "chū quān" in unmatched_reply
        and "管理员" not in unmatched_reply
        and "网页端" not in unmatched_reply
        and not (await ctx.draft()).get("items"),
        f"S39 unmatched reading did not list only available readings: {unmatched_reply}",
    )

    await clean_and_reset("compound suggestion closure")
    messages.append("加词 出圈 jjqt 重新编码")
    compound_reply = await ctx.send_group(messages[-1], to_me=True)
    replies.append(compound_reply)
    narrowed_suggestion = bool(re.search(
        r"(?:可执行命令|可以改为)[\s\S]{0,80}添加「出圈」\s*jjqt"
        r"(?![\s\S]{0,24}(?:重新编码|顺延|腾位))",
        compound_reply,
    ))
    compound_closed = bool(
        "加词 出圈 jjqt 重新编码" in compound_reply
        or "不能保留你要求的添加并腾位操作" in compound_reply
        or (
            "添加" in compound_reply
            and any(marker in compound_reply for marker in ("重新编码", "顺延", "腾位"))
        )
    )
    require(
        compound_closed
        and not narrowed_suggestion
        and not (await ctx.draft()).get("items"),
        f"S39 compound remediation silently narrowed the request: {compound_reply}",
    )

    await clean_and_reset("occupant perspective")
    await require_live_occupant("occupant perspective")
    messages.append(S39_COMMAND)
    occupant_discovery = await ctx.send_group(messages[-1], to_me=True)
    replies.append(occupant_discovery)
    require(
        S39_TARGET_CODE in occupant_discovery
        and S39_OCCUPANT in occupant_discovery,
        f"S39 occupant control lacked a live newcomer state: {occupant_discovery}",
    )
    messages.append(S39_OCCUPANT_COMMAND)
    occupant_reply = await ctx.send_group(messages[-1], to_me=True)
    replies.append(occupant_reply)
    occupant_draft = await ctx.draft()
    require(
        {item_key(item) for item in occupant_draft.get("items", [])}
        == expected_items
        and pending_confirmation_copy() not in occupant_reply,
        f"S39 occupant-perspective command did not resolve {S39_WORD}: "
        f"{occupant_reply}; {occupant_draft}",
    )

    await clean_and_reset("final")
    return {
        "messages": messages,
        "replies": replies,
        "draft": happy_draft,
        "facts": {
            "command": S39_COMMAND,
            "selection": S39_SELECTION,
            "selectedReading": "chū quān",
            "selectedCandidateCodes": ["jjqt", "jjqta", "jjqtai"],
            "happyPathTurnCount": 2,
            "selectionConfirmations": 1,
            "shiftedCode": shifted_code,
            "manualReviewSealed": True,
            "unmatchedReadingListedAvailable": True,
            "compoundSuggestionClosed": compound_closed,
            "occupantPerspectiveResolved": True,
        },
    }


S40_COPY_WORD = "发布会"
S40_OCCUPANT = "重病号"
S40_TARGET_CODE = "fbh"
S40_BATCH_WORDS = S23_BATCH_WORDS[:2]


async def scenario_s40(ctx: ScenarioContext) -> dict[str, Any]:
    """Close executable copy, next-turn submit, recovery, and existing facts."""
    fixture = ctx.fixture_facts["s40"]
    front = fixture["frontCases"][0]
    shifted_code = str(front["shiftedCode"])
    messages: list[str] = []
    replies: list[str] = []

    await ctx.next_client.clean_draft(ctx.platform_id)
    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)

    messages.append(f"喵喵 {S40_OCCUPANT}")
    existing_reply = await ctx.send(messages[-1])
    replies.append(existing_reply)
    first_existing_line = next(
        (line.strip() for line in existing_reply.splitlines() if line.strip()),
        "",
    )
    require(
        first_existing_line == f"「{S40_OCCUPANT}」已在词库（{S40_TARGET_CODE}）。"
        and "无需操作" in existing_reply,
        f"S40 existing-word lookup did not lead with the exact fact: {existing_reply}",
    )

    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)
    messages.append(f"喵喵 {S40_COPY_WORD}")
    recommendation = await ctx.send(messages[-1])
    replies.append(recommendation)
    executable_lines = [
        line.strip()
        for line in recommendation.splitlines()
        if line.strip().startswith("- ")
        and f"「{S40_COPY_WORD}」占 {S40_TARGET_CODE}" in line
        and f"「{S40_OCCUPANT}」顺延" in line
    ]
    require(
        len(executable_lines) == 1,
        f"S40 discovery exposed no unique executable recommendation: {recommendation}",
    )
    copied_line = executable_lines[0]
    require(
        advertised_batch_binding_pairs(copied_line)
        == ((S40_COPY_WORD, S40_TARGET_CODE),),
        f"S40 copied recommendation did not preserve its displayed operands: {copied_line}",
    )

    copy_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append(copied_line)
    copy_reply = await ctx.send(messages[-1])
    replies.append(copy_reply)
    copy_confirmations = 0
    draft = await ctx.draft()
    if not draft.get("items"):
        require(
            pending_confirmation_copy() in copy_reply or "确认" in copy_reply,
            f"S40 copied recommendation neither wrote nor exposed its sealed plan: {copy_reply}",
        )
        messages.append("确认")
        replies.append(await ctx.send(messages[-1]))
        copy_confirmations = 1
        draft = await ctx.draft()
    batch_id = str(draft.get("batchId") or "")
    expected_reorder_items = (
        ("Delete", S40_OCCUPANT, S40_TARGET_CODE),
        ("Create", S40_OCCUPANT, shifted_code),
        ("Create", S40_COPY_WORD, S40_TARGET_CODE),
    )
    actual_reorder_items = tuple(
        item_key(item)
        for item in draft.get("items", [])
        if isinstance(item, dict)
    )
    require(
        batch_id and same_unique_item_set(actual_reorder_items, expected_reorder_items),
        f"S40 copied recommendation did not materialize its exact plan: {draft}",
    )
    copy_calls = [
        event
        for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > copy_cutoff
        and event.get("kind") == "tool"
        and event.get("name") == "keytao_shift_phrase_code"
    ]
    require(copy_calls, "S40 copy-back never reached the shift tool")

    submit_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append("确认并提交")
    submit_reply = await ctx.send(messages[-1])
    replies.append(submit_reply)
    submitted_batch_id = _successful_submit_batch_id(
        ctx.attempt_events(),
        after_sequence=submit_cutoff,
    )
    require(
        submitted_batch_id == batch_id,
        "S40 next-turn 确认并提交 did not submit the actor's just-written batch: "
        f"expected={batch_id}, actual={submitted_batch_id}, reply={submit_reply}",
    )
    submitted_batch = await ctx.next_client.get_admin_batch(
        batch_id=batch_id,
        admin_token=ctx.admin_token,
    )
    require(
        submitted_batch.get("status") in {"Submitted", "Approved"},
        f"S40 copy-back batch never reached submission: {submitted_batch}",
    )

    await ctx.next_client.clean_draft(ctx.platform_id)
    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)
    discovery_message = "喵喵 加词 " + " ".join(S40_BATCH_WORDS)
    messages.append(discovery_message)
    batch_discovery = await ctx.send_group(discovery_message, to_me=True)
    replies.append(batch_discovery)
    stale_message_id = ctx.last_reply_message_id
    displayed_pairs = advertised_batch_binding_pairs(batch_discovery)
    require(
        stale_message_id is not None
        and tuple(word for word, _code in displayed_pairs) == S40_BATCH_WORDS
        and "以上推荐仅用于本次查询" not in batch_discovery,
        f"S40 multi-word query did not expose the exact clean assent set: {batch_discovery}",
    )
    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)
    recovery_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append("加入并提交")
    recovered_reply = await ctx.send_group_reply(
        "加入并提交",
        reply_message_id=stale_message_id,
        to_me=True,
    )
    replies.append(recovered_reply)
    require(
        "回复「加入」写入草稿" not in recovered_reply
        and "回复「加入并提交」" not in recovered_reply,
        f"S40 recovery consumed assent only to ask for it again: {recovered_reply}",
    )
    recovered_batch_id = _successful_submit_batch_id(
        ctx.attempt_events(),
        after_sequence=recovery_cutoff,
    )
    recovery_confirmations = 0
    if not recovered_batch_id:
        recovered_draft = await ctx.draft()
        require(
            not recovered_draft.get("items")
            and (pending_confirmation_copy() in recovered_reply or "确认" in recovered_reply),
            f"S40 recovered assent neither submitted nor reached policy confirmation: {recovered_reply}",
        )
        messages.append("确认")
        replies.append(await ctx.send_group("确认", to_me=True))
        recovery_confirmations = 1
        recovered_batch_id = _successful_submit_batch_id(
            ctx.attempt_events(),
            after_sequence=recovery_cutoff,
        )
    require(recovered_batch_id, "S40 recovered add-and-submit never submitted")
    recovered_batch = await ctx.next_client.get_admin_batch(
        batch_id=recovered_batch_id,
        admin_token=ctx.admin_token,
    )
    recovered_pairs = tuple(
        (str(item.get("word") or ""), str(item.get("code") or "").lower())
        for item in recovered_batch.get("pullRequests", [])
        if isinstance(item, dict) and item.get("action") == "Create"
    )
    require(
        recovered_batch.get("status") in {"Submitted", "Approved"}
        and same_unique_item_set(recovered_pairs, displayed_pairs),
        f"S40 recovery escaped its displayed batch: {recovered_batch}",
    )

    return {
        "messages": messages,
        "replies": replies,
        "draft": await ctx.draft(),
        "facts": {
            "existingLead": first_existing_line,
            "copiedRecommendation": copied_line,
            "copyConfirmationSteps": copy_confirmations,
            "copyBatchId": batch_id,
            "nextTurnSubmitBatchId": submitted_batch_id,
            "recoveredPairs": [list(pair) for pair in recovered_pairs],
            "recoveryConfirmationSteps": recovery_confirmations,
            "recoveredBatchId": recovered_batch_id,
        },
    }


S41_WORD = "畜产品"
S41_READING_MESSAGE = f"{S41_WORD}的畜字怎么读"
S41_CODE_MESSAGE = f"{S41_WORD}怎么编码"
S41_EXISTING_CODES = ("xjpoo", "jjpoo")


async def scenario_s41(ctx: ScenarioContext) -> dict[str, Any]:
    """Keep reading Q&A focused while code Q&A gets bounded unique facts."""
    messages: list[str] = []
    replies: list[str] = []

    await ctx.next_client.clean_draft(ctx.platform_id)
    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)

    messages.append(f"喵喵 {S41_READING_MESSAGE}")
    reading_reply = await ctx.send_group(messages[-1], to_me=True)
    replies.append(reading_reply)
    require(
        S41_WORD in reading_reply
        and re.search(
            r"(?:畜产品[^\n]{0,30}(?:畜[^\n]{0,12})?|畜[^\n]{0,20})"
            r"(?:规范)?读\s*chù",
            reading_reply,
        )
        is not None
        and any(marker in reading_reply for marker in ("这里", "产品", "牲畜", "表示", "用于")),
        f"S41 reading question did not answer the contextual pronunciation: {reading_reply}",
    )
    require(
        re.search(r"畜产品[^\n]{0,40}读\s*xù", reading_reply) is None
        and re.search(r"畜产品[^\n]{0,40}畜[^\n]{0,20}(?:是|为)\s*xù", reading_reply) is None,
        f"S41 reading question assigned the wrong contextual pronunciation: {reading_reply}",
    )
    require(
        not any(
            marker in reading_reply
            for marker in (
                "补充说明：",
                "编码位置说明",
                "常用度对比",
                "当前用",
                *S41_EXISTING_CODES,
            )
        ),
        f"S41 reading question leaked code-position diagnostics: {reading_reply}",
    )
    require(
        "？不，" not in reading_reply
        and "?不," not in reading_reply
        and not duplicate_visible_lines(reading_reply),
        f"S41 reading reply retained scratch correction or duplicate lines: {reading_reply}",
    )

    messages.append(f"喵喵 {S41_CODE_MESSAGE}")
    code_reply = await ctx.send_group(messages[-1], to_me=True)
    replies.append(code_reply)
    existing_fact_lines = [
        line.strip()
        for line in code_reply.splitlines()
        if "「畜产品」已在词库：" in line
        and all(code in line for code in S41_EXISTING_CODES)
    ]
    require(
        "补充说明：" in code_reply
        and "畜产品 的编码位置说明：" in code_reply
        and len(existing_fact_lines) == 1,
        f"S41 code question did not render one existing-entry fact block: {code_reply}",
    )
    require(
        "当前用" not in code_reply
        and code_reply.count("常用度对比：") <= 1
        and not duplicate_visible_lines(code_reply),
        f"S41 code explanation repeated or narrated existing entries as placements: {code_reply}",
    )

    return {
        "messages": messages,
        "replies": replies,
        "draft": await ctx.draft(),
        "facts": {
            "readingFocused": True,
            "existingFact": existing_fact_lines[0],
            "commonnessLineCount": code_reply.count("常用度对比："),
            "readingDuplicateLines": list(duplicate_visible_lines(reading_reply)),
            "codeDuplicateLines": list(duplicate_visible_lines(code_reply)),
        },
    }


S42_WORDS = ("老登", "中登", "小登")


async def scenario_s42(ctx: ScenarioContext) -> dict[str, Any]:
    """Keep every live candidate reply executable without restoring meta prose."""
    messages: list[str] = []
    replies: list[str] = []

    await ctx.next_client.clean_draft(ctx.platform_id)
    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)

    discovery_message = "喵喵 加词 " + " ".join(S42_WORDS)
    messages.append(discovery_message)
    candidate_reply = await ctx.send_group(discovery_message, to_me=True)
    replies.append(candidate_reply)
    displayed_pairs = advertised_batch_binding_pairs(candidate_reply)
    contract = advertised_reply_contract(candidate_reply)
    require(
        tuple(word for word, _code in displayed_pairs) == S42_WORDS,
        f"S42 candidate listing did not bind the exact three words: {candidate_reply}",
    )
    require(
        {"加入", "加入并提交"}.issubset(contract.batch_assent_forms),
        f"S42 live candidate listing omitted assent forms: {candidate_reply}",
    )
    scoped_selection = re.search(
        r"回复「(?P<word>[^」\n]+) 添加(?P<indexes>[1-9]\d*(?:、[1-9]\d*)*)」",
        candidate_reply,
    )
    require(
        scoped_selection is not None
        and scoped_selection.group("word") in S42_WORDS,
        f"S42 restarted numbering omitted a word-scoped selection: {candidate_reply}",
    )
    require(
        not any(
            marker in candidate_reply
            for marker in ("本次仅查询", "只读展示", "执行前系统会重新审词")
        ),
        f"S42 live candidate listing restored meta narration: {candidate_reply}",
    )

    submit_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append("加入并提交")
    submit_reply = await ctx.send_group("加入并提交", to_me=True)
    replies.append(submit_reply)
    submitted_batch_id = _successful_submit_batch_id(
        ctx.attempt_events(),
        after_sequence=submit_cutoff,
    )
    require(
        submitted_batch_id
        and "回复「加入」" not in submit_reply
        and "回复「加入并提交」" not in submit_reply,
        f"S42 bare add-and-submit did not execute in that turn: {submit_reply}",
    )
    submitted_batch = await ctx.next_client.get_admin_batch(
        batch_id=submitted_batch_id,
        admin_token=ctx.admin_token,
    )
    submitted_pairs = tuple(
        (str(item.get("word") or ""), str(item.get("code") or "").lower())
        for item in submitted_batch.get("pullRequests", [])
        if isinstance(item, dict) and item.get("action") == "Create"
    )
    require(
        submitted_batch.get("status") in {"Submitted", "Approved"}
        and same_unique_item_set(submitted_pairs, displayed_pairs),
        f"S42 bare assent escaped its displayed batch: {submitted_batch}",
    )

    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)
    query_message = f"喵喵 {S41_READING_MESSAGE}"
    messages.append(query_message)
    query_reply = await ctx.send_group(query_message, to_me=True)
    replies.append(query_reply)
    query_contract = advertised_reply_contract(query_reply)
    require(
        S41_WORD in query_reply
        and not query_contract.requires_live_state
        and not any(
            marker in query_reply
            for marker in (
                "回复「加入」",
                "加入并提交",
                " 添加1",
                " 添加2",
                "本次仅查询",
                "只读展示",
                "执行前系统会重新审词",
            )
        ),
        f"S42 query-only reply advertised a nonexistent action: {query_reply}",
    )

    return {
        "messages": messages,
        "replies": replies,
        "draft": await ctx.draft(),
        "facts": {
            "advertisedPairs": [list(pair) for pair in displayed_pairs],
            "advertisedForms": list(contract.batch_assent_forms),
            "scopedSelection": scoped_selection.group(0),
            "submittedBatchId": submitted_batch_id,
            "queryAdvertisesAction": query_contract.requires_live_state,
        },
    }


S43_WORD = "钉选"


async def scenario_s43(ctx: ScenarioContext) -> dict[str, Any]:
    """Recover on a longer retry, then degrade read-only after full outage."""
    messages: list[str] = []
    replies: list[str] = []

    await ctx.next_client.clean_draft(ctx.platform_id)
    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)

    retry_message = f"喵喵 查词 {S43_WORD}"
    ctx.encode_delay.arm(ctx.scenario_id, injections=1)
    try:
        messages.append(retry_message)
        recovered_reply = await ctx.send_group(retry_message, to_me=True)
        replies.append(recovered_reply)
        recovered_injections = ctx.encode_delay.injection_count
    finally:
        ctx.encode_delay.disarm()
    require(
        recovered_injections == 1,
        f"S43 did not inject exactly the first encode attempt: {recovered_injections}",
    )
    require(
        S43_WORD in recovered_reply
        and "编码服务重试后仍不可用" not in recovered_reply
        and "未经编码服务确认" not in recovered_reply,
        f"S43 longer retry did not complete through the normal flow: {recovered_reply}",
    )
    recovery_retry_logs = [
        str(event.get("message") or "")
        for event in ctx.attempt_events()
        if event.get("kind") == "log"
        and "[http_client] retry" in str(event.get("message") or "")
        and "next_timeout=20s" in str(event.get("message") or "")
    ]
    require(
        recovery_retry_logs,
        "S43 recovered without logging the longer second-attempt budget",
    )

    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)
    ctx.encode_delay.arm(ctx.scenario_id, injections=3)
    try:
        messages.append(retry_message)
        degraded_reply = await ctx.send_group(retry_message, to_me=True)
        replies.append(degraded_reply)
        failure_injections = ctx.encode_delay.injection_count
    finally:
        ctx.encode_delay.disarm()
    require(
        failure_injections == 3,
        f"S43 did not exhaust all three encode attempts: {failure_injections}",
    )
    degraded_contract = advertised_reply_contract(degraded_reply)
    require(
        S43_WORD in degraded_reply
        and "上游暂时不可用" in degraded_reply
        and "未经编码服务确认" in degraded_reply
        and "dīng xuǎn" in degraded_reply
        and "dgxt" in degraded_reply,
        f"S43 degraded reply omitted offline evidence: {degraded_reply}",
    )
    require(
        f"查词 {S43_WORD}" in degraded_reply
        and "查看草稿" not in degraded_reply,
        f"S43 degraded reply advertised the wrong next step: {degraded_reply}",
    )
    require(
        "本次不提供写入" in degraded_reply
        and not degraded_contract.requires_live_state
        and not any(
            marker in degraded_reply
            for marker in ("回复「加入」", "加入并提交", " 添加1", " 添加2")
        ),
        f"S43 degraded reply exposed a write affordance: {degraded_reply}",
    )
    draft = await ctx.draft()
    require(not draft.get("items"), f"S43 query path wrote to the draft: {draft}")

    return {
        "messages": messages,
        "replies": replies,
        "draft": draft,
        "facts": {
            "recoveredInjections": recovered_injections,
            "failureInjections": failure_injections,
            "longerRetryLogged": True,
            "degradedReadings": ["dīng xuǎn"],
            "degradedCandidateBase": "dgxt",
            "retryCommand": f"查词 {S43_WORD}",
            "writeAdvertised": degraded_contract.requires_live_state,
        },
    }


S44_WORD = "载具"
S44_OCCUPANT = "在距"
S44_OCCUPIED_CODE = "zhjl"
S44_FREE_CODE = "zhjlu"
S44_DISCOVERY = f"加词 {S44_WORD}"
S44_COMMAND = "1 重新编码，并加入 2"


async def scenario_s44(ctx: ScenarioContext) -> dict[str, Any]:
    """Execute one compound selection and keep invalid controls read-only."""
    messages: list[str] = []
    replies: list[str] = []

    cleanup = await ctx.next_client.clean_draft(ctx.platform_id)
    require(cleanup.get("success") is True, f"S44 cleanup failed: {cleanup}")
    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)
    fixture = ctx.fixture_facts.get("s44") or {}
    shifted_code = str(fixture.get("shiftedCode") or "").strip()
    require(
        shifted_code
        and shifted_code not in {S44_OCCUPIED_CODE, S44_FREE_CODE},
        f"S44 fixture omitted a post-reservation shift slot: {fixture}",
    )

    messages.append(S44_DISCOVERY)
    discovery = await ctx.send_group(messages[-1], to_me=True)
    replies.append(discovery)
    require(
        f"1. {S44_OCCUPIED_CODE} — 已有「{S44_OCCUPANT}」" in discovery
        and f"2. {S44_FREE_CODE} — 空位" in discovery,
        f"S44 discovery did not expose the exact live slots: {discovery}",
    )

    controls = (
        ("2 重新编码 + 1", ("编号 2", "空位", "不能使用「重新编码」")),
        ("1 重新编码 + 1", ("编号 1", "重复")),
        ("1 重新编码 + 4", ("编号 4", "超出当前候选范围 1-3")),
    )
    control_facts: list[dict[str, Any]] = []
    for command, markers in controls:
        cutoff = max(
            (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
            default=0,
        )
        messages.append(command)
        reply = await ctx.send_group(command, to_me=True)
        replies.append(reply)
        turn_events = [
            event
            for event in ctx.attempt_events()
            if int(event.get("sequence") or 0) > cutoff
        ]
        model_turns = [
            event for event in turn_events
            if event.get("kind") == "modelExchange"
        ]
        mutation_tools = [
            event for event in turn_events
            if event.get("kind") == "tool"
            and event.get("name") in {
                "keytao_batch_add_to_draft",
                "keytao_create_phrase",
                "keytao_shift_phrase_code",
            }
        ]
        draft = await ctx.draft()
        require(
            all(marker in reply for marker in markers)
            and "本次未写入" in reply
            and not draft.get("items")
            and not model_turns
            and not mutation_tools,
            f"S44 control was not a deterministic read-only ASK: "
            f"command={command}; reply={reply}; draft={draft}; "
            f"models={model_turns}; tools={mutation_tools}",
        )
        control_facts.append({
            "command": command,
            "markers": list(markers),
            "modelTurns": len(model_turns),
            "mutationTools": len(mutation_tools),
            "draftItems": len(draft.get("items") or []),
        })

    happy_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append(S44_COMMAND)
    plan_reply = await ctx.send_group(S44_COMMAND, to_me=True)
    replies.append(plan_reply)
    preview_draft = await ctx.draft()
    preview_events = [
        event
        for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > happy_cutoff
    ]
    preview_calls = [
        event for event in preview_events
        if event.get("kind") == "tool"
        and event.get("name") == "keytao_shift_phrase_code"
        and not event.get("arguments", {}).get("confirmed_plan_digest")
    ]
    preview_model_turns = [
        event for event in preview_events
        if event.get("kind") == "modelExchange"
    ]
    require(
        f"「{S44_WORD}」→ {S44_OCCUPIED_CODE}" in plan_reply
        and f"「{S44_OCCUPANT}」顺延至 {shifted_code}" in plan_reply
        and f"「{S44_WORD}」→ {S44_FREE_CODE}" in plan_reply
        and plan_reply.count(pending_confirmation_copy()) == 1
        and len(preview_calls) == 1
        and not preview_model_turns
        and not preview_draft.get("items"),
        f"S44 compound selection was not one deterministic preview: "
        f"reply={plan_reply}; draft={preview_draft}; "
        f"calls={preview_calls}; models={preview_model_turns}",
    )

    messages.append("确认")
    confirmation_reply = await ctx.send_group(messages[-1], to_me=True)
    replies.append(confirmation_reply)
    draft = await ctx.draft()
    expected_items = {
        ("Delete", S44_OCCUPANT, S44_OCCUPIED_CODE),
        ("Create", S44_WORD, S44_OCCUPIED_CODE),
        ("Create", S44_OCCUPANT, shifted_code),
        ("Create", S44_WORD, S44_FREE_CODE),
    }
    actual_items = {item_key(item) for item in draft.get("items", [])}
    happy_events = [
        event
        for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > happy_cutoff
    ]
    confirmed_calls = [
        event for event in happy_events
        if event.get("kind") == "tool"
        and event.get("name") == "keytao_shift_phrase_code"
        and event.get("arguments", {}).get("confirmed_plan_digest")
    ]
    happy_model_turns = [
        event for event in happy_events
        if event.get("kind") == "modelExchange"
    ]
    require(
        actual_items == expected_items
        and len(draft.get("items", [])) == 4
        and len(confirmed_calls) == 1
        and not happy_model_turns,
        f"S44 confirmation did not write the exact sealed plan once: "
        f"reply={confirmation_reply}; draft={draft}; "
        f"calls={confirmed_calls}; models={happy_model_turns}",
    )

    final_cleanup = await ctx.next_client.clean_draft(ctx.platform_id)
    require(
        final_cleanup.get("success") is True,
        f"S44 final cleanup failed: {final_cleanup}",
    )
    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)
    return {
        "messages": messages,
        "replies": replies,
        "draft": draft,
        "facts": {
            "word": S44_WORD,
            "occupant": S44_OCCUPANT,
            "occupiedCode": S44_OCCUPIED_CODE,
            "additionalCode": S44_FREE_CODE,
            "shiftedCode": shifted_code,
            "controlAsks": control_facts,
            "previewCalls": len(preview_calls),
            "confirmedCalls": len(confirmed_calls),
            "selectionModelTurns": len(happy_model_turns),
            "confirmationSteps": 1,
            "expectedItems": sorted(expected_items),
            "actualItems": sorted(actual_items),
        },
    }


S45_FIRST_WORD = "财宝"
S45_SECOND_WORD = "财报"
S45_FIRST_CODE = "chbz"
S45_SECOND_CODE = "chbza"
S45_SWAP_MESSAGE = "对换财宝和财报的编码"
S45_CHARACTER_QUESTION = "单人旁加个巨字是什么字"
S45_CHARACTER_ANSWER = "佢"


async def scenario_s45(ctx: ScenarioContext) -> dict[str, Any]:
    """Swap two exact codes, then answer a character question without review."""
    messages: list[str] = []
    replies: list[str] = []

    cleanup = await ctx.next_client.clean_draft(ctx.platform_id)
    require(cleanup.get("success") is True, f"S45 cleanup failed: {cleanup}")
    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)

    swap_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append(S45_SWAP_MESSAGE)
    plan_reply = await ctx.send_group(S45_SWAP_MESSAGE, to_me=True)
    replies.append(plan_reply)
    preview_draft = await ctx.draft()
    preview_events = [
        event for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > swap_cutoff
    ]
    preview_calls = [
        event for event in preview_events
        if event.get("kind") == "tool"
        and event.get("name") == "keytao_shift_phrase_code"
        and not event.get("arguments", {}).get("confirmed_plan_digest")
    ]
    require(
        pending_confirmation_copy() in plan_reply
        and all(
            marker in plan_reply
            for marker in (
                S45_FIRST_WORD,
                S45_SECOND_WORD,
                S45_FIRST_CODE,
                S45_SECOND_CODE,
            )
        )
        and len(preview_calls) == 1
        and not preview_draft.get("items"),
        f"S45 verbatim swap did not offer the resolved plan: "
        f"reply={plan_reply}; calls={preview_calls}; draft={preview_draft}",
    )

    messages.append("确认")
    confirmation_reply = await ctx.send_group(messages[-1], to_me=True)
    replies.append(confirmation_reply)
    swap_draft = await ctx.draft()
    expected_swap_items = {
        ("Delete", S45_FIRST_WORD, S45_FIRST_CODE),
        ("Create", S45_FIRST_WORD, S45_SECOND_CODE),
        ("Delete", S45_SECOND_WORD, S45_SECOND_CODE),
        ("Create", S45_SECOND_WORD, S45_FIRST_CODE),
    }
    actual_swap_items = {
        item_key(item) for item in swap_draft.get("items", [])
    }
    swap_events = [
        event for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > swap_cutoff
    ]
    confirmed_calls = [
        event for event in swap_events
        if event.get("kind") == "tool"
        and event.get("name") == "keytao_shift_phrase_code"
        and event.get("arguments", {}).get("confirmed_plan_digest")
    ]
    swap_model_turns = [
        event for event in swap_events
        if event.get("kind") == "modelExchange"
    ]
    require(
        actual_swap_items == expected_swap_items
        and len(swap_draft.get("items") or []) == 4
        and len(confirmed_calls) == 1
        and not swap_model_turns,
        f"S45 confirmation did not exchange both codes exactly once: "
        f"reply={confirmation_reply}; draft={swap_draft}; "
        f"calls={confirmed_calls}; models={swap_model_turns}",
    )

    cleanup = await ctx.next_client.clean_draft(ctx.platform_id)
    require(cleanup.get("success") is True, f"S45 swap cleanup failed: {cleanup}")
    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)
    question_cutoff = max(
        (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
        default=0,
    )
    messages.append(S45_CHARACTER_QUESTION)
    answer_reply = await ctx.send_group(S45_CHARACTER_QUESTION, to_me=True)
    replies.append(answer_reply)
    question_events = [
        event for event in ctx.attempt_events()
        if int(event.get("sequence") or 0) > question_cutoff
    ]
    encode_calls = [
        event for event in question_events
        if event.get("kind") == "tool"
        and event.get("name") == "keytao_encode"
        and event.get("arguments", {}).get("word") == S45_CHARACTER_ANSWER
    ]
    character_data_verified = any(
        isinstance(event.get("result"), dict)
        and any(
            isinstance(char, dict)
            and char.get("char") == S45_CHARACTER_ANSWER
            for char in event["result"].get("chars") or []
        )
        for event in encode_calls
    )
    review_calls = [
        event for event in question_events
        if event.get("kind") == "tool"
        and event.get("name") == "keytao_prepare_reviewed_add"
    ]
    question_draft = await ctx.draft()
    answer_contract = advertised_reply_contract(answer_reply)
    require(
        S45_CHARACTER_ANSWER in answer_reply
        and character_data_verified
        and not review_calls
        and not answer_contract.requires_live_state
        and not any(
            marker in answer_reply
            for marker in (
                "审词：",
                "候选编码",
                "回复「加入」",
                "加入并提交",
                "写入草稿",
            )
        )
        and not question_draft.get("items"),
        f"S45 character question entered review/add flow or lacked data proof: "
        f"reply={answer_reply}; encode={encode_calls}; review={review_calls}; "
        f"draft={question_draft}",
    )

    return {
        "messages": messages,
        "replies": replies,
        "draft": question_draft,
        "facts": {
            "swapMessage": S45_SWAP_MESSAGE,
            "previewCalls": len(preview_calls),
            "confirmedCalls": len(confirmed_calls),
            "confirmationSteps": 1,
            "swapModelTurns": len(swap_model_turns),
            "expectedSwapItems": sorted(expected_swap_items),
            "actualSwapItems": sorted(actual_swap_items),
            "question": S45_CHARACTER_QUESTION,
            "answerCharacter": S45_CHARACTER_ANSWER,
            "characterDataVerified": character_data_verified,
            "reviewCalls": len(review_calls),
            "writeAdvertised": answer_contract.requires_live_state,
        },
    }


S46_WORD = "哲思"
S46_OCCUPANT = "这厮"
S46_FIRST_CODE = "fesk"
S46_FIRST_SHIFTED_CODE = "fesko"
S46_SECOND_CODE = "qesk"
S46_SECOND_SHIFTED_CODE = "qesko"
S46_MESSAGE = (
    '加词 哲思 fesk，并且为"这厮 fesk"顺延\n'
    '加词 哲思 qesk，并且为"这厮 qesk"顺延'
)
S46_PLAN_COMMAND = (
    "添加「哲思」 fesk，这厮顺延\n"
    "添加「哲思」 qesk，这厮顺延"
)


async def scenario_s46(ctx: ScenarioContext) -> dict[str, Any]:
    """Execute and replay one two-line plan with the same occupant twice."""
    messages: list[str] = []
    replies: list[str] = []
    expected_items = {
        ("Delete", S46_OCCUPANT, S46_FIRST_CODE),
        ("Create", S46_WORD, S46_FIRST_CODE),
        ("Create", S46_OCCUPANT, S46_FIRST_SHIFTED_CODE),
        ("Delete", S46_OCCUPANT, S46_SECOND_CODE),
        ("Create", S46_WORD, S46_SECOND_CODE),
        ("Create", S46_OCCUPANT, S46_SECOND_SHIFTED_CODE),
    }

    async def execute_plan(command: str, label: str) -> dict[str, Any]:
        cleanup = await ctx.next_client.clean_draft(ctx.platform_id)
        require(cleanup.get("success") is True, f"S46 {label} cleanup failed: {cleanup}")
        await ctx.bot.reset_conversation(platform_id=ctx.platform_id)
        cutoff = max(
            (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
            default=0,
        )

        messages.append(command)
        plan_reply = await ctx.send_group(command, to_me=True)
        replies.append(plan_reply)
        preview_draft = await ctx.draft()
        preview_events = [
            event
            for event in ctx.attempt_events()
            if int(event.get("sequence") or 0) > cutoff
        ]
        preview_calls = [
            event
            for event in preview_events
            if event.get("kind") == "tool"
            and event.get("name") == "keytao_shift_phrase_code"
            and not event.get("arguments", {}).get("confirmed_plan_digest")
        ]
        preview_model_turns = [
            event
            for event in preview_events
            if event.get("kind") == "modelExchange"
        ]
        contract = advertised_reply_contract(plan_reply)
        require(
            all(
                marker in plan_reply
                for marker in (
                    f"「{S46_WORD}」→ {S46_FIRST_CODE}",
                    f"「{S46_OCCUPANT}」顺延至 {S46_FIRST_SHIFTED_CODE}",
                    f"「{S46_WORD}」→ {S46_SECOND_CODE}",
                    f"「{S46_OCCUPANT}」顺延至 {S46_SECOND_SHIFTED_CODE}",
                )
            )
            and plan_reply.count(pending_confirmation_copy()) == 1
            and not contract.command_suggestions
            and len(preview_calls) == 1
            and not preview_model_turns
            and not preview_draft.get("items"),
            f"S46 {label} did not produce one plan-only preview: "
            f"reply={plan_reply}; calls={preview_calls}; "
            f"models={preview_model_turns}; draft={preview_draft}",
        )

        messages.append("确认")
        confirmation_reply = await ctx.send_group("确认", to_me=True)
        replies.append(confirmation_reply)
        draft = await ctx.draft()
        actual_items = {item_key(item) for item in draft.get("items", [])}
        all_events = [
            event
            for event in ctx.attempt_events()
            if int(event.get("sequence") or 0) > cutoff
        ]
        confirmed_calls = [
            event
            for event in all_events
            if event.get("kind") == "tool"
            and event.get("name") == "keytao_shift_phrase_code"
            and event.get("arguments", {}).get("confirmed_plan_digest")
        ]
        model_turns = [
            event for event in all_events if event.get("kind") == "modelExchange"
        ]
        require(
            actual_items == expected_items
            and len(draft.get("items") or []) == len(expected_items)
            and len(confirmed_calls) == 1
            and not model_turns,
            f"S46 {label} confirmation diverged from the promised end state: "
            f"reply={confirmation_reply}; draft={draft}; "
            f"calls={confirmed_calls}; models={model_turns}",
        )
        return {
            "planReply": plan_reply,
            "confirmationReply": confirmation_reply,
            "draft": draft,
            "actualItems": sorted(actual_items),
            "previewCalls": len(preview_calls),
            "confirmedCalls": len(confirmed_calls),
            "modelTurns": len(model_turns),
        }

    incident = await execute_plan(S46_MESSAGE, "incident")
    replay = await execute_plan(S46_PLAN_COMMAND, "plan-command replay")
    require(
        incident["actualItems"] == replay["actualItems"] == sorted(expected_items),
        f"S46 plan-object command did not reproduce the promised state: "
        f"incident={incident}; replay={replay}",
    )

    final_cleanup = await ctx.next_client.clean_draft(ctx.platform_id)
    require(final_cleanup.get("success") is True, f"S46 final cleanup failed: {final_cleanup}")
    await ctx.bot.reset_conversation(platform_id=ctx.platform_id)
    return {
        "messages": messages,
        "replies": replies,
        "draft": replay["draft"],
        "facts": {
            "word": S46_WORD,
            "occupant": S46_OCCUPANT,
            "targetCodes": [S46_FIRST_CODE, S46_SECOND_CODE],
            "shiftedCodes": [S46_FIRST_SHIFTED_CODE, S46_SECOND_SHIFTED_CODE],
            "sameOccupantShiftCount": 2,
            "confirmationStepsPerRun": 1,
            "advertisedCommandCount": 0,
            "planCommand": S46_PLAN_COMMAND,
            "planCommandReproducedPromise": True,
            "incident": incident,
            "replay": replay,
            "expectedItems": sorted(expected_items),
        },
    }


S47_EVICTION_MESSAGE = "加词 哲思 fesk 重新编码"
S47_RECODE_COMMAND = "删除 这厮 fesk；添加 这厮 fesko"


async def scenario_s47(ctx: ScenarioContext) -> dict[str, Any]:
    """Close choice, suggestion, deterministic recode, and draft-merge paths."""
    messages: list[str] = []
    replies: list[str] = []

    async def reset() -> None:
        cleanup = await ctx.next_client.clean_draft(ctx.platform_id)
        require(cleanup.get("success") is True, f"S47 cleanup failed: {cleanup}")
        await ctx.bot.reset_conversation(platform_id=ctx.platform_id)

    def cutoff() -> int:
        return max(
            (int(event.get("sequence") or 0) for event in ctx.attempt_events()),
            default=0,
        )

    def events_after(sequence: int) -> list[dict[str, Any]]:
        return [
            event for event in ctx.attempt_events()
            if int(event.get("sequence") or 0) > sequence
        ]

    await reset()
    messages.append("删词 这厮")
    refusal = await ctx.send_group("删词 这厮", to_me=True)
    replies.append(refusal)
    suggested_match = re.search(
        r"[-•]\s*[「“『](删词\s+这厮\s+[a-z]+)[」”』]",
        refusal,
    )
    require(
        suggested_match is not None,
        f"S47 refusal did not expose a plan-rendered exact command: {refusal}",
    )
    suggested_command = suggested_match.group(1)
    suggestion_cutoff = cutoff()
    messages.append(suggested_command)
    suggestion_reply = await ctx.send_group(suggested_command, to_me=True)
    replies.append(suggestion_reply)
    suggestion_events = events_after(suggestion_cutoff)
    suggestion_models = [
        event for event in suggestion_events if event.get("kind") == "modelExchange"
    ]
    suggestion_batch_calls = [
        event for event in suggestion_events
        if event.get("kind") == "tool"
        and event.get("name") == "keytao_batch_add_to_draft"
    ]
    require(
        suggestion_batch_calls
        and not suggestion_models
        and pending_confirmation_copy() in suggestion_reply,
        f"S47 advertised command did not execute deterministically: "
        f"command={suggested_command}; reply={suggestion_reply}; "
        f"events={suggestion_events}",
    )

    await reset()
    recode_cutoff = cutoff()
    messages.append(S47_RECODE_COMMAND)
    recode_reply = await ctx.send_group(S47_RECODE_COMMAND, to_me=True)
    replies.append(recode_reply)
    if pending_confirmation_copy() in recode_reply:
        messages.append("确认")
        recode_reply = await ctx.send_group("确认", to_me=True)
        replies.append(recode_reply)
    recode_events = events_after(recode_cutoff)
    recode_models = [
        event for event in recode_events if event.get("kind") == "modelExchange"
    ]
    recode_batch_calls = [
        event for event in recode_events
        if event.get("kind") == "tool"
        and event.get("name") == "keytao_batch_add_to_draft"
    ]
    recode_draft = await ctx.draft()
    recode_items = {item_key(item) for item in recode_draft.get("items", [])}
    require(
        not recode_models
        and recode_batch_calls
        and {
            ("Delete", "这厮", "fesk"),
            ("Create", "这厮", "fesko"),
        }.issubset(recode_items),
        f"S47 explicit recode was not one deterministic plan: "
        f"reply={recode_reply}; draft={recode_draft}; events={recode_events}",
    )

    await reset()
    await ctx.next_client.add_draft_items(
        platform_id=ctx.platform_id,
        items=[{
            "action": "Create",
            "word": "哲思",
            "code": "fesko",
            "type": "Phrase",
            "needsManualReview": False,
        }],
    )
    messages.append(S47_EVICTION_MESSAGE)
    choice_reply = await ctx.send_group(S47_EVICTION_MESSAGE, to_me=True)
    replies.append(choice_reply)
    require(
        all(marker in choice_reply for marker in (
            "A. 删除上述冲突草稿行",
            "B. 保留当前草稿",
            "回复 A 或 B 即可",
        )),
        f"S47 conflict did not create a live choice offer: {choice_reply}",
    )
    choice_a_cutoff = cutoff()
    messages.append("A")
    choice_a_reply = await ctx.send_group("A", to_me=True)
    replies.append(choice_a_reply)
    choice_events = events_after(choice_a_cutoff)
    choice_remove_calls = [
        event for event in choice_events
        if event.get("kind") == "tool"
        and event.get("name") == "keytao_batch_remove_draft_items"
    ]
    require(
        choice_remove_calls
        and not any(
            event.get("kind") == "modelExchange" for event in choice_events
        ),
        f"S47 bare A did not execute the recorded cleanup option: "
        f"reply={choice_a_reply}; events={choice_events}",
    )
    if pending_confirmation_copy() in choice_a_reply:
        messages.append("确认")
        cleanup_reply = await ctx.send_group("确认", to_me=True)
        replies.append(cleanup_reply)
    require(
        not (await ctx.draft()).get("items"),
        f"S47 selected cleanup did not remove the conflicting row: {await ctx.draft()}",
    )

    await reset()
    await ctx.next_client.add_draft_items(
        platform_id=ctx.platform_id,
        items=[{
            "action": "Delete",
            "word": "这厮",
            "code": "fesk",
            "type": "Phrase",
        }],
    )
    seeded_merge_draft = await ctx.draft()
    merge_cutoff = cutoff()
    messages.append(S47_EVICTION_MESSAGE)
    merge_reply = await ctx.send_group(S47_EVICTION_MESSAGE, to_me=True)
    replies.append(merge_reply)
    merge_events = events_after(merge_cutoff)
    shift_previews = [
        event for event in merge_events
        if event.get("kind") == "tool"
        and event.get("name") == "keytao_shift_phrase_code"
        and not event.get("arguments", {}).get("confirmed_plan_digest")
    ]
    merge_plan = next((
        event.get("result", {}).get("shiftPlan", {})
        for event in shift_previews
        if isinstance(event.get("result"), dict)
        and isinstance(event.get("result", {}).get("shiftPlan"), dict)
    ), {})
    require(
        merge_plan.get("mergedDraftItems")
        and merge_plan.get("remainingItems")
        and not any(
            isinstance(event.get("result"), dict)
            and event.get("result", {}).get("requiresDraftCleanup") is True
            for event in shift_previews
        )
        and pending_confirmation_copy() in merge_reply,
        f"S47 exact occupant draft row was not merged into the delta: "
        f"reply={merge_reply}; plan={merge_plan}; events={merge_events}",
    )
    messages.append("确认")
    merge_confirmation = await ctx.send_group("确认", to_me=True)
    replies.append(merge_confirmation)
    final_draft = await ctx.draft()
    final_items = {item_key(item) for item in final_draft.get("items", [])}
    require(
        {
            ("Delete", "这厮", "fesk"),
            ("Create", "哲思", "fesk"),
            ("Create", "这厮", "fesko"),
        }.issubset(final_items)
        and len(final_draft.get("items", [])) == 3,
        f"S47 merged plan did not reach the promised draft: {final_draft}",
    )

    return {
        "messages": messages,
        "replies": replies,
        "draft": final_draft,
        "facts": {
            "suggestedCommand": suggested_command,
            "suggestedCommandExecuted": True,
            "explicitRecodeDeterministic": True,
            "choiceAExecuted": True,
            "mergedDraftItemIds": [
                item.get("id") for item in merge_plan.get("mergedDraftItems") or []
            ],
            "seededMergeDraft": seeded_merge_draft,
            "remainingDelta": merge_plan.get("remainingItems"),
            "finalItems": sorted(final_items),
        },
    }


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("S1", "cold eviction default", scenario_s1),
    Scenario("S2", "explicit duplicate", scenario_s2),
    Scenario("S3", "back placement", scenario_s3),
    Scenario("S4", "stale and single-delete confirmation", scenario_s4),
    Scenario("S5", "self-service convergence", scenario_s5),
    Scenario("S6", "injection controls", scenario_s6),
    Scenario("S7", "timeout retry", scenario_s7),
    Scenario("S8", "admin approval chain", scenario_s8),
    Scenario("S9", "candidate commonness ordering", scenario_s9),
    Scenario("S10", "multi-add authorization", scenario_s10),
    Scenario("S11", "front-insert ticket before extra action", scenario_s11),
    Scenario("S12", "front-insert weight legality", scenario_s12),
    Scenario("S13", "same-turn resend loop breaker", scenario_s13),
    Scenario("S14", "wrong-entry pronunciation poisoning", scenario_s14),
    Scenario("S15", "numbered add-submit and suggestion/direct closure", scenario_s15),
    Scenario("S16", "two-word bare advertised add-submit", scenario_s16),
    Scenario("S17", "semantic common-character auto-pass", scenario_s17),
    Scenario("S18", "multi-number candidate snapshot selection", scenario_s18),
    Scenario("S19", "advertised-set subtraction with chunked progress", scenario_s19),
    Scenario("S20", "native-quoted batch assent", scenario_s20),
    Scenario("S21", "assent modifier and rendered remediation closure", scenario_s21),
    Scenario("S22", "re-review advertisement state coupling", scenario_s22),
    Scenario("S23", "stale advertised assent recovery and fresh closure", scenario_s23),
    Scenario("S24", "single-word natural quoted assent", scenario_s24),
    Scenario("S25", "natural add, record-backed number, and combined submit", scenario_s25),
    Scenario("S26", "server-resolved add with occupant eviction", scenario_s26),
    Scenario("S27", "binding precheck and question-turn reply", scenario_s27),
    Scenario("S28", "reviewed multi-reading cascade closure", scenario_s28),
    Scenario("S29", "quoted same-code commonness reorder", scenario_s29),
    Scenario("S30", "read cancel and natural assent closure", scenario_s30),
    Scenario("S31", "verbatim positional eviction closure", scenario_s31),
    Scenario("S32", "draft-aware and explicit-list chain scope", scenario_s32),
    Scenario("S33", "batch-aware homophone slot allocation", scenario_s33),
    Scenario("S34", "pending submitted word awareness", scenario_s34),
    Scenario("S35", "comparator recommendation is the default add plan", scenario_s35),
    Scenario("S36", "dictionary delete and exact swap incident round", scenario_s36),
    Scenario("S37", "occupant eviction and selected-slot revalidation", scenario_s37),
    Scenario("S38", "reading, query recovery, and modifier incident closure", scenario_s38),
    Scenario("S39", "one-turn reading selection and occupant eviction closure", scenario_s39),
    Scenario("S40", "assent execution and existing-word incident closure", scenario_s40),
    Scenario("S41", "reading focus and code explanation deduplication", scenario_s41),
    Scenario("S42", "live candidate affordances and bare assent execution", scenario_s42),
    Scenario("S43", "encode retry ladder and offline read-only degradation", scenario_s43),
    Scenario("S44", "deterministic compound candidate selection", scenario_s44),
    Scenario("S45", "swap verbs and interrogative review boundary", scenario_s45),
    Scenario("S46", "multi-line promise-preserving double eviction", scenario_s46),
    Scenario("S47", "choice and executable-suggestion closure", scenario_s47),
)


async def run_scenario(scenario: Scenario, context: ScenarioContext) -> dict[str, Any]:
    started = time.monotonic()
    try:
        details = await scenario.execute(context)
        return {
            "verdict": "PASSED",
            "failure": None,
            "durationSeconds": time.monotonic() - started,
            **details,
        }
    except SafetyViolation:
        raise
    except Exception as error:
        try:
            draft = await context.draft()
        except Exception as draft_error:
            draft = {"unavailable": f"{type(draft_error).__name__}: {draft_error}"}
        return {
            "verdict": "FAILED",
            "failure": f"{type(error).__name__}: {error}",
            "durationSeconds": time.monotonic() - started,
            "draft": draft,
            "messages": [
                event.get("text")
                for event in context.attempt_events()
                if event.get("kind") == "message" and event.get("direction") == "input"
            ],
            "replies": [
                event.get("text")
                for event in context.attempt_events()
                if event.get("kind") == "message" and event.get("direction") == "reply"
            ],
            "facts": {},
        }
