"""Scenario pack v1 with end-state and reply-marker assertions."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

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


def batch_link_ids(reply: str) -> set[str]:
    return set(re.findall(r"/batch/([A-Za-z0-9_-]+)", reply))


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

    async def send(self, text: str) -> str:
        return await self.bot.send(
            platform_id=self.platform_id,
            sender_name=self.sender_name,
            text=text,
        )

    async def send_group(self, text: str, *, to_me: bool) -> str:
        return await self.bot.send_group(
            platform_id=self.platform_id,
            sender_name=self.sender_name,
            text=text,
            to_me=to_me,
        )

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
    require(len(draft["items"]) == 1, f"S2 created more than one draft item: {draft}")
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
    reply = await ctx.send("确认")
    draft = await ctx.draft()
    require(not draft["items"], f"stale confirmation changed the draft: {draft}")
    require("已经过期" in reply or "已过期" in reply, f"S4 omitted expiry marker: {reply}")
    require(
        not re.search(r"(?:重新|再次).{0,8}(?:发送|回复).{0,4}[「“]?确认", reply),
        f"S4 advised resending confirmation: {reply}",
    )
    return {
        "messages": ["确认"],
        "replies": [reply],
        "draft": draft,
        "facts": {"draftChanges": 0, "expiryMarker": True},
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
        confirmation = "确认加入 王中王 wfw 微服务 wfwu"
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
    if not second_batch_id and "回复「确认」、「执行」继续" in guidance:
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
            r"是否以编码\s+(?P<code>[a-z]{2,12})\s+将「亮面」加入草稿",
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
            r"请发送(?P<suggestion>「添加\s+亮面\s+(?P<code>[a-z]{2,12})\s+并提交」)",
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
    messages = ["喵喵 加词 载流 载流子"]
    replies = [await ctx.send(messages[-1])]
    discovery_reply = replies[-1]
    require(
        re.search(r"(?m)^-\s*「载流」\s*(?:→|->)\s*zhlq\s*$", discovery_reply)
        is not None,
        f"S16 discovery did not bind 载流@zhlq: {discovery_reply}",
    )
    require(
        re.search(r"(?m)^-\s*「载流子」\s*(?:→|->)\s*zlzu\s*$", discovery_reply)
        is not None,
        f"S16 discovery did not bind 载流子@zlzu: {discovery_reply}",
    )
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
    )
    batch_id = _successful_submit_batch_id(
        ctx.attempt_events(),
        after_sequence=cutoff,
    )
    if not batch_id:
        require(
            "回复「确认」、「执行」继续" in replies[-1],
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
    batch = await ctx.next_client.get_admin_batch(
        batch_id=batch_id,
        admin_token=ctx.admin_token,
    )
    require(
        batch.get("status") in {"Submitted", "Approved"},
        f"S16 batch did not pass through submission: {batch}",
    )
    expected_items = (("载流", "zhlq"), ("载流子", "zlzu"))
    for word, code in expected_items:
        require(
            _submitted_item(batch, word=word, code=code) is not None,
            f"S16 submitted batch lacks {word}@{code}: {batch}",
        )
    return {
        "messages": messages,
        "replies": replies,
        "draft": await ctx.draft(),
        "facts": {
            "submittedWords": [word for word, _code in expected_items],
            "submittedCodes": [code for _word, code in expected_items],
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
        "该词可自动通过",
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
    if not semantic_batch_id and "回复「确认」、「执行」继续" in replies[-1]:
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
    assert_reply_mentions(obscure_reply, "龘季", "该词需管理员审核")
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
    if not obscure_batch_id and "回复「确认」、「执行」继续" in replies[-1]:
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


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("S1", "cold eviction default", scenario_s1),
    Scenario("S2", "explicit duplicate", scenario_s2),
    Scenario("S3", "back placement", scenario_s3),
    Scenario("S4", "stale confirmation", scenario_s4),
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
