"""Scenario pack v1 with end-state and reply-marker assertions."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .recording import ArtifactRecorder
from .runtime import E2EBotHarness, LocalNextClient
from .safety import EncodeDelayController, SafetyViolation


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


@dataclass
class ScenarioContext:
    scenario_id: str
    attempt: int
    identity: dict[str, str]
    next_client: LocalNextClient
    bot: E2EBotHarness
    recorder: ArtifactRecorder
    encode_delay: EncodeDelayController
    fixture_facts: dict[str, Any]

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
    reply = await ctx.send("把吃席同码放在赤溪前面")
    draft = await ctx.draft()
    require(len(draft["items"]) == 1, f"S2 created more than one draft item: {draft}")
    item = draft["items"][0]
    require(item_key(item) == ("Create", "吃席", "wkxk"), f"unexpected S2 item: {item}")
    weight = item.get("weight")
    require(
        isinstance(weight, int) and not isinstance(weight, bool) and weight < 100,
        f"S2 duplicate weight is not ahead of 100: {weight}",
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
        "messages": ["把吃席同码放在赤溪前面"],
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


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("S1", "cold eviction default", scenario_s1),
    Scenario("S2", "explicit duplicate", scenario_s2),
    Scenario("S3", "back placement", scenario_s3),
    Scenario("S4", "stale confirmation", scenario_s4),
    Scenario("S5", "self-service convergence", scenario_s5),
    Scenario("S6", "injection controls", scenario_s6),
    Scenario("S7", "timeout retry", scenario_s7),
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
