"""S58: move-to-code incident replay and structured proposal safety."""

import json
import re
from typing import Any
from urllib.parse import urlsplit

from .scenarios import (
    ScenarioContext, _assert_s56_advertised_reply_closure, item_key, require,
)


S58_WORD = "奈飞"
S58_OCCUPANT = "浓厚氛围"
S58_ORIGINAL_CODE = "nhfwv"
S58_TARGET_CODE = "nhfw"
S58_SHIFTED_CODE = "nhfwa"
S58_EXPLICIT_DESTINATION = "nhfwav"
S58_MANUAL_DESTINATION = "nhfwzz"
S58_DESTINATION_OCCUPANT = "浓厚风味"
S58_PLAN = "把 奈飞 调到 nhfw，把 浓厚氛围 挪到 nhfwa"
S58_GAP_COMMAND = "把 奈飞 迁到 nhfw"
S58_BANNED_COPY = (
    "顺延操作的词条或目标编码未精确绑定",
    "回复中的操作说法没有可验证的服务端绑定记录，已移除",
    "缺少明确的执行动词",
    "这条消息没有明确的执行指令",
    "判定为「缺少明确的执行指令」",
)


def _expected_items(
    destination: str = S58_SHIFTED_CODE,
    extra_moves: tuple[tuple[str, str, str], ...] = (),
) -> list[tuple[str, str, str]]:
    items = [
        ("Delete", S58_WORD, S58_ORIGINAL_CODE),
        ("Create", S58_WORD, S58_TARGET_CODE),
        ("Delete", S58_OCCUPANT, S58_TARGET_CODE),
        ("Create", S58_OCCUPANT, destination),
    ]
    for word, previous, target in extra_moves:
        items.extend((("Delete", word, previous), ("Create", word, target)))
    return sorted(items)


def _structured_shift_calls(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Read actual provider responses without treating model prose as operands."""
    calls = []
    for event in events:
        if event.get("kind") != "modelExchange":
            continue
        response = event.get("response") or {}
        if not isinstance(response, dict):
            continue
        for choice in response.get("choices") or []:
            for call in (choice.get("message") or {}).get("tool_calls") or []:
                function = call.get("function") or {}
                if function.get("name") != "keytao_shift_phrase_code":
                    continue
                arguments = function.get("arguments")
                try:
                    arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                except (TypeError, ValueError):
                    continue
                if isinstance(arguments, dict):
                    calls.append({"sequence": event["sequence"], "arguments": arguments})
    return calls


def _assert_s58_manual_destination_seal(
    plan: dict[str, Any], created: dict[str, Any], preview: str, destination: str,
) -> None:
    """Bind persisted pronunciation and review flags to the actual sealed item."""
    from keytao_bot.utils.keytao_review import normalize_pinyin_sequence

    sealed = next(row for row in plan["items"] if item_key(row) == ("Create", S58_OCCUPANT, destination))
    remark = str(created.get("remark") or "")
    reading_match = re.search(r"读音\s+([^；;\n]+)", remark)
    saved_reading = normalize_pinyin_sequence(reading_match.group(1)) if reading_match else ()
    sealed_reading = normalize_pinyin_sequence(str(sealed.get("_reviewed_pinyin") or ""))
    require(
        sealed_reading == saved_reading == ("nong", "hou", "fen", "wei")
        and sealed.get("needsManualReview") is True
        and created.get("needsManualReview") is True
        and "形码 zz 未能核验" in str(sealed.get("manualReviewReason") or "")
        and "形码 zz 未能核验" in remark
        and "复核" in preview,
        f"S58 explicit unknown suffix lost its exact reviewed reading/manual seal: {preview}; {sealed}; {created}",
    )


async def scenario_s58(ctx: ScenarioContext) -> dict[str, Any]:
    from keytao_bot.harness.conversation import ConversationAddress
    from keytao_bot.harness.state import PendingToolConfirm, server_warning_ticket_is_complete
    from keytao_bot.harness.tools import ToolContext
    from keytao_bot.plugins import chat_routing as routing
    from keytao_bot.utils.pending_confirmation import advertised_command_suggestions

    chat = ctx.bot.openai_chat
    store = chat.conversation_state_store
    address = ConversationAddress.group("qq", str(ctx.bot._group_id(ctx.platform_id)), ctx.platform_id)
    messages: list[str] = []
    replies: list[str] = []
    facts: dict[str, Any] = {"cases": [], "closure": []}
    fixture_words = (S58_WORD, S58_OCCUPANT, S58_DESTINATION_OCCUPANT)

    def cutoff() -> int:
        return max((int(event.get("sequence") or 0) for event in ctx.attempt_events()), default=0)

    def events_after(sequence: int) -> list[dict[str, Any]]:
        return [event for event in ctx.attempt_events() if int(event.get("sequence") or 0) > sequence]

    def model_calls(sequence: int) -> list[dict[str, Any]]:
        return [event for event in events_after(sequence) if (
            event.get("kind") == "modelExchange"
            or (event.get("kind") == "http" and urlsplit(str(event.get("url") or "")).hostname == urlsplit(chat.OPENAI_BASE_URL).hostname)
        )]

    async def closure(reply: str) -> None:
        sequence = cutoff()
        bound = await _assert_s56_advertised_reply_closure(
            reply, chat=chat, record=store.get_record(address), address=address,
            read_draft=ctx.draft,
        )
        require(not model_calls(sequence), "S58 advertised parser/binding closure called the model")
        if bound:
            facts["closure"].append(bound)

    async def send(text: str, *, to_me: bool = True, quote: int | None = None) -> str:
        messages.append(text)
        reply = await (
            ctx.send_group_reply(text, reply_message_id=quote, to_me=to_me)
            if quote is not None else ctx.send_group(text, to_me=to_me)
        )
        replies.append(reply)
        require(not any(copy in reply for copy in S58_BANNED_COPY), f"S58 retained retired copy: {reply}")
        await closure(reply)
        return reply

    async def reset() -> None:
        result = await ctx.next_client.clean_draft(ctx.platform_id)
        require(result.get("success") is True, f"S58 draft cleanup failed: {result}")
        await ctx.bot.reset_conversation(platform_id=ctx.platform_id)

    async def cleanup() -> dict[str, Any]:
        result = await ctx.next_client.remove_rig_owned_dictionary_words(
            platform_id=ctx.platform_id, admin_token=ctx.admin_token,
            scenario_id="S58", fixture_words=fixture_words,
        )
        require(result.get("verified") is True, f"S58 fixture cleanup failed: {result}")
        return result

    async def dictionary_snapshot() -> list[tuple[str, str, str, Any]]:
        rows = []
        for word in fixture_words:
            rows.extend(
                (row["word"], row["code"], row["type"], row.get("weight"))
                for row in await ctx.next_client.phrases_by_word(word)
                if row.get("word") == word
            )
        return sorted(rows)

    def live_plan(
        destination: str = S58_SHIFTED_CODE,
        extra_moves: tuple[tuple[str, str, str], ...] = (),
    ) -> dict[str, Any]:
        record = store.get_record(address)
        require(
            record is not None and record.owner_key == address and not record.execution_id
            and isinstance(record.state, PendingToolConfirm)
            and record.state.function_name == "keytao_shift_phrase_code"
            and server_warning_ticket_is_complete(record.state),
            f"S58 expected one complete actor-owned shift ticket: {record}",
        )
        display = record.state.args.get("_pending_display") or {}
        plan = display.get("shiftPlan")
        require(isinstance(plan, dict), f"S58 ticket lacks its server plan: {record.state.args}")
        require(
            sorted(item_key(row) for row in plan.get("items") or []) == _expected_items(destination, extra_moves),
            f"S58 ticket changed the requested move or omitted an occupant: {plan}",
        )
        return plan

    async def assert_assent_closure() -> None:
        state = store.get_record(address).state
        sequence = cutoff()
        for command in ("确认调整", "确认执行", "确认", "执行", "好", "强制", "硬来", "照做"):
            intent = await chat._classify_message_command_intent(command, state)
            require(
                intent.intent == "pending_confirm"
                and routing._message_authorizes_pending_state_control(state, command, intent),
                f"S58 live assent failed the real parser/binder: {command}; {intent}",
            )
        require(not model_calls(sequence), "S58 live assent closure called the model")

    async def execute_case(
        command: str, *, destination: str = S58_SHIFTED_CODE,
        to_me: bool = True, assent: str = "确认", quoted: bool = False,
        grammar_gap: bool = False,
        reviewed_suffix: bool = False,
        extra_moves: tuple[tuple[str, str, str], ...] = (),
    ) -> dict[str, Any]:
        await reset()
        sequence = cutoff()
        before = await ctx.draft()
        preview = await send(command, to_me=to_me)
        preview_events = events_after(sequence)
        plan = live_plan(destination, extra_moves)
        quote_id = ctx.last_reply_message_id
        require(await ctx.draft() == before, f"S58 proposal wrote before assent: {command}; {preview}")
        require(all(value in preview for value in (S58_WORD, S58_OCCUPANT, S58_ORIGINAL_CODE, S58_TARGET_CODE, destination)), f"S58 preview omitted a live movement: {preview}")
        await assert_assent_closure()
        if grammar_gap:
            calls = _structured_shift_calls(preview_events)
            require(
                any(call["arguments"].get("word") == S58_WORD and call["arguments"].get("target_code") == S58_TARGET_CODE for call in calls),
                f"S58 grammar-gap control did not receive real structured model arguments: {calls}",
            )
            logs = [str(event.get("message") or "") for event in preview_events if event.get("kind") == "log"]
            require(any("[grammar_gap]" in line and "authorization=False" in line for line in logs), f"S58 control was not a real grammar gap: {logs}")
            require(any("grammar_gap_bridged=True" in line for line in logs), f"S58 real structured gap was not recorded as bridged: {logs}")
            shapes = (
                ("请明确回复一句执行指令（例如「确认调整 / 执行」）", ("确认调整", "执行"), False),
                (f"例如：\n执行：{S58_PLAN}", (f"执行：{S58_PLAN}",), False),
                ("比如「执行：把 外部词 调到 zzzz」", ("执行：把 外部词 调到 zzzz",), True),
                ("可发送：\n执行：把 外部词 调到 zzzz", ("执行：把 外部词 调到 zzzz",), True),
            )
            for model_copy, expected_commands, must_redraw in shapes:
                advertised = advertised_command_suggestions(model_copy)
                require(all(value in advertised for value in expected_commands), f"S58 structural advertisement escaped detection: {model_copy}; {advertised}")
                token = chat._current_turn_message.set(command)
                try:
                    redrawn = chat._enforce_advertised_reply_contract(model_copy, address)
                finally:
                    chat._current_turn_message.reset(token)
                if must_redraw or redrawn != model_copy:
                    require(
                        all(value in redrawn for value in (S58_WORD, S58_OCCUPANT, S58_ORIGINAL_CODE, destination))
                        and not any(copy in redrawn for copy in S58_BANNED_COPY),
                        f"S58 live-plan redraw lost its ticketed preview: {model_copy}; {redrawn}",
                    )
                await closure(redrawn)
                live_plan(destination)
                require(await ctx.draft() == before, "S58 advertisement redraw changed the draft")
                facts.setdefault("advertisementShapes", []).append({"input": model_copy, "detected": advertised, "output": redrawn})
        else:
            calls = []
            if not reviewed_suffix:
                require(not model_calls(sequence), f"S58 closed command used a model request: {command}")
        preview_model_requests = len([event for event in preview_events if event.get("kind") == "http" and event.get("isLlm")])
        confirmation_sequence = cutoff()
        require(not quoted or quote_id is not None, "S58 native quote control has no actual preview message ID")
        receipt = await send(assent, quote=quote_id if quoted else None)
        draft = await ctx.draft()
        require(
            sorted(item_key(row) for row in draft.get("items") or []) == _expected_items(destination, extra_moves)
            and all(row.get("type") == "Phrase" for row in draft.get("items") or []),
            f"S58 confirmation did not execute its entire exact plan: {command}; {draft}; {receipt}",
        )
        require(all(value in receipt for value in (S58_WORD, S58_OCCUPANT, S58_TARGET_CODE, destination)) and "✅" in receipt, f"S58 completion lacks the exact shift receipt: {receipt}")
        if not reviewed_suffix:
            require(not model_calls(confirmation_sequence), "S58 ticket assent called the model")
        for extra_word, _previous, extra_target in extra_moves:
            require(extra_word in preview and extra_target in preview and extra_word in receipt and extra_target in receipt, f"S58 omitted the extra displaced occupant: {preview}; {receipt}")
        if reviewed_suffix:
            created = next(row for row in draft["items"] if item_key(row) == ("Create", S58_OCCUPANT, destination))
            _assert_s58_manual_destination_seal(plan, created, preview, destination)
        require(await dictionary_snapshot() == initial_dictionary, "S58 draft operation changed the live dictionary")
        result = {"command": command, "toMe": to_me, "preview": preview, "plan": plan, "assent": assent, "quoted": quoted, "receipt": receipt, "draft": draft, "structuredModelCalls": calls, "grammarGapBridged": grammar_gap, "reviewedSuffix": reviewed_suffix, "previewModelRequests": preview_model_requests, "confirmationModelRequests": len([event for event in events_after(confirmation_sequence) if event.get("kind") == "http" and event.get("isLlm")])}
        facts["cases"].append(result)
        return result

    await reset()
    await cleanup()
    try:
        for word, code in ((S58_WORD, S58_ORIGINAL_CODE), (S58_OCCUPANT, S58_TARGET_CODE)):
            await ctx.next_client.seed_phrase(platform_id=ctx.platform_id, word=word, code=code, phrase_type="Phrase", weight=100)
        await reset()
        initial_dictionary = await dictionary_snapshot()
        require(initial_dictionary == sorted(((S58_WORD, S58_ORIGINAL_CODE, "Phrase", 100), (S58_OCCUPANT, S58_TARGET_CODE, "Phrase", 100))), f"S58 fixture does not match the incident: {initial_dictionary}")
        for code in (S58_SHIFTED_CODE, S58_EXPLICIT_DESTINATION, S58_MANUAL_DESTINATION):
            occupied = [row for row in await ctx.next_client.phrases_by_code(code) if row.get("code") == code and row.get("type") == "Phrase"]
            require(not occupied, f"S58 requires its declared destination to be empty: {code}; {occupied}")

        await execute_case("喵喵 把 奈飞 调到 nhfw", to_me=False)
        await execute_case(S58_PLAN, assent="确认调整")
        await execute_case(f"执行：{S58_PLAN}", assent="确认执行", quoted=True)
        await execute_case(f"把 奈飞 调到 nhfw；把 浓厚氛围 挪到 {S58_EXPLICIT_DESTINATION}", destination=S58_EXPLICIT_DESTINATION, assent="执行")
        await execute_case(f"把 奈飞 调到 nhfw，把 浓厚氛围 挪到 {S58_MANUAL_DESTINATION}", destination=S58_MANUAL_DESTINATION, reviewed_suffix=True)

        await reset()
        await ctx.next_client.seed_phrase(platform_id=ctx.platform_id, word=S58_DESTINATION_OCCUPANT, code=S58_SHIFTED_CODE, phrase_type="Phrase", weight=100)
        occupant_encode = await ctx.next_client.encode(S58_DESTINATION_OCCUPANT)
        occupant_codes = occupant_encode.get("codes") or []
        require(S58_SHIFTED_CODE in occupant_codes and occupant_codes.index(S58_SHIFTED_CODE) + 1 < len(occupant_codes), f"S58 occupied-destination fixture lacks its actual successor: {occupant_encode}")
        occupant_next = occupant_codes[occupant_codes.index(S58_SHIFTED_CODE) + 1]
        require(not [row for row in await ctx.next_client.phrases_by_code(occupant_next) if row.get("code") == occupant_next and row.get("type") == "Phrase"], f"S58 occupied-destination successor must be empty: {occupant_next}")
        initial_dictionary = await dictionary_snapshot()
        await execute_case(S58_PLAN, extra_moves=((S58_DESTINATION_OCCUPANT, S58_SHIFTED_CODE, occupant_next),))
        await reset()
        occupant_cleanup = await ctx.next_client.remove_rig_owned_dictionary_words(
            platform_id=ctx.platform_id, admin_token=ctx.admin_token, scenario_id="S58",
            fixture_words=(S58_DESTINATION_OCCUPANT,),
        )
        require(occupant_cleanup.get("verified") is True, f"S58 occupied-destination fixture cleanup failed: {occupant_cleanup}")
        initial_dictionary = await dictionary_snapshot()
        await execute_case(S58_GAP_COMMAND, grammar_gap=True, assent="强制")

        # This explicit boundary supplement supplies structured arguments to
        # the real executor. It does not replace the model in a chat turn.
        for message, arguments in (
            (S58_GAP_COMMAND, {"word": S58_OCCUPANT, "target_code": S58_TARGET_CODE}),
            ("把 奈飞 迁到 nhfw1", {"word": S58_WORD, "target_code": "nhfw1"}),
            ("把 从未入库虚构词 迁到 nhfw", {"word": "从未入库虚构词", "target_code": S58_TARGET_CODE}),
        ):
            await reset()
            sequence = cutoff()
            before = await ctx.draft()
            raw = await chat.tool_executor.call(
                "keytao_shift_phrase_code", arguments,
                ToolContext(platform="qq", user_id=ctx.platform_id, current_message=message, writes_allowed=False),
            )
            result = json.loads(raw) if isinstance(raw, str) else raw
            require(
                isinstance(result, dict) and result.get("success") is not True
                and result.get("grammar_gap_bridged") is not True
                and not result.get("shiftPlan") and not result.get("requiresConfirmation")
                and not result.get("needs_confirmation")
                and store.get_record(address) is None and await ctx.draft() == before,
                f"S58 unverifiable structured arguments acquired a proposal or write: {message}; {arguments}; {result}",
            )
            require(not model_calls(sequence), "S58 structured boundary supplement called the model")
            facts.setdefault("invalidStructuredBoundary", []).append({"message": message, "arguments": arguments, "result": result, "draftUnchanged": True, "syntheticBoundaryOnly": True})

        await reset()
        first_recordless_reply = None
        for command in ("确认调整", "确认执行", "确认", "执行", "好"):
            sequence = cutoff()
            no_ticket = await send(command)
            require(
                store.get_record(address) is None and not (await ctx.draft()).get("items")
                and len([line for line in no_ticket.splitlines() if line.strip()]) == 1
                and not model_calls(sequence),
                f"S58 recordless assent was not a truthful deterministic one-line reply: {command}; {no_ticket}",
            )
            if first_recordless_reply is None:
                first_recordless_reply = no_ticket
            require(no_ticket == first_recordless_reply, f"S58 recordless assent reinterpreted a refusal as a proposal: {no_ticket}")
            facts.setdefault("noTicketAssent", []).append({"command": command, "reply": no_ticket})

        await reset()
        prior_preview = await send("把 奈飞 调到 nhfw")
        live_plan()
        store.delete(address)
        sequence = cutoff()
        expired_reply = await send("确认调整")
        require(
            S58_WORD in expired_reply and S58_TARGET_CODE in expired_reply
            and len([line for line in expired_reply.splitlines() if line.strip()]) == 1
            and store.get_record(address) is None and not (await ctx.draft()).get("items")
            and not model_calls(sequence),
            f"S58 lost-ticket assent did not truthfully name its last proposal: {expired_reply}",
        )
        facts["lostTicketAssent"] = {"priorPreview": prior_preview, "reply": expired_reply, "ticketRemovalIsSynthetic": True}
    finally:
        await reset()
        facts["cleanup"] = await cleanup()
    return {"messages": messages, "replies": replies, "facts": facts, "draft": await ctx.draft()}
