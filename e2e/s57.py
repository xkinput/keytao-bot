"""S57: exact same-code ordering, offered answer binding, and additional codes."""

from typing import Any
from urllib.parse import urlsplit

from .scenarios import (
    ScenarioContext, require, item_key, _assert_s56_advertised_reply_closure,
)


async def scenario_s57(ctx: ScenarioContext) -> dict[str, Any]:
    from keytao_bot.harness.conversation import ConversationAddress
    from keytao_bot.harness.state import PendingAddWord, PendingToolConfirm, server_warning_ticket_is_complete
    from keytao_bot.plugins import chat_routing as routing
    from keytao_bot.utils.offered_options import structural_option_questions

    chat = ctx.bot.openai_chat
    store = chat.conversation_state_store
    address = ConversationAddress.group("qq", str(ctx.bot._group_id(ctx.platform_id)), ctx.platform_id)
    messages: list[str] = []
    replies: list[str] = []
    facts: dict[str, Any] = {"closure": []}
    fixture_words = ("嘢", "咽", "春暖花开")

    def cutoff() -> int:
        return max((int(event.get("sequence") or 0) for event in ctx.attempt_events()), default=0)

    def model_calls(after: int) -> list[dict[str, Any]]:
        return [event for event in ctx.attempt_events() if int(event.get("sequence") or 0) > after and (
            event.get("kind") == "modelExchange"
            or (event.get("kind") == "http" and urlsplit(str(event.get("url") or "")).hostname == urlsplit(chat.OPENAI_BASE_URL).hostname)
        )]

    async def closure(reply: str) -> None:
        start = cutoff()
        result = await _assert_s56_advertised_reply_closure(
            reply, chat=chat, record=store.get_record(address), address=address,
            read_draft=ctx.draft,
        )
        require(not model_calls(start), "S57 advertised-string closure called the model")
        if result:
            facts["closure"].append(result)

    async def send(text: str) -> str:
        messages.append(text)
        reply = await ctx.send_group(text, to_me=True)
        replies.append(reply)
        await closure(reply)
        return reply

    async def reset() -> None:
        await ctx.next_client.clean_draft(ctx.platform_id)
        await ctx.bot.reset_conversation(platform_id=ctx.platform_id)

    async def cleanup() -> dict[str, Any]:
        return await ctx.next_client.remove_rig_owned_dictionary_words(
            platform_id=ctx.platform_id, admin_token=ctx.admin_token,
            scenario_id="S57", fixture_words=fixture_words,
        )

    async def exact_dictionary() -> list[tuple[str, str, str, Any]]:
        rows = []
        for word in ("嘢", "咽"):
            rows.extend(
                (row["word"], row["code"], row["type"], row.get("weight"))
                for row in await ctx.next_client.phrases_by_word(word)
                if row.get("word") == word
            )
        return sorted(rows)

    await reset()
    require((await cleanup()).get("verified") is True, "S57 fixture cleanup failed")
    try:
        await ctx.next_client.seed_phrase(platform_id=ctx.platform_id, word="咽", code="yeoiav", phrase_type="Single", weight=10)
        await ctx.next_client.seed_phrase(platform_id=ctx.platform_id, word="嘢", code="yeoiav", phrase_type="Single", weight=11)
        await reset()
        initial = await exact_dictionary()
        require(initial == [("咽", "yeoiav", "Single", 10), ("嘢", "yeoiav", "Single", 11)], f"S57 initial dictionary differs: {initial}")
        start = cutoff()
        preview = await send('提高"嘢    yeoiav"的优先级，使之位于"咽    yeoiav"之前')
        record = store.get_record(address)
        require(record is not None and isinstance(record.state, PendingToolConfirm) and server_warning_ticket_is_complete(record.state), f"S57 reorder lacks complete ticket: {preview}; {record}")
        require("常用度提示：咽 更常用，仍按你的要求执行" in preview and "11→10" in preview and "10→11" in preview, f"S57 incomplete reorder preview: {preview}")
        require(not structural_option_questions(preview) and not (await ctx.draft()).get("items"), f"S57 reorder asked another question or wrote early: {preview}")
        for command in ("强制调序", "确认", "好", "强制", "硬来", "硬加", "强行", "就按我说的", "就这样", "照做"):
            intent = await chat._classify_message_command_intent(command, record.state)
            require(intent.intent == "pending_confirm" and routing._message_authorizes_pending_state_control(record.state, command, intent), f"S57 force/assent failed parser and binder: {command}; {intent}")
        receipt = await send("强制调序")
        draft = await ctx.draft()
        changes = sorted((item_key(row), row.get("type"), row.get("weight")) for row in draft.get("items", []))
        require(changes == [(("Change", "咽", "yeoiav"), "Single", 11), (("Change", "嘢", "yeoiav"), "Single", 10)], f"S57 force did not swap weights: {draft}; {receipt}")
        require(all(word in receipt for word in ("嘢", "咽")) and "已" in receipt and not model_calls(start), f"S57 reorder used model or lacks truthful receipt: {receipt}")
        require(await exact_dictionary() == initial, "S57 draft operation changed dictionary before local approval")
        facts["reorder"] = {"preview": preview, "receipt": receipt, "draft": draft, "modelExchanges": 0}
        start = cutoff()
        extra = await send('添加单字"嘢"，编码为"yeoia"')
        extra_draft = await ctx.draft()
        items = extra_draft.get("items") or []
        added = [row for row in items if item_key(row) == ("Create", "嘢", "yeoia")]
        require(
            extra_draft["batchId"] == draft["batchId"] and len(items) == 3 and len(added) == 1
            and added[0].get("type") == "Single" and added[0].get("needsManualReview") is True and added[0].get("weight") == 10
            and sorted((item_key(row), row.get("type"), row.get("weight")) for row in items if row not in added) == changes,
            f"S57 continuous explicit extra code did not preserve weights and add one sealed Single: {extra}; {extra_draft}",
        )
        require("yeoia" in extra and "复核" in extra and not any(marker in extra for marker in ("也/耶", "只核到了音码", "无法据此")), f"S57 extra-code receipt is inaccurate: {extra}")
        require(not model_calls(start), "S57 continuous extra-code request called the model")
        require(await exact_dictionary() == initial, "S57 extra code modified dictionary before local approval")
        facts["extraCode"] = {"receipt": extra, "draft": extra_draft}
        await ctx.next_client.submit_batch(platform_id=ctx.platform_id, batch_id=extra_draft["batchId"], content_version=extra_draft["contentVersion"])
        await ctx.next_client.approve_admin_batch(batch_id=extra_draft["batchId"], admin_token=ctx.admin_token, review_note="S57 local fixture weight and extra-code verification")
        reordered = await exact_dictionary()
        require(reordered == [("咽", "yeoiav", "Single", 11), ("嘢", "yeoia", "Single", 10), ("嘢", "yeoiav", "Single", 10)], f"S57 local dictionary weights/additional code differ: {reordered}")
        facts["reorder"]["approvedLocalDictionary"] = reordered

        await reset()
        existing = await send("嘢")
        require("已在词库" in existing and "加入编码 <code>" in existing, f"S57 existing lookup omits extra-code path: {existing}")
        start = cutoff()
        await send("加入编码yeoiau")
        contextual = await ctx.draft()
        require(len(contextual.get("items", [])) == 1 and item_key(contextual["items"][0]) == ("Create", "嘢", "yeoiau"), f"S57 contextual extra-code path did not bind: {contextual}")
        require(not model_calls(start), "S57 contextual extra-code request called the model")
        facts["contextualExtraCode"] = contextual
        for command in ("给 嘢 加一个码 yeoiau", "嘢 也放到 yeoiau", "添加单字「嘢」，编码为「yeoiau」"):
            await reset()
            start = cutoff()
            variant_reply = await send(command)
            variant_draft = await ctx.draft()
            variant_items = variant_draft.get("items", [])
            require(len(variant_items) == 1 and item_key(variant_items[0]) == ("Create", "嘢", "yeoiau") and variant_items[0].get("type") == "Single" and variant_items[0].get("needsManualReview") is True, f"S57 explicit variant did not write the bound sealed item: {command}; {variant_reply}; {variant_draft}")
            require(not model_calls(start), f"S57 explicit variant called the model: {command}")
            facts.setdefault("explicitVariants", []).append({"command": command, "reply": variant_reply, "draft": variant_draft})

        # Only option aliases are synthetic; operands and changes come from
        # the actual server-sealed plan, and answers use the real chat path.
        from copy import deepcopy
        from keytao_bot.utils.offered_options import option_questions_bind_live_state
        for answer, executes in (("保留现顺序", False), ("按此排序", True)):
            await reset()
            option_preview = await send("嘢 和 咽 调换顺序")
            live = store.get_record(address)
            require(live is not None and server_warning_ticket_is_complete(live.state), "S57 option control lacks actual sealed plan")
            bound_state = deepcopy(live.state)
            bound_state.args["_offered_options"] = {"按此排序": "confirm", "保留现顺序": "cancel"}
            require(store.set(address, bound_state), "S57 option aliases could not persist")
            option_reply = option_preview + "\n按此排序，还是保留现顺序？"
            require(option_questions_bind_live_state(option_reply, bound_state), "S57 option fixture did not bind")
            require(chat._enforce_advertised_reply_contract(option_reply, address) == option_reply, "S57 valid options refused at delivery")
            await closure(option_reply)
            require(chat.add_to_history(address, "选择调序方案", option_reply), "S57 preceding bot option turn was not recorded")
            ctx.inject_bot_message(option_reply)
            start = cutoff()
            answer_reply = await send(answer)
            option_draft = await ctx.draft()
            expected = [(("Change", "咽", "yeoiav"), 10), (("Change", "嘢", "yeoiav"), 11)] if executes else []
            require(sorted((item_key(row), row.get("weight")) for row in option_draft.get("items", [])) == expected and not model_calls(start), f"S57 offered non-force answer did not bind: {answer}; {answer_reply}; {option_draft}")
            facts.setdefault("offeredOptions", []).append({"question": option_reply, "answer": answer, "reply": answer_reply, "draft": option_draft, "modelRequests": 0})

        await reset()
        quoted_preview = await send("嘢 和 咽 调换顺序")
        quoted_id = ctx.last_reply_message_id
        require(quoted_id is not None, "S57 actual preview lacks a quoteable message ID")
        start = cutoff()
        messages.append("好")
        quoted_receipt = await ctx.send_group_reply("好", reply_message_id=quoted_id, to_me=True)
        replies.append(quoted_receipt)
        await closure(quoted_receipt)
        quoted_draft = await ctx.draft()
        require(sorted((item_key(row), row.get("weight")) for row in quoted_draft.get("items", [])) == [(("Change", "咽", "yeoiav"), 10), (("Change", "嘢", "yeoiav"), 11)] and not model_calls(start), f"S57 native quote assent did not execute exact weight plan: {quoted_receipt}; {quoted_draft}")
        facts["nativeQuoteAssent"] = {"preview": quoted_preview, "reply": quoted_receipt, "draft": quoted_draft, "modelRequests": 0}

        await reset()
        control = await send("春暖花开")
        genuine = store.get_record(address)
        require(genuine is not None and isinstance(genuine.state, PendingAddWord) and genuine.state.word == "春暖花开" and not (await ctx.draft()).get("items"), f"S57 genuine four-character query lost discovery: {control}; {genuine}")
        facts["genuineWordQuery"] = {"word": genuine.state.word, "reply": control}

        await reset()
        synthetic = "请问是要强制调序，还是保留现顺序？"
        token = chat._current_turn_message.set('提高"嘢    yeoiav"的优先级，使之位于"咽    yeoiav"之前')
        try:
            refused = chat._enforce_advertised_reply_contract(synthetic, address)
        finally:
            chat._current_turn_message.reset(token)
        require(refused != synthetic and not structural_option_questions(refused) and "未写入" in refused and store.get_record(address) is None, f"S57 no-ticket option question escaped: {refused}")
        facts["boundaryRefusal"] = {"input": synthetic, "output": refused}
        no_ticket = await send("强制调序")
        require("审词" not in no_ticket and "引用" in no_ticket and not (await ctx.draft()).get("items"), f"S57 recordless force became a word query: {no_ticket}")
        facts["noTicketForce"] = no_ticket
    finally:
        await reset()
        await ctx.next_client.clean_submitted_batches(ctx.platform_id)
        facts["cleanup"] = await cleanup()
    return {"messages": messages, "replies": replies, "facts": facts, "draft": await ctx.draft()}
