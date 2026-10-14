# Routing eval label spec v1 (frozen)

Context: 喵喵 is a QQ/Telegram group bot that maintains the KeyTao (键道) Chinese input-method dictionary.
Users look up words, add words with codes (lowercase a-z codes like `wjekuo`), change codes, reorder candidates,
delete entries, manage a personal draft batch and submit it for review. When the bot offers candidate codes it
keeps a pending ticket; the user's next message may answer it (pick 1/2/3 or a code, "加入", "加入并提交", "取消").

Input row: {id, ts, space, user, message, prev_bot_reply}. `prev_bot_reply` is the bot's previous reply in the same
chat (any user) — use it ONLY to judge whether `message` answers a pending ticket. A leading bot name
(喵喵 / @喵喵) is just addressing, ignore it.

Output: one JSON object per input row, same order, written as JSONL:

{"id": <int>, "op": <enum>, "operands": [<words in the order the user gave>], "codes": [<lowercase codes the user
 named>], "submit_after": <bool>, "refers_to_pending": <bool>, "frame": <enum>, "confidence": "high"|"low",
 "note": "<short, only when low confidence or noteworthy, else empty>"}

op enum (pick exactly one; what the user WANTS done this turn):
- query            look up one or more words/codes, or ask about a word's code/frequency/reading
- chat             no dictionary task (greeting, thanks, banter, emoji, talking to other humans)
- help             asks how to use the bot / how to phrase a request (e.g. "应该怎么说")
- add              add new word(s), optionally with code(s)
- select_candidate answer a pending ticket by choosing an index or code (may include add/submit wording)
- confirm          bare assent to a pending ticket/plan ("加入", "好", "确认", "加入并提交" with no new selection)
- cancel           decline/cancel the pending ticket
- pending_recode   ask the pending ticket for a different/other code ("换码", "换一个码")
- swap             swap two existing entries' codes/positions/priority
- delete           delete an existing entry
- recode           change an existing entry to a different code
- reorder          change candidate order on a code / chain (incl. "X 优先于 Y", "提到前面")
- move             move/insert an entry to a specific code position, replace what is at a code
- batch_replace    replace a character across a listed set of codes
- draft_view | draft_submit | draft_recall | draft_clear | draft_keep_only
- operation_recall ask what operations the bot recently performed
- clear_history    reset chat history
- other            a real task not covered above (describe in note)

frame enum (how the operation is expressed — this is the safety signal):
- direct    the sender is instructing the bot now
- question  asks whether/how something could be done ("能调换吗？"), not instructing
- negated   says NOT to do it ("不要删")
- reported  quotes/relays someone else's words or a past message, hypothetical, example
- none      op is query/chat/help

Rules:
- Judge the user's intent the way a human group member would, not by keyword. Typos and mixed verbs
  ("调整调换 A 和 B 词条") still count as the intended op.
- `submit_after` true only if the user asks to also submit ("并提交", "加入并提交").
- `refers_to_pending` true only if the message answers the ticket in prev_bot_reply. A new, complete command
  that happens to mention a ticket word is NOT a ticket answer.
- Multiple ops in one message: label the primary write op, put the rest in note, confidence "low".
- Do not invent operands; only words/codes literally in the message (or, for select/confirm, the ticket's word
  if the message omits it — then copy it from prev_bot_reply).
