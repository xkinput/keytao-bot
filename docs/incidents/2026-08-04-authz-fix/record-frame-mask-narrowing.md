# Record-frame bracket mask narrowing

## Current trade-off

`_mask_quoted_record_frames()` currently masks every supported bracketed span before
`_RECORD_FRAME_RE` runs. This intentionally treats bracket contents as user data, but
the mask is wider than the entry-operand slot: framing words inside brackets are hidden
even when they annotate the user's own bare command. In forms such as a bracketed
“记录一下” next to a bare delete command, the record-frame annotation can therefore be
suppressed and the command can reach the ordinary confirmation prompt.

This is a bounded classification false negative, not an injection path. The text still
comes from the current user's raw message, operand binding is unchanged, and the
confirmation gate still prevents silent execution.

## Why the naive narrowing is not safe to ship

The proposed naive change only treats bracket contents as data when a literal `词条`
appears near the bracket. The review corpus that produced the earlier quantitative claim
was not retained in this repository, so this document makes no numeric claim from it.
The underlying failure is directly checkable from these counterexamples instead:

- `删除【记录】`
- `把【保留记录】加入草稿`
- `将（写入笔记）删除`
- `顺延《登记》到 jlu`
- `添加〈写入笔记〉 jlu`

Each is a direct entry mutation and each deliberately omits the literal `词条`. A rule
that requires that literal near the bracket therefore cannot recognize these operand
slots. The current implementation accepts all five through
`message_authorizes_mutation`; they can be checked offline with:

```bash
.venv/bin/python - <<'PY'
from keytao_bot.harness.tools import message_authorizes_mutation

commands = (
    "删除【记录】",
    "把【保留记录】加入草稿",
    "将（写入笔记）删除",
    "顺延《登记》到 jlu",
    "添加〈写入笔记〉 jlu",
)
assert all(message_authorizes_mutation(command) for command in commands)
PY
```

A correct narrowing needs an explicit, complete grammar for every bracketed
entry-operand slot, not a keyword proxy. At minimum that design must enumerate and test:

- verb-initial operands, including direct add/delete/change/shift forms;
- `把` / `将` constructions where the operand precedes the mutation verb;
- record-shaped entry names that overlap mutation tokens, such as `保留记录` or
  `写入笔记`;
- draft-container deletion forms and postposed/colon operand forms;
- all supported bracket pairs, command lead-ins, optional `词` / `词条` decoration,
  and optional codes.

The allow corpus must then be checked together with the existing record-framing block
corpus so that narrowing the mask neither revives reported-command authorization nor
repeats the legitimate-command false positives.

## Decision

Do not narrow the mask in this follow-up. Keep the current bounded annotation cost and
treat complete entry-verb-slot enumeration plus bidirectional corpus design as separate
design work.
