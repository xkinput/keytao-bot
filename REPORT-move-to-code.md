# S58 — Move-to-code grammar and structured proposal bridge

## Scope and acceptance status

Baseline: clean `9909dbf42a7aacdd57f69ec5811e1dee214f8cb6`.
All changes are local and uncommitted. No staging, commit, push, deployment,
production host/API/data access, or `pnpm test` is authorized or performed.
The local opt-in rig uses localhost Next/PostgreSQL and the explicitly
configured `https://api.deepseek.com` / `deepseek-v4-flash`, without model or
provider fallback. This report is an execution ledger until the final gate
below is populated; targeted checks do not close the monolithic gate.

Current closure blocker: the exact DeepSeek endpoint returned HTTP 402
`Insufficient Balance` during targeted S58 invocation 26. No accepted FULL
exists. The eight rejected FULL invocations remain failed evidence; a fresh
S1–S58 invocation is still required after the test account becomes available.

Current local Next HEAD is `1e421aae905cc201d617f347bd96b97bbbeff534`.
Its only pre-existing dirty file, `pnpm-workspace.yaml`, has SHA-256
`c54f79df5b0c8375ec0d16da1141b2cd6ea91e039c4b4e923cc59e23e1c41287`.
Read-only preflight validated PostgreSQL `localhost:5432`, Next
`localhost:3100`, and the existing loopback proxy `127.0.0.1:7890`.
Secrets were checked for presence and supplied through process environment;
their contents are absent from the ledger. The Next child retains its
outbound proxy confinement to `127.0.0.1:9`.

## Incident diagnosis

The incident is confirmed, with one correction to the suspected first-turn
cause: the keyword is a normalization defect, but is not the sole reason for
`binding_incomplete`. The original real parser and binder still failed after
removing `喵喵`, and also with `@喵喵`.

At baseline, `调到` expressed positive positional intent, so the first message
could receive write authorization. It was absent from the narrower
`_EXISTING_ENTRY_MOVE_RE`, however, so `parse_existing_entry_move` returned
`None`. The binding check therefore fell through to
`_positional_destination_is_bound`, which required the requested code to be
among the target's server-known codes. A word lookup returned existing
`奈飞@nhfwv`; it did not establish candidate `nhfw` in the reviewed code map.
Literal `奈飞` plus literal `nhfw` consequently did not satisfy that alternate
capability path. No model misunderstanding is needed to explain the failure.

The new shared normalization site runs before `trusted_mutation_source`
feeds either parsing or binding. It removes keyword/mention prefixes and
closed imperative wrappers, while retaining quoted/reported/negative/question
guards. The closed existing-item parser now binds the exact requested word
and target code against a current existing identity. The live planner still
owns pronunciation, target-code validation, occupancy, and displacement.
The positional fallback's candidate-provenance requirement was not weakened.

The original unknown-verb reproduction also revealed that `迁到` did not
even set the gap metric: `authorization=False`, `gap=False`, planner calls
zero, and a `verb_not_matched` result. Gap detection now recognizes bounded
positive destination syntax without granting execution authority.

## Command and ticket behavior

The move lexicon covers `调到 / 调至 / 挪到 / 挪至 / 移到 / 移至 / 换到 /
放到 / 放在 / 改到 / 调整到`, as well as `X 用 C` and `把 X 换成 C`.
`把 / 将` are optional in the closed forms. Quotes, extra spaces and
full-width/half-width punctuation normalize at the same source boundary.
Optional named `并顺延 / 挤掉 / 顶替` clauses retain actual occupant binding.

`执行：`, `执行 `, `请执行 `, `帮我 ` and `麻烦 ` wrap a complete command.
Two literal moves separated by comma, semicolon, `、`, `和` or `并且` form one
`EntryMovePlanCommand`; the second destination is retained in `expected_codes`
through the existing ranked shift planner. A model's first-only shift call
cannot discard the second clause. There is no independent move algorithm.

A second explicit destination outside the ordinary candidate list reuses S56's
`validate_explicit_code` against a freshly reviewed reading inventory. A valid
phonetic prefix with an unverified shape suffix is retained exactly and sealed
as `needsManualReview=True`, with the reviewed pinyin and reason on the created
draft row. Missing, ambiguous, unresolved or blocked reading evidence rejects
the preview. Confirmation recomputes the same plan and rejects reading drift;
an occupied explicit destination uses the existing displacement chain.

Closed move-to-code requests intentionally preview first. They do not take
the existing positional single-occupant auto-confirm shortcut. `确认调整`,
`确认执行`, `确认`, `执行`, `好`, native quote-confirmation and the S57 force
lexicon consume the current actor-owned ticket through the existing parser,
binding check, claim and sealed replay. No-ticket assent is deterministic,
one line, and names the previous displayed proposal when available; history
supplies display context only and cannot resurrect a mutation.

## Structured grammar-gap bridge

The executor handles eligible grammar gaps as proposals. The original
structured tool operands, not model prose, must be literal in the current
trusted text. Internal digests, batch anchors, review capabilities and extra
mutation operands cannot be supplied by the model to this path. Attachments,
questions, reported instructions, negation, protected targets and protected
cascade hints cannot enter the proposal lookup.

For a shift, the executor performs an exact live word lookup and requires a
unique existing typed row. It invokes the registered shared shift planner
without a confirmation digest. The planner reads encoding, occupancy and the
current draft, obtains its strict preview, and returns a complete plan,
warning digest and content-version binding. Only that response can produce
`grammar_gap_bridged=True`. The orchestrator persists the actor-owned
`PendingToolConfirm` before rendering and exits the model loop immediately.
No auto-confirmation or subsequent model-prose command runs in this turn.
Failed validation instead returns a deterministic truthful line and no ticket.

The bridge is a registry of existing non-writing preview protocols, not a
permission to invoke arbitrary mutation tools. Besides shift, it handles
single and batch dictionary Create/Change/Delete proposals and single/batch
draft-item removal. Create/Change require an existing server-minted reviewed
capability; dictionary identities and actor-owned draft targets are read live.
Draft removal seals exact ordered targets, batch and content version. Model
arguments cannot supply any of those internal capabilities. Partial, skipped,
collision-replanned or incompletely sealed previews cannot produce a ticket.
At the grammar-gap model boundary, the raw batch schema's `old_word` is validated before conversion to the server
planner's `oldWord`; accepting model-supplied canonical keys would bypass the
public argument contract and is rejected.

Three of the eight public mutation tools remain outside this adapter and
return a truthful no-write result without dispatch. This is an explicit
boundary, not a claim that every mutation tool now bridges:

- `keytao_update_draft_item_weight`: no preview/confirmed interface; the current
  implementation proceeds to PATCH (`keytao-draft/tools.py:3675`, `:3793`).
- `keytao_recall_batch`: omitting `batch_id` selects a latest-batch GET preview,
  while supplying one enters the version/claim/POST path (`:2894`, `:3001`,
  `:3070`, `:3113`). The item adapter must not infer an operation batch.
- `keytao_submit_batch`: it **does** have `preview_only`; unlike the item
  adapter it first audits the entire current draft and seals a separate audit,
  snapshot and warning capability (`:2399`, `:2523`, `:2596`;
  `keytao_bot/harness/state.py:510`, `:587`).
  This lifecycle-specific adapter was not added in this incident round.

Rebuttal to an unrestricted interpretation of C: literal operands alone do not
justify invoking the first two mutation interfaces as previews, or expanding
an item-shaped gap to whole-batch submission. The requested move/shift bridge
and the five supported preview interfaces are implemented. Invalid operands
also cannot yield a pre-validated working command; C's truthful one-line reply
takes precedence over inventing an executable advertisement under E.

Independent review caught and fixed an initially misspelled lookup tool name,
loss of raw model arguments before bridge validation, and a protected-cascade
hint reaching lookup. Negative tests also reject missing plan/digests,
foreign words/codes, model-supplied confirmation data, and absent live rows.
The final review additionally caught action-blind batch previews and loss of
saved batch reading capabilities at confirmation. Rendering now names every
Delete/Change operation and its old/new identities. Complete server-warning
batch tickets restore reviewed capabilities by `(word, code)`; confirmed
Create/Change calls retain exact pinyin, candidates, type and manual flags,
while Delete items receive no reading capability. Existing top-level review
protocols keep their original precedence. Raw model fields remain untrusted.

## Advertisement and retired copy

The advertisement detector uses the structure of a suggested reply rather
than an action-verb allowlist: quoted strings and lines introduced by
`例如 / 比如 / 回复 / 发送 / 请回复 / 可发送`, and lines beginning `执行：`.
It handles the incident's slash-separated assent alternatives, colon/newline
form and ASCII/full-width quotes. Extracted strings must pass the real parser
and the applicable live-state binder.

A complete shift ticket takes precedence over model-authored advertisements:
redraw renders that same sealed server plan, including when model prose names
a wrong destination. State-dependent mutation or assent advertisements require
the applicable live record and binding; closed read-only commands such as
`查看草稿` retain their existing parser-backed route. The banned
authorization-jargon class is checked at final delivery, with a ticketed
preview or plain truthful cause replacing it.

Existing-word placeholder instructions are explanatory formats rather than
advertised executable strings. A multiple-code context record no longer
advertises the unique-location `换码` operation. The record still supports
the S57 explicit additional-code path.

## Verification ledger

Checks and exit metadata are under `/private/tmp/keytao-s58-checks`.
The six offline suites run in separate Python processes. Changes discovered
after any run require source-matched final checks; old tails are not relabeled.

| Invocation | Result |
|---|---|
| Targeted 1: `20260908T025610Z-e3fc0d65` | Rejected. All five move cases, actual structured-model bridge, exact four-row drafts, advertisements and invalid-argument controls passed; lost-ticket response omitted the word/code by recalling only the preview heading. Both attempts failed that assertion. The display-context renderer was corrected. |
| Targeted 2: `20260908T030030Z-27b1af5e` | Rejected. A verification process imported files during a concurrent in-progress two-file copy change and saw an absent `_RETIRED_AUTHORIZATION_COPY_RE`; both attempts failed before model calls or operation execution. Later runs require a stable source handoff. |
| Targeted 3: `20260908T031607Z-980988eb` | Rejected. The new explicit `nhfwzz` case wrote the correct manual-review row, but its assertion assumed tone-marked pinyin rather than the server's actual reading format. Investigation also found a real presentation bug: compacting ranked-plan evidence removed the manual-review notice from the preview. The renderer now emits that notice separately from the sealed item's manual reason; the E2E notice assertion is retained. Both attempts failed; this run is not accepted as S58 evidence. |
| Targeted 4: `20260908T033130Z-f6804afd` | Rejected. Explicit-destination validation and the six-row third-occupant displacement passed, but the ranked preview displayed only the two requested words and omitted the third occupant; the receipt correctly included it. The renderer now supplements the ranked rows from the same sealed `shifted` collection. A focused reproduction failed before this fix; the full 44-test S58 set passes afterward. |
| Targeted 5: `20260908T033341Z-c5df476e` | PASSED, attempt 1, exit 0; superseded after output inspection. All then-current assertions passed (19.2 scenario seconds, 2 model exchanges, 24,499 tokens; process 03:33:41.802–03:34:21.487 UTC). Reading the full artifact revealed that repeated no-ticket assent recursively described prior refusals as proposals, and ordinary shift previews listed unlabeled old/new items separately. The no-ticket renderer is now stable across repeats and ignores ordinary chat; the shift renderer now pairs sealed source/destination codes. New assertions retain these output requirements for the final run. |
| Targeted 6: `20260908T034108Z-348836df` | PASSED, attempt 1, exit 0 on the `final` source snapshot, before the punctuation compatibility adjustment. Scenario: 26.362 seconds, 3 real model exchanges, 49,567 tokens. All seven cases and the strengthened output assertions passed. The actual provider's shift operands were exactly `{"word":"奈飞","target_code":"nhfw"}`; the unknown verb remained unauthorized and produced a proposal ticket. |
| Targeted 8 / S4: `20260908T040154Z-465ab830` | PASSED, attempt 1, exit 0 after the FULL1 fix. The stale-advertisement refusal preserves its expiry-or-missing cause; the subsequent exact deletion previews once and executes on one confirmation. 13.7 scenario seconds, 3 real model exchanges, 48,363 tokens. |
| Targeted 9 / S58: `20260908T040239Z-6f78451e` | PASSED, attempt 1, exit 0 after the FULL1 fix. All seven cases and all negative/advertisement/no-ticket controls remain green. 30.1 scenario seconds, 3 real model exchanges, 29,849 tokens. |
| Targeted 10 / S19: `20260908T042451Z-6f6c7d26` | PASSED, attempt 1, exit 0 after the FULL2 advertisement fix. The original nine-word subset confirmation and execution assertions passed. 63.7 scenario seconds, 10 real model exchanges, 52,194 tokens. |
| Targeted 11 / S58: `20260908T042651Z-c87c44a3` | PASSED, attempt 1, exit 0 on the same frozen source after the FULL2 fix. All seven cases and negative controls passed. 26.3 scenario seconds, 3 real model exchanges, 29,440 tokens. |
| Targeted 12 / S21: `20260908T045626Z-635ba319` | PASSED, attempt 1, exit 0 after the FULL3 fix. Original precedence guidance, unrelated-text no-write control and exact two-word confirmation all passed. 60.2 scenario seconds, 17 real model exchanges, 129,924 tokens. |
| Targeted 13 / S47: `20260908T045810Z-721049cd` | Rejected, both attempts failed, exit 1. The actual conflict producer prefixes the canonical A/B block with a server-derived conflict reason; the newly strict whole-text comparison rejected that valid result. The earlier matrix supplied only the pure block and missed this integration difference. The real producer path is now the required reproduction, and the original S47 assertions remain unchanged. |
| Targeted 14 / S47: `20260908T050504Z-47e629f4` | PASSED, attempt 1, exit 0. The conflict reason is now saved with the A/B ticket and rendered from that record. Original conflict-choice, cleanup and subsequent displacement assertions passed. 11.2 scenario seconds, 2 real model exchanges, 1,182 tokens. |
| Targeted 15 / S58: `20260908T050630Z-dea4444d` | PASSED, attempt 1, exit 0 on `acceptance5`. All seven cases and controls passed. 30.4 scenario seconds, 3 real model exchanges, 29,850 tokens. |
| Targeted 16 / S27: `20260908T053612Z-ad59863a` | PASSED, attempt 1, exit 0 after the FULL4 fix. Exactly one unbound notice, working binding help, no bound-user notice and zero-tool meta answer all satisfy the original scenario. 30.3 scenario seconds, 7 real model exchanges, 27,134 tokens. |
| Targeted 17 / S58: `20260908T053734Z-b9b700af` | PASSED, attempt 1, exit 0 on `acceptance6`. All seven cases and controls passed. 29.7 scenario seconds, 3 real model exchanges, 29,486 tokens. |
| Targeted 18 / S34: `20260908T062020Z-ee00c464` | PASSED, attempt 1, exit 0 after the FULL5 fix. The original pending-query fact, duplicate confirmation/cancellation and different-code paths passed. 31.0 scenario seconds, 8 real model exchanges, 3,251 tokens. |
| Targeted 19 / S58: `20260908T062407Z-30918e68` | PASSED, attempt 1, exit 0 on `acceptance7`. All seven cases and controls passed. 30.6 scenario seconds, 4 real model exchanges, 34,192 tokens. |
| Targeted 20 / S27: `20260908T065125Z-c0c9f708` | PASSED, attempt 1, exit 0 after the FULL6 detector fix. The unchanged binding-notice, remediation and zero-tool meta-answer assertions passed. 30.1 scenario seconds, 7 real model exchanges, 27,236 tokens. |
| Targeted 21 / S58: `20260908T065314Z-09ba2d73` | PASSED, attempt 1, exit 0 on `acceptance8`. All seven cases and controls passed. 26.4 scenario seconds, 3 real model exchanges, 29,459 tokens. |
| Targeted 22 / S39: `20260908T072902Z-0955c04a` | PASSED, attempt 1, exit 0 after the FULL7 fix. The original selected-reading, candidate guidance and later compound-operation assertions passed. 20.0 scenario seconds, 5 real model exchanges, 3,073 tokens. |
| Targeted 23 / S46: `20260908T073005Z-902cce10` | PASSED, attempt 1, exit 0. Both the incident and reproduced plan retain their exact item/write invariants, and confirmation/cancellation advertisements pass the real parser/binder. 5.6 scenario seconds, zero model exchanges/tokens. |
| Targeted 24 / S58: `20260908T073122Z-dd813506` | PASSED, attempt 1, exit 0 on `acceptance9`. All seven cases and controls passed. 21.0 scenario seconds, 2 real model exchanges, 24,499 tokens. |
| Targeted 25 / S45: `20260908T081318Z-c0d0230d` | PASSED, attempt 1, exit 0 after the FULL8 fix. The actual provider again produced an example sentence, the sanitizer logged `preserved_reply=true`, and the unchanged character/tool/no-review/no-write assertions passed. 13.3 scenario seconds, 3 real model exchanges, 42,032 tokens. |
| Targeted 26 / S58: `20260908T081419Z-ac1f991e` | Rejected, both attempts failed, exit 1. Each attempt's command router and main-model request returned HTTP 402 `Insufficient Balance`; no successful model exchange or structured unknown-verb proposal was possible. The assertion failed on the truthful service-error reply instead of a ticket. Process 08:14:18.893–08:15:20.873 UTC; this is an external-provider blocker, not a passing control. |

Evidence boundaries for S58: the seven move cases use the ordinary QQ message
path. In targeted 11, all three real provider exchanges occur in the unknown
`迁到` preview, including one actual structured shift call with exact operands;
the six known-command previews and all confirmations use zero model calls.
The four advertisement-shape controls inject synthetic prose into the real
delivery gate while the actual live ticket exists. The three invalid-argument
controls call the real executor directly; they are marked
`syntheticBoundaryOnly` and do not substitute for the positive provider turn.
The lost-ticket control deliberately deletes the saved record and is marked
`ticketRemovalIsSynthetic`. Advertisement closure covers 24 parser/binding
checks across eight actual previews and four synthetic redraws, in addition
to the per-case assent alias checks. Each completed case verifies exact
Pending draft rows and an unchanged live dictionary snapshot; S58 does not
submit those rows into the dictionary.

Initial state suite: `1998/2019`, 21 failures. These are recorded as failures,
not baseline passes. No-ticket ordinary-chat expectations conflict with S58 B;
the actual no-tool assertions are retained and the new one-line/zero-model
behavior is required. Advertisement-related failures are reviewed separately
for product regressions versus incomplete synthetic ticket fixtures.

The initial memory suite ran all 407 tests and failed 12 methods; targeted
reproductions preserved their original mutation/closure assertions while fixing
the live advertisement handling and completing synthetic server-ticket inputs.
The first complete rerun (`final1`) failed five subcases in three methods:
two required nonexistent `2,4` candidate indices, and three required an
advertised command after a rejected grammar-gap proposal without reviewed
inputs or a preview adapter. Those tests now assert no ticket/no dispatch in
that branch and still verify renderer-generated commands through the original
validator. `final2` passed all seven requested commands. `final3` exposed five
existing punctuation assertions in paired shift previews; restoring the
established colon/arrow spacing preserves those assertions and the newly
required explicit movement pairs. The five exact failed methods plus all S58
modules pass together: 51 tests (`focused-final6.log`). The `final4` offline
invocation preceded one fallback arrow-space adjustment; subsequent complete
offline invocations and source fingerprints are recorded below.

Final implementation evidence:

| Contract | Source |
|---|---|
| Shared keyword/wrapper normalization; closed move grammar | `keytao_bot/harness/authorization_grammar.py:2272`, `:1669`, `:1729` |
| One two-clause plan and exact complete binding | `keytao_bot/harness/authorization_grammar.py:1331`, `:5618` |
| Structured executor proposal bridge and preview registry | `keytao_bot/harness/tools.py:1537`, `:1642` |
| Record before rendering and immediate model-loop termination | `keytao_bot/harness/orchestrator.py:2426` |
| Reviewed explicit destinations using the existing planner | `keytao_bot/skills/keytao-draft/tools.py:5459`, `:5622` |
| Restore sealed batch readings by exact word/code | `keytao_bot/plugins/chat_commands.py:786` |
| Structural advertised-command extraction; preserve independent facts when stripping | `keytao_bot/utils/pending_confirmation.py:2205`, `keytao_bot/harness/orchestrator.py:281` |
| Reviewed candidate assent with an exact resolved-set constraint when present | `keytao_bot/harness/orchestrator.py:220` |
| Action-aware and source/destination-paired sealed previews | `keytao_bot/plugins/chat_render.py:664`, `:695`, `:762`, `:790` |
| Retain persisted candidate reading and manual-review facts | `keytao_bot/plugins/chat_render.py:677`, `keytao_bot/plugins/openai_chat.py:2291` |
| Nonrecursive no-ticket display recovery | `keytao_bot/utils/offered_options.py:153` |
| Actor/task-local display binding fact; reset before new lookup | `keytao_bot/utils/user_resolver.py:20`, `:80` |
| Restore authoritative notice only after successful advertisement closure | `keytao_bot/plugins/openai_chat.py:2523` |
| Complete duplicate request and matching Submitted identity | `keytao_bot/harness/state.py:396`, `:426`, `:437` |
| Canonical duplicate request display, including mixed batches | `keytao_bot/plugins/chat_render.py:896` |

Independent review is same-model review, not cross-model validation. Its
15 generic replay tests and 15 raw-field forgery probes passed against local
strict simulated sinks. Final S58 focused execution runs the five new modules
in one process; integration and live-provider claims depend on the rig below.

Final offline tails (`/private/tmp/keytao-s58-checks/*-final13.log`):

| Literal command | Tail | Exit |
|---|---|---|
| `.venv/bin/python test_state_machine.py` | `Results: 2019/2019 passed, 0 failed` | 0 |
| `.venv/bin/python test_memory_safety.py` | `Ran 407 tests in 201.421s` / `OK` | 0 |
| `.venv/bin/python test_security_fixes.py` | `Results: 268/268 passed, 0 failed` | 0 |
| `.venv/bin/python test_review_gate.py` | `Results: 443/443 passed` / `ALL TESTS PASSED` | 0 |
| `.venv/bin/python test_llm_policy.py` | `Ran 10 tests in 0.101s` / `OK` | 0 |
| `.venv/bin/python test_word_discovery.py` | `Results: 290/290 passed` / `ALL TESTS PASSED` | 0 |
| `.venv/bin/python -m e2e.test_safety` | `Ran 107 tests in 0.739s` / `OK` | 0 |

An earlier focused command combined the five exact memory-suite failures with
`test_s58_grammar test_s58_bridge test_s58_explicit_destination
test_s58_generic_bridge test_s58_advertisement`: `Ran 51 tests in 0.232s`, `OK`,
exit 0 (`focused-final6.log`). There are 46 S58 tests within that invocation.
After the FULL2 fix, the five S58 modules pass 49 tests together
(`focused-acceptance3.log`), and all 21 S54–S57 modules pass 168 tests in
separate processes. Exact commands, tails, exits and unchanged source hashes
are in `focused-adjacent-acceptance3.json`.
`git diff --check` is clean. No full repository command outside the user's
prescribed suites was substituted for them; `pnpm test` was never run.

Immediately before the last attempted FULL (FULL8), the 138 source/config/test
fingerprints matched `source-freeze-acceptance9.json` at 07:33:41.158 UTC.
The subsequent S45 sanitizer correction is frozen separately in
`/private/tmp/keytao-s58-checks/source-freeze-acceptance10.json`; `final13` and
the 62-test focused run apply to that final snapshot. Report/temporary
run-ledger helpers are excluded from the fingerprints so evidence can be
recorded. FULL9 has not started because the configured provider is unavailable.

## Accepted monolithic FULL

| Acceptance item | Current result |
|---|---|
| Accepted fresh S1–S58 invocation | None; gate remains open |
| Rejected FULL invocations | 8; listed below with their first failed attempts |
| Latest source-matched offline commands | All 7 passed (`final13`) |
| Latest source-matched S58 focused modules | 62 tests passed |
| Required remaining execution | S58 replay, then a fresh FULL S1–S58 with all 58 attempt-1 passes and exit 0 |
| External blocker | DeepSeek HTTP 402 `Insufficient Balance` |

All rejected FULL invocations below are retained as failed evidence. Their
passing scenarios are never combined to satisfy the acceptance table.

FULL invocation 1 (`20260908T035028Z-dcb16406`) is rejected. It ran
03:50:28.481–03:52:16.434 UTC and was interrupted at the first failed attempt
(exit 130). S1, S2 and S3 passed on attempt 1; S4 failed because the new
recordless-assent stage intercepted an expired-ticket confirmation and omitted
the existing expiry marker. The original S4 assertion remains required.
Log/exit metadata: `/private/tmp/keytao-s58-checks/full-s1-s58-run1`.

The S4 fix preserves the original uncertainty: its rig fixture contains only
a prior confirmation advertisement, not proof of TTL expiry. The one-line
recordless response now says the confirmation expired **or** its record no
longer exists when the prior advertisement actually parsed as assent. Normal
absence does not imply expiry. Native replies, live operations and eligible
other-owner pending contexts keep their existing scoped arbitration. Actual
expired-store and history-only stage regressions pass, as do the prior 24
stale-confirmation pipeline assertions. S57's existing `将把 ...` proposal
shape also remains recognizable for display-only recovery.

All 21 `test_s54_*` through `test_s57_*` modules (168 tests) passed in independent
`.venv/bin/python -m unittest <module>` processes; exact commands and log paths
are recorded in `neighbor-exits.json`. The S58 modules now contain 48 tests
and pass together (`focused-after-s4.log`). An attempted targeted
`--only S4,S58` was safely rejected by the single-scenario CLI (exit 2, no
scenario execution); it is not a verification pass.

After the fix, all seven prescribed commands passed again (`final5`), and
all 138 fingerprints matched `source-freeze-acceptance2.json` immediately
before FULL2 at 2026-09-08 04:05:49.657 UTC.

FULL invocation 2 (`20260908T040610Z-603062d5`) is rejected. It ran
04:06:10.413–04:16:43.535 UTC and stopped at S19 attempt 1 (exit 130).
S1–S18 passed on attempt 1. S19 correctly persisted the selected nine-word
subset, but the final advertisement gate redrew its confirmation into the
generic candidate-selection footer, losing the required `确认` affordance.
This was a confirmation-display regression, not a restored excluded-word set;
the original scenario assertion is retained. Log/exit prefix:
`full-s1-s58-run2`. No result from either rejected invocation can close the gate.

The S19 root cause was a mismatch between the real confirmation renderer and
the new advertisement validator: `_append_pending_ticket_challenge` appended
`确认 / 取消` to an already CAS-saved resolved word set, while the validator's
local-preview branch recognized only add-style replies. The first correction accepted
the existing assent parser only when `_resolved_advertised_words` exactly
matches the selected Create items, every candidate scope/code is valid, and
all displayed bindings equal the saved bindings. Ordinary candidate previews
did not gain this capability at that stage; S21 later exposed valid legacy
reviewed candidate tickets, addressed below. The new regression uses the real advertised-set
store/CAS, renderer, ticket challenge, final delivery gate and parser/binder;
it checks nine included words, two exclusions, zero first-turn writes, and
rejection of missing/claimed records, changed sets, unverified codes and extra
unbound commands. The original S19 E2E assertion was not changed.

The source-matched `final6` offline invocation completed all seven prescribed
commands successfully at 04:28:53.446 UTC. Independent review also exercised
one positive and 17 negative resolved-set cases, with zero tool calls and
unchanged source fingerprints (`resolved-assent-review-acceptance3.log`).
FULL3 started from S1 after those checks; no prior scenario artifacts were
reused as its evidence.

FULL invocation 3 (`20260908T042903Z-480b3261`) is rejected. It ran
04:29:03.207–04:41:07.066 UTC and stopped at S21 attempt 1 (exit 130).
S1–S20 passed on attempt 1, including both previously failing scenarios.
S21's exact two-word reviewed candidate ticket remained present, but the final
advertisement gate replaced the existing pending-operation guidance with a
generic candidate footer. This removed its single coherent confirmation block
when `提交草稿` was sent while that ticket existed. The original S21 assertions
remain required, including its unrelated-text no-write control and subsequent
exact-set confirmation. Log/exit prefix: `full-s1-s58-run3`.

The S21 investigation distinguishes legacy reviewed candidate tickets from
resolved-set tickets. The real legacy producer seals same-turn review items
and candidate scopes into the actor's `PendingToolConfirm` without a
`_resolved_advertised_words` marker. That marker therefore cannot be required
for every valid candidate confirmation. The required binding is the live,
unclaimed actor record, exact displayed pairs, valid complete candidate scopes
and saved review fields; a present resolved-set marker remains an additional
exact-set constraint. Model prose still cannot populate these records.

A parallel renderer matrix also found that a `PendingAddWord` containing an
already-existing code and additional choices advertised bare `换码`, which only
the trusted-word context parser supports. Its valid candidate-number/code
choices must remain available without that unsupported extra advertisement.
The same matrix found a complete nested-plan A/B offer being parsed as the
single advertised string `A 或 B 即可`, then receiving an invalid generic
`确认 / 取消` challenge. Its advertisements must bind A and B through the actual
choice parser and nested sealed plans. Existing S47 already executes this
cleanup-choice path and requires the original `回复 A 或 B 即可` copy; that
scenario remains an unchanged integration check.

The final correction uses the real legacy reviewed-ticket producer and the
existing parser/binder. The S21 regression fails before the correction and
passes afterward; generic candidate confirmation remains `preview_only=True`
at the executor boundary. Candidate renderers now suppress unsupported bare
context actions while retaining their bound selection commands. A/B detection
splits the complete letter-alternative form, validates exact canonical output
and complete nested plans, and adds no unsupported assent challenge.
The five S58 modules pass 52 tests (`focused-acceptance4.log`). The 21 S54–S57
modules pass 168 tests in separate processes (`neighbor-acceptance4.json`).
The original 30-case renderer matrix now passes all 28 valid records and
rejects two incomplete synthetic records; supplemental producer-derived
records verify the legacy case. The initial supplemental script's exact-string
assertion failed because the equivalent candidate footer was normalized; its
failed log is retained and not represented as a pass. Final matrix/source
evidence is under `renderer-matrix-acceptance4.*`,
`renderer-producer-detail-acceptance4.log` and `choice-ab-final-acceptance4.log`.

The `final7` offline invocation passed six commands but failed one of 407
memory tests: the candidate-copy adjustment inserted `下列` inside the existing
`仍可选择其他编码` phrase. This is recorded as a failed run; restoring the
compatible plain phrase requires no change to its original test or to the
new prohibition on unsupported bare context actions.

The actual S47 conflict producer now saves `_choice_reason` with the complete
A/B ticket. Its canonical renderer emits that persisted server explanation
and original options together. Follow-up reminders and unbound model commands
redraw from the live record; the exact-text comparison and nested-plan checks
remain intact. The new regression runs the real `_execute_confirmed_tool`
conflict path through persistence and final delivery, including protected-add
and forged-command controls. On the resulting `acceptance5` snapshot, the five
S58 modules pass 53 tests (`focused-acceptance5.log`).
All seven prescribed commands pass on the same source (`final8`, completed
05:08:56.721 UTC). Targeted S47 and S58 both pass on attempt 1 with process
exit 0, without changing their original integration assertions.
The final independent review passed the two actual-producer/label regressions
and ten A/B controls with zero tool calls and unchanged source fingerprints.
It covered missing/foreign/claimed records, missing/incomplete/all-empty plans,
invalid reasons and redraw of forged prefixes (`s47-choice-producer-review-acceptance5.log`,
`choice-ab-review-acceptance5.log`, `choice-review-acceptance5-source.json`).

FULL invocation 4 (`20260908T051025Z-1df6b291`) is rejected. It ran
05:10:25.789–05:25:36.565 UTC and stopped at S27 attempt 1 (exit 130).
S1–S26 passed on attempt 1, including the previous S4/S19/S21 failure points.
S27's unbound-user candidate content was valid, but final advertisement redraw
removed its mandatory single account-binding precheck notice. Its original
notice-count, bound-user, binding-remediation and conversational meta-question
assertions remain required. Log/exit prefix: `full-s1-s58-run4`.

S27 has two connected causes: the old notice advertises the placeholder
`/bind 绑定码` as if it were an executable command, and a valid candidate redraw
then drops the earlier authoritative binding notice. The full unbound-account
help has the same placeholder example. Binding guidance must describe the
actual profile-page/code requirement without advertising an unfilled command.
Final candidate delivery must preserve a notice based on this turn's actual
actor binding lookup, never on model-supplied notice text or an older result.
The original hard-coded notice expectation in the state suite is updated to
the shared constant while retaining its binding/absence/count assertions.

The final delivery wrapper strips model-supplied notice text, runs the existing
advertisement gate, then appends the notice only for the same actor's current
verified `False` binding result and an unclaimed, parser/binder-valid live
record. The resolver stores an immutable, display-only task-local tuple;
the inbound turn resets it and each new lookup clears it to `None` before
awaiting. Failures remain unknown, and child tasks cannot overwrite their
parent's result. This fact is not read by mutation authorization and adds no
network request or persistent cache. Independent tests cover real discovery
with a simulated binding lookup, redraw/idempotence, missing/foreign records,
model-forged notices, failed queries, turn reset and parent/child isolation.
Evidence: `s27-notice-before.log`, `s27-notice-focused.log`,
`s27-existing-focused.log`, `s27-original-state-focused.log`,
`s27-binding-review-focused.log` and `s27-binding-review-source.json`.
All five S58 modules pass 55 tests on that final snapshot
(`focused-acceptance6.log`). The seven prescribed offline commands pass again
(`final9`, completed 05:39:57.458 UTC). S27 and S58 targeted replay both pass
on attempt 1 with exit 0 before the fresh FULL5 invocation.

FULL invocation 5 (`20260908T054040Z-89f73b41`) is rejected. It ran
05:40:40.193–06:02:29.065 UTC and stopped at S34 attempt 1 (exit 130).
S1–S33 passed on attempt 1, including S27. The repeated pending-word query
correctly read the actor's Submitted item but appended an unfilled
`添加 开团 <编码>` advertisement and an unbound `撤回提交` advertisement without
creating an executable ticket. The new gate rejected those advertisements and
dropped the valid leading pending fact with them. The producer must describe
the actual-code/withdrawal prerequisites without inventing commands; the
existing S34 exact-duplicate warning and different-code paths remain required.
Log/exit prefix: `full-s1-s58-run5`.

The S34 reproduction also exposed the existing local duplicate-confirmation
path: it saved `_pending_submitted_confirmed` without enough structured
display evidence for the new advertisement binder. Its four existing producers
now retain the tool's validated pending rows and a snapshot of the complete
original request in the existing `_pending_display` field. Canonical display
must bind the unchanged complete request and an exact Submitted Create match;
a mixed batch still displays every requested item. This is the existing local
preview protocol, not a new server-warning digest or execution shortcut.
The original executor still owns the subsequent server preview and replay.
A separate narrow scan found one remaining question fallback advertising
`查词 <字符>`; its unfilled format is replaced by a plain requirement for a
specific character, without inventing a query argument.

Independent S34 review found an important Change-identity gap in the new
duplicate display validator: absent `old_word` or conflicting `old_word` and
`oldWord` could render a different old identity from the executor's argument
precedence. Change now requires a nonempty old identity and exact equality
when both aliases exist; non-Change items reject a valued old identity.
The three producer-derived S34 regressions include those negative controls,
full-request tampering and missing pending facts. Independent review passed
all three with unchanged source fingerprints
(`s34-duplicate-review-focused.log`, `s34-duplicate-review-source.json`).
The five S58 modules pass 58 tests together (`focused-acceptance7.log`).
All seven prescribed commands pass on the same frozen source (`final10`,
completed 06:24:07.098 UTC); targeted S34 and S58 pass before FULL6.

FULL invocation 6 (`20260908T062509Z-95b41fe5`) is rejected. It ran
06:25:08.836–06:41:39.231 UTC and stopped at S27 attempt 1 (exit 130).
S1–S26 passed on attempt 1. The actual provider answered the conversational
account-binding question without tool calls, but the final advertisement gate
replaced that answer with a no-plan refusal. The earlier unbound notice and
binding-help checks had already passed; this is a distinct false-positive
inside the same original S27 meta-question assertion. Log/exit prefix:
`full-s1-s58-run6`. Its passed subset is not acceptance evidence.

The exact 267-character provider response recounted why the earlier
`加入并提交` was blocked. Within the quote's same-sentence lookback,
`_COMMAND_SUGGESTION_LEAD_RE` mistook the noun `当前发送者` for the imperative
lead `发送`. The sanitizer correctly declined to excise an arbitrary inline
clause, but its remaining false advertisement caused the whole answer to be
replaced. Independent reproduction confirmed a detector error, with zero
tool calls in the original question turn. Follow-up review also reproduced
two related past-tense shapes, `你刚才发送了「加入并提交」` and
`上一条回复提到了「加入并提交」`, which require per-lead framing checks rather
than exempting an entire narrative answer from advertisement validation.

The correction excludes the actor noun `发送者` from the bare command lead,
and ignores only the matched lead inside an immediately preceding completed
report fragment. Independent `例如`, `请回复`, `可发送` and execution-line leads
remain active even in the same answer. Conditional instructions such as
`发送了「确认」就会执行` and `回复提到的「确认」即可` still require a ticket.
The exact artifact response is preserved through the orchestrator; the
existing final mechanism-copy filter then renders the ordinary binding-meta
answer, `会的：写入前会校验绑定；未绑定会给出绑定引导。`, with zero tools and no
ticket. Neither the original S27 assertion nor the sanitizer was weakened.
The exact response fixture is byte-for-byte equivalent as decoded text to
the actual model-exchange content. The focused regression and 14 independent
framing controls pass with unchanged source hashes
(`s27-reported-prefix-review-focused.log`,
`s27-reported-prefix-review-source.json`). On `acceptance8`, all five S58
modules pass together: 59 tests (`focused-acceptance8.log`).
All seven prescribed commands pass on the same source (`final11`, completed
06:55:16.201 UTC); targeted S27 and S58 pass before the fresh FULL7.

FULL invocation 7 (`20260908T065545Z-1e1be396`) is rejected. It ran
06:55:44.812–07:19:39.971 UTC and stopped at S39 attempt 1 (exit 130).
S1–S38 passed on attempt 1, including all previous failure points. The
reading-specific request `加词 出圈 圈字读quan` produced the correct candidate
codes but the final advertisement gate's canonical redraw omitted the selected
reading group. The actual review tool had returned `chū quān`; this is a
presentation/evidence-retention regression, not an accepted change to S39's
original requirement. Log/exit prefix: `full-s1-s58-run7`.

The producer's occupied-choice guidance placed a real `回复` instruction and
the later operation object `除权` in the same sentence. The structural
detector consequently treated that object quote as another command. The
canonical `PendingAddWord` redraw then omitted persisted pronunciation and
manual-review fields. The correction must separate the two guidance sentences
and retain the existing structured reading/review fields in canonical output;
no model prose is used to recover them.

The bounded audit of upcoming scenarios also found an obsolete S46 assertion:
it required exactly one `pending_confirmation_copy()` but simultaneously
required no `command_suggestions`. S58 D deliberately detects that copy's
`确认` and `取消`, making the assertions mutually exclusive. The appropriate
replacement permits exactly those two advertised controls and requires the
live ticket's real parser/binder closure. The four plan markers, single preview
call, no-model/no-write preview and exact subsequent execution stay required.
This is an assertion-contract correction, not a weakened operation invariant.

The S39 correction passes its producer-derived regression and eight independent
controls: full/partial/multiple reading groups, foreign-group/prose exclusion,
unknown manual-review status, absent server snapshot, claimed record and
mismatched candidates. Reading evidence is projected only onto persisted
server candidate codes; an explicit manual-review flag retains the saved
reason. Evidence: `s39-reading-before.log`, `s39-reading-after.log`,
`s39-reading-redraw-review-focused.log` and its source fingerprint JSON.
The original S39 integration assertions are unchanged.

The S46 replacement also records each control's real parser/binder closure in
its case facts. A historical server preview plus the actual compound-plan
parser's resolved plan passes the current renderer, final gate and closure
helper with all four movement markers intact. An initial probe lacking the
route's resolved-plan field failed its markers and is an incomplete probe,
not a product defect. That combined exploratory command first observed eleven
archived review-tool results through the real single-word producer and final
gate, covering S43/S53/S55/S56 selected reading/source/manual-review lines
without external requests; it ultimately exited 1 at the incomplete S46 probe
and is not a suite pass. A separate complete S46 reconstruction exited 0.
These transient tool outputs were not retained as log files or used as
source-matched acceptance evidence. The subsequent targeted S46 invocation
above supplies retained actual-service evidence.
On the final `acceptance9` snapshot the five S58 modules pass 60 tests together
(`focused-acceptance9.log`). All seven prescribed commands pass (`final12`,
completed 07:33:06.658 UTC); targeted S39, S46 and S58 pass on attempt 1 with
process exit 0 before the fresh FULL8 invocation.

FULL invocation 8 (`20260908T073341Z-93b5d7d6`) is rejected. It ran
07:33:41.246–08:03:02.912 UTC and stopped at S45 attempt 1 (exit 130).
S1–S44 passed on attempt 1, including S39. The character question
`单人旁加个巨字是什么字` correctly called encoding for `佢` and did not enter
review or write a draft. The actual provider explained its `qú` reading and
used the linguistic example `例如粤语里「佢哋」就是「他们」`. The structural
detector classified those two quoted nouns as advertised commands and replaced
the entire correct answer with a no-plan refusal. The original S45 character,
tool, no-review and no-write assertions remain required. Log/exit prefix:
`full-s1-s58-run8`.

The final S45 correction keeps D's conservative detector unchanged. The shared
sanitizer removes sentences containing unbound advertised strings and retains
independent facts from the same answer. It keeps multiline introducers together
and does not split at punctuation inside quoted commands. A nonempty remaining
answer is retained only after the complete advertisement contract says it no
longer requires live state. Thus the example sentence is removed while the
character/reading and tool-verification paragraphs survive. This also handles
real invalid command advertisements between facts without a linguistic or
question-specific exemption. A valid live proposal still takes the earlier
canonical record-rendering path; no prose creates a ticket or authority.
The exact 101-character provider fixture is retained in the new regression;
controls cover unknown verbs, multiline examples/execution, multiple commands,
quoted punctuation, valid read-only draft viewing and an all-advertisement
answer with no surviving facts.
Independent review passed both new tests and ten framing/remaining-contract
controls, with unchanged source hashes (`s45-sanitizer-review-focused.log`,
`s45-sanitizer-review-source.json`). On `acceptance10`, the five S58 modules
pass together: 62 tests (`focused-acceptance10.log`).

The next targeted S58 replay encountered provider HTTP 402 `Insufficient
Balance`, confirmed in both attempts' HTTP response artifacts. Four failed
model requests are retained in `20260908T081419Z-ac1f991e/S58-attempt-1.json`
and `S58-attempt-2.json`; no provider fallback or further evidence-neutral
manual retry followed the targeted rig's two configured attempts. FULL9 has
not started. Resume requires restoring the
configured test account/key, rerunning S58, verifying the final source
fingerprints, and starting one fresh monolithic S1–S58 invocation from S1.
All seven prescribed offline commands completed successfully on that final
snapshot at 08:17:07.719 UTC (`offline-final13-exits.json`). No ninth FULL was
started and no additional provider request was made after that failed S58
invocation ended. The final 138 fingerprints, unchanged bot/Next HEADs,
empty index and protected Next dirty-file hash matched at 08:18:03.451 UTC;
`git diff --check` passed.
The four owned `.tmp-work/s58-*.py` helpers are retained as resumable local
run/verification utilities; the report, source freeze and `/private/tmp`
execution ledgers record the exact pending work. They have not been staged.

Acceptance is not yet claimed. Closure requires a fresh
`.venv/bin/python -u -m e2e.run` invocation selecting S1–S58, all passing on
attempt 1, with process exit 0 and unchanged frozen source fingerprints.
A failed scenario invalidates the entire invocation; retries inside it cannot
close this gate.
