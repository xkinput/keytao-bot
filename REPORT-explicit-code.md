# S56 — explicit codes, guarded eviction, and completed-operation undo

## Scope and acceptance status

Baseline: clean `b012704af6164438771677e5b499428c0ba56b0b`.
All changes are local and uncommitted. No commit, push, deployment, production
host, production API, or production data access is authorized or performed.
The banned `pnpm test` command is not used. The opt-in E2E rig uses localhost
Next/PostgreSQL and the explicitly configured real
`https://api.deepseek.com` / `deepseek-v4-flash`, without provider fallback.
The read-only Next source is `1e421aae905cc201d617f347bd96b97bbbeff534`;
its pre-existing dirty `pnpm-workspace.yaml` is preserved and fingerprinted.

Acceptance is complete for the local gate. FULL invocation 10,
`e2e/artifacts/20260907T101611Z-453de2dd`, passed S1–S56 in one process:
all 56 scenarios passed on attempt 1, no failed scenario or scenario retry,
process exit 0. The post-run verifier found no drift in the 119 frozen source,
configuration, and prompt files. All six offline suites, 107 E2E safety tests,
and 64 focused regressions also passed on this source. This is local evidence;
the changes remain uncommitted and have not been deployed or production-verified.
The rejected runs below are retained as evidence and are not combined with
the accepted process.
FULL invocation 1 was stopped after S21 attempt 1 failed, artifact
`e2e/artifacts/20260907T065335Z-c6620e28`, log
`/private/tmp/keytao-s56-checks/full-s1-s56-run1.log`.
FULL invocation 2 was stopped after S9 connection errors; its artifact is
`e2e/artifacts/20260907T071126Z-42a60402` and
log is `/private/tmp/keytao-s56-checks/full-s1-s56-run2.log`.
FULL invocation 3 was stopped after S28's obsolete copy assertion. It used
the existing local HTTPS proxy explicitly supplied. Its artifact is
`e2e/artifacts/20260907T071758Z-8df4743b`; its log is
`/private/tmp/keytao-s56-checks/full-s1-s56-run3.log`.
FULL invocation 4 was stopped after S35 lost the default front-insertion
manual seal. Its artifact is
`e2e/artifacts/20260907T075204Z-80e3d1c2`; its log is
`/private/tmp/keytao-s56-checks/full-s1-s56-run4.log`.
FULL invocation 5 was stopped after an independent audit identified incomplete
S56 advertised-string closure coverage. S1–S5 had passed on attempt 1; no
scenario had failed. Its artifact is `e2e/artifacts/20260907T082140Z-a6b3d63a`
and log is `/private/tmp/keytao-s56-checks/full-s1-s56-run5.log`.
FULL invocation 6 was stopped after S39's remaining obsolete generic-compound
remediation expectation failed; S1–S38 passed on attempt 1. Its log is
`/private/tmp/keytao-s56-checks/full-s1-s56-run6.log` and artifact is
`e2e/artifacts/20260907T084600Z-dc95e304`.
FULL invocation 7 was stopped after S27's first meta-answer attempt failed;
its successful second attempt cannot accept the run. Its artifact is
`e2e/artifacts/20260907T091831Z-6b341dea` and log is
`/private/tmp/keytao-s56-checks/full-s1-s56-run7.log`.
FULL invocation 8 was stopped after S22's provider connection failures;
its log is `/private/tmp/keytao-s56-checks/full-s1-s56-run8.log` and
artifact is `e2e/artifacts/20260907T093955Z-169b10b8`.
FULL invocation 9 was stopped after S13 repeated the failed instruction as
remediation. Its artifact is `e2e/artifacts/20260907T100152Z-29303054` and
log is `/private/tmp/keytao-s56-checks/full-s1-s56-run9.log`.

```sh
HTTPS_PROXY=http://127.0.0.1:7890 E2E_ARTIFACT_RETENTION=1000 E2E_OPENAI_API_KEY="$(cat .e2e_key)" E2E_OPENAI_BASE_URL=https://api.deepseek.com E2E_OPENAI_MODEL=deepseek-v4-flash .venv/bin/python -u -m e2e.run
```

## Reproductions and disposition

All three incident diagnoses are confirmed. The initial explicit-code
regression (`.venv/bin/python -m unittest test_s56_explicit_code`) failed seven
assertions: all four direct command forms failed live binding, and all three
invalid-code cases returned the same generic rejection rather than naming the
invalid component. The existing `_handle_pending_add_word` shifted occupied
short codes without a commonness decision. A separate real-comparator
regression observed zero comparison calls for the occupied Single singleton
and an unsafe `qx` recommendation. The no-pending cancellation reproduction
returned `None`, which permitted the general model flow.

## Code-validation rules and trusted capability

The direct forms are `加入编码<code>`, `加入 <code>`, `添加 <word> <code>`,
and `用 <code>`, with optional `并提交`. They consume one entire direct command;
reported instructions, questions, negation, foreign words, and trailing
targets cannot create a code capability. Existing advertised envelopes still
pass through the same complete-command unwrapping and live binding checks.
Explicit numeric values in `加入编码 12` or `添加 鎗 12` are rejected as
invalid characters; the existing ordinal selectors `加入 2` / `用 2` remain
selectors. A focused red/green regression covers this distinction.

For a new Single/Phrase code, the permitted characters are lowercase `a-z`
and the maximum length is six. Its phonetic prefix must belong to the same
persisted reviewed pronunciation. Prefix lengths follow the shared encoder:
Single two keys, two-character word four, three-character word three, and
longer word four. The actual base comes from that reading's trusted inventory,
so the legitimate reviewed `fe*` / `qe*` alternatives are not overwritten by
re-deriving one default reading code. Missing or ambiguous reading evidence
cannot authorize a new suffix.

Wrong prefix, invalid characters, overlength, and an already existing exact
word/code/type identity are objective rejection reasons and are named in the
reply. In particular an identical six-code entry cannot be added twice.
Existing backend code, occupancy, warning-digest, and content-version checks
still run; the new parser does not override a backend duplicate prohibition.
Different words sharing six keys are not categorically invalid: the current
Next `lib/services/conflictDetector.ts:204-226` rejects exact identity,
while `:276-295` permits different-word occupancy with a warning. The existing
bot six-key chain-tail behavior preserves the occupant; its offline control
at `test_state_machine.py:20615` remains passing.

A suffix absent from the reviewed inventory is unverifiable. After an exact
typed occupancy lookup, it is saved in the actor-owned candidate capability
with `needs_manual_review=true` and a concrete reason. `鎗@qxioio` retains
`qiāng` and `形码 ioio 未能核验，需管理员复核` in its review remark and receipt.
The internal reviewed-reading capability reaches the real draft validator;
model-supplied fields cannot create it. The user-selected code is not silently
replaced by a shorter recommendation. Occupied explicit codes use the same
bounded commonness comparator as ordinary reviewed candidates.

## Commonness guard placement

`assess_candidate_chain_commonness` now examines Single and Phrase occupants,
including a chain with no empty candidate. `protected_candidate_occupants`
binds each decision to newcomer, occupant, and code. Every occupant of a
shared slot must be demonstrably weaker before default displacement is
allowed. Ties, stronger occupants, unknown evidence, failed comparison, and
missing occupant identity never authorize a default eviction.

The review recommendation applier, persisted ordering snapshot, shared
renderer, and pending executor all use that same decision. With no safe
default the recommendation is empty, while the reviewed record remains usable
for explicit code selection or named eviction. The renderer retains Single
type, reading evidence, occupancy, and review result, and advertises only
complete named eviction commands. The fuller-code syntax is descriptive
formatting, not an executable placeholder. Bare `加入`, `加入并提交`, and `好`
keep the protected occupant and preserve the candidate record.

Named forms such as `加入，顶替 强`, `加入1，挤掉蛋粉`, and complete
`添加 鎗 qx，顶替 强` retain exact occupant binding. Single shift targets retain
their Single type and reviewed reading through the existing shift planner.
The complete add-plus-shift entry point also uses the same guard and the
same refusal renderer. A generic `顺延其他词条` must not manufacture a shift
ticket for a protected occupant: both fresh and already-reviewed entry
reproductions reached that unsafe preview before the fix. Only an occupant
actually named in the original command supplies override authority; a name
filled in by a lookup does not.
Every selected occupied code is guarded independently; selecting two empty
slots does not accidentally fall back to a protected recommendation. Ordinary
multiple-code additions leave existing occupants in place, so they retain
their existing add behavior without acquiring a displacement capability.

For a mixed bare-word query, a word with no safe default is explicitly left
unselected and rendered as reviewed occupancy information; other safe scopes
continue through the existing batch selector. The informational block carries
no executable ordinal or assent. An all-blocked query has no actionable batch
record and keeps the same per-word diagnostic information.

Observed vendored-reference comparisons with external fallback forbidden:

| New item / occupant | Evidence | Decision |
|---|---|---|
| 鎗 / 强 | frequency 10 / 17,342 | occupant stronger |
| 单份 / 蛋粉 | new absent / occupant corpus-and-dictionary presence | occupant stronger |
| 发布会 / 重病号 | frequency 4,417 / 2 | newcomer stronger |
| 出圈 / 除权 | frequency 3 / 65 | occupant stronger |
| 载具 / 在距 | both corpus frequencies unknown; dictionary presence 2 / 0 | newcomer stronger |

S48's old unnamed `添加1` displacement and S39's old unnamed `1 重新编码`
displacement conflict with the explicit S56 requirement to name a protected
occupant. Their no-write controls now preserve exact snapshots and require
zero model calls and zero mutation dispatch. Their named-eviction positives
retain exact final rows, reviewed seals, and confirmed-plan assertions.
Offline controls additionally prove that ordinal and numeric recode selection
still execute when the occupant is demonstrably weaker.
S38 also contained superseded default-eviction expectations: 耙耙柑 has no
reference frequency or dictionary presence, while 琵琶骨 has frequency 42
and dictionary presence 3 (`behind_more_common`). Its occupied ordinal
selectors and generic shift form are retained as protection controls; named
positives retain the actual shift checks, and bare assent uses a safe empty
recommendation.

## Undo route and window

A durable journal captures real pre/post-write snapshots at the shared
effective-tool dispatch boundary, including deterministic writes and model
ticket replays. It groups the actual writes of one turn and binds the record
to platform, conversation, actor, operation ID, batch, content version, and
exact rows. The last actor write and previous conversation turn are stored
separately; assistant prose is never used to invent inverse mutations.

Literal `取消` / `撤销` / `撤回` / `回滚` enters a deterministic stage before
classification. Automatic undo requires the same actor and conversation, the
immediately previous completed-write turn, no intervening actor write, and
an elapsed interval of 0–600 seconds. A time/turn/write mismatch produces one
line naming the operation and batch link and requesting a native quoted
yes/no. The confirmation remains tied to the original operation and exact
version. Questions, negation, reported text, and pasted untrusted confirmation
text cannot authorize undo.

Dictionary shift proposals are draft Create/Delete rows. Reverting the
incident chain deletes exactly the five operation-owned rows in one existing
version/digest-bound batch delete and verifies the remaining draft. It does
not delete unrelated draft rows or edit the dictionary directly. Submitted
batches first use the existing exact recall flow, followed by the same undo;
the receipt explicitly says that the submission was recalled.

If an operation changed or removed existing draft rows, bounded raw PR detail
reads preserve original nullable weight and other persisted fields. Undo
removes this operation's differences and restores the original semantic rows
through the existing strict batch preview/CAS endpoint. Restored rows may have
new IDs. These are durable phases, not a claim of one backend transaction.
Partial completion and uncertain responses retain exact resumable state and
block unrelated writes until resolved. A successful receipt requires the
actual mutation result and a verified final snapshot.

The existing `_recent_own_write` submission capability is a completed-receipt
affordance, not an unfinished add or warning plan. The undo stage consumes
only its exact actor/batch-bound form before executing the journal undo;
other unfinished tickets retain their normal cancellation semantics.

Independent adversarial tests reproduced and fixed three additional failures:
a rejected read-only deletion preview leaving a permanent actor fence;
a successful recall with a lost response being rejected on replay; and an
old Delete/Change target changing during restoration preview. Recovery now
requires the original recall claim, exact batch/version, verified Draft
status, and actual success state. Restoration rechecks the original target
fingerprint after preview and before confirmed dispatch.

The existing backend has no atomic delete-and-restore operation. Its warning
digest also omits the dictionary target's remark, so the bot's two target
checks cannot establish atomic target CAS across a concurrent remark edit.
This boundary is not presented as covered by local tests or by a draft
content-version check.

An old row whose remark contains either exact server review delimiter
(`--- miao-review:start ---` or `--- miao-review:end ---`) cannot be rebuilt
through the current strict batch API. Undo detects this before any recall or
deletion, preserves all current rows and batch status, and reports that the
whole operation could not be reverted. This remains an explicit unsupported
restoration case; no successful-undo claim is made for it.

### Source entry points

| Boundary | Current source |
|---|---|
| Closed explicit syntax and objective validation | `keytao_bot/utils/explicit_code.py:28`, `:44` |
| Exact occupancy lookup and derived reviewed capability | `keytao_bot/plugins/chat_commands.py:11454` |
| Shared Single/Phrase commonness assessment | `keytao_bot/utils/keytao_review.py:6071` |
| Generic complete-command guard / pending selected-code guard | `keytao_bot/plugins/chat_commands.py:1522`, `:10385` |
| Shared protected-occupant refusal | `keytao_bot/plugins/chat_commands.py:629` |
| Existing default front-insert manual seal | `keytao_bot/plugins/chat_commands.py:10535` |
| Exact same-instruction retry delivery guard | `keytao_bot/harness/orchestrator.py:3742` |
| Early deterministic undo stage / stage registration | `keytao_bot/plugins/openai_chat.py:3323`, `:5451` |
| Actual write journal / undo window and confirmation | `keytao_bot/utils/completed_draft_undo.py:150`, `:279` |
| Exact operation-row deletion / original-row restoration | `keytao_bot/utils/completed_draft_undo.py:464`, `:506` |

## Verification ledger

FULL invocation 9 (`20260907T100152Z-29303054`) is rejected: S1–S12 passed
on attempt 1; S13's complete reply told the user to add the absent draft item
and then `之后再用` the exact original weight instruction. The first line
alone was a correct empty-draft explanation, but the second line violated the
existing no-resend assertion. That assertion remains unchanged. The shared
finalizer recognized resend verbs such as sending and entering, but returned
before its exact-message check for `再用` or `再次使用`. Adding those two verb
forms only when the complete current instruction is repeated retains the
existing send/input rules and interrogative exemption. An independent control
showed that applying the broad `current` reference heuristic to the new verbs
would suppress a different draft-view suggestion, so the new branch is gated
on exact normalized message inclusion. Two reproduction subcases fail before the correction and pass
afterward; plain failure explanations, non-directive quotations, and a
different draft-view command remain unchanged. The only dispatched tool was
read-only `keytao_list_draft_items` (event 1768); the final draft was absent,
version zero, with no rows. The rig was interrupted with exit 130. A fresh
monolithic run is required after this correction and targeted verification.

FULL invocation 8 (`20260907T093955Z-169b10b8`) is rejected and interrupted
(exit 130). S1–S21 passed on attempt 1, then S22's main model request failed
with `ConnectTimeout` after 180.002 seconds (event 4742) and its bounded
transport retry failed with `ConnectError` after 17.638 seconds (event 4743).
There was no model response for those requests and no successful write. The
same configured provider subsequently returned HTTP 200 (event 4746), so a
targeted S22 recovery check precedes a fresh FULL process. No source change,
provider/model fallback, longer timeout, or relaxed assertion is applied for
this transient connection failure.

FULL invocation 7 (`20260907T091831Z-6b341dea`) is rejected and interrupted
(exit 130). S1–S26 passed on attempt 1; S27 failed on attempt 1 and passed
on attempt 2. The provider returned a nonempty answer to the binding-process
question. Its phrase `工具调用` triggered the existing implementation-narration
filter, and the final delivery redraw replaced the whole answer with the
generic no-reply fallback. This was not a provider empty response or the
advertised-live-state guard. The same filter and delivery behavior exist in
baseline `b012704`.
The minimal delivery correction reuses the existing deterministic binding
meta-answer when that redraw has no live record to render. It retains live
record precedence, the narration filter, advertisement validation, and all
authorization checks. A captured-trigger regression changed from the generic
fallback to the existing plain binding answer; unrelated input still receives
the original fallback. S27's direct-answer assertion is unchanged.

FULL invocation 6 (`20260907T084600Z-dc95e304`) is rejected and interrupted
(exit 130): S1–S38 passed on attempt 1, including the unchanged S35 seal gate;
both S39 attempts rejected the same protected-occupant reply. The retained
input `加词 出圈 jjqt 重新编码` does not name `除权`, whose commonness exceeds
`出圈` (65 versus 3). The correct S56 guard retained the current position,
made no draft change, and offered the bound `加入，顶替 除权` override. The
old compound-copy assertion only accepted wording that preserved an unnamed
add-and-shift suggestion. It now follows requirement B while retaining exact
no-write and named-command closure checks. The existing named
selection had already produced its exact three rows and manual seal.
This cold complete command used one ordinary intent-model exchange before
read-only lookup/review; it is not a zero-model pending assent. The pending
numeric protection and advertisement closure retain their zero-model checks.
The original artifact's message slice contains only the four permitted
read-only tool types and zero shift calls; the before/after batch, version 2,
and empty items agree. Later compensating fixture cleanup is outside that
message slice. The final assertion preserves the original input, requires
the protection reason and exact unchanged snapshot, and checks both the named
suggestion and its complete bullet through the real classifier, canonicalizer,
live binder, and whole-reply record binding. An independent read-only audit
of S40–S55 found no further analogous unnamed generic-remediation conflict.

FULL invocation 5 (`20260907T082140Z-a6b3d63a`) is rejected and interrupted
(exit 130). The initial S56 closure loop covered named commands, complete
bullet envelopes, and add assents but omitted the concrete multi-select
example, generic plan confirmation/cancellation, and recent-write submission
advertisements. The earlier targeted run proved the enumerated incident
behaviors, but did not prove the user's every-advertised-string closure gate.
The loop is expanded to cover those additional exact strings with their real
parser and appropriate live capability binding before a new FULL invocation
is accepted. The additional shift cancellation check exposed a product bug:
the shared assent parser recognized cancellation only for add tickets, so a
shift plan advertising `取消` could fall through to the model. Recognized,
closed cancellation now applies to live tool tickets as well. A real pending
executor regression observed `None` before the fix and `已取消。` afterward,
with the ticket removed and no model or tool call. Questions, reported text,
conditional text, and negated cancellation remain non-authorizing.
The expanded check retains the original checks and adds `添加2、4`, the
advertised free-slot ordinal `2`, `确认`, `取消`, and `提交`. Candidate strings
pass the real classifier, exact ticket canonicalizer, and live binding check;
plan controls pass their respective pending-confirm/pending-cancel parser;
submission checks the actual draft and actor-owned recent-write capability.
Each per-reply closure is required to add zero model exchanges. A replay of
all replies from the earlier S56 artifact covers 17 advertising replies and
59 string/envelope checks with zero model calls. That replay reconstructs
capabilities from recorded tool evidence; the fresh targeted/FULL runs below
are the evidence for their actual durable live records.

FULL invocation 4 (`20260907T075204Z-80e3d1c2`) is rejected: S1–S34 passed
on attempt 1, then S35 created the correct three rows but marked the new
`发布会@fbh` row `needsManualReview=false`. This was an implementation
regression, not an obsolete assertion. Baseline `b012704` omitted the target
review override in its recommended-front-insert branch; the existing shift
tool defaults a new target to manual review. The new type/reading transport
also passed the earlier lexical review's false flag, overriding that default.
The branch now explicitly retains the existing manual seal while preserving
type, reading, and review remark. A focused regression failed for both bare
add and add-and-submit before this correction and passes afterward. S35's
original exact-row, one-confirmation, and manual-seal assertions are unchanged.
The process was interrupted (exit 130); a fresh FULL run is required.
The accepted baseline artifact `20260907T034449Z-b7f1c65d/S35-attempt-1.json`
independently confirms this distinction: lexical review was already false,
while the actual default insertion retained `sealedCreate=true`, one
confirmation, and Submitted status.

FULL invocation 3 (`20260907T071758Z-8df4743b`) is rejected: S1–S27 all
passed on attempt 1, including the real S21 control, then S28's invalid-code
control required the old phrases `可选读音链` and `没有执行添加`. The actual
reply correctly identified `zzzzzz` as a wrong phonetic prefix, named the
reviewed bases `htje / htwe`, and said `本次未写入`; the draft remained empty.
This is a superseded copy expectation under requirement A. The correction
requires the precise failure reason while retaining the compactness, no bogus
batch link, no write-tool dispatch, and unchanged draft checks. The process
was interrupted (exit 130). Replaying the original S28 artifact through the
complete current invalid-control assertion block passed: zero model calls,
zero write-tool dispatch, empty version-zero draft, and only the two reviewed
phonetic bases out of six candidates mentioned. The final literal safety
suite passed 103 tests in 0.611s. The S29–S56 invalid-code copy audit found no
other superseded equivalent assertion.

FULL invocation 2 (`20260907T071126Z-42a60402`) is rejected: S1–S8 passed
on attempt 1, then S9 had repeated connection errors and no successful write.
The process was interrupted (exit 130). Diagnostic GET probes failed 3/3
with `trust_env=False` and reached the provider 3/3 through the existing
`http://127.0.0.1:7890` system proxy (HTTP 401 without credentials).
The rig's `NO_PROXY` entries suppress macOS proxy inheritance in this
installed HTTPX environment: no HTTPS proxy was selected until explicitly
setting `HTTPS_PROXY`. Invocation 3 supplies that existing transport route;
localhost remains excluded, Next's outbound safety restriction remains in
place, and the configured provider/model are unchanged. No product code or
test expectations changed for this transport correction.

FULL invocation 1 (`20260907T065335Z-c6620e28`) is rejected: S1–S20 passed
on attempt 1, then S21 treated a read-only identity lookup as a mutation. The
process was interrupted (exit 130); subsequent scenario retries cannot make
this invocation acceptable. S21's exact draft snapshot and tool-policy
assertions are retained.

The mutation allegation is rebutted by the recorded sole offending request,
`POST /api/bot/user/find`, and the current Next source: its route performs
`prisma.user.findFirst` at `app/api/bot/user/find/route.ts:47`, while
`lib/botAuth.ts:8` only compares the bot token. Neither implementation writes
data or allocates a persistent session. The rig previously classified every
non-GET local request as a mutation. Its correction exempts only that exact
read-only POST path; draft POST/PATCH/DELETE requests and exact snapshot
changes still fail the control.

The exact original S21 event slice and original before/after snapshots were
replayed offline: failure before the correction, pass after it, with the same
batch, content version `2`, and empty items. A new negative control rejects
other methods and lookalike paths. The post-correction literal safety suite
passed 103 tests; no product behavior changed between FULL invocations.

The final six-suite run includes the S27 delivery and narrowed S13 retry corrections, with every command
exiting 0. Exact logs and exit metadata are in
`/private/tmp/keytao-s56-checks/final`. Its `source-freeze.json` preserves the
119 source/configuration/prompt SHA-256 values verified unchanged after those
checks and immediately before FULL invocation 10 (freeze timestamp
`2026-09-07T10:12:20.416751+00:00`).
These are offline evidence; the monolithic FULL gate is recorded separately.

| Command | Observed tail | Exit |
|---|---|---:|
| `.venv/bin/python test_state_machine.py` | `Results: 2019/2019 passed, 0 failed` / `ALL TESTS PASSED` | 0 |
| `.venv/bin/python test_memory_safety.py` | `Ran 407 tests in 187.735s` / `OK` | 0 |
| `.venv/bin/python test_security_fixes.py` | `Results: 268/268 passed, 0 failed` | 0 |
| `.venv/bin/python test_review_gate.py` | `Results: 443/443 passed` / `ALL TESTS PASSED` | 0 |
| `.venv/bin/python test_llm_policy.py` | `Ran 10 tests in 0.096s` / `OK` | 0 |
| `.venv/bin/python test_word_discovery.py` | `Results: 290/290 passed` / `ALL TESTS PASSED` | 0 |
| `.venv/bin/python -m e2e.test_safety` | `Ran 107 tests in 0.616s` / `OK` | 0 |
| `.venv/bin/python -m unittest test_s56_explicit_code test_s56_commonness test_s56_commonness_binding test_s56_e2e_closure test_s56_security test_s56_undo test_s56_undo_review test_s56_generic_eviction` | `Ran 64 tests in 0.382s` / `OK` | 0 |

`git diff --check` and an AST parse of 59 Python files (all product modules
and the selected rig files) passed. The final source freeze records 119 source/configuration/prompt
file SHA-256 values, the bot and Next revisions, preserved worktree status,
and absence of staged changes. Key presence was checked without printing it.

| Targeted artifact | Result and observed correction |
|---|---|
| `20260907T061745Z-0a3cec42` | Rejected: real Single row and manual seal were correct, but the success receipt omitted the explicit shape-review reason. |
| `20260907T062035Z-cd7bce40` | Rejected: complete `添加 鎗 qxioio` took the older replacement route. It now enters the same reviewed explicit-code executor. |
| `20260907T062234Z-0d8e2313` | Rejected: all four explicit forms and submitted undo passed, then fuller-code format text in a refusal was misread as an advertised executable placeholder. |
| `20260907T062624Z-34878c97` | Rejected: the undo snapshot hook rejected a legitimate first-write absence-CAS preview UUID. It now accepts only an explicit empty version-zero pre-state for that exact case. |
| `20260907T062954Z-0968cf98` | Rejected: all prior controls and the exact five-row named shift passed; the existing recent-write submission capability then required explicit integration with the undo route. |
| `20260907T064514Z-bbfc6f50` | Rejected and interrupted after attempt 1: repeated model connection errors before the initial review. Subsequent direct and configured-proxy probes reached the configured provider; a fresh invocation followed. |
| `20260907T064737Z-1f4a61c5` | Rejected and interrupted after attempt 1: all original incident controls passed, and the old-row weight update succeeded exactly. The extra fixture incorrectly required that existing general-flow update to use zero model calls. The update now serves as a real model-write journal control; its following undo still requires zero model calls and exact restoration. |
| `20260907T065107Z-4cd18c38` | Targeted S56 passed its then-current assertions, attempt 1, exit 0: 94.2 seconds, 28 model requests, 82,389 tokens. Incident controls, the original partial advertisement checks, submitted undo, exact five-row chain undo, old-row weight restoration, and weaker-word default shift passed. This run does not prove the subsequently expanded every-string closure gate. |
| `20260907T075030Z-47c8d242` | Targeted S38 passed, attempt 1, exit 0: 47.5 seconds, 13 model requests, 6,412 tokens. Protected ordinal and generic inputs did not write; named overrides produced the exact actual shift, and bare assent used a safe empty recommendation. |
| `20260907T082008Z-51926454` | Targeted S35 passed, attempt 1, exit 0: 59.4 seconds, 6 model requests, 2,464 tokens. The exact front insert retained its manual seal and Submitted status after one confirmation; numbered opt-out and free-slot controls also passed. |
| `20260907T083634Z-dcf1b127` | Rejected and interrupted after attempt 1: expanded submission-advertisement closure compared raw journal rows (with two internal target audit fields) to canonical draft-list rows. The exact batch/version, IDs, weights, remarks, and review seals agreed. This comparison requires the same existing snapshot normalization on both sides; no product change is warranted. |
| `20260907T084355Z-2bde2cfb` | Targeted S56 passed the expanded assertions, attempt 1, exit 0: 89.0 seconds, 28 model requests, 82,517 tokens. Its 16 advertising replies produced 58 real parser/binding checks with zero model exchanges; every incident, submitted undo, exact chain undo, existing-row restoration, and weaker-word control passed. This particular weight receipt did not advertise submission; the prior real receipt's submission form is covered by the canonical journal regression. |
| `20260907T091726Z-69cd5de0` | Targeted S39 passed, attempt 1, exit 0: 18.8 seconds, 5 model requests, 3,081 tokens. Protected numeric and compound inputs left the exact draft unchanged; the compound used zero shift calls, one cold intent-model exchange, and zero parser-closure model calls. Named selection and occupant-perspective positives retained the exact sealed shift. |
| `20260907T093807Z-3be92545` | Targeted S27 passed, attempt 1, exit 0: 28.3 seconds, 7 model requests, 27,119 tokens. The original direct binding-process answer assertion remains unchanged. Pure replay of the original failed model response also produces the existing plain binding answer after the delivery correction. |
| `20260907T095702Z-a7c89f84` | Targeted S22 passed after the FULL8 transport failure, attempt 1, exit 0: 224.1 seconds, 10 model requests, 97,367 tokens. Re-review rebuilt the two live advertised bindings, the exact two words were written and approved, and fixture cleanup was verified. Provider, model, timeout, and safety settings were unchanged. |
| `20260907T100956Z-ef2fc2a2` | Targeted S13 passed the original assertion, attempt 1, exit 0: 10.3 seconds, 3 model requests, 47,973 tokens. A subsequent independent different-command control required narrowing the new retry verbs before accepting the source freeze. |
| `20260907T101221Z-ece0ccd5` | Targeted S13 passed after that narrowing, attempt 1, exit 0: 8.7 seconds, 3 model requests, 48,139 tokens. Original-artifact pure replay also passes the unchanged S13 assertion, while the different draft-view suggestion, binding question, successful receipt priority, and plain failure explanation remain intact. |

In targeted artifact `20260907T065107Z-4cd18c38`, the existing weight-write route used
four model exchanges; its following undo used zero. The target's semantic row
was restored from ID `17527` to `17529`, with weight `10` and its original
review seal and remark. Adjacent same-word row `17528` retained its ID and all
fields. The undo's DELETE event `980` and strict confirmed restoration POST
event `988` both returned HTTP 200 with successful exact mutation receipts.

Logs are preserved under `/private/tmp/keytao-s56-checks` and the corresponding
rig artifact directories. None of these targeted invocations is an accepted
monolithic deploy gate or production verification.

## Accepted monolithic FULL run

Artifact: `e2e/artifacts/20260907T101611Z-453de2dd`; log:
`/private/tmp/keytao-s56-checks/full-s1-s56-run10.log` (also archived as
`full-run.log` inside the artifact). UTC start `2026-09-07 10:16:11`,
completion `2026-09-07 10:53:26`; process exit `0`.

`manifest.json`, `summary.json`, and exactly 56 attempt files agree on
S1–S56, all PASSED on attempt 1. The real `openai.AsyncOpenAI` message path
recorded 309 successful provider exchanges and 1,689,125 tokens, using only
`api.deepseek.com` / `deepseek-v4-flash`; no fake model client was present.
The source freeze and final verifier prove 119 unchanged source/configuration/
prompt files, 59 valid Python ASTs, unchanged bot and Next HEADs, preserved
Next dirty file, no staged paths, and a clean `git diff --check`.
`offline-checks/` contains all final command logs and explicit exit metadata;
`completion-verification.json` records the completed local gate.

### S56 in the accepted process

S56 passed in 75.3 seconds, with 28 model requests and 82,556 tokens across
its complete setup, review, and control flows. Its 16 advertising replies
produced 58 real parser/canonicalization/live-binding checks (32 concrete
commands and 26 complete bullet envelopes), adding zero model exchanges.

| Control | Actual accepted-run result |
|---|---|
| Four explicit code forms | Each created only `鎗@qxioio`, `Single`, `needsManualReview=true`, weight 10, with `形码 ioio 未能核验，需管理员复核`; zero model exchanges on each explicit confirmation. |
| Invalid codes and exact duplicate | `qxio1o`, `qxioioa`, and `qkioio` identify charset, maximum length, and phonetic prefix errors respectively; exact same-word/six-code duplication is rejected; all leave the draft unchanged with zero model exchanges. |
| Bare assent | `加入`, `加入并提交`, and `好` leave the draft unchanged, preserve `强`, and use zero model exchanges. |
| Named eviction | `加入，顶替 强` creates the exact five draft rows: three creates for `鎗@qx`, `强@qxa`, `戕@qxai`, and two deletes for `强@qx`, `戕@qxa`. |
| Whole chain undo | `取消` removes all five operation rows, leaves an empty draft, and verifies the original dictionary state; zero model exchanges. The receipt-backed recent submission hint is consumed correctly. |
| Submitted undo | The batch moves from Submitted to Draft through recall before deletion, then has zero remaining items; zero model exchanges. |
| Existing-row undo | Weight 10→11 on row 18601 is restored to 10 on replacement row 18603 with the original remark and manual seal. Adjacent row 18602 remains unchanged. The original weight update uses four model exchanges; undo uses zero and the strict restoration POST is confirmed. |
| Weaker-word control | The shared comparator rates `发布会` above `重病号` (frequency 4417 vs 2); `加入` followed by the existing `确认` keeps default shift, producing the exact three rows for `发布会@fbh` and `重病号 fbh→fbhu`, with zero model exchanges in both execution turns. |

The actual HTTP evidence agrees with those facts: explicit-code confirmation
POSTs 18448/18534/18619/18705 each create one row; named-chain POST 19181
creates five and DELETE 19213 removes exactly IDs 18596–18600. Recall 18949
precedes submitted-row deletion 18961. Existing-row restoration uses DELETE
19292, preview 19299, and confirmed POST 19300; GET 19309 returns the unchanged
sentinel 18602 and restored row 18603. The weaker-word POST 19459 creates the
three rows returned by GET 19473. All cited mutations return HTTP 200 with
successful server receipts. A separate read-only audit matched every actual
advertising reply to its 58 binding checks. This weight receipt does not
advertise submission, so its closure count is not copied from a different run.

The backend restoration limitations documented above still apply: restoring
old rows is a checked two-phase operation, and legacy server review delimiters
cause a safe refusal before recall or deletion. No atomic-backend or production
claim is inferred from this run.

### Accepted scenario table

| Scenario | Name | Verdict | Attempt | Seconds | Model requests | Tokens |
|---|---|---|---:|---:|---:|---:|
| S1 | cold eviction default | PASSED | 1 | 22.1 | 2 | 1,113 |
| S2 | explicit duplicate | PASSED | 1 | 16.6 | 1 | 593 |
| S3 | back placement | PASSED | 1 | 4.9 | 0 | 0 |
| S4 | stale and single-delete confirmation | PASSED | 1 | 11.9 | 3 | 48,019 |
| S5 | self-service convergence | PASSED | 1 | 10.7 | 0 | 0 |
| S6 | injection controls | PASSED | 1 | 32.6 | 7 | 50,861 |
| S7 | timeout retry | PASSED | 1 | 5.5 | 0 | 0 |
| S8 | admin approval chain | PASSED | 1 | 15.6 | 0 | 0 |
| S9 | candidate commonness ordering | PASSED | 1 | 8.3 | 2 | 824 |
| S10 | multi-add authorization | PASSED | 1 | 32.7 | 7 | 104,536 |
| S11 | front-insert ticket before extra action | PASSED | 1 | 20.8 | 5 | 98,838 |
| S12 | front-insert weight legality | PASSED | 1 | 46.5 | 5 | 99,340 |
| S13 | same-turn resend loop breaker | PASSED | 1 | 8.6 | 3 | 47,895 |
| S14 | wrong-entry pronunciation poisoning | PASSED | 1 | 9.1 | 3 | 1,386 |
| S15 | numbered add-submit and suggestion/direct closure | PASSED | 1 | 17.4 | 4 | 1,648 |
| S16 | two-word bare advertised add-submit | PASSED | 1 | 43.0 | 8 | 52,237 |
| S17 | semantic common-character auto-pass | PASSED | 1 | 29.4 | 6 | 2,755 |
| S18 | multi-number candidate snapshot selection | PASSED | 1 | 13.6 | 3 | 1,519 |
| S19 | advertised-set subtraction with chunked progress | PASSED | 1 | 67.9 | 10 | 52,659 |
| S20 | native-quoted batch assent | PASSED | 1 | 22.4 | 5 | 70,990 |
| S21 | assent modifier and rendered remediation closure | PASSED | 1 | 56.1 | 14 | 129,741 |
| S22 | re-review advertisement state coupling | PASSED | 1 | 25.8 | 8 | 96,245 |
| S23 | stale advertised assent recovery and fresh closure | PASSED | 1 | 35.9 | 6 | 58,457 |
| S24 | single-word natural quoted assent | PASSED | 1 | 6.3 | 2 | 824 |
| S25 | natural add, record-backed number, and combined submit | PASSED | 1 | 15.0 | 4 | 2,014 |
| S26 | server-resolved add with occupant eviction | PASSED | 1 | 21.9 | 2 | 1,054 |
| S27 | binding precheck and question-turn reply | PASSED | 1 | 27.2 | 7 | 27,186 |
| S28 | reviewed multi-reading cascade closure | PASSED | 1 | 19.5 | 8 | 3,296 |
| S29 | quoted same-code commonness reorder | PASSED | 1 | 2.6 | 1 | 588 |
| S30 | read cancel and natural assent closure | PASSED | 1 | 32.6 | 6 | 2,488 |
| S31 | verbatim positional eviction closure | PASSED | 1 | 8.8 | 1 | 524 |
| S32 | draft-aware and explicit-list chain scope | PASSED | 1 | 8.9 | 0 | 0 |
| S33 | batch-aware homophone slot allocation | PASSED | 1 | 128.3 | 21 | 278,737 |
| S34 | pending submitted word awareness | PASSED | 1 | 46.7 | 8 | 3,242 |
| S35 | comparator recommendation is the default add plan | PASSED | 1 | 41.6 | 6 | 2,464 |
| S36 | dictionary delete and exact swap incident round | PASSED | 1 | 31.6 | 7 | 2,748 |
| S37 | occupant eviction and selected-slot revalidation | PASSED | 1 | 14.5 | 4 | 1,697 |
| S38 | reading, query recovery, and modifier incident closure | PASSED | 1 | 37.0 | 12 | 5,822 |
| S39 | one-turn reading selection and occupant eviction closure | PASSED | 1 | 11.2 | 4 | 2,368 |
| S40 | assent execution and existing-word incident closure | PASSED | 1 | 25.5 | 9 | 47,549 |
| S41 | reading focus and code explanation deduplication | PASSED | 1 | 37.2 | 7 | 105,722 |
| S42 | live candidate affordances and bare assent execution | PASSED | 1 | 73.5 | 10 | 129,911 |
| S43 | encode retry ladder and offline read-only degradation | PASSED | 1 | 15.4 | 3 | 1,687 |
| S44 | deterministic compound candidate selection | PASSED | 1 | 14.8 | 3 | 1,356 |
| S45 | swap verbs and interrogative review boundary | PASSED | 1 | 10.5 | 3 | 41,970 |
| S46 | multi-line promise-preserving double eviction | PASSED | 1 | 5.1 | 0 | 0 |
| S47 | choice and executable-suggestion closure | PASSED | 1 | 7.3 | 2 | 1,182 |
| S48 | numbered candidate create-with-eviction | PASSED | 1 | 32.4 | 7 | 2,992 |
| S49 | reasoning-only exhaustion bounded recovery | PASSED | 1 | 0.0 | 0 | 0 |
| S50 | relative-position grammar and honest confirmation | PASSED | 1 | 27.9 | 5 | 1,910 |
| S51 | replace-at-code grammar and honest confirmation | PASSED | 1 | 18.1 | 4 | 1,567 |
| S52 | first-render binding and possessive delete closure | PASSED | 1 | 19.0 | 3 | 1,368 |
| S53 | unknown-polyphone reading resolution | PASSED | 1 | 87.5 | 17 | 8,177 |
| S54 | bare multi-word reviewed candidate and selection closure | PASSED | 1 | 42.4 | 7 | 4,005 |
| S55 | Single review, failure traceback, and tripped search backend | PASSED | 1 | 13.9 | 6 | 2,465 |
| S56 | explicit codes, commonness-guarded eviction, and completed-write undo | PASSED | 1 | 75.3 | 28 | 82,556 |
