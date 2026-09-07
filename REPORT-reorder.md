# S57 — Same-code reorder, offered answers, and additional codes

## Scope and current gate status

Baseline: clean `86f9ea79c8ef1d285888184789620fcbdab1ea2c`.
Changes are local and uncommitted. No staging, commit, push, deployment,
production host/API/data access, or `pnpm test` is authorized or performed.
The opt-in rig uses localhost Next/PostgreSQL and the explicitly configured
`https://api.deepseek.com` / `deepseek-v4-flash`, without provider fallback.
Next remains `1e421aae905cc201d617f347bd96b97bbbeff534`; its pre-existing
dirty `pnpm-workspace.yaml` is preserved and fingerprinted in
`/private/tmp/keytao-s57-checks/preflight.json`.

S57 is closed locally: six offline suites, E2E safety and targeted S57 passed.
The first fresh monolithic S1–S57 invocation also passed all 57 scenarios on
attempt 1, with process exit 0 and all frozen source fingerprints unchanged.
The accepted artifact is `e2e/artifacts/20260907T162111Z-742d7f08`.

## Diagnosis and disposition

The supplied incidents are confirmed. The original full reorder instruction
returned `None` through the real deterministic command handler. The actual
delivery boundary returned the synthetic no-ticket option question unchanged.
The six added explicit-entry syntax variants failed the original parser.
`强制调序` returned `PendingAssentPhrase(recognized=False, matched=False)`.
These failures were reproduced before the corresponding fixes.

The existing ranked planner, actor-owned pending store, warning digests and
confirmed replay are reused. Commonness does not grant or withhold the user's
explicit reorder authority. Incorrect/ambiguous identities, invalid codes or
readings and unavailable facts remain objective refusal conditions.

## Same-code route and weights

`utils/same_code_reorder.py` accepts complete, positive two-item commands with
the quoted/space variants from the incident. Questions, negation, reported
instructions and appended actions do not parse. The actual public message
pipeline bypasses semantic routing for these closed commands, then executes
after operation arbitration and before ordinary pending dispatch/discovery.

Typed word lookups and the complete exact-code lookup are merged with the
current draft. A unique shared code/type is required. All members of that
typed same-code chain remain represented: relative insertion preserves the
others' order, while a swap exchanges the two selected positions. Distinct
existing weight slots are reused in ascending order. Tied weights use the
standing type base plus index through the shared planner (`Single=10`,
`Phrase=100`); no plan may assign below the type base.

The planner now scopes explicitly code-bound existing items to that code even
if the item also has another code. Same-code weight changes do not need fresh
encoding, and occupants in a different table do not enter that typed order.
The existing sealed preview/replay machinery still rechecks the real rows.

The preview stores both the exact weight deltas and one advisory in its
server plan's `evidenceLines`. The shared renderer preserves those lines and
does not describe an explicit user order as a conflicting recommendation.
Delivery retains an advisory only from a complete live sealed ticket. The
local reference comparator supplies the commonness fact; missing reference
evidence produces an uncertainty note, never a blocking question.

The incident preview is `将把 嘢 排到 咽 之前：嘢 11→10、咽 10→11`, with
`常用度提示：咽 更常用，仍按你的要求执行`. The receipt compares each same-code
proposal with the successful same-batch final draft snapshot before reporting
its actual weight change. Proposed changes absent from that snapshot are not
reported as completed.

## Offered options, claiming, and force controls

`utils/offered_options.py` identifies question-terminated sentences with
alternative structure (`还是`, `或者`, `或`) or implicit yes/no (`是否`).
It normalizes the offered strings rather than matching a list of action
keywords. An implicit yes/no question without literal answer strings cannot
invent an assent lexicon and is redrawn.

In write flows, both model finalization and final delivery require the exact
normalized question-option set to equal the live complete ticket's
`_offered_options` map. Each alias maps only to confirm/cancel of that existing
sealed operation and round-trips through the real pending parser and binding
checker. The model's prose cannot mint the map or supply mutation operands.
Unbacked options are replaced by the existing deterministic live preview or a
truthful no-write line. Plain explanatory questions remain available during
an unrelated pending operation.

The early claiming stage reads only the immediately preceding assistant turn
from actor-and-conversation-scoped durable history. An exact normalized offered
answer is consumed before word discovery. A missing/expired/unbound ticket
produces the previous proposal and a native-quote confirmation instruction,
without reviving a write from prose. An intervening turn ends that option
claim. Independent controls cover foreign actors and conversations.

Closed `强制`, `硬加`, `硬来`, `强行`, `就按我说的`, `强制调序`, `就这样`, and
`照做` are assent to the live operation. They do not accept questions,
negation, quotes or trailing targets. For an add candidate, override is bound
to one actually selected server candidate; normal assent retains S56's
commonness protection. Forced execution uses the existing confirmed plan
without another user confirmation. With no live operation, these controls
cannot become word queries.

## Additional-code path

`utils/explicit_code.py` parses complete explicit item/code commands, including
`添加单字"嘢"，编码为"yeoia"`, full-width quotes, `「」`, whitespace,
`给 嘢 加一个码 yeoia`, and `嘢 也放到 yeoia`. The existing contextual
`加入编码yeoia` consumes a trusted word record after a lookup.

The cold and contextual paths share reviewed candidate inventory, exact
Single/Phrase type, unambiguous reviewed reading, phonetic prefix and S56
code-length/character validation. Occupancy is read at the actual new code in
the correct table. An exact word/code/type duplicate is rejected; another
existing code does not block addition. Unverifiable suffixes retain a manual
review seal and a plain review note. Occupancy of `ye` cannot stand in for
occupancy of `yeoia`.

The existing-word renderer describes `回复「加入编码 <code>」可再追加一个编码`.
The placeholder is an explanatory template; executable closure uses concrete
codes through the real parser and binder. A multi-code lookup persists a
`context_only` trusted record carrying word/type identity, with that flag
sealed into its digest and durable serialization. It cannot authorize a
unique-location delete/recode or a bare assent. An exact completed-write
submission hint can yield to the next complete additional-code command;
unfinished operation tickets retain precedence.

For an explicit additional-code request, a private task-local capability is
derived only from the same reviewed existing word/type and a fresh empty
target-code lookup. It permits the server's pure `multiple_code` warning to
confirm once, with its exact item, previous code, manual seal and complete
batch/version/digest binding. Mixed risks, changed targets, missing seals or
incomplete CAS do not qualify. The capability is restored in `finally`;
ordinary creates still pause on this warning. Independent offline concurrent
actor checks found no capability leakage; submission remains separately bound
to the actual added item and requires the requested submit suffix.

The contextual command survives the immediate-next-turn record cleanup and
skips both generic and pending semantic classification. A regression exercises
the real scope, classification and execution stages for unique-location and
multi-code records; invalid literal codes reach deterministic validation,
while unrelated word queries consume the old lookup context.

## Validation ledger

Logs are under `/private/tmp/keytao-s57-checks`. Separate Python processes are
used for the six offline suites and E2E safety. Independent review found and
closed the read-only option-question regression, incomplete same-code receipt,
and three S57 evidence gaps: loss of continuous conversation, omitted failed
provider HTTP requests, and uncollected force/structural option advertisements.

Initial six-suite results: state `2019/2019`, security `268/268`, review
`443/443`, policy `10`, and discovery `290/290` passed. Memory reported exactly
three failing tests: `PlatformNeutralPendingTests.test_multiple_code_warning_pauses_on_snapshot_bound_ticket`,
`PlatformNeutralPendingTests.test_owner_duplicate_add_and_submit_authorizes_exact_preview_chain`,
and `CleanBatchAddOrchestratorTests.test_advertised_candidate_reply_persists_sealed_live_batch_ticket`.
The first two were product regressions caused by shortcut classification
running before existing pending parsing; both pass after restoring precedence.
The last expected literal unbound `是否…？` model copy. That expectation is
superseded by S57: only the expected deterministic copy changed; original model
input, complete candidate/owner/nonce/seal facts and no-write assertions remain.

Initial safety execution failed only the old S1–S56 registry length. Updating
the expected registry to S1–S57 preserves every safety assertion; all 107 pass.
Final source-matched suite tails will be recorded below. An intermediate
attempt to invoke `test_security.py` and `test_review.py` exited 2 because
those files do not exist; the prescribed suites are `test_security_fixes.py`
and `test_review_gate.py`, which were run literally afterward. These missing
file invocations are not counted as suite passes.

| Targeted artifact | Disposition |
|---|---|
| `20260907T155152Z-d7263abd` | Infrastructure preflight rejected S57's initially incorrect default reading expectation for 咽. Current local encoder selects `yàn`, retaining `yān/yàn/yè`; fixture metadata now asserts that actual default. No scenario or model call ran. |
| `20260907T155319Z-3d59ba90` | Rejected: helper called unregistered `keytao_lookup_phrase`. Replaced with registered `keytao_lookup_by_code`; regression now checks actual skill registries. |
| `20260907T155638Z-12e50680` | Rejected: actual two-row weight write succeeded with zero model requests, but completion omitted 咽. Corrected using verified final-row evidence. |
| `20260907T160040Z-1ffadcbf` | Rejected: additional-code create paused on the actual backend `multiple_code` warning. Added the narrow same-turn capability and exact CAS confirmation described above. |
| `20260907T160858Z-415efa34` | Rejected: continuous reorder/add succeeded, but the next-turn contextual code was discarded by legacy lookup-scope cleanup. Fixed record retention and deterministic classification, with a red/green stage regression. |
| `20260907T161514Z-c1da111c` | Rejected: local writes and option controls passed, but the synthetic boundary check inherited the prior read-only word query's ContextVar. The control now explicitly sets the original reorder request and restores context in `finally`, evaluating the required write-flow contract without changing production detection. Extra-code paths also now assert zero actual provider requests. |
| `20260907T161800Z-87ad3149` | PASSED, attempt 1, exit 0, 32.390s; 5 actual model requests / 1,889 tokens confined to ordinary query controls. The reorder, assent, all explicit additional-code forms and closure checks assert zero actual model requests. |

Final offline commands and tails (all exit 0; `*-final3.log`):

```text
.venv/bin/python test_state_machine.py
Results: 2019/2019 passed, 0 failed

.venv/bin/python test_memory_safety.py
Ran 407 tests in 206.673s
OK

.venv/bin/python test_security_fixes.py
Results: 268/268 passed, 0 failed

.venv/bin/python test_review_gate.py
Results: 443/443 passed

.venv/bin/python test_llm_policy.py
Ran 10 tests in 0.122s
OK

.venv/bin/python test_word_discovery.py
Results: 290/290 passed

.venv/bin/python -m e2e.test_safety
Ran 107 tests in 0.742s
OK

.venv/bin/python -m unittest test_s57_reorder test_s57_extra_code test_s57_options test_s57_claiming test_s57_security_review test_s57_e2e_closure_review test_s56_explicit_code
Ran 49 tests in 0.333s
OK

git diff --check
exit 0
```

Before FULL, all 116 source/config/test file fingerprints were frozen in
`/private/tmp/keytao-s57-checks/source-freeze.json`; bot HEAD, empty index,
Next HEAD and its original dirty-file hash were verified again. The report
is excluded from that source freeze so the final evidence can be recorded.

Key implementation evidence:

| Contract | Source |
|---|---|
| Closed reorder grammar and complete typed-chain plan | `keytao_bot/utils/same_code_reorder.py:44`, `:92` |
| Structural option detection and exact live lexicon | `keytao_bot/utils/offered_options.py:54`, `:120` |
| Final delivery boundary and early offered-answer claiming | `keytao_bot/plugins/openai_chat.py:2451`, `:3397` |
| Explicit item/code grammar and reviewed additional-code execution | `keytao_bot/utils/explicit_code.py:23`, `keytao_bot/plugins/chat_commands.py:11629` |
| Exact additional-code warning/CAS capability | `keytao_bot/plugins/chat_commands.py:4460` |
| Lookup context digest/serialization | `keytao_bot/harness/state.py:256` |
| Continuous three-turn and local dictionary proof | `e2e/s57.py:80`, `:96`, `:113` |

## Accepted monolithic FULL

Accepted invocation: `.venv/bin/python -u -m e2e.run`, without `--only`.
It ran from 2026-09-07 16:21:11.595 UTC to 17:04:42.346 UTC
(43m 30.750s), exit 0. This was FULL invocation 1; no FULL failed or retried.
All 57 `manifest.json` selections, summary verdicts and individual attempt
artifacts agree: PASSED, attempt 1, no failure. The run's ID is
`742d7f080b144257b02b854264d8c718`.

The manifest confirms localhost:3100, localhost PostgreSQL:5432,
`openai.AsyncOpenAI`, `api.deepseek.com` / `deepseek-v4-flash`, and no fake
client in the message path. All four production-target safety proofs pass.
The rig records 323 successful provider exchanges and 1,683,244 tokens;
provider monetary billing is unavailable locally. Network errors and model
transport failures remain in the raw artifacts; S57's zero-model assertions
count both model exchanges and attempted provider HTTP calls.

Accepted S57 took 19.573s. Its exact receipt is:

```text
✅ 操作已完成
已变更：嘢 yeoiav 权重 11→10、咽 yeoiav 权重 10→11
```

After the continuous three user turns, the actual local draft contained
exactly two Change rows and one sealed Create row. Local fixture approval
then produced these exact Single records:

| Word | Code | Weight |
|---|---|---|
| 咽 | yeoiav | 11 |
| 嘢 | yeoiav | 10 |
| 嘢 | yeoia | 10 |

The new row carried `needsManualReview=true` and the note
`形码 oia 未能核验，需管理员复核`. All advertised concrete commands used the
real parser and binder. The force lexicon, option confirmation/cancellation,
native quote-confirmation, contextual and cold code variants, genuine
four-character query and synthetic no-ticket option rejection passed.
Scenario cleanup was verified.

Artifacts:

- `e2e/artifacts/20260907T162111Z-742d7f08/manifest.json`
- `e2e/artifacts/20260907T162111Z-742d7f08/summary.json`
- `e2e/artifacts/20260907T162111Z-742d7f08/S57-attempt-1.json`
- `/private/tmp/keytao-s57-checks/full-s1-s57-run1.log`
- `/private/tmp/keytao-s57-checks/full-run1-exit.json`

| Scenario | Verdict | Attempt | Seconds | LLM requests (rig) | Tokens |
|---|---|---|---|---|---|
| S1 | PASSED | 1 | 21.813 | 2 | 1069 |
| S2 | PASSED | 1 | 16.451 | 1 | 593 |
| S3 | PASSED | 1 | 4.811 | 0 | 0 |
| S4 | PASSED | 1 | 13.107 | 3 | 48031 |
| S5 | PASSED | 1 | 10.672 | 0 | 0 |
| S6 | PASSED | 1 | 44.174 | 12 | 71179 |
| S7 | PASSED | 1 | 5.504 | 0 | 0 |
| S8 | PASSED | 1 | 15.685 | 0 | 0 |
| S9 | PASSED | 1 | 8.020 | 2 | 824 |
| S10 | PASSED | 1 | 29.387 | 7 | 103843 |
| S11 | PASSED | 1 | 22.214 | 5 | 99198 |
| S12 | PASSED | 1 | 22.988 | 5 | 101004 |
| S13 | PASSED | 1 | 189.034 | 3 | 47866 |
| S14 | PASSED | 1 | 8.135 | 3 | 1353 |
| S15 | PASSED | 1 | 19.261 | 4 | 1648 |
| S16 | PASSED | 1 | 46.547 | 8 | 52561 |
| S17 | PASSED | 1 | 24.903 | 6 | 2795 |
| S18 | PASSED | 1 | 12.591 | 3 | 1525 |
| S19 | PASSED | 1 | 59.534 | 10 | 52069 |
| S20 | PASSED | 1 | 17.861 | 5 | 46580 |
| S21 | PASSED | 1 | 77.501 | 16 | 129963 |
| S22 | PASSED | 1 | 29.974 | 9 | 96704 |
| S23 | PASSED | 1 | 30.574 | 5 | 50859 |
| S24 | PASSED | 1 | 7.033 | 2 | 824 |
| S25 | PASSED | 1 | 13.201 | 3 | 1419 |
| S26 | PASSED | 1 | 20.903 | 2 | 1088 |
| S27 | PASSED | 1 | 31.051 | 7 | 27249 |
| S28 | PASSED | 1 | 23.963 | 8 | 3296 |
| S29 | PASSED | 1 | 2.998 | 1 | 588 |
| S30 | PASSED | 1 | 32.455 | 6 | 2488 |
| S31 | PASSED | 1 | 8.837 | 1 | 520 |
| S32 | PASSED | 1 | 8.864 | 0 | 0 |
| S33 | PASSED | 1 | 129.240 | 22 | 303314 |
| S34 | PASSED | 1 | 40.231 | 8 | 3252 |
| S35 | PASSED | 1 | 24.128 | 6 | 2464 |
| S36 | PASSED | 1 | 28.043 | 7 | 2748 |
| S37 | PASSED | 1 | 13.513 | 4 | 1829 |
| S38 | PASSED | 1 | 33.636 | 12 | 5812 |
| S39 | PASSED | 1 | 10.724 | 4 | 2368 |
| S40 | PASSED | 1 | 43.934 | 10 | 47867 |
| S41 | PASSED | 1 | 42.046 | 6 | 100946 |
| S42 | PASSED | 1 | 73.149 | 12 | 112553 |
| S43 | PASSED | 1 | 14.900 | 3 | 1719 |
| S44 | PASSED | 1 | 13.333 | 3 | 1353 |
| S45 | PASSED | 1 | 190.275 | 3 | 41922 |
| S46 | PASSED | 1 | 5.043 | 0 | 0 |
| S47 | PASSED | 1 | 6.457 | 2 | 1182 |
| S48 | PASSED | 1 | 25.521 | 7 | 2996 |
| S49 | PASSED | 1 | 0.034 | 0 | 0 |
| S50 | PASSED | 1 | 24.039 | 5 | 1902 |
| S51 | PASSED | 1 | 31.039 | 4 | 1567 |
| S52 | PASSED | 1 | 19.798 | 3 | 1378 |
| S53 | PASSED | 1 | 79.214 | 17 | 8176 |
| S54 | PASSED | 1 | 41.249 | 7 | 3994 |
| S55 | PASSED | 1 | 14.066 | 6 | 2465 |
| S56 | PASSED | 1 | 70.648 | 28 | 82412 |
| S57 | PASSED | 1 | 19.573 | 5 | 1889 |

Post-run verification: all 116 frozen source/config/test fingerprints,
bot HEAD and empty index, Next HEAD and its pre-existing dirty-file hash
remain unchanged; `git diff --check` exits 0. Only this report was completed
after the accepted process. These are local real-provider/real-backend rig
results, not live QQ gateway or production deployment verification.
No staging, commit, push, deployment or production access was performed.
