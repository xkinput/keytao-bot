# S55 — Single-character review and search backend recovery

## Scope and current verification status

Implemented locally from clean HEAD `2781a6d`. This work is edited-only: no
commit, push, deployment, production host, production API or production data
access. The banned `pnpm test` command was not run. External search evidence
uses fixtures; any unseeded search attempt is blocked by the E2E allowlist
before DNS/socket dispatch. The opt-in rig uses only localhost Next/PostgreSQL
and the authorized real `https://api.deepseek.com` / `deepseek-v4-flash` model,
with no provider or model fallback.

The six offline suites and 101-test E2E safety passed. The accepted fresh
monolithic FULL is `20260907T034449Z-b7f1c65d`: S1–S55 all PASSED on attempt 1,
exit 0, with no failed scenario attempt. Manifest, summary and all 55 scenario
artifacts agree. The 18 frozen Python source hashes and HEAD are unchanged.
This closes the requested local deploy gate; deployment remains unperformed.

## Diagnosis: confirmed type error, rebutted swallowed exception

**FIXED:** a bare single character reached the word review implementation,
which discarded the encoder's `type="单字"` and `chars`, previewed a `Phrase`,
and queried occupancy without restricting it to the Single table.

**REBUTTED:** the supplied `codes=["qx"]` fixture does not raise an exception
in the original discovery/render path. The exact original failure is an empty
return, not a swallowed exception:

1. `pending_confirmation.py:1704` at baseline rejects every inventory with
   fewer than two candidates in `render_server_backed_single_word_candidates`.
2. `render_server_backed_single_word_lookup` returns `""` at baseline line
   1797 when that renderer supplies no content.
3. `chat_commands.py:3631` assigns the empty result to `reviewed_prompt` after
   the candidate record has already been persisted.
4. `openai_chat.py:5265` sends `处理请求失败，请重试。` from the empty-response
   stage. This branch raises no exception and logs none.

The original full discovery fixture returned `''`, retained a
`PendingAddWord(qx)` and delivered that exact generic line. A separate replay
loads the original renderer directly with `git show 2781a6d`: `["qx"]` and
`["abcd"]` both return `''` without raising, while `["qx", "qxi"]` renders.
Thus candidate count, rather than the two-letter shape, triggers the failure.
The reproduction scripts and logs are archived with the accepted run.

The service legitimately returns `type="单字"`, `chars[0].phoneticCode="qx"`,
`shapeCode=null`, and `codes=["qx"]` for 鎗. Its Single encoder permits the
phonetic base when shape data is absent. It would be incorrect to invent a
shape expansion or claim the service cannot encode a character for which it
has returned a valid base code.

There was a separate real observability defect: `get_ai_response_core` caught
exceptions and logged only `API error: <message>` without traceback before
returning a retry invitation. That boundary and the main stage boundary are
now covered by the common failure reporter.

## Single pipeline and persisted capability

`prepare_reviewed_word` sends one-character inputs through
`_prepare_reviewed_single_character`. The new `keytao_single_char.py` validates
the encoder's Single type, exact character, known selected reading, phonetic
code, shape data, and ordered returned codes against the Single scheme. It
does not derive phrase chains. Only a server-returned, reading-bound alternate
group may satisfy a requested alternate reading; bare `chars.pinyins` do not
authorize additional invented code chains.

Both existing-word and exact-code occupancy results are filtered to `Single`.
A failed occupancy lookup remains unknown and cannot produce a recommendation.
The result retains `type="Single"`, `encodingType="单字"`, and the original
`chars`, with per-code reading and occupancy. New single characters remain
sealed for administrator review.

The 鎗/枪/槍 relationship is a small audited projection of the vendored
CC-CEDICT entry `鎗 枪 [qiang1] /variant of 槍|枪[qiang1]/rifle/spear/`.
The source attribution and license URL are in the module. Runtime never reads
a sibling checkout. The relation is bound to the evidenced `qiang` reading;
it is not guessed for `cheng` or for other characters. Available relation and
missing-shape facts appear in the same `审词` line as the reading. A
deterministic unencodable response is one truthful line:

```text
编码服务无法为「X」生成单字编码。
```

The shared selector and code extractor now accept one valid candidate. Invalid
or empty inventories still fail closed. If a persisted inventory unexpectedly
cannot render, discovery removes that unusable record and raises a logged
contract failure rather than delivering an empty string.

`PendingAddWord.phrase_type` is persisted as `phraseType`, validated on reload,
and compared in state equivalence and revalidation. Discovery fills it only
from the structured review. Single creation arguments and the trusted reviewed
reading capability carry `Single`, matching the existing draft sink's Single
inference and base weight. A singleton still exposes the common `加入` and
`加入并提交` footer. Both forms pass the real state-scoped assent parser and
deterministic classifier; the entire reply passes the live-record contract.
Numeric and literal-code selections use the existing structural binding path.

For an already-present single character, the existing `已在词库` block lists
the exact Single codes and the Single label, with no new add candidate or
assent offer. As with words, one unique exact typed location may persist a
`PendingTrustedWordRecord` for later unambiguous references; it is not an add
ticket. Multiple existing codes do not create that unique-location record.

## Failure reporting

Every former generic-empty-response emission now reports a deterministic
response-contract error at ERROR with a traceback. Actual stage exceptions are
caught at `_handle_ai_chat_serialized`, except the framework's normal
`FinishedException`. The original traceback is formatted without local-variable
dumps. Logs include `turn_id`, `flow`, and the failing stage name, so searching
`Traceback` finds the stack.

Users see the failed step, for example:

```text
审词和生成候选这一步失败，已记录错误供排查。
```

Only timeouts, connection errors, HTTP 429 or upstream 5xx (including chained
causes) invite a retry. Value errors, empty responses, unsupported protocols
and local protocol errors do not. The B boundary regression injects a real
raised discovery exception and requires ERROR, traceback, the exact turn and
flow, stage name, and a non-generic non-retry user reply.

## Search backend health and measured latency

`web-search/tools.py` keeps process-only health per configured backend. Three
consecutive failed attempts open the circuit for 600 seconds. Empty/no-result
responses, errors, timeouts, and caller cancellation all count as failures;
success resets the counter. During cooldown no request is dispatched. At
expiry one coroutine may claim a recovery probe; other callers skip it. A
generation token prevents an older in-flight success from closing a newer
open circuit. State transitions are logged once, not on every skipped query.

A backend is demoted after its first failure. Healthy ordering uses success
within the last ten minutes and a 0.5-weight latency moving average. The old
language-dependent order only breaks ties when there is no recent evidence.
An expired circuit gets one bounded probe. Each dispatched backend has a
2-second timeout, leaving room for another backend inside the existing S53
8-second web budget. so360 does not follow redirects, and a 302 cannot be
scored as a result. `providersTried` and S53 `outboundAttempts` count actual
dispatches only; `skipped` diagnostic entries do not inflate them.

The same fixture was run through the original `2781a6d` and final search
functions: two concurrent `鎗 拼音` / `鎗 读音` calls, so360 sleeping 6 seconds
then raising TimeoutError, and Bing sleeping 0.18 seconds then returning five
results. No search network was used.

| Search implementation | Wall seconds | so360 dispatches | Bing dispatches |
|---|---:|---:|---:|
| Original `2781a6d` | 6.1840 | 2 | 2 |
| New cold process | 2.1834 | 2 | 2 |
| New already-open circuit | 0.1823 | 0 | 2 |

This is a controlled search-segment comparison, not a measured production
speedup. The supplied production transcript's 13–23 seconds in tools remains
an incident baseline, with different environment and evidence paths. The S55
artifact separately records the complete local review and tool timings.

## jsonschema startup warning

The optional fallback is retained; no image dependency change is needed for
this incident. It validates root-object shape, required fields, declared
types (including rejecting booleans as numbers), enum values, and nested
objects/list items with depth capped at 3 and at most 5 reported errors.
It is deliberately not a full JSON Schema implementation. The warning is
truthful and emitted once. Schema validation is an early argument-quality
check, not write authority: explicit authorization, exact actor-owned target
binding, reviewed code membership, table/type validation, warning tickets,
and server-side write validation remain separate enforced boundaries. Those
boundaries are exercised by the six required suites and E2E safety. No claim
of full schema validation is made.

## Verification and run ledger

All commands ran with the repository's `.venv/bin/python`. The final source
inventory contains 18 changed/new Python files, frozen after the last code
change and checked immediately before FULL. No staged path exists.

Both FULL attempts used this exact invocation, with separate redirected logs:

```sh
E2E_ARTIFACT_RETENTION=1000 E2E_OPENAI_API_KEY="$(cat .e2e_key)" E2E_OPENAI_BASE_URL=https://api.deepseek.com E2E_OPENAI_MODEL=deepseek-v4-flash .venv/bin/python -u -m e2e.run
```

Preflight checked the key's presence without printing its value, fixed the
provider and model above, verified unchanged HEAD and no staged paths, and
recorded source hashes. The runner enforces localhost database/API targets,
blocks production URLs before dispatch and rejects production-like identities.

| Command | Observed final tail | Exit |
|---|---|---:|
| `.venv/bin/python test_memory_safety.py` | `Ran 407 tests in 189.053s` / `OK` | 0 |
| `.venv/bin/python test_state_machine.py` | `Results: 2013/2013 passed, 0 failed` / `ALL TESTS PASSED` | 0 |
| `.venv/bin/python test_security_fixes.py` | `Results: 268/268 passed, 0 failed` | 0 |
| `.venv/bin/python test_review_gate.py` | `Results: 443/443 passed` / `ALL TESTS PASSED` | 0 |
| `.venv/bin/python test_llm_policy.py` | `Ran 10 tests` / `OK` | 0 |
| `.venv/bin/python test_word_discovery.py` | `Results: 290/290 passed` / `ALL TESTS PASSED` | 0 |
| `.venv/bin/python -m e2e.test_safety` | `Ran 101 tests in 0.551s` / `OK` | 0 |
| `.venv/bin/python -m unittest test_s55_failure_boundary test_s55_single_pipeline test_s54_multiword test_s54_renderer test_s54_selection` | `Ran 41 tests` / `OK` | 0 |
| `.venv/bin/python -m unittest test_s55_single_review` | `Ran 10 tests` / `OK` | 0 |
| `.venv/bin/python test_s55_search_breaker.py` | `Ran 11 tests` / `OK` | 0 |
| Bounded schema fixture command | `5/5 bounded schema fallback fixtures passed; no HTTP or model call` | 0 |
| Python AST parse of 18 changed/new files; `git diff --check` | passed | 0 |

The initial attempt to put the distinct offline chat and review harnesses in
one unittest process encountered their incompatible fake `httpx` module
replacement (three errors). Each harness is now invoked in its own process,
as are the six existing suites; the final observed results above use those
separate commands. No product assertion was weakened for that harness clash.

Valid current native quotes were checked separately: the live Single record
and its bound prompt digest supply the type. An expired or mismatched explicit
Single quote may parse as the legacy Phrase display shape and is refused by
typed revalidation, requiring a fresh query. It does not restore a record or
write. This round does not broaden stale-reference recovery semantics.

The first monolithic FULL invocation, `20260907T032434Z-7175f2f8`, passed
S1–S20 on attempt 1, then failed S21's unrelated-text control. It was stopped
with Ctrl-C (exit 130) and is rejected as a deploy gate. S21 took 86.454
seconds, 15 real model requests and 129,878 tokens before that assertion.

**REBUTTED:** `S21 unrelated text outside the quote authorized a batch write`
misclassified a rejected model tool intent as a write. Input sequence 4367
was `请阅读确认`. Tool event 4394 had `success=false`, `policyBlocked=true`,
`blockReason=verb_not_matched`, and `missing=["executionVerb"]`. Between that
input and the final reply, there were zero local Next HTTP requests and seven
DeepSeek POSTs. Draft snapshots 4312 and 4409 had the same batch ID, content
version 2, and empty items. The old helper counted every tool event by name,
including policy refusals; this helper and assertion also exist at `2781a6d`.
The correction requires explicit policy refusal for every attempted batch
call, no backend write dispatch, and unchanged draft identity, version and
items. Authorization logic is unchanged. A fresh monolithic run must restart
at S1; neither this rejected run nor a targeted S21 pass can close the gate.
Three new safety regressions reject unblocked failures or previews, backend
dispatch even when state is restored, and changed or missing snapshot fields.
Replaying that exact failed artifact through the corrected assertion yields
one blocked attempt, zero local Next mutations and an unchanged snapshot.
The final 101-test E2E safety run passed before the source was frozen again.
Only `e2e/scenarios.py` and `e2e/test_safety.py` changed between FULL attempts.

The fresh second invocation, `20260907T034449Z-b7f1c65d`, ran the complete
S1–S55 selection from 03:44:49 to 04:19:33 UTC and finished with exit 0.
All 55 scenarios passed on attempt 1. Its strengthened S21 control had zero
batch attempts, zero local Next mutations and an unchanged version-2 draft.
The first run's explicitly blocked-attempt branch is separately preserved and
verified by artifact replay and the added offline controls.

The first targeted S55 artifact, `20260907T031520Z-28f4eba3`, failed both
attempts because its new assertion read only bullet-style command suggestions.
The actual shared singleton footer correctly used inline `回复「加入」…`.
The assertion now combines the contract's assent forms and explicit command
suggestions, verifies the exact common footer, and retains parser, classifier,
and whole-reply live-record checks. The typed record, qiāng, variant note and
qx display were already present in both failed artifacts. This targeted run
is not an accepted gate. Its observed row was 11.0 seconds, four model
requests, 1,648 tokens, exit 1.

The second targeted invocation, `20260907T031950Z-d96a0942`, passed S55 on
attempt 1, with 15.636 scenario seconds, six real model requests and 2,465
tokens. The 鎗 discovery took 5.259 seconds end to end and 0.150 seconds in
recorded tools. It wrote exactly one `Single 鎗@qx`, weight 10, with
`needsManualReview=true`, without regenerating the review on `加入`. The
existing `一@ykv` control retained a `PendingTrustedWordRecord` and created no
candidate or draft row. The tripped S53 fixture took 0.0041 seconds and made
exactly two Bing fixture dispatches and zero so360 dispatches. The error
injection returned the step-specific non-retry reply with ERROR traceback,
turn id, flow and stage. All four manifest safety proofs are true. This
targeted pass is supporting integration evidence only.

The same targeted inspection exposed an inaccurate preview audit sentence
calling Single character evidence an `整词语境判定`. The audit now handles a
typed Single after occupancy, duplicate, and candidate-membership checks and
retains its actual manual-review reason without invoking phrase semantic
auto-pass. The new regression first failed with one semantic call; it now
requires zero. Duplicate, invalid-code and failed-occupancy controls retain
their original precedence.

Independent review of A/B found no blocker within its inspected scope and ran
the Single review, durable candidate and exception-boundary focused checks.
It did not review its own C implementation and is not presented as cross-model
or production validation.

## Accepted monolithic FULL evidence

Accepted artifact directory: `e2e/artifacts/20260907T034449Z-b7f1c65d/`.
`completion-verification.json` records exit 0, 55 first-attempt passes, no
failed attempts, agreement between manifest/summary/individual artifacts,
18 unchanged source hashes, unchanged HEAD and an empty staging area.
`full-run.log`, `offline-checks/`, `reproduction/` and `source-freeze.json`
preserve the execution and diagnosis evidence.

The manifest records 284 successful real `openai.AsyncOpenAI` HTTP exchanges,
no fake client in the message path, and all four safety proofs true. Scenario
cost records total 285 model requests and 1,593,019 tokens, using only
`deepseek-v4-flash` at `api.deepseek.com`. The one unsuccessful model HTTP
exchange was S6 sequence 1070, a 400 for a tool message without a preceding
tool_calls message. It is retained in the artifact; S6 still passed its
assertions on attempt 1. No provider or model fallback was used.

Recorded scenario durations sum to 1,382.587 seconds; the 34m44s wall time
also includes setup, fixture warmup and cleanup. The runner reused an existing
local Next server (`reusedExisting=true`, `startedByRig=false`). This proves
local integration behavior, not a rebuilt image or production behavior.

Final S55 took 13.119 scenario seconds, six model requests and 2,465 tokens.
The actual 鎗 discovery took 4.344 seconds end to end and 0.142 seconds in
recorded tools. It persisted one Single candidate record before rendering
`qx`, qiāng, the 鎗/枪/槍 relation, missing-shape information, and both shared
assent affordances. `加入` wrote exactly `Single 鎗@qx`, weight 10, with
`needsManualReview=true` and without regenerating the review. Existing
`Single 一@ykv` retained its read-only trusted record and left the draft
unchanged. The open-circuit S53 fixture completed in 0.00086 seconds with
zero so360 calls, two Bing fixture calls, and zero external search requests.
The injected discovery ValueError produced ERROR with full traceback,
`turn_id=ec522b17`, `flow=word-discovery`, stage
`_stage_handle_simple_word_query`, and the step-specific non-retry reply.

These local timings and the controlled before/after search comparison are
reported separately from the supplied production incident's 13–23 tool seconds.
No production latency improvement or deployment has been claimed.

| Scenario | Verdict | Attempt | Seconds | Model requests | Tokens |
|---|---|---:|---:|---:|---:|
| S1 | PASSED | 1 | 23.5 | 2 | 1,070 |
| S2 | PASSED | 1 | 16.0 | 1 | 593 |
| S3 | PASSED | 1 | 4.6 | 0 | 0 |
| S4 | PASSED | 1 | 11.4 | 3 | 48,044 |
| S5 | PASSED | 1 | 10.2 | 0 | 0 |
| S6 | PASSED | 1 | 48.5 | 13 | 71,939 |
| S7 | PASSED | 1 | 5.3 | 0 | 0 |
| S8 | PASSED | 1 | 15.4 | 0 | 0 |
| S9 | PASSED | 1 | 7.2 | 2 | 824 |
| S10 | PASSED | 1 | 31.3 | 7 | 105,269 |
| S11 | PASSED | 1 | 22.8 | 5 | 99,733 |
| S12 | PASSED | 1 | 31.6 | 5 | 98,952 |
| S13 | PASSED | 1 | 10.7 | 3 | 48,007 |
| S14 | PASSED | 1 | 9.0 | 3 | 1,362 |
| S15 | PASSED | 1 | 18.4 | 4 | 1,648 |
| S16 | PASSED | 1 | 38.3 | 8 | 52,791 |
| S17 | PASSED | 1 | 26.7 | 6 | 2,755 |
| S18 | PASSED | 1 | 11.2 | 3 | 1,516 |
| S19 | PASSED | 1 | 60.9 | 10 | 52,212 |
| S20 | PASSED | 1 | 14.4 | 4 | 46,506 |
| S21 | PASSED | 1 | 74.8 | 14 | 124,337 |
| S22 | PASSED | 1 | 27.7 | 8 | 97,415 |
| S23 | PASSED | 1 | 31.4 | 5 | 77,718 |
| S24 | PASSED | 1 | 5.2 | 2 | 824 |
| S25 | PASSED | 1 | 13.6 | 4 | 2,014 |
| S26 | PASSED | 1 | 22.3 | 2 | 1,105 |
| S27 | PASSED | 1 | 34.4 | 7 | 27,722 |
| S28 | PASSED | 1 | 22.4 | 8 | 3,296 |
| S29 | PASSED | 1 | 2.4 | 1 | 588 |
| S30 | PASSED | 1 | 62.1 | 6 | 2,488 |
| S31 | PASSED | 1 | 8.7 | 1 | 523 |
| S32 | PASSED | 1 | 14.2 | 3 | 1,667 |
| S33 | PASSED | 1 | 137.5 | 24 | 333,329 |
| S34 | PASSED | 1 | 29.3 | 8 | 3,254 |
| S35 | PASSED | 1 | 23.1 | 6 | 2,464 |
| S36 | PASSED | 1 | 26.1 | 7 | 2,748 |
| S37 | PASSED | 1 | 11.6 | 4 | 1,702 |
| S38 | PASSED | 1 | 47.0 | 10 | 4,996 |
| S39 | PASSED | 1 | 9.4 | 4 | 2,368 |
| S40 | PASSED | 1 | 28.0 | 9 | 48,292 |
| S41 | PASSED | 1 | 33.2 | 6 | 80,361 |
| S42 | PASSED | 1 | 40.8 | 7 | 71,834 |
| S43 | PASSED | 1 | 14.3 | 3 | 1,712 |
| S44 | PASSED | 1 | 12.5 | 3 | 1,359 |
| S45 | PASSED | 1 | 9.7 | 3 | 42,074 |
| S46 | PASSED | 1 | 4.1 | 0 | 0 |
| S47 | PASSED | 1 | 5.2 | 2 | 1,182 |
| S48 | PASSED | 1 | 26.4 | 7 | 2,982 |
| S49 | PASSED | 1 | 0.0 | 0 | 0 |
| S50 | PASSED | 1 | 26.1 | 5 | 1,906 |
| S51 | PASSED | 1 | 15.2 | 4 | 1,567 |
| S52 | PASSED | 1 | 17.9 | 3 | 1,361 |
| S53 | PASSED | 1 | 75.8 | 17 | 8,157 |
| S54 | PASSED | 1 | 39.6 | 7 | 3,988 |
| S55 | PASSED | 1 | 13.1 | 6 | 2,465 |

Final handoff: edited only; not staged, committed, pushed or deployed.
Production behavior, external search availability, and container packaging
remain unverified by design.
