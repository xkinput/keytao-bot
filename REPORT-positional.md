## Positional grammar

### Outcome

- Completed the positional-grammar round on repository HEAD `b1b33cef679c5da16d83150b45e35f49a21fcc2f` without committing, pushing, deploying, or touching production.
- The required single monolithic FULL rig selected exactly S1 through S50 and finished with 50/50 final `PASSED` verdicts.
- Run artifact: `e2e/artifacts/20260904T140029Z-e229c7f1`.
- Rig manifest: API key present, LLM host `api.deepseek.com`, model `deepseek-v4-flash`, and production URL/database/admin safety checks all true. All recorded model requests used `deepseek-v4-flash`; no provider fallback occurred.
- This is local live-model E2E evidence, not deployment or production verification.

### Resume audit and diagnosis

The original ten dirty files were inspected individually before further edits. The half-finished structure was retained where it matched the contract; ambiguous parser/planner behavior and S50 assertions were re-derived and finished rather than trusted.

| File | Decision | Result |
| --- | --- | --- |
| `e2e/README.md` | keep + finish | Documented S50 and the S1-S50 FULL gate. |
| `e2e/run.py` | finish + correct | Corrected S50's after-position fixture so the subject chain is anchored at the destination code; retained monolithic scenario selection and evidence capture. |
| `e2e/runtime.py` | keep + finish | Retained safe local runtime changes and completed S50 support. |
| `e2e/scenarios.py` | finish | Completed S50 transcript, parser/binding closure, ticket-honesty, recursive preview, six-code collision, final draft, receipt, and cleanup assertions. |
| `e2e/test_safety.py` | finish | Completed structural S50 and configuration/contract coverage. |
| `e2e/zdic_seed.py` | keep + finish | Retained bounded S50 fixture data. |
| `keytao_bot/harness/authorization_grammar.py` | finish | Completed relative-position parsing, including compact/spaced/full-width and contextual verbless forms. |
| `keytao_bot/plugins/chat_commands.py` | finish + correct | Re-derived target-code resolution, live-context gating, ticket rendering, and compatibility paths. |
| `keytao_bot/plugins/openai_chat.py` | finish | Completed typo-correction replacement of stale shift state and customer-copy handling. |
| `test_memory_safety.py` | finish | Completed grammar, binding, ticket, copy, correction, and regression coverage. |

Four additional files were changed only where diagnosis showed the original ten-file partial patch was insufficient: `keytao_bot/harness/orchestrator.py`, `keytao_bot/skills/keytao-draft/tools.py`, `keytao_bot/utils/pending_confirmation.py`, and `test_state_machine.py`.

The killed job's S50 failure was reproduced as:

`S50 肖像 server chain does not place xcxxiu immediately after xcxxi: ['xcxx', 'xcxxi', 'xcxxii']`

Root cause: after-position target resolution used the destination word's ordinary candidate list instead of asking the server encoder for the subject word relative to the destination's current code. The corrected path sends the destination code as `requested_code`, accepts the server's next candidate, and does no client-side code arithmetic. This yields `xcxxiu` after `xcxxi`; because that is a six-code terminal already occupied by `小箱`, the planner truthfully creates the only permitted same-code collision instead of inventing a seventh-code shift.

### Contract closure

- A — Relative position: supports `把 X 放在 Y 前面`, `把X放到Y前面`, `X 放在 Y 前面`, their after-position equivalents, full-width/spacing variants, new or existing X, and the composite `把小象放在销项前面，顺延后面的词`. Front placement takes Y's current code; after placement uses the server-returned next code. Recursive shifts remain server-planned.
- B — Contextual verbless claims: `X 在 Y 后面/前面` is actionable only when X is bound to trusted live pending/placement state. Outside that state it follows the normal conversation path; the implementation does not use an enumerated blacklist.
- C — Honest confirmation: the shared finalizer exposes `确认/取消` only while a live confirmable ticket exists. Correction replaces the obsolete ticket, and confirmation executes only the replacement plan.
- D — Copy sweep: product code no longer emits `这条回复里的建议没有对应的可执行计划，因此未发送该命令` or `绑定成完整目标`. The latter remains only in a negative assertion.
- Compatibility found during the FULL gate: cold explicit after-placement auto-confirms only when the server preview has no shifts; `同码放在` remains on the existing same-code weight-order path rather than being intercepted as cross-code relative placement.

### S50 closure

The exact transcript was executed in one scenario:

1. `喵喵 小像`
2. `小像确实比较常用，属于美团超市这块，换到前面`
3. advertised command replayed verbatim: `把 小像 放在 销项 前面`
4. `错了 是小象`
5. `小象在肖像后面`
6. `确认`

Observed facts:

- The advertised command round-tripped through parser and binding as `小像 / 销项 / 前面`.
- The front preview was server-backed and shifted `销项 xcxx→xcxxi` and `肖像 xcxxi→xcxxii`.
- Correction removed the stale `小像` plan and established live context for `小象`.
- The contextual claim bound as `小象 / 肖像 / 后面`.
- The after target was `xcxxiu`; the server reported the six-code collision with `小箱`, with no fictitious downstream shift.
- `确认` produced exactly `Create 小象 xcxxiu`; the receipt named the completed change, and fixture cleanup was verified.

### Verification

Final offline regression pass after all source changes:

| Check | Result |
| --- | --- |
| `test_memory_safety.py` | 392 tests, `OK` |
| `test_state_machine.py` | 1994/1994 passed |
| `test_security_fixes.py` | 268/268 passed |
| `test_review_gate.py` | 406/406 passed |
| `test_llm_policy.py` | 10 tests, `OK` |
| `test_word_discovery.py` | 290/290 passed |
| Six-suite total | 3360 passed |
| `e2e.test_safety` | 92 tests, `OK` |
| Python compile check | passed |
| `git diff --check` | passed |
| Product copy sweep | no product-code hits |

The required FULL command was run with explicit `E2E_OPENAI_API_KEY` sourced from the non-empty `.e2e_key`, `E2E_OPENAI_BASE_URL=https://api.deepseek.com`, and `E2E_OPENAI_MODEL=deepseek-v4-flash`:

| Scenario | Verdict | Attempts | Seconds | LLM requests | Tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| S1 | PASSED | 1 | 21.4 | 2 | 1061 |
| S2 | PASSED | 1 | 15.8 | 1 | 593 |
| S3 | PASSED | 1 | 4.6 | 0 | 0 |
| S4 | PASSED | 1 | 10.1 | 3 | 48024 |
| S5 | PASSED | 1 | 10.1 | 0 | 0 |
| S6 | PASSED | 2 | 70.0 | 16 | 142164 |
| S7 | PASSED | 1 | 5.2 | 0 | 0 |
| S8 | PASSED | 1 | 15.0 | 0 | 0 |
| S9 | PASSED | 1 | 7.2 | 2 | 824 |
| S10 | PASSED | 1 | 31.8 | 6 | 78949 |
| S11 | PASSED | 1 | 16.5 | 5 | 98641 |
| S12 | PASSED | 1 | 19.4 | 5 | 99777 |
| S13 | PASSED | 1 | 6.8 | 3 | 47902 |
| S14 | PASSED | 1 | 7.2 | 3 | 1367 |
| S15 | PASSED | 1 | 16.5 | 4 | 1648 |
| S16 | PASSED | 1 | 38.7 | 8 | 52159 |
| S17 | PASSED | 1 | 24.5 | 6 | 2778 |
| S18 | PASSED | 1 | 11.1 | 3 | 1301 |
| S19 | PASSED | 1 | 58.3 | 10 | 51878 |
| S20 | PASSED | 1 | 13.5 | 4 | 46424 |
| S21 | PASSED | 1 | 69.2 | 15 | 130028 |
| S22 | PASSED | 1 | 23.9 | 8 | 96582 |
| S23 | PASSED | 1 | 39.5 | 6 | 78508 |
| S24 | PASSED | 1 | 5.1 | 2 | 824 |
| S25 | PASSED | 1 | 11.7 | 4 | 2014 |
| S26 | PASSED | 1 | 21.1 | 2 | 1073 |
| S27 | PASSED | 1 | 32.1 | 9 | 28882 |
| S28 | PASSED | 1 | 18.3 | 8 | 3296 |
| S29 | PASSED | 1 | 2.3 | 1 | 588 |
| S30 | PASSED | 1 | 45.7 | 6 | 2488 |
| S31 | PASSED | 1 | 8.5 | 1 | 523 |
| S32 | PASSED | 1 | 14.4 | 3 | 1668 |
| S33 | PASSED | 1 | 119.8 | 24 | 325557 |
| S34 | PASSED | 1 | 26.6 | 8 | 3257 |
| S35 | PASSED | 1 | 22.2 | 6 | 2464 |
| S36 | PASSED | 1 | 28.1 | 7 | 2748 |
| S37 | PASSED | 1 | 13.0 | 4 | 1657 |
| S38 | PASSED | 1 | 42.4 | 10 | 4770 |
| S39 | PASSED | 1 | 10.3 | 4 | 2368 |
| S40 | PASSED | 1 | 29.0 | 10 | 48667 |
| S41 | PASSED | 1 | 29.2 | 6 | 79891 |
| S42 | PASSED | 1 | 44.1 | 9 | 76510 |
| S43 | PASSED | 1 | 15.1 | 3 | 1709 |
| S44 | PASSED | 1 | 14.3 | 3 | 1356 |
| S45 | PASSED | 1 | 9.6 | 3 | 42058 |
| S46 | PASSED | 1 | 4.2 | 0 | 0 |
| S47 | PASSED | 1 | 6.1 | 2 | 1182 |
| S48 | PASSED | 1 | 26.2 | 7 | 2985 |
| S49 | PASSED | 1 | 0.0 | 0 | 0 |
| S50 | PASSED | 1 | 24.4 | 5 | 1862 |

Totals: 50/50 passed, 51 attempts, 1160.3 scenario-seconds, 257 model requests, and 1,621,005 recorded tokens. S6 attempt 1 failed because the model produced a user-facing raw Python representation; the same monolithic run's bounded second attempt passed. No assertion was weakened and no provider fallback was used.

### Boundaries and local residue

- Work is edited only: no commit, push, deploy, migration, message send, or production verification was performed. `pnpm test` was not run.
- A local E2E Next server remains listening on `127.0.0.1:3100` as PID 2015 with cwd `e2e/.runtime/keytao-next`. It survived the killed predecessor job and the sandbox denied terminating it; it is not a production process. It should be stopped by the workspace owner with `kill -TERM 2015` when no longer needed.
