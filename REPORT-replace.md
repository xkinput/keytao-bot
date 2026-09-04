# S51 Incident Closure

## Replace-at-code

Status: implemented and verified locally on base HEAD `c29ffae`. This round was not committed, pushed, deployed, or exercised against production.

The requested scratchpad files `positional-grammar-task.md` and `numbered-eviction-report.md` were not present in the repository or available scratchpad locations. The committed S50 report and current source/test contracts were used for context, and this fallback report records the S51 closure.

### Grammar and authorization

Replace-at-code is now a first-class mutation shape. The real parser accepts and binds:

- `把 <code> 的「A」改成/改为/换成「B」`
- `编码<code> "A"改为"B"`
- `修改 <code>：A → B`
- equivalent Chinese/curly/straight/single/backtick/no-quote styles, spacing variants, and `→`, `->`, `=>`, `⇒`, `⟶`, `➡`
- the advertised wrapper `确认执行：把 <code> 的「A」改成「B」`

Authorization requires an exact old-word lookup binding at the requested code, an exact reviewed new-word/code/type binding, and exactly one `Delete A@code` plus one `Create B@code`. The route is deterministic before semantic model routing. Executable suggestions are rendered only after the same real parser and binding pre-check succeed, preserving advertise-as-contract.

For the incident command `编码nsl  "哪里"改为"那算了"`, one turn creates a sealed ticket containing:

- `Delete 哪里@nsl`
- `Create 那算了@nsl`

The preview truthfully states that `哪里` remains at `nslko`. A later `确认` executes only the sealed two-item delta and renders a receipt from the resulting trusted draft records. If the replacement word cannot pass the normal pronunciation/code review, the existing review flow remains authoritative.

### Read-only honesty and telemetry

The read-only guidance now asks the model to state the understood request in ordinary language and may expose only a validated `suggestedCommand`. It prohibits mechanism disclosure, predictions of future rejection, and generalizations from earlier failures. A final delivery-boundary guard applies the same rule even if generated output ignores the guidance.

When a message plainly expresses write intent but no closed authorization grammar matches it, the orchestrator emits the distinct `[grammar_gap] plain_write_intent=True authorization=False` marker. Recognized replace-at-code forms do not emit that marker.

### Ticket gate and delivery boundary

Confirmation and cancellation affordances are removed when no live ticket exists. The incident's no-ticket `确认执行：...` form is covered, while server-backed pending facts such as the S34 placeholder reminder remain displayable and are not mistaken for a local ticket.

The delivery boundary rejects raw Python or JSON container dumps in user-facing replies. When trusted turn records exist, it redraws the response deterministically from those records; otherwise it emits the truthful no-write result. Unit coverage includes both Python-style dict/list text and JSON literal text.

### Focused and offline verification

All commands ran from `/Users/rea/code/keytao-org/keytao-bot`. `pnpm test` was not run.

| Check | Result |
|---|---|
| `.venv/bin/python -m unittest test_memory_safety.ReplaceAtCodeS51RegressionTests` | `Ran 8 tests in 0.059s` / `OK` |
| `.venv/bin/python test_memory_safety.py` | `Ran 400 tests in 187.627s` / `OK` |
| `.venv/bin/python test_state_machine.py` | `Results: 1994/1994 passed, 0 failed` |
| `.venv/bin/python test_security_fixes.py` | `Results: 268/268 passed, 0 failed` |
| `.venv/bin/python test_review_gate.py` | `Results: 406/406 passed` |
| `.venv/bin/python test_llm_policy.py` | `Ran 10 tests` / `OK` |
| `.venv/bin/python test_word_discovery.py` | `Results: 290/290 passed` |
| `.venv/bin/python -m unittest e2e.test_safety` | `Ran 93 tests` / `OK` |
| `.venv/bin/python -m e2e.run --only S34` | `S34 PASSED`, attempt 1, exit 0 |
| `.venv/bin/python -m e2e.run --only S51` | `S51 PASSED`, attempt 1, exit 0 |
| `.venv/bin/python -m e2e.run --only S19` | `S19 PASSED`, attempt 1, exit 0; diagnostic check before the final FULL rerun |
| `python -m compileall -q keytao_bot e2e test_memory_safety.py` | exit 0 |
| `git diff --check` | exit 0 |

The S51 regression class has eight tests covering real-parser closure, exact ticket staging, trusted rendering and remaining-location disclosure, the S34 placeholder regression, exact sink bindings, Python/JSON literal redraw, no-ticket confirmation/jargon removal, and `grammar_gap` telemetry.

### Monolithic FULL rig

Acceptance run:

```text
E2E_OPENAI_API_KEY="$(<.e2e_key)" \
E2E_OPENAI_BASE_URL=https://api.deepseek.com \
E2E_OPENAI_MODEL=deepseek-v4-flash \
.venv/bin/python -m e2e.run
```

- Exit code: `0`
- Artifact: `e2e/artifacts/20260904T170907Z-a43137b4`
- Manifest run ID: `a43137b43cbc4d62a121ae882335ee29`
- Manifest base HEAD: `c29ffaee1d38efcf207225c17440bf26586e9fe0`
- Selected/result rows: `51/51`, exactly `S1` through `S51`
- Verdicts: `51 PASSED`, `0 FAILED`
- Model confirmation: manifest `llm.model=deepseek-v4-flash`; all 261 successful real LLM HTTP response records have the single unique model `deepseek-v4-flash`
- Provider fallback: none; `openai.AsyncOpenAI` recorded 261 successful exchanges and no other response model appears
- Safety boundary: manifest confirms the production URL was blocked before dispatch, remote database bindings were rejected, and no production-like admin/binding was used
- Attempts: S27 passed on attempt 2 after a model-answer-shape miss; every other scenario, including S51, passed on attempt 1

| Scenario | Verdict | Attempts | Seconds | LLM requests | Tokens |
|---|---:|---:|---:|---:|---:|
| S1 | PASSED | 1 | 21.9 | 2 | 1071 |
| S2 | PASSED | 1 | 15.5 | 1 | 593 |
| S3 | PASSED | 1 | 4.6 | 0 | 0 |
| S4 | PASSED | 1 | 10.3 | 3 | 48022 |
| S5 | PASSED | 1 | 10.2 | 0 | 0 |
| S6 | PASSED | 1 | 45.5 | 12 | 67173 |
| S7 | PASSED | 1 | 5.2 | 0 | 0 |
| S8 | PASSED | 1 | 15.3 | 0 | 0 |
| S9 | PASSED | 1 | 7.2 | 2 | 824 |
| S10 | PASSED | 1 | 30.0 | 7 | 104529 |
| S11 | PASSED | 1 | 21.2 | 5 | 99622 |
| S12 | PASSED | 1 | 19.7 | 5 | 98898 |
| S13 | PASSED | 1 | 6.7 | 3 | 47838 |
| S14 | PASSED | 1 | 7.0 | 3 | 1364 |
| S15 | PASSED | 1 | 17.6 | 4 | 1648 |
| S16 | PASSED | 1 | 37.2 | 8 | 52285 |
| S17 | PASSED | 1 | 25.1 | 6 | 2758 |
| S18 | PASSED | 1 | 10.8 | 3 | 1301 |
| S19 | PASSED | 1 | 60.7 | 10 | 51912 |
| S20 | PASSED | 1 | 15.7 | 4 | 46371 |
| S21 | PASSED | 1 | 56.8 | 14 | 119661 |
| S22 | PASSED | 1 | 43.0 | 9 | 96575 |
| S23 | PASSED | 1 | 36.8 | 5 | 78204 |
| S24 | PASSED | 1 | 4.5 | 2 | 824 |
| S25 | PASSED | 1 | 13.6 | 4 | 2014 |
| S26 | PASSED | 1 | 21.2 | 2 | 1079 |
| S27 | PASSED | 2 | 45.1 | 14 | 55623 |
| S28 | PASSED | 1 | 17.9 | 8 | 3296 |
| S29 | PASSED | 1 | 2.5 | 1 | 588 |
| S30 | PASSED | 1 | 45.6 | 6 | 2488 |
| S31 | PASSED | 1 | 8.5 | 1 | 513 |
| S32 | PASSED | 1 | 14.4 | 3 | 1670 |
| S33 | PASSED | 1 | 131.1 | 26 | 358344 |
| S34 | PASSED | 1 | 27.9 | 8 | 3261 |
| S35 | PASSED | 1 | 22.7 | 6 | 2464 |
| S36 | PASSED | 1 | 28.8 | 7 | 2748 |
| S37 | PASSED | 1 | 12.8 | 4 | 1672 |
| S38 | PASSED | 1 | 26.9 | 10 | 4750 |
| S39 | PASSED | 1 | 9.5 | 4 | 2368 |
| S40 | PASSED | 1 | 25.9 | 9 | 47588 |
| S41 | PASSED | 1 | 33.9 | 6 | 80229 |
| S42 | PASSED | 1 | 42.4 | 8 | 72553 |
| S43 | PASSED | 1 | 14.3 | 3 | 1712 |
| S44 | PASSED | 1 | 12.5 | 3 | 1356 |
| S45 | PASSED | 1 | 10.2 | 3 | 42188 |
| S46 | PASSED | 1 | 4.3 | 0 | 0 |
| S47 | PASSED | 1 | 4.8 | 2 | 1182 |
| S48 | PASSED | 1 | 26.5 | 7 | 2992 |
| S49 | PASSED | 1 | 0.0 | 0 | 0 |
| S50 | PASSED | 1 | 23.9 | 5 | 1842 |
| S51 | PASSED | 1 | 15.0 | 4 | 1567 |

S51's artifact records a real complete ticket, the exact preview/final item sets, `remainingCodes=["nslko"]`, all four advertised grammar closures, zero semantic-routing model requests, one normal review model request, and a truthful confirmed receipt.

Two earlier diagnostic FULL runs were not used as acceptance evidence: the first exposed an over-broad suggestion-prefix match that regressed S34 and was fixed with a narrow `确认执行` condition plus a regression test; the next ended 50/51 because S19 took two model-dependent nonconforming paths. S19 then passed its targeted diagnostic run, and the final monolithic run above passed all 51 rows in one process.

### Delivery boundary

Repository state remains edited-only on base HEAD `c29ffae`. No commit, push, deployment, production API call, or production resource change was performed.
