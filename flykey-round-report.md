# Flykey production incident round report

Date: 2026-08-17

Repository: `/Users/rea/code/keytao-org/keytao-bot`

Starting and current HEAD: `7ea76451a21558c8c565252d6aba9095cc18eb0f`

Scope boundary: local source and offline tests only. No network, SSH, production host, production API, billed model call, or live rig was used. Nothing was committed, pushed, deployed, or production-verified.

## Outcome

P1-P4 and the offline S25 contract are implemented. The six requested Python suites and `e2e.test_safety` are green; `git diff --check` is clean.

The central safety properties remain fail-closed: the parser never obtains executable code operands from model prose; combined submit is tied to the unique batch written by the same message and still consumes the server snapshot/digest/CAS confirmation envelope; an unresolved multi-sense pronunciation cannot enter semantic auto-pass.

## P1 — natural add verbs with explicit word and code

### Root cause

The execution grammar and advertised command vocabulary were not driven by one add-operation lexicon. `message_authorizes_mutation` and Create binding recognized the older imperative family, but natural forms such as `补上炒冷饭的 wlf 编码` did not establish a Create authorization even though both operands were literal in the user's text.

The clause parser also stripped generic command lead-ins before attempting the actual add verb, which could split or consume natural forms, and the Create branch did not explicitly reject a second target or a separate non-negated action in the same clause.

### Fix

- Added the shared `ADD_OPERATION_VERB_FORMS` / `ADD_OPERATION_VERB_PATTERN` source of truth, including `补`, `补上`, `补一个`, `补充`, `加上`, `添上`, `也加`, and `再加`; advertised add forms now derive from the same vocabulary: `keytao_bot/utils/pending_confirmation.py:26-68`.
- Reused that shared regex throughout authorization. `_parse_complete_add_clause` now tries the add verb before removing a lead-in and accepts the literal `词的 code 编码` shape: `keytao_bot/harness/authorization_grammar.py:366-410`.
- Kept the existing negation, question, quotation, and protected-word gates. Added explicit same-clause rejection for a second target and for a separate non-negated change/delete/recall/weight action: `keytao_bot/harness/authorization_grammar.py:2585-2641`; the Create binding applies both checks at `keytao_bot/harness/authorization_grammar.py:3428-3438`.
- A prefix such as `wlf` now reaches the existing occupied/duplicate-code path; it is not rejected merely because the slot is occupied.

Regression coverage: `test_memory_safety.py:5822` exercises every requested natural verb plus negation, question, quotation, other-action, and second-target refusals.

## P2 — state-backed CODE-CHOICE and live-state execution intent

### Root cause

The delivery contract understood state-backed word/batch advertisements but did not classify a model-authored numbered list of candidate codes for one word. Consequently the prose could be sent without a `PendingAddWord` record. On the next turn, mutation intent was classified from the text `3` alone, so it looked read-only and there was no trusted record from which to resolve the index.

### Fix

- Added kind-aware detection for a one-word numbered CODE-CHOICE advertisement and deterministic rendering from trusted data: `advertised_single_word_candidate_codes` and `render_server_backed_single_word_candidates` at `keytao_bot/utils/pending_confirmation.py:827-899`; the delivery contract carries `code_choice_advertisement` at `keytao_bot/utils/pending_confirmation.py:951-973` and classifies it at `keytao_bot/utils/pending_confirmation.py:1048-1052`.
- The orchestrator captures same-turn encode/candidate statuses and builds a `PendingAddWord` only when one word has an exact ordered record set. It saves the record before delivery and replaces model prose with the deterministic renderer: `keytao_bot/harness/orchestrator.py:816-867` and `keytao_bot/harness/orchestrator.py:1664-1734`.
- The final chat delivery guard verifies that the displayed ordered codes exactly match the actor-owned live record; otherwise it redraws from that record or refuses without writing: `keytao_bot/plugins/openai_chat.py:1852-1909`.
- Mutation-intent determination now consults an actor-owned live `PendingAddWord`, so a bare index and a `用 wlfoo`-style pick count as execution intent: `keytao_bot/plugins/chat_routing.py:1039-1083`, `keytao_bot/plugins/chat_routing.py:1171-1193`, and `keytao_bot/plugins/openai_chat.py:1478-1515`.
- Structural code/index intent is resolved before the optional intent model, including when the intent model is disabled: `keytao_bot/plugins/chat_commands.py:5720-5733`.

Execution operands still come only from `server_candidates`; the display parser merely identifies an advertisement and has no execution authority. A mismatch between displayed candidates and the record cannot execute.

Regression coverage: `test_memory_safety.py:9471` proves that untrusted model numbering is redrawn and that the subsequent number executes the code stored in the actor-owned record.

## P3a — truthful partial-success reporting

### Root cause

`AgentOrchestrator` tracked tool results while running, but the final loop-breaker composed failure/no-write copy from the last submit block alone. It did not receive a durable list of successful mutations from earlier in the same turn. Thus a successful `keytao_create_phrase` followed by a blocked `keytao_submit_batch` fell into a generic `无法执行，本次未写入` branch.

The exact no-write branches were introduced by:

- commit `01d5280bbce98c7897d9862e5231e64acd9179f7` — `fix: recover stale advertised assent instead of dead-ending`.

`git log -S` and `git blame 7ea7645` both attribute the relevant `这条指令按当前表述无法执行，本次未写入` finalizer branches to that commit. The regression was not the write itself; it was the later reply composition erasing the earlier success.

### Fix

- `run` now creates a same-turn `successful_write_receipts` ledger and every public final reply passes through `_finalize_reply` with that ledger: `keytao_bot/harness/orchestrator.py:284-305`.
- The loop records receipts only after the actual final mutating result. A confirmation preview is not a receipt; after auto-confirm, the confirmed execution result replaces the preview before receipt capture: `keytao_bot/harness/orchestrator.py:1365-1416`.
- `_successful_write_receipt` projects exact word, code, action, tool, and batch ID from successful results: `keytao_bot/harness/orchestrator.py:1906-1938`.
- `_finalize_reply` consults receipts before every legacy no-write/failed branch. On partial success it reports the exact written `word -> code` and batch, names the unfinished step and its reason, and emits the renderer-backed executable follow-up command. It also corrects a model-written no-write denial even when no structured `failure_state` survived: `keytao_bot/harness/orchestrator.py:2054-2134`.

Regression coverage in `test_memory_safety.py:12935` includes:

1. add succeeds, then submit is policy-blocked;
2. add succeeds, then a later tool errors;
3. confirmation is required but the user/model abandons it, so no success receipt exists and no write is claimed;
4. a model denial is overridden by an actual successful receipt.

## P3b — combined add-and-submit authorization

### Root cause

The submit gate required an explicit standalone submit command and ignored the `并提交` suffix of the already-advertised combined command. The standalone-only gate originated in:

- commit `7ff022ae6afb20595f779df60f6010bd510321e1` — `refactor: staged chat pipeline, prompt seam, grammar module split`.

That left `添加 词 码 并提交` authorized for Create but not Submit.

### Fix

- `explicit_combined_add_submit_item` parses exactly one literal add item followed by `并/并且/然后/再 + 提交/提审/送审`: `keytao_bot/harness/authorization_grammar.py:2219-2241`.
- The tool context carries the batch IDs successfully written during the current turn. Submit authorization accepts the combined form only when there is exactly one such batch and the submitted batch ID is that batch: `keytao_bot/harness/authorization_grammar.py:3250-3269`.
- The combined submit preview must be `success:false`, `requiresConfirmation:true`, match the batch, contain an integer content version, contain exactly the authorized one-item snapshot, contain all three 64-hex digests, and contain no stale/conflict/changed/uncertain marker: `keytao_bot/harness/orchestrator.py:1941-2003`.
- Only one exact matching successful Create receipt permits the confirmed replay. The replay keeps the existing mutation/server-warning confirmation flags, content-version CAS, server snapshot digest, warning digest, audit digest, actor ownership, and one-shot semantics: `keytao_bot/harness/orchestrator.py:2005-2051`.

Regression coverage: `test_memory_safety.py:13339` proves that a combined command submits only the unique batch just written by that same message and refuses cross-batch or malformed snapshot bindings.

## P4 — meaning-backed multi-sense pronunciation

### Root cause

Candidate ordering preferred whole-word authority before resolving whether multiple readings represented different senses. For `出圈`, this allowed authoritative `chū juàn` to outrank the intended modern `chū quān`. The downstream semantic auto-pass lane could then see a seemingly reviewable primary pronunciation rather than an unresolved sense choice.

### Exact rule implemented

For two or more distinct reading groups:

1. Keep every validated reading group and its codes visible.
2. Select and recommend one group only when the semantic/entity proposal is accepted, is `commonTransparent`, maps to exactly one candidate reading, has one pinyin syllable per character, gives a concrete meaning, has finite confidence, and meets `ENTITY_PRONUNCIATION_MIN_CONFIDENCE`.
3. When those conditions hold, move the meaning-supported reading to the front and use its code recommendation; whole-word authority alone does not override the meaning evidence.
4. Otherwise return ASK/sealed state: clear `recommendedCode`, set `pronunciationUnresolved` and manual review, list every reading/code group, and ask the user for the intended reading or meaning.
5. Semantic-context auto-pass additionally requires `multiSenseResolved`; an ambiguous multi-sense choice cannot pass even if its other semantic/commonness checks succeed.

Implementation: resolver at `keytao_bot/utils/keytao_review.py:1960-2044`, invocation after validated reading groups at `keytao_bot/utils/keytao_review.py:2626-2630`, recommendation/ASK result at `keytao_bot/utils/keytao_review.py:2692-2769`, semantic prompt contract at `keytao_bot/utils/keytao_review.py:3298-3309`, and the mandatory auto-pass gate at `keytao_bot/utils/keytao_review.py:5070-5111`.

Regression coverage: `test_review_gate.py:4530-4656` proves that modern-meaning evidence selects `chu quan` for `出圈` while both groups remain visible, and ambiguous evidence yields ASK with no recommendation and no semantic auto-pass.

## E2E S25 design

S25 is registered after S24 and uses one flykey/occupied-series word, `炒冷饭`, with prefix `wlf` and selected free code `wlfoo`: `e2e/scenarios.py:2713-2929`.

It has three isolated actor-state phases so stale draft state cannot make a later phase pass accidentally:

1. `补上炒冷饭的 wlf 编码` must reach the real Create attempt with the literal word/prefix and the existing occupied/duplicate confirmation path.
2. A clean query turn must render a numbered, state-backed series; the scenario derives the displayed index of `wlfoo`, replies with the bare number, and requires the resulting exact record-backed draft item.
3. After another clean/reset, `添加 炒冷饭 wlfoo 并提交` must create and submit/approve exactly that one item, use at most one server-bound confirmation, and emit none of the forbidden refusal/no-write copy.

The deterministic ZDIC fixture for the whole word and characters is at `e2e/zdic_seed.py:236-249`; the offline FakeContext scenario assertion is at `e2e/test_safety.py:1957-2120`; the scenario is documented in `e2e/README.md`.

The live rig was intentionally not run because this round explicitly forbids network/production access. The offline contract and all safety assertions are green.

## Verification

### 1. `test_memory_safety.py`

Command: `.venv/bin/python test_memory_safety.py`

```text
----------------------------------------------------------------------
Ran 266 tests in 136.698s

OK
```

Log: `/private/tmp/flykey-test-memory-safety-final.log`

### 2. `test_state_machine.py`

Command: `.venv/bin/python test_state_machine.py`

```text
Results: 1479/1479 passed, 0 failed
✅ ALL TESTS PASSED
```

Log: `/private/tmp/flykey-test-state-machine-final.log`

### 3. `test_security_fixes.py`

Command: `.venv/bin/python test_security_fixes.py`

```text
Results: 268/268 passed, 0 failed
```

Log: `/private/tmp/flykey-test-security-fixes.log`

### 4. `test_review_gate.py`

Command: `.venv/bin/python test_review_gate.py`

```text
Results: 379/379 passed
✅ ALL TESTS PASSED
```

Log: `/private/tmp/flykey-test-review-gate.log`

### 5. `test_llm_policy.py`

Command: `.venv/bin/python test_llm_policy.py`

```text
Ran 9 tests in 0.107s
OK
```

Log: `/private/tmp/flykey-test-llm-policy.log`

### 6. `test_word_discovery.py`

Command: `.venv/bin/python test_word_discovery.py`

```text
Results: 290/290 passed
✅ ALL TESTS PASSED
```

Log: `/private/tmp/flykey-test-word-discovery.log`

### 7. Offline E2E safety suite

Command: `.venv/bin/python -m unittest e2e.test_safety`

```text
Ran 55 tests in 0.502s
OK
```

Log: `/private/tmp/flykey-e2e-test-safety-final.log`

### Diff hygiene

`git diff --check`: exit status `0`, no output.

## Handoff boundary

- Edited: yes, 14 tracked files, 1,982 insertions and 25 deletions at final inspection.
- Committed: no.
- Pushed: no.
- Deployed: no.
- Production-verified: no; explicitly prohibited and the host was not contacted.
- Live S25: not run; explicitly prohibited. Only the deterministic offline scenario contract was run.
