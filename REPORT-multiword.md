# S54 — bare multi-word query closure

## Status and evidence boundary

Implemented locally on baseline HEAD
`2c88186d784d3624de33b56351f5b1cec0d6d034`. The work is edited-only: no
commit, push, deployment, production API, production database, or production
server was touched. The banned `pnpm test` command was not run.

S54 is complete. All six required offline suites, `e2e.test_safety`, and the
focused S54 checks passed. The accepted sixth fresh monolithic FULL invocation,
`20260906T181247Z-55a0b25f`, passed **S1–S54, all on attempt 1**, with process
exit 0. Its manifest, summary, 54 scenario artifacts, and final verdict table
agree. The 16 frozen Python source files were unchanged through completion.

Earlier stopped runs exposed the routing and delivery issues documented below;
none is counted toward acceptance. The final run uses only real
`deepseek-v4-flash` exchanges at `api.deepseek.com`, with no model or provider
fallback, against the local bot/Next fixtures. This closes the requested local
deploy gate; it is not a deployment or production verification.

The design follows `REPORT-deploy-gap.md` for record-first selectors and
executable suggestions, and the final Phase 2 section of `REPORT-reading.md`
for the S53 reading ladder and truthful evidence labels.

## Diagnosis and route

The incident diagnosis is confirmed. The former bare-word route accepted one
lexical word; a whitespace-separated list could reach the general orchestrator,
where raw `keytao_encode` candidates were narrated without the reviewed-add
pipeline or a persisted candidate capability. A deterministic batch renderer
also existed, but its legacy scope shape could not retain the complete review
evidence used by the single-word display.

One naming detail in the brief differs from this checkout:
`_MAX_AUTHORIZED_MULTI_ADD_ITEMS` does not exist at the baseline (`git grep`
finds no match). The new explicit cap is
`_MAX_BARE_MULTI_WORD_QUERY_ITEMS = 10`. Eleven or more distinct lexical words
are refused before tool execution with a request to split the query.

`_bare_multi_word_query_words` recognizes two or more Chinese lexical words
separated by whitespace, `、`, `，`, or ASCII comma. It strips the existing
address prefix, preserves order, deduplicates words, and rejects command-like
tokens or clauses. `_try_handle_simple_single_word_query` dispatches these
lists to `_prepare_multi_word_query` before the general model flow. Existing
`parse_advertised_set_reference` and `message_mentions_change_request`
predicates exclude set-subtraction and action commands before this route;
explicit batch-query prefixes and negative-action endings are also excluded.
These checks only prevent misrouting and grant no mutation authority.

The batch collector calls the same single-word function sequentially for each
word with an internal `_prepared_scopes` collector. Each absent word therefore
uses the existing lookup, pending-item reminders, public
`keytao_prepare_reviewed_add`, deterministic `_format_reviewed_add_prompt`, and
`select_candidate_inventory` path. It does not start a second independent
review implementation. Per-word collection does not persist temporary
`PendingAddWord` records. When at least one actionable scope is ready, exactly
one actor-owned `PendingToolConfirm` is saved, and only then is the combined
selector rendered. If every word is already present or has no usable scope,
the route clears the old record and returns the informational blocks without
creating a candidate record.
A failed save or invalid render produces a truthful retry response.

Words found in the dictionary use `already_existing_word_copy` with their exact
codes. They are excluded from the actionable item and scope sets. A mixed
query can consequently have one actionable word while keeping the same
word-scoped selector shape. The original query words and these other blocks
are retained in the batch record so the delivery boundary can redraw the
complete reply.

## Persisted record shape

The record uses the existing durable pending-state store and owner, nonce,
execution, and confirmation-source protections. No schema migration or new
storage family was added. Its state is:

```text
PendingToolConfirm
  function_name: keytao_batch_add_to_draft
  confirmation_source: local_preview
  args:
    items: one Create item per actionable word at its recommendation
    _reviewed_multi_word: true
    _query_words: complete original word list
    _query_other_blocks: existing-word or unresolved-result blocks
    _candidate_scopes:
      word
      candidates: ordered (code, occupied) pairs
      occupiedWords: exact occupants per code
      orderingAssessments: server-backed comparator snapshot
      reviewedState: complete _pending_add_word_payload
      reviewedPrompt: deterministic reviewed-add prompt
```

`reviewedState` retains the word, recommendation, display and server candidate
inventories, occupants, weighted entries, ordering assessments, per-code review
remarks, code-to-reading map, pronunciation recommendations, manual-review
boolean, and reason. The batch items carry the corresponding review seal.

The scope validator reconstructs the saved single-word capability, including
the JSON list/tuple round trip, and checks that word, recommendation,
inventories, occupancy, and ordering metadata agree. The renderer also checks
that the reviewed prompt's ordered candidate codes match the sealed inventory.
Malformed or mismatched fields cannot silently become an executable selector.
SQLite reload and cross-actor rejection are covered in the focused route tests.

## Shared rendering and reading selection

`render_server_backed_batch_candidates` now recognizes rich reviewed scopes and
delegates every block to `render_server_backed_single_word_lookup`. Its
`include_controls=False` option suppresses per-word actions while preserving
the same reading/source line, auto-review verdict and reason, numbered
occupancy rows, and recommendation. The existing recommendation helper has a
matching option for descriptive reorder copy; it does not advertise an
ambiguous bare `不重排选 3` inside a batch. The footer is emitted once after all
blocks. Legacy batch scopes retain their existing behavior. This adds no third
renderer.

The stored inventory comes from the same `select_candidate_inventory` call as
single-word queries. For the incident fixture, `大端` is limited to
`dsdt`, `dsdtv`, `dsdtvo`; `小端` is limited to
`xcdt`, `xcdti`, `xcdtio`. The `dhdt*` and `thdt*` chains do not enter the
resolved `da duan` display. Unresolved pronunciation remains governed by the
existing S53 ladder and its labelled reading groups, rather than flattening
raw encode candidates into one ordinal list.

The common renderer can receive marked or normalized pinyin. The real
fixture currently renders `da duan` and `xiao duan`; the check normalizes these
through the real pinyin normalizer and also checks the exact reviewed source
line and the selected code chain. Tone-mark typography is not used as a
substitute for verifying the reading.

## Canonical selection and executable closure

The complete batch footer advertises these three renderer-built commands:

```text
- 「加入」
- 「加入并提交」
- 「大端 3，小端 2」
```

The canonical per-word form is **word, space, ordinal-or-code**, with pairs
separated by `，` or `、` (ASCII comma is also accepted). Examples are
`大端 3，小端 2`, `大端 dsdtvo，小端 xcdti`, and the partial selection `大端 3`.
The display advertises the ordinal example computed from each sealed
recommendation. It does not advertise a second verb-position syntax.

This form makes repeated per-word ordinals unambiguous, remains the same when
only one word is actionable, and represents a partial selection without
implicitly choosing the other words. Duplicate words, trailing separators,
out-of-range ordinals, alien words or codes, and narration do not produce a
partially interpreted command.

`parse_reviewed_multi_word_selection` is a closed parser and grants no generic
mutation authority. `_resolve_multi_word_pending_candidate_selection` and
`message_authorizes_live_pending_mutation` bind its result only against the
current persisted record. The existing whole-message envelope handler accepts
the exact rendered bullet or quoted command; quoted text supplies no target
inventory. The selected item set contains only the named words. The reply
lists omitted words as unselected and not added. A completed partial selection
consumes the original candidate record; it does not create a second executable
ticket for the omitted words.

Explicit ordinal/code selection chooses the exact code and clears automatic
reorder advice for execution. Choosing an occupied code explicitly therefore
uses the existing duplicate-add path with mandatory administrator review.

Bare `加入` and `加入并提交` consume every actionable recommendation. A
recommended reorder uses the sealed reorder plan by default. The execution
path projects pinyin and candidate-chain capabilities from `reviewedState` into
the actual batch/shift sink, rather than regenerating candidates or trusting
fields supplied in prose. Warning replay retains the reviewed capabilities
but cannot reopen ordinal selection after a server warning has sealed the
operation.

The multi-reorder path binds expected occupants for each reorder target.
Ordinary additions, including occupied-code recommendations, are companion
items in the same atomic plan; they do not shift their occupants. The existing
nonempty reorder-occupancy requirement, cross-plan word/code conflict checks,
plan digest, and strict batch CAS remain enforced. No server API changed.
Both reordered targets and ordinary companions retain their exact reviewed
reading through the harness and the final code-validation function. Unknown
warning types or warnings for different targets stop at a new confirmation
ticket; only known warnings bound to the exact requested additions may replay
automatically. Main-chat warning replay uses the same executor and preserves
the saved reading capability.

The footer validates its assent strings and canonical selection through the
real parsers before rendering. S54 then feeds every advertised command, both
its content and the full rendered bullet, through the real live binding
checker. The combined reply must also pass the final live-record contract.

## Delivery boundary: no orphan numbered candidate lists

`_NUMBERED_CANDIDATE_ROW_RE` detects the ordinal/code/occupancy-or-recommendation
shape independently of whether a footer advertises a command.
`_numbered_candidates_match_record` requires an unconsumed trusted candidate
record. Single-word records must retain the exact server inventory; batch
replies must match the complete deterministic rendering of the current record.
That comparison strips line-edge whitespace and ignores empty lines because
the QQ text adapter removes blank separators. It does not remove or normalize
word, code, occupancy, reading, or command content.

An unmatched list is redrawn from the live record when that redraw itself
passes the binding checker. Otherwise the delivery boundary replaces it with:

```text
当前没有可验证的可执行操作，本次未写入。
```

The internal `ServerBackedQueryReply` marker alone cannot bypass this rule.
Explicit read-only contexts remain unable to create pending write state, so a
recordless numbered list from such a context is now refused at delivery.
Source-level lookup results remain available to their internal checks.

The end-to-end receipt regression exposed another normalization boundary:
after a successful batch submission, `_stage_normalize_response` could append
fresh candidate guidance to the completed receipt, causing its truthful batch
link to be rejected as an orphan action advertisement. Guidance is now added
only when `_reply_carries_live_candidate_state` confirms a current candidate
record. Completed receipts retain their result and link.

## S54 scenario and local fixtures

The new contiguous scenario pack runs S1 through S54. S54 uses the local Next
fixture and bot message path, with real DeepSeek model exchanges. Its fixture
occupants are `打断@dsdt`, `大段@dsdtv`, and `小段@xcdt`; controlled whole-word
reading seeds resolve the incident reading. `肌群` is the authoritative
whole-word-miss control. These dictionary seeds are local fixtures, not live
dictionary or external web-provider evidence.

| Case | Required checks |
|---|---|
| `大端 小端` then `加入并提交` | one record before delivery; each absent word goes through reviewed-add; both recommendations submitted in one turn with the exact batch link; no review regeneration |
| `大端、小端` then `大端 3，小端 2` | exactly `Create 大端@dsdtvo` and `Create 小端@xcdti` |
| `大端，小端` then `大端 3` | only `大端@dsdtvo`; reply states `小端` was unselected |
| `小端` already exists at `xcdtio` | existing-word block; only `大端` is actionable; no duplicate `小端` write |
| `大端 小端 肌群` then `加入` | same pipeline and one record for all three actionable words; every recommendation written |

Each discovery case observes store writes and actual reply delivery, checks
that the persisted nonce already exists at delivery, inspects per-word
inventories and public review calls, and performs parser/binding closure.
Fixture cleanup is restricted to the rig-owned local words and batches.

## Final offline verification tails

All commands use the repository's `.venv/bin/python`. Logs are under
`/private/tmp/keytao-s54-checks`. This table records observed tails, not inferred
success from a command being started. The eight final logs for the six suites,
E2E safety, and combined focused checks are also archived under
`e2e/artifacts/20260906T181247Z-55a0b25f/offline-checks/`.

| Command | Observed result |
|---|---|
| `.venv/bin/python test_memory_safety.py` | exit 0; `Ran 407 tests in 191.550s`, `OK` (`memory-final-tail.log`). Initial complete run was `Ran 403 tests in 204.276s`, `FAILED (failures=7)`; the seven obsolete delivery assertions were corrected as described below. Added S27 and first-turn word-set controls raise the final count by four. |
| `.venv/bin/python test_state_machine.py` | exit 0; `Results: 2013/2013 passed, 0 failed`, `ALL TESTS PASSED` (`state-final-tail.log`) |
| `.venv/bin/python test_security_fixes.py` | exit 0; `Results: 268/268 passed, 0 failed` (`security-final-tail.log`) |
| `.venv/bin/python test_review_gate.py` | exit 0; `Results: 443/443 passed`, `ALL TESTS PASSED` (`review-final-tail.log`) |
| `.venv/bin/python test_llm_policy.py` | exit 0; `Ran 10 tests in 0.119s`, `OK` (`llm-final-tail.log`) |
| `.venv/bin/python test_word_discovery.py` | exit 0; `Results: 290/290 passed`, `ALL TESTS PASSED` (`discovery-final-tail.log`) |
| `.venv/bin/python -m e2e.test_safety` | exit 0; `Ran 96 tests in 0.635s`, `OK` (`remaining-final-tail.log`) |
| `.venv/bin/python -m unittest test_s54_multiword test_s54_renderer test_s54_selection` | exit 0; standalone `Ran 37 tests in 0.070s`, `OK` (`s19-s54-final.log`); rerun after the final tail fix together with five affected methods: `Ran 42 tests in 0.082s`, `OK` (`s27-tail-green.log`) |
| Four adjusted memory-boundary methods | `Ran 4 tests in 0.105s`, `OK` (`test_memory_safety_boundary_focused.log`) |
| Renderer-focused initial regression | Before fix: four failures and one missing-API error; after shared-renderer changes: `Ran 6 tests`, `OK` |

All six offline suites, `e2e.test_safety`, and the 37 focused checks passed
after the final mixed-execution, warning-replay, delivery, S19 routing, and S27
nominal-phrase, first-turn word-set, and stale meta-answer tail fixes. The
focused checks include the actual main chat stage, durable SQLite reload,
the actual harness capability binder, and the real code-validation function
inside the mixed shift plan. Independent incremental static review found no
remaining blocker. The final 16 changed Python files were fingerprinted after
the last code change. Their hashes matched immediately before the accepted
FULL invocation and after it completed. The accepted artifact contains
`source-freeze.json` and `completion-verification.json` with the frozen
inventory and final checks. Only this report was finalized afterward.

The memory assertion correction affects exactly these four methods (the
re-review method has four layouts, giving seven failing cases):

- `test_binding_advertisement_uses_only_same_turn_review_records`;
- `test_mixed_advertisement_prefers_specific_review_bindings`;
- `test_read_only_rereview_advertisement_reestablishes_live_ticket`;
- `test_same_chain_batch_candidates_allocate_distinct_slots_by_priority`.

Only `delivered == result` was replaced by an exact truthful no-write response
and an assertion that delivery contains no numbered candidate row. Every
source snapshot, candidate binding, `record is None`, `set_count == 0`, review
call, ranking call, and occupancy-snapshot assertion was retained. The
corresponding state-machine read-only display assertion now requires refusal
of a recordless list. This is the explicitly requested D policy change, not
permission to persist state for read-only callers or to weaken source checks.
The E2E safety pack assertion was extended from contiguous S1–S53 to S1–S54.

## Targeted run evidence and corrections

These are supporting diagnostics only. None is an accepted deployment gate.
Their manifests record `baseHost=api.deepseek.com`,
`model=deepseek-v4-flash`, and a present API key. `.e2e_key` was checked as
non-empty without printing its contents.

| Artifact | Observed result | Diagnosis and disposition |
|---|---|---|
| `e2e/artifacts/20260906T151653Z-39fb7053` | S54 FAILED, 2 attempts; result row 28.7s, 4 model requests, 2,212 tokens | The reply already retained `审词：读音 da duan`, the source line, and the exact `dsdt*` chain. The new scenario initially required tone-mark typography. Corrected the assertion to use the real pinyin normalizer while retaining exact source-line, pronunciation-set, and code-chain checks. |
| `e2e/artifacts/20260906T153525Z-019911c8` | S54 FAILED, 2 attempts; result row 38.9s, 4 model requests, 2,212 tokens | Add-and-submit failed its receipt/link assertion because completed-result normalization appended candidate guidance after the record was consumed. Fixed `_stage_normalize_response` to add guidance only for a live candidate reply; added a completed-receipt regression. |
| `e2e/artifacts/20260906T153852Z-a89d6ff6` | S54 FAILED, 2 attempts; result row 56.6s, 10 model requests, 5,759 tokens | The existing-word case rendered `小端` as already present and advertised `大端 3`. Every command and full bullet passed its binder; the final whole-reply equality failed because QQ removed blank separators between the existing-word block and the selector. Equality now compares stripped nonempty lines without weakening any word/code/occupancy content. |
| `e2e/artifacts/20260906T155104Z-ff190400` | exit 0; S54 PASSED, attempt 1; 39.5s, 7 model requests, 4,007 tokens | All five variants passed. Every discovery saved exactly one candidate record before delivery; every command and complete bullet passed the parser and binding checker. Both recommended submission and explicit/partial selections matched actual local Next rows. |
| `e2e/artifacts/20260906T161312Z-f7f6c720` | exit 0; S19 PASSED, attempt 1; 59.3s, 10 model requests, 52,090 tokens | After the first FULL failure, the routing guard restored the original chunked review flow and all nine expected draft items. The scenario and its progress assertions are unchanged. This targeted pass does not repair the rejected FULL artifact. |
| `e2e/artifacts/20260906T163737Z-eb62f8e8` | exit 1; S33 FAILED, 2 attempts; 12.6s, 2 model requests, 1,172 tokens | The new S33 assertion incorrectly compared the compact persisted ordering snapshot with the richer public review object. Both recommended `缩手@sled`. The exact `126 vs 3` reason is in the public review and trusted `reviewedPrompt`; the delivered shared renderer shows the commonness recommendation, concrete reorder, and non-reorder alternative. Corrected the assertion to verify these distinct evidence layers, and replayed the complete assertion block against both actual artifacts successfully. No production behavior changed for this mismatch. |
| `e2e/artifacts/20260906T164258Z-1d467f27` | exit 0; S33 PASSED, attempt 1; 134.1s, 22 model requests, 306,739 tokens | Real rerun passed the strengthened stored/public review comparison, delivered recommendation and auto-review reason, and every original zero-write, same-chain, same-code, six-code, and collision control. |
| `e2e/artifacts/20260906T170630Z-9d713945` | exit 0; S27 PASSED, attempt 1; 49.5s, 7 model requests, 27,270 tokens | Real rerun preserved the direct account-binding answer after the narrow nominal-phrase correction; original unbound/bound-user and zero-tool meta-question assertions remain unchanged. |
| `e2e/artifacts/20260906T174404Z-f16e0d0b` | exit 0; S19 PASSED, attempt 1; 62.7s, 10 model requests, 51,944 tokens | After the first-turn snapshot fix, the real scenario passed its original alien-exclusion ASK, `8/9` progress, exact nine-word confirmation and write assertions. |
| `e2e/artifacts/20260906T180827Z-f03d5bcb` | exit 0; S27 PASSED, attempt 1; 27.9s, 7 model requests, 27,222 tokens | Real S27 passed after sharing the existing deterministic binding answer for unsafe candidate tails; all original scenario assertions remain intact. |

The failed process rows above report their own recorded timing and usage;
they are not accumulated cost estimates for all retries. No failed targeted
run is presented as S54 completion.

## Accepted monolithic FULL gate

Invocation after targeted and offline closure, using `/bin/sh`:

```text
E2E_ARTIFACT_RETENTION=1000 \
E2E_OPENAI_API_KEY="$(cat .e2e_key)" \
E2E_OPENAI_BASE_URL=https://api.deepseek.com \
E2E_OPENAI_MODEL=deepseek-v4-flash \
.venv/bin/python -u -m e2e.run
```

Accepted artifact:
[`e2e/artifacts/20260906T181247Z-55a0b25f/manifest.json`](e2e/artifacts/20260906T181247Z-55a0b25f/manifest.json).
The fresh process selected exactly S1–S54 and exited 0. All 54 results passed
on attempt 1, with no failed scenario attempt or retry. `summary.json` equals
the manifest's result array; all 54 exact `S<number>-attempt-1.json` files
independently report `PASSED`. A post-run archive check initially included
dictionary warm-up files in a broad glob; using the exact scenario-file shape
corrected that bookkeeping check without changing source or rig results.

The run started at `2026-09-06 18:12:47 UTC` and completed at
`2026-09-06 18:46:09 UTC` (33m22s elapsed; scenario durations sum to 1,292.1s).
It recorded 284 successful real HTTP model exchanges and 1,600,865 tokens.
Every recorded model is `deepseek-v4-flash`; the client is
`openai.AsyncOpenAI`, and no fake client is present in the message path.
Provider monetary billing was not available locally and is not estimated.

The rig used `http://localhost:3100` and PostgreSQL at `localhost:5432`.
It reused the existing local Next server (`startedByRig=false`), so no new
server ownership or cleanup claim is made for this run. The manifest verifies
that production URLs are blocked before dispatch and remote databases,
production-like bindings, and production-like administrators are rejected.
The four safety checks are all true. Dictionary/search seeds remain fixtures.

The final cross-check is recorded in
[`completion-verification.json`](e2e/artifacts/20260906T181247Z-55a0b25f/completion-verification.json):
baseline HEAD unchanged, all 16 source hashes unchanged, `git diff --check`
passed, and no staged paths. The full process log is
`/private/tmp/keytao-s54-checks/full-s1-s54-run6.log`.

| Scenario | Verdict | Attempts | Seconds | LLM requests | Tokens |
|---|---|---|---|---|---|
| S1 | PASSED | 1 | 20.6 | 2 | 1,059 |
| S2 | PASSED | 1 | 15.4 | 1 | 593 |
| S3 | PASSED | 1 | 4.7 | 0 | 0 |
| S4 | PASSED | 1 | 11.4 | 3 | 48,514 |
| S5 | PASSED | 1 | 10.4 | 0 | 0 |
| S6 | PASSED | 1 | 58.3 | 14 | 90,852 |
| S7 | PASSED | 1 | 5.6 | 0 | 0 |
| S8 | PASSED | 1 | 15.4 | 0 | 0 |
| S9 | PASSED | 1 | 6.4 | 2 | 824 |
| S10 | PASSED | 1 | 33.6 | 7 | 107,373 |
| S11 | PASSED | 1 | 16.5 | 5 | 98,573 |
| S12 | PASSED | 1 | 22.3 | 5 | 100,643 |
| S13 | PASSED | 1 | 6.5 | 3 | 47,997 |
| S14 | PASSED | 1 | 6.9 | 3 | 1,363 |
| S15 | PASSED | 1 | 15.1 | 4 | 1,648 |
| S16 | PASSED | 1 | 34.8 | 8 | 52,187 |
| S17 | PASSED | 1 | 23.6 | 6 | 2,735 |
| S18 | PASSED | 1 | 10.4 | 3 | 1,512 |
| S19 | PASSED | 1 | 64.1 | 10 | 52,561 |
| S20 | PASSED | 1 | 23.7 | 6 | 51,142 |
| S21 | PASSED | 1 | 37.1 | 12 | 118,266 |
| S22 | PASSED | 1 | 26.4 | 9 | 97,287 |
| S23 | PASSED | 1 | 53.3 | 7 | 86,738 |
| S24 | PASSED | 1 | 5.1 | 2 | 824 |
| S25 | PASSED | 1 | 13.6 | 4 | 2,014 |
| S26 | PASSED | 1 | 20.6 | 2 | 1,094 |
| S27 | PASSED | 1 | 28.7 | 7 | 27,100 |
| S28 | PASSED | 1 | 18.3 | 8 | 3,296 |
| S29 | PASSED | 1 | 2.4 | 1 | 588 |
| S30 | PASSED | 1 | 29.8 | 6 | 2,488 |
| S31 | PASSED | 1 | 8.6 | 1 | 524 |
| S32 | PASSED | 1 | 13.6 | 3 | 1,668 |
| S33 | PASSED | 1 | 128.7 | 23 | 311,230 |
| S34 | PASSED | 1 | 26.7 | 8 | 3,256 |
| S35 | PASSED | 1 | 21.9 | 6 | 2,464 |
| S36 | PASSED | 1 | 27.6 | 7 | 2,748 |
| S37 | PASSED | 1 | 13.5 | 4 | 1,699 |
| S38 | PASSED | 1 | 26.6 | 10 | 4,993 |
| S39 | PASSED | 1 | 10.1 | 4 | 2,368 |
| S40 | PASSED | 1 | 48.5 | 9 | 48,077 |
| S41 | PASSED | 1 | 28.4 | 6 | 80,134 |
| S42 | PASSED | 1 | 45.3 | 9 | 76,241 |
| S43 | PASSED | 1 | 14.4 | 3 | 1,694 |
| S44 | PASSED | 1 | 12.6 | 3 | 1,353 |
| S45 | PASSED | 1 | 8.4 | 3 | 41,901 |
| S46 | PASSED | 1 | 4.5 | 0 | 0 |
| S47 | PASSED | 1 | 5.2 | 2 | 1,182 |
| S48 | PASSED | 1 | 25.2 | 7 | 3,018 |
| S49 | PASSED | 1 | 0.0 | 0 | 0 |
| S50 | PASSED | 1 | 22.9 | 5 | 1,866 |
| S51 | PASSED | 1 | 30.0 | 4 | 1,568 |
| S52 | PASSED | 1 | 18.6 | 3 | 1,378 |
| S53 | PASSED | 1 | 75.6 | 17 | 8,177 |
| S54 | PASSED | 1 | 34.0 | 7 | 4,055 |

S54's final artifact confirms all five cases saved exactly one candidate
record before reply delivery. Every advertised string and full bullet passed
the real parser and actor-owned live-record binding check. The default
`加入并提交` wrote both recommended items and submitted their batch; explicit
selection wrote the exact two codes; partial selection omitted `小端`; the
existing-word variant added only `大端`; and the three-word variant added the
unknown-word control through the same reviewed pipeline.

## Earlier FULL invocations and fixes

Every invocation below was discarded as a gate result. Each subsequent FULL
started from S1 in a fresh process; targeted reruns did not repair prior runs.

Rejected FULL 1: `e2e/artifacts/20260906T155916Z-97376be6`, log
`/private/tmp/keytao-s54-checks/full-s1-s54-run1.log`. S1–S18 passed on their
first attempts. S19 attempt 1 failed its existing `8/9` chunk-progress
assertion: the new bare-list route treated
`天选打工人先不要，其他可以加，沙县小吃也不要` as three lexical words, bypassing
the live word-set selection. The process was interrupted with exit 130 and
reported that its owned local server stopped. It has no completed all-scenario
summary and is not an accepted gate. The built-in retry had begun but was
stopped; none of its results can close this run.

The fix reuses the existing set-reference parser and change-request precheck,
with regressions for the original sentence, other exclusion forms, ordinary
action clauses, all-negative clauses, and explicit batch queries. Actual
lexical lists such as `加班 加工 加法` still take the deterministic route. The
S19 scenario and its progress assertion were not changed. An independent
offline scan of 3,173 S20–S53 literal strings found no further input that must
be misrouted in its actual scenario context.

Stopped FULL 2: `e2e/artifacts/20260906T162022Z-61162182`, log
`/private/tmp/keytao-s54-checks/full-s1-s54-run2.log`. S1–S16 passed on their
first attempts and no completed scenario failed. The process was deliberately
stopped before S33, with exit 130 and owned-server cleanup, after independent
offline verification found a superseded S33 query expectation. This is not a
claim that S33 failed during this invocation.

The actual rig reference database has `缩手` frequency 126/presence 2 and
`所受` frequency 3/presence 0. Running the existing commonness comparator and
recommendation applier with external fallback forbidden gives
`front_more_common`, `frequency_ratio`, `newCode=sled`, `freeCode=sleda`.
The old S33 pure query only encoded words and fixed `缩手@sleda`; S54 requires
that same bare two-word query to use reviewed recommendations instead. The
S33 assertion now requires the concrete `缩手@sled` recommendation
and its persisted reorder assessment, retaining the original zero-write,
occupant-display, and later collision checks. A new complete invocation is
required after that assertion correction.

Rejected FULL 3: `e2e/artifacts/20260906T164632Z-81935751`, log
`/private/tmp/keytao-s54-checks/full-s1-s54-run3.log`. S1–S26 passed on their
first attempts. S27 attempt 1 failed because the direct model answer to
`你是否会先确认对方有没有绑定账号？` was replaced with the character-data
fallback. The shared contract detected `这些写入草稿` inside the descriptive
noun phrase `这些写入草稿的动作`. Its only positive contract field was
`word_set_advertisement`; no candidate list or copyable command was present.
Both this regex and the finalizer branch existed unchanged at the baseline.
S27's built-in second attempt passed, but that does not erase the first
failure. The process was stopped with exit 130 and owned-server cleanup as
S28 was starting. No all-scenario summary exists for this run.

The correction is limited to excluding the nominal suffixes `的动作`,
`的操作`, `的行为`, and `的流程` from the word-set advertisement regex.
Direct commands, quoted assent, and D's independent numbered-row guard keep
their existing checks. S27's E2E assertion and S45's interrogative finalizer
remain unchanged. The actual model reply is added to the existing S27 offline
regression with positive advertisement controls before a fresh FULL run.

Rejected FULL 4: `e2e/artifacts/20260906T171117Z-2f703b58`, log
`/private/tmp/keytao-s54-checks/full-s1-s54-run4.log`. S1–S18 passed on their
first attempts. S19 attempt 1 reached the required `8/9` progress but its final
candidate reply was refused by D. The first-turn model described `加入词库`
and `写入草稿`, while the existing snapshot-creation gate depended on narrower
model advertisement wording. It failed to create the trusted 11-word record,
despite the user's explicit request to advertise a future batch assent.
The later exclusion message consequently had no trusted snapshot to bind;
its nine public reviews did not grant mutation authority. The orchestrator
returned a recordless read-only selector, and D correctly rejected it. The
process was stopped with exit 130 and owned-server cleanup; its started retry
does not count as acceptance.

The correction uses the existing explicit `requests_future_batch_assent_offer`
predicate in the first-turn word-set record gate. Targets still come only
from the trusted absent-word result, and no write authority is inferred from
model prose or from later review calls. The existing snapshot token,
actor-scoped exclusion resolver, replacement CAS, and shared renderer remain
the path from discovery to the final candidate record. Ordinary read-only
queries and ambiguous multiple snapshots do not gain a guessed executable set.
The offer detector also rejects negation, reported/quoted instructions,
offer-clause questions, and fenced code; the legitimate scan phrase containing
`是否已收录` remains accepted when it separately requests the future offer.
The real FULL4 model wording is preserved in a regression that observes the
11-word record before rendering, deterministically rejects the alien exclusion,
binds the nine remaining words, and writes exactly those after confirmation.
The first regression run failed nine assertions; the additional framing checks
then exposed five further input counterexamples. All were corrected. Final
affected methods passed `Ran 11 tests in 0.143s`, and S54 passed
`Ran 37 tests in 0.070s`. These are offline execution results; the real S19 and
fresh FULL checks are recorded separately.

Rejected FULL 5: `e2e/artifacts/20260906T174821Z-4fbd8393`, log
`/private/tmp/keytao-s54-checks/full-s1-s54-run5.log`. S1–S26 passed on their
first attempts, including the corrected S19. S27 attempt 1 failed on a
different model output: a correct binding explanation ended with the stale
claim `刚才「来都来了」的候选还在，等你绑定完回复「加入并提交」就行。`
The advertisement detector correctly recognized this actual action offer.
The defect was the finalizer replacing the entire binding answer with an
unrelated character-data fallback. This does not invalidate the earlier
nominal-phrase fix. The process was stopped with exit 130 and owned-server
cleanup during the built-in retry. No completed all-scenario summary exists.

The finalizer now reuses the existing deterministic binding-process answer
when that question receives a stale action advertisement. The existing
question predicate and copy were extracted into `_binding_meta_question_reply`
and shared with the previous system-template cleanup; no new authorization
path was introduced. The corrected reply is
`会的：写入前会校验绑定；未绑定会给出绑定引导。`
It retains the useful answer without the stale candidate claim or assent.
The assignment continues through the original failure/receipt checks, and
non-binding character questions retain S45's original fallback. The actual
FULL5 model text failed its reproduction before the change; afterward, five
focused checks plus all 37 S54 checks passed (`42 tests in 0.082s`). D is
unchanged. The S27 E2E assertions remain unchanged.

Production remains untouched. Local mocked checks prove the exercised logic;
the E2E fixtures prove the local bot/Next integration with a real model, not
production deployment, external dictionary availability, or live search
provider behavior. No persistent production browser, stream, watcher, worker,
or container was allocated for this task.
