# September 19–20 bot incident repair

Baseline: `bb01a7ea36122e14b6fc45a70746d30976df680e` (local and Aliyun verified before edits).

## Observed causes and intended behavior

| Incident | Confirmed cause | Required behavior |
| --- | --- | --- |
| Ordinary conversation returns a service/write failure | Two routing calls exhaust the two-call allowance before any main-agent call | Reserve one bounded answer opportunity after routing; keep all calls accounted for |
| Comparing 偿债 and 从众 starts a four-word add flow | The natural comparison grammar misses both reported forms; the bare list parser accepts the question fragment as another word | Route a complete comparison question to the existing read-only commonness tool |
| Rephrased question repeats stale candidates | The send stage appends the previous ticket's confirmation to a plain answer, making the delivery guard replace it with the ticket | Append confirmation only to an answer already advertising a stateful operation |
| Clear draft/cache/history repeats stale candidates | The compound clear instruction is not recognized before pending arbitration | Execute an exact, explicit reset of the actor's draft and current conversation; preserve running writes and other actors |
| Book title receives invented multiple meanings | Semantic JSON is truncated at 450 tokens, treated as a completed negative, cached, and replaced with Cartesian character readings | Use a bounded word-length budget and compact output; do not cache incomplete inference or present unsupported character combinations as word meanings |
| Commonness explanation reverses the evidence numbers | Word names switch for the winning direction, but reference objects do not | Display words and both frequency/dictionary counts in the same order |
| Quoted full proverb is not queried as one entry | Quoted literals and explicit single-entry syntax fall through or are rejected as reported assent to an older ticket; downstream review also counts punctuation as a spoken character | Route a complete literal as a fresh query while retaining active-operation arbitration; preserve the exact literal through lookup, review, pending state, and write; map pronunciation only to spoken characters |

## Verification method

- Original failures are reproduced with fake providers/tools in credential-free temporary source snapshots.
- Tests run with a minimal environment and blocked outbound sockets; no paid model calls or user draft changes are made.
- Production logs establish routing and delivery failures independently of test fixtures.
- Vendored reference data is built inside the isolated test snapshot. It contains the exact punctuated proverb with eight spoken syllables.
- Confirmation authorization, ownership, revision checks, evidence trust, and server candidate validation remain required.

## Verification results

- `python -B -m unittest -q test_sep19_commonness test_sep19_general test_sep19_pronunciation test_sep19_reset test_sep20_phrase test_s59_commonness_route test_s59_commonness_evidence test_s62_commonness test_s62_reading_resolution test_s62_selection test_s62_stale_draft test_s63_general_reply`: 91 tests passed.
- `python -B -m unittest -q test_s57_options test_sep20_phrase_routing test_s57_claiming test_s58_advertisement test_s61_submit_replay test_s62_draft_flow`: 73 tests passed, including the literal-routing matrix's 40 subcases.
- `python -B -m unittest -q test_s59_general_cap`: 11 tests passed in its separate real-NoneBot fixture process.
- `python -B test_state_machine.py`: 2023/2023 checks passed.
- `python -B test_review_gate.py`: 443/443 checks passed.
- Changed Python files pass AST parsing; `git diff --check` passes.
- All commands above use a credential-free snapshot, `env -i`, `PYTHONDONTWRITEBYTECODE=1`, and an outbound socket audit guard. Build the reference database with `scripts/build_pinyin_reference.py` and include tracked `test_fixtures/` before running the legacy scripts. Missing snapshot assets initially caused failures; the complete baseline passes 2023/2023 and 443/443 too.

Independent review found and closed four implementation regressions: applying the advertisement guard too broadly to direct confirmations; clearing a draft before retiring its unsent waiting operation; indexing missing pronunciation defaults; and accepting an invalid explicit pronunciation variant via a fallback. A further routing regression test preserves exact sealed confirmation/cancellation aliases over ordinary quoted-word lookup.

## Deployment and limits

Deploy the verified commit to `origin/main` and `/root/project/keytao-bot`, then restart only `keytao-bot`. Recheck the remote revision and clean tracked tree before changing the bind-mounted source. Verify container health, matching source fingerprints, and read-only parser behavior afterward. The task handoff records the actual deployed revision and observed runtime state.

These tests establish deterministic routing and data-integrity behavior, including a real local CEDICT row through review, pending state, and a fake write sink that runs the actual code validator and review-flag stamping. They do not establish real model or QQ end-to-end behavior. No real user draft was added, submitted, or cleared during verification.
