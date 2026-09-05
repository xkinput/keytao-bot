# S53 Unknown-polyphone Reading Resolution

Status: implemented and verified locally on base HEAD
`9ec5723cf0bfd9433daeef03e5432e1c3b5a1817`. This phase is edited-only: no
commit, push, deployment, production API, production data, or production
resource was touched. The monolithic S1-S53 FULL rig was intentionally not run.

## Incident diagnosis

The supplied diagnosis is confirmed; there is no rebuttal.

The deployed encode result for `薄肌` has
`pronunciationSource=pinyin-pro-context`,
`standardPronunciationStatus=absent`, and
`semanticPronunciationNeeded=false`. Before S53,
`_needs_semantic_pronunciation` accepted an explicit semantic flag, a conflict
between the default and contextual sequences, or the narrow fallback sources
`zdic-character-default` / `zdic-unavailable`. It did not inspect the local
per-character reading inventory when the source was `pinyin-pro-context`.
Therefore `薄` being polyphonic was not enough to start semantic resolution,
and the review fell through to the encode service default `báo jī`.

S53 changes the gate to this invariant: a `found` authoritative whole-word
reading always wins; an `absent` whole-word reading enters the new stage only
when `pinyin_reference.db.readings` gives at least two distinct normalized
readings for one of the word's characters. A non-polyphone miss does not enter
the stage.

## Resolution ladder

The new bounded ladder sits after complete authoritative-source absence and
before any encode-default fallback.

1. **Local composition.** A bounded overlap query considers fixed local words
   containing the target and fixed compounds contained by the target. Every
   polyphonic position must be resolved without conflicting carriers. The
   rendered source is `组合推断（<carrier>）`; this lane remains manual-review.
2. **Web pronunciation evidence.** The existing enabled `web` channel from
   `keytao_bot/skills/web-search/tools.py` receives `<word> 拼音` and
   `<word> 读音` concurrently. Extraction supports tone marks, numbered pinyin,
   and plain pinyin, while binding the sequence to the exact target in the
   title/snippet. Two independent results agreeing, or one declared high-trust
   domain, can resolve a candidate. Conflicting accepted readings remain a
   disagreement. No browser is used.
3. **Semantic judgment.** The existing semantic model step now asks for every
   character's contextual pinyin, confidence, meaning, and one-line rationale,
   explicitly distinguishing literary/compound and colloquial/standalone use.
   Web snippets are untrusted input; their extracted pinyin answer is removed
   before the remaining usage clue reaches the model. Web and model agreement
   is eligible for auto-approval. A high-confidence model-only result selects
   the right chain but remains manual-review. An authoritative whole-word hit
   never reaches this ladder.
4. **Unresolved choice.** Conflicting or insufficient evidence returns all
   reading-bound code chains and asks for a reading/meaning. Short codes are
   owned by the higher-priority reading; only six-code duplicates may appear in
   more than one chain. The encode default is merely one alternative.

The raw encode response for `薄肌` originally exposes only the `bzjk` default
chain even though its character facts list `báo`, `bó`, and `bò`. Review now
uses the shared encoder helpers to derive reading-bound phrase chains from
those same returned character facts. The accepted `bó jī` chain is
`bljk`, `bljki`, `bljkiu`; no guessed second encode request is made.

## Cache, latency, and cost boundary

Web resolution uses a durable SQLite cache beside the local pronunciation data
family. Its path is configurable with
`PRONUNCIATION_RESOLUTION_CACHE_DB`; the default is
`data/pronunciation_resolution_cache.db`. Positive evidence lives for 30 days;
a complete negative/weak/disagreement search lives for 15 minutes. Cache hits
perform zero registry calls. Transient or incomplete searches are not cached as
complete absence.

The web rung has an 8.0-second hard timeout. The complete unknown-polyphone
ladder has a 10.0-second deadline, and the semantic, legacy entity, and
contextual fallbacks all share its remaining time. This is below the existing
72.0-second review-stage budget and independent of the existing encode timeout
ladder `(10.0, 20.0, 30.0)`.

Production cost shape for one previously unseen absent polyphone:

- composition hit: local SQLite/reference reads only;
- web lane: two concurrent calls through the existing web-search registry,
  bounded to 8 seconds total; each call may try the registry's configured HTTP
  search backends until it has results or exhausts its own bounded path;
- semantic lane: normally one semantic-pronunciation model request; web
  disagreement stops before the model; if both web and semantic produce
  nothing, the retained legacy entity/context fallback can add at most one
  entity model request plus bounded HTTP/encode work, still inside the same
  10-second caller deadline;
- repeat word: the web cache removes registry calls; accepted semantic results
  also use the existing six-hour in-process cache (rejected results: ten
  minutes), request rate limits, and max-concurrency gate.

The incident timestamps cover about 17 seconds from the first rejected source
at `02:05:09` to `keytao_prepare_reviewed_add` at `02:05:26`, ending with the
wrong default. This is not a full input-to-reply measurement, so it is only an
observed baseline interval.

The final targeted S53 run measured the `薄肌` input-to-reply interval at
19.897 seconds. Its structured unknown-polyphone resolution took 3.4233
seconds, including the semantic decision; the seeded web registry call itself
took 0.0008 seconds. The new resolution is therefore within both its 10-second
budget and the parent review budget. Relative to the incident's non-identical
17-second interval, the complete local E2E reply was about 2.9 seconds longer
while correcting the reading; no claim of a latency reduction is made.

The S53 fixture made two registry calls and recorded two fixture backend
attempts for `薄肌`; these were deterministic in-process web fixtures, not live
web HTTP. The real DeepSeek semantic-pronunciation request used 558 input and
179 output tokens (737 total). The six-message scenario as a whole used 15 real
model exchanges and 6,780 tokens; those totals include command routing,
commonness work, controls, and all three composition cases, so they are not the
incremental cost of the pronunciation rung alone.

## Truthful rendering and approval

The resolved incident reply now contains:

```text
审词：读音 bó jī；来源 网络（baike.baidu.com） ...；网络（terms.naer.edu.tw） ...；语义判断；
自动审核：权威来源、编码和常用度证据一致，可自动通过
1. bljk
2. bljki
3. bljkiu
```

Web-only and model-only results remain sealed for manual review. Composition
states its carrier and remains manual-review. Disagreement returns both labelled
readings without a recommendation or pending write. All S53 turns were
read-only and the draft was unchanged.

## S53 fixture design

The targeted scenario sends six independent real chat messages through the
public review tool:

| Case | Purpose | Expected resolution |
|---|---|---|
| `薄肌` | incident | two fixed independent web snippets plus real model agree on `bó jī`; `bljk*` only |
| `肌群` | latency control | no local polyphone; zero new web-registry calls |
| `校准器` | contained compound | inherit `jiào zhǔn` from `校准`; zero web calls |
| `长按键` | contained compound | inherit `cháng àn` from `长按`; zero web calls |
| `着陆台` | two polyphonic positions | inherit `zhuó lù` from `着陆`; zero web calls |
| `校肌` | disagreement control | fixture web results disagree on `jiào jī` / `xiào jī`; expose both chains and ask |

The local Next fixture keeps every exact word absent while returning the real
character reading inventories. The web fixture is installed only at the new
registry boundary. Existing external dictionary/search probes remain governed
by the E2E socket allowlist and were blocked before external dispatch. DeepSeek
remained real.

## Verification tails

All commands ran from `/Users/rea/code/keytao-org/keytao-bot`. `pnpm test` was
not run.

| Check | Final result |
|---|---|
| `.venv/bin/python test_memory_safety.py` | `Ran 403 tests in 186.487s` / `OK`; real 187.02s |
| `.venv/bin/python test_state_machine.py` | `Results: 2013/2013 passed, 0 failed`; real 0.95s |
| `.venv/bin/python test_security_fixes.py` | `Results: 268/268 passed, 0 failed`; real 0.48s |
| `.venv/bin/python test_review_gate.py` | `Results: 432/432 passed`; real 21.09s |
| `.venv/bin/python test_llm_policy.py` | `Ran 10 tests in 0.063s` / `OK`; real 0.26s |
| `.venv/bin/python test_word_discovery.py` | `Results: 290/290 passed`; real 0.20s |
| `.venv/bin/python -m e2e.test_safety` | `Ran 96 tests in 0.453s` / `OK`; real 0.62s |
| targeted `E2E_OPENAI_BASE_URL=https://api.deepseek.com E2E_OPENAI_MODEL=deepseek-v4-flash .venv/bin/python -m e2e.run --only S53` | `S53 PASSED`, attempt 1, exit 0 |
| `.venv/bin/python -m py_compile ...` | exit 0 |
| `git diff --check` | exit 0 |

The deadline regression was developed red-to-green: the first run was
`431/432`, with the deliberately slow legacy entity fallback escaping the
shared deadline; after the fix the final suite is `432/432`.

## Targeted S53 result

- Artifact: `e2e/artifacts/20260905T045844Z-f34f8f74`
- Run ID: `f34f8f74fa6046429e72fa4d32ad05b6`
- Selected scenarios: exactly `S53`
- Verdict: `PASSED`, attempt 1, 73.539 seconds for all six messages
- Real provider proof: 15 successful `openai.AsyncOpenAI` HTTP exchanges, all
  `deepseek-v4-flash`; no fake client in the message path
- Incident fact: `bó jī`; codes `bljk`, `bljki`, `bljkiu`; web/model agreement;
  `autoReviewable=true`
- Controls: non-polyphone web calls `0`; all three composition cases web calls
  `0`; disagreement exposed `jcjk*` and `xcjk*`; draft unchanged
- Safety proof: production URL blocked before dispatch, remote database and
  production-like bindings/admin identities rejected

The FULL S1-S53 rig remains deliberately unexecuted for phase 2.

# Phase 2 — cross-model review closure and FULL gate

Phase 2 supersedes the Phase 1 statements about the web clue, 30-day positive
TTL, contained-compound inheritance, short-code ownership, the original S53
case matrix, and the FULL rig being unexecuted. The accepted implementation and
evidence are recorded below. The work remains edited-only on
`9ec5723cf0bfd9433daeef03e5432e1c3b5a1817`: no commit, push, deployment,
production API, production data, or production resource was touched.

## Review finding closure

| Finding | Disposition | Change and evidence |
|---|---|---|
| Authoritative hit always wins | SAFE / unchanged | The ladder still starts only after authoritative whole-word absence; covered by the existing authoritative-hit regression. |
| BLOCKER: weak single-domain web evidence auto-approved | FIXED | Only `web.status == "resolved"` can supply the selected agreement candidate (`keytao_bot/utils/keytao_review.py:3106-3119`). A weak result stays `model_only` or unresolved and manual (`test_review_gate.py:2307-2401`). |
| BLOCKER: semantic model received the web answer/instructions | FIXED | The rung-3 call receives no web evidence hint (`keytao_bot/utils/keytao_review.py:3079-3089`). `_web_semantic_context` is now bounded scrubbed audit/debug output only (`keytao_bot/utils/keytao_review.py:2918-2949`); numbered, tone-marked, and plain pinyin leakage is covered at `test_review_gate.py:1747-1846`. |
| SHOULD: untrusted URL/host and false authority copy | FIXED | Only valid HTTP(S) DNS hostnames survive (`keytao_bot/utils/pronunciation_resolution.py:436-447`); web source labels omit URLs (`keytao_bot/plugins/chat_render.py:1072-1079`); web evidence no longer counts as authority and agreement has its own reason (`keytao_bot/utils/keytao_review.py:3945-3981`). |
| SHOULD: 30-day poisoned positive cache | FIXED | Cache schema version 2 is part of the primary key, positive TTL is 24 hours, and durable payloads contain only normalized readings and validated domains (`keytao_bot/utils/pronunciation_resolution.py:59-61`, `64-153`, `156-263`). Raw title/snippet/URL text is not stored. Regression: `test_review_gate.py:1876`. |
| SHOULD: contained-compound inheritance | FIXED | Composition accepts only longer carriers containing the target (`keytao_bot/utils/pronunciation_resolution.py:390-425`). `便宜` can no longer donate `pián yi` to `便宜行事`; test at `test_review_gate.py:1590-1667`. |
| SHOULD: global encode candidate mutation | FIXED | Ordinary `fetch_keytao_encode` preserves the server candidate scope; character-derived variants are added only to a local ladder copy (`keytao_bot/utils/keytao_review.py:2993-2997`). Regression: `test_review_gate.py:1703-1744`. |
| NIT: unknown pronunciation source failed open | FIXED | Unrecognized values now return `unavailable` (`keytao_bot/utils/keytao_review.py:2124-2140`). |
| NIT: repeated character SQLite lookup | FIXED | Character readings are memoized by resolved DB path, mtime, and size; repeated characters are also deduplicated per word (`keytao_bot/utils/pronunciation_resolution.py:271-322`). |
| NIT: duplicated pinyin character class | SKIPPED | `keytao_bot/utils/pronunciation_resolution.py:26-34` still has the local copy. This is behavior-neutral; removing it after the accepted FULL run would make the verified artifact differ from the final source and require another billed monolithic run solely for symbol deduplication. |
| NIT: regex over unbounded snippets | FIXED | Titles/snippets are truncated before regex matching (`keytao_bot/utils/pronunciation_resolution.py:483-490`). |
| NIT: pronunciation query could be rerouted | FIXED | The registered call explicitly pins `channel="web"` (`keytao_bot/utils/keytao_review.py:2770-2775`); the skill entry accepts the internal keyword-only override. |
| NIT: unresolved short-code ownership favored the first group | FIXED | Colliding codes shorter than six characters are removed from every unresolved group; only six-character duplicates may remain (`keytao_bot/utils/keytao_review.py:3301-3325`, `test_review_gate.py:2799-2810`). |
| NIT: semantic cache keyed only by word | FIXED | Evidence-bearing calls use a SHA-256-qualified key while no-hint calls keep the word key (`keytao_bot/utils/keytao_review.py:4755-4760`, `4817-4844`); regression at `test_review_gate.py:2746-2796`. |
| Fixture gap | FIXED | S53 now includes two agreeing untrusted domains and a same-domain weak pair (`e2e/web_evidence_seed.py:11-47`), asserted through the public tool at `e2e/scenarios.py:8149-8227`. |
| Missing tests (a)–(d) | ADDED | (a) weak agreement: `test_review_gate.py:2307`; (b) suffix lookalike: `test_review_gate.py:1792-1859`; (c) numbered/tone variants: `test_review_gate.py:1747-1846`; (d) wrong contained compound: `test_review_gate.py:1612-1667`. |
| Latency | SAFE | The existing bounded web and overall ladder deadlines remain in force; the accepted S53 took 71.5 seconds end-to-end across all cases, while its individual ladder facts remain bounded in the artifact. |

The only skipped NIT is the local pinyin-character-class duplication listed
above. No BLOCKER or SHOULD-FIX was skipped.

## Final S53 fixture coverage

The Phase 2 S53 scenario sends six independent chat messages through
`keytao_prepare_reviewed_add`:

| Case | Evidence lane | Required result | Accepted-run result |
|---|---|---|---|
| `薄肌` | two agreeing untrusted web domains; independent real model disagrees | conflict stays manual and shows both reading-bound chains | `disagreement`; `báo jī` and `bó jī`; `autoReviewable=false` |
| `薄荷味糖` | two agreeing untrusted domains plus independent real model | agreement may auto-review under the web-specific reason | `web_model_agreement`; `autoReviewable=true` |
| `薄肌腱` | two snippets from one untrusted domain | weak evidence cannot auto-review | `web.status=weak`; final `unresolved`; manual |
| `肌群` | no polyphonic character | stop before web | web calls `0` |
| `不着陆` | longer local carrier `不着陆飞行` | composition remains manual and stops before web | `bu zhuo lu`; web calls `0` |
| `校肌` | conflicting web readings | expose both reading-bound chains and ask | `jiao ji` and `xiao ji`; manual |

The accepted facts also record `draftUnchanged=true`. Fixture web snippets are
deterministic in-process inputs, not live web proof; DeepSeek calls are real.

## Phase 2 verification tails

All commands ran from `/Users/rea/code/keytao-org/keytao-bot`. The banned
`pnpm test` command was not run.

| Check | Result |
|---|---|
| `.venv/bin/python test_memory_safety.py` | `Ran 403 tests in 183.329s` / `OK` |
| `.venv/bin/python test_state_machine.py` | `Results: 2013/2013 passed, 0 failed` |
| `.venv/bin/python test_security_fixes.py` | `Results: 268/268 passed, 0 failed` |
| `.venv/bin/python test_review_gate.py` | `Results: 443/443 passed` / `ALL TESTS PASSED` |
| `.venv/bin/python test_llm_policy.py` | `Ran 10 tests in 0.068s` / `OK` |
| `.venv/bin/python test_word_discovery.py` | `Results: 290/290 passed` |
| `.venv/bin/python -m e2e.test_safety` | `Ran 96 tests in 0.461s` / `OK` |
| `.venv/bin/python -m py_compile ...` | exit `0` |
| `git diff --check` | exit `0` |
| targeted S53, real DeepSeek | `S53 PASSED`, attempt 1, 79.3s, 17 requests, 8,201 tokens; artifact `e2e/artifacts/20260905T060043Z-2fedc9f0` |

The targeted run is supporting evidence only. It is not used to satisfy the
FULL gate.

## Accepted monolithic FULL S1–S53 gate

- Invocation: `E2E_OPENAI_API_KEY="$(<.e2e_key)" E2E_OPENAI_BASE_URL=https://api.deepseek.com E2E_OPENAI_MODEL=deepseek-v4-flash .venv/bin/python -m e2e.run`
- Preflight: `.e2e_key` non-empty (secret not printed); endpoint
  `https://api.deepseek.com`; model `deepseek-v4-flash`; no provider fallback.
- Single process: started `2026-09-05T06:07:04Z` (`14:07:04` Asia/Shanghai),
  completed `2026-09-05T06:39:32Z` (`14:39:32` Asia/Shanghai), exit `0`.
- Artifact: `e2e/artifacts/20260905T060704Z-5558751d`; run ID
  `5558751da1ba421fb1f0093cd4998de8`.
- Manifest agreement: selected `53`, results `53`, passed `53`, failed `0`;
  every scenario used attempt `1`; repo HEAD
  `9ec5723cf0bfd9433daeef03e5432e1c3b5a1817`.
- Provider proof: manifest records `281` successful HTTP exchanges through
  `openai.AsyncOpenAI`, no fake client in the message path, and the result model
  set is exactly `{deepseek-v4-flash}`. Per-scenario accounting totals 282 model
  requests and 1,636,947 tokens; provider billing prices were unavailable
  locally.
- Safety proof: production URL blocked before dispatch; remote DB,
  production-like binding, and production-like admin were rejected.

Complete accepted per-scenario table:

| Scenario | Verdict | Attempts | Seconds | Model requests | Tokens |
|---|---:|---:|---:|---:|---:|
| S1 | PASSED | 1 | 21.1 | 2 | 1238 |
| S2 | PASSED | 1 | 15.4 | 1 | 593 |
| S3 | PASSED | 1 | 4.5 | 0 | 0 |
| S4 | PASSED | 1 | 11.2 | 3 | 48285 |
| S5 | PASSED | 1 | 10.2 | 0 | 0 |
| S6 | PASSED | 1 | 75.7 | 17 | 102250 |
| S7 | PASSED | 1 | 5.2 | 0 | 0 |
| S8 | PASSED | 1 | 15.1 | 0 | 0 |
| S9 | PASSED | 1 | 7.3 | 2 | 824 |
| S10 | PASSED | 1 | 30.8 | 7 | 105298 |
| S11 | PASSED | 1 | 19.6 | 5 | 99715 |
| S12 | PASSED | 1 | 20.1 | 5 | 99468 |
| S13 | PASSED | 1 | 7.0 | 3 | 47864 |
| S14 | PASSED | 1 | 7.6 | 3 | 1385 |
| S15 | PASSED | 1 | 16.3 | 4 | 1648 |
| S16 | PASSED | 1 | 56.0 | 8 | 52460 |
| S17 | PASSED | 1 | 24.0 | 6 | 2760 |
| S18 | PASSED | 1 | 11.0 | 3 | 1523 |
| S19 | PASSED | 1 | 60.7 | 10 | 51998 |
| S20 | PASSED | 1 | 10.0 | 4 | 46147 |
| S21 | PASSED | 1 | 66.4 | 15 | 124142 |
| S22 | PASSED | 1 | 31.6 | 10 | 100666 |
| S23 | PASSED | 1 | 38.7 | 5 | 77970 |
| S24 | PASSED | 1 | 4.5 | 2 | 824 |
| S25 | PASSED | 1 | 14.1 | 4 | 2014 |
| S26 | PASSED | 1 | 21.2 | 2 | 1070 |
| S27 | PASSED | 1 | 27.5 | 7 | 27211 |
| S28 | PASSED | 1 | 34.9 | 8 | 3296 |
| S29 | PASSED | 1 | 2.8 | 1 | 588 |
| S30 | PASSED | 1 | 30.3 | 6 | 2488 |
| S31 | PASSED | 1 | 8.5 | 1 | 515 |
| S32 | PASSED | 1 | 14.6 | 3 | 1672 |
| S33 | PASSED | 1 | 116.6 | 25 | 352238 |
| S34 | PASSED | 1 | 27.7 | 8 | 3255 |
| S35 | PASSED | 1 | 23.0 | 6 | 2464 |
| S36 | PASSED | 1 | 28.4 | 7 | 2748 |
| S37 | PASSED | 1 | 12.9 | 4 | 1686 |
| S38 | PASSED | 1 | 27.9 | 10 | 4990 |
| S39 | PASSED | 1 | 9.6 | 4 | 2368 |
| S40 | PASSED | 1 | 43.3 | 10 | 47977 |
| S41 | PASSED | 1 | 24.6 | 5 | 74405 |
| S42 | PASSED | 1 | 46.3 | 9 | 76570 |
| S43 | PASSED | 1 | 14.4 | 3 | 1719 |
| S44 | PASSED | 1 | 12.4 | 3 | 1352 |
| S45 | PASSED | 1 | 10.2 | 3 | 42078 |
| S46 | PASSED | 1 | 4.2 | 0 | 0 |
| S47 | PASSED | 1 | 5.8 | 2 | 1182 |
| S48 | PASSED | 1 | 25.6 | 7 | 2991 |
| S49 | PASSED | 1 | 0.0 | 0 | 0 |
| S50 | PASSED | 1 | 23.8 | 5 | 1866 |
| S51 | PASSED | 1 | 14.4 | 4 | 1567 |
| S52 | PASSED | 1 | 18.8 | 3 | 1375 |
| S53 | PASSED | 1 | 71.5 | 17 | 8204 |

The manifest/result/table counts agree. This is a local E2E gate with a real
DeepSeek provider; it is not deployment or production verification.
