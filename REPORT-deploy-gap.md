# S52 deploy-gap incident closure

## Status

Implemented locally on baseline HEAD `00540014aa7bc6609d8cef8a7493d5f9aa251132`.
Nothing in this round was committed, pushed, deployed, or exercised against
production. `pnpm test` was not run.

The one requested monolithic real-provider run executed S1 through S52 in one
process. S52 passed on attempt 1. The aggregate result was 51/52 because S30
still asserted the superseded rule that a plain candidate render followed by
`好` must not write. This round intentionally makes every trusted candidate
render executable, so the observed S30 write was the new required behavior.
That stale S30 assertion was corrected after the run. It was not followed by a
second real-provider run, preserving the instruction to run exactly one FULL
rig. The resulting S30 assertion change therefore has offline and artifact
evidence but no post-change real-provider rerun.

## Root-cause and closure ledger

| Area | Finding | Closure |
|---|---|---|
| A: restart gap | OneBot messages received after the old connection stopped and before the new connection completed were not visible to the new process. There was no durable per-group processed boundary or reconnect replay. | Added a durable `(message_id, timestamp)` marker per group, advanced only after a handled reply boundary. On OneBot connect, configured and previously served groups are fetched sequentially, exactly once each, filtered by age, marker, existing later bot reply, and the existing `should_handle` rule, then replayed through `handle_ai_chat` in chronological order. Processing stops at the first delivery failure so a later message cannot move the marker past it. The completion log emits one `[startup_replay] complete` line with group, call, replay, skip, and failure counts. |
| B: first render | `_try_handle_simple_single_word_query` persisted `PendingAddWord` only when `explicit_add_word`, a prefixed review, or `actionable_lookup` was true. A plain `细品` query could render candidates with `本次仅查询` and no trusted record. | Removed the alternate read-only candidate path. Every structured candidate inventory is persisted before rendering the common selector. Bare assent consumes that exact record. S52 proves `细品` renders `xkpb`, then `加入并提交` submits it in the next turn with zero candidate regeneration. |
| C: rebuild persistence | The default SQLite path was already on a host bind mount, but `SQLiteConversationStateStore` explicitly persisted only `PendingTrustedWordRecord` and `PendingToolConfirm`; `PendingAddWord` remained process-local and was lost on restart. | Added validated serialization and reload for the complete `PendingAddWord` capability. A disk-reload test verifies equality, owner, nonce, and reconfirmation fields. No compose mount change was needed. |
| D: delete grammar/tool mismatch | The deterministic grammar lacked code-first possessive delete forms, so the model could select a create tool and reach a misleading generic binding error. | Added closed grammar for `删除/删掉 <code> 的/上的 <word>` and `把 <code> 的/上的 <word> 删了/删掉`, including supported spacing and quote styles. A delete-shaped turn selecting `keytao_create_phrase` or `keytao_shift_phrase_code` is now rejected as `intent_mismatch` with an accurate delete-vs-create/recode explanation, then immediately handed to the deterministic delete route. The false “词条或编码与这条消息不一致” copy is not used for this case. |

The stale-record regeneration path remains available as a safety net, but the
normal first-render flow no longer reaches it.

## Startup replay and production resource cost

The implementation uses the documented OneBot
[`get_group_msg_history`](https://napneko.github.io/onebot/api) action. It makes
one bounded history call per configured/served group on each OneBot connect;
the default page size is 50 and the hard cap is 200. Groups are processed
sequentially, and messages older than 10 minutes are skipped by default (hard
cap 60 minutes).

Configuration:

- `KEYTAO_STARTUP_REPLAY_GROUP_IDS`: explicit replay groups;
- `KEYTAO_STARTUP_REPLAY_MAX_AGE_MINUTES`: default `10`;
- `KEYTAO_STARTUP_REPLAY_HISTORY_COUNT`: default `50`;
- existing `KEYTAO_SYNC_NOTIFY_GROUP_IDS` are also treated as configured;
- groups with durable processed markers are included automatically.

For the shared 16 GB/no-swap host, this adds transient bounded JSON history
reads only. It does not create a browser, stream, watcher, worker, or container,
and it does not add concurrent calls: at default settings only one page of at
most 50 rows is resident for the current group. This is materially different
from the persistent Chrome allocation in the cited production incident. No
production history endpoint was called in this round.

The double-reply test includes an older message already followed by a bot
message. It is skipped, while the later unanswered addressed message is
replayed once. A second reconnect reloads the marker from SQLite and does not
replay it again.

## Volume finding

The bot image uses `WORKDIR /app`, and the default pending store resolves to
`/app/data/pending_confirmations.db`. `docker-compose.yml` already bind-mounts
the repository at `.:/app`, so the database is the host file
`./data/pending_confirmations.db` and survives `docker compose up --build` and
container recreation. The same SQLite database now contains both
`pending_confirmations` and `startup_replay_markers`.

The mount itself was therefore not defective and was left unchanged. The real
C defect was the store's previous decision not to serialize candidate records.

## Scenario S52

S52 contains both incident closures rather than creating S53:

1. `喵喵 细品` renders the trusted `xkpb/xkpba/xkpbao` inventory with selector
   affordances and a persisted `xkpb` recommendation.
2. Immediate `加入并提交` writes and submits only `细品@xkpb`, in one turn,
   without `候选记录已失效` or a second `keytao_prepare_reviewed_add` call.
3. Six possessive-delete spacing/quote variants each seal and confirm only
   `Delete 哪里@nsl`; every receipt names `哪里 nsl`, never `nslko`, and every
   case records zero semantic-routing model requests.
4. The wrong create/shift-tool branch is forced in offline tests because the
   corrected deterministic production route claims all six S52 turns before a
   model can improvise.

The S52 artifact records `PASSED`, attempt 1, 18.0 seconds, 3 real model
requests, 1,383 tokens, first-render persistence, zero regeneration, the exact
submitted batch, and all six deterministic delete cases.

## Focused and offline verification

All commands ran from `/Users/rea/code/keytao-org/keytao-bot`.

| Check | Result |
|---|---|
| New first-render, possessive-delete, wrong-tool fallback focused checks | 19 assertions passed |
| Candidate disk reload + startup replay focused unittest | `Ran 3 tests` / `OK` |
| `.venv/bin/python test_memory_safety.py` | `Ran 403 tests in 188.311s` / `OK` |
| `.venv/bin/python test_state_machine.py` | `Results: 2013/2013 passed, 0 failed` |
| `.venv/bin/python test_security_fixes.py` | `Results: 268/268 passed, 0 failed` |
| `.venv/bin/python test_review_gate.py` | `Results: 406/406 passed` |
| `.venv/bin/python test_llm_policy.py` | `Ran 10 tests in 0.088s` / `OK` |
| `.venv/bin/python test_word_discovery.py` | `Results: 290/290 passed` |
| `.venv/bin/python -m unittest e2e.test_safety` | final run: `Ran 94 tests in 0.542s` / `OK` |
| `.venv/bin/python -m compileall -q keytao_bot e2e test_memory_safety.py test_state_machine.py` | exit 0 |
| `git diff --check` | exit 0 |

The first complete `test_memory_safety.py` run found that the initial intent
guard was too broad and blocked a legitimate unqualified delete whose unique
code was supplied by the server record. The guard was narrowed to the required
wrong create/shift tools; the focused regression and the full 403-test rerun
then passed.

## One monolithic FULL real-provider rig

Command:

```text
E2E_OPENAI_API_KEY="$(<.e2e_key)" \
E2E_OPENAI_BASE_URL=https://api.deepseek.com \
E2E_OPENAI_MODEL=deepseek-v4-flash \
.venv/bin/python -m e2e.run
```

- Exit code: `1` because of the stale S30 assertion described above.
- Artifact: `e2e/artifacts/20260904T182222Z-cb90f242`.
- Selected/result rows: 52/52, exactly S1 through S52.
- Verdicts: 51 PASSED, 1 FAILED (S30 only); S52 PASSED on attempt 1.
- Real-provider proof: 253 successful `openai.AsyncOpenAI` HTTP exchanges.
- Model proof: manifest model `deepseek-v4-flash`; scenario cost rows list no
  other response model and no provider fallback.
- Aggregate usage: 253 model requests and 1,548,763 recorded tokens.
- Safety proof: local KeyTao origin and local database only; production KeyTao
  URLs and remote database bindings remained blocked by the rig.

| Scenario | Verdict | Attempts | Seconds | LLM requests | Tokens |
|---|---:|---:|---:|---:|---:|
| S1 | PASSED | 1 | 21.7 | 2 | 1118 |
| S2 | PASSED | 1 | 15.9 | 1 | 593 |
| S3 | PASSED | 1 | 4.6 | 0 | 0 |
| S4 | PASSED | 1 | 10.8 | 3 | 48203 |
| S5 | PASSED | 1 | 10.2 | 0 | 0 |
| S6 | PASSED | 1 | 41.2 | 10 | 66929 |
| S7 | PASSED | 1 | 5.2 | 0 | 0 |
| S8 | PASSED | 1 | 15.1 | 0 | 0 |
| S9 | PASSED | 1 | 7.3 | 2 | 824 |
| S10 | PASSED | 1 | 36.3 | 6 | 80641 |
| S11 | PASSED | 1 | 16.5 | 5 | 99155 |
| S12 | PASSED | 1 | 16.6 | 5 | 98743 |
| S13 | PASSED | 1 | 7.2 | 3 | 47906 |
| S14 | PASSED | 1 | 6.4 | 3 | 1363 |
| S15 | PASSED | 1 | 17.1 | 4 | 1648 |
| S16 | PASSED | 1 | 41.9 | 8 | 52530 |
| S17 | PASSED | 1 | 25.2 | 6 | 2748 |
| S18 | PASSED | 1 | 10.5 | 3 | 1285 |
| S19 | PASSED | 1 | 68.2 | 10 | 53118 |
| S20 | PASSED | 1 | 20.0 | 4 | 46215 |
| S21 | PASSED | 1 | 43.3 | 14 | 119079 |
| S22 | PASSED | 1 | 33.2 | 10 | 100813 |
| S23 | PASSED | 1 | 34.7 | 6 | 59615 |
| S24 | PASSED | 1 | 4.5 | 2 | 824 |
| S25 | PASSED | 1 | 12.2 | 4 | 2014 |
| S26 | PASSED | 1 | 21.1 | 2 | 1067 |
| S27 | PASSED | 1 | 28.3 | 7 | 27217 |
| S28 | PASSED | 1 | 17.3 | 8 | 3296 |
| S29 | PASSED | 1 | 2.7 | 1 | 588 |
| S30 | FAILED | 2 | 23.3 | 4 | 1648 |
| S31 | PASSED | 1 | 8.5 | 1 | 523 |
| S32 | PASSED | 1 | 14.5 | 3 | 1661 |
| S33 | PASSED | 1 | 131.9 | 25 | 351493 |
| S34 | PASSED | 1 | 28.1 | 8 | 3242 |
| S35 | PASSED | 1 | 21.4 | 6 | 2464 |
| S36 | PASSED | 1 | 27.7 | 7 | 2748 |
| S37 | PASSED | 1 | 13.1 | 4 | 1680 |
| S38 | PASSED | 1 | 27.5 | 10 | 4772 |
| S39 | PASSED | 1 | 6.5 | 2 | 1184 |
| S40 | PASSED | 1 | 27.9 | 9 | 48055 |
| S41 | PASSED | 1 | 31.5 | 6 | 80239 |
| S42 | PASSED | 1 | 46.1 | 9 | 77052 |
| S43 | PASSED | 1 | 14.6 | 3 | 1713 |
| S44 | PASSED | 1 | 28.8 | 3 | 1356 |
| S45 | PASSED | 1 | 12.1 | 3 | 42403 |
| S46 | PASSED | 1 | 4.3 | 0 | 0 |
| S47 | PASSED | 1 | 5.9 | 2 | 1182 |
| S48 | PASSED | 1 | 40.4 | 7 | 3008 |
| S49 | PASSED | 1 | 0.0 | 0 | 0 |
| S50 | PASSED | 1 | 23.8 | 5 | 1858 |
| S51 | PASSED | 1 | 14.5 | 4 | 1567 |
| S52 | PASSED | 1 | 18.0 | 3 | 1383 |

## Remaining verification boundary

The corrected S30 scenario now treats the first-render `好` write as required,
cleans that draft, and continues to test cancellation and natural
add-and-submit. Existing offline state-machine coverage proves the same
first-render `好` binding, and the only FULL artifact failure contains the exact
successful `Create 吃席@wkxko` evidence that the stale assertion rejected.
Because no second paid FULL run was allowed, the corrected S30 scenario has not
been rerun end to end against the real provider.

Startup replay was verified with constructed OneBot history and the real
durable store, not by restarting production NapCat. No production-side resource
or behavior claim is made.

## FULL rig (accepted run)

The accepted run started at `2026-09-04T19:33:15Z` and executed exactly S1
through S52 in one process. It exited `0` with 52 `PASSED` verdicts and no
second scenario attempts. The accepted artifact is
`e2e/artifacts/20260904T193315Z-0c5fc35c`.

The manifest records HEAD `00540014aa7bc6609d8cef8a7493d5f9aa251132`, LLM
host `api.deepseek.com`, model `deepseek-v4-flash`, 262 successful real
`openai.AsyncOpenAI` HTTP exchanges, and no response model other than
`deepseek-v4-flash`. Aggregate accepted-run usage was 262 model requests and
1,722,469 recorded tokens. All four fail-closed safety self-checks passed:
production KeyTao URL blocking, remote database rejection, production-like
binding rejection, and production-like admin rejection.

Model confirmation line from the accepted run:

```text
09-05 03:33:59 [INFO] keytao_bot | LLM usage: operation=entity_knowledge model=deepseek-v4-flash response_id=c48341e2-061a-4e4d-a106-e26fe8aa6c32 system_fingerprint=a26a7955944dc5c60445bff77fac9c8e finish_reason=stop input_tokens=413 output_tokens=138 total_tokens=551 cached_tokens=384 cache_miss_tokens=29
```

Assertion change made in this task:

- `_recommended_empty_code` now recognizes the common selector's explicit
  `推荐编码：<code>` line. The rejected fresh FULL run at
  `e2e/artifacts/20260904T190148Z-1ae3a4d3` rendered both
  `2. wkxko — ✅ 推荐（空位）` and `推荐编码：wkxko`, but the old assertion
  parser recognized only legacy actionable, bullet, and `不重排选` phrasings.
  The S52 user-facing rule that every structured candidate inventory uses one
  executable selector with an explicit recommendation supersedes those older
  presentation shapes. This is not an assertion weakening: after parsing the
  explicit recommendation, the existing check still requires that exact code
  to appear in a numbered row marked as an empty slot.

| Scenario | Verdict | Attempts | Seconds | LLM requests | Tokens |
|---|---:|---:|---:|---:|---:|
| S1 | PASSED | 1 | 22.4 | 2 | 1106 |
| S2 | PASSED | 1 | 15.9 | 1 | 593 |
| S3 | PASSED | 1 | 4.6 | 0 | 0 |
| S4 | PASSED | 1 | 11.7 | 3 | 48532 |
| S5 | PASSED | 1 | 10.3 | 0 | 0 |
| S6 | PASSED | 1 | 38.4 | 9 | 65556 |
| S7 | PASSED | 1 | 5.2 | 0 | 0 |
| S8 | PASSED | 1 | 15.3 | 0 | 0 |
| S9 | PASSED | 1 | 7.1 | 2 | 824 |
| S10 | PASSED | 1 | 31.1 | 7 | 106329 |
| S11 | PASSED | 1 | 18.5 | 5 | 99019 |
| S12 | PASSED | 1 | 31.2 | 5 | 99710 |
| S13 | PASSED | 1 | 8.0 | 3 | 48038 |
| S14 | PASSED | 1 | 6.8 | 3 | 1379 |
| S15 | PASSED | 1 | 16.5 | 4 | 1648 |
| S16 | PASSED | 1 | 40.1 | 8 | 52175 |
| S17 | PASSED | 1 | 46.2 | 6 | 2821 |
| S18 | PASSED | 1 | 10.6 | 3 | 1294 |
| S19 | PASSED | 1 | 57.1 | 10 | 52051 |
| S20 | PASSED | 1 | 19.6 | 6 | 51411 |
| S21 | PASSED | 1 | 63.4 | 15 | 129643 |
| S22 | PASSED | 1 | 24.5 | 9 | 96504 |
| S23 | PASSED | 1 | 33.3 | 5 | 77897 |
| S24 | PASSED | 1 | 4.6 | 2 | 824 |
| S25 | PASSED | 1 | 12.7 | 4 | 2014 |
| S26 | PASSED | 1 | 20.4 | 2 | 1113 |
| S27 | PASSED | 1 | 26.7 | 7 | 27209 |
| S28 | PASSED | 1 | 18.1 | 8 | 3296 |
| S29 | PASSED | 1 | 2.3 | 1 | 588 |
| S30 | PASSED | 1 | 30.1 | 6 | 2488 |
| S31 | PASSED | 1 | 8.5 | 1 | 525 |
| S32 | PASSED | 1 | 14.2 | 3 | 1657 |
| S33 | PASSED | 1 | 129.1 | 27 | 401770 |
| S34 | PASSED | 1 | 27.2 | 8 | 3256 |
| S35 | PASSED | 1 | 21.8 | 6 | 2464 |
| S36 | PASSED | 1 | 26.3 | 7 | 2748 |
| S37 | PASSED | 1 | 13.6 | 4 | 1693 |
| S38 | PASSED | 1 | 26.7 | 10 | 4772 |
| S39 | PASSED | 1 | 30.3 | 4 | 2368 |
| S40 | PASSED | 1 | 32.7 | 10 | 71571 |
| S41 | PASSED | 1 | 29.9 | 6 | 100034 |
| S42 | PASSED | 1 | 52.7 | 10 | 101281 |
| S43 | PASSED | 1 | 14.4 | 3 | 1682 |
| S44 | PASSED | 1 | 12.6 | 3 | 1355 |
| S45 | PASSED | 1 | 12.2 | 3 | 42227 |
| S46 | PASSED | 1 | 4.2 | 0 | 0 |
| S47 | PASSED | 1 | 5.9 | 2 | 1182 |
| S48 | PASSED | 1 | 25.4 | 7 | 3001 |
| S49 | PASSED | 1 | 0.0 | 0 | 0 |
| S50 | PASSED | 1 | 24.4 | 5 | 1891 |
| S51 | PASSED | 1 | 14.6 | 4 | 1567 |
| S52 | PASSED | 1 | 18.3 | 3 | 1363 |
