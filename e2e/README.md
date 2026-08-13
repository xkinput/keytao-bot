# Real-LLM E2E rig

This directory contains an opt-in scenario rig. It is intentionally not part of
the repository's normal test suites: it calls a real paid LLM endpoint and a
local `keytao-next` server backed by the local development database.

## Safety model

The runner constructs `KEYTAO_API_BASE` itself as
`http://localhost:<port>`. It refuses a non-local KeyTao origin, a non-local
PostgreSQL host in `keytao-next/.env`, an LLM origin that names a KeyTao
production host, and any test binding without the reserved E2E name/email and a
30-plus-digit synthetic QQ ID. Scenario identities are freshly generated per
run and verified through `/api/bot/user/find` before use. S8 also provisions a
fresh reserved local manager by first using `keytao-next/scripts/initBotUser.ts`,
then applying a rig-owned, fail-closed `R:MANAGER` promotion. The runner logs in
through `/api/auth/login` and re-verifies the JWT through `/api/auth/me`; ordinary
names, email domains, QQ-shaped IDs, missing bot roles, and missing admin roles
are rejected before the identity can approve anything.

Every Python HTTP hop, including redirects, is checked immediately before
transport dispatch. Raw Python sockets are also limited to loopback addresses
and the resolved IP/port of the one configured LLM origin. In particular,
`keytao.vercel.app` is always blocked. The local next process is started only
after its effective database URL has been verified as loopback-only. Its
telemetry is disabled, and HTTP(S) proxy variables confine any incidental child
process HTTP attempt to a closed loopback port while localhost bypasses remain
available. Conversation history, compressed memory, and uncertain-mutation
fences are initialized inside the current run's artifact directory before the
real chat plugin is imported.

Scenarios can declare an intentionally narrow direct-database pronunciation
fixture. After the same loopback-only `DATABASE_URL` validation, the runner
upserts only the selected scenarios' declared rows into `zdic_pinyin_cache` and
no other table. `keytao-next` has no API for its internal pronunciation cache,
while the child-process proxy confinement makes live `zdic.net` fetches
impossible by design. S9 declares six rows for `射覆` and `慑服`. S10 declares
found character readings for `王`, `中`, `微`, `服`, and `务`, plus absent
whole-word entries for `王中王` and `微服务`. Duplicate declarations are collapsed
by `(kind, entry)` before the upsert.
S14 declares found character readings for `亮` and `面` plus an absent
whole-word entry for `亮面`. S15 reuses the existing S9 `射覆`/`慑服`
candidate fixture and the S14 `亮面` pronunciation fixture.
S16 declares the `zài liú`, `zài liú zǐ`, and `zuò luò zài` character reality
plus absent whole-word entries for `载流`, `载流子`, and `座落在`.

## Prerequisites

- `keytao-next` exists at `../keytao-next`, has dependencies installed, and its
  `.env` points to a running localhost PostgreSQL development database.
- `BOT_API_TOKEN` exists in `keytao-next/.env`.
- The bot `.env` contains the real OpenAI-compatible key, base URL, and model.
  The current repository configuration uses the Doubao endpoint.
- keytao-next's local `node_modules/.bin/next` and `tsx`, plus this repository's
  `.venv`, are available. The rig never invokes Corepack or a package registry.
- The local dictionary is either empty at `wkxk` or already contains exactly
  `赤溪@wkxk`, Phrase, weight 100. The rig seeds that fixture through local next
  bot APIs when it is absent. It also repairs the exact interrupted S8 post-world
  through a compensating API batch. Because `/by-code?code=wkxk` is a prefix
  query, repair selects only rows whose code is exactly `wkxk` or the recorded
  shifted code. Every removed row must be owned by a user whose name starts with
  `keytao-e2e-llm-rig-`; non-rig or otherwise ambiguous rows fail closed.

Real model calls can incur provider charges. The artifacts record request count
and token usage. Monetary cost is left unset because provider pricing is not
available to the runner.

## Run

```bash
.venv/bin/python -m e2e.run
.venv/bin/python -m e2e.run --only S4
.venv/bin/python -m e2e.run --only S8
.venv/bin/python -m e2e.run --only S9
.venv/bin/python -m e2e.run --port 3101
```

The runner reuses a compatible local next server on the selected port or starts
the installed local Next.js binary. A process started by the rig is stopped on exit. Results and
full transcripts are written under `e2e/artifacts/<timestamp>-<run-id>/`.
Each failed scenario is cleaned and rerun once; two consecutive failures produce
a FAILED verdict and a non-zero exit code.

S8 reuses the S1 real-LLM positional move, submits the resulting batch through
the bot preview/confirm API, and approves it through the real admin endpoint. It
asserts the persisted `needsManualReview=true` seal on the new `吃席@wkxk` row,
the Approved batch/PR states, reviewer identity, and final dictionary layout.
Admin mode is the product's human-review gate, so it legitimately accepts the
sealed row without clearing or bypassing the flag. After all scenario assertions
pass, the runner creates, submits, and admin-approves a compensating API batch
that removes exact-code rig-owned fixture rows and restores sole
`赤溪@wkxk` weight 100. Cleanup failure is reported as rig infrastructure failure,
not as a failed product assertion. Fixture preflight recognizes the same exact
post-S8 world and performs the API-only repair after an interrupted prior run;
ambiguous state still fails closed. No raw SQL is used for dictionary cleanup.

S9 provisions `慑服@eefj` when necessary, then sends the uncommon new word
`射覆` through the real chat entry point. It asserts that the reply contains the
candidate commonness assessment and one of the supported reorder, keep-order,
or insufficient-signal recommendations. It also proves the user's draft remains
exactly empty: candidate ordering is advisory until the user explicitly selects
an action. Its preflight requires the seeded `射覆` character lookups to be
`found` and the exact candidate chain `eefj`, `eefju`, `eefjuv`; it no longer
accepts the proxy-induced `zdic-unavailable` fallback. A rig-owned S9 fixture is
removed through an approved API batch after the scenario; a non-rig row at any
declared S9 word slot fails closed before the scenario runs.

Before any scenario with a ZDIC declaration dispatches an attempt, the rig first
derives its dictionary slots from that scenario's probe words and whole-word
fixture rows. Rig-owned leftovers are removed through an approved API batch;
non-rig rows fail closed. The rig then probes the declared whole words through
local next four times with the existing `4s`, `5s`, `6s` warm-up backoff. Only
the final probe is asserted. It must report an absent whole-word lookup and the
exact seeded found reading for every character; a cold or stale dev server that
still reports `zdic-unavailable` therefore fails as rig infrastructure before
any scenario assertion or model call. The same API-only removal runs after each
attempt so preflight repair remains an aborted-run safety net.

S11 keeps the confirmation path when the server returns a wider live ticket and
always rejects provisional batch links; the narrow named-occupant shape may
instead complete through its single server-bound replay. S12 asserts that narrow
shape materializes in one message with the exact persisted weight cascade
(`吃席@wkxk` 100 and `赤溪@wkxk` 101), with no value below its type base. S13 sends an explicit weight
adjustment against an empty draft and asserts a deterministic failure that never
asks the user to resend the same current message. The orchestrating session owns
live-rig execution; ordinary offline verification must not start a server.

S14 injects a 汉典-shaped search hit and page for the different entry `光面`
with pinyin `guāng miàn` while the requested word is `亮面`. The injection is
armed only for S14 and returns synthetic responses before the network allowlist;
no external request is dispatched. The scenario rejects any `guang` syllable or
`gxmm*` candidate and accepts only an `lxmm*` chain or the existing fail-closed
manual-pronunciation path.

S15 first discovers the server-issued `射覆` candidate list, replies
`2 添加并提交`, and verifies candidate 2 reaches a submitted batch without the
old execution-verb rejection. It then sends bare `添加并提交` for `亮面`. Direct
completion, with at most one legitimate server-bound `确认`, satisfies the
sub-case only when the discovered code reaches a submitted batch with its
manual-review seal intact. If the flow instead stops with a corrected
add-and-submit suggestion, S15 copies the rendered `「...」` string literally
and verifies that command reaches a submitted batch without another correction.

S16 replays the two-word 载流 production transcript with `座落在@zlz` seeded as
a Phrase, weight 100 dictionary occupant: it discovers `载流@zhlq` and
`载流子@zlzu` in one turn, sends bare `加入并提交` without a native quote, and
requires both exact items to reach the same submitted batch.
At most one server-bound `确认` step may intervene. The flow rejects
target-completion guidance and any submit-only remediation.

Optional overrides:

- `E2E_OPENAI_API_KEY`, `E2E_OPENAI_BASE_URL`, `E2E_OPENAI_MODEL`
- `E2E_OPENAI_TIMEOUT`, `E2E_OPENAI_MAX_TOKENS`, `E2E_OPENAI_TEMPERATURE`
- `E2E_KEYTAO_PORT`, `E2E_NEXT_START_TIMEOUT`, `E2E_MESSAGE_TIMEOUT`
- `E2E_ENCODE_DELAY_ONCE_SECONDS`, `E2E_ENCODE_ATTEMPT_TIMEOUT_SECONDS`

The encode delay is armed only by S7. Its first matching GET is delayed and
failed before dispatch, so the production retry path must issue the subsequent
real localhost request and emit a retry log.

## Add a scenario

Add an async scenario function in `scenarios.py`, assert the local next draft
snapshot and only stable reply markers, then append it to `SCENARIOS`. Give it a
new ID so it receives a dedicated generated user. Do not assert exact model
wording, patch model responses, replace tool implementations, or add the rig to
the normal offline suites.

Offline safety checks can be run without keys or network:

```bash
.venv/bin/python -m unittest e2e.test_safety
```
