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

S9 has one intentionally narrow direct-database fixture: after the same
loopback-only `DATABASE_URL` validation, the runner upserts six fixed rows into
`zdic_pinyin_cache` and no other table. `keytao-next` has no API for its internal
pronunciation cache, while the child-process proxy confinement makes live
`zdic.net` fetches impossible by design. The rows mark `射覆` and the fixture word
`慑服` as absent whole-word entries and provide their required character readings,
so both the fixture API flow and S9's review flow remain deterministic offline.

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
removed through an approved API batch after the scenario; a compatible
pre-existing non-rig fixture is preserved.

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
