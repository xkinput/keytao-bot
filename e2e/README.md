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
run and verified through `/api/bot/user/find` before use.

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
  bot APIs when it is absent; it never repairs ambiguous dictionary state.

Real model calls can incur provider charges. The artifacts record request count
and token usage. Monetary cost is left unset because provider pricing is not
available to the runner.

## Run

```bash
.venv/bin/python -m e2e.run
.venv/bin/python -m e2e.run --only S4
.venv/bin/python -m e2e.run --port 3101
```

The runner reuses a compatible local next server on the selected port or starts
the installed local Next.js binary. A process started by the rig is stopped on exit. Results and
full transcripts are written under `e2e/artifacts/<timestamp>-<run-id>/`.
Each failed scenario is cleaned and rerun once; two consecutive failures produce
a FAILED verdict and a non-zero exit code.

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
