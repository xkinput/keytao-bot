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
S17 declares known readings for `产`, `季`, and the obscure control character
`龘`, while keeping both `产季` and `龘季` absent as whole-word entries. The rig
also builds the complete vendored pronunciation/commonness database inside the
run artifact state before importing the bot, so S17 exercises real offline
character frequencies instead of reduced test rows.
S18 declares the `huán/hái`, `chē`, and `huàn` character readings for `还`,
`车`, and `换`, mirrors the authoritative `还车` whole-word entry as
`huán chē`, keeps `换车` absent as a pronunciation entry, and seeds `换车@htwe`
as the exact local dictionary occupant.
S19 declares authoritative whole-word readings for its 11 advertised common
words while the scenario repair step removes those words from the local
dictionary. This keeps the scan result absent and the later review deterministic.
S20 reuses the first three S19 word/character declarations for a native-quoted
batch confirmation and applies the same rig-owned dictionary cleanup.
S22 reuses the first two S19 word/character declarations for the orphaned
re-review advertisement replay and applies the same rig-owned cleanup.
S23 keeps the first nine S19 declarations for stale advertised-assent recovery;
the forced state loss is rig-owned and does not touch the dictionary.
S24 reuses the S18 `还车` pronunciation fixture for one single-word candidate;
it does not reuse S18's multi-selection action and adds no new dictionary fact.
S27 seeds the local Next ZDIC cache with the authoritative `lái dōu lái le`
whole-word reading and its three unique character rows, while user-dictionary
repair keeps `来都来了` absent. The separately built vendored bot reference DB
has no exact whole-word reading; the scenario follows the seeded local encode
reality.
S29 declares authoritative readings for `火锅` and `电脑`, clears only those
rig-owned dictionary words, then seeds the exact `mkdr` Phrase chain at weights
100 and 101. The complete vendored commonness database supplies the ordering
evidence; the scenario never depends on an external search result.
S32 declares authoritative readings for `米等`, `幂等`, and `迷瞪`, clears only
those rig-owned dictionary words, then seeds `米等@mkdr(100)`,
`迷瞪@mkdro(100)`, and a current-draft `幂等@mkdr(101)` projection.
S35 declares authoritative readings for the isolated `发布会`/`重病号`,
`计算机`/`建三江`, and `无事忙` controls. It clears only those declared
rig-owned words, seeds the exact `重病号@fbh` and `建三江@jsj` occupants, and
uses the vendored commonness reference for both front-insert verdicts.

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
The writable Next build cache and checksum-idempotent pinyin reference database
are reused under ignored `e2e/.runtime/`; per-run `state/` contains only the
conversation databases. At startup, artifacts are pruned to the newest five
runs including the run in progress.
Each failed scenario is cleaned and rerun once; two consecutive failures produce
a FAILED verdict and a non-zero exit code.

S4 first proves that an expired confirmation cannot write or invite a blind
resend. It then seeds one exact draft record, requests its deletion through the
real chat path, and requires the first user prompt to be composed from the
server-locked target facts. Exactly one `确认` must remove the record; a second
confirmation prompt or any `PR#...` identifier fails the scenario.

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
the comparator-backed `载流子@zlz` front insert in one turn, sends bare
`加入并提交` without a native quote, and requires the free add, newcomer create,
and sealed `座落在` shift to reach the same submitted batch.
At most one server-bound `确认` step may intervene. The flow rejects
target-completion guidance and any submit-only remediation.

S17 exercises the common-characters-plus-LLM semantic auto-pass lane. It asks
for `产季`, requires the compact review line to name the concrete semantic basis
and exact offline character frequencies, submits the selected server-issued
code, and verifies the persisted item has `needsManualReview=false` and the
batch reaches `Approved` through the bot auto-approve route. It then submits
`龘季` as an obscure-character control and requires
`needsManualReview=true` with the batch left `Submitted`. Both whole-word ZDIC
entries and the whole-word offline commonness rows are absent. The lane has only
the whole-word `corpus_frequency` and `common_characters_and_llm` routes, so the
pass case must take the latter and cannot silently use whole-word evidence.

S18 replays the multi-number candidate incident for `还车`. It parses the live
rendered indices for the empty `htjev` slot and occupied `htwe` slot across the
authoritative `huán chē` and offline `hái chē` candidate readings, proves an
out-of-range mixed selection writes nothing and leaks no policy identifiers,
then sends one multi-number reply. The exact two-item set must land in one batch,
the duplicate item keeps its manual-review seal, the empty item keeps its own
audit verdict, and submission may use at most one server-bound `确认` step.

S19 replays the oversized advertised-set incident. It scans and renders 11
server-derived absent words, proves an out-of-snapshot exclusion asks without
writing, then sends `天选打工人先不要，其他可以加，沙县小吃也不要`. The nine-word
remainder must emit an 8/9 progress line, require one confirmation, and reach
one draft batch with no excluded or extra words and no `参数格式错误` diagnosis.

S20 replays native-quoted batch assent in a QQ group. It discovers and persists
the exact three-word `显眼包`/`嘴替`/`松弛感` batch, then sends a real OneBot
`reply` segment quoting that bot advertisement plus bare `都加`. The displayed
word/code pairs must reach one draft batch exactly, with at most one additional
server-bound confirmation and no `引用文字不能创建或恢复确认权限` response.

S21 replays the 2026-08-16 advertised-contract incidents against a live
actor-owned two-word batch ticket. Before each ticket is created, the scenario
cleans and verifies the actor draft and resets its conversation state. It proves
`都加 跳过X` writes exactly the one-word record-derived remainder, an
out-of-ticket exclusion asks without a write, and unrelated text outside a
quoted command still cannot authorize. It then takes the bot's own fully
rendered remediation line (bullet, command quote, and record-derived
parenthetical included), sends that whole line back verbatim, and requires the
exact live ticket set to reach one batch.

S22 replays the orphaned re-review advertisement incident with the minimum
multi-word shape. It first establishes an exact two-word candidate list, clears
the actor conversation to force state loss, then quotes that list while asking a
later read-only review to recompute the same exact words. The refreshed codes may
legitimately differ from the first display. If the re-review advertises
`加入并提交`, the delivery contract must rebuild an actor-owned ticket whose
bindings exactly equal the refreshed display. A following unquoted bare
`加入并提交` must write exactly those two displayed pairs to one batch. The
scenario rejects invented/dropped words, placeholder commands, dishonest
completion copy, and any response claiming no quote was present.

S23 replays the stale advertised-assent production incident directly. It first
establishes the exact nine-word candidate list, clears only the actor's live
conversation state, then natively quotes the stale bot message and sends
`加入并提交`. That same turn must run the existing review path for exactly the
display-bound words, emit a fresh state-backed candidate list, and perform zero
draft writes. A following unquoted `加入并提交` must write exactly the fresh
displayed set to one batch. Offline safety coverage also forces an unresolvable
quoted display and requires an honest renderer-backed `加词 ...` alternative
without minting a write ticket or inventing any word.

S24 replays the single-word natural-assent incident. It establishes the exact
`还车@htje` live candidate, verifies that this shape advertises only `加入` and
`加入并提交` as whole-state assent, then natively quotes that bot message and
sends `加入草稿，然后就提交。`. The exact state-derived item must reach one
submitted batch with at most one server-bound confirmation. No reply may ask
for a full word-plus-code line or claim that the selecting quote did not match.

S25 replays the 炒冷饭 production incident while separating explicit-code and
reviewed-code contracts. It first sends `补上炒冷饭的 wlf 编码` and requires the
natural add verb to reach the occupied-code write gate. In a clean actor state
it renders the encode-service `jlf` series, rejects leakage from the unrelated
explicit `wlf` fixture, sends the bare index for `jlf`, and requires the exact
trusted record item to reach the draft. After another explicit cleanup it sends
`添加 炒冷饭 jlf 并提交`; that same-turn item must reach one submitted
batch with at most one server-bound confirmation. No executable sub-path may
emit read-only, no-write, impossible-to-execute, or no-safe-next-command copy.
The rig seeds and verifies `窝里反@wlf` and `晚礼服@wlfo` as the exact occupied
fixtures for the explicit duplicate-code subcase; reviewed candidates remain
strictly bound to the encode service's `jlf` chain.

S26 replays the add-plus-eviction incident with
`添加 吃席 wkxk，赤溪顺延`. It requires one atomic draft batch containing the
newcomer at `wkxk` and the server-resolved occupant moved to its next free
candidate, with at most one confirmation. The completion must name both
outcomes, may not deny a completed write, and may never auto-confirm a
`duplicate_code` creation in place of the requested eviction.

S37 sends the verbatim occupant-derived eviction
`加词 耙耙柑 把琵琶骨顶掉`, requiring direct same-turn materialization because
the only shifted word is the named live occupant. The receipt must place
`耙耙柑@ppg`, move `琵琶骨` to its fixture-proven next free slot, name both
results, include the materialized batch link, and ask for no user confirmation.
The runner removes any prior rig-owned S37 rows before seeding and cleans them
through approved APIs after each attempt. A separate native-quote leg injects
an obsolete selected slot, requires a fresh current candidate list, and proves
the same normalized selection cannot repeat the old candidate-set refusal. The
offline cascade control still requires one explicit confirmation whenever the
shift would also displace an unnamed third word.

S38 pins the reading-chain incident round. It sends the verbatim
`加词 出圈，读音是 chū quān` and requires `requested_reading` to reach the
server review before candidates are rendered, then completes the add. It also
pins the explanatory `耙耙柑为pá pá gān，因此这三个字的声母分别为p, p, g`
turn and rejects reasoning-only exhaustion. Three clean query-only candidate
snapshots must recover through `1`, `回复1`, and `加入`; the negative modifier
`加词 耙耙柑 ppg，不要顺延其他相关的词条` must create the duplicate code without
shifting its occupant. The positive trailing-modifier control may advertise a
shift only when replay reaches the same reviewed encode record that authorized
the suggestion.

S39 collapses the explicit-reading eviction flow. `加词 出圈 圈字读quan`
must select the `chū quān` group from the six candidates returned by one
encode result and render `jjqt` occupied by `除权` plus the two free
successors, the manual-review seal, and the standard selectors in the first
turn. `1 重新编码` is the single confirmation in the second turn: it must
materialize `出圈@jjqt` and shift `除权` to its next free code without a
third user message. Controls require an unmatched reading to list both returned
readings, forbid an add-and-evict remediation from narrowing to add-only, and
prove `重新编码 "除权" jjqt` resolves the newcomer from the live candidate state.

S27 replays the binding-precheck incident with an intentionally unprovisioned
synthetic actor and the scenario's provisioned bound control. The unbound actor
asks for `来都来了`: the first deterministic candidate reply must include one
short binding notice without blocking the read-only review. `加入并提交` must then
retain the existing binding guidance. A following binding-process question must
receive a direct answer, call no tool, and contain none of the registered system
refusal markers. The bound control asks for the same word and must not receive
the notice.

S28 replays the multi-reading candidate cascade. It first proves that the exact
review-advertised recommendation survives write-side validation, then rebuilds
a live state and sends a complete same-word word+code command selecting a
different valid reading chain. A third clean state sends `添加4并提交` and must
retain the fourth candidate's code-to-reading binding through submission. The
invalid-code control must name compact per-reading chains, call no write tool,
and expose neither a raw candidate dump nor a root-only draft URL.

S29 replays the 2026-08-20 quoted-summary incident. It quotes a bot-authored
operation summary containing the `mkdr` diff and submit instruction, sends the
code-first actionable chain-reorder request, and requires a complete current-to-proposed
plan with evidence and exactly one confirmation. Before confirmation the draft
must remain empty; after confirmation it must contain only the two sealed
same-word Change operations at weights 100 and 101. A real quoted presence
question against the same summary remains on the presence-lookup route and
cannot change that draft.

S30 pins three intent-coverage boundaries in one actor flow. A bare `吃席`
lookup followed by `好` must make zero mutation calls; a fresh reviewed-add
ticket followed by `先别加` must cancel without advertising an add command;
and a second fresh ticket must accept `那就加入并提交吧`, preserving its exact
displayed word/code pair through submission.

S31 executes the verbatim incident command
`把 幂等 放到 米等 前面，米等顺延到下一个空位`. The rig seeds only the declared
local `米等@mkdr` fixture, requires the bot to traverse lookup plus the
server-generated circular shift plan, and verifies the exact three draft rows
before API-only fixture cleanup.

S32 replays both 2026-08-20 chain-scope incidents verbatim. The first command
must merge the live `mkdr` row with the current draft row and render one locked
weight plan; the scenario cancels that plan to retain the incident baseline.
The newline-separated three-word command must then resolve every word, include
`迷瞪@mkdro` in one `mkdr*` prefix-chain plan, and render only old-to-new moves,
one evidence summary, one confirmation line, and the draft link in at most eight
lines. The sealed incident shape is exactly `Delete 米等@mkdr`,
`Create 米等@mkdrou`, and `幂等@mkdr 101→100`; one confirmation must return
successful dictionary and draft-weight receipts and materialize all three
changes before the compact completion reply.

S33 replays both 2026-08-21 homophone batch shapes. The read-only control seeds
`所受@sled`, then sends the verbatim `缩手 所售`: one reviewed turn must show
the occupant and advertise `缩手→sleda` plus `所售→sledi` from one shared
snapshot, without a write ticket, binding error, or speculative commonness
copy. The write control with `洒漏` and `撒漏` must advertise two different
recommendations from their one shared `ssld*` occupancy snapshot, ordered by
the commonness comparator, and a bare `加入并提交` must submit those exact
distinct pairs without another prompt.
Controls preserve an explicit `同码` duplicate and a full six-code duplicate,
while a model-composed short-code collision must be replanned at the batch sink,
shown as one compact changed-code line, and stopped on one new confirmation.

S34 replays the 2026-08-21 pending-batch incident with `开团→khtt`. It creates
and submits one actor-owned manual-review batch through the local server, then
queries `开团` again. The reply must lead with the exact Submitted batch link,
must not restart reviewed-add discovery, and may advertise only actions that
remain available. A fresh explicit add followed by `加入` must stop at one
local confirmation before an exact duplicate reaches the server write route.
After cancellation, selecting a different reviewed code must proceed through
the same sink while preserving the Submitted reminder as the first line.

S35 replays the 2026-08-22 default-reorder incident without touching its
production-shaped words or batch. A comparator-backed front-insert review must
render one recommendation plus one numbered no-reorder opt-out. Bare
`加入并提交` must consume one sealed plan confirmation, shift the named occupant,
create the newcomer at the occupied slot, and submit that exact batch. A second
front-insert pair pins numbered fallback selection, while a free-chain control
pins the unchanged no-recommendation default.

S36 replays the 2026-08-23 delete-and-swap incident round: deleting an existing
dictionary word through one locked Delete confirmation and submit, a
code-qualified delete, a two-move code swap as one sealed plan, a bare
previous-turn record-backed action-name follow-up with its no-record word-query control, and an
exact same-chain priority swap over the draft-aware merged view. Every reply is
also checked for normalized internal tool identifiers and model-facing
`禁止重复调用` / `请直接根据…回复用户` directives.

Optional overrides:

- `E2E_OPENAI_API_KEY`, `E2E_OPENAI_BASE_URL`, `E2E_OPENAI_MODEL`
- `E2E_OPENAI_TIMEOUT`, `E2E_OPENAI_MAX_TOKENS`, `E2E_OPENAI_TEMPERATURE`
- `E2E_KEYTAO_PORT`, `E2E_NEXT_START_TIMEOUT`, `E2E_MESSAGE_TIMEOUT`
- `E2E_ARTIFACT_RETENTION` (default: `5`, minimum: `1`)
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
