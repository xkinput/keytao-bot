"""CLI entry point for the opt-in real-LLM E2E rig."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from .recording import ArtifactRecorder
from .runtime import (
    E2EBotHarness,
    LocalNextClient,
    NextServer,
    RigInfrastructureError,
    assert_runtime_configuration,
    provision_test_user,
    test_identity,
)
from .safety import (
    EncodeDelayController,
    NetworkAllowlist,
    SafetyViolation,
    validate_keytao_base,
    validate_llm_base,
    validate_next_database_url,
    validate_test_binding,
)
from .scenarios import (
    SCENARIOS,
    ScenarioContext,
    ordered_candidate_codes,
    run_scenario,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NEXT_DIR = REPO_ROOT.parent / "keytao-next"


def _nonempty(value: Any) -> str:
    return str(value or "").strip()


def _first_value(*values: Any) -> str:
    for value in values:
        text = _nonempty(value)
        if text:
            return text
    return ""


def load_configuration(args: argparse.Namespace) -> dict[str, Any]:
    bot_env_path = REPO_ROOT / ".env"
    next_env_path = args.next_dir / ".env"
    if not bot_env_path.is_file():
        raise RigInfrastructureError(f"Missing bot env file: {bot_env_path}")
    if not next_env_path.is_file():
        raise RigInfrastructureError(f"Missing keytao-next env file: {next_env_path}")
    bot_values = dotenv_values(bot_env_path)
    next_values = dotenv_values(next_env_path)
    database_url = _nonempty(next_values.get("DATABASE_URL"))
    database = validate_next_database_url(database_url)
    bot_token = _nonempty(next_values.get("BOT_API_TOKEN"))
    if not bot_token:
        raise SafetyViolation("keytao-next .env has no BOT_API_TOKEN")
    llm = {
        "api_key": _first_value(
            os.getenv("E2E_OPENAI_API_KEY"),
            bot_values.get("OPENAI_API_KEY"),
            bot_values.get("ARK_API_KEY"),
            bot_values.get("GEMINI_API_KEY"),
        ),
        "base_url": validate_llm_base(
            _first_value(
                os.getenv("E2E_OPENAI_BASE_URL"),
                bot_values.get("OPENAI_BASE_URL"),
                bot_values.get("ARK_BASE_URL"),
                bot_values.get("GEMINI_BASE_URL"),
            )
        ),
        "model": _first_value(
            os.getenv("E2E_OPENAI_MODEL"),
            bot_values.get("OPENAI_MODEL"),
            bot_values.get("ARK_MODEL"),
            bot_values.get("GEMINI_MODEL"),
        ),
    }
    if not llm["api_key"] or not llm["model"]:
        raise SafetyViolation("A real E2E LLM key and model are required")
    port = int(args.port)
    if port < 1024 or port > 65535:
        raise SafetyViolation("The local keytao-next port must be between 1024 and 65535")
    keytao_base = validate_keytao_base(f"http://localhost:{port}")
    child_env = dict(os.environ)
    child_env.pop("DATABASE_URL", None)
    for key, value in next_values.items():
        if value is not None:
            child_env[str(key)] = str(value)
    child_env["DATABASE_URL"] = database_url
    child_env["BOT_API_TOKEN"] = bot_token
    child_env["NODE_ENV"] = "development"
    child_env["NEXT_TELEMETRY_DISABLED"] = "1"
    child_env["NODE_USE_ENV_PROXY"] = "1"
    child_env["HTTP_PROXY"] = "http://127.0.0.1:9"
    child_env["HTTPS_PROXY"] = "http://127.0.0.1:9"
    child_env["ALL_PROXY"] = "http://127.0.0.1:9"
    child_env["NO_PROXY"] = "localhost,127.0.0.1,::1"
    return {
        "bot_env_path": bot_env_path,
        "next_env_path": next_env_path,
        "database": database,
        "bot_token": bot_token,
        "llm": llm,
        "keytao_base": keytao_base,
        "next_child_env": child_env,
        "bot_values": bot_values,
    }


def apply_bot_environment(config: dict[str, Any]) -> None:
    llm = config["llm"]
    bot_values = config["bot_values"]
    os.environ.update(
        {
            "KEYTAO_API_BASE": config["keytao_base"],
            "BOT_API_TOKEN": config["bot_token"],
            "KEYTAO_USER_API_KEYS": "{}",
            "BOT_USER_API_KEYS": "{}",
            "OPENAI_API_KEY": llm["api_key"],
            "OPENAI_BASE_URL": llm["base_url"],
            "OPENAI_MODEL": llm["model"],
        }
    )
    optional = {
        "OPENAI_TIMEOUT": _first_value(
            os.getenv("E2E_OPENAI_TIMEOUT"), bot_values.get("OPENAI_TIMEOUT")
        ),
        "OPENAI_MAX_TOKENS": _first_value(
            os.getenv("E2E_OPENAI_MAX_TOKENS"), bot_values.get("OPENAI_MAX_TOKENS")
        ),
        "OPENAI_TEMPERATURE": _first_value(
            os.getenv("E2E_OPENAI_TEMPERATURE"), bot_values.get("OPENAI_TEMPERATURE")
        ),
    }
    for key, value in optional.items():
        if value:
            os.environ[key] = value


async def build_fixture_facts(client: LocalNextClient) -> dict[str, Any]:
    chixi = await client.phrases_by_word("赤溪")
    wkxk = await client.phrases_by_code("wkxk")
    exact_chixi = [
        item
        for item in chixi
        if item.get("word") == "赤溪"
        and item.get("code") == "wkxk"
        and item.get("type") == "Phrase"
        and item.get("weight") == 100
    ]
    if len(chixi) != 1 or len(exact_chixi) != 1:
        raise RigInfrastructureError(
            f"Local fixture requires exactly 赤溪@wkxk weight 100 Phrase; found {chixi}"
        )
    if len(wkxk) != 1 or wkxk[0].get("word") != "赤溪" or wkxk[0].get("weight") != 100:
        raise RigInfrastructureError(
            f"Local fixture requires wkxk to contain only 赤溪@100; found {wkxk}"
        )
    chixi_encode = await client.encode("赤溪")
    chixi_codes = ordered_candidate_codes(chixi_encode)
    if "wkxk" not in chixi_codes or chixi_codes.index("wkxk") + 1 >= len(chixi_codes):
        raise RigInfrastructureError(f"赤溪 encode chain has no served successor: {chixi_encode}")
    chixi_next = chixi_codes[chixi_codes.index("wkxk") + 1]
    chixi_next_occupants = await client.phrases_by_code(chixi_next)
    if chixi_next_occupants:
        raise RigInfrastructureError(
            f"赤溪 immediate successor {chixi_next} is occupied: {chixi_next_occupants}"
        )
    chixi_subject_encode = await client.encode("吃席")
    subject_codes = ordered_candidate_codes(chixi_subject_encode)
    if "wkxk" not in subject_codes:
        raise RigInfrastructureError(f"吃席 encode chain does not include wkxk: {chixi_subject_encode}")
    subject_next_free = ""
    checked_slots: list[dict[str, Any]] = []
    for code in subject_codes[subject_codes.index("wkxk") + 1 :]:
        occupants = await client.phrases_by_code(code)
        checked_slots.append({"code": code, "occupants": occupants})
        if not occupants:
            subject_next_free = code
            break
    if not subject_next_free:
        raise RigInfrastructureError(f"吃席 has no free served candidate after wkxk: {checked_slots}")
    return {
        "dictionary": {"byWord": chixi, "byCode": wkxk},
        "chixiCandidateCodes": chixi_codes,
        "chixi_next_code": chixi_next,
        "chixiSubjectCandidateCodes": subject_codes,
        "chixiSubjectCheckedSlots": checked_slots,
        "chixi_subject_next_free_code": subject_next_free,
    }


async def ensure_fixture(
    *,
    client: LocalNextClient,
    seed_identity: dict[str, str],
) -> dict[str, Any]:
    chixi = await client.phrases_by_word("赤溪")
    if not chixi:
        wkxk = await client.phrases_by_code("wkxk")
        if wkxk:
            raise RigInfrastructureError(
                f"Cannot safely seed 赤溪 because wkxk is already occupied: {wkxk}"
            )
        await client.clean_draft(seed_identity["platform_id"])
        await client.seed_phrase(
            platform_id=seed_identity["platform_id"],
            word="赤溪",
            code="wkxk",
        )
    return await build_fixture_facts(client)


def initialize_openai_chat(config: dict[str, Any], *, state_dir: Path) -> Any:
    apply_bot_environment(config)
    import nonebot

    nonebot.init()
    from keytao_bot.utils import draft_mutation_store
    from keytao_bot.utils import history_store as history_store_module
    from keytao_bot.utils import memory_store as memory_store_module
    from keytao_bot.utils.draft_mutation_store import DraftMutationClaimStore
    from keytao_bot.utils.history_store import HistoryStore
    from keytao_bot.utils.memory_store import ScopedMemoryStore

    state_dir.mkdir(parents=True, exist_ok=True)
    history_store_module._history_store = HistoryStore(str(state_dir / "history.db"))
    memory_store_module._memory_store = ScopedMemoryStore(str(state_dir / "memory.db"))
    draft_mutation_store._DEFAULT_STORE = DraftMutationClaimStore(
        str(state_dir / "draft-mutation-claims.db")
    )
    from keytao_bot.plugins import openai_chat

    assert_runtime_configuration(
        openai_chat,
        keytao_base=config["keytao_base"],
        llm=config["llm"],
    )
    module_name = str(getattr(openai_chat.AsyncOpenAI, "__module__", ""))
    if not module_name.startswith("openai"):
        raise SafetyViolation(
            f"The real OpenAI SDK client is not active: AsyncOpenAI module={module_name}"
        )
    return openai_chat


def make_safety_proof(guard: NetworkAllowlist) -> dict[str, Any]:
    proof: dict[str, Any] = {
        "productionUrlBlockedBeforeDispatch": False,
        "remoteDatabaseRejected": False,
        "productionLikeBindingRejected": False,
    }
    try:
        guard.assert_url_allowed("https://keytao.vercel.app/api/phrases")
    except SafetyViolation:
        proof["productionUrlBlockedBeforeDispatch"] = True
    try:
        validate_next_database_url("postgresql://user:pass@db.example.com:5432/keytao")
    except SafetyViolation:
        proof["remoteDatabaseRejected"] = True
    try:
        validate_test_binding(
            platform_id="12345678",
            expected_name="real-user",
            expected_email="person@example.com",
            user={"name": "real-user", "email": "person@example.com", "roles": []},
        )
    except SafetyViolation:
        proof["productionLikeBindingRejected"] = True
    if not all(proof.values()):
        raise SafetyViolation(f"Safety self-check did not fail closed: {proof}")
    return proof


def print_table(results: list[dict[str, Any]]) -> None:
    headers = ("Scenario", "Verdict", "Attempts", "Seconds", "LLM requests", "Tokens")
    rows = []
    for result in results:
        cost = result.get("cost", {})
        rows.append(
            (
                result["scenarioId"],
                result["verdict"],
                str(result["attempts"]),
                f"{result['durationSeconds']:.1f}",
                str(cost.get("modelRequests", 0)),
                str(cost.get("totalTokens", 0)),
            )
        )
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print("\nScenario verdicts")
    print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


async def async_main(args: argparse.Namespace) -> int:
    config = load_configuration(args)
    run_id = uuid.uuid4().hex
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    artifact_dir = REPO_ROOT / "e2e" / "artifacts" / f"{timestamp}-{run_id[:8]}"
    recorder = ArtifactRecorder(artifact_dir)
    encode_delay = EncodeDelayController(
        delay_seconds=float(os.getenv("E2E_ENCODE_DELAY_ONCE_SECONDS", "0.20")),
        attempt_timeout_seconds=float(os.getenv("E2E_ENCODE_ATTEMPT_TIMEOUT_SECONDS", "0.05")),
    )
    guard = NetworkAllowlist(
        llm_base_url=config["llm"]["base_url"],
        recorder=recorder,
        scenario_getter=recorder.current_scenario,
        encode_delay=encode_delay,
    )
    safety_proof = make_safety_proof(guard)
    selected_ids = {args.only.upper()} if args.only else {item.scenario_id for item in SCENARIOS}
    scenarios = [item for item in SCENARIOS if item.scenario_id in selected_ids]
    if not scenarios:
        raise RigInfrastructureError(f"Unknown scenario: {args.only}")
    server = NextServer(
        next_dir=args.next_dir,
        base_url=config["keytao_base"],
        artifact_dir=artifact_dir,
        start_timeout=float(args.next_start_timeout),
        child_env=config["next_child_env"],
    )
    bot_harness: E2EBotHarness | None = None
    results: list[dict[str, Any]] = []
    identities = {
        scenario.scenario_id: test_identity(run_id, scenario.scenario_id)
        for scenario in scenarios
    }
    seed_identity = test_identity(run_id, "seed")
    manifest = {
        "runId": run_id,
        "startedAt": timestamp,
        "repoHead": "",
        "keytaoBase": config["keytao_base"],
        "nextDatabase": config["database"],
        "llm": {
            "baseHost": guard.llm_origin[1],
            "model": config["llm"]["model"],
            "apiKeyPresent": True,
        },
        "selectedScenarios": [item.scenario_id for item in scenarios],
        "safetyProof": safety_proof,
        "identities": identities,
    }
    try:
        head = await asyncio.to_thread(
            subprocess_check_output,
            ["git", "rev-parse", "HEAD"],
            REPO_ROOT,
        )
        manifest["repoHead"] = head.strip()
        guard.install()
        await server.ensure_running()
        manifest["nextServer"] = {
            "reusedExisting": server.reused_existing,
            "startedByRig": not server.reused_existing,
        }
        client = LocalNextClient(
            base_url=config["keytao_base"],
            bot_token=config["bot_token"],
        )
        all_identities = [seed_identity, *identities.values()]
        for identity in all_identities:
            await provision_test_user(
                client=client,
                next_dir=args.next_dir,
                next_env=config["next_child_env"],
                identity=identity,
            )
        fixture_facts = await ensure_fixture(client=client, seed_identity=seed_identity)
        recorder.write_json("fixture-facts.json", fixture_facts)
        state_dir = artifact_dir / "state"
        openai_chat = initialize_openai_chat(config, state_dir=state_dir)
        recorder.install_log_sink()
        bot_harness = E2EBotHarness(
            openai_chat=openai_chat,
            recorder=recorder,
            state_dir=state_dir,
            message_timeout=float(args.message_timeout),
        )
        for scenario in scenarios:
            final: dict[str, Any] | None = None
            total_duration = 0.0
            aggregate_cost = {
                "modelRequests": 0,
                "promptTokens": 0,
                "completionTokens": 0,
                "totalTokens": 0,
                "models": set(),
                "monetaryCost": None,
                "costNote": "Provider billing price is not available locally; token usage is recorded.",
            }
            for attempt in (1, 2):
                with recorder.scope(scenario.scenario_id, attempt):
                    await client.clean_draft(identities[scenario.scenario_id]["platform_id"])
                    await bot_harness.reset_conversation(
                        platform_id=identities[scenario.scenario_id]["platform_id"]
                    )
                    context = ScenarioContext(
                        scenario_id=scenario.scenario_id,
                        attempt=attempt,
                        identity=identities[scenario.scenario_id],
                        next_client=client,
                        bot=bot_harness,
                        recorder=recorder,
                        encode_delay=encode_delay,
                        fixture_facts=fixture_facts,
                    )
                    attempt_result = await run_scenario(scenario, context)
                    cost = recorder.cost_summary(scenario.scenario_id, attempt)
                    attempt_result["cost"] = cost
                    total_duration += float(attempt_result["durationSeconds"])
                    aggregate_cost["modelRequests"] += cost["modelRequests"]
                    aggregate_cost["promptTokens"] += cost["promptTokens"]
                    aggregate_cost["completionTokens"] += cost["completionTokens"]
                    aggregate_cost["totalTokens"] += cost["totalTokens"]
                    aggregate_cost["models"].update(cost["models"])
                    artifact_path = recorder.write_attempt(
                        scenario_id=scenario.scenario_id,
                        attempt=attempt,
                        result=attempt_result,
                    )
                    attempt_result["artifact"] = str(artifact_path.relative_to(REPO_ROOT))
                    final = attempt_result
                if attempt_result["verdict"] == "PASSED":
                    break
                print(
                    f"{scenario.scenario_id} attempt {attempt} failed: "
                    f"{attempt_result['failure']}"
                )
            assert final is not None
            aggregate_cost["models"] = sorted(aggregate_cost["models"])
            results.append(
                {
                    "scenarioId": scenario.scenario_id,
                    "name": scenario.name,
                    "verdict": final["verdict"],
                    "attempts": 1 if final.get("artifact", "").endswith("attempt-1.json") else 2,
                    "durationSeconds": total_duration,
                    "failure": final.get("failure"),
                    "cost": aggregate_cost,
                    "finalAttemptArtifact": final.get("artifact"),
                    "facts": final.get("facts", {}),
                }
            )
            await client.clean_draft(identities[scenario.scenario_id]["platform_id"])
        live_model_exchanges = [
            event
            for event in recorder.events
            if event.get("kind") == "modelExchange" and event.get("status") == 200
        ]
        if not args.only and not live_model_exchanges:
            raise RigInfrastructureError("The full rig completed without a real successful LLM exchange")
        manifest["realLlmProof"] = {
            "successfulHttpExchanges": len(live_model_exchanges),
            "clientClass": f"{openai_chat.AsyncOpenAI.__module__}.{openai_chat.AsyncOpenAI.__name__}",
            "fakeClientPresentInMessagePath": False,
        }
        manifest["completedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        manifest["results"] = results
        recorder.write_json("manifest.json", manifest)
        recorder.write_json("summary.json", {"results": results})
        print_table(results)
        print(f"\nArtifacts: {artifact_dir.relative_to(REPO_ROOT)}")
        return 1 if any(item["verdict"] != "PASSED" for item in results) else 0
    except BaseException as error:
        manifest["abortedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        manifest["abort"] = {"type": type(error).__name__, "message": str(error)}
        recorder.write_json("manifest.json", manifest)
        raise
    finally:
        if bot_harness is not None:
            await bot_harness.close()
        recorder.remove_log_sink()
        await server.stop()
        guard.restore()


def subprocess_check_output(command: list[str], cwd: Path) -> str:
    import subprocess

    return subprocess.check_output(command, cwd=cwd, text=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="Run one scenario, for example S4")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("E2E_KEYTAO_PORT", "3100")),
        help="Local keytao-next port (default: 3100)",
    )
    parser.add_argument(
        "--next-dir",
        type=Path,
        default=DEFAULT_NEXT_DIR,
        help="Read-only keytao-next source directory",
    )
    parser.add_argument(
        "--next-start-timeout",
        type=float,
        default=float(os.getenv("E2E_NEXT_START_TIMEOUT", "120")),
    )
    parser.add_argument(
        "--message-timeout",
        type=float,
        default=float(os.getenv("E2E_MESSAGE_TIMEOUT", "360")),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return asyncio.run(async_main(args))
    except (SafetyViolation, RigInfrastructureError) as error:
        print(f"E2E aborted safely: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("E2E interrupted; owned local server was stopped.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
