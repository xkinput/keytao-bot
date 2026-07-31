"""DeepSeek-aware Chat Completions request policy and usage metrics."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


_SAMPLING_PARAMETERS = (
    "temperature",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
)
_REASONING_EFFORTS = frozenset({"high", "max"})


def is_deepseek_model(model: Any) -> bool:
    """Return whether a configured model uses DeepSeek-specific parameters."""
    return str(model or "").strip().lower().startswith("deepseek-")


def with_deepseek_chat_policy(
    request: Mapping[str, Any],
    *,
    thinking: Optional[bool] = None,
    reasoning_effort: str = "high",
    json_output: bool = False,
) -> Dict[str, Any]:
    """Apply task-specific DeepSeek options without changing other providers."""
    configured = dict(request)
    if not is_deepseek_model(configured.get("model")):
        return configured

    if thinking is not None:
        extra_body = dict(configured.get("extra_body") or {})
        extra_body["thinking"] = {"type": "enabled" if thinking else "disabled"}
        configured["extra_body"] = extra_body

        if thinking:
            normalized_effort = str(reasoning_effort or "high").strip().lower()
            if normalized_effort not in _REASONING_EFFORTS:
                raise ValueError(f"Unsupported DeepSeek reasoning effort: {reasoning_effort}")
            configured["reasoning_effort"] = normalized_effort
            for parameter in _SAMPLING_PARAMETERS:
                configured.pop(parameter, None)
        else:
            configured.pop("reasoning_effort", None)

    if json_output:
        configured["response_format"] = {"type": "json_object"}

    return configured


def _value(source: Any, *names: str) -> Any:
    for name in names:
        if isinstance(source, Mapping):
            value = source.get(name)
        else:
            value = getattr(source, name, None)
        if value is not None:
            return value
    return None


def _token_count(value: Any) -> Optional[int]:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    return max(count, 0)


def chat_usage_metrics(response: Any) -> Dict[str, int]:
    """Normalize Chat Completions and Responses-style token usage fields."""
    usage = _value(response, "usage")
    if usage is None:
        return {}

    input_tokens = _token_count(_value(usage, "prompt_tokens", "input_tokens"))
    output_tokens = _token_count(_value(usage, "completion_tokens", "output_tokens"))
    total_tokens = _token_count(_value(usage, "total_tokens"))

    input_details = _value(usage, "prompt_tokens_details", "input_tokens_details")
    output_details = _value(usage, "completion_tokens_details", "output_tokens_details")
    cached_tokens = _token_count(
        _value(usage, "prompt_cache_hit_tokens")
        if _value(usage, "prompt_cache_hit_tokens") is not None
        else _value(input_details, "cached_tokens")
    )
    cache_miss_tokens = _token_count(_value(usage, "prompt_cache_miss_tokens"))
    reasoning_tokens = _token_count(_value(output_details, "reasoning_tokens"))

    if cache_miss_tokens is None and input_tokens is not None and cached_tokens is not None:
        cache_miss_tokens = max(input_tokens - cached_tokens, 0)

    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    metrics = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cached_tokens": cached_tokens,
        "cache_miss_tokens": cache_miss_tokens,
    }
    return {name: value for name, value in metrics.items() if value is not None}


def log_chat_usage(logger: Any, response: Any, *, operation: str, model: str) -> None:
    """Log non-sensitive token usage for cost and response attribution."""
    metrics = chat_usage_metrics(response)
    if not metrics:
        return
    response_model = str(_value(response, "model") or model)
    response_id = _value(response, "id")
    system_fingerprint = _value(response, "system_fingerprint")
    choices = _value(response, "choices") or []
    finish_reason = _value(choices[0], "finish_reason") if choices else None
    metadata = [f"operation={operation}", f"model={response_model}"]
    if response_id:
        metadata.append(f"response_id={response_id}")
    if system_fingerprint:
        metadata.append(f"system_fingerprint={system_fingerprint}")
    if finish_reason:
        metadata.append(f"finish_reason={finish_reason}")
    details = " ".join(f"{name}={value}" for name, value in metrics.items())
    logger.info(f"LLM usage: {' '.join(metadata)} {details}")
