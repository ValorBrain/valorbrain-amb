import logging
import os

from .base import LLM, Schema
from .gemini import GeminiLLM
from .groq import GroqLLM
from .gateway import GatewayLLM, ServedModelMismatch, parse_extra_body
from .openai import OpenAILLM
from .glm import GlmLLM
from .agy_direct import AgyDirectLLM

logger = logging.getLogger(__name__)

REGISTRY: dict[str, type[LLM]] = {
    "gemini": GeminiLLM,
    "groq": GroqLLM,
    "openai": OpenAILLM,
    "gateway": GatewayLLM,
    "glm": GlmLLM,
    "agy-direct": AgyDirectLLM,
}


def get_llm(name: str = "gemini") -> LLM:
    if name not in REGISTRY:
        raise ValueError(f"Unknown LLM: '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name]()


def _gateway_role_kwargs(role: str) -> dict:
    """GatewayLLM kwargs from the per-role env vars OMB_<ROLE>_*."""
    kwargs = {}
    eff = os.environ.get(f"OMB_{role}_REASONING_EFFORT")
    if eff:
        kwargs["reasoning_effort"] = eff
    for suffix, dst in [("BASE_URL", "base_url"), ("API_KEY", "api_key")]:
        value = os.environ.get(f"OMB_{role}_{suffix}")
        if value:
            kwargs[dst] = value
    extra_var = f"OMB_{role}_EXTRA_BODY"
    raw_extra = os.environ.get(extra_var, "").strip()
    if raw_extra:
        kwargs["extra_body"] = parse_extra_body(raw_extra, extra_var)
    strict_var = f"OMB_{role}_STRICT_MODEL"
    strict = os.environ.get(strict_var, "").strip()
    if strict not in ("", "0", "1"):
        raise ValueError(f"{strict_var} must be 0 or 1, got {strict!r}")
    if strict == "1":
        kwargs["strict_model"] = True
    return kwargs


def _role_llm(role: str, default_provider: str) -> LLM:
    """Build the LLM for a role from OMB_<ROLE>_LLM / OMB_<ROLE>_MODEL (+ gateway settings)."""
    provider_var = f"OMB_{role}_LLM"
    provider = os.environ.get(provider_var, default_provider)
    model = os.environ.get(f"OMB_{role}_MODEL")
    cls = REGISTRY.get(provider)
    if cls is None:
        raise ValueError(f"Unknown {provider_var}: '{provider}'. Available: {list(REGISTRY)}")
    if provider == "gateway":
        kwargs = _gateway_role_kwargs(role)
    else:
        kwargs = {}
        if os.environ.get(f"OMB_{role}_STRICT_MODEL", "").strip() == "1":
            logger.warning(
                "OMB_%s_STRICT_MODEL=1 has no effect: the served-model check exists only "
                "for the gateway provider, and %s is %r", role, provider_var, provider,
            )
    return cls(model, **kwargs) if model else cls(**kwargs)


def get_answer_llm() -> LLM:
    """Return the LLM to use for RAG answer generation.
    Configured via OMB_ANSWER_LLM (provider) and OMB_ANSWER_MODEL (optional model override);
    the gateway provider also reads OMB_ANSWER_{REASONING_EFFORT,BASE_URL,API_KEY,EXTRA_BODY,STRICT_MODEL}."""
    return _role_llm("ANSWER", "groq")


def get_judge_llm() -> LLM:
    """Return the LLM to use for evaluation/judging.
    Configured via OMB_JUDGE_LLM (provider) and OMB_JUDGE_MODEL (optional model override);
    the gateway provider also reads OMB_JUDGE_{REASONING_EFFORT,BASE_URL,API_KEY,EXTRA_BODY,STRICT_MODEL}."""
    return _role_llm("JUDGE", "gemini")
