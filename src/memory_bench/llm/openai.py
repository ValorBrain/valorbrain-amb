import json
import os
import time

from .base import LLM, Schema

_MAX_RETRIES = 6
_RETRY_BASE_DELAY = 5



def _extract_json(text: str | None) -> dict:
    """Parse a JSON object out of an LLM reply.

    Reasoning models routed via OpenRouter may ignore strict json_schema and
    answer in prose, wrap the object in ```json fences, or prepend commentary.
    Try direct parse first, then the outermost balanced {...} block."""
    if not text or not text.strip():
        raise ValueError("empty LLM response")
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if "```" in text:
        for chunk in text.split("```"):
            c = chunk.strip()
            if c.startswith("json"):
                c = c[4:].strip()
            if c.startswith("{"):
                try:
                    return json.loads(c)
                except json.JSONDecodeError:
                    pass
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    start = None
    raise ValueError(f"no JSON object found in LLM response: {text[:120]!r}")

class OpenAILLM(LLM):
    def __init__(self, model: str = "gpt-4o"):
        from openai import OpenAI
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self._model = model

    @property
    def model_id(self) -> str:
        return f"openai:{self._model}"

    def generate(self, prompt: str, schema: Schema) -> dict:
        schema_json = {
            "type": "object",
            "properties": schema.properties,
            "required": schema.required,
            "additionalProperties": False,
        }
        delay = _RETRY_BASE_DELAY
        last_exc = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "response", "schema": schema_json, "strict": True},
                    },
                )
                text = response.choices[0].message.content
                try:
                    data = _extract_json(text)
                except Exception:
                    data = None
                if data is None:
                    # Prosa sem JSON nenhum: usa o texto cru como answer em vez
                    # de derrubar o eval inteiro; o juiz avalia o conteúdo.
                    return {"answer": (text or "").strip()[:8000]}
                required = [k for k in (schema.required or []) if k not in data]
                if required:
                    # Modelo ignorou o schema (típico de reasoning models via
                    # OpenRouter): uma tentativa corretiva explicitando as chaves.
                    fix_prompt = (
                        prompt
                        + "\n\nSua resposta anterior não seguiu o formato pedido "
                        f"(faltaram as chaves: {', '.join(required)}). "
                        + "Responda APENAS com um único objeto JSON contendo "
                        + "EXATAMENTE estas chaves: "
                        + ", ".join(required) + "."
                    )
                    response2 = self._client.chat.completions.create(
                        model=self._model,
                        messages=[{"role": "user", "content": fix_prompt}],
                        response_format={
                            "type": "json_schema",
                            "json_schema": {"name": "response", "schema": schema_json, "strict": True},
                        },
                    )
                    try:
                        fixed = _extract_json(response2.choices[0].message.content)
                        if all(k in fixed for k in schema.required or []):
                            return fixed
                    except Exception:
                        pass
                    # nem o retry voltou JSON utilizável
                    return {"answer": (text or "").strip()[:8000]}
                return data
            except Exception as e:
                last_exc = e
                msg = str(e)
                if "429" in msg or "rate" in msg.lower():
                    if attempt < _MAX_RETRIES - 1:
                        time.sleep(delay)
                        delay *= 2
                        continue
                raise
        raise RuntimeError(f"OpenAI request failed after {_MAX_RETRIES} retries: {last_exc}")
