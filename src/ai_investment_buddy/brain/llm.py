"""LLM abstraction: one structured-output call, three interchangeable backends.

Every backend takes (system prompt, user message, tool spec) and returns the
validated argument dict for the forced tool call. The tool spec is the standard
``{name, description, input_schema}`` shape (see prompts.DECISION_TOOL); each
client adapts it to its provider's function-calling format.

Select the backend with AIB_LLM_PROVIDER = anthropic | openai | gemini.
"""

from __future__ import annotations

import json
from typing import Callable, Protocol

from ..config import SETTINGS

# Default cap on tool-use rounds in an agentic loop (memory lookups + final).
_MAX_AGENT_ITERS = 8

# Extended-thinking token budgets per effort level (Anthropic & Gemini). "off" is
# absent → no thinking requested. Anthropic needs max_tokens > budget.
_ANTHROPIC_THINK_BUDGET = {"low": 2048, "medium": 6144, "high": 12288}
_GEMINI_THINK_BUDGET = {"low": 2048, "medium": 8192, "high": 24576}


class LLMClient(Protocol):
    def structured_call(self, system: str, user: str, tool: dict) -> dict:
        """Return the tool-call arguments as a dict."""
        ...

    def agentic_call(
        self,
        system: str,
        user: str,
        helper_tools: list[dict],
        final_tool: dict,
        executor: Callable[[str, dict], str],
        max_iters: int = _MAX_AGENT_ITERS,
    ) -> dict:
        """Run a tool-use loop: the model may call helper_tools repeatedly (each
        executed via ``executor``) until it calls ``final_tool``, whose arguments
        are returned. All tool specs use the {name, description, input_schema}
        shape; each client adapts to its provider's format."""
        ...


# --- Anthropic / Claude ------------------------------------------------------
class AnthropicClient:
    def __init__(self) -> None:
        from anthropic import Anthropic

        self.client = Anthropic(
            api_key=SETTINGS.anthropic_api_key,
            timeout=SETTINGS.llm_timeout,
            max_retries=SETTINGS.llm_max_retries,
        )

    def _tuning(self, forced_choice: dict) -> dict:
        """Per-call knobs. With extended thinking ON, Anthropic requires
        temperature=1 and forbids forced tool_choice (must be 'auto'), and
        max_tokens must exceed the thinking budget — so we trade strict
        determinism for reasoning. With thinking OFF we keep the low-temp + forced
        tool call for reproducibility."""
        budget = _ANTHROPIC_THINK_BUDGET.get(SETTINGS.reasoning_effort, 0)
        if budget:
            return {
                "thinking": {"type": "enabled", "budget_tokens": budget},
                "temperature": 1.0,
                "max_tokens": SETTINGS.max_decision_tokens + budget,
                "tool_choice": {"type": "auto"},
            }
        return {
            "temperature": SETTINGS.decision_temperature,
            "max_tokens": SETTINGS.max_decision_tokens,
            "tool_choice": forced_choice,
        }

    def structured_call(self, system: str, user: str, tool: dict) -> dict:
        resp = self.client.messages.create(
            model=SETTINGS.decision_model,
            system=system,
            tools=[tool],
            messages=[{"role": "user", "content": user}],
            **self._tuning({"type": "tool", "name": tool["name"]}),
        )
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and block.name == tool["name"]:
                return dict(block.input)
        raise RuntimeError(
            f"Anthropic returned no tool call (stop_reason={getattr(resp, 'stop_reason', '?')})."
        )

    def agentic_call(self, system, user, helper_tools, final_tool, executor,
                     max_iters=_MAX_AGENT_ITERS):
        tools = helper_tools + [final_tool]
        messages = [{"role": "user", "content": user}]
        for _ in range(max_iters):
            resp = self.client.messages.create(
                model=SETTINGS.decision_model,
                system=system,
                tools=tools,
                messages=messages,
                **self._tuning({"type": "any"}),
            )
            messages.append({"role": "assistant", "content": resp.content})
            tool_results = []
            for block in resp.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                if block.name == final_tool["name"]:
                    return dict(block.input)
                out = executor(block.name, dict(block.input))
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": out}
                )
            if not tool_results:
                break
            messages.append({"role": "user", "content": tool_results})
        # Out of iterations: force the final tool once.
        return self.structured_call(
            system, user + "\n\n(Stop researching; submit your decision now.)", final_tool
        )


# --- OpenAI / OpenAI-compatible ----------------------------------------------
class OpenAIClient:
    def __init__(self) -> None:
        from openai import OpenAI

        self.client = OpenAI(
            api_key=SETTINGS.openai_api_key,
            base_url=SETTINGS.openai_base_url,  # None => default OpenAI endpoint
            timeout=SETTINGS.llm_timeout,
            max_retries=SETTINGS.llm_max_retries,
        )
        # Some reasoning models reject temperature/seed; some non-reasoning models
        # reject reasoning_effort. Each flag is disabled after a 400 names it, so we
        # don't keep sending an unsupported param for the rest of the process.
        self._sampling = True
        self._reasoning = True

    def _create(self, **kwargs):
        """chat.completions.create adding low-temp+seed (reproducibility) and
        reasoning_effort (think harder), each stripped and retried if the model
        rejects it. Reasoning models typically drop the former; classic chat models
        drop the latter."""
        # Reasoning tokens are charged against max_completion_tokens, so give the
        # model headroom to think AND still emit the tool call (else it can burn the
        # budget reasoning and return no tool_calls).
        if (self._reasoning and SETTINGS.reasoning_effort != "off"
                and "max_completion_tokens" in kwargs):
            kwargs = {**kwargs, "max_completion_tokens": kwargs["max_completion_tokens"] + 16_000}
        for _ in range(3):  # at most a couple of param-strip retries
            extra: dict = {}
            if self._sampling:
                extra["temperature"] = SETTINGS.decision_temperature
                if SETTINGS.decision_seed is not None:
                    extra["seed"] = SETTINGS.decision_seed
            if self._reasoning and SETTINGS.reasoning_effort != "off":
                extra["reasoning_effort"] = SETTINGS.reasoning_effort
            try:
                return self.client.chat.completions.create(**kwargs, **extra)
            except Exception as e:
                msg = str(e).lower()
                changed = False
                if self._sampling and any(w in msg for w in ("temperature", "seed", "sampling")):
                    self._sampling = False
                    changed = True
                if self._reasoning and "reasoning" in msg:
                    self._reasoning = False
                    changed = True
                if not changed:
                    raise
        return self.client.chat.completions.create(**kwargs)

    def structured_call(self, system: str, user: str, tool: dict) -> dict:
        fn_tool = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool["input_schema"],
            },
        }
        resp = self._create(
            model=SETTINGS.decision_model,
            max_completion_tokens=SETTINGS.max_decision_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tools=[fn_tool],
            tool_choice={"type": "function", "function": {"name": tool["name"]}},
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            raise RuntimeError("OpenAI-compatible model returned no tool call.")
        return json.loads(msg.tool_calls[0].function.arguments)

    @staticmethod
    def _fn_tool(t: dict) -> dict:
        return {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t["input_schema"],
            },
        }

    def agentic_call(self, system, user, helper_tools, final_tool, executor,
                     max_iters=_MAX_AGENT_ITERS):
        tools = [self._fn_tool(t) for t in helper_tools + [final_tool]]
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        for _ in range(max_iters):
            resp = self._create(
                model=SETTINGS.decision_model,
                max_completion_tokens=SETTINGS.max_decision_tokens,
                messages=messages,
                tools=tools,
                tool_choice="required",
            )
            msg = resp.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))
            if not msg.tool_calls:
                break
            final = None
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                if tc.function.name == final_tool["name"]:
                    final = args
                    continue
                out = executor(tc.function.name, args)
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": out}
                )
            if final is not None:
                return final
        return self.structured_call(
            system, user + "\n\n(Stop researching; submit your decision now.)", final_tool
        )


# --- Google Gemini -----------------------------------------------------------
class GeminiClient:
    def __init__(self) -> None:
        from google import genai

        self.genai = genai
        self.client = genai.Client(api_key=SETTINGS.gemini_api_key)

    def _thinking(self):
        """A ThinkingConfig with a budget for the chosen effort, or None ('off' →
        the model's default)."""
        budget = _GEMINI_THINK_BUDGET.get(SETTINGS.reasoning_effort, 0)
        if not budget:
            return None
        from google.genai import types

        try:
            return types.ThinkingConfig(thinking_budget=budget)
        except Exception:
            return None  # older SDK without thinking support → use default

    def structured_call(self, system: str, user: str, tool: dict) -> dict:
        from google.genai import types

        fn = types.FunctionDeclaration(
            name=tool["name"],
            description=tool.get("description", ""),
            parameters=tool["input_schema"],
        )
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=SETTINGS.max_decision_tokens,
            temperature=SETTINGS.decision_temperature,
            seed=SETTINGS.decision_seed,
            thinking_config=self._thinking(),
            tools=[types.Tool(function_declarations=[fn])],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY", allowed_function_names=[tool["name"]]
                )
            ),
        )
        resp = self.client.models.generate_content(
            model=SETTINGS.decision_model,
            contents=user,
            config=config,
        )
        for part in resp.candidates[0].content.parts:
            if getattr(part, "function_call", None):
                return dict(part.function_call.args)
        raise RuntimeError("Gemini returned no function call.")

    def agentic_call(self, system, user, helper_tools, final_tool, executor,
                     max_iters=_MAX_AGENT_ITERS):
        from google.genai import types

        decls = [
            types.FunctionDeclaration(
                name=t["name"],
                description=t.get("description", ""),
                parameters=t["input_schema"],
            )
            for t in helper_tools + [final_tool]
        ]
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=SETTINGS.max_decision_tokens,
            temperature=SETTINGS.decision_temperature,
            seed=SETTINGS.decision_seed,
            thinking_config=self._thinking(),
            tools=[types.Tool(function_declarations=decls)],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY")
            ),
        )
        contents = [types.Content(role="user", parts=[types.Part(text=user)])]
        for _ in range(max_iters):
            resp = self.client.models.generate_content(
                model=SETTINGS.decision_model, contents=contents, config=config
            )
            cand = resp.candidates[0]
            contents.append(cand.content)
            responses = []
            for part in cand.content.parts:
                fc = getattr(part, "function_call", None)
                if not fc:
                    continue
                if fc.name == final_tool["name"]:
                    return dict(fc.args)
                out = executor(fc.name, dict(fc.args))
                responses.append(
                    types.Part.from_function_response(
                        name=fc.name, response={"result": out}
                    )
                )
            if not responses:
                break
            contents.append(types.Content(role="user", parts=responses))
        return self.structured_call(
            system, user + "\n\n(Stop researching; submit your decision now.)", final_tool
        )


_REGISTRY = {
    "anthropic": AnthropicClient,
    "openai": OpenAIClient,
    "gemini": GeminiClient,
}


def get_llm_client() -> LLMClient:
    provider = SETTINGS.llm_provider
    if provider not in _REGISTRY:
        raise ValueError(
            f"Unknown AIB_LLM_PROVIDER '{provider}'. "
            f"Choose one of: {', '.join(_REGISTRY)}."
        )
    if not SETTINGS.llm_api_key:
        raise RuntimeError(
            f"{SETTINGS.llm_key_env_name()} is not set (required for "
            f"AIB_LLM_PROVIDER={provider}). Add it to your environment or .env file."
        )
    return _REGISTRY[provider]()
