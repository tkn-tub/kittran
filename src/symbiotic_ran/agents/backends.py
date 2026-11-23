from __future__ import annotations

import random
import threading
import time
from typing import List

from .. import security
from ..config import Tenant
from ..energy import render_report
from ..schemas import (
    Critique,
    L2Decision,
    MediatorDecision,
    PCDecision,
    RRMDecision,
    TenantProposal,
    ULRADecision,
)
from .prompts import (
    L2_SYSTEM_PROMPT,
    MEDIATOR_SYSTEM_PROMPT,
    PC_SYSTEM_PROMPT,
    RRM_MONOLITHIC_SYSTEM_PROMPT,
    TENANT_SYSTEM_PROMPT,
    UL_RA_SYSTEM_PROMPT,
    reflection_prompt,
)

ROLES = ("tenant", "mediator", "l2", "pc", "ulra")

DEFAULT_ROLES = {role: "llm" for role in ROLES}

STRUCTURED_OUTPUT_ATTEMPTS = 3

PROPOSER_MAX_TOKENS = 512
CRITIC_MAX_TOKENS = 1024

_UE_IDS = ("fwa_1", "fwa_2", "mtc_1", "iot_1")
_SLICE_OF_TENANT = {"tenant_A": "FWA", "tenant_B": "MTC", "tenant_C": "BE"}


class Agents:

    def __init__(self, tenant_fn, mediator_fn, l2_fn, pc_fn,
                 ulra_fn, critic_fn, roles: dict, rrm_fn=None):
        self.tenant_fn = tenant_fn
        self.mediator_fn = mediator_fn
        self.l2_fn = l2_fn
        self.pc_fn = pc_fn
        self.ulra_fn = ulra_fn
        self.critic_fn = critic_fn
        self.roles = roles
        self.rrm_fn = rrm_fn

    def uses_llm(self) -> bool:
        return "llm" in self.roles.values()


def build_agents(
    *,
    model: str,
    roles: dict | None = None,
    ran_off_threshold_mbps: float = 6.0,
    temperature: float = 0.4,
    thinking: bool | None = None,
    mock_profile: str = "good",
    need_critic: bool = True,
    engine: str = "ollama",
    base_url: str | None = None,
) -> Agents:
    resolved = dict(DEFAULT_ROLES)
    for role, backend in (roles or {}).items():
        if role not in ROLES:
            raise ValueError(
                f"unknown agent role {role!r}; known roles: {', '.join(ROLES)}")
        if backend not in ("llm", "mock"):
            raise ValueError(
                f"agent role {role!r} must be 'llm' or 'mock', got {backend!r}")
        resolved[role] = backend

    if mock_profile not in ("good", "flawed"):
        raise ValueError(f"mock_profile must be 'good' or 'flawed', got {mock_profile!r}")

    mocks = build_mock_agents(
        ran_off_threshold_mbps=ran_off_threshold_mbps,
        profile=mock_profile,
    )

    llm = None
    if "llm" in resolved.values() or need_critic:
        llm = build_llm_agents(
            model, temperature=temperature,
            thinking=thinking, engine=engine, base_url=base_url)

    def pick(role: str, key: str):
        source = llm if resolved[role] == "llm" else mocks
        assert source is not None
        return source[key]

    critic = llm["critic"] if llm is not None else _unavailable_critic

    return Agents(
        tenant_fn=pick("tenant", "tenant"),
        mediator_fn=pick("mediator", "mediator"),
        l2_fn=pick("l2", "l2"),
        pc_fn=pick("pc", "pc"),
        ulra_fn=pick("ulra", "ulra"),
        critic_fn=critic,
        roles=resolved,
        rrm_fn=pick("pc", "rrm"),
    )


def _unavailable_critic(role, context, decision):
    raise RuntimeError(
        "the Reflection critic is always LLM-backed, but no model client was "
        "built. Either enable a role that needs the model, or turn reflection "
        "off with --no-reflect."
    )


def backend_id_for(roles: dict, model: str, engine: str = "ollama") -> str:
    writers = {role: roles.get(role, "llm") for role in ("tenant", "mediator")}
    if all(v == "mock" for v in writers.values()):
        return "mock"
    if all(v == "llm" for v in writers.values()):
        return f"{engine}:{model}"
    mix = ",".join(f"{role}={writers[role]}" for role in ("tenant", "mediator"))
    return f"mix({mix}):{engine}:{model}"


def _preflight_ollama(model: str, host: str | None = None) -> None:
    import os

    import httpx

    host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    if not host.startswith("http"):
        host = f"http://{host}"
    try:
        resp = httpx.get(f"{host}/api/tags", timeout=5.0)
        resp.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            f"cannot reach Ollama at {host} ({exc}). Start it with `ollama serve`, "
            f"or run with --mock."
        ) from exc

    available = {m["name"] for m in resp.json().get("models", [])}
    if available and model not in available and f"{model}:latest" not in available:
        raise RuntimeError(
            f"model {model!r} is not pulled on {host}. Run `ollama pull {model}`, "
            f"pick another with --model, or use --mock. "
            f"Available: {', '.join(sorted(available)) or '(none)'}"
        )


_CALL_LOG = threading.local()


def _call_log() -> list[tuple[str, float, int]]:
    log = getattr(_CALL_LOG, "calls", None)
    if log is None:
        log = _CALL_LOG.calls = []
    return log


def drain_call_log() -> list[tuple[str, float, int]]:
    log = _call_log()
    drained = list(log)
    log.clear()
    return drained


def record_calls(calls) -> None:
    _call_log().extend(calls)


def timing_summary(calls) -> dict | None:
    if not calls:
        return None
    def usage(c):
        return c[4] if len(c) > 4 and c[4] else None

    summary = {
        "llm_calls": len(calls),
        "llm_seconds": round(sum(c[1] for c in calls), 2),
        "llm_retries": sum(c[2] - 1 for c in calls),
        "llm_fallbacks": sum(1 for c in calls if len(c) > 3 and c[3]),
        "llm_calls_detail": [{"what": c[0], "seconds": round(c[1], 2),
                              "attempts": c[2],
                              **({"fallback": True} if len(c) > 3 and c[3]
                                 else {}),
                              **({"tokens_in": usage(c)["in"],
                                  "tokens_out": usage(c)["out"]} if usage(c)
                                 else {})}
                             for c in calls],
    }
    if any(usage(c) for c in calls):
        summary["llm_tokens_in"] = sum(usage(c)["in"] for c in calls if usage(c))
        summary["llm_tokens_out"] = sum(usage(c)["out"] for c in calls if usage(c))
    return summary


RATE_LIMIT_BACKOFF_S = (5.0, 20.0, 45.0, 90.0)


def _is_rate_limited(exc: Exception) -> bool:
    if "ratelimit" in type(exc).__name__.lower():
        return True
    return getattr(exc, "status_code", None) == 429


INJECTABLE_ROLES = {
    "tenant": "tenant ", "mediator": "mediator", "l2": "l2_manager",
    "pc": "pc_agent", "ulra": "ulra_agent", "rrm": "rrm_agent",
    "critic": "reflection on",
}

_INJECTION = {"rates": {}, "rng": random.Random(0)}


def configure_failure_injection(rates: dict | None, *, seed: int | None = 0) -> None:
    rates = dict(rates or {})
    unknown = set(rates) - set(INJECTABLE_ROLES)
    if unknown:
        raise ValueError(f"cannot inject failures into {sorted(unknown)}; "
                         f"roles: {', '.join(INJECTABLE_ROLES)}")
    for role, rate in rates.items():
        if not 0.0 <= float(rate) <= 1.0:
            raise ValueError(f"injection rate for {role} must be in [0, 1], got {rate}")
    _INJECTION["rates"] = {INJECTABLE_ROLES[r]: float(v) for r, v in rates.items()}
    _INJECTION["rng"] = random.Random(seed)


def parse_injection_spec(spec: str | None) -> dict:
    out = {}
    for item in (spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        role, _, rate = item.partition(":")
        if not rate:
            raise ValueError(f"expected ROLE:RATE, got {item!r}")
        out[role.strip()] = float(rate)
    return out


def _poisoned(what: str) -> bool:
    for prefix, rate in _INJECTION["rates"].items():
        if what.startswith(prefix):
            return rate > 0 and _INJECTION["rng"].random() < rate
    return False


class _UsageTally:

    def __init__(self):
        from langchain_core.callbacks import BaseCallbackHandler

        class Handler(BaseCallbackHandler):
            def __init__(inner):
                super().__init__()
                inner.tokens_in = 0
                inner.tokens_out = 0
                inner.seen = False

            def on_llm_end(inner, response, **_kwargs):
                usage = None
                try:
                    usage = response.generations[0][0].message.usage_metadata
                except (AttributeError, IndexError):
                    usage = None
                if usage:
                    inner.tokens_in += int(usage.get("input_tokens", 0))
                    inner.tokens_out += int(usage.get("output_tokens", 0))
                    inner.seen = True
                    return
                token_usage = (response.llm_output or {}).get("token_usage") or {}
                if token_usage:
                    inner.tokens_in += int(token_usage.get("prompt_tokens", 0))
                    inner.tokens_out += int(token_usage.get("completion_tokens", 0))
                    inner.seen = True

        self.handler = Handler()

    @property
    def usage(self) -> dict | None:
        h = self.handler
        return {"in": h.tokens_in, "out": h.tokens_out} if h.seen else None


def _invoke_structured(chain, messages, *, what: str, verbose: bool = True,
                       fallback=None):
    from langchain_core.messages import HumanMessage

    last_error: Exception | None = None
    started = time.perf_counter()
    throttled = 0
    attempt = 0
    tally = _UsageTally()
    config = {"callbacks": [tally.handler]}
    poisoned = _poisoned(what)
    while attempt < STRUCTURED_OUTPUT_ATTEMPTS:
        attempt += 1
        try:
            if poisoned:
                raise ValueError("injected failure")
            result = chain.invoke(messages, config=config)
            if result is None:
                raise ValueError("structured output parsed to None")
            _call_log().append((what, time.perf_counter() - started, attempt,
                                False, tally.usage))
            return result
        except Exception as exc:
            last_error = exc
            if _is_rate_limited(exc) and throttled < len(RATE_LIMIT_BACKOFF_S):
                wait = RATE_LIMIT_BACKOFF_S[throttled]
                throttled += 1
                attempt -= 1
                if verbose:
                    print(f"      (rate limited on {what}; waiting {wait:.0f}s, "
                          f"backoff {throttled}/{len(RATE_LIMIT_BACKOFF_S)})")
                time.sleep(wait)
                continue
            if attempt == STRUCTURED_OUTPUT_ATTEMPTS:
                break
            if verbose:
                print(f"      (retrying {what} after {type(exc).__name__}; "
                      f"attempt {attempt + 1}/{STRUCTURED_OUTPUT_ATTEMPTS})")
            messages = list(messages) + [HumanMessage(
                "Your previous reply could not be parsed. Respond with ONLY the "
                "requested JSON object -- no prose, no code fences -- using "
                "exactly the field names from the schema."
            )]
    if fallback is not None:
        _call_log().append((what, time.perf_counter() - started, attempt, True,
                            tally.usage))
        if verbose:
            print(f"      ({what}: no valid answer after "
                  f"{STRUCTURED_OUTPUT_ATTEMPTS} attempts "
                  f"({type(last_error).__name__}); holding -- this decision "
                  f"changes nothing)")
        return fallback()
    raise RuntimeError(
        f"{what} did not return valid structured output after "
        f"{STRUCTURED_OUTPUT_ATTEMPTS} attempts: {last_error}"
    ) from last_error


DEFAULT_BASE_URL = {
    "ollama": "http://localhost:11434",
    "vllm": "http://localhost:8000/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
}

ENGINES = tuple(DEFAULT_BASE_URL)


def _preflight_vllm(model: str, base_url: str) -> None:
    import httpx

    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/models", timeout=5.0)
        resp.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            f"cannot reach vLLM at {base_url} ({exc}). Start the server, point "
            f"--base-url at it, or run with --mock."
        ) from exc

    available = {m["id"] for m in resp.json().get("data", [])}
    if available and model not in available:
        raise RuntimeError(
            f"model {model!r} is not served by {base_url}. Use --model with one "
            f"of: {', '.join(sorted(available))}"
        )


def _preflight_gemini(model: str, base_url: str) -> None:
    import os

    import httpx

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Export it, or use --engine ollama/vllm "
            "or --mock.")

    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/models",
                         headers={"Authorization": f"Bearer {key}"}, timeout=20.0)
        resp.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            f"cannot reach the Gemini API at {base_url} ({exc}). Check the key "
            f"and the network, or run with --mock."
        ) from exc

    available = {m["id"].removeprefix("models/") for m in resp.json().get("data", [])}
    if available and model.removeprefix("models/") not in available:
        raise RuntimeError(
            f"model {model!r} is not offered by the Gemini API. Try one of: "
            f"{', '.join(sorted(m for m in available if 'gemini' in m))}"
        )


def _build_chat_model(model: str, *, engine: str, temperature: float,
                      thinking: bool | None, base_url: str | None,
                      keep_alive: str, max_tokens: int = PROPOSER_MAX_TOKENS,
                      preflight: bool = True):
    if engine not in ENGINES:
        raise ValueError(
            f"unknown engine {engine!r}; known engines: {', '.join(ENGINES)}")

    if engine == "ollama":
        import os

        from langchain_ollama import ChatOllama
        host = base_url or os.environ.get(
            "OLLAMA_HOST", DEFAULT_BASE_URL["ollama"])
        if not host.startswith("http"):
            host = f"http://{host}"
        if preflight:
            _preflight_ollama(model, host)
        options = {"temperature": temperature, "keep_alive": keep_alive,
                   "base_url": host, "num_predict": max_tokens}
        if thinking is not None:
            options["reasoning"] = thinking
        return ChatOllama(model=model, **options), {}

    from langchain_openai import ChatOpenAI

    if engine == "gemini":
        import os

        url = base_url or DEFAULT_BASE_URL["gemini"]
        if preflight:
            _preflight_gemini(model, url)
        llm = ChatOpenAI(model=model, base_url=url,
                         api_key=os.environ["GEMINI_API_KEY"],
                         temperature=temperature, timeout=900, max_retries=0,
                         max_tokens=max_tokens)
        return llm, {"method": "function_calling"}

    url = base_url or DEFAULT_BASE_URL["vllm"]
    if preflight:
        _preflight_vllm(model, url)

    think = bool(thinking)
    extra_body = {"chat_template_kwargs": {"enable_thinking": think}}
    max_tokens = 32768 if think else max_tokens

    llm = ChatOpenAI(model=model, base_url=url, api_key="EMPTY",
                     temperature=temperature, timeout=900, max_retries=0,
                     max_tokens=max_tokens, extra_body=extra_body)
    return llm, {"method": "function_calling"}


def build_llm_agents(model: str, temperature: float = 0.4,
                     thinking: bool | None = None,
                     keep_alive: str = "30m",
                     engine: str = "ollama",
                     base_url: str | None = None) -> dict:
    from langchain_core.messages import HumanMessage, SystemMessage

    llm, structured_kwargs = _build_chat_model(
        model, engine=engine, temperature=temperature, thinking=thinking,
        base_url=base_url, keep_alive=keep_alive,
        max_tokens=PROPOSER_MAX_TOKENS)
    critic_llm, _ = _build_chat_model(
        model, engine=engine, temperature=temperature, thinking=thinking,
        base_url=base_url, keep_alive=keep_alive,
        max_tokens=CRITIC_MAX_TOKENS, preflight=False)

    def structured(schema, client=llm):
        return client.with_structured_output(schema, **structured_kwargs)

    tenant_chain = structured(TenantProposal)
    mediator_chain = structured(MediatorDecision)
    l2_chain = structured(L2Decision)
    pc_chain = structured(PCDecision)
    ulra_chain = structured(ULRADecision)
    rrm_chain = structured(RRMDecision)
    critic_chain = structured(Critique, critic_llm)

    last_ruling: dict = {"decision": None}

    def tenant_fn(tenant, profile_text, interval, achievable,
                  mediator_suggestion, round_idx) -> TenantProposal:
        system = TENANT_SYSTEM_PROMPT.format(
            tenant_profile=profile_text, achievable_mbps=achievable)
        user = (
            f"Negotiation round {round_idx}.\n"
            f"Recommended interval: [{interval[0]:.1f}, {interval[1]:.1f}] Mbps.\n"
            f"Mediator's current suggested SLA: {mediator_suggestion:.1f} Mbps.\n"
            f"Achievable total throughput right now: {achievable:.1f} Mbps.\n\n"
            "State your proposed SLA (Mbps) and whether you accept the "
            "mediator's suggestion."
        )
        return _invoke_structured(
            tenant_chain, [SystemMessage(system), HumanMessage(user)],
            what=f"tenant {tenant.id} round {round_idx}",
            fallback=lambda: TenantProposal(
                proposed_mbps=float(mediator_suggestion), accept=True,
                reasoning="held: no valid answer after retries; deferring to "
                          "the mediator's suggestion"))

    def mediator_fn(proposals, interval, achievable, tenant_objs,
                    critique=None, capacity_report=None) -> MediatorDecision:
        system = MEDIATOR_SYSTEM_PROMPT.format(achievable_mbps=achievable)
        lines = [
            f"Recommended interval: [{interval[0]:.1f}, {interval[1]:.1f}] Mbps.",
            f"Achievable total throughput: {achievable:.1f} Mbps.",
            "",
            "Tenant proposals this round:",
        ]
        for t, p in zip(tenant_objs, proposals, strict=True):
            starved = (" [CHANNEL CANNOT REACH THIS TENANT'S FLOOR]"
                       if achievable < t.min_acceptable_mbps else "")
            lines.append(
                f"- {t.name} ({t.id}): proposes {p.proposed_mbps:.1f} Mbps, "
                f"accept={p.accept}, floor {t.min_acceptable_mbps:.1f} Mbps, "
                f"utility weight {t.utility_weight}, willingness to pay "
                f"{t.willingness_to_pay}{starved}. {p.reasoning}"
            )
        if capacity_report is not None:
            lines.append("")
            lines.append(render_report(capacity_report))
        lines.append("")
        lines.append(
            "Give the consensus SLA, the on/off decision, and the share vector "
            "(share_tenant_A + share_tenant_B + share_tenant_C = 1.0).")
        messages = [SystemMessage(system), HumanMessage("\n".join(lines))]
        if critique is not None:
            messages.append(HumanMessage(
                "A reviewer rejected your previous ruling for these reasons:\n"
                + (critique.issues or "")
                + "\n\nApply this corrective guidance and produce a corrected "
                  "ruling:\n" + (critique.guidance or "")
            ))
        def hold():
            if last_ruling["decision"] is not None:
                return last_ruling["decision"].model_copy(update={
                    "reasoning": "held: no valid answer after retries; the "
                                 "previous ruling stands"})
            shares = _shares_from_tenants(tenant_objs, achievable)
            return MediatorDecision(
                suggested_sla_mbps=float((interval[0] + interval[1]) / 2),
                consensus_reached=False, switch_ran_off=False,
                share_tenant_A=shares["tenant_A"],
                share_tenant_B=shares["tenant_B"],
                share_tenant_C=shares["tenant_C"],
                reasoning="held: no valid answer after retries and no "
                          "previous ruling; the round is not concluded")

        ruling = _invoke_structured(mediator_chain, messages, what="mediator",
                                    fallback=hold)
        last_ruling["decision"] = ruling
        return ruling

    def _rrm_call(chain, system, context, critique, what, hold):
        messages = [SystemMessage(security.system_prompt(system, context)),
                    HumanMessage(context)]
        if critique is not None:
            messages.append(HumanMessage(
                "A reviewer rejected your previous proposal for these reasons:\n"
                + (critique.issues or "")
                + "\n\nApply this corrective guidance and produce a corrected "
                  "decision:\n"
                + (critique.guidance or "")
            ))
        return _invoke_structured(chain, messages, what=what, fallback=hold)

    held = "held: no valid answer after retries"

    def l2_fn(context, state=None, critique=None) -> L2Decision:
        def hold():
            prev = state or {}
            return L2Decision(
                pc_subintent=prev.get("pc_subintent")
                or f"{held}; carry on under the operator intent: "
                   f"{prev.get('operator_intent', '')}",
                ul_ra_subintent=prev.get("ul_ra_subintent")
                or f"{held}; carry on under the operator intent: "
                   f"{prev.get('operator_intent', '')}")
        return _rrm_call(l2_chain, L2_SYSTEM_PROMPT, context, critique,
                         "l2_manager", hold)

    def pc_hold(state) -> PCDecision:
        ues = (state or {}).get("ues", {})
        fields = {}
        for ue_id in ("fwa_1", "fwa_2", "mtc_1", "iot_1"):
            fields[f"{ue_id}_target_snr_db"] = float(
                ues.get(ue_id, {}).get("target_snr_db", 0.0))
            fields[f"{ue_id}_reasoning"] = held
        return PCDecision(**fields)

    def ulra_hold(state) -> ULRADecision:
        slices = (state or {}).get("slices", {})
        fields = {}
        for name in ("fwa", "mtc", "be"):
            sl = slices.get(name, {})
            fields[f"{name}_throttle_mbps"] = float(
                sl.get("throttle_mbps", 0.0))
            fields[f"{name}_priority"] = sl.get("priority", "normal")
            fields[f"{name}_reasoning"] = held
        return ULRADecision(**fields)

    def pc_fn(context, state=None, critique=None) -> PCDecision:
        return _rrm_call(pc_chain, PC_SYSTEM_PROMPT, context, critique,
                         "pc_agent", lambda: pc_hold(state))

    def ulra_fn(context, state=None, critique=None) -> ULRADecision:
        return _rrm_call(ulra_chain, UL_RA_SYSTEM_PROMPT, context, critique,
                         "ulra_agent", lambda: ulra_hold(state))

    pc_fn.hold = pc_hold
    ulra_fn.hold = ulra_hold

    def rrm_fn(context, state=None, critique=None) -> RRMDecision:
        def hold():
            return RRMDecision(**pc_fn.hold(state).model_dump(),
                               **ulra_fn.hold(state).model_dump())
        return _rrm_call(rrm_chain, RRM_MONOLITHIC_SYSTEM_PROMPT, context,
                         critique, "rrm_agent", hold)

    def critic_fn(role, context, decision) -> Critique:
        user = (
            context
            + "\n\nPROPOSED DECISION (under review):\n"
            + decision.model_dump_json(indent=2)
            + "\n\nReview this proposal."
        )
        return _invoke_structured(
            critic_chain,
            [SystemMessage(security.filter_prompt(reflection_prompt(role))),
             HumanMessage(user)],
            what=f"reflection on {role}",
            fallback=lambda: Critique(
                verdict="advise",
                issues="held: the reviewer returned no valid critique after "
                       "retries; the proposal was not reviewed",
                guidance=""))

    return {
        "tenant": tenant_fn, "mediator": mediator_fn,
        "l2": l2_fn, "pc": pc_fn, "ulra": ulra_fn, "critic": critic_fn,
        "rrm": rrm_fn,
    }


def _shares_from_tenants(tenants: List[Tenant], achievable: float) -> dict:
    raw = {}
    for t in tenants:
        w = max(0.0, t.utility_weight * t.willingness_to_pay)
        if achievable < t.min_acceptable_mbps:
            w *= 0.25
        raw[t.id] = w
    total = sum(raw.values()) or 1.0
    return {k: round(v / total, 4) for k, v in raw.items()}


def build_mock_agents(ran_off_threshold_mbps: float = 6.0,
                      profile: str = "good") -> dict:


    def tenant_fn(tenant: Tenant, profile_text, interval, achievable,
                  mediator_suggestion, round_idx) -> TenantProposal:
        low, high = interval
        if achievable < tenant.min_acceptable_mbps:
            proposed = max(0.0, round(achievable * tenant.willingness_to_pay, 1))
            return TenantProposal(
                proposed_mbps=proposed, accept=True,
                reasoning="Channel below my minimum; willing to accept a very low "
                          "SLA or RAN-off.")
        target = low + (high - low) * tenant.willingness_to_pay
        blend = min(1.0, 0.4 + 0.2 * round_idx)
        proposed = mediator_suggestion + blend * (target - mediator_suggestion)
        proposed = round(max(low, min(high, proposed)), 1)
        return TenantProposal(
            proposed_mbps=proposed,
            accept=abs(proposed - mediator_suggestion) <= 2.0,
            reasoning="Compromising toward the recommended interval.")

    def mediator_fn(proposals, interval, achievable, tenant_objs,
                    critique=None, capacity_report=None) -> MediatorDecision:
        low, high = interval
        vals = [p.proposed_mbps for p in proposals]
        suggested = max(low, min(high, round(sum(vals) / len(vals), 1)))
        spread = max(vals) - min(vals)
        accepts = sum(1 for p in proposals if p.accept)
        consensus = spread <= 3.0 and accepts >= max(1, len(proposals) - 1)
        switch_off = achievable < ran_off_threshold_mbps
        shares = _shares_from_tenants(list(tenant_objs), achievable)
        return MediatorDecision(
            suggested_sla_mbps=suggested,
            consensus_reached=consensus or switch_off,
            switch_ran_off=switch_off,
            share_tenant_A=shares.get("tenant_A", 0.34),
            share_tenant_B=shares.get("tenant_B", 0.33),
            share_tenant_C=shares.get("tenant_C", 0.33),
            reasoning="Averaged tenant proposals within the recommended interval; "
                      "shares weighted by utility and willingness to pay, with "
                      "starved tenants demoted.")


    def good_l2(context, state=None, critique=None) -> L2Decision:
        state = state or {}
        if state.get("ran_off"):
            return L2Decision(
                pc_subintent="The RAN is switched off and transmit power is "
                             "already zero. Hold every UE's target SNR steady so "
                             "the cell can recover immediately when it comes back.",
                ul_ra_subintent="The RAN is switched off. Keep all three slices at "
                                "normal priority and maximum throttle "
                                "(100 Mbit/s); favour no slice.")
        shares = state.get("tenant_shares") or {}
        sla = state.get("sla_mbps", 0.0)
        ranked = sorted(shares.items(), key=lambda kv: kv[1], reverse=True)
        if ranked:
            top_slice = _SLICE_OF_TENANT[ranked[0][0]]
            bottom_slice = _SLICE_OF_TENANT[ranked[-1][0]]
            split = ", ".join(
                f"{_SLICE_OF_TENANT[tid]} {frac * 100:.0f}%" for tid, frac in ranked)
        else:
            top_slice, bottom_slice, split = "FWA", "BE", "unspecified"
        return L2Decision(
            pc_subintent=(
                f"Raise every UE's target SNR toward its channel ceiling to lift "
                f"cell capacity toward the {sla:.1f} Mbit/s SLA. Prioritise the "
                f"UEs in slice {top_slice}."),
            ul_ra_subintent=(
                f"Divide capacity {split} of the {sla:.1f} Mbit/s SLA. Raise slice "
                f"{top_slice} to high priority AND lower slice {bottom_slice} to "
                f"low priority to make the split real; keep throttles at the "
                f"maximum 100 Mbit/s unless a slice must be capped."),
        )

    def good_pc(context, state=None, critique=None) -> PCDecision:
        state = state or {}
        ues = state.get("ues", {})
        text = (state.get("pc_subintent") or "").lower()
        if state.get("ran_off"):
            return PCDecision(**{
                f"{ue_id}_{suffix}": value
                for ue_id in _UE_IDS
                for suffix, value in (
                    ("target_snr_db", ues.get(ue_id, {}).get("target_snr_db", 0.0)),
                    ("reasoning", "RAN is off; holding the setpoint for a fast restart"),
                )
            })

        save_battery = any(
            phrase in text for phrase in
            ("save battery", "save as much battery", "save power",
             "minimise transmit power", "minimize transmit power",
             "reduce power", "lower the target snr", "conserve battery"))

        fields = {}
        for ue_id in _UE_IDS:
            ue = ues.get(ue_id, {})
            ceiling = ue.get("snr_ceiling_db", ue.get("target_snr_db", 0.0))
            battery_powered = ue.get("battery_pct") is not None
            if save_battery and battery_powered:
                target = ceiling - 3.0
                why = "backing off the ceiling to save battery"
            else:
                target = ceiling
                why = "driving toward the channel ceiling to lift cell capacity"
            fields[f"{ue_id}_target_snr_db"] = round(target, 2)
            fields[f"{ue_id}_reasoning"] = why
        return PCDecision(**fields)

    def good_ulra(context, state=None, critique=None) -> ULRADecision:
        state = state or {}
        shares = state.get("tenant_shares") or {}
        max_throttle = 100.0
        if state.get("ran_off") or not shares:
            return ULRADecision(
                fwa_throttle_mbps=max_throttle, fwa_priority="normal",
                fwa_reasoning="no differentiation required",
                mtc_throttle_mbps=max_throttle, mtc_priority="normal",
                mtc_reasoning="no differentiation required",
                be_throttle_mbps=max_throttle, be_priority="normal",
                be_reasoning="no differentiation required")

        ranked = sorted(shares.items(), key=lambda kv: kv[1], reverse=True)
        priority_by_slice = {}
        for rank, (tid, _frac) in enumerate(ranked):
            priority_by_slice[_SLICE_OF_TENANT[tid]] = (
                "high" if rank == 0 else "low" if rank == len(ranked) - 1 else "normal")
        sla = state.get("sla_mbps", 0.0)
        throttle_by_slice = {}
        for tid, frac in shares.items():
            throttle_by_slice[_SLICE_OF_TENANT[tid]] = round(
                min(max_throttle, max(5.0, frac * sla * 1.8)), 1)

        return ULRADecision(
            fwa_throttle_mbps=throttle_by_slice.get("FWA", max_throttle),
            fwa_priority=priority_by_slice.get("FWA", "normal"),
            fwa_reasoning="priority set from the mediator's share vector",
            mtc_throttle_mbps=throttle_by_slice.get("MTC", max_throttle),
            mtc_priority=priority_by_slice.get("MTC", "normal"),
            mtc_reasoning="priority set from the mediator's share vector",
            be_throttle_mbps=throttle_by_slice.get("BE", max_throttle),
            be_priority=priority_by_slice.get("BE", "normal"),
            be_reasoning="priority set from the mediator's share vector",
        )


    def flawed_l2(context, state=None, critique=None) -> L2Decision:
        if critique is not None:
            return good_l2(context, state)
        return L2Decision(
            pc_subintent="Keep all UE target SNRs unchanged.",
            ul_ra_subintent=(
                "Set throttle to 0 (unlimited) on all three slices and set every "
                "slice to high priority. Leave the competing slices unchanged."),
        )

    def flawed_pc(context, state=None, critique=None) -> PCDecision:
        if critique is not None:
            return good_pc(context, state)
        state = state or {}
        ues = state.get("ues", {})
        text = (state.get("pc_subintent") or "").lower()
        wants_up = any(w in text for w in ("raise", "increase", "lift", "toward"))
        if not wants_up:
            return good_pc(context, state)
        fields = {}
        for ue_id in _UE_IDS:
            current = ues.get(ue_id, {}).get("target_snr_db", 0.0)
            fields[f"{ue_id}_target_snr_db"] = round(current - 4.0, 2)
            fields[f"{ue_id}_reasoning"] = (
                "(flawed) lowering target SNR despite a sub-intent asking to raise it")
        return PCDecision(**fields)

    def flawed_ulra(context, state=None, critique=None) -> ULRADecision:
        if critique is not None:
            return good_ulra(context, state)
        return ULRADecision(
            fwa_throttle_mbps=0.0, fwa_priority="high",
            fwa_reasoning="(flawed) throttle 0 = 'unlimited'; everyone high",
            mtc_throttle_mbps=0.0, mtc_priority="high",
            mtc_reasoning="(flawed) throttle 0 = 'unlimited'; everyone high",
            be_throttle_mbps=0.0, be_priority="high",
            be_reasoning="(flawed) throttle 0 = 'unlimited'; everyone high",
        )

    l2_fn, pc_fn, ulra_fn = (
        (flawed_l2, flawed_pc, flawed_ulra) if profile == "flawed"
        else (good_l2, good_pc, good_ulra))

    def rrm_fn(context, state=None, critique=None) -> RRMDecision:
        return RRMDecision(**pc_fn(context, state, critique).model_dump(),
                           **ulra_fn(context, state, critique).model_dump())

    return {
        "tenant": tenant_fn, "mediator": mediator_fn,
        "l2": l2_fn, "pc": pc_fn, "ulra": ulra_fn, "rrm": rrm_fn,
    }
