"""Recommends "Zasady ładowania" (runtime settings) for this Mac and model.

A normal user has no way to know a good RAM limit, how many models may stay
loaded, how many parallel slots to run, which KV cache type to pick, and so
on. This module answers that in two layers:

1. :func:`recommend` — a **pure, deterministic** function: given
   :class:`MachineFacts` (real RAM, core counts, Apple Silicon or not — read
   with :func:`gather_machine_facts`, which mirrors how ``runtime/memory.py``
   already measures the machine) and optionally a chosen
   :class:`~localcode.hub.types.LocalModel`, it returns a
   :class:`TuneProposal`: one :class:`Recommendation` per tunable setting,
   each with a one-sentence Polish reason and the concrete number (RAM in GB,
   core count, model size…) it was derived from. Same inputs, same output,
   no model call — this is what makes it unit-testable and safe to run before
   any model is even loaded. Memory sizing reuses ``runtime/memory.py``'s
   estimator (the same math the RAM-budget admission checks and the store
   use) instead of a second, independent guess.
2. :func:`refine_with_model` — **optional**: asks the model that is already
   loaded (never loads or evicts anything on its own) for a second opinion,
   given the facts and the deterministic proposal as context. Whatever the
   model answers is re-validated field by field with
   :func:`~localcode.runtime.manager.validate_settings_patch`; a field that
   is missing, unparsable or out of range is silently dropped and the
   deterministic value is kept. A model can therefore only ever narrow a
   value that was already itself in range — it never writes settings
   straight through.

Both layers return the same :class:`TuneProposal` shape, so the HTTP route and
the CLI treat them identically; applying a proposal always goes through
:meth:`~localcode.runtime.manager.RuntimeManager.update_settings`, the same
validated path any other settings change uses.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import platform
import subprocess
from dataclasses import dataclass
from typing import Any, Protocol

from localcode.hub.types import LocalModel
from localcode.runtime import memory
from localcode.runtime.base import LoadConfig, SamplingParams
from localcode.runtime.manager import SettingsError, validate_settings_patch

log = logging.getLogger(__name__)

__all__ = [
    "MachineFacts",
    "Recommendation",
    "TuneProposal",
    "gather_machine_facts",
    "recommend",
    "refine_with_model",
]

GiB = memory.GiB

#: snake_case runtime setting name -> camelCase (the ten tunable knobs; matches
#: ``runtime.manager._SETTING_KEYS`` — duplicated rather than imported so this
#: module has no dependency on manager internals beyond the public validator).
_CAMEL: dict[str, str] = {
    "max_loaded_models": "maxLoadedModels",
    "max_instances_per_model": "maxInstancesPerModel",
    "default_parallel": "defaultParallel",
    "ram_limit_gb": "ramLimitGb",
    "idle_ttl_minutes": "idleTtlMinutes",
    "default_ctx": "defaultCtx",
    "flash_attn": "flashAttn",
    "kv_cache_type": "kvCacheType",
    "gpu_layers": "gpuLayers",
    "evict_policy": "evictPolicy",
}


def _pl(n: int, one: str, few: str, many: str) -> str:
    """Polish plural form for ``n`` (1 / 2-4 / 5+, with the 12-14 exception)."""
    if n == 1:
        return one
    tail = n % 10
    teen = n % 100
    if 2 <= tail <= 4 and not 12 <= teen <= 14:
        return few
    return many


# --------------------------------------------------------------------------- #
# Machine facts
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MachineFacts:
    """What the recommender knows about the machine it is tuning for."""

    total_ram_bytes: int
    is_apple_silicon: bool
    performance_cores: int
    efficiency_cores: int
    cpu_count: int          # logical cores in total (performance + efficiency)

    @property
    def total_ram_gb(self) -> float:
        return self.total_ram_bytes / GiB

    def to_dict(self) -> dict[str, Any]:
        return {
            "totalRamBytes": self.total_ram_bytes,
            "isAppleSilicon": self.is_apple_silicon,
            "performanceCores": self.performance_cores,
            "efficiencyCores": self.efficiency_cores,
            "cpuCount": self.cpu_count,
        }


def _sysctl_int(name: str) -> int | None:
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=2)
    except (OSError, subprocess.SubprocessError):
        return None
    text = (out.stdout or "").strip()
    return int(text) if text.isdigit() else None


def gather_machine_facts() -> MachineFacts:
    """Read the real machine: total RAM, Apple Silicon, performance/efficiency cores.

    This is the only impure function in the module — it is not exercised with
    fake machines in tests, :func:`recommend` is. It never raises: anything it
    cannot read falls back to a conservative guess.
    """
    total = memory.system_memory()["total"]
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    cpu_count = os.cpu_count() or 4
    perf: int | None = None
    eff: int | None = None
    if is_apple_silicon:
        perf = _sysctl_int("hw.perflevel0.physicalcpu")
        eff = _sysctl_int("hw.perflevel1.physicalcpu")
    if not perf or perf <= 0:
        perf, eff = cpu_count, 0
    elif not eff or eff < 0:
        eff = max(0, cpu_count - perf)
    return MachineFacts(total_ram_bytes=total, is_apple_silicon=is_apple_silicon,
                        performance_cores=perf, efficiency_cores=eff, cpu_count=cpu_count)


# --------------------------------------------------------------------------- #
# Recommendation / proposal
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Recommendation:
    """One proposed runtime setting, with the Polish reason a user would read."""

    key: str            # snake_case RuntimeSettings field name
    value: Any
    reason: str          # Polish, one sentence — why this value
    basis: str            # the fact (with its number) the value was derived from

    def to_dict(self) -> dict[str, Any]:
        return {"key": _CAMEL.get(self.key, self.key), "value": self.value,
                "reason": self.reason, "basis": self.basis}


@dataclass(frozen=True)
class TuneProposal:
    """A full set of recommendations, ready to show or to apply."""

    facts: MachineFacts
    model_key: str | None
    recommendations: tuple[Recommendation, ...]
    source: str = "deterministic"       # "deterministic" | "model"
    note: str | None = None              # Polish, e.g. why the model layer changed nothing

    def as_patch(self) -> dict[str, Any]:
        """snake_case patch — ``RuntimeSettings`` field names."""
        return {r.key: r.value for r in self.recommendations}

    @property
    def patch(self) -> dict[str, Any]:
        """camelCase patch — the wire spelling ``update_settings``/the HTTP API use elsewhere.

        Both spellings validate identically (``validate_settings_patch`` accepts
        either), this is just what the route and the backends apply so the
        request body matches every other runtime-settings write.
        """
        return {_CAMEL.get(r.key, r.key): r.value for r in self.recommendations}

    def to_dict(self) -> dict[str, Any]:
        return {
            "machine": self.facts.to_dict(),
            "modelKey": self.model_key,
            "source": self.source,
            "note": self.note,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "patch": self.patch,
        }


# --------------------------------------------------------------------------- #
# Deterministic recommender
# --------------------------------------------------------------------------- #

#: (upper bound GB inclusive, RAM fraction for the budget, baseline max loaded
#: models, idle TTL minutes, KV cache type, default context with no model
#: chosen yet). Tiers mirror ``memory.AUTO_BUDGET_FRACTION`` (0.70) at the
#: 20-40 GB tier and become more conservative below it / more generous above.
_RAM_TIERS: tuple[tuple[float, float, int, int, str, int], ...] = (
    (10.0, 0.55, 1, 15, "q8_0", 4096),
    (20.0, 0.60, 2, 20, "q8_0", 8192),
    (40.0, 0.70, 2, 30, "f16", 16384),
    (80.0, 0.72, 3, 30, "f16", 32768),
    (float("inf"), 0.75, 4, 45, "f16", 32768),
)


def _ram_tier(total_gb: float) -> tuple[float, float, int, int, str, int]:
    for row in _RAM_TIERS:
        if total_gb <= row[0]:
            return row
    return _RAM_TIERS[-1]  # pragma: no cover - last tier's bound is +inf


def recommend(facts: MachineFacts, model: LocalModel | None = None) -> TuneProposal:
    """Pure: the same ``facts``/``model`` always produce the same proposal.

    Covers the ten "Zasady ładowania" knobs (``RuntimeSettings``): RAM limit,
    max loaded models, instances per model, parallel slots, idle TTL, default
    context, KV cache type, flash attention, GPU layers and eviction policy.
    """
    total_gb = facts.total_ram_gb
    upper, fraction, base_loaded, idle_ttl, kv_type, ctx_no_model = _ram_tier(total_gb)

    ram_limit_gb = round(total_gb * fraction, 1)
    ram_limit_bytes = int(ram_limit_gb * GiB)
    left_pct = round((1 - fraction) * 100)
    recs: list[Recommendation] = [Recommendation(
        "ram_limit_gb", ram_limit_gb,
        f"Modelom przydziela {round(fraction * 100)}% pamięci, a systemowi i innym aplikacjom "
        f"zostawia ~{left_pct}% z {total_gb:.0f} GB RAM tego Maca.",
        f"{total_gb:.0f} GB RAM")]

    # max_instances_per_model: more than one copy only pays off with enough
    # performance cores and enough RAM tier to also feed a second copy.
    if facts.performance_cores >= 6 and upper >= 40.0:
        max_instances = 2
        instances_reason = (f"{facts.performance_cores} {_pl(facts.performance_cores, 'rdzeń wydajności', 'rdzenie wydajności', 'rdzeni wydajności')} "
                            f"i {ram_limit_gb:.0f} GB limitu RAM starczają na dwie równoległe kopie tego samego modelu.")
    else:
        max_instances = 1
        instances_reason = (f"Przy {facts.performance_cores} "
                            f"{_pl(facts.performance_cores, 'rdzeniu wydajności', 'rdzeniach wydajności', 'rdzeniach wydajności')} "
                            "bezpieczniej trzymać jedną kopię modelu naraz, zamiast dzielić rdzenie i pamięć.")
    recs.append(Recommendation("max_instances_per_model", max_instances, instances_reason,
                               f"{facts.performance_cores} rdzeni wydajności"))

    # default_parallel: request slots scale with performance cores, capped by
    # the RAM tier (each extra slot multiplies the KV cache).
    parallel = max(1, min(4, facts.performance_cores // 2))
    if upper <= 10.0:
        parallel = min(parallel, 1)
    elif upper <= 20.0:
        parallel = min(parallel, 2)
    recs.append(Recommendation(
        "default_parallel", parallel,
        f"{facts.performance_cores} {_pl(facts.performance_cores, 'rdzeń wydajności obsłuży', 'rdzenie wydajności obsłużą', 'rdzeni wydajności obsłuży')} "
        f"z zapasem {parallel} {_pl(parallel, 'równoległe żądanie', 'równoległe żądania', 'równoległych żądań')} "
        f"bez wydłużania kolejki, a każde kolejne żądanie zabiera kontekst z limitu RAM.",
        f"{facts.performance_cores} rdzeni wydajności"))

    # max_loaded_models: the RAM tier's baseline, narrowed when the chosen
    # model's own footprint (weights + a minimal KV cache) already takes a
    # large share of the budget.
    max_loaded = base_loaded
    instance_bytes: int | None = None
    if model is not None:
        probe = LoadConfig(n_ctx=memory.MIN_USEFUL_CTX, n_parallel=1, kv_cache_type=kv_type,
                           embedding=model.is_embedding)
        instance_bytes = memory.breakdown(model, probe).total
        by_size = max(1, ram_limit_bytes // max(instance_bytes, 1))
        max_loaded = max(1, min(base_loaded, by_size))
        loaded_reason = (f"Model {model.display_name} zajmuje już samymi wagami i minimalnym kontekstem "
                         f"~{memory.format_gb(instance_bytes)}, więc przy limicie {ram_limit_gb:.0f} GB RAM "
                         f"sensownie trzymać naraz maksymalnie {max_loaded} "
                         f"{_pl(max_loaded, 'model', 'modele', 'modeli')}.")
        loaded_basis = f"~{memory.format_gb(instance_bytes)} na instancję {model.display_name}"
    else:
        loaded_reason = (f"Przy {total_gb:.0f} GB RAM (limit {ram_limit_gb:.0f} GB) sensownie trzymać naraz "
                         f"maksymalnie {max_loaded} {_pl(max_loaded, 'model', 'modele', 'modeli')}, "
                         "żeby został zapas na system; wybór modelu doprecyzuje tę liczbę.")
        loaded_basis = f"{ram_limit_gb:.0f} GB limitu RAM"
    recs.append(Recommendation("max_loaded_models", max_loaded, loaded_reason, loaded_basis))

    # idle_ttl_minutes: the RAM tier's baseline (tighter machines free memory sooner).
    recs.append(Recommendation(
        "idle_ttl_minutes", idle_ttl,
        f"Przy {total_gb:.0f} GB RAM model bezczynny przez {idle_ttl} "
        f"{_pl(idle_ttl, 'minutę', 'minuty', 'minut')} zwalnia pamięć, zanim komuś innemu jej zabraknie.",
        f"{total_gb:.0f} GB RAM"))

    # default_ctx: the largest window that fits a per-instance share of the
    # RAM budget (reusing runtime/memory.py's own admission math), or a
    # tier-based default when no model was chosen yet.
    per_instance_budget = int(ram_limit_bytes / max(1, max_loaded))
    if model is not None:
        native = memory.native_context(model)
        fit = memory.max_ctx_that_fits(model, per_instance_budget, n_parallel=parallel,
                                       kv_cache_type=kv_type, flash_attn="auto",
                                       embedding=model.is_embedding)
        ctx = max(min(memory.MIN_USEFUL_CTX, native), min(fit, native)) if fit > 0 else min(memory.MIN_USEFUL_CTX, native)
        ctx_reason = (f"Największe okno kontekstu modelu {model.display_name} "
                     f"(natywnie {memory.format_tokens(native)} tokenów), które mieści się w "
                     f"~{memory.format_gb(per_instance_budget)} przydzielonych na jedną instancję "
                     f"przy {parallel} {_pl(parallel, 'równoległym żądaniu', 'równoległych żądaniach', 'równoległych żądaniach')}.")
        ctx_basis = f"~{memory.format_gb(per_instance_budget)} na instancję, model {model.display_name}"
    else:
        ctx = ctx_no_model
        ctx_reason = (f"Bez wybranego modelu to bezpieczny, umiarkowany rozmiar kontekstu dla {total_gb:.0f} GB "
                     "RAM; po wyborze modelu przelicz ponownie, żeby dobrać go dokładnie.")
        ctx_basis = f"{total_gb:.0f} GB RAM (bez wybranego modelu)"
    recs.append(Recommendation("default_ctx", ctx, ctx_reason, ctx_basis))

    # kv_cache_type: the RAM tier's baseline.
    if kv_type == "q8_0":
        kv_reason = (f"Przy {total_gb:.0f} GB RAM cache KV w Q8_0 zajmuje o połowę mniej pamięci niż f16 "
                     "przy praktycznie niezauważalnej stracie jakości — to zwykle więcej kontekstu "
                     "niż pełna precyzja na tej samej pamięci.")
    else:
        kv_reason = (f"Przy {total_gb:.0f} GB RAM starcza zapasu, by trzymać cache KV w pełnej precyzji f16 "
                     "(najlepsza jakość, bez kompromisu w pamięci).")
    recs.append(Recommendation("kv_cache_type", kv_type, kv_reason, f"{total_gb:.0f} GB RAM"))

    # flash_attn: "auto" is always at least as good (enables on Metal, cuts
    # the compute-buffer cost that grows with the context).
    recs.append(Recommendation(
        "flash_attn", "auto",
        f"Tryb auto włącza flash attention na Metalu i obniża pamięć buforów obliczeń przy oknie "
        f"{memory.format_tokens(ctx)} tokenów, bez ręcznego przełączania.",
        f"{memory.format_tokens(ctx)} tokenów kontekstu"))

    # gpu_layers: Apple Silicon shares RAM between CPU and GPU, so -1 (every
    # layer on the GPU) is essentially always correct.
    if model is not None and model.meta and model.meta.n_layers:
        gpu_reason = (f"Pamięć zunifikowana Apple Silicon ({total_gb:.0f} GB) mieści wszystkie "
                      f"{model.meta.n_layers} warstw modelu {model.display_name} na GPU (Metal) naraz "
                      "— brak powodu, by liczyć je ręcznie.")
        gpu_basis = f"{model.meta.n_layers} warstw modelu"
    elif facts.is_apple_silicon:
        gpu_reason = (f"Pamięć zunifikowana Apple Silicon ({total_gb:.0f} GB) mieści cały model na GPU "
                      "(Metal) — -1 ładuje wszystkie warstwy bez ręcznego liczenia.")
        gpu_basis = f"{total_gb:.0f} GB pamięci zunifikowanej"
    else:
        gpu_reason = (f"Nie wykryto Apple Silicon na tym {total_gb:.0f}-gigabajtowym komputerze — LocalCode "
                      "jest pomyślany dla takich Maców, ale -1 nadal próbuje załadować wszystkie warstwy "
                      "na GPU, jeśli sterownik na to pozwala.")
        gpu_basis = f"{total_gb:.0f} GB RAM, brak wykrytego Apple Silicon"
    recs.append(Recommendation("gpu_layers", -1, gpu_reason, gpu_basis))

    # evict_policy: LRU is the safe default whenever a RAM budget exists at all.
    recs.append(Recommendation(
        "evict_policy", "lru",
        f"Przy limicie {ram_limit_gb:.0f} GB RAM polityka LRU automatycznie zwalnia najdawniej używany "
        "model, gdy zabraknie miejsca na kolejny, zamiast odmawiać załadowania.",
        f"{ram_limit_gb:.0f} GB limitu RAM"))

    note = None if facts.is_apple_silicon else (
        "Nie wykryto Apple Silicon (arm64) — te reguły są kalibrowane dla takich Maców.")
    return TuneProposal(facts=facts, model_key=model.key if model is not None else None,
                        recommendations=tuple(recs), source="deterministic", note=note)


# --------------------------------------------------------------------------- #
# Model layer (optional)
# --------------------------------------------------------------------------- #

_MODEL_SYSTEM_PROMPT = (
    "You help tune local LLM runtime settings. You will be given machine facts "
    "and a rule-based proposal with reasons. Reply with a single JSON object "
    "only — no prose, no markdown fences — containing ONLY the keys you want to "
    "change from the proposal, each mapped to your new value. Keep every value "
    "sane for the stated machine; when in doubt, leave a key out entirely so "
    "the proposal's own value is kept. If the proposal already looks right, "
    "reply with {}.")


class RuntimeLike(Protocol):
    """What :func:`refine_with_model` needs from ``RuntimeManager`` (duck-typed)."""

    def loaded_keys(self) -> set[str]: ...

    def acquire(self, model_key: str, *, n_ctx: int | None = None, purpose: str = "chat") -> Any: ...


def _model_prompt(proposal: TuneProposal) -> str:
    facts = proposal.facts
    lines = [
        f"Machine: {facts.total_ram_gb:.0f} GB RAM total, "
        f"{'Apple Silicon' if facts.is_apple_silicon else 'not Apple Silicon'}, "
        f"{facts.performance_cores} performance cores, {facts.efficiency_cores} efficiency cores.",
        "Rule-based proposal (key: value — reason):",
    ]
    for rec in proposal.recommendations:
        lines.append(f"- {_CAMEL.get(rec.key, rec.key)}: {rec.value!r} — {rec.reason}")
    lines.append("Reply with a JSON object of the keys you would change, or {} to accept everything.")
    return "\n".join(lines)


async def _ask_model(engine: Any, proposal: TuneProposal, *, max_tokens: int) -> str:
    """One short, tool-free, thinking-off completion; returns the raw answer text."""
    from localcode.runtime.formatter import ChatFormatter
    from localcode.runtime.toolcalls import StreamSplitter

    formatter = ChatFormatter(engine)
    messages = [{"role": "system", "content": _MODEL_SYSTEM_PROMPT},
                {"role": "user", "content": _model_prompt(proposal)}]
    prompt = await formatter.render(messages, thinking=False)
    splitter = StreamSplitter(formatter.parse_format(), formatter.starts_in_reasoning(prompt))
    params = SamplingParams(max_tokens=max(1, int(max_tokens)), temperature=0.1,
                            stop=formatter.stop_strings())
    text = ""
    iterator = engine.complete(prompt, params)
    try:
        async for chunk in iterator:
            if chunk.text:
                text += "".join(piece for kind, piece in splitter.feed(chunk.text) if kind == "text")
            if chunk.done or len(text) > 2000:
                break
    finally:
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            await asyncio.shield(aclose())
    text += "".join(piece for kind, piece in splitter.finish() if kind == "text")
    return text


def _merge_model_answer(proposal: TuneProposal, parsed: dict[str, Any]) -> TuneProposal:
    """Apply only the fields of ``parsed`` that pass ``validate_settings_patch``."""
    changed: list[str] = []
    rejected: list[str] = []
    new_recs: list[Recommendation] = []
    for rec in proposal.recommendations:
        candidate = None
        found = False
        for name in (rec.key, _CAMEL.get(rec.key, rec.key)):
            if name in parsed:
                candidate, found = parsed[name], True
                break
        if not found:
            new_recs.append(rec)
            continue
        try:
            validated = validate_settings_patch({rec.key: candidate})
        except SettingsError:
            rejected.append(_CAMEL.get(rec.key, rec.key))
            new_recs.append(rec)
            continue
        value = validated.get(rec.key, rec.value)
        if value == rec.value:
            new_recs.append(rec)
            continue
        changed.append(_CAMEL.get(rec.key, rec.key))
        new_recs.append(dataclasses.replace(
            rec, value=value,
            reason=f"{rec.reason} Model doprecyzował tę wartość dla wybranego modelu.",
            basis=f"{rec.basis}; korekta modelu"))

    parts: list[str] = []
    if changed:
        parts.append("Model skorygował: " + ", ".join(changed) + ".")
    if rejected:
        parts.append("Odrzucono spoza dozwolonego zakresu: " + ", ".join(rejected) + " — zostały reguły.")
    if not changed and not rejected:
        parts.append("Model nie zmienił żadnej z proponowanych wartości.")
    note = " ".join(parts)
    if proposal.note:
        note = f"{proposal.note} {note}"
    return dataclasses.replace(proposal, recommendations=tuple(new_recs),
                               source="model" if changed else proposal.source, note=note)


async def refine_with_model(
    runtime: RuntimeLike,
    model_key: str,
    proposal: TuneProposal,
    *,
    max_tokens: int = 300,
    timeout: float = 20.0,
) -> TuneProposal:
    """Ask the model, if it is already loaded, for a second opinion on ``proposal``.

    Never loads, unloads or reloads anything — it only borrows a slot of an
    instance that is already ``ready``. Every failure (model not loaded, no
    slot available in time, an unparsable or out-of-range answer) falls back
    to ``proposal`` unchanged, with a Polish ``note`` explaining why; this
    function never raises for a bad model answer.
    """
    try:
        loaded = set(runtime.loaded_keys())
    except Exception:  # noqa: BLE001 - a broken status call must not break tuning
        loaded = set()
    if model_key not in loaded:
        note = "Model nie jest teraz załadowany, więc druga opinia modelu została pominięta — użyto reguł."
        if proposal.note:
            note = f"{proposal.note} {note}"
        return dataclasses.replace(proposal, note=note)

    async def _run() -> TuneProposal:
        lease = await runtime.acquire(model_key, purpose="chat")
        try:
            text = await _ask_model(lease.engine, proposal, max_tokens=max_tokens)
        finally:
            release = getattr(lease, "release", None)
            if callable(release):
                release()

        from localcode.runtime.toolcalls import loads_lenient

        try:
            parsed = loads_lenient(text)
        except ValueError:
            note = "Odpowiedź modelu nie była poprawnym JSON-em — użyto reguł."
            if proposal.note:
                note = f"{proposal.note} {note}"
            return dataclasses.replace(proposal, note=note)
        if not isinstance(parsed, dict):
            note = "Odpowiedź modelu nie była obiektem JSON — użyto reguł."
            if proposal.note:
                note = f"{proposal.note} {note}"
            return dataclasses.replace(proposal, note=note)
        if not parsed:
            note = "Model nie zaproponował żadnych zmian — użyto reguł."
            if proposal.note:
                note = f"{proposal.note} {note}"
            return dataclasses.replace(proposal, note=note)
        return _merge_model_answer(proposal, parsed)

    try:
        return await asyncio.wait_for(_run(), timeout)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - a model that fails must not break tuning
        log.info("tuner: model refinement of %s failed: %s", model_key, exc)
        note = "Nie udało się zapytać modelu o drugą opinię — użyto reguł."
        if proposal.note:
            note = f"{proposal.note} {note}"
        return dataclasses.replace(proposal, note=note)
