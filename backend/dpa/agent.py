"""Agent loop: Claude (Anthropic Messages API, tool use) orchestrates the deterministic analytics tools.

Flow per user message:
  1. Build the system prompt = fixed rules + live session context (tables, relationships, demand definition,
     scenarios, assumptions, KPIs used, recent result ids).
  2. Call the model with the tool library. Execute every tool_use block, return tool_result blocks.
  3. Repeat until the model calls `final_answer` (or the iteration cap is hit, in which case the model is
     forced to answer with what it has).
  4. Check number provenance: every number in the answer must appear in a tool result, the user's own
     messages or a logged assumption; anything else is flagged in the UI as "unverified".
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from .semantic import get_semantics
from .session import now_iso
from .tools import FINAL_ANSWER, ToolContext, anthropic_tools, result_for_llm, run_tool
from .util import to_jsonable

MAX_ITERATIONS = int(os.environ.get("DPA_MAX_TOOL_ITERATIONS", "12"))
MAX_SECONDS = int(os.environ.get("DPA_MAX_SECONDS", "240"))

RULES = """You are an AI demand-planning data scientist inside a planning app for a beauty & personal-care company.
You work ONLY through the tools. The tools run deterministic Python analytics on the user's uploaded tables.

NON-NEGOTIABLE RULES
1. Never invent, estimate or recall numbers. Every number you state must be copied from a tool result in this
   conversation, from the user's own words, or from a logged assumption. If you need arithmetic (a % change, a
   difference, a ratio), call `calculate` with the input numbers - do not compute in your head.
2. Label every number in key_results with its source: "computed" (straight from data and user-stated event
   parameters, e.g. landed cost per unit), "estimated" (depends on a model: forecasts, elasticities, any horizon
   total built on forecast units), or "assumed" (user-provided or hypothetical, e.g. a demand change).
3. Do NOT make your own assumptions without checking with the user. When something essential is missing or
   ambiguous (event rate, start date, which suppliers, pass-through %, demand effect), ask ONE focused question
   (set needs_user_input=true) or propose clearly labelled default assumptions and ask the user to confirm them.
4. Scenarios / external events, in this order:
   a. Parse the event into a structured scenario with `create_scenario` (draft). Show the parsed object back to the
      user (event type, scope and what it resolved to, dates, rates, defaults) and ask for confirmation. Do not
      apply it in the same turn unless the user already explicitly confirmed these exact parameters.
   b. After the user confirms: `confirm_scenario`, then `apply_scenario` for the direct, computable effects first
      (landed cost, COGS, margin per SKU / supplier / country).
   c. Demand effects only if defensible: estimated from data (`estimate_price_elasticity`, only if the result is
      marked usable for a permanent price change) or supplied by the user. Otherwise say the data cannot estimate
      it, run `sensitivity_analysis` (e.g. demand 0/-5/-10/-20%) labelled hypothetical, and ask which to apply.
   d. Every disruption answer covers: (1) expected under normal conditions, (2) what changes (deltas + charts),
      (3) what is affected (ranked SKUs, suppliers, countries by financial exposure), (4) viable mitigations from
      `evaluate_mitigations`, including what the data cannot evaluate.
   e. If the scope matches nothing in the data (e.g. a country with no suppliers), say so plainly with the
      list of what does exist, and offer relevant alternatives.
5. Forecasts: use `forecast_weekly` (it always runs baselines, ETS and LightGBM with a rolling-origin backtest).
   Recommend a model only if the tool says it beats the baselines; otherwise recommend the baseline and say so.
   Always mention the prediction interval. Report WAPE, MAE, RMSE, bias and MASE; MAPE only with its zero caveat.
   Any additive weekly planning measure can be forecast (units, revenue, purchase quantities, returns...).
5b. Inventory: for "what should we order", stockout risk or safety stock questions use `inventory_plan`. The service level
   is an assumption: use the user's if given, otherwise say the 95% default is assumed and offer to change it. Items with
   status "stockout within lead time - expedite" cannot be saved by a normal order: say so plainly.
5c. Promotions: for promotion planning use `estimate_promo_lift`; report lifts with their intervals and flag effects marked
   not reliable. For a promotion scenario, a demand effect may be taken from a reliable lift estimate (source: estimated).
6. Joins: rely on the confirmed relationships. If a tool warns about fan-out or grain mismatch, explain it.
   If relationships are not yet confirmed, mention it and ask the user to confirm them.
7. Charts: create at least one chart for analytical answers when it helps (make_chart, forecast_weekly,
   apply_scenario create charts). Put the chart ids you want shown in final_answer.chart_ids. Label axes with units.
8. Finish EVERY turn by calling `final_answer` exactly once:
   - answer: 2-3 sentences with the key number(s).
   - key_results: small set of labelled values with source badges.
   - how_computed: the method in plain English (the tool-call log is attached automatically).
   - data_notes: gaps, breaks, outliers and how they were handled, join caveats, sample-size limits.
   - assumptions: especially for scenarios.
   - recommendations: specific and actionable, each tied (based_on) to a number above.
   - next_questions: 2-3 follow-ups generated from THIS answer's results, each naming the specific SKU, supplier,
     category or number it refers to (e.g. "Compare SUP01 vs SUP09 for the 4 SKUs where shifting saves the most").
   - kpis_used: ids of KPIs used (wape, mae, rmse, bias, mase, mape, prediction_interval, gross_margin,
     gross_margin_pct, landed_cost, cogs, revenue, price_elasticity, robust_z, stockout_rate, lost_sales, cv,
     yoy_growth, safety_stock, holding_cost, average_selling_price, weeks_of_cover, reorder_quantity,
     service_level, promo_lift).
9. Audience: business users, some without data-science background. Be clear and business-friendly, define jargon
   on first use (e.g. "WAPE - total forecast error as a % of actual demand"), and never overstate confidence.
10. If a question cannot be answered from the data, say what is missing. If a tool errors, fix the arguments and
   retry once; if it still fails, explain the failure instead of guessing.
"""


def session_context(session) -> str:
    s = session
    lines = [f"TODAY: {s.settings.get('today') or time.strftime('%Y-%m-%d')}"]
    if not s.tables:
        lines.append("NO DATA LOADED. Ask the user to upload a ZIP/CSV/Excel or click 'Load demo data'.")
        return "\n".join(lines)
    lines.append("TABLES (name | rows | role | key | date range):")
    for n in sorted(s.tables):
        t = s.tables[n]
        p = s.table_profile(n)
        dr = p.get("date_range") or {}
        lines.append(f"- {n} | {p['rows']} | {t.role} | {t.pk} | {dr.get('min', '')}..{dr.get('max', '')} {dr.get('granularity') or ''}")
        cols = ", ".join(t.df.columns[:30])
        lines.append(f"    columns: {cols}")
    n_conf = sum(r.status == "confirmed" for r in s.relationships)
    lines.append(f"RELATIONSHIPS: {len(s.relationships)} inferred, {n_conf} confirmed, "
                 f"{sum(r.status == 'rejected' for r in s.relationships)} rejected. All confirmed: {s.relationships_confirmed}")
    sem = get_semantics(s)
    dm = sem.get("demand") or {}
    if dm:
        lines.append(f"DEMAND: table={dm['table']} measure={dm['measure']} (forecast target by default; the user may ask for any other measure), "
                     f"date={dm['date_col']}, keys={dm['keys']}")
        last = s.df(dm["table"])[dm["date_col"]].max()
        lines.append(f"LAST ACTUAL WEEK: {last:%Y-%m-%d}")
    if s.scenarios:
        lines.append("SCENARIOS:")
        for n, sc in s.scenarios.items():
            sp = sc["spec"]
            lines.append(f"- '{n}' [{sc['status']}{', applied' if sc.get('summary') else ''}] {sp['event_type']} scope={ {k: v for k, v in sp['scope'].items() if v} } "
                         f"start={sp['start_date']} end={sp['end_date']} tariff={sp.get('tariff_pct')} pass_through={sp.get('price_pass_through_pct')} "
                         f"demand_effect={sp['demand_effect']}")
    if s.assumptions:
        lines.append("ASSUMPTIONS LOG:")
        for a in s.assumptions[-15:]:
            lines.append(f"- {a['id']} [{a['source']}] {a['text']}" + (f" (scenario {a['scenario']})" if a.get("scenario") else ""))
    if s.kpi_usage:
        lines.append("KPIS USED SO FAR: " + ", ".join(s.kpi_usage))
    fcs = s.cache.get("forecast_by_id", {})
    if fcs:
        lines.append("FORECASTS AVAILABLE: " + "; ".join(f"{k}: {v['summary']['measure']} {v['summary']['level']} h={v['summary']['horizon_weeks']}"
                                                         for k, v in list(fcs.items())[-6:]))
    res = s.cache.get("results", {})
    if res:
        lines.append("RECENT RESULT IDS: " + "; ".join(f"{k}: {v['meta'].get('tool')} {v['meta'].get('table', '')} by {v['meta'].get('group_by', '')} {v['meta'].get('time_bucket') or ''}"
                                                       for k, v in list(res.items())[-6:]))
    return "\n".join(lines)


# ============================================================================ provenance

NUM_RE = re.compile(r"(?<![A-Za-z0-9_])[-+]?\$?\d[\d,]*(?:\.\d+)?\s?(?:%|k|K|M|m|bn|B)?(?![A-Za-z0-9_])")


def _parse_num(tok: str) -> tuple[float, float] | None:
    """Return (value, tolerance) of a displayed number token."""
    t = tok.strip().replace("$", "").replace(",", "").replace(" ", "")
    mult = 1.0
    if t.endswith("%"):
        t = t[:-1]
    elif t[-1:] in ("k", "K"):
        mult, t = 1e3, t[:-1]
    elif t[-1:] in ("M", "m"):
        mult, t = 1e6, t[:-1]
    elif t.endswith("bn") or t.endswith("B"):
        mult, t = 1e9, t.rstrip("bnB")
    try:
        v = float(t)
    except ValueError:
        return None
    dec = len(t.split(".")[1]) if "." in t else 0
    tol = 0.5 * 10 ** (-dec) * mult
    return v * mult, tol


def _collect_numbers(obj: Any, out: list[float]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, str):
        for m in NUM_RE.finditer(obj):
            p = _parse_num(m.group())
            if p:
                out.append(p[0])
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_numbers(v, out)


def _skip_token(text: str, start: int, end: int, tok: str) -> bool:
    t = tok.strip()
    pre = text[max(0, start - 1):start]
    post = text[end:end + 1]
    if pre in ("-", "/") and re.match(r"\d", text[max(0, start - 2):start - 1] or "x"):
        return True  # part of a date like 2026-10-29
    if post in ("-", "/") and re.match(r"\d", text[end + 1:end + 2] or "x"):
        return True
    raw = t.replace(",", "").replace("$", "").rstrip("%kKMmB").lstrip("+-")
    if re.fullmatch(r"(19|20)\d\d", raw) and "%" not in t and "$" not in t:
        return True  # a year
    if re.fullmatch(r"\d", raw) and "%" not in t and "$" not in t:
        return True  # single digits (counts like "3 suppliers", list numbering)
    if re.fullmatch(r"\d{1,3}", raw) and int(raw) <= 104 and re.match(r"[- ]?weeks?\b", text[end:end + 7]):
        return True  # durations / horizons such as "13-week" or "52 weeks" (request parameters, not results)
    return False


def check_provenance(answer: dict, sources: list[Any]) -> dict:
    pool: list[float] = []
    for s in sources:
        _collect_numbers(s, pool)
    pool_arr = sorted(set(round(x, 9) for x in pool))
    texts = [answer.get("answer", "")]
    texts += [f"{k.get('value', '')}" for k in answer.get("key_results", []) or []]
    texts += [r.get("text", "") for r in answer.get("recommendations", []) or []]
    unverified, checked = [], 0
    import bisect
    for text in texts:
        for m in NUM_RE.finditer(text or ""):
            tok = m.group()
            if _skip_token(text, m.start(), m.end(), tok):
                continue
            p = _parse_num(tok)
            if not p:
                continue
            v, tol = p
            checked += 1
            cands = [v, -v, v / 100, v * 100] if "%" in tok else [v, -v]
            ok = False
            for c in cands:
                tol_c = tol / 100 if (c == v / 100) else (tol * 100 if c == v * 100 else tol)
                tol_c = max(tol_c, abs(c) * 1e-9)
                i = bisect.bisect_left(pool_arr, c - tol_c - 1e-12)
                if i < len(pool_arr) and pool_arr[i] <= c + tol_c + 1e-12:
                    ok = True
                    break
            if not ok:
                unverified.append(tok.strip())
    return {"numbers_checked": checked, "unverified": sorted(set(unverified))}


# ============================================================================ loop

@dataclass
class AgentResult:
    final: dict
    calls: list[dict]
    charts: list[str]
    kpis: list[str]
    provenance: dict
    iterations: int
    seconds: float
    error: str | None = None
    exports: list | None = None


def _blocks_to_dicts(content) -> list[dict]:
    out = []
    for b in content:
        t = getattr(b, "type", None) or (b.get("type") if isinstance(b, dict) else None)
        if t == "text":
            out.append({"type": "text", "text": getattr(b, "text", None) if not isinstance(b, dict) else b["text"]})
        elif t == "tool_use":
            g = (lambda k: b[k]) if isinstance(b, dict) else (lambda k: getattr(b, k))
            out.append({"type": "tool_use", "id": g("id"), "name": g("name"), "input": g("input")})
    return out


def run_agent(session, user_text: str, client, model: str, message_id: str,
              progress: Callable[[str], None] | None = None, max_tokens: int = 4096,
              should_stop: Callable[[], bool] | None = None) -> AgentResult:
    t0 = time.time()
    progress = progress or (lambda m: None)
    ctx = ToolContext(session=session, message_id=message_id)
    system = [{"type": "text", "text": RULES, "cache_control": {"type": "ephemeral"}},
              {"type": "text", "text": "SESSION CONTEXT\n" + session_context(session)}]
    history = list(session.llm_history[-12:])
    messages = history + [{"role": "user", "content": user_text}]
    tools = anthropic_tools()
    tool_results_seen: list[Any] = []
    final = None
    iterations = 0
    err = None
    stopped = False
    while iterations < MAX_ITERATIONS + 1:
        if should_stop and should_stop():
            stopped = True
            break
        iterations += 1
        force = iterations > MAX_ITERATIONS or (time.time() - t0) > MAX_SECONDS
        kwargs = dict(model=model, max_tokens=max_tokens, system=system, tools=tools, messages=messages)
        if force:
            kwargs["tool_choice"] = {"type": "tool", "name": "final_answer"}
            progress("Writing the answer")
        else:
            progress("Thinking" if iterations == 1 else "Reviewing results")
        resp = client.messages.create(**kwargs)
        blocks = _blocks_to_dicts(resp.content)
        messages.append({"role": "assistant", "content": blocks})
        uses = [b for b in blocks if b["type"] == "tool_use"]
        if not uses:
            # model answered in plain text: ask it to use final_answer
            txt = " ".join(b["text"] for b in blocks if b["type"] == "text").strip()
            if force:
                final = {"answer": txt or "I could not complete the analysis.", "next_questions": []}
                break
            messages.append({"role": "user", "content": "Please deliver your answer by calling the final_answer tool."})
            continue
        results = []
        for u in uses:
            if u["name"] == FINAL_ANSWER.name:
                final = u["input"]
                results.append({"type": "tool_result", "tool_use_id": u["id"], "content": "delivered"})
                continue
            progress(f"Running {u['name']}")
            res, is_err = run_tool(ctx, u["name"], u["input"])
            tool_results_seen.append(res)
            results.append({"type": "tool_result", "tool_use_id": u["id"], "content": result_for_llm(res), "is_error": is_err})
        messages.append({"role": "user", "content": results})
        if final is not None:
            break
    if final is None and stopped:
        err = "Stopped by the user."
        final = {"answer": "Stopped before finishing. The steps that already ran are listed under 'How this was computed'.",
                 "next_questions": []}
    if final is None:
        err = "The analysis did not finish within the step limit."
        final = {"answer": "I ran out of analysis steps before finishing. Try a narrower question.", "next_questions": []}
    # Provenance: tool results + user's own messages + logged assumptions + scenario specs
    sources = tool_results_seen + [user_text] + [m["content"] for m in history if m["role"] == "user" and isinstance(m["content"], str)] + \
        [a for a in session.assumptions] + [sc["spec"] for sc in session.scenarios.values()] + \
        [m.get("final") for m in session.messages[-6:] if m.get("role") == "assistant"]
    prov = check_provenance(final, sources)
    # keep only charts the model chose (fall back to all produced)
    chosen = [c for c in (final.get("chart_ids") or []) if c in ctx.charts or c in session.cache.get("charts", {})]
    charts = chosen or ctx.charts[-3:]
    kpis = list(dict.fromkeys((final.get("kpis_used") or []) + ctx.kpis))
    for k in kpis:
        if k in session.kpi_usage:
            uses_ = session.kpi_usage[k]["uses"]
            if not any(u.get("message_id") == message_id for u in uses_):
                uses_.append({"inputs": "-", "period_level": "-", "example": None, "message_id": message_id, "at": now_iso()})
    # compact history for the next turn
    summary_txt = final.get("answer", "")
    if final.get("key_results"):
        summary_txt += "\nKey results: " + "; ".join(f"{k.get('label')}: {k.get('value')} ({k.get('source')})" for k in final["key_results"])
    session.llm_history += [{"role": "user", "content": user_text}, {"role": "assistant", "content": summary_txt}]
    return AgentResult(final=to_jsonable(final), calls=ctx.calls, charts=charts, kpis=kpis, provenance=prov,
                       iterations=iterations, seconds=round(time.time() - t0, 1), error=err, exports=ctx.exports)


def make_client():
    """Anthropic client from .env; returns (client, model, error_message)."""
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5").strip()
    stub = os.environ.get("DPA_LLM_STUB")
    if stub:  # offline testing only: "module:function" returning a client with messages.create()
        import importlib
        mod, fn = stub.split(":")
        return getattr(importlib.import_module(mod), fn)(), "scripted-test-client", None
    if not key or key.startswith("sk-ant-...") or key == "your-key-here":
        return None, model, ("No Anthropic API key found. Add ANTHROPIC_API_KEY=... to the .env file in the project folder and restart "
                             "the app. Data upload, profiling and the quality report still work without it.")
    try:
        import anthropic
    except ImportError:
        return None, model, "The 'anthropic' package is not installed. Run: uv sync"
    return anthropic.Anthropic(api_key=key, timeout=90.0, max_retries=2), model, None
