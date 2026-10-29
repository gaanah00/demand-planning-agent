"""Downloads: the data behind an answer as an Excel workbook, and single results as CSV."""
from __future__ import annotations

import io
import re

import pandas as pd

from .analytics import ToolError, get_result


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.tz_localize(None) if getattr(out[c].dt, "tz", None) is not None else out[c]
        elif out[c].dtype == object:
            out[c] = out[c].map(lambda v: ", ".join(map(str, v)) if isinstance(v, (list, tuple, set)) else v)
    return out


def frames_for(session, item: dict) -> list[tuple[str, pd.DataFrame]]:
    kind, iid = item["kind"], item["id"]
    if kind == "result":
        df, meta = get_result(session, iid)
        return [(f"{meta.get('tool', 'result')} {iid}", df)]
    if kind == "forecast":
        from .forecasting import get_forecast
        f = get_forecast(session, iid)
        s = f["summary"]
        mets = pd.DataFrame([{"model": m, **{k: v.get(k) for k in ("wape_pct", "mae", "rmse", "bias_pct", "mase", "mape_pct")}}
                             for m, v in s["backtest"]["metrics"].items()])
        return [(f"forecast {iid}", f["forecast"]), (f"backtest {iid}", mets)]
    if kind == "inventory_plan":
        from .inventory import get_plan
        return [(f"reorder plan {iid}", get_plan(session, iid)["plan"])]
    if kind == "scenario":
        sc = session.scenarios.get(iid)
        if not sc or not sc.get("summary"):
            return []
        summ = sc["summary"]
        out = [(f"{iid} kpis", pd.DataFrame(summ["2_what_changes"]["kpis"])),
               (f"{iid} unit costs", pd.DataFrame(summ["2_what_changes"]["per_unit_cost_changes_computed"])),
               (f"{iid} skus", pd.DataFrame(summ["3_what_is_affected"]["skus_ranked_by_margin_impact"])),
               (f"{iid} suppliers", pd.DataFrame(summ["3_what_is_affected"]["suppliers"]))]
        return out
    return []


def _sheet_name(name: str, used: set) -> str:
    base = re.sub(r"[\[\]\*\?/\\:]", " ", name)[:28].strip() or "data"
    n, i = base, 2
    while n.lower() in used:
        n = f"{base[:25]} {i}"
        i += 1
    used.add(n.lower())
    return n


def answer_workbook(session, message: dict) -> bytes:
    f = message.get("final") or {}
    buf = io.BytesIO()
    used: set = set()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        rows = [["Question", next((m.get("text") for m in session.messages if m.get("id") and m["role"] == "user"
                                   and int(m["id"][1:]) == int(message["id"][1:]) - 1), "")],
                ["Answer", f.get("answer", "")], ["How computed", f.get("how_computed", "")], [], ["Key result", "Value", "Source", "Note"]]
        rows += [[k.get("label"), k.get("value"), k.get("source"), k.get("note", "")] for k in f.get("key_results", []) or []]
        rows += [[], ["Assumptions"]] + [[a] for a in f.get("assumptions", []) or []]
        rows += [[], ["Recommendations", "Based on"]] + [[r.get("text"), r.get("based_on", "")] for r in f.get("recommendations", []) or []]
        rows += [[], ["Data notes"]] + [[n] for n in f.get("data_notes", []) or []]
        pd.DataFrame(rows).to_excel(w, sheet_name=_sheet_name("Answer", used), index=False, header=False)
        for item in message.get("exports") or []:
            try:
                for name, df in frames_for(session, item):
                    _clean(df).to_excel(w, sheet_name=_sheet_name(name, used), index=False)
            except ToolError:
                continue  # result no longer cached (e.g. after a restart); skipped
        calls = pd.DataFrame([{"tool": c["tool"], "seconds": c["seconds"], "error": c["error"], "args": str(c["args"])[:3000]}
                              for c in message.get("calls", [])])
        if len(calls):
            calls.to_excel(w, sheet_name=_sheet_name("Tool calls", used), index=False)
        for ws in w.book.worksheets:
            for col in ws.columns:
                width = max((len(str(c.value)) for c in col[:200] if c.value is not None), default=8)
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 60)
    return buf.getvalue()


def result_csv(session, kind: str, iid: str) -> bytes:
    frames = frames_for(session, {"kind": kind, "id": iid})
    if not frames:
        raise ToolError("Nothing to export for that id.")
    return _clean(frames[0][1]).to_csv(index=False).encode("utf-8-sig")
