"""Chart builder: turns stored results into Plotly figure JSON (rendered by plotly.js in the browser).

Figures are plain dicts in Plotly's JSON schema, so the backend needs no plotting library and charts are
fully interactive in the UI. Design rules (see README): one y-axis only, fixed categorical colour order
(colour follows the entity, never its rank), at most 8 series (the rest fold into "Other"), labelled
axes with units, hover on every mark, event/outlier annotations on time axes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .analytics import ToolError, get_result
from .util import to_jsonable

# Reference categorical palette (validated order; light-mode steps). Dark mode is applied in the UI.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ACTUAL = "#52514e"
BASELINE = SERIES[0]
SCENARIO = SERIES[1]
MAX_SERIES = 8
CHART_TYPES = {"line", "bar", "stacked_bar", "grouped_bar", "area", "scatter", "horizontal_bar", "heatmap"}


def _store(session, fig: dict, meta: dict) -> str:
    charts = session.cache.setdefault("charts", {})
    cid = f"chart_{len(charts) + 1}"
    charts[cid] = {"figure": to_jsonable(fig), "meta": meta}
    return cid


def get_chart(session, cid: str) -> dict:
    ch = session.cache.get("charts", {}).get(cid)
    if ch is None:
        raise ToolError(f"Unknown chart_id '{cid}'")
    return ch


def base_layout(title: str, x_title: str | None, y_title: str | None, legend: bool = True) -> dict:
    return {
        "title": {"text": title, "x": 0, "xanchor": "left", "font": {"size": 15}},
        "xaxis": {"title": {"text": x_title or ""}, "showgrid": False, "zeroline": False},
        "yaxis": {"title": {"text": y_title or ""}, "gridcolor": "rgba(128,128,128,0.18)", "zeroline": False, "rangemode": "tozero"},
        "hovermode": "x unified",
        "legend": {"orientation": "h", "y": -0.18} if legend else {},
        "showlegend": legend,
        "margin": {"l": 60, "r": 20, "t": 50, "b": 60},
        "font": {"family": "Inter, system-ui, sans-serif", "size": 12},
        "shapes": [], "annotations": [],
    }


def _event_annotations(session, x_min, x_max, max_events: int = 6) -> tuple[list, list, list[str]]:
    from .semantic import get_semantics
    sem = get_semantics(session)
    ev = sem.get("events")
    if not ev:
        return [], [], []
    e = session.df(ev)
    if "start_date" not in e.columns:
        return [], [], []
    e = e[(e["start_date"] >= x_min) & (e["start_date"] <= x_max)].sort_values("start_date").head(max_events)
    shapes, anns, used = [], [], []
    for i, (_, r) in enumerate(e.iterrows()):
        x = r["start_date"]
        shapes.append({"type": "line", "x0": x, "x1": x, "yref": "paper", "y0": 0, "y1": 1,
                       "line": {"color": "rgba(120,120,120,0.55)", "width": 1, "dash": "dot"}})
        anns.append({"x": x, "yref": "paper", "y": 1.0 - (i % 3) * 0.07, "text": str(r.get("event_id", "")),
                     "hovertext": f"{r.get('event_id', '')}: {r.get('event_name', '')}", "showarrow": False,
                     "font": {"size": 10, "color": "#6b6a66"}, "xanchor": "left"})
        used.append(f"{r.get('event_id', '')} {r.get('event_name', '')} ({pd.Timestamp(x):%Y-%m-%d})")
    return shapes, anns, used


def make_chart(session, result_id: str, chart_type: str, x: str, y: list[str] | str, color: str | None = None,
               title: str = "", x_title: str | None = None, y_title: str | None = None,
               annotate_events: bool = True, mark_outliers: bool = False, top_n: int | None = None) -> dict:
    if chart_type not in CHART_TYPES:
        raise ToolError(f"chart_type must be one of {sorted(CHART_TYPES)}")
    df, meta = get_result(session, result_id)
    ys = [y] if isinstance(y, str) else list(y)
    for c in [x, *ys] + ([color] if color else []):
        if c not in df.columns:
            raise ToolError(f"Column '{c}' not in result {result_id}. Columns: {list(df.columns)}")
    if len(ys) > 1 and color:
        raise ToolError("Use either several y columns or a color column, not both.")
    notes: list[str] = []
    if len(ys) > 1:
        # One-axis rule: refuse to mix measures of very different magnitude on one axis.
        mags = [np.nanmedian(np.abs(df[c].astype(float))) for c in ys]
        mags = [m for m in mags if m and not np.isnan(m)]
        if mags and max(mags) / max(min(mags), 1e-9) > 20:
            raise ToolError(f"Columns {ys} differ in scale by more than 20x; a single axis would hide the smaller one. "
                            "Make separate charts (no dual axes).")
    d = df.copy()
    is_time = pd.api.types.is_datetime64_any_dtype(d[x])
    if color:
        order = d.groupby(color)[ys[0]].sum().sort_values(ascending=False).index.tolist()
        keep = order[: (top_n or MAX_SERIES) - (1 if len(order) > (top_n or MAX_SERIES) else 0)]
        if len(order) > len(keep):
            folded = len(order) - len(keep)
            d[color] = d[color].where(d[color].isin(keep), "Other")
            d = d.groupby([x, color], as_index=False)[ys[0]].sum()
            notes.append(f"{folded} smaller series folded into 'Other' (max {MAX_SERIES} colours)")
            order = keep + ["Other"]
        # colour follows the entity: stable alphabetical assignment, not rank
        stable = sorted([o for o in order if o != "Other"], key=str)
        cmap = {k: SERIES[i % MAX_SERIES] for i, k in enumerate(stable)}
        cmap["Other"] = "#9a9993"
        groups = [(k, d[d[color] == k]) for k in order]
    else:
        groups = [(c, d) for c in ys]
        cmap = {c: SERIES[i] for i, c in enumerate(ys)}
    if not is_time and chart_type in ("bar", "horizontal_bar") and not color and top_n:
        d = d.sort_values(ys[0], ascending=False).head(top_n)
        groups = [(c, d) for c in ys]
    traces = []
    for name, g in groups:
        ycol = ys[0] if color else name
        g = g.sort_values(x) if is_time else g
        col = cmap.get(name, SERIES[0])
        label = str(name)
        if chart_type in ("line", "area"):
            tr = {"type": "scatter", "mode": "lines", "x": g[x], "y": g[ycol], "name": label,
                  "line": {"color": col, "width": 2}}
            if chart_type == "area":
                tr["stackgroup"] = "one"
                tr["line"]["width"] = 1
        elif chart_type == "scatter":
            tr = {"type": "scatter", "mode": "markers", "x": g[x], "y": g[ycol], "name": label,
                  "marker": {"color": col, "size": 8, "line": {"color": "white", "width": 1}}}
        elif chart_type == "horizontal_bar":
            tr = {"type": "bar", "orientation": "h", "y": g[x], "x": g[ycol], "name": label,
                  "marker": {"color": col}}
        elif chart_type == "heatmap":
            if not color:
                raise ToolError("heatmap needs x, y (value) and color (row category)")
            piv = d.pivot_table(index=color, columns=x, values=ys[0], aggfunc="sum")
            tr = {"type": "heatmap", "z": piv.values, "x": list(piv.columns), "y": [str(i) for i in piv.index],
                  "colorscale": [[0, "#cde2fb"], [0.5, "#3987e5"], [1, "#0d366b"]], "colorbar": {"title": ys[0]}}
            traces = [tr]
            break
        else:
            tr = {"type": "bar", "x": g[x], "y": g[ycol], "name": label, "marker": {"color": col}}
        traces.append(tr)
    lay = base_layout(title or f"{', '.join(ys)} by {x}", x_title or x.replace("_", " "), y_title or ys[0].replace("_", " "),
                      legend=len(traces) > 1)
    if chart_type == "stacked_bar":
        lay["barmode"] = "stack"
    elif chart_type in ("grouped_bar", "bar") and len(traces) > 1:
        lay["barmode"] = "group"
    if chart_type == "horizontal_bar":
        lay["xaxis"], lay["yaxis"] = lay["yaxis"], {**lay["xaxis"], "automargin": True, "autorange": "reversed"}
        lay["hovermode"] = "closest"
    if chart_type in ("scatter", "heatmap"):
        lay["hovermode"] = "closest"
    events_used: list[str] = []
    if is_time and annotate_events and len(d):
        sh, an, events_used = _event_annotations(session, d[x].min(), d[x].max())
        lay["shapes"] += sh
        lay["annotations"] += an
    outl_marked = 0
    if is_time and mark_outliers and not color and len(ys) == 1 and len(d) >= 26:
        from .timeseries import ROBUST_Z_THRESHOLD, regularise_weekly, seasonal_decompose_robust
        s = d.set_index(x)[ys[0]].astype(float)
        if (s.index.to_series().diff().dt.days.dropna() == 7).mean() > 0.9:
            s = regularise_weekly(s)
            dec = seasonal_decompose_robust(s, log=True)
            mask = (dec["robust_z"].abs() > ROBUST_Z_THRESHOLD) & ((s - dec["expected"]).abs() >= 5)
            if mask.any():
                traces.append({"type": "scatter", "mode": "markers", "x": s.index[mask], "y": s[mask],
                               "name": "Flagged outlier", "marker": {"color": "#e34948", "size": 9, "symbol": "circle-open", "line": {"width": 2}},
                               "hovertemplate": "%{x|%Y-%m-%d}: %{y:,.0f} (flagged, |robust z| > 3.5)<extra></extra>"})
                outl_marked = int(mask.sum())
                lay["showlegend"] = True
    fig = {"data": traces, "layout": lay}
    cid = _store(session, fig, {"result_id": result_id, "chart_type": chart_type, "x": x, "y": ys, "color": color})
    return to_jsonable({"chart_id": cid, "title": lay["title"]["text"], "series": [t.get("name") for t in traces],
                        "points": int(sum(len(t.get("x", [])) if t.get("x") is not None else 0 for t in traces)),
                        "events_annotated": events_used, "outliers_marked": outl_marked, "notes": notes})
