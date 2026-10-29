"""Weekly forecasting with baselines, ETS, a gradient-boosting ML model, rolling-origin backtests and
empirical prediction intervals.

Method (also in the README):
* Target: any additive weekly planning measure (default: unconstrained customer demand), summed to the
  requested level (total / category / SKU / any dimension value) and re-indexed to a continuous weekly grid.
* Baselines (always): naive (last value), seasonal naive (same week last year, needs >= 52 weeks of
  history), moving average (last 8 weeks).
* Candidates: Croston/SBA for intermittent demand, ETS / Holt-Winters (additive damped trend, additive 52-week seasonality when >= 2 years of
  history; statsmodels) and LightGBM (global model across series, recursive multi-step, features: lags,
  recent level, same week last year, week-of-year, calendar events, promotion share and discount, price
  index; direct multi-horizon strategy, so no forecast is fed back as an input).
* Backtest: rolling origin inside the holdout (default: last 52 weeks, origins every 13 weeks, horizon 13).
  Each origin re-fits every model on data strictly before the origin. Metrics pooled over all origins.
* Metrics: WAPE, MAE, RMSE, bias, MASE (scaled by the in-sample seasonal-naive error), MAPE with a zero caveat.
* Selection: the candidate with the lowest WAPE is recommended only if it beats the best baseline;
  otherwise the best baseline is recommended and this is said plainly.
* Intervals: empirical, from the backtest's relative errors (pooled by horizon band), so the band reflects
  how wrong each model actually was on this data. Fallback when too few errors: sqrt(h)-scaled residual std.
"""
from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .analytics import ToolError, apply_filters, enrich, store_result
from .semantic import get_semantics
from .util import to_jsonable

BASELINES = ["naive", "seasonal_naive", "moving_average"]
CANDIDATES = ["ets", "lightgbm", "croston"]
MA_WINDOW = 8
SEASON = 52


# ============================================================================ metrics

def metrics(actual: np.ndarray, fc: np.ndarray, insample: np.ndarray | None = None) -> dict:
    a = np.asarray(actual, float)
    f = np.asarray(fc, float)
    m = ~(np.isnan(a) | np.isnan(f))
    a, f = a[m], f[m]
    if len(a) == 0:
        return {}
    err = f - a
    sa = np.abs(a).sum()
    out = {
        "wape_pct": float(np.abs(err).sum() / sa * 100) if sa else None,
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "bias_pct": float(err.sum() / a.sum() * 100) if a.sum() else None,
        "n": int(len(a)),
    }
    nz = a != 0
    out["mape_pct"] = float(np.mean(np.abs(err[nz] / a[nz])) * 100) if nz.any() else None
    out["mape_excluded_zero_weeks"] = int((~nz).sum())
    out["mase"] = None
    if insample is not None:
        ins = np.asarray(insample, float)
        lag = SEASON if len(ins) > SEASON + 4 else 1
        scale = np.nanmean(np.abs(ins[lag:] - ins[:-lag])) if len(ins) > lag else np.nan
        if scale and not np.isnan(scale):
            out["mase"] = float(np.abs(err).mean() / scale)
            out["mase_scale"] = "seasonal naive (52w)" if lag == SEASON else "naive (1w)"
    return out


def pooled_mase(errors: list[tuple[np.ndarray, np.ndarray, float]]) -> float | None:
    num, den = 0.0, 0.0
    for a, f, scale in errors:
        if scale and not np.isnan(scale):
            num += np.abs(f - a).mean() / scale
            den += 1
    return float(num / den) if den else None


# ============================================================================ series building

@dataclass
class SeriesSet:
    y: pd.DataFrame            # index=week, columns=series ids (continuous weekly grid, NaN before launch/after discontinuation)
    exog: dict                 # name -> DataFrame same shape (historical + future)
    measure: str
    level: str
    table: str
    notes: list


def _weekly_date_col(session, table: str) -> str:
    t = session.tables[table]
    cand = [c for c in (t.pk or []) if pd.api.types.is_datetime64_any_dtype(t.df[c])]
    cand += [c for c in t.df.columns if pd.api.types.is_datetime64_any_dtype(t.df[c]) and c not in cand]
    if not cand:
        raise ToolError(f"Table '{table}' has no date column to build a weekly series.")
    return cand[0]


def build_series(session, measure: str | None = None, table: str | None = None, level: str = "total",
                 values: list[str] | None = None, filters: list[dict] | None = None, agg: str = "sum",
                 date_col: str | None = None, top_n: int | None = None) -> SeriesSet:
    sem = get_semantics(session)
    dm = sem.get("demand")
    table = table or (dm["table"] if dm else None)
    if not table:
        raise ToolError("No weekly demand table identified; specify table and measure.")
    df = session.df(table)
    measure = measure or (dm["measure"] if dm and table == dm["table"] else None)
    if not measure:
        raise ToolError("Specify the measure (column) to forecast.")
    date_col = date_col or (dm["date_col"] if dm and table == dm["table"] else _weekly_date_col(session, table))
    notes: list[str] = []
    level_col = None
    if level in ("sku", "product"):
        level_col = (dm or {}).get("product_col") or "sku_id"
    elif level not in ("total",):
        level_col = level
    needed = [c for c in [measure, date_col, level_col] + [f["column"] for f in (filters or [])] if c and c not in df.columns]
    joins: list[dict] = []
    df = enrich(session, table, df, needed, joins)
    if measure not in df.columns or not pd.api.types.is_numeric_dtype(df[measure]):
        raise ToolError(f"'{measure}' is not a numeric column of {table}.")
    df = apply_filters(df, filters)
    if level_col and values:
        df = df[df[level_col].astype(str).isin([str(v) for v in values])]
        if df.empty:
            raise ToolError(f"No rows for {level_col} in {values}.")
    wk = df[date_col].dt.to_period("W-SUN").dt.start_time  # Monday-start weeks
    if not (df[date_col].dt.dayofweek == 0).all():
        notes.append(f"'{date_col}' is not weekly (Monday); values were summed into Monday-start weeks.")
    df = df.assign(_wk=wk)
    key = level_col or "_all"
    if not level_col:
        df = df.assign(_all="Total")
    func = {"sum": "sum", "mean": "mean"}[agg]
    y = df.pivot_table(index="_wk", columns=key, values=measure, aggfunc=func)
    full = pd.date_range(y.index.min(), y.index.max(), freq="7D")
    y = y.reindex(full)
    # within each series' active span, a missing week means zero activity for additive measures
    for c in y.columns:
        s = y[c]
        first, last = s.first_valid_index(), s.last_valid_index()
        if first is not None and agg == "sum":
            y.loc[first:last, c] = s.loc[first:last].fillna(0.0)
    if top_n and y.shape[1] > top_n:
        keep = y.sum().sort_values(ascending=False).index[:top_n]
        notes.append(f"limited to the top {top_n} of {y.shape[1]} series by volume")
        y = y[keep]
    y.columns = [str(c) for c in y.columns]
    exog = build_exog(session, table, df, key, date_col, list(y.columns), y.index, dm)
    return SeriesSet(y=y, exog=exog, measure=measure, level=level, table=table, notes=notes)


def build_exog(session, table, df, key, date_col, series, hist_index, dm) -> dict:
    """Weekly exogenous features per series for history + 52 future weeks (planned promos, prices, calendar)."""
    sem = get_semantics(session)
    last = hist_index.max()
    idx = pd.date_range(hist_index.min(), last + pd.Timedelta(weeks=52), freq="7D")
    ex: dict[str, pd.DataFrame] = {}
    is_dem = dm and table == dm["table"]
    if is_dem and dm.get("promo_col"):
        w = df.assign(_p=df[dm["promo_col"]].notna().astype(float))
        promo = w.pivot_table(index="_wk", columns=key, values="_p", aggfunc="mean").reindex(index=idx, columns=series)
        disc = None
        if dm.get("discount_col"):
            disc = w.pivot_table(index="_wk", columns=key, values=dm["discount_col"], aggfunc="mean").reindex(index=idx, columns=series)
        # Future: planned promotions (scope x calendar) when a promotion table with start/end weeks exists
        fut_promo, fut_disc = _future_promos(session, key, series, idx[idx > last], dm)
        if fut_promo is not None:
            promo.loc[fut_promo.index] = fut_promo.values
            if disc is not None and fut_disc is not None:
                disc.loc[fut_disc.index] = fut_disc.values
        ex["promo_share"] = promo.fillna(0.0)
        if disc is not None:
            ex["discount_pct"] = disc.fillna(0.0)
    if is_dem and dm.get("list_price_col"):
        lp = df.pivot_table(index="_wk", columns=key, values=dm["list_price_col"], aggfunc="mean").reindex(index=idx, columns=series)
        lp = lp.ffill()
        ex["price_index"] = (lp / lp.loc[:last].mean()).fillna(1.0)
    cal = sem.get("calendar_table")
    if cal:
        c = session.df(cal)
        ccol = session.tables[cal].pk[0]
        c = c.set_index(ccol).reindex(idx)
        for col in ("is_peak_season", "is_federal_holiday_week"):
            if col in c.columns:
                ex[col] = pd.DataFrame(np.repeat(c[col].fillna(0).values[:, None], len(series), 1), index=idx, columns=series)
        if "shopping_event" in c.columns:
            v = c["shopping_event"].notna().astype(float).values
            ex["shopping_event"] = pd.DataFrame(np.repeat(v[:, None], len(series), 1), index=idx, columns=series)
    return ex


def _future_promos(session, key, series, fut_idx, dm):
    sem = get_semantics(session)
    pt = sem.get("promotion_table")
    if not pt or len(fut_idx) == 0:
        return None, None
    promo = session.df(pt)
    scope_name = next((n for n, t in session.tables.items() if t.role == "bridge" and "promo_id" in t.df.columns), None)
    if scope_name is None or not {"start_week", "end_week"} <= set(promo.columns):
        return None, None
    scope = session.df(scope_name)
    prod = dm.get("product_col")
    if key not in scope.columns:
        if key == "_all":
            scope = scope.assign(_all="Total")
        else:
            # level is an attribute (e.g. category): map product -> attribute
            pt2 = sem.get("product_table")
            if pt2 and key in session.df(pt2).columns and prod in scope.columns:
                scope = scope.merge(session.df(pt2)[[prod, key]], on=prod, how="left")
            else:
                return None, None
    n_rows = scope.groupby(key).size()  # crude share base: rows in scope
    rows = []
    disc_col = "effective_discount_pct" if "effective_discount_pct" in promo.columns else None
    for _, p in promo.iterrows():
        weeks = fut_idx[(fut_idx >= p["start_week"]) & (fut_idx <= p["end_week"])]
        if len(weeks) == 0:
            continue
        sc = scope[scope["promo_id"] == p["promo_id"]]
        for k, cnt in sc.groupby(key).size().items():
            for w in weeks:
                rows.append({"_wk": w, key: str(k), "cnt": cnt, "disc": float(p[disc_col]) if disc_col else 0.0})
    if not rows:
        return pd.DataFrame(0.0, index=fut_idx, columns=series), pd.DataFrame(0.0, index=fut_idx, columns=series)
    r = pd.DataFrame(rows)
    # share of the series' (sku, dc, channel) cells under promotion, approximated from scope rows
    denom = _cells_per_key(session, dm, key)
    r["share"] = r.apply(lambda x: min(1.0, x["cnt"] / max(denom.get(x[key], x["cnt"]), 1)), axis=1)
    share = r.groupby(["_wk", key])["share"].sum().clip(upper=1).unstack().reindex(index=fut_idx, columns=series).fillna(0.0)
    # Same definition as history: average discount over ALL rows of the series-week (non-promo rows count as 0),
    # i.e. sum over promotions of (share of rows covered x promotion discount).
    r["wdisc"] = r["share"] * r["disc"]
    disc = r.groupby(["_wk", key])["wdisc"].sum().unstack().reindex(index=fut_idx, columns=series).fillna(0.0)
    return share, disc


def _cells_per_key(session, dm, key) -> dict:
    df = session.df(dm["table"])
    last = df[dm["date_col"]].max()
    cur = df[df[dm["date_col"]] == last]
    if key == "_all":
        return {"Total": len(cur)}
    if key in cur.columns:
        return cur.groupby(key).size().astype(float).to_dict()
    sem = get_semantics(session)
    pt = sem.get("product_table")
    if pt and key in session.df(pt).columns:
        cur = cur.merge(session.df(pt)[[dm["product_col"], key]], on=dm["product_col"], how="left")
        return cur.groupby(key).size().astype(float).to_dict()
    return {}


# ============================================================================ models

def fc_naive(y: np.ndarray, h: int, **_) -> np.ndarray:
    return np.repeat(y[-1], h)


def fc_seasonal_naive(y: np.ndarray, h: int, **_) -> np.ndarray:
    if len(y) < SEASON:
        return np.full(h, np.nan)
    return np.array([y[len(y) - SEASON + (i % SEASON)] for i in range(h)], float)


def fc_moving_average(y: np.ndarray, h: int, **_) -> np.ndarray:
    return np.repeat(np.mean(y[-MA_WINDOW:]), h)


def _hw_numpy(y: np.ndarray, h: int, seasonal: bool) -> np.ndarray:
    """Fallback additive damped Holt-Winters fitted by SSE minimisation (used only without statsmodels)."""
    from scipy.optimize import minimize
    m = SEASON if seasonal else 1
    n = len(y)

    def run(params, out_h=0):
        a, b, g, phi = params
        level = y[:m].mean() if seasonal else y[0]
        trend = (y[m:2 * m].mean() - y[:m].mean()) / m if seasonal and n >= 2 * m else 0.0
        season = list(y[:m] - y[:m].mean()) if seasonal else [0.0]
        sse = 0.0
        start = m if seasonal else 1
        for t in range(start, n):
            s = season[t % m] if seasonal else 0.0
            pred = level + phi * trend + s
            e = y[t] - pred
            sse += e * e
            new_level = a * (y[t] - s) + (1 - a) * (level + phi * trend)
            trend = b * (new_level - level) + (1 - b) * phi * trend
            if seasonal:
                season[t % m] = g * (y[t] - new_level) + (1 - g) * s
            level = new_level
        if out_h:
            fc, damp = [], 0.0
            for k in range(1, out_h + 1):
                damp += phi ** k
                fc.append(level + damp * trend + (season[(n + k - 1) % m] if seasonal else 0.0))
            return np.array(fc)
        return sse

    x0 = [0.3, 0.05, 0.1, 0.95]
    bounds = [(0.01, 0.99), (0.0, 0.5), (0.0, 0.5), (0.8, 0.98)]
    res = minimize(run, x0, bounds=bounds, method="L-BFGS-B", options={"maxiter": 60})
    return run(res.x, out_h=h)


def fc_ets(y: np.ndarray, h: int, **_) -> np.ndarray:
    seasonal = len(y) >= 2 * SEASON + 4
    if len(y) < 10:
        return fc_moving_average(y, h)
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(y, trend="add", damped_trend=True,
                                         seasonal="add" if seasonal else None,
                                         seasonal_periods=SEASON if seasonal else None,
                                         initialization_method="estimated")
            fit = model.fit(optimized=True)
            out = np.asarray(fit.forecast(h), float)
    except ImportError:
        out = _hw_numpy(y, h, seasonal)
    except Exception:
        try:
            out = _hw_numpy(y, h, False)
        except Exception:
            out = fc_moving_average(y, h)
    if not np.all(np.isfinite(out)):
        out = fc_moving_average(y, h)
    return np.clip(out, 0, None)


def fc_croston(y: np.ndarray, h: int, alpha: float = 0.1, **_) -> np.ndarray:
    """Croston's method with the Syntetos-Boylan bias correction (SBA), for intermittent demand.

    Smooths the non-zero demand sizes (z) and the intervals between them (p) separately;
    forecast per week = (1 - alpha/2) * z / p, flat over the horizon.
    """
    y = np.asarray(y, float)
    nz = np.where(y > 0)[0]
    if len(nz) == 0:
        return np.zeros(h)
    z = y[nz[0]]
    p = float(nz[0] + 1)
    last = nz[0]
    for t in nz[1:]:
        z = z + alpha * (y[t] - z)
        p = p + alpha * ((t - last) - p)
        last = t
    return np.repeat((1 - alpha / 2) * z / max(p, 1.0), h)


def demand_class(y: np.ndarray) -> str:
    """Syntetos-Boylan classification: ADI = average interval between non-zero weeks, CV2 of non-zero sizes."""
    y = np.asarray(y, float)
    nz = y[y > 0]
    if len(nz) < 2:
        return "lumpy"
    adi = len(y) / len(nz)
    cv2 = (nz.std() / nz.mean()) ** 2 if nz.mean() else 0
    if adi < 1.32:
        return "smooth" if cv2 < 0.49 else "erratic"
    return "intermittent" if cv2 < 0.49 else "lumpy"


SIMPLE_MODELS: dict[str, Callable] = {"naive": fc_naive, "seasonal_naive": fc_seasonal_naive,
                                      "moving_average": fc_moving_average, "ets": fc_ets, "croston": fc_croston}


# ---------------------------------------------------------------------------- gradient boosting (global)

def _gbm():
    try:
        import lightgbm as lgb
        return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                                 subsample=0.9, subsample_freq=1, colsample_bytree=0.9, verbose=-1, random_state=0), "LightGBM"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, random_state=0), \
            "scikit-learn HistGradientBoosting (LightGBM not installed)"


def _roll_mean(a: np.ndarray, end: np.ndarray, w: int) -> np.ndarray:
    """nan-aware mean of a[end-w:end] for each end position (vectorised)."""
    v = np.nan_to_num(a, nan=0.0)
    c = np.concatenate([[0.0], np.cumsum(v)])
    n = np.concatenate([[0.0], np.cumsum(~np.isnan(a))])
    lo = np.clip(end - w, 0, None)
    hi = np.clip(end, 0, None)
    cnt = n[hi] - n[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (c[hi] - c[lo]) / cnt, np.nan)


def _take(a: np.ndarray, pos: np.ndarray) -> np.ndarray:
    out = np.full(len(pos), np.nan)
    ok = (pos >= 0) & (pos < len(a))
    out[ok] = a[pos[ok]]
    return out


def _direct_rows(yl: np.ndarray, origins: np.ndarray, h: int, exog_arr: dict, woy: np.ndarray, code: int):
    """Features for target week = origin + h - 1, using only information available before the origin."""
    o = origins
    tgt = o + h - 1
    lvl13 = _roll_mean(yl, o, 13)
    feats = {
        "h": np.full(len(o), h, float),
        "last": _take(yl, o - 1) - lvl13,
        "lvl4": _roll_mean(yl, o, 4) - lvl13,
        "ly": _take(yl, tgt - 52) - lvl13,
        "ly_prev": _take(yl, tgt - 53) - lvl13,
        "ly_next": np.where(tgt - 51 < o, _take(yl, tgt - 51), np.nan) - lvl13,
        "ly_level": _roll_mean(yl, o - 52, 13) - lvl13,     # last year's level at the same point -> YoY trend
        "woy_sin": np.sin(2 * np.pi * _take(woy, tgt) / 52.18),
        "woy_cos": np.cos(2 * np.pi * _take(woy, tgt) / 52.18),
        "series": np.full(len(o), code, float),
    }
    for k in sorted(exog_arr):
        feats["x_" + k] = _take(exog_arr[k], tgt)
        if k in ("promo_share", "discount_pct"):
            feats["xly_" + k] = _take(exog_arr[k], tgt - 52)   # last year's promo at the same week
    X = np.column_stack([feats[k] for k in feats])
    return X, lvl13, tgt


def fc_gbm_global(ys: pd.DataFrame, exog: dict, origin_pos: int, h: int, min_hist: int = 26) -> tuple[dict, str]:
    """Direct multi-horizon global model.

    One gradient-boosting model is trained on all series and all horizons 1..h. Each training row is
    (origin, horizon): features use only data before the origin (recent level, last value, same week
    last year, last year's level, calendar/promo/price for the target week); the target is
    log(1 + demand) minus the recent 13-week log level, so series of different sizes share one model.
    No forecast is fed back as an input, so errors do not compound across the horizon.
    """
    model, name = _gbm()
    idx = ys.index
    ex_names = sorted(exog)
    full_idx = exog[ex_names[0]].index if ex_names else pd.date_range(idx[0], idx[-1] + pd.Timedelta(weeks=52), freq="7D")
    woy = full_idx.isocalendar().week.astype(int).values.astype(float)
    N = len(full_idx)
    Xs, Ys = [], []
    preds_in = []
    cols = list(ys.columns)
    for ci, c in enumerate(cols):
        yl = np.full(N, np.nan)
        yl[: len(idx)] = np.log1p(ys[c].values.astype(float).clip(min=0))
        yl[origin_pos:] = np.nan                       # hide everything at/after the origin
        ex_arr = {k: exog[k][c].values.astype(float) for k in ex_names}
        valid = np.where(~np.isnan(yl[:origin_pos]))[0]
        if len(valid) < min_hist:
            continue
        first = valid[0]
        for hh in range(1, h + 1):
            origins = np.arange(first + 13, origin_pos - hh + 1)
            if len(origins) == 0:
                continue
            X, lvl, tgt = _direct_rows(yl, origins, hh, ex_arr, woy, ci)
            yt = _take(yl, tgt) - lvl
            ok = ~np.isnan(yt) & ~np.isnan(lvl)
            Xs.append(X[ok])
            Ys.append(yt[ok])
        if not np.isnan(yl[origin_pos - 1]):
            preds_in.append((c, ci, yl, ex_arr))
    if not Xs or sum(len(x) for x in Xs) < 100:
        return {}, name
    X = np.vstack(Xs)
    Y = np.concatenate(Ys)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, Y)
    out = {}
    for c, ci, yl, ex_arr in preds_in:
        rows, lvls = [], []
        for hh in range(1, h + 1):
            if origin_pos + hh - 1 >= N:
                break
            Xr, lvl, _ = _direct_rows(yl, np.array([origin_pos]), hh, ex_arr, woy, ci)
            rows.append(Xr[0])
            lvls.append(lvl[0])
        if not rows:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = model.predict(np.array(rows))
        out[c] = np.clip(np.expm1(np.array(lvls) + p), 0, None)
    return out, name


# ============================================================================ backtest

def _active(y: pd.Series) -> pd.Series:
    first, last = y.first_valid_index(), y.last_valid_index()
    return y.loc[first:last] if first is not None else y.iloc[0:0]


def backtest(ss: SeriesSet, holdout: int = 52, step: int = 13, horizon: int = 13,
             models: list[str] | None = None, n_jobs: int = 1) -> dict:
    models = models or BASELINES + CANDIDATES
    unknown = [m for m in models if m not in BASELINES + CANDIDATES]
    if unknown:
        raise ToolError(f"Unknown models {unknown}. Choose from {BASELINES + CANDIDATES}")
    y = ss.y
    n = len(y)
    if n < 30:
        raise ToolError(f"Only {n} weeks of history; at least 30 are needed for a backtest.")
    holdout = min(holdout, max(horizon, n // 3))
    origins = list(range(n - holdout, n - horizon + 1, step)) or [n - horizon]
    records = []  # (model, series, origin_idx, k, actual, forecast)
    gbm_name = None
    t0 = time.time()
    for o in origins:
        for c in y.columns:
            hist = _active(y[c].iloc[:o])
            if len(hist) < 8:
                continue
            act = y[c].iloc[o:o + horizon].values
            for m in models:
                if m in SIMPLE_MODELS:
                    f = SIMPLE_MODELS[m](hist.values.astype(float), len(act))
                    for k in range(len(act)):
                        records.append((m, c, o, k + 1, act[k], f[k]))
        if "lightgbm" in models:
            gf, gbm_name = fc_gbm_global(y, ss.exog, o, horizon)
            gf = {c: f for c, f in gf.items() if not np.isnan(y[c].iloc[o - 1])}
            for c, f in gf.items():
                act = y[c].iloc[o:o + horizon].values
                for k in range(min(len(act), len(f))):
                    records.append(("lightgbm", c, o, k + 1, act[k], f[k]))
    bt = pd.DataFrame(records, columns=["model", "series", "origin", "h", "actual", "forecast"])
    bt = bt.dropna(subset=["actual"])
    res = {}
    for m, g in bt.groupby("model"):
        g = g.dropna(subset=["forecast"])
        if g.empty:
            continue
        met = metrics(g["actual"].values, g["forecast"].values)
        # pooled MASE: per series/origin scale = in-sample seasonal naive MAE before that origin
        scaled = []
        for (c, o), gg in g.groupby(["series", "origin"]):
            ins = _active(y[c].iloc[:o]).values.astype(float)
            lag = SEASON if len(ins) > SEASON + 4 else 1
            sc = np.nanmean(np.abs(ins[lag:] - ins[:-lag])) if len(ins) > lag else np.nan
            scaled.append((gg["actual"].values, gg["forecast"].values, sc))
        met["mase"] = pooled_mase(scaled)
        met["mase_scale"] = "in-sample seasonal naive (52w) error, or naive (1w) for series under ~1 year"
        met["coverage_series"] = int(g["series"].nunique())
        res[m] = met
    classes = {c: demand_class(_active(y[c]).values) for c in y.columns}
    by_class: dict = {}
    bt2 = bt.dropna(subset=["forecast"]).assign(_cls=bt["series"].map(classes))
    for (m, k), g in bt2.groupby(["model", "_cls"]):
        by_class.setdefault(k, {})[m] = {"wape_pct": metrics(g["actual"].values, g["forecast"].values).get("wape_pct"),
                                          "series": int(g["series"].nunique())}
    return {"metrics": res, "by_demand_class": by_class, "demand_classes": pd.Series(classes).value_counts().to_dict(),
            "records": bt, "origins": [y.index[o].strftime("%Y-%m-%d") for o in origins],
            "holdout_weeks": holdout, "horizon": horizon, "step": step, "gbm_impl": gbm_name,
            "seconds": round(time.time() - t0, 1)}


def choose_model(metrics_by_model: dict) -> dict:
    base = {m: v for m, v in metrics_by_model.items() if m in BASELINES and v.get("wape_pct") is not None}
    cand = {m: v for m, v in metrics_by_model.items() if m in CANDIDATES and v.get("wape_pct") is not None}
    if not base:
        raise ToolError("No baseline could be evaluated.")
    best_base = min(base, key=lambda m: base[m]["wape_pct"])
    if cand:
        best_cand = min(cand, key=lambda m: cand[m]["wape_pct"])
        if cand[best_cand]["wape_pct"] < base[best_base]["wape_pct"]:
            imp = (base[best_base]["wape_pct"] - cand[best_cand]["wape_pct"]) / base[best_base]["wape_pct"] * 100
            return {"recommended": best_cand, "beats_baselines": True, "best_baseline": best_base,
                    "wape_improvement_vs_best_baseline_pct": imp,
                    "reason": f"{best_cand} has the lowest WAPE and beats the best baseline ({best_base}) by {imp:.1f}% (relative)."}
        return {"recommended": best_base, "beats_baselines": False, "best_baseline": best_base,
                "wape_improvement_vs_best_baseline_pct": 0.0,
                "reason": f"No candidate model beat the best baseline ({best_base}) on WAPE, so the baseline is recommended."}
    return {"recommended": best_base, "beats_baselines": False, "best_baseline": best_base,
            "wape_improvement_vs_best_baseline_pct": 0.0, "reason": "Only baselines were evaluated."}


def interval_quantiles(bt: pd.DataFrame, model: str, level: float = 0.8) -> dict:
    """Empirical relative-error quantiles by horizon band from backtest errors."""
    g = bt[(bt["model"] == model) & bt["forecast"].notna() & (bt["forecast"] > 0)]
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    bands = {"1-4": (1, 4), "5-8": (5, 8), "9-13": (9, 13), "14+": (14, 10_000)}
    out = {}
    for b, (a, z) in bands.items():
        gg = g[(g["h"] >= a) & (g["h"] <= z)]
        if len(gg) >= 8:
            rel = gg["actual"] / gg["forecast"] - 1
            out[b] = (float(rel.quantile(lo_q)), float(rel.quantile(hi_q)), int(len(gg)))
    if not out and len(g) >= 8:
        rel = g["actual"] / g["forecast"] - 1
        out["all"] = (float(rel.quantile(lo_q)), float(rel.quantile(hi_q)), int(len(g)))
    return out


def _band_for(h: int, q: dict):
    for b, (a, z) in {"1-4": (1, 4), "5-8": (5, 8), "9-13": (9, 13), "14+": (14, 10_000)}.items():
        if a <= h <= z and b in q:
            return q[b]
    if q:
        # widest available band for longer horizons
        return max(q.values(), key=lambda v: v[1] - v[0])
    return None


# ============================================================================ main tool

def forecast_weekly(session, measure: str | None = None, table: str | None = None, level: str = "total",
                    values: list[str] | None = None, filters: list[dict] | None = None, horizon: int = 13,
                    models: list[str] | None = None, holdout_weeks: int = 52, interval: float = 0.8,
                    outlier_treatment: str = "keep", top_n: int | None = 25, agg: str = "sum") -> dict:
    if not 1 <= horizon <= 52:
        raise ToolError("horizon must be between 1 and 52 weeks")
    if not 0.5 <= interval <= 0.99:
        raise ToolError("interval must be between 0.5 and 0.99")
    key = ("fc", measure, table, level, tuple(values or []), str(filters), horizon, tuple(models or []), holdout_weeks,
           interval, outlier_treatment, top_n, agg)
    cache = session.cache.setdefault("forecasts", {})
    if key in cache:
        return cache[key]["summary"]
    # one computation per key: concurrent callers (e.g. background warm-up + a chat turn) wait for it
    import threading
    locks = session.cache.setdefault("forecast_locks", {})
    lock = locks.setdefault(key, threading.Lock())
    with lock:
        if key in cache:
            return cache[key]["summary"]
        return _forecast_weekly_compute(session, key, cache, measure, table, level, values, filters, horizon, models,
                                        holdout_weeks, interval, outlier_treatment, top_n, agg)


def _forecast_weekly_compute(session, key, cache, measure, table, level, values, filters, horizon, models, holdout_weeks,
                             interval, outlier_treatment, top_n, agg) -> dict:
    ss = build_series(session, measure, table, level, values, filters, agg=agg, top_n=top_n)
    treat_notes = []
    if outlier_treatment != "keep":
        from .quality import treat_outliers
        for c in ss.y.columns:
            s = _active(ss.y[c])
            if len(s) >= 26:
                new, eff = treat_outliers(s, outlier_treatment)
                ss.y.loc[new.index, c] = new.values
                if eff["weeks_flagged"]:
                    treat_notes.append({"series": c, **eff})
    bt = backtest(ss, holdout=holdout_weeks, horizon=min(13, horizon), models=models)
    choice = choose_model(bt["metrics"])
    rec = choice["recommended"]
    # Final forecasts from the full history for every model
    n = len(ss.y)
    future_idx = pd.date_range(ss.y.index[-1] + pd.Timedelta(weeks=1), periods=horizon, freq="7D")
    fc = {m: {} for m in bt["metrics"]}
    for c in ss.y.columns:
        hist = _active(ss.y[c])
        if len(hist) < 8 or hist.index[-1] != ss.y.index[-1]:
            continue  # discontinued series are not forecast
        for m in fc:
            if m in SIMPLE_MODELS:
                fc[m][c] = SIMPLE_MODELS[m](hist.values.astype(float), horizon)
    if "lightgbm" in fc:
        g, _ = fc_gbm_global(ss.y, ss.exog, n, horizon)
        fc["lightgbm"] = g
    q = interval_quantiles(bt["records"], rec, interval)
    rows = []
    for c, f in fc.get(rec, {}).items():
        resid_sd = float(np.nanstd(np.diff(_active(ss.y[c]).values.astype(float))))
        for k, d in enumerate(future_idx[: len(f)]):
            band = _band_for(k + 1, q)
            if band:
                lo, hi = f[k] * (1 + band[0]), f[k] * (1 + band[1])
                method = "empirical backtest errors"
            else:
                z = 1.2816 if interval == 0.8 else abs(float(np.round(__import__("scipy").stats.norm.ppf((1 - interval) / 2), 4)))
                lo, hi = f[k] - z * resid_sd * math.sqrt(k + 1), f[k] + z * resid_sd * math.sqrt(k + 1)
                method = "normal approximation (too few backtest errors)"
            rows.append({"series": c, "week": d, "forecast": max(0.0, f[k]), "lower": max(0.0, lo), "upper": max(0.0, hi),
                         "model": rec, "interval_method": method})
    fc_df = pd.DataFrame(rows)
    hist_df = ss.y.reset_index().melt(id_vars="index", var_name="series", value_name="actual").rename(columns={"index": "week"}).dropna()
    all_models = []
    for m, d in fc.items():
        for c, f in d.items():
            for k, wk in enumerate(future_idx[: len(f)]):
                all_models.append({"series": c, "week": wk, "model": m, "forecast": float(max(0.0, f[k]))})
    am = pd.DataFrame(all_models)
    fid = f"fc_{len(cache) + 1}"
    # Per-series backtest WAPE for the recommended model (worst series first)
    rb = bt["records"][(bt["records"]["model"] == rec)].dropna(subset=["forecast"])
    per_series = (rb.assign(ae=(rb["forecast"] - rb["actual"]).abs())
                  .groupby("series").agg(abs_err=("ae", "sum"), actual=("actual", "sum")))
    per_series["wape_pct"] = per_series["abs_err"] / per_series["actual"].replace(0, np.nan) * 100
    per_series = per_series.sort_values("wape_pct", ascending=False)
    total_fc = fc_df.groupby("week")[["forecast", "lower", "upper"]].sum() if len(fc_df) else pd.DataFrame()
    summary = {
        "forecast_id": fid,
        "measure": ss.measure, "table": ss.table, "level": level, "values": values, "filters": filters,
        "series_count": int(ss.y.shape[1]), "series_forecast": int(fc_df["series"].nunique()) if len(fc_df) else 0,
        "history": {"first_week": ss.y.index[0], "last_week": ss.y.index[-1], "weeks": n},
        "horizon_weeks": horizon, "forecast_weeks": {"first": future_idx[0], "last": future_idx[-1]},
        "backtest": {"origins": bt["origins"], "holdout_weeks": bt["holdout_weeks"], "horizon": bt["horizon"],
                     "step": bt["step"], "metrics": bt["metrics"], "ml_implementation": bt["gbm_impl"],
                     "demand_classes": bt["demand_classes"], "wape_by_demand_class": bt["by_demand_class"],
                     "demand_class_note": "Syntetos-Boylan classes: smooth / erratic / intermittent / lumpy. Croston (SBA) is "
                                          "designed for intermittent and lumpy series; compare it there, not on the pooled total."},
        "model_choice": choice,
        "forecast_total": {"sum_forecast": float(fc_df["forecast"].sum()) if len(fc_df) else None,
                           "sum_lower": float(total_fc["lower"].sum()) if len(total_fc) else None,
                           "sum_upper": float(total_fc["upper"].sum()) if len(total_fc) else None,
                           "note": "interval sums are conservative (errors assumed perfectly correlated across weeks/series)"},
        "forecast_by_week_total": [{"week": i, **{k: float(v) for k, v in r.items()}} for i, r in total_fc.head(52).iterrows()],
        "last_52w_actual": float(ss.y.iloc[-52:].sum().sum()) if n >= 52 else None,
        "same_period_last_year_actual": float(ss.y.iloc[n - 52:n - 52 + horizon].sum().sum()) if n >= 52 else None,
        "interval": {"level": interval, "method": "empirical quantiles of backtest relative errors by horizon band",
                     "quantiles_by_band": q},
        "worst_series_backtest": [{"series": i, "wape_pct": float(r["wape_pct"]), "actual_units": float(r["actual"])}
                                  for i, r in per_series.head(5).iterrows() if not np.isnan(r["wape_pct"])],
        "notes": ss.notes + ([f"discontinued or inactive series not forecast: {int(ss.y.shape[1] - fc_df['series'].nunique())}"]
                             if len(fc_df) and fc_df["series"].nunique() < ss.y.shape[1] else []),
        "outlier_treatment": {"method": outlier_treatment, "series_changed": len(treat_notes), "details": treat_notes[:10]},
        "exogenous_features": sorted(ss.exog),
        "seconds": bt["seconds"],
    }
    # Sanity guardrail: flag when the recommended forecast diverges strongly from simple references.
    sanity = []
    rec_total = sum(float(np.nansum(v)) for v in fc.get(rec, {}).values())
    for ref in ("seasonal_naive", "ets"):
        if ref != rec and fc.get(ref):
            ref_total = sum(float(np.nansum(v)) for v in fc[ref].values())
            if ref_total > 0 and abs(rec_total / ref_total - 1) > 0.25:
                sanity.append(f"The {rec} forecast total differs from {ref} by {(rec_total / ref_total - 1) * 100:+.0f}%; "
                              f"review drivers before using it.")
    summary["sanity_warnings"] = sanity
    summary = to_jsonable(summary)
    cache[key] = {"summary": summary, "forecast": fc_df, "history": hist_df, "all_models": am, "backtest": bt["records"]}
    session.cache.setdefault("forecast_by_id", {})[fid] = cache[key]
    store_result(session, fc_df, {"tool": "forecast_weekly", "forecast_id": fid})
    return summary


def get_forecast(session, fid: str) -> dict:
    f = session.cache.get("forecast_by_id", {}).get(fid)
    if f is None:
        raise ToolError(f"Unknown forecast_id '{fid}'")
    return f


def forecast_chart(session, fid: str, series: str | None = None, history_weeks: int = 104, show_models: bool = False) -> dict:
    """Actuals + recommended forecast with prediction interval (and optionally other models), total or one series."""
    from .charts import ACTUAL, SERIES, _event_annotations, _store, base_layout
    f = get_forecast(session, fid)
    s = f["summary"]
    hist, fc, am = f["history"], f["forecast"], f["all_models"]
    if series:
        hist, fc, am = hist[hist["series"] == series], fc[fc["series"] == series], am[am["series"] == series]
        if hist.empty:
            raise ToolError(f"Series '{series}' not in forecast {fid}")
    h = hist.groupby("week")["actual"].sum().sort_index().iloc[-history_weeks:]
    f2 = fc.groupby("week")[["forecast", "lower", "upper"]].sum().sort_index()
    rec = s["model_choice"]["recommended"]
    lvl = int(s["interval"]["level"] * 100)
    traces = [
        {"type": "scatter", "mode": "lines", "x": list(h.index), "y": list(h.values), "name": "Actual",
         "line": {"color": ACTUAL, "width": 2}},
        {"type": "scatter", "mode": "lines", "x": list(f2.index) + list(f2.index[::-1]),
         "y": list(f2["upper"]) + list(f2["lower"][::-1]), "fill": "toself", "fillcolor": "rgba(42,120,214,0.15)",
         "line": {"width": 0}, "name": f"{lvl}% prediction interval", "hoverinfo": "skip"},
        {"type": "scatter", "mode": "lines", "x": list(f2.index), "y": list(f2["forecast"]),
         "name": f"Forecast ({rec})", "line": {"color": SERIES[0], "width": 2}},
    ]
    if show_models:
        for i, (m, g) in enumerate(am.groupby("model")):
            if m == rec:
                continue
            gg = g.groupby("week")["forecast"].sum().sort_index()
            traces.append({"type": "scatter", "mode": "lines", "x": list(gg.index), "y": list(gg.values), "name": m,
                           "line": {"color": SERIES[(i + 1) % len(SERIES)], "width": 1, "dash": "dot"}})
    title = f"{s['measure'].replace('_', ' ')} - {series or s['level']}: actuals and {s['horizon_weeks']}-week forecast"
    lay = base_layout(title, "Week", f"{s['measure'].replace('_', ' ')} (weekly)")
    if len(h):
        lay["shapes"].append({"type": "line", "x0": h.index[-1], "x1": h.index[-1], "yref": "paper", "y0": 0, "y1": 1,
                              "line": {"color": "rgba(0,0,0,0.35)", "width": 1}})
        lay["annotations"].append({"x": h.index[-1], "yref": "paper", "y": 1.02, "text": "forecast start", "showarrow": False,
                                   "font": {"size": 10}})
        sh, an, _ = _event_annotations(session, h.index[0], f2.index[-1] if len(f2) else h.index[-1])
        lay["shapes"] += sh
        lay["annotations"] += an
    cid = _store(session, {"data": traces, "layout": lay}, {"forecast_id": fid, "series": series})
    return {"chart_id": cid, "title": title}


def backtest_chart(session, fid: str) -> dict:
    """Bar chart of backtest WAPE by model (baselines vs candidates)."""
    from .charts import SERIES, _store, base_layout
    s = get_forecast(session, fid)["summary"]
    mets = s["backtest"]["metrics"]
    order = [m for m in BASELINES + CANDIDATES if m in mets]
    colors = ["#9a9993" if m in BASELINES else SERIES[0] for m in order]
    rec = s["model_choice"]["recommended"]
    colors = [SERIES[1] if m == rec else c for m, c in zip(order, colors)]
    tr = {"type": "bar", "x": order, "y": [mets[m]["wape_pct"] for m in order], "marker": {"color": colors},
          "text": [f"{mets[m]['wape_pct']:.1f}%" for m in order], "textposition": "outside",
          "hovertemplate": "%{x}: WAPE %{y:.1f}%<extra></extra>"}
    lay = base_layout(f"Backtest WAPE by model (lower is better; grey = baseline, orange = recommended)", "Model", "WAPE (%)", legend=False)
    lay["hovermode"] = "closest"
    cid = _store(session, {"data": [tr], "layout": lay}, {"forecast_id": fid})
    return {"chart_id": cid}
