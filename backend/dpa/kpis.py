"""KPI & formula glossary. Only KPIs actually used in the session's answers appear in the sidebar; each
usage records inputs, period/level, a worked example with this session's numbers and the messages that used it."""
from __future__ import annotations

from .session import now_iso

KPI_DEFS: dict[str, dict] = {
    "wape": {"name": "WAPE (weighted absolute percentage error)",
             "meaning": "Total forecast error as a share of total actual demand.",
             "formula": r"\mathrm{WAPE} = \frac{\sum_t |F_t - A_t|}{\sum_t A_t} \times 100\%",
             "good": "Lower is better. Compare against the baselines on the same data; at SKU level 20-40% is common for weekly demand.",
             "limitations": ["Dominated by high-volume items and weeks", "Says nothing about direction of error (see bias)"]},
    "mae": {"name": "MAE (mean absolute error)", "meaning": "Average size of the forecast miss, in units.",
            "formula": r"\mathrm{MAE} = \frac{1}{n}\sum_t |F_t - A_t|",
            "good": "Lower is better; in the same units as demand.", "limitations": ["Not comparable across series of different size"]},
    "rmse": {"name": "RMSE (root mean squared error)", "meaning": "Typical miss size, penalising big misses more.",
             "formula": r"\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_t (F_t - A_t)^2}",
             "good": "Lower is better; RMSE much larger than MAE means occasional large misses.", "limitations": ["Sensitive to outliers", "Scale dependent"]},
    "bias": {"name": "Forecast bias", "meaning": "Whether the forecast is systematically too high (+) or too low (-).",
             "formula": r"\mathrm{Bias} = \frac{\sum_t (F_t - A_t)}{\sum_t A_t} \times 100\%",
             "good": "Close to 0%. Persistent positive bias builds excess stock; negative bias causes stockouts.",
             "limitations": ["Over- and under-forecasts cancel out, so a 0% bias can hide large errors"]},
    "mase": {"name": "MASE (mean absolute scaled error)", "meaning": "Forecast error relative to a simple seasonal-naive forecast's in-sample error.",
             "formula": r"\mathrm{MASE} = \frac{\mathrm{MAE}}{\frac{1}{n-m}\sum_{t=m+1}^{n}|A_t - A_{t-m}|},\; m = 52",
             "good": "Below 1 means better than the naive benchmark; above 1 means worse.",
             "limitations": ["Needs at least one season of history for the 52-week scale (otherwise 1-week naive is used)"]},
    "mape": {"name": "MAPE (mean absolute percentage error)", "meaning": "Average percentage miss per week.",
             "formula": r"\mathrm{MAPE} = \frac{1}{n}\sum_{t: A_t \neq 0} \left|\frac{F_t - A_t}{A_t}\right| \times 100\%",
             "good": "Lower is better.", "limitations": ["Undefined when actual demand is zero (those weeks are excluded)",
                                                      "Explodes for small-volume weeks; prefer WAPE"]},
    "prediction_interval": {"name": "Prediction interval", "meaning": "Range the actual value is expected to fall in with the stated probability.",
                            "formula": r"[\hat F_t (1 + q_{\alpha/2}),\; \hat F_t (1 + q_{1-\alpha/2})],\; q = \text{quantiles of backtest } A/F - 1",
                            "good": "About 80% of actuals should fall inside an 80% interval.",
                            "limitations": ["Based on past errors; new kinds of disruption can fall outside", "Summing weekly bounds overstates the range of a total"]},
    "gross_margin": {"name": "Gross margin", "meaning": "Revenue left after paying the landed cost of the goods sold.",
                     "formula": r"\mathrm{GM} = \sum (\text{units} \times \text{price}) - \sum (\text{units} \times \text{landed cost})",
                     "good": "Higher is better; compare scenario vs baseline over the same weeks.",
                     "limitations": ["Excludes marketing, warehousing and overheads", "Uses recent average selling price"]},
    "gross_margin_pct": {"name": "Gross margin %", "meaning": "Gross margin as a share of revenue.",
                         "formula": r"\mathrm{GM\%} = \frac{\mathrm{GM}}{\mathrm{Revenue}} \times 100\%",
                         "good": "Higher is better.", "limitations": ["Can rise while total margin dollars fall"]},
    "landed_cost": {"name": "Landed cost per unit", "meaning": "Full cost of getting one unit into the warehouse.",
                    "formula": r"\text{landed} = \text{unit cost} \times (1 + \text{tariff}\%) + \frac{\text{freight}}{\text{units}}",
                    "good": "Lower is better.", "limitations": ["Tariff assumed on supplier unit cost (customs value)", "Freight averaged over recent POs"]},
    "cogs": {"name": "COGS (landed)", "meaning": "Cost of the goods sold at landed cost.",
             "formula": r"\mathrm{COGS} = \sum \text{units} \times \text{landed cost (sourcing-share weighted)}",
             "good": "-", "limitations": ["Uses planned sourcing shares, not actual PO mix"]},
    "revenue": {"name": "Revenue", "meaning": "Sales value.", "formula": r"\text{Revenue} = \sum \text{units sold} \times \text{selling price}",
                "good": "-", "limitations": ["Net of discounts; returns not deducted"]},
    "price_elasticity": {"name": "Price elasticity of demand", "meaning": "% change in units for a 1% change in price.",
                         "formula": r"\ln(\text{units}) = \beta \ln(\text{price}) + \text{SKU FE} + \text{seasonality} + \varepsilon,\;\; \text{elasticity} = \beta",
                         "good": "Negative. Below -1 = elastic (revenue falls when price rises).",
                         "limitations": ["Promotional elasticity overstates response to permanent price increases",
                                         "Average across SKUs", "Unreliable outside historical price range"]},
    "robust_z": {"name": "Robust z-score (outliers)", "meaning": "How unusual a week is versus its expected seasonal level.",
                 "formula": r"z_t = \frac{r_t - \mathrm{median}(r)}{\mathrm{MAD}(r)/0.6745},\; r = \log(1+y) - \text{trend} - \text{seasonal}",
                 "good": "|z| > 3.5 is flagged.", "limitations": ["Needs 2 years for seasonal adjustment", "Flags real events (promotions) too - flags are not errors"]},
    "stockout_rate": {"name": "Stockout rate", "meaning": "Share of item-location-weeks that were out of stock.",
                      "formula": r"\frac{\#\{\text{rows with stockout flag}=1\}}{\#\text{rows}}", "good": "Lower is better.",
                      "limitations": ["Row-weighted, not volume-weighted"]},
    "lost_sales": {"name": "Lost sales share", "meaning": "Demand that could not be served because of stockouts.",
                   "formula": r"\frac{\sum \text{lost units}}{\sum \text{customer demand}}", "good": "Lower is better.",
                   "limitations": ["Depends on how lost demand was recorded"]},
    "cv": {"name": "Coefficient of variation", "meaning": "Volatility relative to the average.",
           "formula": r"\mathrm{CV} = \sigma / \mu", "good": "Below 0.5 is fairly stable; above 1 is volatile.", "limitations": ["Undefined when the mean is 0"]},
    "yoy_growth": {"name": "Year-over-year growth", "meaning": "Change versus the same period a year earlier.",
                   "formula": r"\frac{X_{t} - X_{t-52}}{X_{t-52}} \times 100\%", "good": "-", "limitations": ["Compare like-for-like weeks; partial years distort"]},
    "safety_stock": {"name": "Safety stock", "meaning": "Buffer stock held against demand and supply variability.",
                     "formula": r"SS = z \times \sigma_{week} \times \sqrt{L + R};\quad SS_{new} = SS \times \sqrt{\frac{L + \Delta L}{L}}", "good": "Enough to hit the chosen service level; more ties up cash.",
                     "limitations": ["Square-root rule assumes independent weekly demand"]},
    "weeks_of_cover": {"name": "Weeks of cover", "meaning": "How many weeks current stock lasts at forecast demand.",
                       "formula": r"\text{cover} = \frac{\text{on hand}}{\text{average weekly forecast}}",
                       "good": "Above lead time + review period; far above it means excess stock.",
                       "limitations": ["Ignores incoming purchase orders", "Uses average, not peak, demand"]},
    "reorder_quantity": {"name": "Recommended order quantity", "meaning": "Units to order now so stock covers demand until the next order can arrive.",
                         "formula": r"Q = \max\big(0,\; F_{L+R} + SS - (\text{on hand} + \text{on order})\big),\ \text{rounded up to case pack, at least MOQ}",
                         "good": "Zero when the position already covers the protection period.",
                         "limitations": ["Location forecast is a share of the SKU forecast", "Does not check supplier capacity"]},
    "service_level": {"name": "Cycle service level", "meaning": "Probability of not running out during a replenishment cycle.",
                      "formula": r"z = \Phi^{-1}(\text{service level})", "good": "A business choice: often 95-98% for A items, lower for C items.",
                      "limitations": ["Not the same as fill rate (share of units served)"]},
    "promo_lift": {"name": "Promotion lift", "meaning": "% change in weekly demand while a promotion runs, versus comparable weeks without it.",
                   "formula": r"\text{lift}_k = e^{\beta_k} - 1,\quad \log(1+y) = \textstyle\sum_k \beta_k\,\text{promo}_k + \gamma\,\text{post} + \text{FE} + \text{season} + \varepsilon",
                   "good": "Higher lift per point of discount is better; a negative post-promotion effect means sales were pulled forward.",
                   "limitations": ["Includes display and advertising that came with the promotion", "Average across products"]},
    "holding_cost": {"name": "Holding cost", "meaning": "Cost of keeping inventory (capital, storage, risk).",
                     "formula": r"\text{units} \times \text{unit cost} \times \text{rate} \times \frac{\text{weeks held}}{52}", "good": "-",
                     "limitations": ["Rate is an assumption unless provided"]},
    "average_selling_price": {"name": "Average selling price", "meaning": "Revenue per unit sold.",
                              "formula": r"\text{ASP} = \frac{\sum \text{revenue}}{\sum \text{units sold}}", "good": "-",
                              "limitations": ["Mix effects: changes when the product mix changes"]},
}


def record_kpi(session, kpi_id: str, inputs: str, period_level: str, example: str | None = None, message_id: str | None = None) -> None:
    if kpi_id not in KPI_DEFS:
        return
    u = session.kpi_usage.setdefault(kpi_id, {"id": kpi_id, **KPI_DEFS[kpi_id], "uses": [], "first_used": now_iso()})
    entry = {"inputs": inputs, "period_level": period_level, "example": example, "message_id": message_id, "at": now_iso()}
    # keep the latest few distinct usages
    if not any(x["inputs"] == inputs and x["period_level"] == period_level and x["message_id"] == message_id for x in u["uses"]):
        u["uses"].append(entry)
        u["uses"] = u["uses"][-8:]


def glossary(session) -> list[dict]:
    return sorted(session.kpi_usage.values(), key=lambda x: x["first_used"])
