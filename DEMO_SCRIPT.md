---
output:
  pdf_document: default
  html_document: default
---

# Live demo script (5-7 minutes)

## Before class (10 minutes, same laptop)

1.  `uv run python start.py`: confirm the header says **"23 tables loaded, model claude-…"** after step 1 below, and that no yellow "AI not connected" banner shows.
2.  Do one full dry run of the prompts below. This warms the operating-system file cache and confirms the API key works.
3.  Reload the page (a fresh session) right before presenting. Keep this file open on a second screen.
4.  Turn off sleep / screen saver. Have a phone hotspot ready in case the classroom Wi-Fi blocks the Anthropic API.

## The path

| Time | Do / type exactly | What to point out |
|----|----|----|
| 0:00 | Click **Load demo data** | It runs the bundled **ZIP of Excel files** through the same upload pipeline as any user upload. The progress steps show as it works. |
| 0:40 | Upload summary appears | 23 tables with rows × columns, keys, date ranges and a quality badge each. Click **Review relationships**: 45 inferred joins, all with 100% keys matched. Note *category* and *month* are flagged "aggregate first". Click **Confirm 45 relationships**. |
| 1:10 | `Give me the data quality report and tell me what needs fixing.` | No missing weeks; 25 short-history series; stockouts are explicit (lost-sales column), so we forecast **unconstrained demand**. Outliers are flagged but never deleted, and each comes with its likely cause (Black Friday, promotions, winter storm). |
| 2:00 | `Forecast total weekly demand for the next 13 weeks. Does the model beat the baselines?` | Chart with an 80% prediction interval. All three baselines were computed and the ML model beats seasonal naive in a rolling-origin backtest. Open **How this was computed** to show the exact tool calls. Click the **wape** tag: the glossary opens with the formula and a worked example using these numbers. |
| 3:00 | `A 25% tariff is imposed on imports from India starting 10/29/2026.` | The honest answer: **no supplier is in India**, so 0 SKUs are exposed. It lists the countries we do import from instead of inventing an impact. |
| 3:30 | `A 25% tariff is imposed on imports from South Korea starting 10/29/2026.` | It parses the event into a **draft** scenario: 10 SKUs, 2 suppliers, and default assumptions (additive tariff, no price pass-through, FIFO timing, no demand effect). **Nothing is applied yet.** |
| 4:00 | In the sidebar under *Scenarios & assumptions*, click **Confirm and apply** | Four parts: normal conditions, what changes, what is affected, mitigations. Show the **Computed / Estimated / Assumed** badges. Pre-buy is **not feasible** (8-week lead time vs 9 days); shifting volume to existing US/French suppliers is. 7 SKUs have no alternate supplier, and this is flagged as not evaluable. |
| 5:15 | Click the next-question chip about price pass-through, or type `What if we pass 50% of the cost into price? Can the data estimate how demand reacts?` | The promotional elasticity is about −3, but the list-price elasticity is **not statistically distinguishable from zero**. So the app shows a hypothetical sensitivity table and asks which assumption to use, instead of guessing. |
| 6:15 | Point at the bottom of any answer | "All N numbers trace back to tool results": the automatic provenance check. Close with the architecture diagram in README.md. |

## Fallback plan

| Problem | What to do |
|----|----|
| Answer shows "Could not reach the Anthropic API" | Switch to the phone hotspot and click the last question chip again. The app keeps running, and nothing needs restarting. |
| Rate limit or overload message | Wait 10 seconds and resend. |
| A chat answer is slow (\> 60 s) | Talk through the progress steps shown on screen. Forecasts are cached, so a repeated question is instant. |
| Charts don't render | Restart with `uv run python start.py` (plotly.js is served locally from the Python package). |
| The API is unusable for the whole session | Restart in **scripted backup mode**. It runs the same tools and data with a scripted stand-in for Claude, and the header shows "model scripted-test-client". **Say so to the audience.** Windows PowerShell: `$env:DPA_LLM_STUB="e2e_stub:make"; $env:PYTHONPATH="tests"; uv run python start.py`. The demo prompts above (data quality, forecast, India, South Korea, confirm) are supported. |
| Laptop fails | Present screenshots from the dry run (take them during the dry run). |
