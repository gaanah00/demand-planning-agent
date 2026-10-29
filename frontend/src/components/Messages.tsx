import React, { useState } from "react";
import { fmt } from "../api";
import { Md, PlotlyChart, SourceBadge } from "./bits";

type Actions = {
  send: (text: string) => void;
  confirmAllRelationships: () => void;
  rejectRelationship: (id: string) => void;
  resolveWorkbook: (file: string, mode: "first" | "all") => void;
  openKpi: (id: string) => void;
  busy: boolean;
  sid?: string | null;
  themeTick?: number;
};

export function UploadSummary({ msg, state, actions }: { msg: any; state: any; actions: Actions }) {
  const s = msg.summary;
  const [showRels, setShowRels] = useState(false);
  const rels = state?.relationships || s.relationships || [];
  const pending = rels.filter((r: any) => r.status === "inferred").length;
  const q = state?.quality;
  return (
    <article className="card upload" id={msg.id}>
      <header className="card-head">
        <h3>Upload summary: {s.upload}</h3>
        <span className="muted">
          {s.tables.length} table{s.tables.length === 1 ? "" : "s"} loaded in {s.seconds}s
        </span>
      </header>
      {s.warnings?.length > 0 && (
        <ul className="warnings">
          {s.warnings.map((w: string, i: number) => (
            <li key={i}>{w}</li>
          ))}
        </ul>
      )}
      {s.tables.length > 0 && (
        <div className="table-wrap">
          <table className="grid">
            <thead>
              <tr>
                <th>Table</th>
                <th className="num">Rows × columns</th>
                <th>Role</th>
                <th>Key</th>
                <th>Date range</th>
                <th>Quality</th>
              </tr>
            </thead>
            <tbody>
              {s.tables.map((t: any) => (
                <tr key={t.table}>
                  <td>
                    <strong>{t.table}</strong>
                    <div className="muted small">{t.source_file?.split("/").pop()}</div>
                  </td>
                  <td className="num">
                    {fmt(t.rows)} × {t.columns}
                  </td>
                  <td>{t.role}</td>
                  <td className="mono-ish">{(t.primary_key || []).join(" + ") || "none found"}</td>
                  <td className="small">
                    {t.date_range ? <span className="nowrap">{t.date_range.min} to {t.date_range.max}</span> : "–"}
                    {t.date_range?.granularity && <div className="muted">{t.date_range.granularity}</div>}
                  </td>
                  <td>
                    <span className={`status status-${q?.table_status?.[t.table]?.status || "na"}`}>
                      <span className="status-dot" />
                      {q?.table_status?.[t.table]?.status === "ok" ? "OK" : q?.table_status?.[t.table]?.issues ? `${q.table_status[t.table].issues} to check` : "…"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {(s.skipped?.length > 0 || s.duplicates?.length > 0) && (
        <details className="sub">
          <summary>
            {s.skipped.length} file{s.skipped.length === 1 ? "" : "s"} skipped, {s.duplicates.length} duplicate table file
            {s.duplicates.length === 1 ? "" : "s"}
          </summary>
          <ul className="plain">
            {s.duplicates.map((d: any, i: number) => (
              <li key={"d" + i}>
                <strong>{d.table}</strong>: kept {d.kept.split("/").pop()}, skipped {d.skipped.split("/").pop()} ({d.detail})
              </li>
            ))}
            {s.skipped.map((d: any, i: number) => (
              <li key={"s" + i}>
                <strong>{d.file}</strong>: {d.reason}
              </li>
            ))}
          </ul>
        </details>
      )}
      {s.pending_workbooks?.length > 0 &&
        s.pending_workbooks.map((w: any) => (
          <div className="ask" key={w.file}>
            <p>
              <strong>{w.file}</strong> has {w.sheets.length} sheets ({w.sheets.join(", ")}). How should I load it?
            </p>
            <div className="row">
              <button onClick={() => actions.resolveWorkbook(w.file, "first")}>Use the first sheet only</button>
              <button onClick={() => actions.resolveWorkbook(w.file, "all")}>Load each sheet as a table</button>
            </div>
          </div>
        ))}
      {rels.length > 0 && (
        <div className="ask">
          <p>
            I inferred <strong>{rels.length}</strong> relationships between the tables (
            {rels.filter((r: any) => r.coverage_pct === 100).length} with every key matched).{" "}
            {pending > 0 ? "Please review and confirm them before I build analyses on them." : "All relationships are reviewed."}
          </p>
          <div className="row">
            {pending > 0 && (
              <button className="primary" onClick={actions.confirmAllRelationships}>
                Confirm {pending} relationships
              </button>
            )}
            <button onClick={() => setShowRels(!showRels)}>{showRels ? "Hide" : "Review"} relationships</button>
          </div>
          {showRels && (
            <div className="table-wrap">
              <table className="grid small">
                <thead>
                  <tr>
                    <th>From</th>
                    <th>To</th>
                    <th>Type</th>
                    <th className="num">Keys matched</th>
                    <th>Status</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {rels.map((r: any) => (
                    <tr key={r.id} className={r.status === "rejected" ? "struck" : ""}>
                      <td>
                        {r.child_table}.{r.child_columns.join("+")}
                      </td>
                      <td>
                        {r.parent_table}.{r.parent_columns.join("+")}
                      </td>
                      <td>
                        {r.kind}
                        {r.note && <div className="muted">{r.note}</div>}
                      </td>
                      <td className="num">
                        {r.coverage_pct}%{r.orphan_count ? ` (${fmt(r.orphan_count)} orphans)` : ""}
                      </td>
                      <td>{r.status}</td>
                      <td>
                        {r.status !== "rejected" && (
                          <button className="link" onClick={() => actions.rejectRelationship(r.id)}>
                            Reject
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
      {q?.timeseries && (
        <p className="muted small">
          Demand data: {fmt(q.timeseries.series)} weekly series of <strong>{q.timeseries.measure}</strong> in {q.timeseries.table},{" "}
          {q.timeseries.date_range?.min} to {q.timeseries.date_range?.max}. {q.timeseries.missing_weeks_total} missing weeks,{" "}
          {fmt(q.timeseries.zero_demand_rows)} zero-demand rows, {q.timeseries.short_series_lt52} series shorter than a year.
        </p>
      )}
      {s.tables.length > 0 && (
        <div className="chips">
          <button className="chip" onClick={() => actions.send("Give me the data quality report and tell me what needs fixing.")} disabled={actions.busy}>
            Show the data quality report
          </button>
          <button className="chip" onClick={() => actions.send("Summarise weekly demand by category over the full history with a chart.")} disabled={actions.busy}>
            Weekly demand by category
          </button>
          <button className="chip" onClick={() => actions.send("What should we reorder this week, and which items will stock out before an order can arrive?")} disabled={actions.busy}>
            What should we reorder?
          </button>
        </div>
      )}
    </article>
  );
}

export function AssistantMessage({ msg, actions, isLast }: { msg: any; actions: Actions; isLast: boolean }) {
  const f = msg.final || {};
  const prov = msg.provenance;
  return (
    <article className={"card answer" + (msg.error ? " is-error" : "")} id={msg.id}>
      <Md text={f.answer} className="lead" />
      {f.key_results?.length > 0 && (
        <div className="kpis">
          {f.key_results.map((k: any, i: number) => (
            <div className="kpi" key={i}>
              <div className="kpi-label">{k.label}</div>
              <div className="kpi-value">{k.value}</div>
              <div className="kpi-foot">
                <SourceBadge source={k.source} />
                {k.note && <span className="muted small">{k.note}</span>}
              </div>
            </div>
          ))}
        </div>
      )}
      {msg.charts?.map((c: any) => (
        <PlotlyChart key={c.id} figure={c.figure} themeTick={actions.themeTick} />
      ))}
      {(f.how_computed || msg.calls?.length > 0) && (
        <details className="sub">
          <summary>How this was computed</summary>
          <Md text={f.how_computed} />
          {msg.calls?.length > 0 && (
            <ol className="calls">
              {msg.calls.map((c: any, i: number) => (
                <li key={i} className={c.error ? "err" : ""}>
                  <div>
                    <code>{c.tool}</code> <span className="muted small">{c.seconds}s{c.error ? ", returned an error" : ""}</span>
                  </div>
                  <pre>{JSON.stringify(c.args, null, 1)}</pre>
                  <details>
                    <summary className="small">Result</summary>
                    <pre>{c.result_preview}</pre>
                  </details>
                </li>
              ))}
            </ol>
          )}
        </details>
      )}
      {f.data_notes?.length > 0 && (
        <section className="block">
          <h4>Data notes</h4>
          <ul>
            {f.data_notes.map((n: string, i: number) => (
              <li key={i}>{n}</li>
            ))}
          </ul>
        </section>
      )}
      {f.assumptions?.length > 0 && (
        <section className="block">
          <h4>Assumptions</h4>
          <ul>
            {f.assumptions.map((n: string, i: number) => (
              <li key={i}>{n}</li>
            ))}
          </ul>
        </section>
      )}
      {f.recommendations?.length > 0 && (
        <section className="block">
          <h4>Recommendations</h4>
          <ul>
            {f.recommendations.map((r: any, i: number) => (
              <li key={i}>
                {r.text}
                {r.based_on && <span className="muted small"> (based on {r.based_on})</span>}
              </li>
            ))}
          </ul>
        </section>
      )}
      <footer className="answer-foot">
        {msg.kpis?.length > 0 && (
          <div className="tags">
            {msg.kpis.map((k: string) => (
              <button key={k} className="tag" onClick={() => actions.openKpi(k)} title="Open in the KPI glossary">
                {k.replace(/_/g, " ")}
              </button>
            ))}
          </div>
        )}
        {prov && (
          <span className={"prov " + (prov.unverified?.length ? "prov-warn" : "prov-ok")}>
            {prov.unverified?.length
              ? `${prov.unverified.length} number(s) not found in any tool result: ${prov.unverified.slice(0, 6).join(", ")}`
              : `All ${prov.numbers_checked} numbers trace back to tool results or your inputs`}
          </span>
        )}
        {msg.seconds && <span className="muted small">{msg.seconds}s</span>}
        {actions.sid && !msg.error && (
          <a className="download small" href={`/api/session/${actions.sid}/export/answer/${msg.id}.xlsx`} download>
            Download data (Excel)
          </a>
        )}
      </footer>
      {f.next_questions?.length > 0 && (
        <div className="chips">
          {f.next_questions.map((q: string, i: number) => (
            <button key={i} className="chip" onClick={() => actions.send(q)} disabled={actions.busy || !isLast}>
              {q}
            </button>
          ))}
        </div>
      )}
    </article>
  );
}

export function Pending({ job, text, onStop }: { job: any; text: string; onStop?: () => void }) {
  const steps = job?.progress || [];
  return (
    <article className="card pending" aria-live="polite">
      <div className="pending-head">
        <span className="spinner" /> {text}
        {onStop && (
          <button className="small stop" onClick={onStop} disabled={!!job?.cancel}>
            {job?.cancel ? "Stopping…" : "Stop"}
          </button>
        )}
      </div>
      {steps.length > 0 && (
        <ol className="steps">
          {steps.slice(-6).map((s: any, i: number) => (
            <li key={i}>
              <span className="muted small">{s.t}s</span> {s.msg}
            </li>
          ))}
        </ol>
      )}
    </article>
  );
}
