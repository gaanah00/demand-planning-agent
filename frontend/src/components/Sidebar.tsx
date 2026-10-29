import React, { useEffect, useMemo, useState } from "react";
import { fmt } from "../api";
import { Formula, SourceBadge, StatusDot } from "./bits";

type Props = {
  state: any;
  open: boolean;
  onToggle: () => void;
  focusKpi: string | null;
  onEditDescription: (table: string, column: string, description: string) => void;
  onConfirmScenario: (name: string) => void;
  onConfirmAll: () => void;
};

const SECTIONS = [
  { id: "dict", label: "Data dictionary" },
  { id: "rels", label: "Table relationships" },
  { id: "kpi", label: "KPI & formula glossary" },
  { id: "scen", label: "Scenarios & assumptions" },
  { id: "dq", label: "Data quality" },
];

function match(q: string, ...xs: any[]) {
  if (!q) return true;
  const s = q.toLowerCase();
  return xs.some((x) => String(x ?? "").toLowerCase().includes(s));
}

export function Sidebar({ state, open, onToggle, focusKpi, onEditDescription, onConfirmScenario, onConfirmAll }: Props) {
  const [q, setQ] = useState("");
  const [openSec, setOpenSec] = useState<Record<string, boolean>>({ dict: false, rels: false, kpi: true, scen: true, dq: true });
  const [tableFilter, setTableFilter] = useState<string>("");
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");

  useEffect(() => {
    if (focusKpi) {
      setOpenSec((o) => ({ ...o, kpi: true }));
      setQ("");
      setTimeout(() => document.getElementById("kpi-" + focusKpi)?.scrollIntoView({ behavior: "smooth", block: "start" }), 60);
    }
  }, [focusKpi]);

  const dict = useMemo(
    () =>
      (state?.dictionary || []).filter(
        (r: any) => (!tableFilter || r.table === tableFilter) && match(q, r.table, r.column, r.original_name, r.description, r.key),
      ),
    [state, q, tableFilter],
  );
  const glossary = (state?.glossary || []).filter((g: any) => match(q, g.name, g.meaning, g.id));
  const rels = (state?.relationships || []).filter((r: any) => match(q, r.child_table, r.parent_table, r.child_columns, r.note));
  const assumptions = (state?.assumptions || []).filter((a: any) => match(q, a.text, a.source, a.scenario));
  const scen = (state?.scenarios || []).filter((s: any) => match(q, s.name, s.spec?.event_type, s.spec?.description));
  const counts: Record<string, number> = {
    dict: dict.length,
    rels: rels.length,
    kpi: glossary.length,
    scen: scen.length + assumptions.length,
    dq: state?.tables?.length || 0,
  };
  const jump = (mid: string) => document.getElementById(mid)?.scrollIntoView({ behavior: "smooth", block: "start" });

  if (!open) {
    return (
      <aside className="sidebar collapsed">
        <button className="collapse" onClick={onToggle} aria-label="Open Session Reference" title="Open Session Reference">
          ›
        </button>
      </aside>
    );
  }
  return (
    <aside className="sidebar">
      <div className="side-head">
        <h2>Session reference</h2>
        <button className="collapse" onClick={onToggle} aria-label="Collapse Session Reference" title="Collapse">
          ‹
        </button>
      </div>
      <input className="search" placeholder="Search columns, KPIs, assumptions…" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search the session reference" />
      {!state?.tables?.length && <p className="side-empty">Upload data or load the demo to fill this panel.</p>}
      {SECTIONS.map((sec) => (
        <section key={sec.id} className="side-sec">
          <button className="side-sec-head" onClick={() => setOpenSec((o) => ({ ...o, [sec.id]: !o[sec.id] }))} aria-expanded={!!openSec[sec.id]}>
            <span>{sec.label}</span>
            <span className="count">{counts[sec.id]}</span>
          </button>
          {openSec[sec.id] && sec.id === "dict" && (
            <div className="side-body">
              <select value={tableFilter} onChange={(e) => setTableFilter(e.target.value)} aria-label="Filter by table">
                <option value="">All tables</option>
                {(state?.tables || []).map((t: any) => (
                  <option key={t.name} value={t.name}>
                    {t.name}
                  </option>
                ))}
              </select>
              {dict.slice(0, 250).map((r: any) => {
                const k = r.table + "." + r.column;
                return (
                  <div className="dict-row" key={k}>
                    <div className="dict-name">
                      <strong>{r.column}</strong>
                      {r.key && <span className="key">{r.key}</span>}
                    </div>
                    <div className="dict-meta">
                      {r.table}, {r.type}
                      {r.unit ? `, ${r.unit}` : ""}, {r.missing_pct}% missing
                      {r.original_name !== r.column && <>, originally “{r.original_name}”</>}
                    </div>
                    {editing === k ? (
                      <form
                        onSubmit={(e) => {
                          e.preventDefault();
                          onEditDescription(r.table, r.column, draft);
                          setEditing(null);
                        }}
                      >
                        <input autoFocus value={draft} onChange={(e) => setDraft(e.target.value)} aria-label="Column description" />
                        <button type="submit">Save</button>
                      </form>
                    ) : (
                      <button
                        className="dict-desc"
                        title="Click to edit"
                        onClick={() => {
                          setEditing(k);
                          setDraft(r.description);
                        }}
                      >
                        {r.description} <em className={r.description_status}>{r.description_status}</em>
                      </button>
                    )}
                    <div className="dict-ex">e.g. {(r.examples || []).map(String).join(", ")}</div>
                  </div>
                );
              })}
              {dict.length > 250 && <p className="muted small">Showing 250 of {dict.length}; narrow with search.</p>}
            </div>
          )}
          {openSec[sec.id] && sec.id === "rels" && (
            <div className="side-body">
              {state?.relationships?.length > 0 && !state.relationships_confirmed && (
                <button className="primary small" onClick={onConfirmAll}>
                  Confirm all inferred relationships
                </button>
              )}
              {rels.map((r: any) => (
                <div className={"rel " + r.status} key={r.id}>
                  <div>
                    {r.child_table}.<strong>{r.child_columns.join("+")}</strong> → {r.parent_table}
                  </div>
                  <div className="muted small">
                    {r.kind}, {r.coverage_pct}% keys matched, {r.status}
                  </div>
                </div>
              ))}
              {state?.joins?.length > 0 && (
                <>
                  <h4>Joins used</h4>
                  {state.joins
                    .slice(-12)
                    .reverse()
                    .map((j: any, i: number) => (
                      <div className="rel" key={i}>
                        <div>
                          {j.left} ⟶ {j.right} on {Array.isArray(j.on) ? j.on.join(", ") : String(j.on)}
                        </div>
                        <div className={"small " + (j.warnings?.length ? "warn-text" : "muted")}>
                          rows {fmt(j.left_rows)} → {fmt(j.result_rows)} ({j.how})
                          {j.warnings?.length ? `: ${j.warnings[0].slice(0, 120)}` : ""}
                        </div>
                      </div>
                    ))}
                </>
              )}
            </div>
          )}
          {openSec[sec.id] && sec.id === "kpi" && (
            <div className="side-body">
              {glossary.length === 0 && <p className="muted small">KPIs appear here as answers use them.</p>}
              {glossary.map((g: any) => (
                <div className={"kpi-entry" + (focusKpi === g.id ? " focus" : "")} id={"kpi-" + g.id} key={g.id}>
                  <h4>{g.name}</h4>
                  <p>{g.meaning}</p>
                  <Formula tex={g.formula} />
                  {g.uses?.filter((u: any) => u.example).slice(-2).map((u: any, i: number) => (
                    <div className="worked" key={i}>
                      <div className="small">Worked example: {u.example}</div>
                    </div>
                  ))}
                  {(() => {
                    const u = [...(g.uses || [])].reverse().find((x: any) => x.inputs && x.inputs !== "-");
                    return u ? (
                      <dl className="small">
                        <dt>Inputs</dt>
                        <dd>{u.inputs}</dd>
                        <dt>Period and level</dt>
                        <dd>{u.period_level}</dd>
                      </dl>
                    ) : null;
                  })()}
                  <p className="small">
                    <strong>Good looks like:</strong> {g.good}
                  </p>
                  <ul className="small caveats">
                    {g.limitations.map((l: string, i: number) => (
                      <li key={i}>{l}</li>
                    ))}
                  </ul>
                  <div className="used-in small">
                    Used in:{" "}
                    {[...new Set((g.uses || []).map((u: any) => u.message_id).filter(Boolean))].map((m: any) => (
                      <button className="link" key={m} onClick={() => jump(m)}>
                        answer {String(m).replace("m", "#")}
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
          {openSec[sec.id] && sec.id === "scen" && (
            <div className="side-body">
              {scen.length === 0 && assumptions.length === 0 && <p className="muted small">Describe an event (for example a tariff) to create a scenario.</p>}
              {scen.map((s: any) => (
                <div className="scen" key={s.name}>
                  <div className="scen-head">
                    <strong>{s.name}</strong>
                    <span className={"pill " + s.status}>{s.applied ? "applied" : s.status}</span>
                  </div>
                  <div className="small muted">
                    {s.spec.event_type}
                    {s.spec.tariff_pct != null ? `, +${s.spec.tariff_pct}% tariff` : ""}, from {s.spec.start_date || "?"}
                    {s.spec.end_date ? ` to ${s.spec.end_date}` : ""}
                  </div>
                  {s.status === "draft" && (
                    <button className="primary small" onClick={() => onConfirmScenario(s.name)}>
                      Confirm and apply
                    </button>
                  )}
                </div>
              ))}
              {assumptions.length > 0 && <h4>Assumptions log</h4>}
              {assumptions
                .slice()
                .reverse()
                .map((a: any) => (
                  <div className="assump" key={a.id}>
                    <div>
                      <SourceBadge source={a.source === "estimated" ? "estimated" : a.source === "user" ? "user" : "default"} /> {a.text}
                    </div>
                    <div className="muted small">
                      {a.scenario ? `${a.scenario}, ` : ""}
                      {a.added_at}
                    </div>
                  </div>
                ))}
            </div>
          )}
          {openSec[sec.id] && sec.id === "dq" && (
            <div className="side-body">
              {!state?.quality && state?.tables?.length > 0 && <p className="muted small">Running checks…</p>}
              {state?.quality &&
                Object.entries(state.quality.table_status)
                  .filter(([n]) => match(q, n))
                  .map(([n, v]: any) => (
                    <div className="dq-row" key={n}>
                      <span>{n}</span>
                      <StatusDot status={v.status} />
                    </div>
                  ))}
              {state?.quality?.outliers?.flagged_weeks != null && (
                <p className="small muted">
                  Outliers: {state.quality.outliers.flagged_weeks} SKU-weeks flagged in {state.quality.outliers.series_with_flags} of{" "}
                  {state.quality.outliers.series_checked} SKUs (kept, not deleted).
                </p>
              )}
            </div>
          )}
        </section>
      ))}
    </aside>
  );
}
