import React, { useCallback, useEffect, useRef, useState } from "react";
import { api, Job, pollJob } from "./api";
import { AssistantMessage, Pending, UploadSummary } from "./components/Messages";
import { Sidebar } from "./components/Sidebar";
import { Workspaces } from "./components/Workspaces";

const SID_KEY = "dpa_session_id";
const THEME_KEY = "dpa_theme";

function store(key: string, v?: string | null): string | null {
  try {
    if (v === undefined) return localStorage.getItem(key);
    if (v === null) localStorage.removeItem(key);
    else localStorage.setItem(key, v);
  } catch {
    /* private mode */
  }
  return null;
}

type Theme = "light" | "dark" | "system";

function applyTheme(t: Theme) {
  const root = document.documentElement;
  if (t === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", t);
}

const STARTERS = [
  "Give me the data quality report and tell me what needs fixing.",
  "Forecast total weekly demand for the next 13 weeks. Does the model beat the baselines?",
  "What should we reorder this week, and which items will stock out before an order can arrive?",
  "Which promotion types lift demand the most, by category?",
  "A 25% tariff is imposed on imports from South Korea starting 10/29/2026.",
];

function Login({ onDone }: { onDone: () => void }) {
  const [pw, setPw] = useState("");
  const [err, setErr] = useState<string | null>(null);
  return (
    <div className="login">
      <form
        className="card login-card"
        onSubmit={async (e) => {
          e.preventDefault();
          try {
            await api.login(pw);
            onDone();
          } catch (x: any) {
            setErr(x.message);
          }
        }}
      >
        <h1>Demand planning analyst</h1>
        <p className="muted">This workspace is password protected.</p>
        <label>
          Password
          <input type="password" autoFocus value={pw} onChange={(e) => setPw(e.target.value)} />
        </label>
        {err && <div className="error">{err}</div>}
        <button className="primary" type="submit" disabled={!pw}>
          Sign in
        </button>
      </form>
    </div>
  );
}

export default function App() {
  const [sid, setSid] = useState<string | null>(null);
  const [state, setState] = useState<any>(null);
  const [pending, setPending] = useState<{ job: Job | null; text: string; kind: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [sideOpen, setSideOpen] = useState(true);
  const [focusKpi, setFocusKpi] = useState<string | null>(null);
  const [needLogin, setNeedLogin] = useState(false);
  const [theme, setTheme] = useState<Theme>(((store(THEME_KEY) as Theme) || "system"));
  const [themeTick, setThemeTick] = useState(0);
  const fileRef = useRef<HTMLInputElement>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const busy = !!pending;

  useEffect(() => {
    applyTheme(theme);
    store(THEME_KEY, theme);
    setThemeTick((x) => x + 1);
    const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
    const on = () => setThemeTick((x) => x + 1);
    mq?.addEventListener?.("change", on);
    return () => mq?.removeEventListener?.("change", on);
  }, [theme]);

  useEffect(() => {
    const h = () => setNeedLogin(true);
    window.addEventListener("dpa-auth-required", h);
    return () => window.removeEventListener("dpa-auth-required", h);
  }, []);

  const refresh = useCallback(
    async (id?: string) => {
      const s = id || sid;
      if (!s) return;
      try {
        setState(await api.state(s));
      } catch (e: any) {
        if (String(e.message).includes("not found")) {
          const n = await api.createSession();
          store(SID_KEY, n.session_id);
          setSid(n.session_id);
          setState(await api.state(n.session_id));
        } else if (!String(e.message).includes("Sign in")) setError(e.message);
      }
    },
    [sid],
  );

  const boot = useCallback(async () => {
    const a = await api.auth();
    if (a.required && !a.authenticated) {
      setNeedLogin(true);
      return;
    }
    setNeedLogin(false);
    let s = store(SID_KEY);
    if (s) {
      try {
        await api.state(s);
      } catch {
        s = null;
      }
    }
    if (!s) {
      const n = await api.createSession();
      s = n.session_id as string;
      store(SID_KEY, s);
    }
    setSid(s);
    refresh(s);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    boot().catch((e) => setError(e.message));
  }, [boot]);

  useEffect(() => {
    if (state?.warmup?.status !== "running") return;
    const t = setInterval(() => refresh(), 4000);
    return () => clearInterval(t);
  }, [state?.warmup?.status, refresh]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [state?.messages?.length, pending?.job?.progress?.length]);

  async function runJob(start: () => Promise<any>, label: string, kind = "other") {
    if (!sid || busy) return;
    setError(null);
    setPending({ job: null, text: label, kind });
    try {
      const { job_id } = await start();
      const j = await pollJob(job_id, (jj) => setPending({ job: jj, text: label, kind }));
      if (j.status === "error") setError(j.error || "Something went wrong.");
    } catch (e: any) {
      setError(e.message);
    } finally {
      setPending(null);
      refresh();
    }
  }

  const send = (text: string) => {
    const t = text.trim();
    if (!t) return;
    setInput("");
    runJob(() => api.chat(sid!, t), "Working on it", "chat");
  };

  const switchTo = (id: string) => {
    if (busy) return;
    store(SID_KEY, id);
    setSid(id);
    setState(null);
    refresh(id);
  };

  const actions = {
    send,
    busy,
    sid,
    themeTick,
    openKpi: (id: string) => {
      setSideOpen(true);
      setFocusKpi(null);
      setTimeout(() => setFocusKpi(id), 0);
    },
    confirmAllRelationships: async () => {
      await api.relationships(sid!, { confirm_all: true });
      refresh();
    },
    rejectRelationship: async (id: string) => {
      await api.relationships(sid!, { reject_ids: [id] });
      refresh();
    },
    resolveWorkbook: (file: string, mode: "first" | "all") => runJob(() => api.workbook(sid!, file, mode), "Loading the workbook"),
  };

  const onFile = (f?: File | null) => {
    if (!f) return;
    runJob(() => api.upload(sid!, f), `Uploading ${f.name}`);
    if (fileRef.current) fileRef.current.value = "";
  };

  if (needLogin) return <Login onDone={() => boot()} />;

  const messages = state?.messages || [];
  const lastAssistant = [...messages].reverse().find((m: any) => m.role === "assistant")?.id;
  const llm = state?.api;
  const hasData = !!state?.tables?.length;

  return (
    <div className={"app" + (sideOpen ? "" : " side-closed")}>
      <Sidebar
        state={state}
        open={sideOpen}
        onToggle={() => setSideOpen(!sideOpen)}
        focusKpi={focusKpi}
        onEditDescription={async (t, c, d) => {
          await api.dictionary(sid!, t, c, d);
          refresh();
        }}
        onConfirmScenario={async (name) => {
          try {
            await api.confirmScenario(sid!, name);
            send(`I confirm the scenario '${name}' as shown. Apply it and walk me through the impact.`);
          } catch (e: any) {
            setError(e.message);
          }
        }}
        onConfirmAll={actions.confirmAllRelationships}
      />
      <main
        className="main"
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          onFile(e.dataTransfer.files?.[0]);
        }}
      >
        <header className="topbar">
          <div className="brand">
            <h1>Demand planning analyst</h1>
            <Workspaces
              current={state}
              disabled={busy}
              onSwitch={switchTo}
              onRenamed={() => refresh()}
              onError={setError}
            />
          </div>
          <div className="row">
            <span className="muted small status-line">
              {hasData ? `${state.tables.length} tables` : "No data loaded"}
              {llm && (llm.llm_ready ? `, ${llm.model}` : ", AI not connected")}
            </span>
            <select className="theme" value={theme} onChange={(e) => setTheme(e.target.value as Theme)} aria-label="Colour theme">
              <option value="system">System theme</option>
              <option value="light">Light</option>
              <option value="dark">Dark</option>
            </select>
            <button onClick={() => window.print()} disabled={!messages.length} title="Print or save the conversation as PDF">
              Save as PDF
            </button>
            <input ref={fileRef} type="file" accept=".zip,.csv,.xlsx,.xls,.xlsm" hidden onChange={(e) => onFile(e.target.files?.[0])} />
            <button onClick={() => fileRef.current?.click()} disabled={busy}>
              Upload files
            </button>
            <button className="primary" onClick={() => runJob(() => api.demo(sid!), "Loading demo data through the zip pipeline")} disabled={busy}>
              Load demo data
            </button>
          </div>
        </header>
        {llm && !llm.llm_ready && <div className="banner">{llm.message}</div>}
        {state?.save_error && <div className="banner">This workspace could not be saved to disk: {state.save_error}</div>}
        <section className="thread" aria-live="polite">
          <div className="print-only print-head">
            <h1>{state?.name || "Demand planning analyst"}</h1>
            <p>Exported {new Date().toLocaleString()}</p>
          </div>
          {messages.length === 0 && !pending && (
            <div className="empty">
              <h2>Start with your data</h2>
              <p>
                Upload a .zip of Excel or CSV files (one file per table), a single file, or load the demo dataset. I will profile every
                table, infer how they join, check data quality, and then answer planning questions with forecasts, reorder plans,
                charts and scenario analysis.
              </p>
              <p className="muted small">You can also drop a file anywhere on this page. Your work is saved automatically.</p>
            </div>
          )}
          {messages.map((m: any) =>
            m.role === "user" ? (
              <div className="bubble" key={m.id} id={m.id}>
                {m.text}
              </div>
            ) : m.kind === "upload_summary" ? (
              <UploadSummary key={m.id} msg={m} state={state} actions={actions} />
            ) : m.kind === "note" ? (
              <div className="note" key={m.id}>
                {m.text}
              </div>
            ) : (
              <AssistantMessage key={m.id} msg={m} actions={actions} isLast={m.id === lastAssistant} />
            ),
          )}
          {hasData && !busy && messages.filter((m: any) => m.role === "user").length === 0 && (
            <div className="starters">
              <h3>Try asking</h3>
              <div className="chips">
                {STARTERS.map((q) => (
                  <button key={q} className="chip" onClick={() => send(q)}>
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}
          {pending && (
            <Pending
              job={pending.job}
              text={pending.text}
              onStop={
                pending.kind === "chat" && pending.job?.id
                  ? () => api.cancel(pending.job!.id).catch((e) => setError(e.message))
                  : undefined
              }
            />
          )}
          {error && (
            <div className="error" role="alert">
              {error}
              <button className="link" onClick={() => setError(null)}>
                Dismiss
              </button>
            </div>
          )}
          <div ref={endRef} />
        </section>
        <form
          className="composer"
          onSubmit={(e) => {
            e.preventDefault();
            send(input);
          }}
        >
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send(input);
              }
            }}
            placeholder={
              hasData
                ? "Ask a question or describe an event, e.g. “A 25% tariff on imports from South Korea starts 10/29/2026”"
                : "Load data first, then ask a question"
            }
            rows={2}
            disabled={busy}
            aria-label="Message"
          />
          <button className="primary" type="submit" disabled={busy || !input.trim()}>
            Send
          </button>
        </form>
      </main>
    </div>
  );
}
