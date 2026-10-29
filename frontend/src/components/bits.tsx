import React, { useEffect, useRef } from "react";
import katex from "katex";
import { marked } from "marked";

marked.use({ renderer: { html: () => "" } }); // never render raw HTML coming from the model

export function Md({ text, className }: { text?: string; className?: string }) {
  if (!text) return null;
  const html = marked.parse(text, { async: false }) as string;
  return <div className={"md " + (className || "")} dangerouslySetInnerHTML={{ __html: html }} />;
}

const BADGE: Record<string, { label: string; glyph: string; title: string }> = {
  computed: { label: "Computed", glyph: "=", title: "Computed from data and the stated event parameters" },
  estimated: { label: "Estimated", glyph: "≈", title: "Depends on a model (forecast or fitted elasticity)" },
  assumed: { label: "Assumed", glyph: "?", title: "User-supplied or hypothetical assumption" },
  user: { label: "User", glyph: "✓", title: "Provided by you" },
  default: { label: "Default", glyph: "◇", title: "Default assumption, waiting for your confirmation" },
};

export function SourceBadge({ source }: { source?: string }) {
  const k = (source || "").toLowerCase();
  const b = BADGE[k] || { label: source || "?", glyph: "", title: "" };
  return (
    <span className={`badge badge-${k}`} title={b.title}>
      <span className="badge-glyph" aria-hidden>
        {b.glyph}
      </span>
      {b.label}
    </span>
  );
}

export function StatusDot({ status }: { status?: string }) {
  const label = status === "ok" ? "OK" : status === "warning" ? "Check" : status === "error" ? "Problem" : status || "";
  return (
    <span className={`status status-${status}`} title={label}>
      <span className="status-dot" aria-hidden />
      {label}
    </span>
  );
}

export function Formula({ tex }: { tex: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    try {
      katex.render(tex, ref.current, { throwOnError: false, displayMode: true });
    } catch {
      ref.current.textContent = tex;
    }
  }, [tex]);
  return <span ref={ref} className="formula" />;
}

declare global {
  interface Window {
    Plotly?: any;
  }
}

export function isDark(): boolean {
  const t = document.documentElement.getAttribute("data-theme");
  if (t) return t === "dark";
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;
}

// Dark-mode steps of the categorical palette (same hues, re-stepped for the dark surface)
const LIGHT_TO_DARK: Record<string, string> = {
  "#2a78d6": "#3987e5", "#eb6834": "#d95926", "#1baf7a": "#199e70", "#eda100": "#c98500",
  "#e87ba4": "#d55181", "#008300": "#008300", "#6250d6": "#9085e9", "#e34948": "#e66767", "#52514e": "#c3c2b7",
};

function themed(fig: any, dark: boolean) {
  if (!dark) return fig;
  const swap = (v: any): any => (typeof v === "string" && LIGHT_TO_DARK[v.toLowerCase()] ? LIGHT_TO_DARK[v.toLowerCase()] : Array.isArray(v) ? v.map(swap) : v);
  const data = (fig.data || []).map((t: any) => ({
    ...t,
    line: t.line ? { ...t.line, color: swap(t.line.color) } : t.line,
    marker: t.marker ? { ...t.marker, color: swap(t.marker.color) } : t.marker,
  }));
  return { ...fig, data };
}

export function PlotlyChart({ figure, themeTick }: { figure: any; themeTick?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el || !figure) return;
    let tries = 0;
    let timer: any;
    const draw = () => {
      if (!window.Plotly) {
        if (tries++ < 40) {
          timer = setTimeout(draw, 250);
        } else {
          el.textContent = "The chart library did not load. Restart the app (it serves /vendor/plotly.min.js from the plotly package).";
          el.classList.add("chart-missing");
        }
        return;
      }
      render();
    };
    const render = () => {
      const dark = isDark();
      const fig = themed(figure, dark);
      const ink = dark ? "#ece8f1" : "#2a2233";
      const grid = dark ? "rgba(236,232,241,0.12)" : "rgba(128,128,128,0.18)";
      const layout = {
        ...fig.layout,
        autosize: true,
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { ...(fig.layout?.font || {}), family: "Instrument Sans, system-ui, sans-serif", color: ink },
        xaxis: { ...(fig.layout?.xaxis || {}), gridcolor: grid, linecolor: grid },
        yaxis: { ...(fig.layout?.yaxis || {}), gridcolor: grid, linecolor: grid },
        hoverlabel: { bgcolor: dark ? "#31283b" : "#ffffff", font: { color: ink } },
      };
      window.Plotly.react(el, fig.data, layout, { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"] });
    };
    draw();
    return () => {
      clearTimeout(timer);
      try {
        window.Plotly.purge(el);
      } catch {
        /* ignore */
      }
    };
  }, [figure, themeTick]);
  return <div className="chart" ref={ref} />;
}

export function Spinner() {
  return <span className="spinner" aria-label="Working" />;
}
