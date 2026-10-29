// Thin client for the backend API. Long operations return a job id that is polled for progress.

export type Job = {
  id: string;
  kind: string;
  status: "running" | "done" | "error";
  progress: { t: number; msg: string }[];
  result?: any;
  error?: string;
  seconds?: number;
};

async function req(path: string, opts: RequestInit = {}): Promise<any> {
  let res: Response;
  try {
    res = await fetch(path, opts);
  } catch {
    throw new Error("Cannot reach the app server. Check that it is still running (uv run python start.py).");
  }
  let body: any = null;
  try {
    body = await res.json();
  } catch {
    /* non-JSON */
  }
  if (res.status === 401 && body?.auth) {
    window.dispatchEvent(new Event("dpa-auth-required"));
    throw new Error("Sign in required.");
  }
  if (!res.ok) throw new Error(body?.error || `Request failed (${res.status})`);
  return body;
}

export const api = {
  createSession: () => req("/api/session", { method: "POST" }),
  state: (sid: string) => req(`/api/session/${sid}/state`),
  upload: (sid: string, file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return req(`/api/session/${sid}/upload`, { method: "POST", body: fd });
  },
  demo: (sid: string) => req(`/api/session/${sid}/demo`, { method: "POST" }),
  chat: (sid: string, text: string) => post(`/api/session/${sid}/chat`, { text }),
  relationships: (sid: string, body: any) => post(`/api/session/${sid}/relationships`, body),
  workbook: (sid: string, file: string, mode: "first" | "all") => post(`/api/session/${sid}/workbook`, { file, mode }),
  dictionary: (sid: string, table: string, column: string, description: string, unit?: string) =>
    post(`/api/session/${sid}/dictionary`, { table, column, description, unit }),
  confirmScenario: (sid: string, name: string) =>
    req(`/api/session/${sid}/scenarios/${encodeURIComponent(name)}/confirm`, { method: "POST" }),
  job: (jid: string): Promise<Job> => req(`/api/jobs/${jid}`),
  cancel: (jid: string) => req(`/api/jobs/${jid}/cancel`, { method: "POST" }),
  workspaces: () => req("/api/workspaces"),
  newWorkspace: (name?: string) => post("/api/workspaces", { name }),
  renameWorkspace: (sid: string, name: string) => post(`/api/workspaces/${sid}`, { name }),
  deleteWorkspace: (sid: string) => req(`/api/workspaces/${sid}`, { method: "DELETE" }),
  auth: () => req("/api/auth"),
  login: (password: string) => post("/api/login", { password }),
  answerXlsx: (sid: string, mid: string) => `/api/session/${sid}/export/answer/${mid}.xlsx`,
};

function post(path: string, body: any) {
  return req(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}

export async function pollJob(jid: string, onProgress: (j: Job) => void, intervalMs = 700): Promise<Job> {
  for (;;) {
    const j = await api.job(jid);
    onProgress(j);
    if (j.status !== "running") return j;
    await new Promise((r) => setTimeout(r, intervalMs));
  }
}

export function fmt(v: any, digits = 1): string {
  if (v === null || v === undefined || v === "") return "–";
  if (typeof v !== "number") return String(v);
  const a = Math.abs(v);
  if (a >= 1e6) return (v / 1e6).toFixed(2) + "M";
  if (a >= 1e4) return Math.round(v).toLocaleString();
  if (Number.isInteger(v)) return v.toLocaleString();
  return v.toLocaleString(undefined, { maximumFractionDigits: digits });
}
