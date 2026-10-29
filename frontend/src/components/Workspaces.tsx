import React, { useEffect, useRef, useState } from "react";
import { api } from "../api";

type Props = {
  current: any;
  disabled: boolean;
  onSwitch: (id: string) => void;
  onRenamed: () => void;
  onError: (msg: string) => void;
};

/** Workspace switcher: current name (click to rename), list of saved workspaces, new and delete. */
export function Workspaces({ current, disabled, onSwitch, onRenamed, onError }: Props) {
  const [open, setOpen] = useState(false);
  const [list, setList] = useState<any[]>([]);
  const [renaming, setRenaming] = useState(false);
  const [name, setName] = useState("");
  const boxRef = useRef<HTMLDivElement>(null);

  const load = () =>
    api
      .workspaces()
      .then((r) => setList(r.workspaces))
      .catch((e) => onError(e.message));

  useEffect(() => {
    if (!open) return;
    load();
    const close = (e: MouseEvent) => {
      if (boxRef.current && !boxRef.current.contains(e.target as Node)) setOpen(false);
    };
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", esc);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  if (!current) return <span className="muted small">Loading workspace…</span>;

  return (
    <div className="ws" ref={boxRef}>
      {renaming ? (
        <form
          className="ws-rename"
          onSubmit={async (e) => {
            e.preventDefault();
            try {
              await api.renameWorkspace(current.session_id, name);
              setRenaming(false);
              onRenamed();
            } catch (x: any) {
              onError(x.message);
            }
          }}
        >
          <input autoFocus value={name} onChange={(e) => setName(e.target.value)} aria-label="Workspace name" maxLength={60} />
          <button type="submit" className="small">
            Save
          </button>
          <button type="button" className="small" onClick={() => setRenaming(false)}>
            Cancel
          </button>
        </form>
      ) : (
        <div className="ws-current">
          <button className="ws-toggle" onClick={() => setOpen(!open)} disabled={disabled} aria-expanded={open} aria-haspopup="listbox">
            {current.name || "Workspace"} <span aria-hidden>▾</span>
          </button>
          <button
            className="link small"
            onClick={() => {
              setName(current.name || "");
              setRenaming(true);
            }}
          >
            Rename
          </button>
        </div>
      )}
      {open && (
        <div className="ws-menu" role="listbox">
          <button
            className="ws-new"
            onClick={async () => {
              try {
                const r = await api.newWorkspace();
                setOpen(false);
                onSwitch(r.session_id);
              } catch (x: any) {
                onError(x.message);
              }
            }}
          >
            + New workspace
          </button>
          {list.length === 0 && <p className="muted small">No saved workspaces yet.</p>}
          {list.map((w) => (
            <div key={w.id} className={"ws-item" + (w.id === current.session_id ? " active" : "")}>
              <button
                className="ws-pick"
                role="option"
                aria-selected={w.id === current.session_id}
                onClick={() => {
                  setOpen(false);
                  if (w.id !== current.session_id) onSwitch(w.id);
                }}
              >
                <strong>{w.name}</strong>
                <span className="muted small">
                  {w.tables} tables, {w.questions} questions{w.scenarios ? `, ${w.scenarios} scenarios` : ""}; updated {w.updated}
                </span>
              </button>
              {w.id !== current.session_id && (
                <button
                  className="link small danger"
                  onClick={async () => {
                    if (!window.confirm(`Delete the workspace "${w.name}" and its uploaded data? This cannot be undone.`)) return;
                    try {
                      await api.deleteWorkspace(w.id);
                      load();
                    } catch (x: any) {
                      onError(x.message);
                    }
                  }}
                >
                  Delete
                </button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
