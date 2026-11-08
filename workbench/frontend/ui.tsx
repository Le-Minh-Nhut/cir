import { useId, useState, type ReactNode } from "react";

/** Shared primitives.
 *
 *  One surface style, one type scale, one state vocabulary, so a reader learns the workbench once.
 *  The rules that matter scientifically are encoded here rather than repeated per page:
 *
 *  * an empty region always names WHICH emptiness it is (nothing selected / no results yet / no
 *    match for the filter / capability missing / request failed), because those imply different
 *    next steps;
 *  * a disabled control carries its reason as text next to it, never as a silent no-op;
 *  * long technical detail (hashes, paths, environment dumps, tracebacks) is disclosure, not row
 *    content.
 */

export function Badge({ tone = "neutral", children, title }: {
  tone?: "neutral" | "ok" | "warn" | "bad" | "info";
  children: ReactNode;
  title?: string;
}) {
  return <span className={`badge badge-${tone}`} title={title}>{children}</span>;
}

export function Card({ title, subtitle, actions, children, className = "", testId }: {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  testId?: string;
}) {
  return (
    <section className={`card ${className}`} data-testid={testId}>
      {(title || actions) && (
        <header className="card-head">
          <div>
            {title && <h2 className="card-title">{title}</h2>}
            {subtitle && <p className="card-sub">{subtitle}</p>}
          </div>
          {actions && <div className="card-actions">{actions}</div>}
        </header>
      )}
      {children}
    </section>
  );
}

export type EmptyKind = "none-selected" | "no-results" | "no-match" | "capability" | "error" | "loading" | "partial";

const EMPTY_TITLES: Record<EmptyKind, string> = {
  "none-selected": "Chưa chọn gì",
  "no-results": "Chưa có kết quả",
  "no-match": "Không có mục nào khớp bộ lọc",
  capability: "Không có dữ liệu cho khả năng này",
  error: "Yêu cầu thất bại",
  loading: "Đang tải",
  partial: "Dữ liệu chưa đầy đủ",
};

/** The only way to render "nothing here". `kind` is required so a page cannot blur the five cases
 *  into one grey sentence, and `next` carries the actual next action. */
export function EmptyState({ kind, title, detail, next, testId = "empty-state" }: {
  kind: EmptyKind;
  title?: string;
  detail?: ReactNode;
  next?: ReactNode;
  testId?: string;
}) {
  return (
    <div className={`empty empty-${kind}`} data-testid={testId} data-empty-kind={kind} role="status">
      <b className="empty-title">{title ?? EMPTY_TITLES[kind]}</b>
      {detail && <div className="empty-detail">{detail}</div>}
      {next && <div className="empty-next">{next}</div>}
    </div>
  );
}

export function ErrorState({ message, onRetry, retryLabel = "Thử lại" }: {
  message: string;
  onRetry?: () => void;
  retryLabel?: string;
}) {
  return (
    <div className="empty empty-error" data-testid="empty-state" data-empty-kind="error" role="alert">
      <b className="empty-title">Yêu cầu thất bại</b>
      <div className="empty-detail"><code>{message}</code></div>
      {onRetry && (
        <div className="empty-next">
          <button type="button" data-testid="error-retry" onClick={onRetry}>{retryLabel}</button>
        </div>
      )}
    </div>
  );
}

export function LoadingState({ label = "Đang tải…" }: { label?: string }) {
  return (
    <div className="empty empty-loading" data-testid="empty-state" data-empty-kind="loading"
         role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" />
      <span className="empty-detail">{label}</span>
    </div>
  );
}

/** A control that cannot act says why, in text, adjacent. */
export function Disabled({ reason, children }: { reason: string; children: ReactNode }) {
  return (
    <span className="disabled-wrap" title={reason}>
      {children}
      <span className="disabled-reason">{reason}</span>
    </span>
  );
}

export function Field({ label, hint, children, htmlFor }: {
  label: string;
  hint?: ReactNode;
  children: ReactNode;
  htmlFor?: string;
}) {
  return (
    <div className="field">
      <label className="field-label" htmlFor={htmlFor}>{label}</label>
      {children}
      {hint && <span className="field-hint">{hint}</span>}
    </div>
  );
}

/** A disclosure for the technical layer: identifiers, digests, paths, environment, tracebacks. */
export function Detail({ summary, children, testId }: { summary: ReactNode; children: ReactNode; testId?: string }) {
  return (
    <details className="detail" data-testid={testId}>
      <summary>{summary}</summary>
      <div className="detail-body">{children}</div>
    </details>
  );
}

export function KeyValue({ rows }: { rows: Array<[string, ReactNode]> }) {
  return (
    <dl className="kv">
      {rows.map(([name, value]) => (
        <div key={name} className="kv-row">
          <dt>{name}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  );
}

/** The control that opens a row-detail panel. It stays in the last cell; the panel itself renders
 *  BELOW the table, not inside a cell.
 *
 *  The previous markup put a `<details>` panel inside that cell. The cell is ~100 px wide and `.kv`
 *  is a `max-content 1fr` grid, so the panel grew to 7 711 px tall with 12 px-wide value columns —
 *  measured, not guessed — while the table widened from 1 252 to 1 304 px and the whole row reflowed
 *  into single-character columns. A panel outside the table cannot widen it and gives hashes and
 *  paths the container's full width.
 */
export function DetailToggle({ open, onToggle, label = "Chi tiết" }: {
  open: boolean;
  onToggle: () => void;
  label?: string;
}) {
  return (
    <button type="button" className="rowbtn" aria-expanded={open} onClick={onToggle}>
      <span aria-hidden="true">{open ? "▾" : "▸"}</span> {label}
    </button>
  );
}

export function Progress({ value, max, label }: { value: number; max: number; label: string }) {
  const pct = max > 0 ? Math.min(100, Math.max(0, (value / max) * 100)) : 0;
  return (
    <div className="progress" role="img" aria-label={label} title={label}>
      <span style={{ width: `${pct}%` }} />
    </div>
  );
}

/** Sort/filter state for a compact table. Sorting is by a declared column key and never by
 *  "whatever the API returned", so a reader can trust the order. */
export interface SortState { key: string; dir: "asc" | "desc" }

export function useSort(initial: SortState) {
  const [sort, setSort] = useState<SortState>(initial);
  const toggle = (key: string) => setSort((current) => current.key === key
    ? { key, dir: current.dir === "asc" ? "desc" : "asc" }
    : { key, dir: "desc" });
  return { sort, toggle };
}

export function sortRows<T>(rows: T[], sort: SortState, value: (row: T, key: string) => number | string | null): T[] {
  const copy = [...rows];
  copy.sort((left, right) => {
    const a = value(left, sort.key);
    const b = value(right, sort.key);
    // Nulls always sort last, in both directions: a missing measurement is not the smallest value.
    if (a === null && b === null) return 0;
    if (a === null) return 1;
    if (b === null) return -1;
    if (typeof a === "number" && typeof b === "number") return sort.dir === "asc" ? a - b : b - a;
    const cmp = String(a).localeCompare(String(b));
    return sort.dir === "asc" ? cmp : -cmp;
  });
  return copy;
}

export function SortHeader({ label, column, sort, onSort, numeric, unit }: {
  label: string;
  column: string;
  sort: SortState;
  onSort: (key: string) => void;
  numeric?: boolean;
  unit?: string;
}) {
  const active = sort.key === column;
  return (
    <th scope="col" className={numeric ? "num" : ""} aria-sort={active ? (sort.dir === "asc" ? "ascending" : "descending") : "none"}>
      <button type="button" className={`sort-head ${active ? "active" : ""}`} onClick={() => onSort(column)}>
        <span>{label}{unit && <span className="unit"> {unit}</span>}</span>
        <span aria-hidden="true" className="sort-arrow">{active ? (sort.dir === "asc" ? "▲" : "▼") : "↕"}</span>
      </button>
    </th>
  );
}

/** A drawer for one row's technical detail. Rows stay one line tall; the drawer is where hashes,
 *  paths, environment knobs and tracebacks live. */
export function Drawer({ open, title, onClose, children }: {
  open: boolean;
  title: ReactNode;
  onClose: () => void;
  children: ReactNode;
}) {
  const titleId = useId();
  if (!open) return null;
  return (
    <div className="drawer-scrim" onClick={onClose} role="presentation">
      <aside className="drawer" role="dialog" aria-modal="true" aria-labelledby={titleId}
             onClick={(event) => event.stopPropagation()}>
        <header className="drawer-head">
          <h2 id={titleId}>{title}</h2>
          <button type="button" className="ghost" onClick={onClose} aria-label="Đóng chi tiết">Đóng</button>
        </header>
        <div className="drawer-body">{children}</div>
      </aside>
    </div>
  );
}

/** Copy-to-clipboard with an honest state: it reports what happened, including failure. */
export function CopyLink({ href }: { href: string }) {
  const [state, setState] = useState<"idle" | "ok" | "failed">("idle");
  return (
    <span className="copy-link">
      <button type="button" className="secondary copy-button" data-testid="copy-link" onClick={async () => {
        try {
          await navigator.clipboard.writeText(href);
          setState("ok");
        } catch {
          setState("failed");
        }
        window.setTimeout(() => setState("idle"), 2500);
      }}>
        <span aria-hidden="true">⧉</span> Sao chép liên kết
      </button>
      {state === "ok" && <span className="hint-ok" role="status">Đã sao chép</span>}
      {state === "failed" && (
        <input className="mono copy-fallback" readOnly value={href}
               aria-label="Liên kết để sao chép thủ công" onFocus={(event) => event.currentTarget.select()} />
      )}
    </span>
  );
}

/** Save action with its three real states. "Save" that cannot fail cannot be trusted. */
export type SaveState = { status: "idle" | "saving" | "saved" | "error"; message?: string };

export function SaveFeedback({ state, count }: { state: SaveState; count?: number }) {
  // `data-status` is the machine-readable face of the same fact, so a harness asserts the state
  // instead of parsing Vietnamese prose. It is always rendered (including `idle`) so "nothing
  // happened yet" is observable rather than an absent element.
  return (
    <span data-testid="save-feedback" data-status={state.status} data-count={count ?? ""}>
      {state.status === "saving" && <span className="hint" role="status">Đang lưu…</span>}
      {state.status === "saved" && (
        <span className="hint-ok" role="status">
          Đã lưu{count === undefined ? "" : ` ${count.toLocaleString("en-US")} query`}
        </span>
      )}
      {state.status === "error" && (
        <span className="hint-bad" role="alert">Lưu thất bại: {state.message}</span>
      )}
    </span>
  );
}
