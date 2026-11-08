import { useEffect, useMemo, useRef, useState } from "react";
import { fetchJson } from "./useRegion";
import type { WorkbenchData } from "./data";
import type { Location } from "./routes";
import { Badge, Card, Detail, EmptyState, ErrorState, Field, KeyValue, LoadingState, SaveFeedback, type SaveState } from "./ui";
import { fmtDateStamp, fmtInt } from "./format";
import type { Cohort } from "./types";

/** Sổ nghiên cứu: annotations bound to a query identity, and cohorts/hypotheses.
 *
 *  Two defects fixed here:
 *  * the annotation page was an empty heading when no query was selected. Now it either shows a
 *    working form for the selected query, or explains how to select one AND offers the next action;
 *  * the note was keyed by `query_id` alone. The CIRCUS release and the FashionIQ index both use
 *    `7:603`, so the same note could appear under two dataset identities. The record now names the
 *    protocol and the query identity together, and the UI displays both, so a note cannot silently
 *    move to another dataset.
 */
const LABELS = [
  "preservation_failure", "under_edit", "over_edit", "partial_edit", "wrong_attribute",
  "wrong_entity_binding", "reference_dominance", "text_dominance", "fine_grained_visual_confusion",
  "ambiguous_ground_truth", "possible_dataset_issue", "duplicate_or_near_duplicate", "other", "unknown",
];

export function NotebookPage({ data, location, view, onGoto }: {
  data: WorkbenchData;
  location: Location;
  view: string;
  onGoto: (patch: Partial<Location>) => void;
}) {
  if (view === "cohorts") return <CohortsView data={data} onGoto={onGoto} location={location} />;
  return <AnnotationsView data={data} location={location} onGoto={onGoto} />;
}

function AnnotationsView({ data, location, onGoto }: {
  data: WorkbenchData;
  location: Location;
  onGoto: (patch: Partial<Location>) => void;
}) {
  const queryId = location.query;
  const protocol = location.protocol;
  const [labels, setLabels] = useState<string[]>([]);
  const [note, setNote] = useState("");
  const [state, setState] = useState<SaveState>({ status: "idle" });
  const [loadedFor, setLoadedFor] = useState<string | null>(null);
  const requestRef = useRef(0);

  const key = `${protocol}\u001f${queryId}`;
  useEffect(() => {
    const token = ++requestRef.current;
    setLoadedFor(null);
    setLabels([]);
    setNote("");
    if (!queryId) return;
    const controller = new AbortController();
    fetchJson<Array<{ labels: string[]; note: string }>>(
      `/api/annotations?query_id=${encodeURIComponent(queryId)}&protocol_id=${encodeURIComponent(protocol)}`,
      controller.signal)
      .then((rows) => {
        if (token !== requestRef.current) return; // superseded: another query became active
        setLabels(rows[0]?.labels ?? []);
        setNote(rows[0]?.note ?? "");
        setLoadedFor(key);
      })
      .catch((cause: unknown) => {
        if (controller.signal.aborted || token !== requestRef.current) return;
        setLoadedFor(key);
        setState({ status: "error", message: cause instanceof Error ? cause.message : String(cause) });
      });
    return () => controller.abort();
  }, [queryId, protocol, key]);

  async function save() {
    if (!queryId) return;
    // The server owns the identity rule. It accepts both `category:index:reference:target` for the
    // registry protocols and `7:<n>` for the CIRCUS release (when the id exists in the release that
    // protocol declares), and it refuses a cross-protocol pair by name
    // (`query_id_protocol_mismatch`). A client-side copy of that rule blocked a flow the server had
    // already learned to accept, which is the same defect class as two key builders: a duplicated
    // validator that drifts. So there is no pre-check here — the request is sent and the server's
    // own named error is surfaced verbatim.
    setState({ status: "saving" });
    try {
      const response = await fetch("/api/annotations", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query_id: queryId, protocol_id: protocol, labels, note }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        const inner = payload?.detail ?? payload;
        const code = inner?.error ? `${inner.error}: ` : "";
        throw new Error(code + (inner?.message || `${response.status}`));
      }
      setState({ status: "saved" });
    } catch (cause) {
      setState({ status: "error", message: cause instanceof Error ? cause.message : String(cause) });
    }
  }

  if (!queryId) {
    return (
      <div className="stack" data-testid="area-notebook">
      <Card title="Ghi chú" subtitle="Gắn nhận xét vào một query cụ thể">
        <EmptyState kind="none-selected"
                    detail="Chưa chọn query nào. Ghi chú được lưu theo cặp (query, protocol) nên phải chọn query trước."
                    next={(
                      <>
                        <button type="button" className="primary"
                                onClick={() => onGoto({ area: "samples", view: "explorer" })}>
                          Mở inspector để chọn query
                        </button>
                        <button type="button" className="ghost"
                                onClick={() => onGoto({ area: "failures", view: "consensus" })}>
                          Xem cohort lỗi có sẵn
                        </button>
                      </>
                    )} />
      </Card>
      </div>
    );
  }

  return (
    <div className="stack" data-testid="area-notebook">
      <Card
        title="Ghi chú"
        subtitle={(
          <span data-testid="annotation-query-id">
            Query <code>{queryId}</code> · protocol <code>{protocol}</code> · dataset do protocol quyết định
          </span>
        )}
        actions={<button type="button" className="ghost" onClick={() => onGoto({ area: "samples", view: "explorer" })}>Đổi query</button>}
      >
        {loadedFor !== key && state.status !== "error"
          ? <LoadingState label="Đang đọc ghi chú đã lưu…" />
          : (
            <div className="stack">
              <div className="field">
                <span className="field-label">Nhãn lỗi</span>
                <div className="row" role="group" aria-label="Nhãn lỗi">
                  {LABELS.map((label) => (
                    <label key={label} className="badge" style={{ cursor: "pointer", textTransform: "none" }}>
                      <input type="checkbox" checked={labels.includes(label)}
                             onChange={() => setLabels((current) => current.includes(label)
                               ? current.filter((item) => item !== label)
                               : [...current, label])} />
                      {label}
                    </label>
                  ))}
                </div>
              </div>
              <Field label="Ghi chú" htmlFor="annotation-note">
                <textarea id="annotation-note" data-testid="annotation-note" value={note}
                          onChange={(event) => setNote(event.target.value)}
                          placeholder="Vì sao query này đáng chú ý: mô hình bỏ qua thuộc tính nào, ảnh đích có mơ hồ không…" />
              </Field>
              <div className="row">
                <button type="button" className="primary" data-testid="annotation-save"
                        onClick={() => void save()} disabled={state.status === "saving"}>
                  Lưu / cập nhật
                </button>
                <SaveFeedback state={state} />
              </div>
              <Detail summary="Định danh đang lưu">
                <KeyValue rows={[
                  ["query_id", <code key="q">{queryId}</code>],
                  ["protocol_id", <code key="p">{protocol}</code>],
                  ["khoá lưu", <code key="k">(query_id, protocol_id)</code>],
                  ["nhãn đã chọn", labels.length ? labels.join(", ") : "chưa có"],
                ]} />
              </Detail>
            </div>
          )}
      </Card>
    </div>
  );
}

function CohortsView({ data, onGoto, location }: {
  data: WorkbenchData;
  onGoto: (patch: Partial<Location>) => void;
  location: Location;
}) {
  const region = data.cohorts.region;
  const cohorts = useMemo(() => region.data ?? [], [region.data]);
  return (
    <div className="stack" data-testid="area-notebook">
      <Card
        title="Cohort & giả thuyết"
        subtitle={`${fmtInt(cohorts.length)} cohort đã lưu · mỗi cohort ghi phạm vi, preset và số query`}
        actions={<button type="button" onClick={data.cohorts.retry}>Làm mới</button>}
      >
        {region.status === "error"
          ? <ErrorState message={region.error ?? ""} onRetry={data.cohorts.retry} />
          : region.status === "loading" && !region.data
            ? <LoadingState />
            : cohorts.length === 0
              ? <EmptyState kind="no-results" title="Chưa có cohort nào"
                            detail="Cohort được tạo từ khu vực Phân tích lỗi: lọc theo preset rồi lưu. Toàn bộ tập đã lọc được lưu, không chỉ trang đang xem."
                            next={<button type="button" className="primary"
                                          onClick={() => onGoto({ area: "failures", view: "consensus" })}>
                              Đi tới Phân tích lỗi
                            </button>} />
              : (
                <div className="stack">
                  {cohorts.map((cohort) => <CohortRow key={cohort.cohort_id} cohort={cohort} />)}
                </div>
              )}
      </Card>
      {location.query && (
        <p className="hint">Query đang xem: <code>{location.query}</code> — ghi chú của query này nằm ở tab Ghi chú.</p>
      )}
    </div>
  );
}

function CohortRow({ cohort }: { cohort: Cohort }) {
  const definition = cohort.definition as Record<string, unknown>;
  return (
    <article className="card" style={{ padding: 12 }} data-testid={`cohort-${cohort.cohort_id}`}>
      <div className="row">
        <b>{cohort.cohort_id}</b>
        <Badge tone="info">{fmtInt(cohort.query_ids.length)} query</Badge>
        <Badge>{fmtInt(cohort.run_ids.length)} lượt</Badge>
        <span className="hint" style={{ marginLeft: "auto" }}>{fmtDateStamp(cohort.created_at)}</span>
      </div>
      <Detail summary="Phạm vi và định nghĩa">
        <KeyValue rows={[
          ["protocol", <code key="p">{cohort.protocol_id}</code>],
          ["preset", String(definition?.preset ?? "—")],
          ["cutoff @K", String(definition?.k ?? "—")],
          ["cohort", String(definition?.cohort ?? "—")],
          ["partial", String(definition?.partial ?? "—")],
          ["run_ids", <code key="r">{cohort.run_ids.join(", ")}</code>],
        ]} />
      </Detail>
    </article>
  );
}
