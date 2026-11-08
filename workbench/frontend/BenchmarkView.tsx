import type { BenchmarkRun, GateVerdict } from "./gate";
import { isCertifiedEvidence, warningLabels } from "./gate";

export function BenchmarkView({
  benchmarkRuns,
  thresholdPct,
}: {
  benchmarkRuns: BenchmarkRun[];
  thresholdPct: number;
}) {
  return (
    <section className="benchmark-section">
      <h2>Benchmark</h2>
      <div className="benchmark-banner">
        <p>
          <b>Official Aggregate Evaluations:</b> These results represent certified macro benchmarks corroborated by the evidence ledger.
          Per-query sample rankings were not exported by design (<code>per_query_state: CAPABILITY_ABSENT</code>); Top-K exploration is disabled for these runs.
        </p>
        <p className="paper-sanity-note">
          <small>
            <b>Notice on Paper Scores:</b> Values labelled <em>Paper Reference</em> are published literature sanity baselines, NOT measured execution numbers.
            Cross-protocol comparisons fail closed.
          </small>
        </p>
      </div>

      {benchmarkRuns.length === 0 ? (
        <p>No aggregate benchmark reports available.</p>
      ) : (
        <div className="benchmark-list">
          {benchmarkRuns.map((run) => {
            const isCertified = isCertifiedEvidence(run);
            const warningLabelsList = warningLabels(run, thresholdPct);
            const obs = run.observed_metrics || {};
            const paper = run.paper_sanity_metrics || {};
            const cats = run.category_metrics || {};
            const gate = (run.benchmark_gate || {}) as GateVerdict;
            const refEv = gate.reference_evidence && typeof gate.reference_evidence === "object"
              ? (gate.reference_evidence as { category_metrics?: Record<string, { r10?: number; r50?: number }> })
              : undefined;
            const paperCats = refEv?.category_metrics || {};
            const comparisons = gate.comparisons || [];

            return (
              <article key={run.run_id} className="benchmark-run-card">
                <header className="benchmark-run-header">
                  <div>
                    <h3>
                      {run.model_id?.toUpperCase()} · {run.checkpoint_id}
                    </h3>
                    <div className="run-meta">
                      <code>{run.run_id}</code> · protocol: <b>{run.protocol_id}</b>
                    </div>
                  </div>
                  <div className="run-badges">
                    <span className={`badge ${isCertified ? "badge-pass" : "badge-warn"}`}>
                      {run.verification?.state?.toUpperCase() || (isCertified ? "CERTIFIED" : "UNCERTIFIED")}
                    </span>
                    <span className="badge badge-aggregate">
                      aggregate-only — per-query Top-K unavailable (CAPABILITY_ABSENT)
                    </span>
                    {warningLabelsList.map((label) => (
                      <span key={label.kind} className={`warning-label ${label.tone}`}>
                        {label.text}
                      </span>
                    ))}
                  </div>
                </header>

                <div className="table-container">
                  <h4>Macro Evaluation Results vs Paper Reference</h4>
                  <table className="benchmark-table">
                    <thead>
                      <tr>
                        <th>Metric</th>
                        <th>Measured (Evaluated)</th>
                        <th>Paper Reference (Sanity Only)</th>
                        <th>Difference</th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td><b>Recall@10</b></td>
                        <td className="font-mono">{obs.r10 != null ? `${obs.r10.toFixed(2)}% (${obs.r10})` : "—"}</td>
                        <td className="font-mono text-muted">{paper.r10 != null ? `${paper.r10.toFixed(2)}% (${paper.r10})` : "—"}</td>
                        <td className="font-mono">
                          {obs.r10 != null && paper.r10 != null
                            ? `${obs.r10 - paper.r10 > 0 ? "+" : ""}${(obs.r10 - paper.r10).toFixed(2)} pp`
                            : "—"}
                        </td>
                        <td>
                          {obs.r10 != null && paper.r10 != null && Math.abs((obs.r10 - paper.r10) / paper.r10) <= 0.05
                            ? <span className="status-ok">within 5% tolerance</span>
                            : "—"}
                        </td>
                      </tr>
                      <tr>
                        <td><b>Recall@50</b></td>
                        <td className="font-mono">{obs.r50 != null ? `${obs.r50.toFixed(2)}% (${obs.r50})` : "—"}</td>
                        <td className="font-mono text-muted">{paper.r50 != null ? `${paper.r50.toFixed(2)}% (${paper.r50})` : "—"}</td>
                        <td className="font-mono">
                          {obs.r50 != null && paper.r50 != null
                            ? `${obs.r50 - paper.r50 > 0 ? "+" : ""}${(obs.r50 - paper.r50).toFixed(2)} pp`
                            : "—"}
                        </td>
                        <td>
                          {obs.r50 != null && paper.r50 != null && Math.abs((obs.r50 - paper.r50) / paper.r50) <= 0.05
                            ? <span className="status-ok">within 5% tolerance</span>
                            : "—"}
                        </td>
                      </tr>
                      <tr>
                        <td><b>Mean R@10 / R@50</b></td>
                        <td className="font-mono">{obs.mean != null ? `${obs.mean.toFixed(2)}% (${obs.mean})` : "—"}</td>
                        <td className="font-mono text-muted">{paper.mean != null ? `${paper.mean.toFixed(2)}% (${paper.mean})` : "—"}</td>
                        <td className="font-mono">
                          {obs.mean != null && paper.mean != null
                            ? `${obs.mean - paper.mean > 0 ? "+" : ""}${(obs.mean - paper.mean).toFixed(2)} pp`
                            : "—"}
                        </td>
                        <td>
                          {obs.mean != null && paper.mean != null && Math.abs((obs.mean - paper.mean) / paper.mean) <= 0.05
                            ? <span className="status-ok">within 5% tolerance</span>
                            : "—"}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>

                <div className="table-container">
                  <h4>Per-Category Breakdown</h4>
                  <table className="benchmark-table">
                    <thead>
                      <tr>
                        <th>Category</th>
                        <th>Measured R@10</th>
                        <th>Paper Ref R@10</th>
                        <th>Measured R@50</th>
                        <th>Paper Ref R@50</th>
                      </tr>
                    </thead>
                    <tbody>
                      {["dress", "shirt", "toptee"].map((cat) => {
                        const catObs = cats[cat] || {};
                        const catPaper = paperCats[cat] || {};
                        return (
                          <tr key={cat}>
                            <td><b>{cat}</b></td>
                            <td className="font-mono">{catObs.r10 != null ? `${catObs.r10.toFixed(2)}% (${catObs.r10.toFixed(4)})` : "—"}</td>
                            <td className="font-mono text-muted">{catPaper.r10 != null ? `${catPaper.r10.toFixed(2)}% (${catPaper.r10})` : "—"}</td>
                            <td className="font-mono">{catObs.r50 != null ? `${catObs.r50.toFixed(2)}% (${catObs.r50.toFixed(4)})` : "—"}</td>
                            <td className="font-mono text-muted">{catPaper.r50 != null ? `${catPaper.r50.toFixed(2)}% (${catPaper.r50})` : "—"}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                <details className="gate-detail">
                  <summary>Full Gate Verification & Provenance</summary>
                  <p><b>Status:</b> {gate.status || "—"} {gate.detail ? `(${gate.detail})` : ""}</p>
                  <p><b>Parity:</b> {run.parity_status || "—"}</p>
                  <p>
                    <b>Evaluator Knobs:</b>{" "}
                    {run.evaluator_environment_knobs
                      ? Object.entries(run.evaluator_environment_knobs).map(([k, v]) => `${k}=${v}`).join(", ")
                      : "evaluator_environment_knobs: null (unrecorded)"}
                  </p>
                  {comparisons.length > 0 && (
                    <div className="table-container">
                      <table className="benchmark-table">
                        <thead>
                          <tr>
                            <th>Metric</th>
                            <th>Paper Ref</th>
                            <th>Reproduced</th>
                            <th>&Delta; (pp)</th>
                            <th>Rel. Error</th>
                            <th>Within Tol.</th>
                            <th>Provenance</th>
                          </tr>
                        </thead>
                        <tbody>
                          {comparisons.map((c) => (
                            <tr key={c.metric} className={c.within_tolerance === false ? "deviating" : ""}>
                              <td>{c.metric}</td>
                              <td>{c.paper ?? "—"}</td>
                              <td>{c.reproduced ?? "—"}</td>
                              <td>{c.signed_delta_pp ?? "—"}</td>
                              <td>{c.relative_error_pct != null ? `${c.relative_error_pct.toFixed(2)}%` : "—"}</td>
                              <td>{c.within_tolerance ? "yes" : "no"}</td>
                              <td>{c.paper_provenance || "—"}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </details>
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}
