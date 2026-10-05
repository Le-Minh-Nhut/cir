import { BASE } from "./client";
import type { Run } from "./types";

export function runLabel(run: Run): string {
  return `${run.method_name} / ${run.checkpoint_id}${run.checkpoint_training_noise_pct === null ? "" : ` / train-noise ${run.checkpoint_training_noise_pct}%`}`;
}

export function Image({ category, id, kind }: { category: string; id: string; kind: string }) {
  return <figure><img src={`${BASE}/api/images/${category}/${encodeURIComponent(id)}`} alt={`${kind}: ${id}`} onError={event => { event.currentTarget.style.display = "none"; }}/><figcaption>{kind}: {id}</figcaption></figure>;
}
