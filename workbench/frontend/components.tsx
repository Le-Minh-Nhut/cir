import { BASE } from "./client";
import type { Run } from "./types";

export function noiseConditionLabel(modelId: string, trainingNoise: number | null, evaluationNoise: number | null): string {
  const model = modelId.toLowerCase() === "conesep" ? "ConeSep" : modelId.toUpperCase();
  return `${model} · train noise ${trainingNoise == null ? "unrecorded" : `${trainingNoise}%`} · eval noise ${evaluationNoise == null ? "unrecorded" : `${evaluationNoise}%`}`;
}

export function runLabel(run: Run): string {
  return `${noiseConditionLabel(run.model_id, run.checkpoint_training_noise_pct, run.evaluation_noise_pct)} · ${run.method_name} / ${run.checkpoint_id}`;
}

export function Image({ category, id, kind }: { category: string; id: string; kind: string }) {
  return <figure><img src={`${BASE}/api/images/${category}/${encodeURIComponent(id)}`} alt={`${kind}: ${id}`} onError={event => { event.currentTarget.style.display = "none"; }}/><figcaption>{kind}: {id}</figcaption></figure>;
}
