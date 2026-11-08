export type TopResult = { rank:number; image_id:string; score:number };
export type Query = { query_id:string; category:string; annotation_index:number; reference_id:string; target_id:string; raw_captions:string[]; model_input_text:string; target_rank:number; top_results:TopResult[] };
export type Run = { run_id:string; protocol_id:string; model_id:string; method_name:string; checkpoint_id:string; checkpoint_training_noise_pct:number|null; evaluation_noise_pct:number|null; data_kind:"mock"|"experiment"; top_k_saved:number; gallery_size:number; category_gallery_sizes?:Record<string,number>|null; category_eligible_counts?:Record<string,number>|null; evidence_status?:"PER_QUERY_VERIFIED"|"SYNTHETIC_OR_MOCK"|string|null };
export type CheckpointAvailability = { checkpoint_id:string; checkpoint_training_noise_pct:number|null; evaluation_noise_pct:number|null; automatic_download_available:boolean; checkpoint_downloaded:boolean; local_sha256:string|null; official_sha256_known:boolean; mapping_verified:boolean; adapter_command_ready:boolean; runtime_verified:boolean; runnable:boolean; blocking_reasons:string[] };
export type CheckpointBundle = { bundle_id:string; required_checkpoint_ids:string[]; structure?:string; mapping_status?:string; notes?:string; members?:CheckpointAvailability[] };
export type Model = { model_id:string; method_name:string; paper_title?:string|null; paper_url?:string|null; native_protocol:string|null; literature_split_label:"original"|"val"|null; reproduction_status:string; strict_clean_checkpoint_available:boolean|null; checkpoints:CheckpointAvailability[]; checkpoint_bundles?:CheckpointBundle[] };
export type Analysis = { query_id:string; category:string; target_rank:Record<string,number>; consensus_fail_fraction:number; min_rank:number; max_rank:number; median_rank:number; rank_range:number; rank_std:number; mean_topk_jaccard:number; common_distractors:{image_id:string;run_count:number;run_ids:string[];best_rank:number;mean_rank:number}[] };
export type Cohort = { cohort_id:string; protocol_id:string; query_ids:string[]; run_ids:string[]; definition:Record<string,unknown>; notes:string; created_at:string };
export type Job = { job_id:string; model_id:string; checkpoint_id:string; protocol_id:string; command:string[]; cwd:string; log_directory:string; status:"queued"|"preparing"|"running"|"blocked"|"completed"|"failed"|"cancelled"; created_at:string; started_at:string|null; completed_at:string|null; return_code:number|null; error:string|null; stdout_tail:string; stderr_tail:string };
/** Per-flag evidence from `capability_detail`, with a NAMED status when absent. A false flag with a
 *  status and a reason is a capability the run does not have — not an error, not a broken run. */
export type CapabilityDetail = {
  available?: boolean;
  status?: string | null;
  reason?: string | null;
  query_count?: number | null;
  conditions?: string[];
  depth?: number | null;
  source?: string | null;
  sampled?: number | null;
  resolved?: number | null;
  layout?: string | null;
};
export type CapabilityFlags = {
  aggregate_metrics: boolean;
  per_query_target_ranks: boolean;
  candidate_topk: boolean;
  image_availability: boolean;
};
/** Sample count, denominator and the unscored/failure split. `unscored_query_count` is a DIFFERENT
 *  counter from retrieval failure and must never be summed into one. */
export type Measurement = {
  sample_count?: number | null;
  denominator?: number | null;
  denominator_reason?: string | null;
  coverage?: number | null;
  scored_count?: number | null;
  unscored_query_count?: number | null;
  unscored_by_kind?: Record<string, number> | null;
  image_availability_sample_size?: number | null;
};
export type TransferInfo = {
  is_transfer: boolean;
  training_dataset?: string | null;
  evaluation_dataset?: string | null;
  label?: string | null;
};
export type DuplicateStatus = "unique" | "duplicate" | "undetermined" | "synthetic_fixture";

export type CompositionMetrics = {
  ndcg_multimodal?: number | null;
  ndcg_text_only?: number | null;
  ndcg_image_only?: number | null;
  composition_gap_ndcg?: number | null;
  gap_cohort_count?: number | null;
  gap_multimodal_ndcg?: number | null;
  gap_text_only_ndcg?: number | null;
  gap_image_only_ndcg?: number | null;
  gap_unavailable_reason?: string | null;
  multimodal?: { scored_count?: number | null; query_count?: number | null; coverage?: number | null;
                 recall_at_10?: number | null; recall_at_50?: number | null; ndcg_full?: number | null;
                 mrr?: number | null; unavailable_reason?: string | null } | null;
  text_only?: { scored_count?: number | null; query_count?: number | null; coverage?: number | null;
                recall_at_10?: number | null; recall_at_50?: number | null; ndcg_full?: number | null;
                mrr?: number | null; unavailable_reason?: string | null } | null;
  image_only?: { scored_count?: number | null; query_count?: number | null; coverage?: number | null;
                 recall_at_10?: number | null; recall_at_50?: number | null; ndcg_full?: number | null;
                 mrr?: number | null; unavailable_reason?: string | null } | null;
  recall_at_1_multimodal?: number | null;
  recall_at_10_multimodal?: number | null;
  recall_at_50_multimodal?: number | null;
  mrr_gap?: number | null;
  coverage?: number;
  common_cohort_count?: number;
  outcomes?: {
    composition_required: number;
    text_only: number;
    image_only: number;
    both: number;
    unresolved: number;
    neither_scored: number;
  };
};
export type CompositionRun = {
  model_id: string;
  checkpoint_id: string;
  run_id: string;
  protocol_id: string;
  subset: "full" | "sf" | "validated";
  gallery_count: number;
  query_count: number;
  query_count_per_subset?: Record<string, number>;
  conditions: string[];
  execution_status: string;
  condition_status?: Record<string, { status: string; unscored_count?: number; reason?: string }>;
  checkpoint_training_noise_pct?: number | null;
  evaluation_noise_pct?: number | null;
  checkpoint_transfer?: boolean | string | null;
  checkpoint_identity?: string | null;
  available_subsets?: string[] | null;
  capabilities?: CapabilityFlags | null;
  capability_detail?: Partial<Record<keyof CapabilityFlags, CapabilityDetail>> | null;
  measurement?: Measurement | null;
  statistics?: {
    conditions?: Record<string, { denominator?: number; scored_count?: number; unscored_count?: number;
                                   unscored_by_kind?: Record<string, number>; coverage?: number }>;
    query_count?: number | null;
  } | null;
  transfer?: TransferInfo | null;
  duplicate_status?: DuplicateStatus | null;
  duplicate_attempt_count?: number | null;
  measurement_group_id?: string | null;
  measurement_identity_confidence?: "full" | "partial" | null;
  measurement_identity_missing_inputs?: string[] | null;
  provenance_source?: string | null;
  checkpoint_training_dataset?: string | null;
  reproduction_status?: string | null;
  interpreter_verification_status?: string | null;
  training_or_transfer_label?: string;
  is_synthetic?: boolean;
  synthetic_fixture?: boolean;
  candidate_topk_exported?: boolean | null;
  /** How this run's `score()` combines features, derived by the backend from the registered adapter's
   *  own source (`einsum` / `einsum-optimized` / `matmul`). Two runs with different implementations
   *  are not bit-comparable at query level, so the table must not rank them as if they were. */
  scoring_impl?: { impl: string; basis?: string; adapter_hash_known?: boolean; scoring_hash_known?: boolean } | null;
  /** Backend `is_failed_summary`: no numbers were measured, so metric cells must blank. */
  is_failed?: boolean;
  metrics: CompositionMetrics;
  smoke?: { limited: boolean; query_limit?: number | null; label?: string | null };
};
export type GalleryTopResult = {
  rank: number;
  image_id: string;
  /** Position in the full released gallery: a different fact from `rank`, keep both. */
  gallery_position?: number | null;
  /** null when the artifact stored an order but no scores — render "—", never 0.0. */
  score?: number | null;
};
export type QueryState = "unscored" | "retrieval_failure" | "hit";
export type CompositionQuery = {
  query_id: string;
  category: string | null;
  query_txt: string;
  reference_id: string;
  positive_ids: string[];
  ranks: Record<string, number | null>;
  target_rank: number | null;
  outcome: "composition_required" | "text_only" | "image_only" | "both" | "unresolved" | "unscored";
  failure?: string | null;
  top_results?: (TopResult | GalleryTopResult)[];
  query_state?: QueryState;
  unscored?: boolean;
  unscored_kind?: string | null;
  retrieval_failure?: boolean;
  candidate_status?: string | null;
  candidate_reason?: string | null;
  candidate_depth?: number | null;
};
