/** TypeScript contracts mirroring `workbench/backend/edir_pinpoint_api.py`. */

export type BenchmarkDataset = "edir" | "pinpoint";
export type BenchmarkRegime = "indomain" | "fiq_transfer" | "cirr_transfer";
export type ModelStatus = "ready" | "pending" | "unavailable";

export interface EdirOverallMetrics {
  matched_queries?: number;
  total_queries?: number;
  "Recall@1": number;
  "Recall@5": number;
  "Recall@10": number;
  "Recall@50": number;
  MRR: number;
}

export interface EdirSubcatMetrics extends EdirOverallMetrics {
  count: number;
}

export interface EdirRegimeBlock {
  status: ModelStatus;
  has_predictions: boolean;
  variant?: string | null;
  best_epoch?: number | null;
  best_val_score?: number | null;
  finished_at?: string | null;
  elapsed_sec?: number | null;
  unavailable_reason?: string | null;
  overall?: EdirOverallMetrics | null;
  macro_subcategory?: Partial<EdirOverallMetrics> | null;
  per_subcategory?: Record<string, EdirSubcatMetrics> | null;
}

export interface EdirModelRow {
  key: string;
  run_id: string;
  display: string;
  family: string;
  backbone: string;
  stage0?: Partial<EdirOverallMetrics> | null;
  indomain: EdirRegimeBlock;
  cirr_transfer: EdirRegimeBlock;
  fiq_transfer: EdirRegimeBlock;
  deltas: {
    indomain_minus_cirr_r1: number | null;
    indomain_minus_fiq_r1: number | null;
    indomain_minus_s0_r1: number | null;
  };
}

export interface EdirSummaryResponse {
  dataset: "edir";
  split_tag: string;
  corpus_size: number;
  test_query_count: number;
  primary_metric: "Recall@1";
  completion: {
    completed: Record<string, number>;
    expected: Record<string, number>;
  };
  subcategories: Array<{ id: string; count: number }>;
  models: EdirModelRow[];
}

export interface PinPointSliceMetrics {
  total_queries?: number;
  "mAP@1"?: number;
  "mAP@5"?: number;
  "mAP@10": number;
  "mAP@50"?: number;
  "mAP@10_noNeg"?: number;
  "precision@1"?: number;
  "precision@5"?: number;
  "precision@10"?: number;
  "recall@10"?: number;
  "NegRecall@10"?: number;
  ling_sens_range?: number;
}

export interface PinPointRegimeBlock {
  status: ModelStatus;
  has_predictions: boolean;
  variant?: string | null;
  best_epoch?: number | null;
  best_val_score?: number | null;
  finished_at?: string | null;
  elapsed_sec?: number | null;
  unavailable_reason?: string | null;
  overall?: PinPointSliceMetrics | null;
  per_category?: Record<string, PinPointSliceMetrics> | null;
  slices: Record<string, PinPointSliceMetrics>;
}

export interface PinPointModelRow {
  key: string;
  run_id: string;
  display: string;
  family: string;
  backbone: string;
  stage0?: Partial<PinPointSliceMetrics> | null;
  indomain: PinPointRegimeBlock;
  fiq_transfer: PinPointRegimeBlock;
  slice_deltas: Record<string, number | null>;
}

export interface PinPointSliceMeta {
  id: string;
  label: string;
  description: string;
  count: number;
}

export interface PinPointSummaryResponse {
  dataset: "pinpoint";
  split_tag: string;
  corpus_size: number;
  test_query_count: number;
  primary_metric: "mAP@10";
  completion: {
    completed: Record<string, number>;
    expected: Record<string, number>;
  };
  slices: PinPointSliceMeta[];
  categories: Array<{ id: string; count: number }>;
  published_baselines: Array<{ label: string; map10: number }>;
  models: PinPointModelRow[];
}

export interface BenchmarksOverviewResponse {
  split_tag: string;
  edir: {
    corpus_size: number;
    test_query_count: number;
    completion: { completed: Record<string, number>; expected: Record<string, number> };
    top3: Array<{ key: string; display: string; family: string; recall1: number | null; mrr: number | null }>;
    remove_r1_min: number | null;
    remove_r1_max: number | null;
  };
  pinpoint: {
    corpus_size: number;
    test_query_count: number;
    completion: { completed: Record<string, number>; expected: Record<string, number> };
    top3: Array<{ key: string; display: string; family: string; map10: number | null; p1: number | null }>;
    models_beating_gpt5: number;
    gpt5_reference_map10: number;
  };
  availability: Array<{
    key: string;
    display: string;
    family: string;
    backbone: string;
    edir_indomain: ModelStatus;
    pinpoint_indomain: ModelStatus;
    fiq_to_pinpoint: ModelStatus;
    fiq_variant?: string | null;
    fiq_reason?: string | null;
    cirr_to_edir: ModelStatus;
    cirr_variant?: string | null;
    cirr_reason?: string | null;
    fiq_to_edir: ModelStatus;
  }>;
}

export interface SingleQueryScore {
  primary_score: number;
  best_gt_rank: number | null;
  rr?: number;
  ap10?: number;
  p10?: number;
  neg_recall10?: number;
  hit1: boolean;
  hit10: boolean;
  hit50: boolean;
  top1_image_id: string | null;
  top1_is_gt: boolean;
  top1_is_ref: boolean;
  top1_is_hneg: boolean;
}

export interface BenchmarkQueryListItem {
  query_id: string;
  text: string;
  category: string;
  l1_interest?: string | null;
  ref_image_id: string;
  ref_image_url: string;
  gt_count: number;
  hard_neg_count: number;
  primary: SingleQueryScore;
  compare: SingleQueryScore | null;
  delta: number | null;
}

export interface BenchmarkQueriesResponse {
  dataset: BenchmarkDataset;
  model: string;
  regime: BenchmarkRegime;
  variant?: string | null;
  compare_active: boolean;
  compare_model: string | null;
  compare_regime: BenchmarkRegime | null;
  total_queries: number;
  filtered_total: number;
  offset: number;
  limit: number;
  outcome_counts: Record<string, number>;
  items: BenchmarkQueryListItem[];
}

export interface CandidateImageItem {
  rank: number;
  image_id: string;
  image_url: string;
  is_gt: boolean;
  is_ref: boolean;
  is_hard_neg: boolean;
}

export interface BenchmarkQueryDetailResponse {
  dataset: BenchmarkDataset;
  query: {
    query_id: string;
    text: string;
    category: string;
    l1_interest?: string | null;
    length_category?: string | null;
    is_imperative?: boolean | null;
    is_question?: boolean | null;
    ref_image_id: string;
    ref_image_url: string;
  };
  ground_truths: Array<{
    image_id: string;
    image_url: string;
    primary_rank: number | null;
    compare_rank: number | null;
  }>;
  hard_negatives: Array<{
    image_id: string;
    image_url: string;
    is_ref: boolean;
    primary_rank: number | null;
    compare_rank: number | null;
  }>;
  primary: {
    model: string;
    display: string;
    regime: BenchmarkRegime;
    variant?: string | null;
    metrics: SingleQueryScore;
    candidates: CandidateImageItem[];
  };
  compare: {
    model: string;
    display: string;
    regime: BenchmarkRegime;
    variant?: string | null;
    metrics: SingleQueryScore;
    candidates: CandidateImageItem[];
  } | null;
  all_models: Array<{
    model: string;
    display: string;
    family: string;
    regime: BenchmarkRegime;
    primary_score: number;
    best_gt_rank: number | null;
    hit1: boolean;
    hit10: boolean;
    top1_is_ref: boolean;
    top1_is_hneg: boolean;
  }>;
}

export interface InspectorSeed {
  dataset: BenchmarkDataset;
  model?: string;
  regime?: BenchmarkRegime;
  category?: string;
  slice?: string;
  compareModel?: string;
  compareRegime?: BenchmarkRegime;
  outcome?: string;
}
