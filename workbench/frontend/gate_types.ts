export type MetricComparison = {
  metric: string;
  paper: number | null;
  reproduced: number | null;
  absolute_error_pp: number | null;
  relative_error_pct: number | null;
  within_tolerance: boolean | null;
  paper_provenance: string;
  note: string;
  direction: "higher" | "lower" | "equal";
  signed_delta_pp: number | null;
};
