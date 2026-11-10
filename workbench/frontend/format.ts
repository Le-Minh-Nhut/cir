/** Formatting with explicit units.
 *
 *  The served numbers carry three different meanings and the previous UI printed them all with the
 *  same helper: nDCG is a fraction in [0,1], recall is ALREADY a percentage (30.06 means 30.06 %),
 *  coverage is a fraction, and the composition gap is a dimensionless fraction that is not a
 *  percentage-point delta. Printing 30.0645 as "30.06%" is right; printing a fraction as "%" or a
 *  gap as "pp" is not, and a reader cannot tell which happened once the number is on screen.
 *
 *  Every function here names its unit in the output, and `null` renders as an em dash with no fake
 *  zero. Pure; checked by `workbench/tests/test_frontend_format.py`.
 */

const DASH = "—";

export function fmtInt(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  return Math.round(value).toLocaleString("en-US");
}

/** A rank is 1-based; anything else is "unscored", never 0 and never "last". */
export function fmtRank(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "unscored";
  return `#${value.toLocaleString("en-US")}`;
}

export function fmtNdcg(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  return value.toFixed(4);
}

/** A value that is already expressed in percent (recall@K). */
export function fmtPercentValue(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  return `${value.toFixed(digits)}%`;
}

/** A fraction that must be read as a share (coverage, fail fraction). */
export function fmtFractionPercent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  return `${(value * 100).toFixed(digits)}%`;
}

/** Recall@K on the scale the SERVED ARTIFACT states.
 *
 *  W-01: `0.2793` (a fraction) and `27.93` (a percent) are the same measurement, and printing both
 *  with `fmtPercentValue` showed them as 0.28 % and 27.93 % — two different scores for one number.
 *  The unit is a property of the row, not of the caller's guess, so it is a required argument: a
 *  `null` unit renders the raw value with an explicit "đơn vị không ghi" marker instead of a `%`
 *  the artifact never promised.
 */
export function fmtRecall(value: number | null | undefined, unit: "fraction" | "percent" | null | undefined,
                          digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  if (unit === "fraction") return `${(value * 100).toFixed(digits)}%`;
  if (unit === "percent") return `${value.toFixed(digits)}%`;
  return `${value.toFixed(digits)} (đơn vị không ghi)`;
}

/** The composition gap: dimensionless, sign preserved, 3 decimals. */
export function fmtGap(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  return `${value > 0 ? "+" : ""}${value.toFixed(3)}`;
}

/** A signed difference in percentage POINTS, from two percentage-valued numbers. */
export function fmtPp(a: number | null | undefined, b: number | null | undefined, digits = 2): string {
  if (a === null || a === undefined || b === null || b === undefined
      || !Number.isFinite(a) || !Number.isFinite(b)) return DASH;
  const delta = a - b;
  return `${delta > 0 ? "+" : ""}${delta.toFixed(digits)} pp`;
}

/** "scored / total (coverage)" — a metric without its denominator is not a metric. */
export function fmtDenominator(scored: number | null | undefined, total: number | null | undefined): string {
  const scoredText = fmtInt(scored);
  const totalText = fmtInt(total);
  if (scored === null || scored === undefined || total === null || total === undefined) {
    return `${scoredText} / ${totalText}`;
  }
  if (!total) return `${scoredText} / —`;
  return `${scoredText} / ${totalText} (${((scored / total) * 100).toFixed(1)}%)`;
}

export function fmtDateStamp(value: string | null | undefined): string {
  if (!value) return DASH;
  // Run ids are `20261107T054343058424Z_…`: the first 15 characters are the UTC timestamp.
  const match = /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})/.exec(value);
  if (!match) return value;
  const [, y, mo, d, h, mi, s] = match;
  return `${y}-${mo}-${d} ${h}:${mi}:${s}Z`;
}

/** A short, stable identifier for a column header: enough to tell two checkpoints apart, never the
 *  whole digest (long hashes belong in a drawer). */
export function shortId(value: string | null | undefined, length = 12): string {
  if (!value) return DASH;
  return value.length <= length ? value : value.slice(0, length);
}
