import type { CompositionQuery } from "./types";
import { CONDITIONS } from "./routes";

export interface ConditionReading {
  id: string;
  short: string;
  rank: number | null;
}

/** The three condition readings for one query, in protocol order.
 *
 *  Extracted so the labels can never drift from the ids the API uses, and so a null rank is
 *  distinguishable from a rank of 0 (the API uses `null` for "no exact rank"; rank 0 is impossible).
 */
export function conditionsOf(query: Pick<CompositionQuery, "ranks">): ConditionReading[] {
  const short: Record<string, string> = {
    multimodal: "Multimodal (MM)",
    text_only_black_image: "Text-only",
    image_only_empty_text: "Image-only",
  };
  return CONDITIONS.map((condition) => {
    const raw = query.ranks?.[condition.id];
    return {
      id: condition.id,
      short: short[condition.id] ?? condition.label,
      rank: typeof raw === "number" && raw > 0 ? raw : null,
    };
  });
}

export type Tone = "ok" | "warn" | "bad" | "info" | "neutral";

export function outcomeTone(outcome: string): Tone {
  switch (outcome) {
    case "composition_required": return "ok";
    case "text_only":
    case "image_only": return "info";
    case "both": return "warn";
    case "unscored": return "neutral";
    default: return "warn";
  }
}

/** Why a per-condition rank is what it is, in one sentence. Used both in the inspector and in the
 *  evaluation table so the two cannot give different accounts of the same number. */
export function explainRank(rank: number | null, cutoff: number): string {
  if (rank === null) return "no exact rank: pipeline/mapping failure, not a retrieval failure";
  return rank <= cutoff ? `inside @${cutoff}` : `outside @${cutoff}`;
}
