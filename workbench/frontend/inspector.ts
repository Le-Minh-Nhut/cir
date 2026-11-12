import type { CompositionQuery } from "./types";
import { CONDITIONS } from "./routes";

export interface ConditionReading {
  id: string;
  short: string;
  rank: number | null;
}

const SHORT_LABEL: Record<string, string> = {
  multimodal: "Multimodal (MM)",
  text_only_black_image: "Text-only",
  image_only_empty_text: "Image-only",
  clvc_image_encoder_only: "Image-only (image encoder)",
};

/** The condition readings for one query, over the conditions THAT RUN scored.
 *
 *  Extracted so the labels can never drift from the ids the API uses, and so a null rank is
 *  distinguishable from a rank of 0 (the API uses `null` for "no exact rank"; rank 0 is impossible).
 *
 *  The readings come from the ROW's own `ranks` keys, not from the protocol triple. A run whose
 *  image role is an adapter ablation (`clvc_image_encoder_only`) serves no `image_only_empty_text`
 *  key at all, and rendering the protocol names regardless printed "Chưa chấm" for a condition the
 *  run never measured -- three of them, beside a real target rank. Protocol order first (so a normal
 *  run's order is unchanged), then any extra condition the run actually ranked, in the order the
 *  payload lists it.
 */
export function conditionsOf(query: Pick<CompositionQuery, "ranks">): ConditionReading[] {
  const ranks = query.ranks ?? {};
  const protocolIds: string[] = CONDITIONS.map((condition) => condition.id);
  const extra = Object.keys(ranks).filter((id) => !protocolIds.includes(id));
  return [...protocolIds, ...extra]
    .filter((id) => id in ranks)
    .map((id) => {
      const raw = ranks[id];
      const declared = CONDITIONS.find((condition) => condition.id === (id as typeof condition.id));
      return {
        id,
        short: SHORT_LABEL[id] ?? declared?.label ?? id,
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
