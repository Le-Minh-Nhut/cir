export const protocolDefinitions = [
  { id: "fashioniq_original_split", label: "FashionIQ - Original Split (Full Gallery, Reference Included)", defaultVisible: true },
  { id: "fashioniq_full_gallery_ref_excluded", label: "FashionIQ - Original Full Gallery, Reference Excluded", defaultVisible: false },
  { id: "fashioniq_val_split", label: "FashionIQ - Val Split (Pair-Union Gallery, Reference Excluded)", defaultVisible: true },
] as const;

export type ProtocolId = (typeof protocolDefinitions)[number]["id"];
export type ProtocolVisibility = Record<ProtocolId, boolean>;
export const protocolVisibilityStorageKey = "cir-workbench.protocolVisibility";

export function defaultProtocolVisibility(): ProtocolVisibility {
  return Object.fromEntries(protocolDefinitions.map(({ id, defaultVisible }) => [id, defaultVisible])) as ProtocolVisibility;
}

export function parseProtocolVisibility(value: string | null): ProtocolVisibility {
  const defaults = defaultProtocolVisibility();
  if (value === null) return defaults;
  try {
    const saved: unknown = JSON.parse(value);
    if (typeof saved !== "object" || saved === null || Array.isArray(saved)) return defaults;
    for (const { id } of protocolDefinitions) {
      if (typeof (saved as Record<string, unknown>)[id] === "boolean") defaults[id] = (saved as Record<string, boolean>)[id];
    }
  } catch {
    return defaults;
  }
  return Object.values(defaults).some(Boolean) ? defaults : defaultProtocolVisibility();
}

export function setProtocolVisibility(visibility: ProtocolVisibility, id: ProtocolId, visible: boolean): ProtocolVisibility {
  if (!visible && visibleProtocolIds(visibility).length === 1) return visibility;
  return { ...visibility, [id]: visible };
}

export function selectedVisibleProtocol(protocol: ProtocolId, visibility: ProtocolVisibility): ProtocolId {
  return visibility[protocol] ? protocol : visibleProtocolIds(visibility)[0];
}

export function visibleProtocolIds(visibility: ProtocolVisibility): ProtocolId[] {
  return protocolDefinitions.filter(({ id }) => visibility[id]).map(({ id }) => id);
}
