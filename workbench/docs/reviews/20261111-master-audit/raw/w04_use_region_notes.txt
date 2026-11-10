import { useCallback, useEffect, useRef, useState } from "react";
import { RequestRunner, beginRegion, failRegion, idleRegion, settleRegion, type Region } from "./requests";

/** Load one region's data, keyed by the context that determines it.
 *
 *  Every fetch in this UI goes through here, so no component can accidentally apply an answer for
 *  a context the operator has left: `RequestRunner` decides (token AND context key), and the region
 *  it produces has an explicit `status` so the component can render loading / error / ready for the
 *  region it belongs to rather than the whole page.
 *
 *  `fetcher` may be null to mean "this context needs no request" (for example, no run selected);
 *  the region then reports `idle` and holds no data — which is a different statement from an error.
 */
export interface RegionHandle<T> {
  region: Region<T>;
  retry: () => void;
}

export function useRegion<T>(
  family: string,
  key: string,
  fetcher: ((signal: AbortSignal) => Promise<T>) | null,
): RegionHandle<T> {
  const [region, setRegion] = useState<Region<T>>(() => idleRegion<T>(key));
  const [nonce, setNonce] = useState(0);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;
  const runnerRef = useRef<RequestRunner | null>(null);
  if (runnerRef.current === null) {
    runnerRef.current = new RequestRunner(family, {
      commit: (committedKey, value) => {
        setRegion((previous) => settleRegion(previous, committedKey, value as T));
      },
      onError: (failedKey, message) => {
        setRegion((previous) => failRegion(previous, failedKey, message));
      },
    });
  }

  useEffect(() => {
    const runner = runnerRef.current!;
    const active = fetcherRef.current;
    if (!active) {
      runner.reset(key);
      setRegion((previous) => (previous.key === key && previous.status === "idle" && previous.data === null
        ? previous
        : idleRegion<T>(key)));
      return;
    }
    setRegion((previous) => beginRegion(previous, key));
    void runner.run(key, active);
    return () => {
      // Leaving the component (or changing key) aborts the bytes in flight. It is a cost
      // optimization only: correctness is the token+key check inside the runner.
      runner.abortPending();
    };
  }, [key, nonce]);

  const retry = useCallback(() => setNonce((value) => value + 1), []);
  return { region, retry };
}

/** Abortable JSON fetch. A non-2xx answer becomes a thrown Error carrying the API's own message, so
 *  the region shows what the server said instead of a generic failure. */
export async function fetchJson<T>(path: string, signal: AbortSignal): Promise<T> {
  const response = await fetch(path, { signal });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const message = payload?.message || payload?.detail?.message || payload?.error
      || JSON.stringify(payload) || `${response.status} ${response.statusText}`;
    throw new Error(message);
  }
  return payload as T;
}
