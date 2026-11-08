/** One fetch family, ordered by explicit context identity.
 *
 *  The defect this module exists to prevent: an in-flight response arriving AFTER the operator
 *  changed the thing it was about, and painting the previous context's data under the new
 *  context's labels. The UI showed subset SF while the pager and cards were Full's, and a
 *  Full→SF switch on a slow network could leave a stale answer on screen indefinitely.
 *
 *  A boolean is not enough. `loading = true` cannot tell "the response for the context I asked
 *  about" from "a response for a context I have since left", and it cannot order two responses
 *  that both belong to the same component. So each request carries:
 *
 *  * a **generation token** — a monotonically increasing number minted per request in the family;
 *  * a **context key** — the serialized identity of the data asked for (protocol, subset, run,
 *    condition, filters, sort, page, ...).
 *
 *  A response is applied only when it is both the newest request in its family AND its context key
 *  still equals the live key. The two conditions are separate because they fail differently: a
 *  newest-token-with-changed-key answer is a response to a question nobody is asking any more,
 *  and an old-token-with-same-key answer is a response whose *data* is still right but whose
 *  arrival is out of order (a newer request for the same context is still pending).
 *
 *  `AbortController` is used at the network layer as a cost optimization: it stops bytes for a
 *  context the operator has left. It is NOT the correctness mechanism — a response can already be
 *  parsed when the context changes — so the token+key guard is what decides, and it is what the
 *  tests in `workbench/tests/test_frontend_requests.py` execute.
 *
 *  Pure, dependency-free and transport-agnostic: the runner takes a `Fetcher`, so a test can
 *  force the reversed order (older resolves after newer) and the newer-request-rejects case
 *  deterministically, with no network and no browser.
 */

export type ContextKey = string;

/** Serialize a context identity. Field order is fixed by the caller's literal, and the separator
 *  is a character that cannot appear in any id, so no two different contexts collide. */
export function contextKey(parts: Record<string, string | number | boolean | null | undefined>): ContextKey {
  return Object.keys(parts)
    .sort()
    .map((name) => `${name}=${parts[name] === null || parts[name] === undefined ? "" : String(parts[name])}`)
    .join("\u001f");
}

/** The identity of a request to whichever fetch family is being guarded, for the two-field case
 *  (protocol + active run) that the explorer needed first.
 *
 *  It delegates to `contextKey` on purpose: two key builders in one codebase are two separator rules
 *  that can drift apart, and a drifted separator is exactly the collision the encoding prevents. The
 *  two-field form exists because the EMPTY case is the subtle one: when the newly selected protocol
 *  has no compatible run the run id is `""`, and a guard written as "is there still a run?" cannot
 *  tell "this run is gone" from "there was never a run here". Both must drop the previous answer, so
 *  the identity is a key and not a run id.
 */
export function requestKey(protocol: string, runId: string): ContextKey {
  return contextKey({ protocol, run: runId });
}

export interface RequestTicket {
  /** Family name, for the region that owns the state. */
  family: string;
  token: number;
  key: ContextKey;
}

export interface FamilyState {
  token: number;
  key: ContextKey;
}

/** Mint the next ticket in a family and make it the live one. */
export function openRequest(family: FamilyState, key: ContextKey): RequestTicket {
  family.token += 1;
  family.key = key;
  return { family: "", token: family.token, key };
}

/** Whether an answer may be applied to the live state of its own family. */
export function isCurrent(ticket: RequestTicket, family: FamilyState): boolean {
  return ticket.token === family.token && ticket.key === family.key;
}

/** The lifecycle of one region's data. Every read of a region's data goes through a `Region`. */
export type RegionStatus = "idle" | "loading" | "ready" | "error";

export interface Region<T> {
  key: ContextKey;
  status: RegionStatus;
  data: T | null;
  error: string | null;
  /** True when `data` belongs to `key`; false when it is the previous context's data, which must
   *  never be displayed as if it described the current context. */
  fresh: boolean;
}

export function idleRegion<T>(key: ContextKey): Region<T> {
  return { key, status: "idle", data: null, error: null, fresh: true };
}

/** Loading a NEW context clears the previous context's data. Keeping it visible is exactly the
 *  bug: a failed request would then leave another context's rows under the new labels. */
export function beginRegion<T>(previous: Region<T>, key: ContextKey): Region<T> {
  if (previous.key === key && previous.status === "ready") return previous;
  return { key, status: "loading", data: previous.key === key ? previous.data : null, error: null, fresh: true };
}

export function settleRegion<T>(previous: Region<T>, key: ContextKey, data: T): Region<T> {
  if (previous.key !== key) return previous;
  return { key, status: "ready", data, error: null, fresh: true };
}

export function failRegion<T>(previous: Region<T>, key: ContextKey, message: string): Region<T> {
  if (previous.key !== key) return previous;
  return { key, status: "error", data: null, error: message, fresh: true };
}

export type Fetcher<T> = (signal: AbortSignal) => Promise<T>;

export interface RunnerOptions {
  /** Called with every applied value. Never called for a superseded request. */
  commit: (key: ContextKey, value: unknown) => void;
  onError?: (key: ContextKey, message: string) => void;
  /** Injected for tests; defaults to the platform implementation. */
  controller?: () => AbortController;
  /** Called when a request is superseded, so a test can count real aborts. */
  onAbort?: (ticket: RequestTicket) => void;
}

/** A single-family request runner. One instance per region, so no two regions can cancel each
 *  other's legitimate answer (a shared counter is data loss dressed as safety). */
export class RequestRunner {
  private live: FamilyState = { token: 0, key: "" };
  private inFlight: AbortController | null = null;
  private readonly opts: RunnerOptions;

  constructor(family: string, options: RunnerOptions) {
    this.family = family;
    this.opts = options;
  }

  readonly family: string;

  /** The key of the newest request. A region renders from its own state, but a caller that
   *  re-renders after the fact (an async callback) can read this to label what it holds. */
  get currentKey(): ContextKey {
    return this.live.key;
  }

  /** Supersede whatever is pending and start a request for `key`. */
  run<T>(key: ContextKey, fetch: Fetcher<T>): Promise<T | null> {
    this.abortPending();
    const ticket = openRequest(this.live, key);
    const controller = (this.opts.controller ?? (() => new AbortController))();
    this.inFlight = controller;
    let settled = false;
    const finish = () => {
      if (!settled) {
        settled = true;
        if (this.inFlight === controller) this.inFlight = null;
      }
    };
    return fetch(controller.signal)
      .then((value) => {
        finish();
        if (!isCurrent(ticket, this.live)) return null;
        this.opts.commit(ticket.key, value);
        return value;
      })
      .catch((error: unknown) => {
        finish();
        if (controller.signal.aborted) return null;
        if (!isCurrent(ticket, this.live)) return null;
        this.opts.onError?.(ticket.key, error instanceof Error ? error.message : String(error));
        return null;
      });
  }

  /** Abort the pending request, if any. The token is NOT advanced: aborting must not make a
   *  legitimate later answer look superseded. */
  abortPending(): void {
    if (!this.inFlight) return;
    const pending = this.inFlight;
    this.inFlight = null;
    pending.abort();
    this.opts.onAbort?.({ family: this.family, token: this.live.token, key: this.live.key });
  }

  /** Abandon the current context without starting a new request. */
  reset(key: ContextKey): void {
    this.abortPending();
    this.live.token += 1;
    this.live.key = key;
  }
}
