/** When a response may be applied, and to what state it belongs.
 *
 *  Every fetch in the explorer is answered asynchronously, and the operator can change the thing
 *  the request was about while it is in flight. The rule is one sentence: a response may only be
 *  applied while the state it was requested for is still the current state. Anything else -- a
 *  protocol switch, a different active run, a protocol whose run list is now empty -- makes it a
 *  description of something the operator is no longer looking at.
 *
 *  The empty case is why the identity is a key and not a run id. When the newly selected protocol
 *  has no compatible run, the run id is `""`, and a guard written as "is there still a run?" cannot
 *  tell "this run is gone" from "there was never a run here"; the previous request must be dropped
 *  in both. Two dimensions, one key, one comparison.
 *
 *  Pure and dependency-free on purpose: this is the piece a test can execute, so the rule is
 *  verified rather than described.
 */

export function requestKey(protocol: string, runId: string): string {
  // A separator that cannot occur in either field, so no pair of values can collide with another.
  return `${protocol}\u0000${runId}`;
}

export function responseIsCurrent(requestedKey: string, currentKey: string): boolean {
  return requestedKey === currentKey;
}

/** The identity of a request to whichever fetch family is being guarded.
 *
 *  Counters are NOT shared between families. A single counter would let the query page's request
 *  cancel a legitimate in-flight annotation answer, which is data loss rather than safety: the
 *  operator would be told their selection failed when it did not. Each family therefore owns its
 *  own counter, and both conditions -- newest request, unchanged key -- must hold.
 */
export function mayApply(
  requested: { token: number; key: string },
  current: { token: number; key: string },
): boolean {
  return requested.token === current.token && requested.key === current.key;
}
