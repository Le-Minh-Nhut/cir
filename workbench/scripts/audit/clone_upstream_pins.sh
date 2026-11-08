#!/usr/bin/env bash
# Clone each pinned upstream repo into an audit root at its exact commit, detached.
# Usage: clone_upstream_pins.sh <audit_root> [pins_tsv]
# pins_tsv lines: name<TAB>url<TAB>full_sha
# Never substitutes upstream HEAD for the pinned revision: if the clone lacks the SHA it fetches
# that SHA explicitly, and reports SHA_ABSENT rather than checking out something else.
set -uo pipefail
AUD="${1:?audit root}"
PINS="${2:-/tmp/pins.tsv}"
mkdir -p "$AUD"
log="$AUD/clone.log"
: > "$log"

clone_one() {
  name="$1"; url="$2"; sha="$3"
  dest="$AUD/$name"
  echo "=== $name $url $sha" >>"$log"
  if [ -d "$dest/.git" ]; then echo "SKIP_EXISTS $name" >>"$log"; return 0; fi
  if ! git clone --no-checkout --filter=blob:none --quiet "$url" "$dest" >>"$log" 2>&1; then
    echo "CLONE_FAILED $name" >>"$log"; return 1
  fi
  cd "$dest" || return 1
  git cat-file -e "${sha}^{commit}" 2>/dev/null || git fetch --quiet origin "$sha" >>"$log" 2>&1
  if ! git cat-file -e "${sha}^{commit}" 2>/dev/null; then
    echo "SHA_ABSENT $name $sha" >>"$log"; return 1
  fi
  if git checkout --quiet --detach "$sha" >>"$log" 2>&1; then
    echo "CHECKOUT_OK $name $(git rev-parse HEAD)" >>"$log"
  else
    echo "CHECKOUT_FAILED $name" >>"$log"; return 1
  fi
  cd "$AUD" || return 1
}

running=0
while IFS=$'\t' read -r name url sha; do
  [ -z "${name:-}" ] && continue
  clone_one "$name" "$url" "$sha" &
  running=$((running+1))
  if [ "$running" -ge 4 ]; then wait -n; running=$((running-1)); fi
done <"$PINS"
wait
grep -E '^(===|CLONE_FAILED|SHA_ABSENT|CHECKOUT_OK|CHECKOUT_FAILED|SKIP_EXISTS)' "$log"
