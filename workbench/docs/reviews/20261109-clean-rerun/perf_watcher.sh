#!/usr/bin/env bash
# Wait for the serial sweep to exit, confirm the box is idle, then measure decode throughput.
#
# DETACHED ON PURPOSE: the measurement window opens when the sweep exits and may not come again
# this session. Nothing here waits for a human.
#
# FAILS CLOSED. If the box is not idle the watcher writes perf_run.FAILED and exits non-zero
# rather than producing contaminated numbers that look like a baseline.
set -uo pipefail

REPO=/home/heheboiz/data/cir
REVIEW=$REPO/workbench/docs/reviews/20261109-clean-rerun
LOG=$REVIEW/logs/perf_run.log
FAILED=$REVIEW/perf_run.FAILED
PY=$REPO/workbench/envs/ctl312h/bin/python
SAMPLE=$REVIEW/perf_sample_ids.txt
REF_RANKS=$REPO/workbench/artifacts/composition/runs/5dee7425dd46c5f154808aac/ranks/encoder_fashioniq.json

mkdir -p "$(dirname "$LOG")"
rm -f "$FAILED"

log() { printf '%s %s\n' "$(date -Is)" "$*" >>"$LOG"; }

log "watcher start pid=$$ ; waiting for sweep to exit"
log "watching: pgrep -f 'pytest workbench/tests' ; pgrep -f composition_eval"

# 1) wait for BOTH to be empty. No upper bound: this is the whole point of the watcher.
waited=0
while :; do
    pytest_pids=$(pgrep -f 'pytest workbench/tests' || true)
    eval_pids=$(pgrep -f composition_eval || true)
    if [ -z "$pytest_pids" ] && [ -z "$eval_pids" ]; then
        break
    fi
    sleep 20
    waited=$((waited + 20))
    # heartbeat every 5 minutes so the log shows it is alive, not wedged
    if [ $((waited % 300)) -eq 0 ]; then
        log "still waiting (${waited}s); pytest=[${pytest_pids}] eval=[${eval_pids}]"
    fi
done
log "sweep exited after ~${waited}s of waiting"

# 2) let the OS settle, then record the load
sleep 30
LOADAVG=$(cat /proc/loadavg)
log "post-settle loadavg: $LOADAVG"

# 3) re-verify idleness after the settle. Something new could have started.
pytest_pids=$(pgrep -f 'pytest workbench/tests' || true)
eval_pids=$(pgrep -f composition_eval || true)
if [ -n "$pytest_pids" ] || [ -n "$eval_pids" ]; then
    msg="NOT IDLE after the sweep: pytest=[${pytest_pids}] eval=[${eval_pids}]. Measurement skipped so no contaminated baseline is produced. loadavg: $LOADAVG"
    log "$msg"
    printf '%s\n' "$msg" >"$FAILED"
    exit 3
fi

# 4) measure. --allow-busy is deliberately NOT passed: the harness re-checks and must agree.
cd "$REVIEW" || { log "cannot cd to $REVIEW"; echo "cannot cd" >"$FAILED"; exit 4; }
log "starting measurement: encoder, workers 0,2,4,6,8, trials 3"
log "loadavg at measurement start: $LOADAVG"

"$PY" "$REVIEW/perf_bench.py" \
    --model encoder \
    --sample-ids "$SAMPLE" \
    --reference-ranks "$REF_RANKS" \
    --workers 0,2,4,6,8 \
    --trials 3 \
    --out-jsonl "$REVIEW/perf_raw.jsonl" \
    --out-md "$REVIEW/PERF_TABLE.md" >>"$LOG" 2>&1
status=$?
log "perf_bench exit=$status"

if [ $status -ne 0 ]; then
    msg="perf_bench exited $status; see $LOG. No performance numbers were produced, so PERFORMANCE.vi.md must stay NOT DONE."
    printf '%s\n' "$msg" >"$FAILED"
    exit $status
fi

# 5) sanity-check the output actually has steady-state rows before declaring success
if ! grep -q '"kind": "steady"' "$REVIEW/perf_raw.jsonl" 2>/dev/null; then
    msg="perf_bench exited 0 but perf_raw.jsonl has no steady-state rows; treating as failure."
    log "$msg"; printf '%s\n' "$msg" >"$FAILED"; exit 5
fi

log "measurement complete; loadavg after: $(cat /proc/loadavg)"

# 6) GENERATE the document from the raw file rather than leaving it to be typed later.
#    The generator re-checks the evidence floor and refuses thin data, so a partly successful run
#    cannot turn into a document that looks measured.
log "generating PERFORMANCE.vi.md from perf_raw.jsonl"
"$PY" "$REVIEW/gen_performance_doc.py" \
    --raw "$REVIEW/perf_raw.jsonl" \
    --out "$REVIEW/PERFORMANCE.vi.md" \
    --perf-table "$REVIEW/PERF_TABLE.md" >>"$LOG" 2>&1
gen_status=$?
log "gen_performance_doc exit=$gen_status"

if [ $gen_status -ne 0 ]; then
    msg="measurement ran but the document generator refused (exit $gen_status) - the raw file does not meet the evidence floor. PERFORMANCE.vi.md is left as the NOT DONE placeholder. See $LOG."
    printf '%s\n' "$msg" >"$FAILED"
    exit $gen_status
fi

log "outputs: perf_raw.jsonl PERF_TABLE.md PERFORMANCE.vi.md"
exit 0
