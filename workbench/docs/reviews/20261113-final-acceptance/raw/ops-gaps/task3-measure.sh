#!/bin/bash
# Task 3: isolated stop measurement, one pass, no collisions. Spare ports only.
# Reports: exit time, child survival, port release, temp/lock leftovers, restart-on-same-port.
set -u
REPO=/home/heheboiz/data/cir
PY=$REPO/workbench/envs/ctl312h/bin/python
NPM=/home/heheboiz/.nvm/versions/node/v25.9.0/bin/npm
export CIR_DATA_ROOT=$REPO/data CIR_REPO_ROOT=$REPO
export FASHIONIQ_ROOT=$REPO/data/FashionIQ
export WORKBENCH_ARTIFACTS_ROOT=$REPO/workbench/artifacts
export WORKBENCH_CHECKPOINT_ROOT=$REPO/workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads
export WORKBENCH_CIRCUS_RELEASE_ROOT=$REPO/workbench/artifacts/circus_release
export WORKBENCH_RESULTS_ROOT=$REPO/workbench/artifacts/results
export WORKBENCH_THIRD_PARTY_ROOT=$REPO/workbench/third_party

owner() { ss -ltnpH "sport = :$1" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2; }
wait_free() { for _ in $(seq 1 400); do [ -z "$(owner "$1")" ] && return 0; sleep 0.05; done; return 1; }
wait_listen() { for _ in $(seq 1 1200); do p=$(owner "$1"); [ -n "$p" ] && { echo "$p"; return 0; }; sleep 0.05; done; return 1; }
kids() { pgrep -P "$1" 2>/dev/null; }

measure() { # name port signal cmd...
  local name=$1 port=$2 sig=$3; shift 3
  wait_free "$port" || { echo "$name PORT_BUSY"; return; }
  setsid "$@" >/tmp/opsgaps/$name.log 2>&1 </dev/null &
  local launcher=$!
  local listen_pid; listen_pid=$(wait_listen "$port") || { echo "$name NEVER_READY"; kill -9 -$launcher 2>/dev/null; return; }
  local child_list; child_list=$(kids "$launcher" | tr '\n' ' ')
  local group; group=$(ps -o pgid= -p "$launcher" | tr -d ' ')
  local start; start=$(date +%s%N)
  kill -"$sig" "$launcher"
  for _ in $(seq 1 2400); do kill -0 "$launcher" 2>/dev/null || break; sleep 0.02; done
  local stop_ns=$(( ($(date +%s%N) - start) / 1000000 ))
  local alive_children=""; for p in $child_list; do kill -0 "$p" 2>/dev/null && alive_children="$alive_children $p"; done
  local port_after; port_after=$(owner "$port")
  # Same-port restart, which is what Restart=always + RestartSec tries 2 s later.
  local restart_ok=NO
  if wait_free "$port"; then
    setsid "$@" >/tmp/opsgaps/$name.restart.log 2>&1 </dev/null &
    local rl=$!
    wait_listen "$port" >/dev/null && restart_ok=YES
    kill -TERM "$rl" 2>/dev/null; sleep 0.5; kill -9 -"$rl" 2>/dev/null
    wait_free "$port" >/dev/null
  fi
  kill -9 -"$group" 2>/dev/null; kill -9 "$launcher" 2>/dev/null
  for p in $child_list; do kill -9 "$p" 2>/dev/null; done
  sleep 0.3
  echo "$name sig=$sig listen_pid=$listen_pid stop_ms=$stop_ns children_at_signal=[$child_list] children_surviving=[$alive_children] port_owner_after=${port_after:-none} restart_same_port=$restart_ok"
}

echo "== backend :8795 =="
measure be-term 8795 TERM $PY -m uvicorn workbench.backend.main:app --host 127.0.0.1 --port 8795
measure be-int  8795 INT  $PY -m uvicorn workbench.backend.main:app --host 127.0.0.1 --port 8795
echo "== frontend :8796 =="
( cd $REPO/workbench/frontend && export PATH=/home/heheboiz/.nvm/versions/node/v25.9.0/bin:/usr/local/bin:/usr/bin:/bin WORKBENCH_BACKEND_URL=http://127.0.0.1:8000 && measure fe-term 8796 TERM $NPM run preview -- --host 127.0.0.1 --port 8796 --strictPort )
( cd $REPO/workbench/frontend && export PATH=/home/heheboiz/.nvm/versions/node/v25.9.0/bin:/usr/local/bin:/usr/bin:/bin WORKBENCH_BACKEND_URL=http://127.0.0.1:8000 && measure fe-int 8796 INT $NPM run preview -- --host 127.0.0.1 --port 8796 --strictPort )
echo "== cleanup proof =="
ss -ltnp 2>/dev/null | grep -E ':(8795|8796) ' || echo "ports 8795/8796 free"
pgrep -af "port 8795|port 8796" | grep -v pgrep || echo "no processes on spare ports"
