#!/usr/bin/env sh
if [ "${1-}" = "autopilot" ]; then
    shift
    exec python3 "$(dirname "$0")/scripts/autopilot.py" "$@"
fi
exec python3 "$(dirname "$0")/scripts/pipeline.py" "$@"
