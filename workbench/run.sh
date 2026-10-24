#!/usr/bin/env sh
# Launch a workbench CLI with the VERIFIED controller interpreter.
#
# `python3` from PATH is a guess: on this project's own host it resolved to an interpreter without
# FastAPI while the verified environment sat in `./.venv`, and the operator had to arrange PATH
# before every invocation -- which a documented "one command" route cannot assume. The repository's
# own environment is therefore preferred when it exists, and PATH remains the fallback for a host
# whose controller environment is arranged some other way. Model interpreters are NEVER taken from
# here: each model's own environment is resolved per model from the registry.
set -e
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
if [ -x "$root/.venv/bin/python" ]; then
    python="$root/.venv/bin/python"
else
    python=python3
fi
if [ "${1-}" = "autopilot" ]; then
    shift
    exec "$python" "$root/workbench/scripts/autopilot.py" "$@"
fi
exec "$python" "$root/workbench/scripts/pipeline.py" "$@"
