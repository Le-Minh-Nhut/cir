# CIR Failure Analysis Workbench — single-container image.
# Backend: uvicorn (python -m). Frontend: built at image build time, served by `vite preview`
# with /api proxied to the backend in the same container.
#
# Heavy external assets (FashionIQ dataset, model checkpoints, third-party sources, run
# artifacts) are NOT baked in: mount them at run time (see docker-compose.yml / run.sh).
FROM python:3.13-slim-bookworm

# Node only builds the frontend. It stays in the image because `vite preview` serves the
# built frontend at runtime and proxies /api to the backend.
FROM node:22-slim AS frontend
WORKDIR /build
COPY workbench/frontend/package.json workbench/frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY workbench/frontend/ ./
RUN npx tsc -b && npx vite build --outDir dist

FROM python:3.13-slim-bookworm
WORKDIR /repo

# Node runtime is needed at runtime too: `vite preview` serves the built frontend and
# proxies /api to the backend in this same container. The frontend stage IS node:22-slim,
# so copying its /usr/local brings the exact node+npm the frontend was built with.
COPY --from=frontend /usr/local /usr/local

# git is required: provenance checks read the repository's own commit state.
# micromamba is the static conda solver the environment planner provisions model
# environments with; without it every GPU-model environment install fails with
# "No conda/mamba/micromamba solver available".
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl bzip2 \
    && curl -fsSL "https://micro.mamba.pm/api/micromamba/linux-64/latest" | tar -xvj -C /opt bin/micromamba \
    && ln -s /opt/bin/micromamba /usr/local/bin/micromamba \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY workbench/ ./workbench/
COPY --from=frontend /build/dist ./workbench/frontend/dist
# `vite preview` serves the built frontend but resolves vite from node_modules.
COPY --from=frontend /build/node_modules ./workbench/frontend/node_modules

# The workbench backend needs: fastapi, uvicorn, duckdb, pyyaml, packaging, numpy, pillow,
# torch/torchvision (CPU wheels suffice for orchestration; model environments are separate).
# Pinned to what the repository is developed against; update with care.
RUN python -m pip install --no-cache-dir \
    "fastapi==0.142.2" "uvicorn==0.54.0" "duckdb==1.5.6" "pyyaml==6.0.3" \
    "pydantic==2.13.5" "packaging==26.3" numpy pillow \
    "torch==2.14.0" "torchvision==0.29.0"

# The compose file runs the container as the operator's uid (WORKBENCH_UID/GID, default
# 1000:1000) so files written into host bind mounts are not owned by root. The runtime user
# needs: a real HOME (user caches resolve under it), a writable micromamba root, and write
# access to node_modules (vite preview creates .vite-temp there).
#
# The micromamba root is a container-local solver cache, NOT a mount point: /artifacts is a
# host bind mount, so a directory baked into it would be shadowed by the mount and its
# ownership would come from the host — leaving an operator whose uid is not 1000 unable to
# create the prefix. It is 0777 inside the image instead, which makes provisioning work for
# any uid the compose file selects.
RUN useradd --create-home --uid 1000 --user-group appuser \
    && mkdir -p /opt/mamba \
    && chmod 0777 /opt/mamba \
    && chown -R 1000:1000 /repo/workbench/frontend/node_modules

ENV HOME=/home/appuser \
    MAMBA_ROOT_PREFIX=/opt/mamba \
    PYTHONUNBUFFERED=1 \
    CIR_REPO_ROOT=/repo \
    CIR_DATA_ROOT=/data \
    FASHIONIQ_ROOT=/data/FashionIQ \
    WORKBENCH_ARTIFACTS_ROOT=/artifacts \
    WORKBENCH_CHECKPOINT_ROOT=/checkpoints \
    WORKBENCH_RESULTS_ROOT=/results \
    WORKBENCH_THIRD_PARTY_ROOT=/third_party \
    WORKBENCH_HOST=0.0.0.0 \
    WORKBENCH_BACKEND_PORT=8000 \
    WORKBENCH_FRONTEND_PORT=5173

# Frontend proxy target: backend in the same container.
ENV WORKBENCH_BACKEND_URL=http://127.0.0.1:8000

EXPOSE 8000 5173

# serve_workbench.py builds the uvicorn command with `python -m uvicorn` and runs the frontend
# via npm preview, so a single process tree serves both halves with one configuration.
CMD ["python", "workbench/scripts/serve_workbench.py", "--production-frontend", "--host", "0.0.0.0"]
