# syntax=docker/dockerfile:1
#
# pbisim-app container image.
#
# The private engine `pbisim` is installed from its private GitHub repo using a
# READ-ONLY token supplied at build time. The token lives only in the *builder*
# stage (a build ARG), so it never appears in the final image's layers or
# `docker history`. NEVER hard-code the token here or commit it — set it as a
# secret environment variable on the host (see DEPLOY.md).
#
# Build args:
#   PBISIM_TOKEN  fine-grained GitHub PAT with read-only access to the pbisim repo
#   PBISIM_REF    git ref/tag/SHA of pbisim to install (default: main).
#                 Change it to force a fresh engine pull (busts the pip layer cache).

########################  builder  ########################
FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# --- 1. Private engine (cached until PBISIM_REF changes) ---
ARG PBISIM_TOKEN
ARG PBISIM_REF=main
RUN test -n "$PBISIM_TOKEN" || (echo "ERROR: PBISIM_TOKEN build arg is required" >&2; exit 1) \
    && pip install "git+https://${PBISIM_TOKEN}@github.com/phage-therapy-sim/pbisim.git@${PBISIM_REF}"

# --- 2. App source + remaining deps (editable so prompts/ resolves at runtime) ---
WORKDIR /app
COPY . /app
RUN pip install -e .

# --- 3. Scrub any token left in VCS install metadata ---
RUN find /opt/venv -name direct_url.json -delete || true

########################  runtime  ########################
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    # headless matplotlib backend (belt-and-suspenders with matplotlib.use in app.py)
    MPLBACKEND=Agg \
    # cap native (BLAS/OMP) thread pools — multi-threaded BLAS forked by joblib on a
    # small shared instance is a common SIGSEGV (exit 139) source
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

RUN useradd --create-home --uid 10001 appuser

# venv + source (the editable install links pbisim_app -> /app)
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

WORKDIR /app
USER appuser

# Render/most hosts inject $PORT; default to 8501 for local `docker run`.
ENV PORT=8501
EXPOSE 8501

CMD streamlit run pbisim_app/app.py \
    --server.port "${PORT}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
