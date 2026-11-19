# syntax=docker/dockerfile:1

# =======================================================
# Stage 1: Builder (installs production dependencies with uv)
# =======================================================
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /uvx /bin/

WORKDIR /app

# Compile bytecode, copy files instead of hardlinking, and use the container's Python 3.13
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=python3.13

# 1. Install third-party dependencies first for layer caching
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# 2. Copy application source and install the project non-editably into .venv
COPY README.md ./
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable


# =======================================================
# Stage 2: Runtime (minimal, non-root image)
# =======================================================
FROM python:3.13-slim AS runtime

WORKDIR /app

# Create a non-root user and a mount point for read-only target repositories (-v /repo:/workspace:ro)
RUN groupadd --system appgroup \
    && useradd --system --gid appgroup --create-home appuser \
    && mkdir -p /workspace \
    && chown appuser:appgroup /app /workspace

# Copy only the built virtualenv and source code from builder (never host .venv or .env)
COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appgroup /app/src /app/src

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER appuser

# Replace with your MCP server entrypoint once ready (e.g. CMD ["python", "-m", "server"])
CMD ["python"]
