FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy FASTEMBED_CACHE_PATH=/app/.models
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.12.6 /uv /usr/local/bin/uv

# Dependencies first, so code changes do not reinstall them.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY canapply ./canapply
COPY data ./data
RUN uv sync --frozen --no-dev

# Bake the embedding model into the image so the first search does not wait on a download.
RUN uv run python -c "from canapply import embedder; embedder.model()"

EXPOSE 8080
CMD ["uv", "run", "--no-sync", "python", "-m", "canapply.serve"]
