FROM python:3.13-slim AS base
WORKDIR /app
COPY pyproject.toml .
COPY active_inference/ active_inference/
RUN pip install --no-cache-dir ".[dev]"

FROM base AS test
COPY tests/ tests/
CMD ["python", "-m", "pytest", "tests/", "-v"]

FROM base AS demo
COPY examples/ examples/
ENTRYPOINT ["python", "-m"]
CMD ["examples.grid_world_demo"]
