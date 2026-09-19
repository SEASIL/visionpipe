# CPU inference image: no PyTorch/Ultralytics, so it stays small. Uses ONNX Runtime + OpenCV.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY configs ./configs
COPY scripts ./scripts
ENTRYPOINT ["python", "-m", "visionpipe"]
CMD ["--help"]
