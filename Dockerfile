FROM python:3.11-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --prefix=/install --no-cache-dir -r requirements.txt

FROM python:3.11-slim AS runtime
LABEL org.opencontainers.image.title="HVAC RL"
WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .
RUN mkdir -p models logs plots experiments
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
