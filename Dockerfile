FROM python:3.13-slim

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY data ./data
COPY artifacts ./artifacts
COPY .env.example ./

# Create necessary directories and set permissions
RUN mkdir -p /app/data /app/artifacts && \
    chown -R appuser:appuser /app

EXPOSE 8000

# Switch to non-root user
USER appuser

# Healthcheck using existing health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.serving.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]