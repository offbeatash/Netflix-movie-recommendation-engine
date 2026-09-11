FROM python:3.13-slim

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

EXPOSE 8000
CMD ["uvicorn", "src.serving.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
