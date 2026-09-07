FROM python:3.10-slim

WORKDIR /app

#Install system dependencies required for C-based ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

#Expose FastAPI default port
EXPOSE 8000

#Launch the REST API
CMD ["uvicorn", "src.serving.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]