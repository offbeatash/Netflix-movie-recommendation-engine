FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .

#Expose FastAPI default port
EXPOSE 8000

#Launch the REST API
CMD ["uvicorn", "src.serving.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
