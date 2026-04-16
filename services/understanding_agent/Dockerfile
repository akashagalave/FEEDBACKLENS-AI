FROM python:3.11-slim

WORKDIR /app

COPY services/understanding-agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/understanding-agent/app ./app
COPY shared ./shared

EXPOSE 8002

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]