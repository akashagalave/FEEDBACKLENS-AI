FROM python:3.11-slim

WORKDIR /app

COPY services/orchestrator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/orchestrator/app ./app
COPY shared ./shared

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]