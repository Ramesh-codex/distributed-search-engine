# Single image, reused for every service in docker-compose.yml (shards,
# coordinator, and the standalone single-node api) -- each container just
# overrides `command` to point uvicorn at a different ASGI app.
FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

ENV PYTHONPATH=/app/src
EXPOSE 8000

CMD ["uvicorn", "search.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
