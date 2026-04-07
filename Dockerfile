FROM python:3.11-slim

LABEL maintainer="hrshkr"
LABEL description="RCTD Environment — Research Coordination & Truth Discovery"
LABEL openenv="true"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=7860
ENV HOST=0.0.0.0
ENV WORKERS=2

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD uvicorn rctd_env.server.app:app --host 0.0.0.0 --port 7860 --workers 2
