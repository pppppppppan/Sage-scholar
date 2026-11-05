# syntax=docker/dockerfile:1
FROM node:24.21.0-bookworm-slim AS mcp-node
WORKDIR /opt/mcp
COPY deploy/mcp/package.json deploy/mcp/package-lock.json ./
RUN npm ci --omit=dev --no-audit --no-fund

FROM python:3.12.12-slim-bookworm AS builder
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libxml2-dev libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY requirements*.txt ./
COPY deploy/backend-constraints.txt ./constraints.txt
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
    && /opt/venv/bin/pip install --no-cache-dir -c constraints.txt -r requirements.txt -r requirements-observability.txt paper-search-mcp==0.1.4
ARG INSTALL_VISUAL=false
RUN case "$INSTALL_VISUAL" in \
      true) /opt/venv/bin/pip install --no-cache-dir -c constraints.txt -r requirements-visual.txt ;; \
      false) ;; *) echo 'INSTALL_VISUAL must be true or false'; exit 1 ;; \
    esac && /opt/venv/bin/pip check
# fetch needs httpx<0.28; pdfmux also owns its own dependency tree. No runtime pip/uvx installs.
RUN python -m venv /opt/fetch \
    && /opt/fetch/bin/pip install --no-cache-dir 'mcp-server-fetch==2025.4.7' 'mcp==1.30.0' \
    && /opt/fetch/bin/pip check \
    && python -m venv /opt/pdfmux \
    && /opt/pdfmux/bin/pip install --no-cache-dir 'pdfmux[serve]==1.8.7' 'mcp==1.30.0' \
    && /opt/pdfmux/bin/pip check

FROM python:3.12.12-slim-bookworm AS runtime
# ps is used by MCP descendant cleanup; poppler/tesseract serve PDF text/OCR helpers.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates libgomp1 libglib2.0-0 libgl1 libxml2 libxslt1.1 procps poppler-utils tesseract-ocr \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 sage
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/fetch /opt/fetch
COPY --from=builder /opt/pdfmux /opt/pdfmux
COPY --from=mcp-node /usr/local/bin/node /usr/local/bin/node
COPY --from=mcp-node /opt/mcp /opt/mcp
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/sage HF_HOME=/home/sage/.cache/huggingface XDG_CACHE_HOME=/home/sage/.cache
WORKDIR /app
COPY --chown=sage:sage sage_scholar/ sage_scholar/
COPY --chown=sage:sage skills/ skills/
COPY --chown=sage:sage configs/agents.json deploy/configs/agents.json
COPY --chown=sage:sage deploy/configs/mcp_servers.json deploy/configs/mcp_servers.json
COPY --chown=sage:sage deploy/entrypoint.py deploy/prepare_models.py deploy/
RUN mkdir -p /app/data/tmp /app/logs /home/sage/.cache/huggingface \
    && chown -R sage:sage /app/data /app/logs /home/sage/.cache
USER sage
EXPOSE 8000
CMD ["python", "-m", "deploy.entrypoint"]
