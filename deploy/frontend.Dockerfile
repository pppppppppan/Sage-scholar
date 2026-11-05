# syntax=docker/dockerfile:1
FROM node:24.21.0-bookworm-slim AS dependencies
WORKDIR /app
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund

FROM dependencies AS builder
ENV NEXT_TELEMETRY_DISABLED=1
# Empty means same-origin /api in browser code. Never embed Docker DNS in the browser.
ENV NEXT_PUBLIC_API_BASE=""
COPY web/ ./
RUN mkdir -p public && npm run build

FROM node:24.21.0-bookworm-slim AS runtime
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1 HOSTNAME=0.0.0.0 PORT=3000
WORKDIR /app
COPY --from=builder --chown=node:node /app/.next/standalone ./
COPY --from=builder --chown=node:node /app/.next/static ./.next/static
COPY --from=builder --chown=node:node /app/public ./public
USER node
EXPOSE 3000
CMD ["node", "server.js"]
