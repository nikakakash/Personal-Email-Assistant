# Web image: build the SPA with Node, serve it (and proxy /api) with Caddy.
FROM node:26-slim AS build
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM caddy:2-alpine
COPY infra/Caddyfile /etc/caddy/Caddyfile
COPY --from=build /web/dist /srv
EXPOSE 8080
