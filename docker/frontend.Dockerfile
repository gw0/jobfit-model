# Optional self-hosted alternative to the HF Spaces (Static) deployment (specs §7).
FROM node:22-alpine AS build
WORKDIR /app
# `npm install`, not `npm ci`: the committed lockfile is bun's.
COPY frontend/package.json frontend/bun.lock ./
RUN npm install
COPY frontend/ ./
# src/App.vue imports ../../questions.json, the same file the pipeline reads.
COPY questions.json /questions.json
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY docker/frontend.nginx.conf /etc/nginx/conf.d/default.conf
# Non-root (specs §8): the cache dirs and pid file must be writable by that user.
RUN adduser -D -H -u 1000 static \
    && chown -R static:static /usr/share/nginx/html /var/cache/nginx /etc/nginx/conf.d \
    && sed -i "s#pid .*#pid /tmp/nginx.pid;#" /etc/nginx/nginx.conf
USER static
EXPOSE 8080
