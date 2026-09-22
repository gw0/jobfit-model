# One Python image for every Argo pipeline stage (specs §8); stages differ only in command.
FROM python:3.13-slim

WORKDIR /app

COPY pipeline/requirements.txt pipeline/requirements.txt
RUN pip install --no-cache-dir -r pipeline/requirements.txt \
    && python -m spacy download en_core_web_sm

# The image has no .git; `make build` passes the commit so reports record it.
ARG GIT_SHA=unknown
ENV GIT_SHA=$GIT_SHA

# data/ provides corpus.py and pii_scrub.py, which pipeline/ imports.
COPY aspects.json aspects.json
COPY data/ data/
COPY pipeline/ pipeline/
