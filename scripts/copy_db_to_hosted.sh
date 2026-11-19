#!/bin/sh
# Copy the local database (index, readings, labels, users) into an empty hosted Postgres.
# Reads HOSTED_DATABASE_URL from .env so the connection string never appears on the command line or in output.
set -eu
cd "$(dirname "$0")/.."
HOSTED_DATABASE_URL=$(grep -E '^HOSTED_DATABASE_URL=' .env | head -1 | cut -d= -f2- | sed -e 's/^"//' -e 's/"$//')
[ -n "$HOSTED_DATABASE_URL" ] || { echo "Add HOSTED_DATABASE_URL=... to .env first"; exit 1; }
LOCAL=${LOCAL_DATABASE:-canapply}

echo "Hosted server: $(psql "$HOSTED_DATABASE_URL" -Atc 'show server_version')"
existing=$(psql "$HOSTED_DATABASE_URL" -Atc "select count(*) from information_schema.tables where table_schema='public'")
[ "$existing" = "0" ] || { echo "Hosted database already has $existing tables. Refusing to overwrite."; exit 1; }

psql "$HOSTED_DATABASE_URL" -qc 'create extension if not exists vector'
echo "Copying (about 450 MB, a few minutes)..."
pg_dump --no-owner --no-privileges --format=custom "$LOCAL" \
  | pg_restore --no-owner --no-privileges --exit-on-error --dbname="$HOSTED_DATABASE_URL" 2>&1 \
  | grep -v 'extension "vector" already exists' || true

psql "$HOSTED_DATABASE_URL" -Atc "select 'open jobs: ' || count(*) filter (where closed_at is null) || ', embedded: ' || count(embedding) from jobs; select 'readings: ' || count(*) from requirements; select 'size: ' || pg_size_pretty(pg_database_size(current_database()))"
