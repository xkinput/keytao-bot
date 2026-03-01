#!/usr/bin/env bash
set -e

NAPCAT_UID=$(id -u) NAPCAT_GID=$(id -g) docker compose up -d --build

echo ""
echo "Services started:"
echo "  NapCat WebUI  → http://localhost:6099/webui  (token: napcat)"
echo "  Bot           → running in background"
echo ""
echo "Run 'docker compose logs -f' to follow logs."
