#!/usr/bin/env bash
set -e

docker compose down --rmi local --volumes --remove-orphans

echo "All containers, local images and anonymous volumes removed."
