#!/bin/bash
export PYTHONPATH=$PWD
echo "[INFO] Iniciando Neural Forge..."
echo "[INFO] API rodando em: http://localhost:8080"

# Roda o servidor
./venv/bin/uvicorn backend.server:app --reload --port 8080 --host 0.0.0.0
