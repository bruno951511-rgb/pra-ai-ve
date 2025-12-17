#!/bin/bash
echo "[INFO] Configurando ambiente Linux (Neural Forge)..."

# 1. Cria o ambiente virtual se não existir
if [ ! -d "venv" ]; then
    echo "[INFO] Criando ambiente virtual (venv)..."
    python3 -m venv venv
fi

# 2. Atualiza o pip dentro do ambiente
./venv/bin/pip install --upgrade pip

# 3. Instala as dependências dentro do venv
echo "[INFO] Instalando bibliotecas..."
./venv/bin/pip install fastapi uvicorn python-dotenv motor pydantic google-generativeai openai

echo "=========================================="
echo "[SUCESSO] Tudo instalado!"
echo "Para rodar, use: ./start.sh"
echo "=========================================="
