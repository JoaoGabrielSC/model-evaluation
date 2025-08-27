#!/bin/bash

# Script para parar todos os serviços do projeto
echo "🛑 Parando serviços do Model Evaluation..."

# Definir comando do compose (nova versão ou legada)
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# Parar banco de dados
if $DOCKER_COMPOSE ps | grep -q "Up"; then
    echo "🐘 Parando banco de dados..."
    $DOCKER_COMPOSE down
    echo "✅ Banco de dados parado"
else
    echo "ℹ️  Banco de dados já está parado"
fi

# Desativar ambiente virtual (se estiver ativo)
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "🐍 Desativando ambiente virtual..."
    deactivate
    echo "✅ Ambiente virtual desativado"
fi

echo "✅ Todos os serviços foram parados"
