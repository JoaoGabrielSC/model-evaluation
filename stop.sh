#!/bin/bash

# Script para parar todos os serviços do projeto
echo "🛑 Parando serviços do Model Evaluation..."

# Parar banco de dados
if docker-compose ps | grep -q "Up"; then
    echo "🐘 Parando banco de dados..."
    docker-compose down
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
