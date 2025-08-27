#!/bin/bash

# Script de configuração inicial do projeto Model Evaluation
# Este script automatiza a configuração do ambiente de desenvolvimento

set -e

echo "🚀 Configurando projeto Model Evaluation..."

# Verificar se Docker está instalado
if ! command -v docker &> /dev/null; then
    echo "❌ Docker não encontrado. Por favor, instale o Docker primeiro."
    exit 1
fi

# Verificar se Docker Compose está disponível
if ! docker compose version &> /dev/null && ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose não encontrado. Por favor, instale o Docker Compose primeiro."
    exit 1
fi

# Definir comando do compose (nova versão ou legada)
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# Verificar se Python está instalado
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 não encontrado. Por favor, instale o Python 3 primeiro."
    exit 1
fi

echo "✅ Pré-requisitos verificados"

# Criar arquivo .env se não existir
if [ ! -f .env ]; then
    echo "📝 Criando arquivo .env..."
    cp .env.example .env
    echo "✅ Arquivo .env criado. Edite-o se necessário."
fi

# Criar ambiente virtual Python
if [ ! -d "venv" ]; then
    echo "🐍 Criando ambiente virtual Python..."
    python3 -m venv venv
    echo "✅ Ambiente virtual criado"
fi

# Ativar ambiente virtual e instalar dependências
echo "📦 Instalando dependências Python..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Dependências instaladas"

# Subir banco de dados
echo "🐘 Iniciando banco de dados PostgreSQL..."
$DOCKER_COMPOSE up -d
echo "✅ Banco de dados iniciado"

# Aguardar banco ficar disponível
echo "⏳ Aguardando banco de dados ficar disponível..."
timeout=60
counter=0
while ! $DOCKER_COMPOSE exec -T postgres pg_isready -U compvis -d visaocomputacional > /dev/null 2>&1; do
    sleep 2
    counter=$((counter + 2))
    if [ $counter -ge $timeout ]; then
        echo "❌ Timeout aguardando banco de dados"
        exit 1
    fi
done
echo "✅ Banco de dados está disponível"

# Criar diretórios necessários
echo "📁 Criando diretórios necessários..."
mkdir -p datasets/{faces_cropped,original_images,embedding_faces,test,val,one,augmented_faces}
mkdir -p results
echo "✅ Diretórios criados"

echo ""
echo "🎉 Configuração concluída com sucesso!"
echo ""
echo "📋 Próximos passos:"
echo "1. Adicione suas imagens em datasets/original_images/"
echo "2. Execute: source venv/bin/activate"
echo "3. Execute: make split (para dividir o dataset)"
echo "4. Execute: make populate MODEL=vit (para gerar embeddings)"
echo "5. Execute: make search MODEL=vit (para avaliar o modelo)"
echo ""
echo "📚 Para mais informações, consulte o README.md"
