# Sistema de Avaliação de Modelos de Visão Computacional

Este projeto implementa um sistema completo para avaliação e comparação de diferentes modelos de visão computacional para reconhecimento facial usando embeddings vetoriais armazenados em PostgreSQL com extensão pgvector.

> **📊 Nota sobre Resultados**: Os resultados e métricas apresentados neste projeto foram obtidos utilizando apenas os modelos **Vision Transformer (VIT)** e **CLIP**. Embora o sistema suporte outros modelos (ResNet, EfficientNet, MobileNet, etc.), as análises e conclusões são baseadas especificamente nos experimentos com VIT e CLIP.


## 📈 Métricas e Resultados

O melhor modelo foi o modelo CLIP da OPENAI
<img width="1126" height="381" alt="image" src="https://github.com/user-attachments/assets/50349909-2291-4d78-90d5-f8294f00c42c" />



O sistema calcula automaticamente:

- **Acurácia**: Percentual de predições corretas
- **Precisão**: Proporção de verdadeiros positivos
- **F1-Score**: Média harmônica entre precisão e recall
- **Matriz de Confusão**: Visualização detalhada dos resultados

Os resultados são salvos em:

- `results/[timestamp]/`: Contém logs detalhados
- `results/[timestamp]/summary/`: Gráficos e visualizações


## 🚀 Quick Start

```bash
# 1. Configuração automática
./setup.sh

# 2. Ativar ambiente virtual
source venv/bin/activate

# 3. Preparar dados (adicione suas imagens em datasets/original_images/ primeiro)
make split

# 4. Gerar embeddings
make populate MODEL=vit

# 5. Avaliar modelo
make search MODEL=vit

# 6. Ver resultados em results/[timestamp]/
```

## 📋 Funcionalidades

- **Extração de Embeddings**: Suporte a múltiplos modelos de visão computacional
- **Busca por Similaridade**: Sistema de busca usando diferentes métricas de distância
- **Avaliação de Performance**: Cálculo de métricas como acurácia, precisão e F1-score
- **Visualização de Resultados**: Geração de gráficos e mosaicos para análise
- **Pipeline Completo**: Desde o processamento dos dados até a avaliação final

## 🚀 Configuração Inicial

### ⚡ Configuração Automática (Recomendada)

Execute o script de configuração automática:

```bash
./setup.sh
```

Este script irá:

- Verificar pré-requisitos (Docker, Python)
- Criar ambiente virtual Python
- Instalar todas as dependências
- Configurar banco de dados PostgreSQL
- Criar estrutura de diretórios

### 🛑 Parar Serviços

Para parar todos os serviços:

```bash
./stop.sh
```

### 🔧 Configuração Manual

Se preferir configurar manualmente:

### 1. Pré-requisitos

- Python 3.8+
- Docker e Docker Compose
- Git

### 2. Por que Docker com pgvector?

Este projeto utiliza **PostgreSQL com a extensão pgvector** por várias razões importantes:

#### 🎯 **pgvector - Extensão Especializada em Vetores**

- **Armazenamento nativo de embeddings**: pgvector é especificamente projetado para armazenar e consultar vetores de alta dimensionalidade
- **Índices otimizados**: Suporte a índices HNSW (Hierarchical Navigable Small World) e IVFFlat para busca eficiente de vizinhos mais próximos
- **Operadores de similaridade**: Implementa nativamente operadores para distância euclidiana (`<->`), similaridade cosseno (`<=>`) e produto interno (`<#>`)
- **Performance superior**: Muito mais rápido que soluções baseadas em arrays tradicionais do PostgreSQL

#### 🐳 **Vantagens do Docker**

- **Isolamento**: Ambiente consistente independente do sistema operacional
- **pgvector pré-instalado**: A imagem `pgvector/pgvector:pg16` já vem com a extensão configurada
- **Facilidade de setup**: Um comando (`docker-compose up -d`) configura todo o ambiente
- **Reprodutibilidade**: Garantia de que o ambiente funciona igual em qualquer máquina
- **Cleanup simples**: Fácil de remover sem afetar o sistema host

#### 📊 **Comparação com Alternativas**

| Solução | Complexidade Setup | Performance | Escalabilidade | Manutenção |
|---------|-------------------|-------------|----------------|------------|
| **PostgreSQL + pgvector** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| SQLite + embeddings | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐⭐ |
| Chroma/Pinecone | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Weaviate/Qdrant | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |

#### 🔍 **Casos de Uso Ideais**

- **Pesquisa por similaridade** em grandes volumes de embeddings
- **Sistemas de recomendação** baseados em similaridade vetorial
- **Reconhecimento facial/de imagens** (como neste projeto)
- **Busca semântica** em documentos
- **Análise de sentimentos** e clustering de textos

### 3. Configuração do Banco de Dados

O projeto utiliza PostgreSQL com a extensão pgvector para armazenamento e busca de embeddings vetoriais.

#### Usando Docker Compose (Recomendado)

```bash
# Subir o banco de dados
docker compose up -d
# ou (versão legada)
docker-compose up -d

# Verificar se está rodando
docker compose ps
# ou (versão legada)
docker-compose ps
```

#### Configuração Manual do PostgreSQL

Se preferir instalar manualmente:

```bash
# Instalar PostgreSQL e pgvector
brew install postgresql pgvector  # macOS
# ou
sudo apt-get install postgresql postgresql-contrib  # Ubuntu

# Criar banco e usuário
createdb visaocomputacional
createuser -s compvis
```

### 3. Configuração do Ambiente Python

```bash
# Clonar o repositório
git clone <repository-url>
cd model-evaluation

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
# ou
venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt
```

### 4. Variáveis de Ambiente

Copie o arquivo de exemplo e configure suas variáveis:

```bash
cp .env.example .env
```

Edite o arquivo `.env` conforme necessário:

```env
# Configurações do banco de dados
DB_NAME=visaocomputacional
DB_USER=compvis
DB_PASSWORD=compvis
DB_HOST=localhost
DB_PORT=5432

# Configurações do projeto
SOURCE_CROPPED_DIR=./datasets/faces_cropped
SOURCE_ORIGINAL_DIR=./datasets/original_images
DEST_BASE_DIR=./datasets
```

### 5. Preparação dos Dados

```bash
# Dividir dataset em treino/validação/teste
make split
# ou
python split_dataset.py
```

## 🤖 Modelos Suportados

O sistema suporta os seguintes modelos:

| Modelo | Descrição | Dimensões |
|--------|-----------|-----------|
| `vit` | Vision Transformer base | 768 |
| `clip` | OpenAI CLIP ViT-base-patch32 | 512 |
| `dino` | Facebook DINO ViT-S16 | 384 |
| `deit` | Data-efficient Image Transformer | 768 |
| `resnet50` | Torchvision ResNet50 | 2048 |
| `resnet18` | Torchvision ResNet18 | 512 |
| `effnet_b0` | Torchvision EfficientNet B0 | 1280 |
| `mobilenet` | Torchvision MobileNet v3 large | 960 |

## 📊 Uso do Sistema

### 1. Popular Banco com Embeddings

Extrai embeddings das imagens e armazena no banco de dados:

```bash
# Usando Makefile (recomendado)
make populate MODEL=dino
make populate MODEL=vit PERCENT=50

# Ou diretamente com Python
python populate_pgvector.py --model dino
python populate_pgvector.py --model vit --percent 50

# Popular com todos os modelos
python populate_pgvector.py
```

**Parâmetros:**

- `--model`: Modelo para extrair embeddings (obrigatório se não quiser todos)
- `--percent`: Porcentagem das imagens a processar (padrão: 100)

### 2. Busca por Similaridade e Avaliação

Realiza validação dos modelos usando diferentes métricas de distância:

```bash
# Avaliar um modelo específico
make search MODEL=resnet50
python search_similar.py --model resnet50

# Avaliar todos os modelos
python search_similar.py
```

**Métricas de Distância Suportadas:**

- `cosine`: Similaridade cosseno (padrão)
- `euclidean`: Distância euclidiana  
- `dot_product`: Produto interno

### 3. Comandos Úteis

```bash
# Limpar/resetar banco de dados
make reset
python reset_db.py

# Ver ajuda do Makefile
make help

# Busca em uma imagem específica
make one
python search_one.py
```

## 🛠️ Estrutura do Projeto

```text
model-evaluation/
├── config/                 # Configurações do projeto
├── db/                     # Conexão com banco de dados
├── enums/                  # Enumerações (tipos de modelo)
├── factory/                # Factory para criação de modelos
├── process/                # Processamento de dados e embeddings
├── repositories/           # Acesso aos dados (padrão Repository)
├── strategies/             # Estratégias para diferentes modelos
├── utils/                  # Utilitários diversos
├── validator/              # Validação e métricas
├── datasets/               # Datasets organizados
├── results/                # Resultados das avaliações
├── docker-compose.yml      # Configuração do banco
├── requirements.txt        # Dependências Python
├── Makefile               # Comandos automatizados
└── .env.example           # Exemplo de variáveis de ambiente
```

## 🔧 Solução de Problemas

### Erro de Conexão com Banco

```bash
# Verificar se o container está rodando
docker-compose ps

# Ver logs do banco
docker-compose logs postgres

# Reiniciar banco
docker-compose restart postgres
```

### Problemas com SSL/Certificados

```bash
# Instalar certificados necessários
pip install certifi
```

### Erro de Memória

Para datasets grandes, reduza o processamento:

```bash
# Processar apenas 20% das imagens
make populate MODEL=vit PERCENT=20
```

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
