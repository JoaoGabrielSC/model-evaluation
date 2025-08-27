MODEL ?= vit
MODEL_ARG := $(if $(MODEL),--model $(MODEL),)
PERCENT_ARG := $(if $(PERCENT),--percent $(PERCENT),)
SUPPRESS_WARNINGS ?= 2>/dev/null

.PHONY: help populate search reset split one

help:
	@echo "Comandos disponíveis:"
	@echo "  make populate MODEL=<modelo> PERCENT=<percent>   - Popular banco com embeddings"
	@echo "  MODEL: vit | clip | dino | resnet50 | resnet18 | effnet_b0 | mobilenet"
	@echo ""
	@echo "  make search MODEL=<modelo> METRIC=<métrica>    - Buscar imagem similar"
	@echo "  MODEL: vit | clip | dino | resnet50 | resnet18 | effnet_b0 | mobilenet"
	@echo ""
	@echo "Exemplos:"
	@echo "  make populate MODEL=dino"
	@echo "  make populate MODEL=vit"
	@echo "  make search MODEL=resnet50"
	@echo "  make search MODEL=vit"

split:
	python split_dataset.py

populate:
	python populate_pgvector.py $(MODEL_ARG) $(PERCENT_ARG) $(SUPPRESS_WARNINGS)

search:
	python search_similar.py $(SUPPRESS_WARNINGS)

reset:
	python reset_db.py

one:
	python search_one.py $(SUPPRESS_WARNINGS)
