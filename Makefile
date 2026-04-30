# Makefile
.PHONY: install train validate ci cd

install:
	pip install -r requirements.txt
train:
	python src/train.py
validate:
	python src/validate.py
ci: install train validate
cd:
	@echo "Simulación de despliegue del modelo"

api-test:
	python -c "from app.main import app; print('API OK')"