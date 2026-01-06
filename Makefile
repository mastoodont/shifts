.PHONY: install run lint package

install:
	pip install -r backend/requirements.txt

run:
	uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

lint:
	# замените на используемый линтер (ruff/flake8/black)
	@echo "Запустите ваш линтер (например, ruff или flake8)."

package:
	python -m build
