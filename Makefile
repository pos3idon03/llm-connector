.PHONY: install run test

PORT ?= 8000

install:
	bash setup.sh

run:
	poetry run llm-connector

test:
	@echo "--- /health ---"
	@curl -sf http://localhost:$(PORT)/health | python3 -m json.tool
	@echo "--- /v1/models ---"
	@curl -sf http://localhost:$(PORT)/v1/models | python3 -m json.tool
