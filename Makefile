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
	@echo "--- / (chat UI) ---"
	@curl -sf -o /dev/null -w "HTTP %{http_code}\n" http://localhost:$(PORT)/
	@echo "--- /mcp (config UI) ---"
	@curl -sf -o /dev/null -w "HTTP %{http_code}\n" http://localhost:$(PORT)/mcp
	@echo "--- /mcp/api/servers ---"
	@curl -sf http://localhost:$(PORT)/mcp/api/servers | python3 -m json.tool
	@echo "--- /settings ---"
	@curl -sf -o /dev/null -w "HTTP %{http_code}\n" http://localhost:$(PORT)/settings
