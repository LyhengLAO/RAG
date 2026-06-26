.PHONY: install lint format test build-dataset build-eval run-baseline run-optimized eval comparison compare serve ui up down logs clean

PYTHON := python
PIP    := pip

install:
	$(PIP) install -e ".[dev]"

lint:
	ruff check src/ tests/

format:
	black src/ tests/
	ruff check --fix src/ tests/

test:
	pytest tests/ -v

build-dataset:
	$(PYTHON) scripts/build_dataset.py

build-eval:
	$(PYTHON) scripts/build_eval_set.py

run-baseline:
	$(PYTHON) -m mmrag.pipelines.baseline

run-optimized:
	$(PYTHON) -m mmrag.pipelines.optimized

# Run the full ablation matrix → results/metrics.json + results/figures/
comparison:
	$(PYTHON) scripts/run_comparison.py

# Retrieval + system metrics only (no LLM judge) — fast smoke run
eval:
	$(PYTHON) scripts/run_comparison.py --no-ragas

# Backwards-compatible alias
compare: comparison

# Start the FastAPI server (reload for dev; use --no-reload in production)
serve:
	uvicorn mmrag.serving.api:app --reload --port 8000

# Start the Streamlit frontend (needs the API running first)
ui:
	streamlit run app/streamlit_app.py

# ── Docker ────────────────────────────────────────────────────────────────────

# Build images and start the full stack (ollama → api → app).
# First run pulls the LLM model (~2 GB) and HF models; subsequent runs are instant.
up:
	docker compose up --build -d
	@echo ""
	@echo "  Stack starting — check readiness with: docker compose ps"
	@echo "  API:      http://localhost:8000/health"
	@echo "  Frontend: http://localhost:8501"

# Stop and remove containers (volumes kept — models/index not lost).
down:
	docker compose down

# Follow logs from all services (Ctrl-C to exit)
logs:
	docker compose logs -f

# Remove containers AND all named volumes (wipes models — requires re-download).
clean-volumes:
	docker compose down -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache dist build *.egg-info
