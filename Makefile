.PHONY: build run run-verbose run-cached shell clean help

# ── Config ──────────────────────────────────────────────────────────────────────
IMAGE   := clinical-graph-builder
COMPOSE := docker compose

# ── Targets ─────────────────────────────────────────────────────────────────────

## help: Show this help message
help:
	@grep -E '^## ' Makefile | sed 's/## /  /'

## build: Build the Docker image
build:
	$(COMPOSE) build

## run: Run with settings from .env (input file must be in ./data/input/)
run:
	@test -f .env || (echo "❌  .env not found — copy .env.example and fill in ANTHROPIC_API_KEY" && exit 1)
	@mkdir -p data/input data/output data/cache
	$(COMPOSE) run --rm clinical-graph-builder

## run-verbose: Run with verbose logging
run-verbose:
	@mkdir -p data/input data/output data/cache
	VERBOSE=1 $(COMPOSE) run --rm clinical-graph-builder

## run-cached: Re-run using cached pipeline stages (faster, no re-extraction)
run-cached:
	@mkdir -p data/input data/output data/cache
	USE_CACHE=1 $(COMPOSE) run --rm clinical-graph-builder

## shell: Open a bash shell inside the container for debugging
shell:
	$(COMPOSE) run --rm --entrypoint bash clinical-graph-builder

## clean: Remove image and data cache
clean:
	$(COMPOSE) down --rmi local
	rm -rf data/cache/*

# ── Advanced one-liner examples ────────────────────────────────────────────────

## example-hip: Run on hip fracture guidelines (edit paths as needed)
example-hip:
	@mkdir -p data/input data/output data/cache
	docker run --rm \
		-e ANTHROPIC_API_KEY=$(ANTHROPIC_API_KEY) \
		-v $(PWD)/data/input:/data/input:ro \
		-v $(PWD)/data/output:/data/output \
		-v $(PWD)/data/cache:/app/pipeline_cache \
		$(IMAGE):latest \
		--input  /data/input/guidelines.pdf \
		--output /data/output/graph.json \
		--metrics /data/output/metrics.json \
		--section "переломы шейки бедра" \
		--verbose
