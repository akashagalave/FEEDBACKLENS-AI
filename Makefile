# ============================================================
# FeedbackLens AI — Makefile
# ============================================================

.PHONY: help build up down restart logs test ingest push-ecr

# ─── HELP ────────────────────────────────────────────────────
help:
	@echo "FeedbackLens AI — Available Commands:"
	@echo ""
	@echo "  make build       → Build all Docker images"
	@echo "  make up          → Start all services"
	@echo "  make down        → Stop all services"
	@echo "  make restart     → Down + Up"
	@echo "  make logs        → Show all logs"
	@echo "  make ingest      → Run full ingestion pipeline"
	@echo "  make flush       → Flush Redis cache"
	@echo "  make test        → Run health checks"
	@echo "  make push-ecr    → Push images to AWS ECR"

# ─── DOCKER ──────────────────────────────────────────────────
build:
	docker-compose up --build

up:
	docker-compose up

down:
	docker-compose down

restart:
	docker-compose down
	docker-compose up --build

logs:
	docker-compose logs -f

logs-gateway:
	docker logs gateway -f

logs-orchestrator:
	docker logs orchestrator -f

logs-insight:
	docker logs insight-agent -f

logs-understanding:
	docker logs understanding-agent -f

logs-recommendation:
	docker logs recommendation-agent -f

# ─── DATA ────────────────────────────────────────────────────
ingest:
	dvc repro --force

flush:
	docker exec -it redis redis-cli FLUSHALL

# ─── HEALTH CHECK ────────────────────────────────────────────
test:
	@echo "Checking all services..."
	curl -s http://localhost:8000/health
	curl -s http://localhost:8002/health
	curl -s http://localhost:8003/health
	curl -s http://localhost:8004/health
	@echo "All services healthy!"

# ─── ECR PUSH ────────────────────────────────────────────────
AWS_REGION=ap-south-1
AWS_ACCOUNT_ID=$(shell aws sts get-caller-identity --query Account --output text)
ECR_BASE=$(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com

ecr-login:
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(ECR_BASE)

push-ecr: ecr-login
	docker tag feedbacklens-ai-gateway:latest $(ECR_BASE)/feedbacklens-gateway:latest
	docker push $(ECR_BASE)/feedbacklens-gateway:latest

	docker tag feedbacklens-ai-orchestrator:latest $(ECR_BASE)/feedbacklens-orchestrator:latest
	docker push $(ECR_BASE)/feedbacklens-orchestrator:latest

	docker tag feedbacklens-ai-understanding-agent:latest $(ECR_BASE)/feedbacklens-understanding-agent:latest
	docker push $(ECR_BASE)/feedbacklens-understanding-agent:latest

	docker tag feedbacklens-ai-insight-agent:latest $(ECR_BASE)/feedbacklens-insight-agent:latest
	docker push $(ECR_BASE)/feedbacklens-insight-agent:latest

	docker tag feedbacklens-ai-recommendation-agent:latest $(ECR_BASE)/feedbacklens-recommendation-agent:latest
	docker push $(ECR_BASE)/feedbacklens-recommendation-agent:latest

	@echo "All images pushed to ECR!"

# ─── DVC ─────────────────────────────────────────────────────
dvc-push:
	dvc push

dvc-pull:
	dvc pull