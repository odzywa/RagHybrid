.PHONY: build up down restart logs shell test

build:
	docker compose build app

up:
	docker compose up -d

down:
	docker compose down

restart:
	docker compose restart app

logs:
	docker compose logs -f app

shell:
	docker compose exec app bash

test:
	docker compose exec app python -m pytest tests/ -v
