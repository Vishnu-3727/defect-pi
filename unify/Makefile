# UNIFY-RUL. Works with GNU make (Git Bash on Windows, or WSL/Linux).
# On plain PowerShell just run the commands after each target by hand.
PY ?= .venv/Scripts/python.exe          # Linux/Pi: .venv/bin/python
DATASETS ?= cmapss femto xjtu
SMOKE ?= synthetic synthetic_small

.PHONY: help install data smoke train baseline export verify bench test serve node clean

help:
	@echo "make install   install training deps into .venv"
	@echo "make data      parse + cache all datasets to data/interim"
	@echo "make smoke     full pipeline on generated data, no downloads needed"
	@echo "make train     joint training on \$$(DATASETS)"
	@echo "make baseline  RandomForest baseline for the results table"
	@echo "make export    ONNX export + verify + stage the 3 Pi artifacts"
	@echo "make bench     latency p50/p95/p99, memory, thermals"
	@echo "make test      the whole test suite"
	@echo "make serve     run the edge service on :8000"
	@echo "make deploy    copy artifacts to \$$(PI)"

install:
	$(PY) -m pip install -r requirements-train.txt

data:
	$(PY) -m unify.datasets.build --datasets $(DATASETS)

# Everything, on generated data. Run this first — it proves the stack works
# before you spend an evening downloading 15 GB.
smoke:
	$(PY) -m unify.datasets.build --datasets $(SMOKE)
	$(PY) -m unify.train.train --datasets $(SMOKE) --overfit
	$(PY) -m unify.train.train --datasets $(SMOKE) --epochs 40
	$(PY) -m unify.deploy.export_onnx --quantize
	$(PY) -m pytest tests/ -q

train:
	$(PY) -m unify.train.train --datasets $(DATASETS)

baseline:
	$(PY) -m unify.train.baseline_rf --datasets $(DATASETS)

export:
	$(PY) -m unify.deploy.export_onnx --quantize

verify:
	$(PY) -m unify.deploy.verify_onnx

bench:
	cd edge && ../$(PY) bench.py --sweep

test:
	$(PY) -m pytest tests/ -q

serve:
	cd edge && ../$(PY) -m uvicorn app:app --host 0.0.0.0 --port 8000

node:
	$(PY) edge/fake_node.py --degrade

# mDNS, not a fixed IP: the Pi's eth0 is often unplugged and wlan0's lease moves.
PI ?= vishnu@defect-pi.local
deploy: export
	ssh $(PI) 'mkdir -p ~/unify/edge/artifacts'
	# model.sha256 is not optional: inference.py reports the model version from it,
	# and without it /health says "onnx:unknown" and you cannot tell which build
	# is on the Pi.
	scp edge/artifacts/unify_rul.onnx edge/artifacts/norm_stats.json \
	    edge/artifacts/sensor_vocab.json edge/artifacts/manifest.json \
	    edge/artifacts/model.sha256 \
	    $(PI):~/unify/edge/artifacts/
	scp edge/app.py edge/inference.py edge/bench.py edge/fake_node.py $(PI):~/unify/edge/
	scp edge/requirements-pi.txt $(PI):~/unify/edge/
	# app.py resolves STATIC as Path(__file__).parent/"static", so it belongs
	# under edge/, not the repo root. config/ does go at the root.
	scp -r edge/static $(PI):~/unify/edge/
	scp -r config $(PI):~/unify/
	# unify/common is numpy-only and shared with the trainer — see edge/inference.py
	ssh $(PI) 'mkdir -p ~/unify/unify'
	scp -r unify/__init__.py unify/common $(PI):~/unify/unify/
	# --user, not sudo: deploy/unify-rul.service is a user unit. `|| true` because
	# a first deploy happens before the unit is installed.
	ssh $(PI) 'systemctl --user restart unify-rul || true'

clean:
	rm -rf data/processed results/*.pt edge/*.db edge/*.db-wal edge/*.db-shm
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
