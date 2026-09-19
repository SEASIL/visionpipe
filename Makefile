.PHONY: install test lint demo smoke docker clean
PY ?= python

install:
	$(PY) -m pip install -e ".[cpu,dev]"

lint:
	ruff check src tests scripts

test:
	$(PY) -m pytest --cov=visionpipe --cov-report=term-missing

demo:
	$(PY) scripts/make_synthetic_video.py --out data/synthetic.mp4
	$(PY) -m visionpipe -c configs/demo.yaml -v

# End-to-end regression gate: synthetic clip -> pipeline -> expected events + tracking quality
smoke:
	$(PY) scripts/smoke.py

docker:
	docker build -t visionpipe:cpu .

clean:
	rm -rf outputs .pytest_cache .ruff_cache .coverage
