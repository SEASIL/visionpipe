.PHONY: install test lint demo smoke docker clean
PY ?= python

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	ruff check src tests scripts

test:
	$(PY) -m pytest --cov=visionpipe --cov-report=term-missing

demo:
	$(PY) scripts/make_synthetic_video.py --out data/synthetic.mp4
	$(PY) -m visionpipe -c configs/demo.yaml -v

# End-to-end regression gate: synthetic clip -> pipeline -> expected events + tracking quality
smoke:
	$(PY) scripts/make_synthetic_video.py --out data/synthetic.mp4
	rm -rf outputs
	$(PY) -m visionpipe -c configs/demo.yaml --stats-out outputs/stats.json > /dev/null
	$(PY) scripts/check_events.py outputs/cam0/events.jsonl --expect loitering=1 zone_intrusion=2 line_crossing=1
	$(PY) scripts/eval_tracking.py --gt data/synthetic.gt.txt --pred outputs/cam0/tracks.txt --min-mota 0.9 --min-idf1 0.9

docker:
	docker build -t visionpipe:cpu .

clean:
	rm -rf outputs .pytest_cache .ruff_cache .coverage
