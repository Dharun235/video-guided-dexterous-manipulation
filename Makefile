PYTHON := .venv-g1/bin/python

.PHONY: setup test video
setup:
	python3 -m scripts.setup_g1
	python3 -m scripts.setup_book_g1

test:
	$(PYTHON) -m pytest -q

video: OUTPUT ?= runs/video
video:
	$(PYTHON) -m video_g1.run_video --output $(OUTPUT)
