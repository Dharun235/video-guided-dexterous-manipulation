PYTHON := .venv-g1/bin/python

.PHONY: setup test review-object transfer
setup:
	python3 -m scripts.setup_g1
	python3 -m scripts.setup_book_g1

test:
	$(PYTHON) -m pytest -q

# Validate the published object candidate; no GPU inference or robot execution.
review-object:
	$(PYTHON) -m scripts.taa.check_object_track \
	  --track examples/object-motion \
	  --scene examples/object-motion/scene/scene.xml \
	  --surface examples/object-motion/case_surface.npy

transfer: OUTPUT ?= runs/transfer
transfer:
	$(PYTHON) -m manipulation.run --output $(OUTPUT)

.PHONY: grasp
grasp: OUTPUT ?= runs/grasp
grasp:
	$(PYTHON) -m video_g1.run_grasp --output $(OUTPUT)

.PHONY: video
video: OUTPUT ?= runs/video
video:
	$(PYTHON) -m video_g1.run_video --output $(OUTPUT)
