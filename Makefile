PYTHON := .venv-g1/bin/python

.PHONY: setup test review-object
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
