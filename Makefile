PYTHON := .venv-g1/bin/python

.PHONY: setup test
setup:
	python3 -m scripts.setup_g1
	python3 -m scripts.setup_book_g1

test:
	$(PYTHON) -m pytest -q humanoid/tests reconstruction/tests
