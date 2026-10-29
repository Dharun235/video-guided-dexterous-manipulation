PYTHON := .venv-g1/bin/python

.PHONY: setup test simulate audit preview collect check-data
setup:
	python3 -m scripts.setup_g1
	python3 -m scripts.setup_book_g1

test:
	$(PYTHON) -m pytest -q humanoid/tests

simulate:
	$(PYTHON) -m humanoid.book_g1 $(OUTPUT)

audit:
	$(PYTHON) -m humanoid.book_dataset_audit $(EPISODE) --check-render

preview:
	$(PYTHON) -m humanoid.book_ego_preview $(EPISODE)

collect:
	$(PYTHON) -m humanoid.book_variations $(REFERENCE) $(OUTPUT)

check-data:
	$(PYTHON) -m humanoid.book_collection_audit $(COLLECTION)
