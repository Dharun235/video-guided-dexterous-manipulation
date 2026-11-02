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

.PHONY: setup-scene scene
setup-scene:
	python3 -m scripts.setup_scene

scene:
	.venv-scene/bin/python -m reconstruction.scene_check "$(VIDEO)"

.PHONY: fuse
fuse: OUTPUT ?= runs/fused-scene
fuse:
	.venv-scene/bin/python -m reconstruction.video_fusion "$(VIDEO)" --output "$(OUTPUT)"
