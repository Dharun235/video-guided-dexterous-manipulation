"""Ensure the published physical evidence matches its recorded checksums."""

import hashlib
from pathlib import Path


def test_published_sequence_bundle_checksums():
    folder = Path(__file__).resolve().parents[2] / "examples/video-to-g1"
    for line in (folder / "SHA256SUMS").read_text().splitlines():
        expected, filename = line.split("  ", 1)
        assert hashlib.sha256((folder / filename).read_bytes()).hexdigest() == expected, filename
