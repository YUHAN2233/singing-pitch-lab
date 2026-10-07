"""Read-only local NCM format check; temporary decoded audio is removed."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import uuid
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.audio import decode_audio
from backend.ncm import MAGIC, extract_ncm

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    source = args.path.resolve(strict=True)
    test_root = (ROOT / ".test-runs").resolve()
    test_root.mkdir(exist_ok=True)
    work = test_root / f"check-ncm-{uuid.uuid4().hex}"
    work.mkdir()
    try:
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            signature = stream.read(8)
            stream.seek(0)
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if signature != MAGIC:
            raise ValueError("not a CTENFDAM NCM container")
        kind = extract_ncm(source, work / "extracted.bin")
        audio, rate = decode_audio(source, work / "original.wav", original_name=source.name)
        print(json.dumps({"name": source.name, "bytes": source.stat().st_size, "sha256": digest.hexdigest(),
                          "container": "CTENFDAM NCM", "codec": kind, "sample_rate": rate,
                          "duration_seconds": round(len(audio) / rate, 3), "channels": audio.shape[1]}, ensure_ascii=False))
    finally:
        assert work.resolve().parent == test_root
        shutil.rmtree(work)


if __name__ == "__main__":
    main()
