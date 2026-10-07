"""Build a source-only zip (no recordings, local environments, model or caches)."""
from pathlib import Path
import compileall
import hashlib
import json
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    if not compileall.compile_dir(str(ROOT / "backend"), quiet=1):
        raise SystemExit("Python compilation failed")
    required = ["web/index.html", "web/app.js", "web/pitch.js", "web/worklet.js", "web/styles.css",
                "backend/app.py", "backend/pitch.py", "backend/separation.py", "start.ps1", "setup.ps1", "README.md"]
    for relative in required:
        if not (ROOT / relative).is_file():
            raise SystemExit(f"Missing build input: {relative}")
    for path in (ROOT / "web").glob("*.js"):
        subprocess.run(["node", "--check", str(path)], check=True)
    paths = []
    for directory in ("backend", "web", "scripts", "tests"):
        paths.extend(p for p in (ROOT / directory).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".js", ".mjs", ".html", ".css", ".json"))
    paths.extend(p for p in ROOT.iterdir() if p.is_file() and p.name != "HANDOFF.md"
                 and (p.suffix in (".md", ".txt", ".ps1", ".cmd", ".toml")
                      or p.name == "LICENSE"))
    (ROOT / "dist").mkdir(exist_ok=True)
    target = ROOT / "dist/singing-lab-source.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, f"singing-lab/{path.relative_to(ROOT).as_posix()}")
    result = {"artifact": str(target), "files": len(paths), "bytes": target.stat().st_size,
              "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "excludes": ["audio", "models", "environment", "caches"]}
    (ROOT / "evidence/build.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
