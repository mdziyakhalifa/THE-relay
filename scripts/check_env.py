"""Phase-2 environment verification.

Imports every runtime dependency, prints versions, and checks that a Groq API
key is configured WITHOUT ever printing its value. Run from the project root:

    .venv\\Scripts\\python.exe scripts\\check_env.py

Exit code 0 = all checks passed.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys
from pathlib import Path

# (module_to_import, distribution_name_for_version)
CHECKS: list[tuple[str, str]] = [
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("httpx", "httpx"),
    ("pydantic", "pydantic"),
    ("pydantic_settings", "pydantic-settings"),
    ("dotenv", "python-dotenv"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("fitz", "pymupdf"),  # legacy import name; PyMuPDF 2.x prefers `import pymupdf`
    ("sentence_transformers", "sentence-transformers"),
    ("pytest", "pytest"),
    ("pytest_asyncio", "pytest-asyncio"),
]

failures: list[str] = []
print(f"Python: {sys.version.split()[0]} on {platform.system()} {platform.release()}")
print("-" * 60)

for module_name, dist_name in CHECKS:
    try:
        module = importlib.import_module(module_name)
        version = importlib.metadata.version(dist_name)
        print(f"PASS  {dist_name:<24} {version}")
    except Exception as exc:  # noqa: BLE001 - report any import failure precisely
        failures.append(dist_name)
        print(f"FAIL  {dist_name:<24} {type(exc).__name__}: {exc}")

print("-" * 60)

# Secret check: report presence only, never the value.
try:
    from dotenv import load_dotenv

    load_dotenv()
    import os

    if os.environ.get("GROQ_API_KEY"):
        print(f"GROQ_API_KEY: SET (length {len(os.environ['GROQ_API_KEY'])}, value not shown)")
    else:
        print("GROQ_API_KEY: NOT SET (required before WP5 real-provider smoke test)")
except Exception:  # noqa: BLE001
    print("GROQ_API_KEY: check skipped (dotenv missing)")
env_file_exists = (Path(__file__).resolve().parent.parent / ".env").is_file()
print(f".env file present: {env_file_exists}")

if failures:
    print(f"\nRESULT: {len(failures)} package(s) failed -> {failures}")
    sys.exit(1)
print("\nRESULT: all dependencies import cleanly.")
