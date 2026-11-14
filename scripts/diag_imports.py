"""Read-only import map: which compiled packages load, which are blocked?

Run: .venv\\Scripts\\python.exe scripts\\diag_imports.py
Prints one PASS/FAIL line per target with the first 160 chars of any error,
so we can see exactly which DLLs this machine's Application Control policy
refuses to load. Changes nothing on the system.
"""
import importlib

TARGETS = [
    "pandas",
    "scipy",
    "sklearn",
    "sklearn.linear_model",
    "transformers",
    "tokenizers",
    "onnxruntime",
    "safetensors",
    "huggingface_hub",
    "torch",  # already known PASS from check_env, re-check to confirm consistency
    "matplotlib",
    "pyarrow",  # pandas 3.x optional dependency, likely pulled in by pandas
]

if __name__ == "__main__":
    for name in TARGETS:
        try:
            importlib.import_module(name)
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001 - any import failure is reportable
            print(f"FAIL {name} -> {type(exc).__name__}: {str(exc)[:160]}")
