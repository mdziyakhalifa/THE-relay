"""One-time helper: write .env with GROQ_API_KEY, input hidden.

Run (any terminal, path does not depend on the current directory):

    C:\\ai app\\the-relay\\.venv\\Scripts\\python.exe -m scripts.set_groq_key

The key is read with getpass (masked, not echoed, not in shell history),
trimmed, and written to the project root .env next to this script's
package. The VALUE is never printed, logged, or returned — only its
length is reported. Re-running overwrites the file.
"""
from __future__ import annotations

import getpass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"


def main() -> None:
    print(f"Writing: {ENV_PATH}")
    key = getpass.getpass("Paste GROQ_API_KEY (input hidden): ").strip().strip('"').strip("'")
    if not key:
        print("NOTHING WRITTEN — input was empty.")
        return
    if not key.startswith("gsk_"):
        print(
            "WARNING: Groq keys normally start with 'gsk_'. The pasted value "
            "does not. Nothing was written — check for truncation and retry."
        )
        return
    ENV_PATH.write_text(f"GROQ_API_KEY={key}\n", encoding="utf-8")

    # ---- self-verification: read back, then run the SAME detection
    # check_env.py uses, so the user sees the truth on their own screen
    read_back = ENV_PATH.read_text(encoding="utf-8")
    print(f"read back OK: {ENV_PATH.name}, {len(read_back)} chars")
    from dotenv import load_dotenv

    load_dotenv()
    import os

    detected = os.environ.get("GROQ_API_KEY")
    if detected:
        print(f"DETECTION: SET (length {len(detected)}) — this terminal can see it")
    else:
        print(
            "DETECTION: NOT SET in this terminal — the file exists but dotenv "
            "did not load it here; report this exact line to the agent"
        )


if __name__ == "__main__":
    main()
