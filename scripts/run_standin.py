"""Run the stand-in service (WP6) as a local process.

Usage (from the project root):
    .venv\\Scripts\\python.exe -m scripts.run_standin [--port 8100] [--host 127.0.0.1]

The default port matches `app.core.settings` standin.base_url
(http://127.0.0.1:8100), so the StandInProvider client needs no override.
"""
from __future__ import annotations

import argparse

import uvicorn

from app.standin.service import STANDIN_APP


def main() -> None:
    parser = argparse.ArgumentParser(description="the-relay stand-in simulator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()

    uvicorn.run(STANDIN_APP, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
