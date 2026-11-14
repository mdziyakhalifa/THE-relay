"""The-relay stand-in service package (WP6).

Exports the documented tokenizer, the FastAPI app factory, and the
failure-mode enum used to script load-test conditions.
"""
from app.standin.service import FailureMode, STANDIN_APP, create_app
from app.standin.tokenizer import (
    DEFAULT_COMPLETION_TOKENS,
    PER_MESSAGE_OVERHEAD_TOKENS,
    count_tokens,
    encode,
    prompt_tokens,
)

__all__ = [
    "FailureMode",
    "STANDIN_APP",
    "create_app",
    "DEFAULT_COMPLETION_TOKENS",
    "PER_MESSAGE_OVERHEAD_TOKENS",
    "count_tokens",
    "encode",
    "prompt_tokens",
]
