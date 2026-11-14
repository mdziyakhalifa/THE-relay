"""API package (WP8): the FastAPI service exposing /ask, /ask/stream, /health."""
from app.api.service import create_api_app

__all__ = ["create_api_app"]
