"""Local workspace -- a single-page UI served from loopback, no CDN, no build."""

from .server import serve

__all__ = ["serve"]
