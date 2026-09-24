"""Same-origin playground assets, with no inference or model administration."""
from importlib.resources import files

from fastapi import APIRouter
from starlette.responses import Response


_ASSETS = {
    "/": ("index.html", "text/html"),
    "/playground.css": ("playground.css", "text/css"),
    "/playground.js": ("playground.js", "text/javascript"),
}
PUBLIC_WEB_PATHS = frozenset(_ASSETS)
_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; "
        "connect-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; "
        "font-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
}


def playground_router() -> APIRouter:
    router = APIRouter()

    def asset_response(filename: str, media_type: str):
        async def serve() -> Response:
            content = files("model_service").joinpath("static", filename).read_bytes()
            return Response(content, media_type=media_type, headers=_HEADERS)
        return serve

    for path, (filename, media_type) in _ASSETS.items():
        router.add_api_route(path, asset_response(filename, media_type), methods=["GET"],
                             include_in_schema=False)
    return router
