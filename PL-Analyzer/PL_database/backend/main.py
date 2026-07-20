from __future__ import annotations

import uvicorn

from .app import app, settings


def main() -> None:
    uvicorn.run(
        app,
        host=settings.api.host,
        port=settings.api.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
