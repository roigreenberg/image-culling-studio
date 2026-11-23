"""Launch the ImageCulling Studio Web UI."""

from __future__ import annotations

import argparse
import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="ImageCulling Studio Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    args = parser.parse_args()

    print(f"Starting ImageCulling Studio at http://{args.host}:{args.port}")
    uvicorn.run(
        "webapp.server:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
