"""Launch the ImageCulling Studio Web UI."""

from __future__ import annotations

import argparse
import threading
import webbrowser
import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="ImageCulling Studio Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    parser.add_argument(
        "--open",
        action="store_true",
        help="Automatically open ImageCulling Studio in your default web browser",
    )
    args = parser.parse_args()

    browser_host = "127.0.0.1" if args.host in ("0.0.0.0", "::", "") else args.host
    url = f"http://{browser_host}:{args.port}"
    print(f"Starting ImageCulling Studio at {url}")

    if args.open:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    uvicorn.run(
        "webapp.server:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
