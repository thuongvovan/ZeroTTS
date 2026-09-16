"""Command-line entry point for the unified local TTS API server."""

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "zerotts_api.app:app",
        host=os.getenv("ZEROTTS_HOST", "0.0.0.0"),
        port=int(os.getenv("ZEROTTS_PORT", "8000")),
        workers=1,
    )


if __name__ == "__main__":
    main()
