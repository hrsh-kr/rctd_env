"""
Server entry point — re-exports the FastAPI app from the rctd_env package.

This file exists at server/app.py as required by the OpenEnv deployment spec.
The actual implementation lives in rctd_env/server/app.py.
"""

import uvicorn

from rctd_env.server.app import app  # noqa: F401


def main():
    """Start the RCTD Environment server."""
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
