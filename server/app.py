"""
Server entry point — re-exports the FastAPI app from the rctd_env package.

This file exists at server/app.py as required by the OpenEnv deployment spec.
The actual implementation lives in rctd_env/server/app.py.
"""

from rctd_env.server.app import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
