"""
AccentSense API Service Launcher
Runs the FastAPI backend service on http://localhost:8000 with CORS and hot-reload.

HOST env var controls the bind address (default 127.0.0.1; set HOST=0.0.0.0
to expose the service on a LAN/deployment target).
"""

import contextlib
import os
import sys

import uvicorn

if sys.platform == "win32":
    with contextlib.suppress(Exception):  # optional console encoding tweak
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("                 ACCENTSENSE BACKEND API SERVICE")
    print("   Listening at : http://localhost:8000")
    print("   Swagger Docs : http://localhost:8000/docs")
    print("   Health Check : http://localhost:8000/health")
    print("   Predict API  : http://localhost:8000/api/predict")
    print("   ASR API      : http://localhost:8000/api/downstream-asr")
    print("=" * 70 + "\n")
    uvicorn.run(
        "src.api.main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
