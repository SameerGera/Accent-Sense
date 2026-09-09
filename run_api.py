"""
AccentSense API Service Launcher
Runs the FastAPI backend service on http://localhost:8000 with CORS and hot-reload.
"""

import sys
import uvicorn

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("                 ACCENTSENSE BACKEND API SERVICE")
    print("   Listening at : http://localhost:8000")
    print("   Swagger Docs : http://localhost:8000/docs")
    print("   Health Check : http://localhost:8000/health")
    print("   Predict API  : http://localhost:8000/api/predict")
    print("   ASR API      : http://localhost:8000/api/downstream-asr")
    print("=" * 70 + "\n")
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
