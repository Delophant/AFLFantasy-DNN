"""Run the AFL Predictions web app: python -m web"""

import os
import uvicorn

if __name__ == "__main__":
    host = os.environ.get("AFLFANTASY_HOST", "0.0.0.0")
    port = int(os.environ.get("AFLFANTASY_PORT", "8001"))
    reload = os.environ.get("AFLFANTASY_RELOAD", "1").lower() not in ("0", "false", "no")
    uvicorn.run("web.main:app", host=host, port=port, reload=reload)
