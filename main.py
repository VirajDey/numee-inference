"""Development entrypoint. In containers the app is served directly:

    uvicorn app.main:app --host 0.0.0.0 --port 8100
"""
import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8100")),
        reload=True,
    )
