"""
Entry point for running distributed server as a module.
"""
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from api.rest.distributed_server import app
import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("NODE_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
