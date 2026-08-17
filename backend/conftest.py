import os
import sys
import tempfile
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Route the persistent job store into a temp location before any app module
# (which binds JOBS_DB_PATH at import time) is loaded.
os.environ.setdefault("JOBS_DB_PATH", str(Path(tempfile.gettempdir()) / "satellite_test_jobs.db"))