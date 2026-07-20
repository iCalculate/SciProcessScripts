from __future__ import annotations

import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import load_settings
from backend.services.database import DatabaseService
from backend.services.importer import ImportManager
from backend.services.matlab_bridge import MatlabBridge


def main() -> None:
    settings = load_settings()
    database = DatabaseService(settings)
    bridge = MatlabBridge(settings)
    manager = ImportManager(settings, database, bridge)

    job = manager.start_job(input_path="mock://demo", recursive=True, force_reimport=True)
    job_id = job["job_id"]
    print(f"Started job: {job_id}")
    while True:
        snapshot = manager.get_job(job_id)
        if snapshot is None:
            raise RuntimeError("Job disappeared")
        print(
            snapshot["status"],
            snapshot.get("processed_files"),
            snapshot.get("exported_spectra"),
            snapshot.get("current_file"),
        )
        if snapshot["status"] not in {"pending", "running"}:
            break
        time.sleep(1.0)

    print("Dashboard:", database.dashboard_summary())


if __name__ == "__main__":
    main()
