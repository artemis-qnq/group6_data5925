from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT
RAW_DATA_DIR = PROJECT_ROOT
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = ARTIFACTS_DIR / "results"
DOCS_DIR = PROJECT_ROOT / "docs"
NOTEBOOKS_DIR = PROJECT_ROOT


def ensure_project_dirs() -> None:
    """Create the shared project directories if they do not already exist."""
    for path in (PROCESSED_DATA_DIR, RESULTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
