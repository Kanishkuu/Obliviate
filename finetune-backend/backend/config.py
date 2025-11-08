from .settings import settings
from pathlib import Path

# Ensure folders exist
Path(settings.artifacts_dir).mkdir(parents=True, exist_ok=True)
Path(settings.uploads_dir).mkdir(parents=True, exist_ok=True)
