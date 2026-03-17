import os
from pathlib import Path

_prefix = os.environ.get("CACHE_DIR_HELICAL_PREFIX")
if _prefix is not None:
    CACHE_DIR_HELICAL = Path(_prefix, ".cache", "helical", "models")
else:
    CACHE_DIR_HELICAL = Path(Path.home(), ".cache", "helical", "models")
