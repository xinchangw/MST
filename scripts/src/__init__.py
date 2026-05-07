from pathlib import Path
import sys


_SRC_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SRC_DIR.parent.parent

for _path in (str(_SRC_DIR), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
