"""Thin CLI wrapper for unified manifest validation."""

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from src.common.manifests.schema import ManifestValidationError, main


if __name__ == "__main__":
    try:
        main()
    except ManifestValidationError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
