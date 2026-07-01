"""Thin CLI wrapper for lightweight dataset manifest building."""

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from src.common.manifests.build_dataset_manifests import main


if __name__ == "__main__":
    main()
