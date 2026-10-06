"""支持 `python -m scholargraph` 启动。"""

import sys

from scholargraph.cli import main

if __name__ == "__main__":
    sys.exit(main())
