"""Console-script launcher for pbisim-app.

The Streamlit app in ``app.py`` executes its whole body at import time and
requires an active Streamlit runtime, so it cannot be imported directly by a
console entry point (doing so crashes before any function runs). This launcher
instead invokes ``streamlit run`` on the app file, forwarding any extra
command-line arguments (e.g. ``pbisim-app --server.port 8600``).
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    """Launch the Streamlit app via ``streamlit run``."""
    from streamlit.web import cli as stcli

    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    sys.argv = ["streamlit", "run", app_path, *sys.argv[1:]]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
