"""Desktop launcher for pbisim-app.

Runs the ordinary Streamlit server locally and wraps it in a native OS window (via
`pywebview`), so the app looks like a desktop application instead of a browser tab.

This is **purely additive** — it launches the exact same ``app.py`` as the browser
mode / the ``pbisim-app`` command; nothing about the app itself or the web (Render)
deployment changes. If ``pywebview`` isn't installed, or no native webview backend is
available (e.g. missing WebKit2GTK on Linux), it degrades gracefully to opening the
default browser.

Entry point: the ``pbisim-app-desktop`` console script (see ``pyproject.toml``).
Install the extra first: ``pip install -e .[desktop]`` (pulls ``pywebview``).
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request

_WINDOW_TITLE = "pbisim-app"
_HEALTH_TIMEOUT = 90.0   # seconds to wait for the Streamlit server to come up


def _app_path() -> str:
    """Absolute path to the installed ``pbisim_app/app.py``."""
    import pbisim_app
    return os.path.join(os.path.dirname(os.path.abspath(pbisim_app.__file__)), "app.py")


def _free_port() -> int:
    """An unused localhost TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _server_env() -> dict:
    """Environment for the Streamlit subprocess — cap BLAS/OMP threads (as the
    Dockerfile does) so numpy/scipy don't oversubscribe cores, and force a headless
    matplotlib backend."""
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(k, "1")
    return env


def _start_streamlit(port: int) -> subprocess.Popen:
    """Launch ``streamlit run app.py`` headless on ``port`` (127.0.0.1 only)."""
    args = [
        sys.executable, "-m", "streamlit", "run", _app_path(),
        "--server.address=127.0.0.1",
        f"--server.port={port}",
        "--server.headless=true",           # don't auto-open a browser / prompt for email
        "--browser.gatherUsageStats=false",
    ]
    return subprocess.Popen(args, env=_server_env())


def _wait_until_up(health_url: str, timeout: float = _HEALTH_TIMEOUT) -> bool:
    """Poll the Streamlit health endpoint until it returns 200 (or times out)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=1.5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.3)
    return False


def _terminate(proc: subprocess.Popen) -> None:
    """Stop the Streamlit subprocess, escalating to kill if it doesn't exit."""
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception:
        pass


def _open_in_browser(url: str, proc: subprocess.Popen) -> None:
    """Fallback: open the default browser and keep the server alive until it exits
    (or the user interrupts the launcher)."""
    import webbrowser
    print(f"pbisim-app: opening in your browser at {url}", file=sys.stderr)
    webbrowser.open(url)
    try:
        proc.wait()
    except KeyboardInterrupt:
        pass


def _force_browser() -> bool:
    """User opted out of the native window (``PBISIM_APP_BROWSER=1``). Escape hatch for
    machines whose native webview engine is too old to render current Streamlit — the
    window would open but stay blank (a JS error in the page, not a Python exception, so
    it can't auto-fall-back). Old Qt WebEngine on old Linux is the known case; Windows
    (WebView2) / macOS (WKWebView) / up-to-date Linux are fine."""
    return os.environ.get("PBISIM_APP_BROWSER", "").strip().lower() in ("1", "true", "yes")


def main() -> None:
    """Start the local server and show it in a native window (browser fallback)."""
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    proc = _start_streamlit(port)
    try:
        if not _wait_until_up(f"{url}/_stcore/health"):
            print("pbisim-app: the local server did not start in time.", file=sys.stderr)
            _open_in_browser(url, proc)
            return

        if _force_browser():
            print("pbisim-app: PBISIM_APP_BROWSER set — using the browser.", file=sys.stderr)
            _open_in_browser(url, proc)
            return

        try:
            import webview  # pywebview (optional dependency)
        except Exception:
            print("pbisim-app: pywebview not installed — install the desktop extra "
                  "(`pip install -e .[desktop]`) for a native window.", file=sys.stderr)
            _open_in_browser(url, proc)
            return

        webview.create_window(_WINDOW_TITLE, url, width=1400, height=900)
        try:
            webview.start()   # blocks until the window is closed
        except Exception as e:  # no native backend (e.g. missing WebKit2GTK on Linux)
            print(f"pbisim-app: native webview unavailable ({e}); using the browser.",
                  file=sys.stderr)
            _open_in_browser(url, proc)
    finally:
        _terminate(proc)


if __name__ == "__main__":
    main()
