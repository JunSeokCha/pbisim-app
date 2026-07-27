"""Desktop launcher (pbisim_app.desktop) — the pywebview wrapper is additive and must
import without pywebview installed; its helpers resolve the app + a port + a safe env.
The full window/server launch is exercised manually (needs a display + pywebview)."""

from __future__ import annotations

import os


def test_desktop_module_imports_without_pywebview():
    # pywebview is an optional extra; the launcher must import regardless (it's imported
    # lazily inside main()), so `pbisim-app-desktop` can degrade to the browser.
    from pbisim_app import desktop
    assert callable(desktop.main)


def test_desktop_targets_installed_app():
    from pbisim_app import desktop
    p = desktop._app_path()
    assert os.path.exists(p) and p.endswith("app.py")


def test_desktop_free_port_and_server_env():
    from pbisim_app import desktop
    port = desktop._free_port()
    assert isinstance(port, int) and 1024 < port < 65536
    env = desktop._server_env()
    # thread caps + headless matplotlib are always present (Dockerfile parity)
    assert "MPLBACKEND" in env and "OMP_NUM_THREADS" in env
