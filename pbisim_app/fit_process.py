"""Run an NLS calibration fit in a SEPARATE PROCESS.

A pbisim-fit NLS fit is CPU-heavy (dozens–hundreds of ODE solves, ~1 min). Running it
in a *thread* can't keep the Streamlit app responsive — CPython's GIL means the fit
starves the main thread, so navigation freezes. A separate process has its own
interpreter/GIL, so the app stays fully usable and navigable while the fit runs.

This module is intentionally MINIMAL and Streamlit-free so the child process (started
with the ``spawn`` method, which re-imports the target's module) is cheap and safe to
launch. The child extracts the fit result into PICKLABLE primitives here — the
pbisim-fit posterior object itself is not guaranteed to pickle, but plain dicts and the
returned ``ModelConfig`` are — and hands them back through the queue.
"""
from __future__ import annotations


def _extract(fp):
    """Pull a finished fit posterior into picklable primitives (MAP, 95% CI, config)."""
    try:
        ci = fp.credible_interval(0.95)
    except Exception:
        ci = {}
    return {
        "map": {k: float(v) for k, v in fp.map().items()},
        "ci": {k: [float(a), float(b)] for k, (a, b) in ci.items()},
        "fitted_config": fp.to_config(),
    }


def run_fit_in_process(queue, payload):
    """Child-process entry point. Runs the NLS fit described by ``payload`` and puts
    ``("done", <picklable result dict>)`` or ``("error", <message>)`` on ``queue``.

    Must stay importable at module level (``spawn`` re-imports it). Imports the fit
    machinery lazily so importing this module never drags in pbisim-fit/torch."""
    try:
        from pbisim_app.nls_fit import run_nls_fit_v2
        fp = run_nls_fit_v2(
            payload["base_config"], payload["targets"], payload["thetas"],
            payload["mappings"], payload["dataset"], payload["obs_keys"],
            od_to_cfu=payload.get("od_to_cfu"),
            n_restarts=payload.get("n_restarts", 3),
            max_nfev=payload.get("max_nfev", 300),
            estimate_b0=payload.get("estimate_b0", "none"),
            obs_compartments=payload.get("obs_compartments"),
            covariate_effects=payload.get("covariate_effects"))
        queue.put(("done", _extract(fp)))
    except Exception as e:  # noqa: BLE001 — report any failure to the parent
        queue.put(("error", f"{type(e).__name__}: {e}"))
