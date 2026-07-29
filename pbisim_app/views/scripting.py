"""Notebook-style Scripting page (rendered by app.py only when scripting is enabled —
see app.py's PBISIM_ENABLE_SCRIPTING gate).

A scratchpad for power users: run Python against the live pbisim API in a shared kernel
(variables persist across cells). It reuses the SAME sandbox as the AI Assistant
(``executor.execute_code``) — so when the executor is later hardened (subprocess
isolation), both surfaces are hardened at once. The sandbox is research-grade, NOT a
security boundary: code runs in this server process. Gate it to trusted/authenticated use.
"""
import io as _io

from pbisim_app.common import *  # noqa: F401,F403
from pbisim_app.executor import execute_code, new_namespace

_PRELUDE = (
    "# The sandbox pre-loads np, plt and the pbisim API (ModelBuilder, PBIModel,\n"
    "# solve_ode, DoseEvent, StrainSet, BinaryResistanceGenotypes, stationary_phase_ic,\n"
    "# time_to_clearance, ...). Variables persist across cells (one shared kernel).\n"
    "cfg = ModelBuilder(n_bacteria=1, n_phages=1).with_growth_rates(1.2).with_phage_params(\n"
    "    burst_sizes=50.0, latent_periods=0.5, adsorption_rates=1e-8, phage_decay_rates=0.1).build()\n"
    "m = PBIModel(cfg, initial_B=np.array([1e7]), initial_P=np.array([1e6]), initial_S=1.0)\n"
    "r = solve_ode(m, t_end=48, dt=0.5)\n"
    "cfu = np.maximum(r.sum_prefixes('B', 'D'), 1.0)   # culturable CFU (B+D)\n"
    "plt.plot(r.time, cfu); plt.yscale('log'); plt.xlabel('time (h)'); plt.ylabel('CFU/mL')\n"
    "print('final CFU:', cfu[-1])\n"
)


def _kernel():
    """The persistent sandbox namespace (created lazily; reset by Restart kernel)."""
    if "script_ns" not in st.session_state:
        st.session_state.script_ns = new_namespace()
    return st.session_state.script_ns


def _run_cell(i):
    """Execute cell *i*'s source in the shared kernel; store stdout/PNGs/error.

    Figures are rendered to PNG bytes here and the Figure objects closed, so nothing
    leaks into matplotlib's global registry and the output survives reruns."""
    src = st.session_state.get(f"script_src_{i}", "")
    res = execute_code(src, namespace=_kernel())
    pngs = []
    for fig in res.figures:
        buf = _io.BytesIO()
        try:
            fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
            pngs.append(buf.getvalue())
        finally:
            plt.close(fig)
    st.session_state.script_outputs[i] = {
        "success": res.success, "stdout": res.stdout, "error": res.error, "pngs": pngs,
    }


def render():
    st.title("Scripting")
    st.caption("A notebook-style scratchpad — run Python against the live pbisim API. "
               "Variables persist across cells (one shared kernel).")
    st.warning(
        "Research sandbox — **not isolated**. Code runs in this app's server process with "
        "access to its environment; it is not safe for untrusted input. Keep it to "
        "trusted / authenticated use. A long-running cell blocks the page until it finishes.",
        icon="⚠️")

    # ── state ──
    if "script_cells" not in st.session_state:
        st.session_state.script_cells = 1
        st.session_state.script_outputs = {}
        st.session_state["script_src_0"] = _PRELUDE
    st.session_state.setdefault("script_outputs", {})
    _n = int(st.session_state.script_cells)

    # ── toolbar ──
    _t = st.columns([1, 1, 1, 3])
    if _t[0].button("Run all", width="stretch", type="primary"):
        for _i in range(_n):
            _run_cell(_i)
    if _t[1].button("Add cell", width="stretch"):
        st.session_state.script_cells = _n + 1
        st.rerun()
    if _t[2].button("Restart kernel", width="stretch",
                    help="Discard all variables and outputs; start a fresh namespace."):
        st.session_state.script_ns = new_namespace()
        st.session_state.script_outputs = {}
        st.rerun()

    # ── cells ──
    for _i in range(_n):
        st.session_state.setdefault(f"script_src_{_i}", "")
        st.markdown(f"<div class='section-label'>CELL [{_i + 1}]</div>", unsafe_allow_html=True)
        st.text_area("code", key=f"script_src_{_i}", height=180, label_visibility="collapsed")
        _rc = st.columns([1, 1, 5])
        if _rc[0].button("Run", key=f"script_run_{_i}", width="stretch"):
            _run_cell(_i)
        if _rc[1].button("Clear", key=f"script_clear_{_i}", width="stretch",
                         help="Empty this cell and its output (variables stay in the kernel)."):
            st.session_state[f"script_src_{_i}"] = ""
            st.session_state.script_outputs.pop(_i, None)
            st.rerun()
        _out = st.session_state.script_outputs.get(_i)
        if _out:
            for _png in _out.get("pngs", []):
                st.image(_png)
            if (_out.get("stdout") or "").strip():
                st.code(_out["stdout"], language="text")
            if not _out.get("success"):
                st.error(_out.get("error", "Execution failed."))
        st.divider()
