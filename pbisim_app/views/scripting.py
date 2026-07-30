"""Notebook-style Scripting page (rendered by app.py only when scripting is enabled —
see app.py's PBISIM_ENABLE_SCRIPTING gate).

A scratchpad for power users: run Python against the live pbisim API in a shared kernel
(variables persist across cells). It reuses the SAME sandbox as the AI Assistant
(``executor.execute_code``) — so when the executor is later hardened (subprocess
isolation), both surfaces are hardened at once. The sandbox is research-grade, NOT a
security boundary: code runs in this server process. Gate it to trusted/authenticated use.

Cells are keyed by **stable ids** (not positional index) and all mutations run in
``on_click`` callbacks (never ``st.rerun`` mid-page) so that adding/deleting a cell never
purges another cell's text — Streamlit drops the session-state key of any widget that
isn't rendered during a run, which a mid-page rerun would trigger.
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


def _init_state():
    if "script_cell_ids" not in st.session_state:
        st.session_state.script_cell_ids = [0]
        st.session_state.script_next_id = 1
        st.session_state.script_outputs = {}
        st.session_state["script_src_0"] = _PRELUDE


def _kernel():
    """The persistent sandbox namespace (created lazily; reset by Restart kernel)."""
    if "script_ns" not in st.session_state:
        st.session_state.script_ns = new_namespace()
    return st.session_state.script_ns


# ── callbacks (run before the rerun, so no widget keys get purged) ──
def _add_cell():
    nid = st.session_state.script_next_id
    st.session_state.script_cell_ids.append(nid)
    st.session_state.script_next_id = nid + 1


def _delete_cell(cid):
    ids = st.session_state.script_cell_ids
    if cid in ids and len(ids) > 1:
        ids.remove(cid)
    st.session_state.script_outputs.pop(cid, None)
    st.session_state.pop(f"script_src_{cid}", None)


def _restart_kernel():
    """Discard all variables + outputs (fresh namespace); the cell CODE is kept."""
    st.session_state.script_ns = new_namespace()
    st.session_state.script_outputs = {}


def _run_cell(cid):
    """Execute cell *cid*'s source in the shared kernel; store stdout/PNGs/error.

    Figures are rendered to PNG bytes here and the Figure objects closed, so nothing
    leaks into matplotlib's global registry and the output survives reruns."""
    src = st.session_state.get(f"script_src_{cid}", "")
    res = execute_code(src, namespace=_kernel())
    pngs = []
    for fig in res.figures:
        buf = _io.BytesIO()
        try:
            fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
            pngs.append(buf.getvalue())
        finally:
            plt.close(fig)
    st.session_state.script_outputs[cid] = {
        "success": res.success, "stdout": res.stdout, "error": res.error, "pngs": pngs,
    }


def _run_all():
    for cid in list(st.session_state.script_cell_ids):
        _run_cell(cid)


def render():
    st.title("Scripting")
    st.caption("A notebook-style scratchpad — run Python against the live pbisim API. "
               "Variables persist across cells (one shared kernel).")
    st.warning(
        "Research sandbox — **not isolated**. Code runs in this app's server process with "
        "access to its environment; it is not safe for untrusted input. Keep it to "
        "trusted / authenticated use. A long-running cell blocks the page until it finishes.",
        icon="⚠️")

    _init_state()
    ids = st.session_state.script_cell_ids

    # ── toolbar ──
    _t = st.columns([1, 1, 4])
    _t[0].button("Run all", width="stretch", type="primary", on_click=_run_all)
    _t[1].button("Restart kernel", width="stretch", on_click=_restart_kernel,
                 help="Discard all variables and outputs (fresh namespace). Keeps your code.")

    # ── cells (keyed by stable id) ──
    for _pos, _cid in enumerate(ids):
        st.session_state.setdefault(f"script_src_{_cid}", "")
        st.markdown(f"<div class='section-label'>CELL [{_pos + 1}]</div>", unsafe_allow_html=True)
        st.text_area("code", key=f"script_src_{_cid}", height=180, label_visibility="collapsed")
        _rc = st.columns([1, 1, 5])
        _rc[0].button("Run", key=f"script_run_{_cid}", width="stretch",
                      on_click=_run_cell, args=(_cid,))
        _rc[1].button("Delete", key=f"script_del_{_cid}", width="stretch",
                      on_click=_delete_cell, args=(_cid,), disabled=(len(ids) == 1),
                      help="Remove this cell (variables stay in the kernel).")
        _out = st.session_state.script_outputs.get(_cid)
        if _out:
            for _png in _out.get("pngs", []):
                st.image(_png)
            if (_out.get("stdout") or "").strip():
                st.code(_out["stdout"], language="text")
            if not _out.get("success"):
                st.error(_out.get("error", "Execution failed."))
        st.divider()

    # ── add a cell (at the bottom, next to the newest cell) ──
    st.button("＋ Add cell", key="script_add_bottom", width="stretch", on_click=_add_cell)
