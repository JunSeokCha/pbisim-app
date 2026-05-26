"""
app.py — Streamlit UI for the pbisim simulation agent.

Run with:
    streamlit run pbisim_app/app.py

or after installing the package:
    pbisim-app
"""

from __future__ import annotations

import io
import os

import matplotlib.pyplot as plt
import streamlit as st

from pbisim_app.agent import SimulationAgent
from pbisim_app.executor import execute_code


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="pbisim — Phage-Bacteria Simulation",
    page_icon="🦠",
    layout="wide",
)


# ── Session state ─────────────────────────────────────────────────────────────
def _init_state():
    if "agent" not in st.session_state:
        st.session_state.agent = SimulationAgent()
    if "history" not in st.session_state:
        st.session_state.history = []   # list of (role, text/result) tuples


_init_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    api_key = st.text_input(
        "Anthropic API key",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Required to run the AI agent.  Set ANTHROPIC_API_KEY env var to avoid re-entering.",
    )
    if api_key:
        st.session_state.agent.client.api_key = api_key

    show_code = st.toggle("Show generated code", value=True)
    show_assumptions = st.toggle("Show assumptions", value=True)

    if st.button("🔄 New simulation (clear history)"):
        st.session_state.agent.reset()
        st.session_state.history.clear()
        st.rerun()

    st.markdown("---")
    st.markdown(
        "**pbisim-app** uses [Claude](https://anthropic.com) to translate "
        "natural-language requests into [pbisim](https://github.com/phage-therapy-sim/pbisim) "
        "simulation code."
    )


# ── Main layout ───────────────────────────────────────────────────────────────
st.title("🦠 pbisim Simulation Assistant")
st.caption(
    "Describe your phage-bacteria simulation scenario in plain English. "
    "The assistant will generate and run the pbisim code for you."
)

# ── Example prompts ───────────────────────────────────────────────────────────
with st.expander("💡 Example prompts"):
    examples = [
        "Simulate a 2-phage cocktail + ciprofloxacin in a chronic pneumonia patient "
        "with poor nutritional environment for 5 days with BID dosing.",
        "Show bacterial growth with and without a single phage dose of 1e8 PFU/mL.",
        "Compare phage-only vs antibiotic-only vs combination therapy for 48 hours. "
        "Show which achieves fastest bacterial clearance.",
        "Model resistance evolution with 1 phage species and 2 bacterial strains "
        "(susceptible and resistant) over 72 hours.",
    ]
    for ex in examples:
        if st.button(ex, key=ex[:40]):
            st.session_state["_queued_prompt"] = ex
            st.rerun()

# ── Render conversation history ───────────────────────────────────────────────
for role, content in st.session_state.history:
    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    else:
        exec_result, agent_resp = content
        with st.chat_message("assistant"):
            st.markdown(agent_resp.narrative)

            if show_assumptions and agent_resp.assumptions:
                with st.expander("📋 Assumptions"):
                    st.markdown(agent_resp.assumptions)

            if show_code and agent_resp.code:
                with st.expander("🐍 Generated code"):
                    st.code(agent_resp.code, language="python")

            if exec_result.success:
                for fig in exec_result.figures:
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
                    buf.seek(0)
                    st.image(buf)
                    plt.close(fig)
                if exec_result.stdout:
                    with st.expander("📄 Output"):
                        st.text(exec_result.stdout)
            else:
                st.error("Execution failed:")
                st.code(exec_result.error, language="text")

# ── Chat input ────────────────────────────────────────────────────────────────
queued = st.session_state.pop("_queued_prompt", None)
prompt = st.chat_input("Describe your simulation...") or queued

if prompt:
    st.session_state.history.append(("user", prompt))

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating simulation code..."):
            try:
                agent_resp = st.session_state.agent.ask(prompt)
            except Exception as e:
                st.error(f"Agent error: {e}")
                st.stop()

        with st.spinner("Running simulation..."):
            exec_result = execute_code(agent_resp.code) if agent_resp.code else None

        st.markdown(agent_resp.narrative)

        if show_assumptions and agent_resp.assumptions:
            with st.expander("📋 Assumptions"):
                st.markdown(agent_resp.assumptions)

        if show_code and agent_resp.code:
            with st.expander("🐍 Generated code"):
                st.code(agent_resp.code, language="python")

        if exec_result:
            if exec_result.success:
                for fig in exec_result.figures:
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
                    buf.seek(0)
                    st.image(buf)
                    plt.close(fig)
                if exec_result.stdout:
                    with st.expander("📄 Output"):
                        st.text(exec_result.stdout)
            else:
                st.error("Execution failed:")
                st.code(exec_result.error, language="text")

    st.session_state.history.append(("assistant", (exec_result, agent_resp)))


def main():
    """Entry point for `pbisim-app` console script."""
    # Streamlit handles everything above; this function is a no-op but
    # must exist for the console_scripts entry point.
    pass
