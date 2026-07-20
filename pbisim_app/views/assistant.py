"""Rendered by app.py when this page is selected."""
from pbisim_app.common import *  # noqa: F401,F403


def render():
    theme_mode = st.session_state.get("theme_mode", "Light")
    st.title("AI Simulation Assistant")
    st.caption("Instruct Claude to design, simulate, and analyze phage therapy setups using natural language.")

    # Output display preferences — live here, next to the output they control
    # (moved out of the sidebar's API settings, where they were buried).
    _sc1, _sc2 = st.columns(2)
    with _sc1:
        show_code = st.toggle("Show generated code", value=True, key="ai_show_code")
    with _sc2:
        show_assumptions = st.toggle("Show assumptions", value=True, key="ai_show_assumptions")

    # Check key
    if not st.session_state.agent.client.api_key:
        st.warning("Please enter your Anthropic API Key in the sidebar to use the AI Assistant.")

    # Chat UI
    for turn in st.session_state.history:
        role, val = turn
        if role == "user":
            st.chat_message("user").markdown(val)
        else:
            exec_result, agent_resp = val
            with st.chat_message("assistant"):
                st.markdown(agent_resp.narrative)
                if getattr(agent_resp, "assumptions", "") and show_assumptions:
                    with st.expander("Model Assumptions"):
                        st.markdown(agent_resp.assumptions)
                if agent_resp.code and show_code:
                    with st.expander("Generated python code"):
                        st.code(agent_resp.code, language="python")

                # Show figures and outputs
                if exec_result:
                    if exec_result.success:
                        for fig in exec_result.figures:
                            st.pyplot(fig)
                            plt.close(fig)
                        if exec_result.stdout:
                            with st.expander("Print outputs"):
                                st.text(exec_result.stdout)
                    else:
                        st.error("Execution failed:")
                        st.code(exec_result.error, language="text")

    # input
    if prompt := st.chat_input("Ask a question, or describe a simulation to run (e.g. 'what's a realistic adsorption rate?' or 'simulate 1 strain + 1 phage, burst 50')..."):
        st.chat_message("user").markdown(prompt)

        # Agentic generation: the model decides whether the request is a question (answer
        # directly) or a simulation (write + run + self-correct code via run_pbisim_code),
        # and only runs code when a simulation is actually asked for.
        run = None
        try:
            with st.spinner("Claude is thinking..."):
                run = st.session_state.agent.generate(
                    prompt, execute_code,
                    configure=apply_ai_configuration,
                    summarize=summarize_current_results,
                )
        except Exception as e:
            st.error(f"AI Assistant Error: {e}")
            st.info("If you are getting a 401 Authentication Error, please verify that your Anthropic API key is correct, active, and has remaining usage credits.")

        if run is not None:
            exec_result = run.result
            with st.chat_message("assistant"):
                st.markdown(run.narrative or "_(no explanation returned)_")
                if run.code and show_code:
                    _n = run.tool_calls
                    with st.expander(f"Generated python code · {_n} execution{'s' if _n != 1 else ''}"):
                        st.code(run.code, language="python")

                if exec_result is not None:
                    if exec_result.success:
                        for fig in exec_result.figures:
                            st.pyplot(fig)
                            plt.close(fig)
                        if exec_result.stdout:
                            with st.expander("Print outputs"):
                                st.text(exec_result.stdout)
                    else:
                        st.error("The code still failed after self-correction:")
                        st.code(exec_result.error or "", language="text")

                # The assistant populated the Interactive Simulator's widgets — offer to open it.
                if getattr(run, "configured", False):
                    st.success("I've set up the **Interactive Simulator** with this configuration — open it to review, tweak, and run.")
                    if st.button("▶ Open in Interactive Simulator", key=f"nav_sim_{len(st.session_state.history)}", width="stretch"):
                        st.session_state["_nav_to"] = "Interactive Simulator"
                        st.rerun()

            # Release every figure the tool loop created (the model may have run code several
            # times; only the displayed set was closed above). Unclosed matplotlib figures
            # accumulate in the global registry and are a real memory leak on the small
            # Render container — a prime cause of the app being OOM-killed mid-chat.
            plt.close("all")

            st.session_state.history.append(("assistant", (exec_result, run)))
            # Keep a generous on-page chat log; only drop the oldest turns in very long
            # sessions (bounds stored figures/results without truncating normal use).
            st.session_state.history = st.session_state.history[-40:]



# ── Clinical Trials & Cohorts Page ────────────────────────────────────────────
