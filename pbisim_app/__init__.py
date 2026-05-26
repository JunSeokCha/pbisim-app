"""
pbisim-app — AI-powered natural-language interface for pbisim simulations.

Users describe a simulation scenario in plain English; the app translates
the request into pbisim Python code via the Claude API, executes it, and
returns figures, a narrative summary, and optionally the generated code.

Entry point
-----------
Run the Streamlit UI::

    streamlit run pbisim_app/app.py

or, after installing the package::

    pbisim-app
"""
