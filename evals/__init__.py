"""Evaluation harness for the pbisim-app AI assistant.

Measures how reliably the assistant turns natural-language requests into working
pbisim code. The checkers and runner logic are unit-tested offline (tests/test_evals.py);
the live measurement — which calls the real Claude API — is run on demand via
``python -m evals.run_eval`` (see that module's docstring).
"""
