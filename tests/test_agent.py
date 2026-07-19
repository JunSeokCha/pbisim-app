"""Tests for agent parsing logic (no API calls required)."""

import pytest
from pbisim_app.agent import _parse_response, AgentResponse


SAMPLE_RESPONSE = """\
Here is the simulation code for your request.

```python
import numpy as np
from pbisim.builder import ModelBuilder
from pbisim.core.model import PBIModel
from pbisim.core.solver import solve_ode

cfg = ModelBuilder(n_bacteria=1, n_phages=1).with_growth_rates(1.2).build()
model = PBIModel(cfg, initial_B=np.array([1e7]), initial_P=np.array([1e4]))
res = solve_ode(model, t_end=24.0, dt=0.5)
print("Done")
```

The simulation shows bacterial clearance within 24 hours under high phage pressure.
Phage replicate rapidly, driving bacteria below detection threshold by hour 18.

Assumptions:
- Growth rate 1.2 h^-1 (typical for E. coli at 37C)
- Adsorption rate 1e-7 mL/PFU/h
- Burst size 80 PFU per cell
"""


class TestParseResponse:
    def test_code_extracted(self):
        resp = _parse_response(SAMPLE_RESPONSE)
        assert "ModelBuilder" in resp.code
        assert "solve_ode" in resp.code

    def test_narrative_non_empty(self):
        resp = _parse_response(SAMPLE_RESPONSE)
        assert len(resp.narrative) > 10

    def test_assumptions_extracted(self):
        resp = _parse_response(SAMPLE_RESPONSE)
        assert "Growth rate" in resp.assumptions or "growth rate" in resp.assumptions.lower()

    def test_no_code_block(self):
        resp = _parse_response("No code here, just narrative text.")
        assert resp.code == ""
        assert len(resp.narrative) > 0

    def test_returns_named_tuple(self):
        resp = _parse_response(SAMPLE_RESPONSE)
        assert isinstance(resp, AgentResponse)


class TestRobustExtraction:
    """The extractor must tolerate fence variants and multiple blocks (a common cause
    of spurious 'no code' retries)."""

    def test_py_fence(self):
        resp = _parse_response("Here:\n```py\nfrom pbisim.builder import ModelBuilder\ncfg = ModelBuilder(n_bacteria=1, n_phages=0).build()\n```")
        assert "ModelBuilder" in resp.code

    def test_bare_fence(self):
        resp = _parse_response("```\nimport numpy as np\nfrom pbisim.core.solver import solve_ode\nsolve_ode\n```")
        assert "solve_ode" in resp.code

    def test_multiple_blocks_prefers_pbisim_script(self):
        text = (
            "First a shell snippet:\n```bash\npip install pbisim\n```\n"
            "Then the simulation:\n```python\nfrom pbisim.builder import ModelBuilder\n"
            "from pbisim.core.solver import solve_ode\ncfg = ModelBuilder(n_bacteria=1, n_phages=1).build()\n```"
        )
        resp = _parse_response(text)
        assert "ModelBuilder" in resp.code and "pip install" not in resp.code

    def test_narrative_strips_all_fences(self):
        text = "Intro.\n```bash\nls\n```\n```python\nsolve_ode\n```\nOutro."
        resp = _parse_response(text)
        assert "```" not in resp.narrative
        assert "Intro." in resp.narrative and "Outro." in resp.narrative


class TestApiLookup:
    """pbisim_api_lookup returns ground-truth signatures (the drift-proof grounding)."""

    def test_method_signature(self):
        from pbisim_app.agent import _pbisim_api_lookup
        s = _pbisim_api_lookup("ModelBuilder.with_phage_params")
        assert "phage_decay_rates" in s   # the arg whose shape the model kept guessing

    def test_function_signature_includes_B0(self):
        from pbisim_app.agent import _pbisim_api_lookup
        s = _pbisim_api_lookup("stationary_phase_ic")
        assert "B0" in s                  # the required arg the model kept omitting

    def test_dotted_prefix_and_missing(self):
        from pbisim_app.agent import _pbisim_api_lookup
        assert "from_strains" in _pbisim_api_lookup("pbisim.BinaryResistanceGenotypes.from_strains")
        assert "not in the pbisim public API" in _pbisim_api_lookup("no_such_symbol_xyz")
