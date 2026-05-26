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
