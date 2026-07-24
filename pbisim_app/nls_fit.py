"""pbisim-fit NLS integration for the Calibration page.

Everything here imports ``pbisim_fit`` LAZILY (inside the functions), so the app
starts without pulling pbisim-fit / torch — the fit machinery loads only when the
user actually runs a fit. Requires ``pip install pbisim-fit`` in the environment.
"""

from __future__ import annotations

import numpy as np


# Curated free-parameter catalog: (label, pbisim-fit path, lo, hi, log_scale).
# Only entries whose array index fits the current model are offered (see
# available_free_params). Paths follow pbisim-fit's grammar; bounds are explicit
# so no PARAM_BOUNDS lookup is required.
FREE_PARAM_CATALOG = [
    ("Growth rate — strain {i}",              "growth_rates[{i}]",                 0.1,  3.0,  False, "strain"),
    ("Bacteria/resource ratio — strain {i}",  "bacteria_to_resource_ratio[{i}]",   1e6,  1e10, True,  "strain"),
    ("Natural death rate — strain {i}",       "death_rate_B[{i}]",                 0.0,  1.0,  False, "strain"),
    ("Monod Ks (global)",                     "monod_constant",                    0.01, 5.0,  True,  "global"),
    ("Adsorption — strain {i} × phage {j}",   "adsorption_rates[{i},{j}]",         1e-11, 1e-6, True, "pair"),
    ("Burst size — strain {i} × phage {j}",   "burst_sizes[{i},{j}]",              1.0,  500.0, False, "pair"),
    ("Latent period — strain {i} × phage {j}", "latent_periods[{i},{j}]",          0.1,  5.0,  False, "pair"),
    ("Phage decay rate — phage {j}",          "phage_decay_rates[{j}]",            0.0,  2.0,  False, "phage"),
]


import ast as _ast
import operator as _op
import re as _re

# Bounds + log-scale templates per parameter family (from pbisim-fit PARAM_BOUNDS,
# widened where useful). Used when a target is freed without explicit bounds.
_FAMILY = {
    # family key: (lo, hi, log)
    "growth_rates": (0.05, 3.0, False),
    "bacteria_to_resource_ratio": (1e6, 1e11, True),
    "death_rate_B": (0.0, 2.0, False),
    "monod_constant": (0.01, 5.0, True),
    "adsorption_rates": (1e-12, 1e-6, True),
    "burst_sizes": (1.0, 500.0, False),
    "latent_periods": (0.1, 3.0, False),
    "phage_decay_rates": (0.01, 2.0, False),
    "mutation_rates": (1e-9, 1e-4, True),
    "debris_u": (0.01, 1.0, False),
    "debris_v": (0.001, 1.0, False),
    "debris_kdis": (0.001, 5.0, False),
    "imm_max": (1e4, 1e9, True),
    "imm_kill50": (1e4, 1e8, True),
    "recycle_fraction": (0.0, 1.0, False),
    "dormancy_diffusion_rate": (1e-3, 3.0, True),
    "death_rate_D": (0.0, 1.0, False),
    "dormant_od_fraction": (0.0, 1.0, False),
    # pbisim-fit-side "virtual" estimables (setattr'd on the config; the engine never
    # sees them — the fit interprets them). See _apply_fitness_cost / _resolve_ic_override.
    "fitness_cost": (0.0, 1.0, False),
    "init_resistant_fraction": (0.0, 0.999, False),
    "fit_initial_cfu": (1e3, 1e11, True),
    "fit_initial_pfu": (1e2, 1e12, True),
}


def _get_path(config, path):
    """Read a parameter value from a config by pbisim-fit path grammar, e.g.
    ``growth_rates[0]``, ``adsorption_rates[1,0]``, ``monod_constant``, ``debris_u``,
    ``pd_config.abx_ec50[0,0]``. Returns float or raises."""
    m = _re.match(r"([\w.]+)(?:\[(\d+)(?:,(\d+))?\])?$", path)
    if not m:
        raise ValueError(f"bad path {path!r}")
    attr, i, j = m.group(1), m.group(2), m.group(3)
    obj = config
    for part in attr.split("."):
        obj = getattr(obj, part)
    if i is None:
        return float(obj)
    arr = np.asarray(obj, dtype=float)
    return float(arr[int(i)] if j is None else arr[int(i), int(j)])


def available_targets(config, initial_cfu=None, initial_pfu=None):
    """Comprehensive list of estimable model parameters for this config:
    [(label, path, current_value, lo, hi, log)]. Only families present in the config
    are emitted (mutation only off-diagonal, debris/immune only when enabled, etc.).
    ``initial_cfu``/``initial_pfu`` seed the start values for the estimable initial
    conditions (else fall back to placeholders)."""
    nb = int(getattr(config, "n_bacteria", 1))
    npg = int(getattr(config, "n_phages", 0))
    out = []

    def add(label, path, family, value=None):
        if value is None:
            try:
                value = _get_path(config, path)
            except Exception:
                return
        lo, hi, log = _FAMILY.get(family, (None, None, False))
        out.append((label, path, float(value), lo, hi, bool(log)))

    for i in range(nb):
        add(f"Growth rate — strain {i} (h⁻¹)", f"growth_rates[{i}]", "growth_rates")
        add(f"Bacteria/resource ratio — strain {i}", f"bacteria_to_resource_ratio[{i}]", "bacteria_to_resource_ratio")
        add(f"Natural death rate — strain {i} (h⁻¹)", f"death_rate_B[{i}]", "death_rate_B")
    add("Monod half-saturation Ks", "monod_constant", "monod_constant")
    for i in range(nb):
        for j in range(npg):
            add(f"Adsorption — strain {i} × phage {j}", f"adsorption_rates[{i},{j}]", "adsorption_rates")
            add(f"Burst size — strain {i} × phage {j}", f"burst_sizes[{i},{j}]", "burst_sizes")
            add(f"Latent period — strain {i} × phage {j} (h)", f"latent_periods[{i},{j}]", "latent_periods")
    for j in range(npg):
        add(f"Phage decay rate — phage {j} (h⁻¹)", f"phage_decay_rates[{j}]", "phage_decay_rates")
    # mutation network (off-diagonal transitions only)
    if getattr(config, "mutation_rates", None) is not None:
        for i in range(nb):
            for j in range(nb):
                if i != j:
                    add(f"Mutation rate — strain {j} → {i}", f"mutation_rates[{i},{j}]", "mutation_rates")
    # debris (OD) — only when the module is on
    if getattr(config, "debris_u", None) is not None:
        add("Debris yield from deaths (u)", "debris_u", "debris_u")
        add("Debris yield from lysis (v)", "debris_v", "debris_v")
        add("Debris dissolution rate (k_dis)", "debris_kdis", "debris_kdis")
    # immunity
    if getattr(config, "imm_max", None):
        add("Immune capacity (imm_max)", "imm_max", "imm_max")
        add("Immune kill-half (imm_kill50)", "imm_kill50", "imm_kill50")
    # nutrient recycling + (when dormancy on) depth-diffusion & dormant death & OD weight
    add("Nutrient recycle fraction", "recycle_fraction", "recycle_fraction")
    if getattr(config, "dormancy_diffusion_rate", None) is not None:
        for i in range(nb):
            add(f"Dormancy diffusion rate — strain {i}", f"dormancy_diffusion_rate[{i}]", "dormancy_diffusion_rate")
            add(f"Dormant death rate — strain {i}", f"death_rate_D[{i}]", "death_rate_D")
    if getattr(config, "debris_u", None) is not None:
        add("Dormant OD contribution fraction", "dormant_od_fraction", "dormant_od_fraction")
    # ── Estimable initial conditions & selection parameters (pbisim-fit-side) ──
    # These are set BY the fit (setattr on the config); the engine never reads them.
    add("Initial bacterial density (B₀, CFU/mL)", "fit_initial_cfu", "fit_initial_cfu",
        value=(float(initial_cfu) if initial_cfu else 1e7))
    if npg >= 1:
        add("Co-inoculated phage titre (PFU/mL)", "fit_initial_pfu", "fit_initial_pfu",
            value=(float(initial_pfu) if initial_pfu else 1e6))
    if nb >= 2:
        add("Fitness cost (resistant vs WT growth)", "fitness_cost", "fitness_cost", value=0.0)
        add("Initial resistant fraction", "init_resistant_fraction", "init_resistant_fraction", value=0.0)
    return out


# ── Safe expression evaluator for theta→target mappings ────────────────────────
_EXPR_FUNCS = {"exp": np.exp, "log": np.log, "log10": np.log10, "sqrt": np.sqrt, "abs": abs}
_EXPR_BINOPS = {_ast.Add: _op.add, _ast.Sub: _op.sub, _ast.Mult: _op.mul,
                _ast.Div: _op.truediv, _ast.Pow: _op.pow}


def _eval_expr(node, env):
    if isinstance(node, _ast.Expression):
        return _eval_expr(node.body, env)
    if isinstance(node, _ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, _ast.Name):
        if node.id in env:
            return env[node.id]
        raise ValueError(f"unknown name {node.id!r}")
    if isinstance(node, _ast.BinOp) and type(node.op) in _EXPR_BINOPS:
        return _EXPR_BINOPS[type(node.op)](_eval_expr(node.left, env), _eval_expr(node.right, env))
    if isinstance(node, _ast.UnaryOp) and isinstance(node.op, (_ast.UAdd, _ast.USub)):
        v = _eval_expr(node.operand, env)
        return +v if isinstance(node.op, _ast.UAdd) else -v
    if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name) and node.func.id in _EXPR_FUNCS:
        return _EXPR_FUNCS[node.func.id](*[_eval_expr(a, env) for a in node.args])
    raise ValueError("unsupported expression")


def validate_expr(expr, theta_names):
    """Parse-check a mapping expression against the declared theta names. Returns
    (ok, message)."""
    try:
        tree = _ast.parse(expr, mode="eval")
        _eval_expr(tree, {n: 1.0 for n in theta_names})
        return True, ""
    except Exception as e:
        return False, str(e)


def _make_expr_fn(expr, theta_names):
    tree = _ast.parse(expr, mode="eval")
    def fn(t):
        return _eval_expr(tree, {n: getattr(t, n) for n in theta_names})
    return fn


def available_free_params(config):
    """Return [(label, path, lo, hi, log_scale)] valid for this config's dimensions."""
    n_b = int(getattr(config, "n_bacteria", 1))
    n_p = int(getattr(config, "n_phages", 0))
    out = []
    for label, path, lo, hi, logs, kind in FREE_PARAM_CATALOG:
        if kind == "global":
            out.append((label, path, lo, hi, logs))
        elif kind == "strain":
            for i in range(n_b):
                out.append((label.format(i=i), path.format(i=i), lo, hi, logs))
        elif kind == "phage":
            for j in range(n_p):
                out.append((label.format(j=j), path.format(j=j), lo, hi, logs))
        elif kind == "pair":
            for i in range(n_b):
                for j in range(n_p):
                    out.append((label.format(i=i, j=j), path.format(i=i, j=j), lo, hi, logs))
    return out


def estimate_od_to_cfu(agg, sel_arms):
    """Data-driven CFU-per-OD ratio = median(cfu / od) over matched (arm, time) points
    where both a CFU and an OD reading exist. Returns None if not estimable."""
    m = agg[agg["arm"].isin(sel_arms)]
    ratios = []
    for a in sel_arms:
        cfu = m[(m["arm"] == a) & (m["observable"] == "cfu")].set_index("time")["value"]
        od = m[(m["arm"] == a) & (m["observable"] == "od")].set_index("time")["value"]
        common = cfu.index.intersection(od.index)
        for t in common:
            c, o = float(cfu.loc[t]), float(od.loc[t])
            if np.isfinite(c) and np.isfinite(o) and c > 0 and o > 0:
                ratios.append(c / o)
    return float(np.median(ratios)) if ratios else None


def build_dataset(agg, sel_arms, sel_obs, arm_cond, *, od_to_cfu=None, dose_unit="moi"):
    """Construct a pbisim-fit ExperimentalDataset from the app's aggregated calibration
    data + per-arm conditions (growth-phase pre-run, B₀, phage dose).

    ``dose_unit`` is how each arm's dose value is interpreted: ``"moi"`` (× the arm's
    B₀) or ``"pfu"`` (absolute PFU/mL). pbisim-fit's ``DoseRecord`` supports both and
    ``_solve_arm`` scales MOI by the initial CFU while using PFU verbatim."""
    from pbisim_fit.data.ingestion import (
        ExperimentalDataset, TreatmentRecord, DoseRecord, DatasetMetadata)

    m = agg[agg["arm"].isin(sel_arms) & agg["observable"].isin(sel_obs)]
    times = np.array(sorted(float(t) for t in m["time"].unique()), dtype=float)
    tidx = {t: i for i, t in enumerate(times)}

    arms = []
    for a in sel_arms:
        cond = arm_cond.get(a, {})
        kw = {}
        for ok in sel_obs:
            col = np.full(len(times), np.nan)
            d = m[(m["arm"] == a) & (m["observable"] == ok)]
            for _, r in d.iterrows():
                col[tidx[float(r["time"])]] = float(r["value"])
            if np.isfinite(col).any():
                kw[ok] = col
        if not kw:
            continue
        doses = []
        _dose = float(cond.get("moi", 0.0) or 0.0)   # dose value in `dose_unit`
        if _dose > 0:
            doses = [DoseRecord(time=0.0, amount=_dose,
                                unit=("pfu" if dose_unit == "pfu" else "moi"), target="phage")]
        pr = float(cond.get("prerun", 0.0) or 0.0)
        arms.append(TreatmentRecord(
            label=str(a), dose_events=doses,
            pretreatment_h=(pr if pr > 0 else None),
            pretreatment_inoculum=(float(cond["b0"]) if cond.get("b0") else None),
            **kw,
        ))
    meta = DatasetMetadata(od_to_cfu=(float(od_to_cfu) if od_to_cfu else None))
    return ExperimentalDataset(data_type="time_kill", time=times, arms=arms, metadata=meta)


def _midpoint(lo, hi, log):
    lo, hi = float(lo), float(hi)
    if log and lo > 0 and hi > 0:
        return float(10.0 ** (0.5 * (np.log10(lo) + np.log10(hi))))
    return 0.5 * (lo + hi)


def _safe_initial(lo, hi, log, fallback):
    """Start value for a parameter. Bounds may be infinite (unconstrained), so fall
    back to an anchor (the parameter's current value / a theta default) whenever a
    finite midpoint can't be formed."""
    lo, hi = float(lo), float(hi)
    if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
        return _midpoint(lo, hi, log)
    return float(fallback)


def has_unbounded(targets, thetas):
    """True if any FREED target or any theta has an infinite bound. pbisim-fit's
    multi-start sampler draws uniformly within the bounds and overflows on ±inf, so
    such fits must run with a single start (n_restarts=1)."""
    for t in (targets or []):
        if t.get("free") and not (np.isfinite(t["lo"]) and np.isfinite(t["hi"])):
            return True
    for th in (thetas or []):
        if not (np.isfinite(th["lo"]) and np.isfinite(th["hi"])):
            return True
    return False


def build_param_spec(base_config, free_params, shared_groups=None, fitness_links=None):
    """Build a pbisim-fit parameter spec for NLS.

    - ``free_params`` = [(label, path, lo, hi, log_scale)] — each an independent
      estimated value bound 1:1 to its path.
    - ``shared_groups`` = [{"paths":[...], "lo","hi","log"}] — one estimated value
      TIED across every path in the group (reparameterization: sharing).
    - ``fitness_links`` = [{"source":path, "target":path, "lo","hi"}] — derive
      ``target = source_value × (1 − cost)`` with ``cost`` estimated in [lo,hi]
      (reparameterization: deriving a resistant param from a fit one via a cost).

    Uses pbisim-fit ``Reparam`` when any reparameterization is present (sharing /
    fitness links), else the simpler 1:1 ``FreeParamSpec``. Returns (config, spec)."""
    shared_groups = shared_groups or []
    fitness_links = fitness_links or []
    if not shared_groups and not fitness_links:
        from pbisim_fit import FreeParamSpec
        fs = FreeParamSpec(base_config)
        for _label, path, lo, hi, logs in free_params:
            fs = fs.free(path, lo, hi, log_scale=logs)
        return fs.build()

    from pbisim_fit import Reparam
    rp = Reparam(base_config)
    # Map each estimated path -> its theta name, so a derived link can reference an
    # estimated source (individual or shared). Sources not estimated fall back to the
    # base-config value via the Reparam ``base`` argument.
    _path_theta = {}
    for k, (label, path, lo, hi, logs) in enumerate(free_params):
        nm = f"ind{k}"
        rp = rp.param(nm, lo, hi, log_scale=bool(logs), initial=_midpoint(lo, hi, logs))
        rp = rp.set(path, (lambda t, n=nm: getattr(t, n)))
        _path_theta[path] = nm
    for k, g in enumerate(shared_groups):
        nm = f"shr{k}"
        rp = rp.param(nm, g["lo"], g["hi"], log_scale=bool(g.get("log")),
                      initial=_midpoint(g["lo"], g["hi"], g.get("log")))
        for p in g["paths"]:
            rp = rp.set(p, (lambda t, n=nm: getattr(t, n)))
            _path_theta[p] = nm
    for k, fl in enumerate(fitness_links):
        nm = f"link{k}"
        kind = fl.get("kind", "fitness_cost")
        lo, hi = float(fl["lo"]), float(fl["hi"])
        rp = rp.param(nm, lo, hi, log_scale=bool(fl.get("log")), initial=_midpoint(lo, hi, fl.get("log")))
        src = _path_theta.get(fl["source"])
        # target = source × factor, where factor = theta (scale) or (1 − theta) (fitness cost).
        _mul = (lambda th: th) if kind == "scale" else (lambda th: 1.0 - th)
        if src is not None:            # source is an estimated theta
            rp = rp.set(fl["target"],
                        (lambda t, s=src, n=nm, m=_mul: getattr(t, s) * m(getattr(t, n))))
        else:                          # source fixed at its base-config value
            rp = rp.set(fl["target"],
                        (lambda t, base, sp=fl["source"], n=nm, m=_mul: base.get(sp) * m(getattr(t, n))))
    return rp.build()


def build_param_spec_v2(base_config, targets, thetas=None, mappings=None):
    """Build a parameter spec from the table-driven UI.

    - ``targets`` = [{"path","free"(bool),"value","lo","hi","log"}] — every model
      parameter. ``free`` → estimated 1:1 with [lo,hi]; else fixed at ``value``.
    - ``thetas`` = [{"name","lo","hi","log","initial"}] — user-defined estimated
      quantities referenced by mappings.
    - ``mappings`` = [{"path","expr"}] — bind a target path to an expression of thetas
      (e.g. ``theta1*(1-theta2)``). A mapped path overrides its ``targets`` row.

    Returns (config, spec). Uses ``Reparam`` (thetas + mappings + free targets +
    changed fixed values all compose)."""
    thetas = thetas or []
    mappings = [m for m in (mappings or []) if str(m.get("expr", "")).strip()]
    mapped_paths = {m["path"] for m in mappings}
    theta_names = [th["name"] for th in thetas]

    from pbisim_fit import Reparam
    rp = Reparam(base_config)

    def _prior(d):
        """Optional Gaussian prior (mu, sd) — a MAP penalty toward mu with spread sd
        (natural scale; log params map to log-normal). Both required, sd > 0."""
        mu, sd = d.get("prior_mu"), d.get("prior_sd")
        if mu is not None and sd is not None and float(sd) > 0:
            return (float(mu), float(sd))
        return None

    for th in thetas:
        init = th.get("initial")
        _log = bool(th.get("log"))
        _fallback = float(init) if init not in (None, "") else 1.0
        rp = rp.param(th["name"], th["lo"], th["hi"], log_scale=_log,
                      initial=(float(init) if init not in (None, "")
                               else _safe_initial(th["lo"], th["hi"], _log, _fallback)),
                      prior=_prior(th))
    for k, t in enumerate(targets):
        p = t["path"]
        if p in mapped_paths:
            continue
        if t.get("free"):
            nm = f"free{k}"
            _log = bool(t.get("log"))
            rp = rp.param(nm, t["lo"], t["hi"], log_scale=_log,
                          initial=_safe_initial(t["lo"], t["hi"], _log, t["value"]),
                          prior=_prior(t))
            rp = rp.set(p, (lambda tt, n=nm: getattr(tt, n)))
        else:
            # fixed: pin only when the value differs from the base config. Params that
            # can't be read from the config are pbisim-fit-side virtuals (fitness_cost,
            # fit_initial_cfu, …) — leave them UNSET when fixed (setting fitness_cost=0
            # would wipe a BRG's baked-in resistant growth); they act only when freed.
            try:
                if abs(float(t["value"]) - _get_path(base_config, p)) > 1e-30:
                    rp = rp.fix(p, float(t["value"]))
            except Exception:
                pass
    for m in mappings:
        rp = rp.set(m["path"], _make_expr_fn(m["expr"], theta_names))
    return rp.build()


def run_nls_fit_v2(base_config, targets, thetas, mappings, dataset, obs_keys, *,
                   od_to_cfu=None, n_restarts=3, max_nfev=300):
    """Run NLS from the table-driven spec (see build_param_spec_v2)."""
    from pbisim_fit.refinement.nls import refine_nls, NLSConfig
    if od_to_cfu and "od" in obs_keys:
        try:
            base_config.od_to_cfu_conversion_factor = float(od_to_cfu)
        except Exception:
            pass
    # Multi-start sampling can't draw from an infinite range → force a single start
    # (from the initial) whenever any freed/theta parameter is unbounded.
    if has_unbounded(targets, thetas):
        n_restarts = 1
    cfg, pspec = build_param_spec_v2(base_config, targets, thetas, mappings)
    return refine_nls(dataset, pspec, cfg,
                      cfg=NLSConfig(obs_keys=list(obs_keys), n_restarts=int(n_restarts),
                                    max_nfev=int(max_nfev), n_arm_jobs=1))


def run_nls_fit(base_config, free_params, dataset, obs_keys, *, od_to_cfu=None,
                shared_groups=None, fitness_links=None, n_restarts=3, max_nfev=300):
    """Run pbisim-fit's NLS on the dataset. ``free_params`` = [(label, path, lo, hi,
    log_scale)]; optional ``shared_groups`` / ``fitness_links`` add a reparameterization
    layer (see build_param_spec). Returns the FitPosterior (``.map()``,
    ``.credible_interval()``, ``.to_config()``).

    ``od_to_cfu`` sets ``base_config.od_to_cfu_conversion_factor`` — pbisim-fit reads
    the OD link off the config (``_extract_obs_log10``), not the dataset metadata, so
    fitting the ``od`` observable REQUIRES it (else model-OD = biomass ≫ data-OD and
    the fit collapses to the parameter bounds)."""
    from pbisim_fit.refinement.nls import refine_nls, NLSConfig

    if od_to_cfu and "od" in obs_keys:
        try:
            base_config.od_to_cfu_conversion_factor = float(od_to_cfu)
        except Exception:
            pass
    cfg, pspec = build_param_spec(base_config, free_params, shared_groups, fitness_links)
    return refine_nls(dataset, pspec, cfg,
                      cfg=NLSConfig(obs_keys=list(obs_keys), n_restarts=int(n_restarts),
                                    max_nfev=int(max_nfev), n_arm_jobs=1))
