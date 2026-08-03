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
    # growth-signal-specific parameters (emitted only for the active growth function)
    "carrying_capacity": (1e6, 1e11, True),        # logistic density ceiling K
    "gompertz_sinf": (0.01, 5.0, True),            # gompertz inflection S∞ (nutrient scale)
    "density_growth_constant": (1e6, 1e11, True),  # density-throttle Kd
    "monod_K_low": (0.01, 50.0, True),             # smooth-efficiency inefficient (low-S) K
    "monod_efficiency_theta": (0.01, 0.99, False), # smooth-efficiency transition midpoint
    "monod_efficiency_hill": (1.0, 12.0, False),   # smooth-efficiency transition sharpness
    "growth_phase_rate_factors": (0.0, 2.0, False),  # diauxic per-phase rate factor
    "growth_phase_monod": (0.01, 5.0, True),         # diauxic per-phase Monod K
    "growth_phase_thresholds": (0.001, 0.999, False),# diauxic phase threshold θ
    # nutrient environment (when nutrients are tracked)
    "s_in": (0.0, 5.0, False),
    "s_out": (0.0, 2.0, False),
    "infected_nutrient_consumption": (0.0, 5.0, False),
    "monod_constant_lysis": (0.01, 5.0, True),     # frac_lysis nutrient half-saturation
    "dormancy_monod_constant": (0.01, 5.0, True),  # nutrient dormancy half-saturation
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
    # ── Growth-signal parameters — depend on the active growth function (which nutrient
    # constants it reads + any shape parameters it introduces). Curated per signal so a
    # Gompertz / smooth-efficiency / diauxic model exposes ITS estimables, not just Ks. ──
    _gfn = getattr(getattr(config, "growth_function", None), "__name__", "monod_growth")
    if _gfn in ("monod_growth", "monod_logistic_growth", "density_throttled_growth"):
        add("Monod half-saturation Ks", "monod_constant", "monod_constant")
    elif _gfn == "gompertz_growth":
        add("Gompertz shape k", "monod_constant", "monod_constant")
        add("Gompertz inflection S∞", "carrying_capacity", "gompertz_sinf")
    elif _gfn == "smooth_efficiency_monod":
        add("Efficient Monod K (high S)", "monod_constant", "monod_constant")
        add("Inefficient Monod K (low S)", "monod_K_low", "monod_K_low")
        add("Efficiency transition θ", "monod_efficiency_theta", "monod_efficiency_theta")
        add("Efficiency transition Hill", "monod_efficiency_hill", "monod_efficiency_hill")
    if _gfn in ("logistic_growth", "monod_logistic_growth"):
        add("Carrying capacity K (CFU/mL)", "carrying_capacity", "carrying_capacity")
    elif _gfn == "density_throttled_growth":
        add("Density throttle Kd (CFU/mL)", "density_growth_constant", "density_growth_constant")
    elif _gfn == "sequential_monod":
        _rf = getattr(config, "growth_phase_rate_factors", None)
        _mc = getattr(config, "growth_phase_monod", None)
        _th = getattr(config, "growth_phase_thresholds", None)
        for i in range(1, len(_rf) if _rf is not None else 0):  # phase 0 factor is pinned to 1.0
            add(f"Diauxic rate factor — phase {i+1}", f"growth_phase_rate_factors[{i}]", "growth_phase_rate_factors")
        for i in range(len(_mc) if _mc is not None else 0):
            add(f"Diauxic Monod K — phase {i+1}", f"growth_phase_monod[{i}]", "growth_phase_monod")
        for i in range(len(_th) if _th is not None else 0):
            add(f"Diauxic threshold θ{i+1}", f"growth_phase_thresholds[{i}]", "growth_phase_thresholds")
    for i in range(nb):
        for j in range(npg):
            add(f"Adsorption — strain {i} × phage {j}", f"adsorption_rates[{i},{j}]", "adsorption_rates")
            add(f"Burst size — strain {i} × phage {j}", f"burst_sizes[{i},{j}]", "burst_sizes")
            add(f"Latent period — strain {i} × phage {j} (h)", f"latent_periods[{i},{j}]", "latent_periods")
    _inc_per_phage = np.ndim(getattr(config, "infected_nutrient_consumption", 0.0)) > 0
    for j in range(npg):
        add(f"Phage decay rate — phage {j} (h⁻¹)", f"phage_decay_rates[{j}]", "phage_decay_rates")
        if _inc_per_phage:   # per-phage infected-cell nutrient draw (nutrient-tracking models)
            add(f"Infected-cell nutrient consumption — phage {j} (×)",
                f"infected_nutrient_consumption[{j}]", "infected_nutrient_consumption")
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
    # nutrient environment (only meaningful when nutrients are tracked)
    # infected_nutrient_consumption is emitted PER PHAGE in the phage loop above.
    if getattr(config, "track_nutrients", False):
        add("Nutrient inflow (s_in)", "s_in", "s_in")
        add("Nutrient washout (s_out)", "s_out", "s_out")
    # nutrient-coupled lysis / dormancy half-saturations, when those signals are active
    if getattr(getattr(config, "lysis_progression_function", None), "__name__", "") == "frac_lysis":
        add("Lysis Monod Ks (Ks_lysis)", "monod_constant_lysis", "monod_constant_lysis")
    if getattr(config, "dormancy_monod_constant", None):
        add("Dormancy Monod Ks", "dormancy_monod_constant", "dormancy_monod_constant")
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


# ── Fit-spec DSL (statement-based, two-way with the fix/free tables) ───────────
# Grammar (one statement per line; # comments):
#   free  <path>  [init=X] [bounds=LO..HI] [prior=MU,SD] [log]
#   fix   <path> = <value>
#   theta <name>  [init=X] [bounds=LO..HI] [prior=MU,SD] [log]
#   map   <path> = <expression of thetas>
# Unmentioned parameters stay fixed at their model value.

def _g(x):
    """Compact numeric render (scientific where sensible)."""
    try:
        return f"{float(x):g}"
    except (TypeError, ValueError):
        return str(x)


def _bounds_prior_tokens(lo, hi, log, mu, sd):
    toks = []
    lo, hi = float(lo), float(hi)
    _lo_unb = (not np.isfinite(lo)) or (log and lo <= 0)
    _hi_unb = not np.isfinite(hi)
    if not (_lo_unb and _hi_unb):
        toks.append("bounds=" + ("" if _lo_unb else _g(lo)) + ".." + ("" if _hi_unb else _g(hi)))
    if log:
        toks.append("log")
    if mu is not None and sd is not None:
        toks.append(f"prior={_g(mu)},{_g(sd)}")
    return toks


def _parse_opts(tokens):
    """Parse `key=value` / bare `log` option tokens into a dict."""
    o = {}
    for tk in tokens:
        if tk == "log":
            o["log"] = True
        elif "=" in tk:
            k, v = tk.split("=", 1)
            o[k.strip()] = v.strip()
        else:
            raise ValueError(f"unrecognised option {tk!r}")
    return o


def _opts_to_cells(o):
    """Map free/theta options to df string cells (value/initial handled by caller)."""
    cells = {}
    if "bounds" in o:
        b = o["bounds"]
        if ".." not in b:
            raise ValueError("bounds must be LO..HI (either side may be blank)")
        lo, hi = b.split("..", 1)
        cells["lower"], cells["upper"] = lo.strip(), hi.strip()
    if o.get("log"):
        cells["log"] = True
    if "prior" in o:
        pr = o["prior"]
        if "," not in pr:
            raise ValueError("prior must be MU,SD")
        mu, sd = pr.split(",", 1)
        cells["prior μ"], cells["prior σ"] = mu.strip(), sd.strip()
    return cells


def parse_fit_spec(text, catalog):
    """Parse DSL text into (targets_df, thetas_df, errors), matching the unified
    parameter table (role Fixed/Free/Derived + expression) and the thetas table.
    ``catalog`` = available_targets(config). Mappings live on the target rows as
    role='Derived' + an expression, so there is no separate map dataframe."""
    import pandas as pd
    by_path = {p: (lab, v, lo, hi, log) for (lab, p, v, lo, hi, log) in catalog}
    errors = []
    rows = {p: {"parameter": lab, "path": p, "role": "Fixed", "value": f"{v:g}",
                "lower": "", "upper": "", "log": bool(log), "prior μ": "", "prior σ": "",
                "expression": ""}
            for (lab, p, v, lo, hi, log) in catalog}
    thetas, theta_names = [], set()

    for ln, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            head, _, rest = line.partition(" ")
            head, rest = head.lower(), rest.strip()
            if head == "free":
                toks = rest.split()
                path = toks[0]
                if path not in by_path:
                    raise ValueError(f"unknown parameter {path!r}")
                o = _parse_opts(toks[1:])
                r = rows[path]; r["role"] = "Free"
                if "init" in o:
                    r["value"] = o["init"]
                r.update(_opts_to_cells(o))
            elif head == "fix":
                path, eq, val = rest.partition("=")
                if not eq:
                    raise ValueError("fix needs '= value'")
                path = path.strip()
                if path not in by_path:
                    raise ValueError(f"unknown parameter {path!r}")
                rows[path]["role"] = "Fixed"
                rows[path]["value"] = val.strip()
            elif head == "theta":
                toks = rest.split()
                name = toks[0]
                if not name.isidentifier():
                    raise ValueError(f"bad theta name {name!r}")
                o = _parse_opts(toks[1:])
                th = {"name": name, "lower": "", "upper": "", "log": False,
                      "initial": (o.get("init", "")), "prior μ": "", "prior σ": ""}
                th.update(_opts_to_cells(o))
                thetas.append(th); theta_names.add(name)
            elif head == "map":
                path, eq, expr = rest.partition("=")
                if not eq:
                    raise ValueError("map needs '= expression'")
                path = path.strip()
                if path not in by_path:
                    raise ValueError(f"unknown parameter {path!r}")
                rows[path]["role"] = "Derived"
                rows[path]["expression"] = expr.strip()
            else:
                raise ValueError(f"unknown statement {head!r} (use free / fix / theta / map)")
        except Exception as e:  # noqa: BLE001
            errors.append(f"line {ln}: {e}")

    for r in rows.values():
        if r["role"] == "Derived" and r["expression"]:
            ok, msg = validate_expr(r["expression"], list(theta_names))
            if not ok:
                errors.append(f"{r['path']} = {r['expression']}: invalid expression ({msg})")

    tdf = pd.DataFrame(list(rows.values()))
    thdf = (pd.DataFrame(thetas) if thetas else
            pd.DataFrame(columns=["name", "lower", "upper", "log", "initial", "prior μ", "prior σ"]))
    return tdf, thdf, errors


def serialize_fit_spec(targets, thetas, mappings, catalog):
    """Serialize the current fit spec (the view's _targets/_thetas/_mappings) to DSL
    text, prefixed with a commented list of the model's available parameters so the
    paths are always to hand. Fixed params left at the model value are omitted."""
    default = {p: v for (lab, p, v, lo, hi, log) in catalog}
    all_paths = [p for (lab, p, *_r) in catalog]
    lines = ["# available parameters (this model):", "#   " + ", ".join(all_paths), ""]
    for th in thetas:
        parts = [f"theta {th['name']}"]
        if th.get("initial") is not None:
            parts.append(f"init={_g(th['initial'])}")
        parts += _bounds_prior_tokens(th["lo"], th["hi"], th.get("log"),
                                      th.get("prior_mu"), th.get("prior_sd"))
        lines.append(" ".join(parts))
    mapped = {m["path"] for m in mappings}
    for t in targets:
        if t["path"] in mapped:
            continue
        if t["free"]:
            parts = [f"free {t['path']}"]
            if t.get("value") is not None:
                parts.append(f"init={_g(t['value'])}")
            parts += _bounds_prior_tokens(t["lo"], t["hi"], t.get("log"),
                                          t.get("prior_mu"), t.get("prior_sd"))
            lines.append(" ".join(parts))
        else:
            dv = default.get(t["path"])
            if dv is None or abs(float(t["value"]) - float(dv)) > 1e-30:
                lines.append(f"fix {t['path']} = {_g(t['value'])}")
    for m in mappings:
        lines.append(f"map {m['path']} = {m['expr']}")
    return "\n".join(lines)


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


def build_dataset(agg, sel_arms, sel_obs, arm_cond, *, od_to_cfu=None, dose_unit="moi",
                  arm_doses=None, arm_covariates=None):
    """Construct a pbisim-fit ExperimentalDataset from the app's aggregated calibration
    data + per-arm conditions (growth-phase pre-run, B₀, phage dose).

    ``dose_unit`` is how each arm's manual dose value is interpreted: ``"moi"`` (× the
    arm's B₀) or ``"pfu"`` (absolute PFU/mL). ``arm_doses`` = ``{arm: [{time, target,
    amount, unit}]}`` imported from NONMEM/Monolix dose rows; these are emitted verbatim
    and, for whichever targets they specify, override the manual per-arm dose/inoculum.
    ``arm_covariates`` = ``{arm: {name: value}}`` attached to each TreatmentRecord for
    per-arm covariate-link fitting (MOI also resolves from the phage dose automatically)."""
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
        # Imported NONMEM/Monolix dose rows — emitted verbatim; the targets they cover
        # are then NOT re-emitted from the manual per-arm fields.
        _dsdoses = (arm_doses or {}).get(a, [])
        _ds_targets = set()
        for d in _dsdoses:
            doses.append(DoseRecord(time=float(d.get("time", 0.0)), amount=float(d["amount"]),
                                    unit=(d.get("unit") or "cfu"), target=d["target"]))
            _ds_targets.add(d["target"])
        # Manual phage dose (only when the dataset didn't specify one for this arm).
        _dose = float(cond.get("moi", 0.0) or 0.0)   # dose value in `dose_unit`
        if _dose > 0 and "phage" not in _ds_targets:
            doses.append(DoseRecord(time=0.0, amount=_dose,
                                    unit=("pfu" if dose_unit == "pfu" else "moi"), target="phage"))
        # Additive-B0 model: a KNOWN inoculum (Shared / Per-arm mode → b0_is_dose) is a
        # t=0 bacteria dose (like the phage dose), consumed literally by pbisim-fit's
        # _solve_arm. A pre-run arm instead carries it as the pre-run's fresh inoculum
        # (pretreatment_inoculum). First-observation mode records NEITHER, so pbisim-fit
        # falls back to cfu[0] (its baseline warning; we already warn in the UI).
        pr = float(cond.get("prerun", 0.0) or 0.0)
        b0 = float(cond.get("b0", 0.0) or 0.0)
        _b0_is_dose = bool(cond.get("b0_is_dose", False)) and b0 > 0 and "bacteria" not in _ds_targets
        if _b0_is_dose and pr <= 0:
            doses.append(DoseRecord(time=0.0, amount=b0, unit="cfu", target="bacteria"))
        _cov = {k: float(v) for k, v in (arm_covariates or {}).get(a, {}).items()
                if v is not None}
        arms.append(TreatmentRecord(
            label=str(a), dose_events=doses,
            pretreatment_h=(pr if pr > 0 else None),
            pretreatment_inoculum=(b0 if (_b0_is_dose and pr > 0) else None),
            covariates=_cov,
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


def build_param_spec_v2(base_config, targets, thetas=None, mappings=None,
                        *, dataset=None, estimate_b0="none", covariate_effects=None):
    """Build a parameter spec from the table-driven UI.

    ``estimate_b0`` ("shared" | "per_arm") wires an estimated additive B0 offset via
    ``free_initial_conditions`` (needs ``dataset``); the role-table ``fit_initial_cfu``
    target, if any, is then skipped so the two don't double-wire.

    ``covariate_effects`` = [{"path","covariate","form","ref","beta_lo","beta_hi",
    "beta_init"}] — per-arm covariate links (NONMEM/Monolix style ``θ_i = θ_ref·(cov/ref)^β``)
    wired via pbisim-fit's ``with_covariate``; each adds one estimated β to the fit.

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
        if estimate_b0 in ("shared", "per_arm") and p == "fit_initial_cfu":
            continue  # handled by free_initial_conditions below (the B0-source radio)
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
    if estimate_b0 in ("shared", "per_arm") and dataset is not None:
        from pbisim_fit import free_initial_conditions
        rp = free_initial_conditions(rp, dataset, cfu=estimate_b0)
    # Per-arm covariate links: each adds one estimated β so a path varies by the arm's
    # covariate (θ_i = θ_ref·(cov/ref)^β). with_covariate registers β + attaches the
    # CovariateEffect to the Reparam's base config.
    for e in (covariate_effects or []):
        from pbisim_fit import with_covariate
        rp = with_covariate(
            rp, e["path"], e["covariate"], form=e.get("form", "power"),
            ref=float(e.get("ref", 1.0) or 1.0),
            beta_bounds=(float(e.get("beta_lo", -3.0)), float(e.get("beta_hi", 3.0))),
            beta_init=float(e.get("beta_init", 0.0) or 0.0))
    return rp.build()


def run_nls_fit_v2(base_config, targets, thetas, mappings, dataset, obs_keys, *,
                   od_to_cfu=None, n_restarts=3, max_nfev=300, estimate_b0="none",
                   obs_compartments=None, covariate_effects=None):
    """Run NLS from the table-driven spec (see build_param_spec_v2). ``estimate_b0``
    ("shared"|"per_arm") estimates an additive B0 offset via free_initial_conditions.

    ``obs_compartments`` = the observation model: ``{obs_key: (prefixes...)}`` (e.g.
    ``{"cfu": ("B","D")}`` — culturable CFU excludes non-culturable I/H). The CFU set is
    threaded to pbisim-fit via ``NLSConfig.cfu_compartments`` so the FIT residual uses the
    SAME compartments as the app overlay. (pbisim-fit already defaults CFU to B+D; this only
    matters when the user overrides the set. Older pbisim-fit without the field → warn.)

    ``covariate_effects`` = per-arm covariate links (see build_param_spec_v2). Requires
    pbisim-fit's ``with_covariate``; if the installed version lacks it we warn and drop the
    links (the fit still runs, without the per-arm modulation)."""
    from pbisim_fit.refinement.nls import refine_nls, NLSConfig
    if covariate_effects:
        try:
            from pbisim_fit import with_covariate  # noqa: F401 — feature-detect
        except ImportError:
            import warnings
            warnings.warn(
                "Installed pbisim-fit lacks with_covariate — covariate-link effects are "
                "ignored (the fit runs without per-arm covariate modulation). Update pbisim-fit.",
                RuntimeWarning)
            covariate_effects = None
    if od_to_cfu and "od" in obs_keys:
        try:
            base_config.od_to_cfu_conversion_factor = float(od_to_cfu)
        except Exception:
            pass
    # Multi-start sampling can't draw from an infinite range → force a single start
    # (from the initial) whenever any freed/theta parameter is unbounded.
    if has_unbounded(targets, thetas):
        n_restarts = 1
    cfg, pspec = build_param_spec_v2(base_config, targets, thetas, mappings,
                                     dataset=dataset, estimate_b0=estimate_b0,
                                     covariate_effects=covariate_effects)
    _nls_kw = dict(obs_keys=list(obs_keys), n_restarts=int(n_restarts),
                   max_nfev=int(max_nfev), n_arm_jobs=1)
    _cfu_comps = (obs_compartments or {}).get("cfu")
    if _cfu_comps:
        try:                       # feature-detect pbisim-fit's cfu_compartments field
            NLSConfig(cfu_compartments=tuple(_cfu_comps))
            _nls_kw["cfu_compartments"] = tuple(_cfu_comps)
        except TypeError:
            import warnings
            warnings.warn(
                "Installed pbisim-fit predates NLSConfig.cfu_compartments — the CFU fit "
                "residual will use its built-in compartment set, which may differ from the "
                "app's observation model (overlay ≠ fit). Update pbisim-fit.", RuntimeWarning)
    return refine_nls(dataset, pspec, cfg, cfg=NLSConfig(**_nls_kw))


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
