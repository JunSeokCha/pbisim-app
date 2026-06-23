"""
presets.py — Preset configurations for the pbisim-app.
Contains structured parameters and custom scripts representing the 13 pbisim tutorials.
"""

from __future__ import annotations

TUTORIALS = [
    {
        "id": "tutorial_01",
        "name": "Tutorial 01: First Simulation",
        "description": """
        **First Simulation: Build → Assemble → Integrate → Read**
        
        This preset introduces the basic pbisim workflow:
        - 1 bacterial strain growing on a nutrient substrate (Monod kinetics).
        - challenged by 1 free-phage species.
        - no immunity, no antibiotics, no dormancy.
        """,
        "type": "single",
        "parameters": {
            "t_end": 48.0,
            "dt": 0.25,
            "extinction_threshold": 1.0,
            "solver_method": "BDF",
            "track_nutrients": True,
            "initial_S": 1.0,
            "monod_constant": 0.3,
            "recycle_fraction": 0.0,
            "s_in": 0.0,
            "s_out": 0.0,
            "immunity_enabled": False,
            "debris_enabled": False,
            "strains": [
                {
                    "name": "Strain 0 (WT)",
                    "initial_B": 1e7,
                    "growth_rate": 1.2,
                    "bacteria_to_resource_ratio": 1e9,
                    "death_rate_B": 0.0,
                    "dormancy_enabled": False,
                }
            ],
            "phages": [
                {
                    "name": "Phage 0",
                    "initial_P": 1e6,
                    "adsorption_rates": 1e-8,
                    "adsorption_rates_dormant": 0.0,
                    "burst_sizes": 50.0,
                    "latent_periods": 0.5,
                    "phage_decay_rates": 0.1,
                    "pk_mode": "None",
                }
            ],
            "antibiotics": [],
            "doses": [],
        },
        "script_code": """# Tutorial 01: First Simulation
import numpy as np
import matplotlib.pyplot as plt
from pbisim import ModelBuilder, PBIModel, solve_ode
from pbisim.analysis.single import plot_simulation

cfg = (
    ModelBuilder(n_bacteria=1, n_phages=1, n_latent=5)
    .with_growth_rates(1.2, bacteria_to_resource_ratio=1e9)
    .with_phage_params(
        adsorption_rates=1e-8,
        burst_sizes=50.0,
        latent_periods=0.5,
        phage_decay_rates=0.1
    )
    .with_nutrient(monod_constant=0.3, recycle_fraction=0.0)
    .build()
)

model = PBIModel(cfg, initial_B=np.array([1e7]), initial_P=np.array([1e6]), initial_S=1.0)
result = solve_ode(model, t_end=48.0, dt=0.25)

fig, axes = plot_simulation(result)
fig.suptitle("Tutorial 01 — Single strain, single phage", y=1.02)
plt.tight_layout()
""",
    },
    {
        "id": "tutorial_02",
        "name": "Tutorial 02: Growth & Nutrients",
        "description": """
        **Growth Functions and Nutrient Dynamics**
        
        This preset compares density-limited and nutrient-limited growth:
        - Uses `logistic_growth` (nutrient tracking disabled, S is frozen).
        - Model limited by carrying capacity ($K = 10^9$ cells/mL).
        - Includes natural cell death ($d_B = 0.05$ h⁻¹) and recycling fraction.
        """,
        "type": "single",
        "parameters": {
            "t_end": 48.0,
            "dt": 0.5,
            "extinction_threshold": 1.0,
            "solver_method": "BDF",
            "track_nutrients": False,
            "growth_function": "logistic_growth",
            "carrying_capacity": 1e9,
            "immunity_enabled": False,
            "debris_enabled": False,
            "strains": [
                {
                    "name": "Strain 0 (Logistic)",
                    "initial_B": 1e5,
                    "growth_rate": 1.2,
                    "bacteria_to_resource_ratio": 1e9,
                    "death_rate_B": 0.05,
                    "dormancy_enabled": False,
                }
            ],
            "phages": [],
            "antibiotics": [],
            "doses": [],
        },
        "script_code": """# Tutorial 02: Growth & Nutrients (Logistic Growth)
import numpy as np
import matplotlib.pyplot as plt
from pbisim import ModelBuilder, PBIModel, solve_ode, logistic_growth
from pbisim.analysis.single import plot_simulation

cfg = (
    ModelBuilder(n_bacteria=1, n_phages=0)
    .with_growth_function(logistic_growth)
    .with_nutrient(carrying_capacity=1e9, track_nutrients=False)
    .build()
)

model = PBIModel(cfg, initial_B=np.array([1e5]), initial_P=np.zeros(0), initial_S=1.0)
result = solve_ode(model, t_end=48.0, dt=0.5)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(result.time, result.get("B0"), label="Bacteria (cells/mL)", lw=2)
ax.axhline(1e9, color="r", ls=":", label="Carrying Capacity")
ax.set(xlabel="Time (h)", ylabel="Density", title="Logistic Bacterial Growth (No Nutrients)")
ax.legend()
plt.tight_layout()
""",
    },
    {
        "id": "tutorial_03",
        "name": "Tutorial 03: Dormancy & Resuscitation",
        "description": """
        **Dormancy Depth Layers and Resuscitation**
        
        Models reversible entry into a dormant state:
        - 4 depth layers ($Q=4$) modeling progressive shutdown.
        - symmetric diffusion between layers ($D_{diff} = 0.05$ h⁻¹).
        - dormant bacteria are $200\\times$ less susceptible to phage infection ($10^{-11}$ mL/h vs $2\\times 10^{-9}$ mL/h).
        """,
        "type": "single",
        "parameters": {
            "t_end": 72.0,
            "dt": 0.25,
            "extinction_threshold": 1.0,
            "solver_method": "BDF",
            "track_nutrients": True,
            "initial_S": 1.0,
            "monod_constant": 0.3,
            "recycle_fraction": 0.0,
            "s_in": 0.0,
            "s_out": 0.0,
            "immunity_enabled": False,
            "debris_enabled": False,
            "strains": [
                {
                    "name": "Strain 0",
                    "initial_B": 1e7,
                    "growth_rate": 1.2,
                    "bacteria_to_resource_ratio": 1e9,
                    "death_rate_B": 0.0,
                    "dormancy_enabled": True,
                    "dormancy_depth": 4,
                    "dormancy_rate": 0.2,
                    "resuscitation_rate": 0.1,
                    "dormancy_diffusion_rate": 0.05,
                    "dormancy_signal": "nutrient",
                    "resuscitation_signal": "nutrient",
                }
            ],
            "phages": [
                {
                    "name": "Phage 0",
                    "initial_P": 1e6,
                    "adsorption_rates": 1e-8,
                    "adsorption_rates_dormant": 1e-11,
                    "burst_sizes": 50.0,
                    "latent_periods": 0.5,
                    "phage_decay_rates": 0.1,
                    "pk_mode": "None",
                }
            ],
            "antibiotics": [],
            "doses": [],
        },
        "script_code": """# Tutorial 03: Dormancy
import numpy as np
import matplotlib.pyplot as plt
from pbisim import ModelBuilder, PBIModel, solve_ode
from pbisim.analysis.single import plot_compartments

cfg = (
    ModelBuilder(n_bacteria=1, n_phages=1, n_latent=5, n_depth=4)
    .with_growth_rates(1.2, bacteria_to_resource_ratio=1e9)
    .with_phage_params(
        adsorption_rates=1e-8,
        burst_sizes=50.0,
        latent_periods=0.5,
        phage_decay_rates=0.1,
        adsorption_rates_dormant=1e-11
    )
    .with_dormancy(
        dormancy_rate=0.2,
        resuscitation_rate=0.1,
        dormancy_diffusion_rate=0.05,
        dormancy_signal="nutrient",
        resuscitation_signal="nutrient"
    )
    .build()
)

model = PBIModel(cfg, initial_B=np.array([1e7]), initial_P=np.array([1e6]), initial_S=1.0)
result = solve_ode(model, t_end=72.0, dt=0.25)

fig, ax = plot_compartments(result, strain_idx=0)
fig.suptitle("Tutorial 03 — Dormancy Depth Compartments", y=1.02)
plt.tight_layout()
""",
    },
    {
        "id": "tutorial_04",
        "name": "Tutorial 04: Immune Response & Synergy",
        "description": """
        **Innate Immunity and Phage Synergy**
        
        Models immune clearance of bacteria:
        - Innate immune cells kill active bacteria ($kill_{rate} = 10^7$ h⁻¹).
        - Non-linear Hill-type killing (half-saturation $K_{kill} = 10^8$ cells/mL).
        - Phage dosing triggers clearance by driving bacteria density below the immune threshold.
        """,
        "type": "single",
        "parameters": {
            "t_end": 48.0,
            "dt": 0.25,
            "extinction_threshold": 1.0,
            "solver_method": "BDF",
            "track_nutrients": True,
            "initial_S": 1.0,
            "monod_constant": 0.3,
            "recycle_fraction": 0.0,
            "s_in": 0.0,
            "s_out": 0.0,
            "immunity_enabled": True,
            "immune_module": "innate",
            "imm_stim_rate": 1.0,
            "imm_stim50": 1e6,
            "innate_kill_rate": 1e7,
            "innate_kill50": 1e8,
            "innate_decay_rate": 0.05,
            "innate_max": 1e7,
            "imm_initial": 0.0,
            "debris_enabled": False,
            "strains": [
                {
                    "name": "Strain 0 (Infection)",
                    "initial_B": 1e7,
                    "growth_rate": 1.2,
                    "bacteria_to_resource_ratio": 1e9,
                    "death_rate_B": 0.0,
                    "dormancy_enabled": False,
                }
            ],
            "phages": [
                {
                    "name": "Phage 0",
                    "initial_P": 0.0,
                    "adsorption_rates": 1e-8,
                    "adsorption_rates_dormant": 0.0,
                    "burst_sizes": 50.0,
                    "latent_periods": 0.5,
                    "phage_decay_rates": 0.1,
                    "pk_mode": "None",
                }
            ],
            "antibiotics": [],
            "doses": [
                {
                    "time": 2.0,
                    "amount": 1e8,
                    "target_type": "phage",
                    "target_idx": 0,
                    "route": "bolus",
                    "duration": 0.0,
                }
            ],
        },
        "script_code": """# Tutorial 04: Innate Immunity and Phage Synergy
import numpy as np
import matplotlib.pyplot as plt
from pbisim import ModelBuilder, PBIModel, solve_ode, DoseSchedule, DoseEvent
from pbisim.analysis.single import plot_simulation

cfg = (
    ModelBuilder(n_bacteria=1, n_phages=1, n_latent=5)
    .with_growth_rates(1.2, bacteria_to_resource_ratio=1e9)
    .with_phage_params(
        adsorption_rates=1e-8,
        burst_sizes=50.0,
        latent_periods=0.5,
        phage_decay_rates=0.1
    )
    .with_immunity(
        imm_kill_rate=1e7,
        imm_kill50=1e8,
        imm_max=1e7
    )
    .with_dose_schedule(DoseSchedule([
        DoseEvent(time=2.0, amount=1e8, target="phage")
    ]))
    .build()
)

model = PBIModel(cfg, initial_B=np.array([1e7]), initial_P=np.array([0.0]), initial_S=1.0)
result = solve_ode(model, t_end=48.0, dt=0.25)

fig, axes = plot_simulation(result)
fig.suptitle("Tutorial 04 — Innate Immune Synergy", y=1.02)
plt.tight_layout()
""",
    },
    {
        "id": "tutorial_05",
        "name": "Tutorial 05: Phage Pharmacokinetics",
        "description": """
        **Phage PK site-delivery and peripheral distribution**
        
        This preset models systemic intravenous (IV) delivery of phage:
        - Uses the bidirectional **Mass-Conserving PK Mode** (`Vi` specified).
        - Includes phage central blood compartment $Pc$ and distribution to tissues.
        - Phage is dosed systemically into the central compartment.
        """,
        "type": "single",
        "parameters": {
            "t_end": 48.0,
            "dt": 0.1,
            "extinction_threshold": 1.0,
            "solver_method": "BDF",
            "track_nutrients": True,
            "initial_S": 1.0,
            "monod_constant": 0.3,
            "recycle_fraction": 0.0,
            "s_in": 0.0,
            "s_out": 0.0,
            "immunity_enabled": False,
            "debris_enabled": False,
            "strains": [
                {
                    "name": "Strain 0",
                    "initial_B": 1e7,
                    "growth_rate": 1.2,
                    "bacteria_to_resource_ratio": 1e9,
                    "death_rate_B": 0.0,
                    "dormancy_enabled": False,
                }
            ],
            "phages": [
                {
                    "name": "IV Phage",
                    "initial_P": 0.0,
                    "adsorption_rates": 1e-8,
                    "adsorption_rates_dormant": 0.0,
                    "burst_sizes": 50.0,
                    "latent_periods": 0.5,
                    "phage_decay_rates": 0.1,
                    "pk_mode": "Mass-Conserving",
                    "Vc": 5000.0,
                    "k_elim": 0.2,
                    "k_in": 0.1,
                    "k_out": 0.05,
                    "Vi": 10.0,
                }
            ],
            "antibiotics": [],
            "doses": [
                {
                    "time": 0.0,
                    "amount": 1e11,
                    "target_type": "phage",
                    "target_idx": 0,
                    "route": "bolus",
                    "duration": 0.0,
                }
            ],
        },
        "script_code": """# Tutorial 05: Phage PK (Mass-Conserving Mode)
import numpy as np
import matplotlib.pyplot as plt
from pbisim import ModelBuilder, PBIModel, solve_ode, DoseSchedule, DoseEvent, PhagePKConfig
from pbisim.analysis.single import plot_simulation

# Phage PK configuration
phage_pk = PhagePKConfig(
    Vc=5000.0,       # central volume mL (5 L)
    k_elim=0.2,      # blood elimination rate h^-1
    k_in=0.1,        # transfer blood -> infection site h^-1
    k_out=0.05,      # transfer infection site -> blood h^-1
    Vi=np.array([10.0]) # volume of infection site (10 mL)
)

cfg = (
    ModelBuilder(n_bacteria=1, n_phages=1)
    .with_growth_rates(1.2, bacteria_to_resource_ratio=1e9)
    .with_phage_params(
        adsorption_rates=1e-8,
        burst_sizes=50.0,
        latent_periods=0.5,
        phage_decay_rates=0.1,
        pk_config=phage_pk
    )
    .with_dose_schedule(DoseSchedule([
        # Dose phage systemically (amount is total phage count)
        DoseEvent(time=0.0, amount=1e11, target="phage")
    ]))
    .build()
)

model = PBIModel(cfg, initial_B=np.array([1e7]), initial_P=np.array([0.0]), initial_S=1.0)
result = solve_ode(model, t_end=48.0, dt=0.1)

# Plot blood vs tissue phage
fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(result.time, result.get("Pc0")/5000.0, label="Blood Conc (phages/mL)", color="C3")
ax.semilogy(result.time, result.get("P0"), label="Infection Site Conc (phages/mL)", color="C0")
ax.set(xlabel="Time (h)", ylabel="Concentration", title="IV Phage PK: Systemic and Local Dynamics")
ax.legend()
plt.tight_layout()
""",
    },
    {
        "id": "tutorial_06",
        "name": "Tutorial 06: Antibiotic PK/PD",
        "description": """
        **Antibiotic PK/PD & Combination Therapy**
        
        Models antibiotic pharmacokinetics and pharmacodynamics:
        - 2-compartment antibiotic PK ($V_c$, $k_{elim}$, $k_{12}$, $k_{21}$).
        - Non-linear PD (Hill equation: $E_{max}$, $EC_{50}$, Hill coefficient).
        - Models combination therapy of phage + antibiotic.
        """,
        "type": "single",
        "parameters": {
            "t_end": 48.0,
            "dt": 0.25,
            "extinction_threshold": 1.0,
            "solver_method": "BDF",
            "track_nutrients": True,
            "initial_S": 1.0,
            "monod_constant": 0.3,
            "recycle_fraction": 0.0,
            "s_in": 0.0,
            "s_out": 0.0,
            "immunity_enabled": False,
            "debris_enabled": False,
            "strains": [
                {
                    "name": "Strain 0",
                    "initial_B": 1e7,
                    "growth_rate": 1.2,
                    "bacteria_to_resource_ratio": 1e9,
                    "death_rate_B": 0.0,
                    "dormancy_enabled": False,
                }
            ],
            "phages": [],
            "antibiotics": [
                {
                    "name": "Cipro",
                    "Vc": 250.0,
                    "k_elim": 0.3,
                    "k12": 0.2,
                    "k21": 0.1,
                    "emax": 3.0,
                    "ec50": 0.2,
                    "hill": 1.5,
                    "inoculum_effect_constant": 0.0,
                    "f_lyse": 0.0,
                }
            ],
            "doses": [
                {
                    "time": 0.0,
                    "amount": 500.0,
                    "target_type": "antibiotic",
                    "target_idx": 0,
                    "route": "bolus",
                    "duration": 0.0,
                },
                {
                    "time": 12.0,
                    "amount": 500.0,
                    "target_type": "antibiotic",
                    "target_idx": 0,
                    "route": "bolus",
                    "duration": 0.0,
                },
            ],
        },
        "script_code": """# Tutorial 06: Antibiotic PK/PD
import numpy as np
import matplotlib.pyplot as plt
from pbisim import ModelBuilder, PBIModel, solve_ode, DoseSchedule, DoseEvent
from pbisim.analysis.single import plot_simulation

cfg = (
    ModelBuilder(n_bacteria=1, n_phages=0)
    .with_growth_rates(1.2, bacteria_to_resource_ratio=1e9)
    .with_antibiotic(
        name="Cipro",
        Vc=250.0, k_elim=0.3, k12=0.2, k21=0.1,
        emax=3.0, ec50=0.2, hill=1.5
    )
    .with_dose_schedule(DoseSchedule([
        DoseEvent(time=0.0, amount=500.0, target="antibiotic"),
        DoseEvent(time=12.0, amount=500.0, target="antibiotic")
    ]))
    .build()
)

model = PBIModel(cfg, initial_B=np.array([1e7]), initial_P=np.zeros(0), initial_S=1.0)
result = solve_ode(model, t_end=48.0, dt=0.25)

fig, axes = plot_simulation(result)
fig.suptitle("Tutorial 06 — Antibiotic Only PK/PD", y=1.02)
plt.tight_layout()
""",
    },
    {
        "id": "tutorial_07",
        "name": "Tutorial 07: Dosing Schedules",
        "description": """
        **Dosing Regimens: Bolus, Infusions, and Multi-dose**
        
        This preset configures complex dosing schedules:
        - 1 antibiotic dosed via **constant-rate IV infusion** (duration = 2 hours).
        - 1 phage dosed via bolus injections.
        - demonstrates how to build and assign a multi-event `DoseSchedule`.
        """,
        "type": "single",
        "parameters": {
            "t_end": 24.0,
            "dt": 0.1,
            "extinction_threshold": 1.0,
            "solver_method": "BDF",
            "track_nutrients": True,
            "initial_S": 1.0,
            "monod_constant": 0.3,
            "recycle_fraction": 0.0,
            "s_in": 0.0,
            "s_out": 0.0,
            "immunity_enabled": False,
            "debris_enabled": False,
            "strains": [
                {
                    "name": "Strain 0",
                    "initial_B": 1e7,
                    "growth_rate": 1.2,
                    "bacteria_to_resource_ratio": 1e9,
                    "death_rate_B": 0.0,
                    "dormancy_enabled": False,
                }
            ],
            "phages": [
                {
                    "name": "Phage 0",
                    "initial_P": 0.0,
                    "adsorption_rates": 1e-8,
                    "adsorption_rates_dormant": 0.0,
                    "burst_sizes": 50.0,
                    "latent_periods": 0.5,
                    "phage_decay_rates": 0.1,
                    "pk_mode": "None",
                }
            ],
            "antibiotics": [
                {
                    "name": "Drug A",
                    "Vc": 200.0,
                    "k_elim": 0.4,
                    "emax": 2.5,
                    "ec50": 0.5,
                    "hill": 1.0,
                }
            ],
            "doses": [
                {
                    "time": 0.0,
                    "amount": 1000.0,
                    "target_type": "antibiotic",
                    "target_idx": 0,
                    "route": "infusion",
                    "duration": 2.0,
                },
                {
                    "time": 4.0,
                    "amount": 1e8,
                    "target_type": "phage",
                    "target_idx": 0,
                    "route": "bolus",
                    "duration": 0.0,
                },
            ],
        },
        "script_code": """# Tutorial 07: Dosing Schedules
import numpy as np
import matplotlib.pyplot as plt
from pbisim import ModelBuilder, PBIModel, solve_ode, DoseSchedule, DoseEvent
from pbisim.analysis.single import plot_simulation

cfg = (
    ModelBuilder(n_bacteria=1, n_phages=1)
    .with_growth_rates(1.2, bacteria_to_resource_ratio=1e9)
    .with_phage_params(adsorption_rates=1e-8, burst_sizes=50.0, latent_periods=0.5, phage_decay_rates=0.1)
    .with_antibiotic(name="Drug A", Vc=200.0, k_elim=0.4, emax=2.5, ec50=0.5, hill=1.0)
    .with_dose_schedule(DoseSchedule([
        # 2-hour infusion of antibiotic starting at t=0
        DoseEvent(time=0.0, amount=1000.0, target="antibiotic", route="infusion", duration=2.0),
        # Bolus injection of phage at t=4
        DoseEvent(time=4.0, amount=1e8, target="phage")
    ]))
    .build()
)

model = PBIModel(cfg, initial_B=np.array([1e7]), initial_P=np.array([0.0]), initial_S=1.0)
result = solve_ode(model, t_end=24.0, dt=0.1)

fig, axes = plot_simulation(result)
fig.suptitle("Tutorial 07 — Infusion & Bolus Dosing Schedule", y=1.02)
plt.tight_layout()
""",
    },
    {
        "id": "tutorial_08",
        "name": "Tutorial 08: Multi-Strain Models & StrainSet",
        "description": """
        **StrainSet API for Clinical Isolates**
        
        This preset configures multiple bacterial strains using the StrainSet API:
        - Models a wild-type (WT) strain and a resistant mutant.
        - The resistant mutant has a fitness cost (reduced growth rate) and is insensitive to Phage 0.
        - Demonstrates how to build multi-strain setups using `StrainSet`.
        """,
        "type": "single",
        "parameters": {
            "t_end": 48.0,
            "dt": 0.25,
            "extinction_threshold": 1.0,
            "solver_method": "BDF",
            "track_nutrients": True,
            "initial_S": 1.0,
            "monod_constant": 0.3,
            "recycle_fraction": 0.0,
            "s_in": 0.0,
            "s_out": 0.0,
            "immunity_enabled": False,
            "debris_enabled": False,
            "strains": [
                {
                    "name": "WT",
                    "initial_B": 1e7,
                    "growth_rate": 1.2,
                    "bacteria_to_resource_ratio": 1e9,
                    "death_rate_B": 0.0,
                    "dormancy_enabled": False,
                },
                {
                    "name": "Resistant",
                    "initial_B": 1e3,
                    "growth_rate": 0.9,  # fitness cost
                    "bacteria_to_resource_ratio": 1e9,
                    "death_rate_B": 0.0,
                    "dormancy_enabled": False,
                },
            ],
            "phages": [
                {
                    "name": "Phage 0",
                    "initial_P": 1e6,
                    # Phage 0 adsorbs WT at 1e-8 but Resistant at 1e-12 (resistant)
                    "adsorption_rates": [1e-8, 1e-12],
                    "adsorption_rates_dormant": 0.0,
                    "burst_sizes": 50.0,
                    "latent_periods": 0.5,
                    "phage_decay_rates": 0.1,
                    "pk_mode": "None",
                }
            ],
            "antibiotics": [],
            "doses": [],
        },
        "script_code": """# Tutorial 08: Multi-Strain with ModelBuilder
import numpy as np
import matplotlib.pyplot as plt
from pbisim import ModelBuilder, PBIModel, solve_ode
from pbisim.analysis.single import plot_simulation

cfg = (
    ModelBuilder(n_bacteria=2, n_phages=1, n_latent=5)
    .with_growth_rates([1.2, 0.9], bacteria_to_resource_ratio=1e9)
    .with_phage_params(
        adsorption_rates=np.array([[1e-8], [1e-12]]),
        burst_sizes=50.0,
        latent_periods=0.5,
        phage_decay_rates=0.1,
    )
    .with_nutrient(monod_constant=0.3)
    .build()
)

model = PBIModel(cfg, initial_B=np.array([1e7, 1e3]), initial_P=np.array([1e6]), initial_S=1.0)
result = solve_ode(model, t_end=48.0, dt=0.25)

fig, axes = plot_simulation(result, strain_labels=["WT", "Resistant"])
fig.suptitle("Tutorial 08 — WT vs Resistant Strain Dynamics", y=1.02)
plt.tight_layout()
""",
    },
    {
        "id": "tutorial_09",
        "name": "Tutorial 09: Binary Resistance Genotypes",
        "description": """
        **Resistance Genotype Space (BRG)**
        
        Enumerates $2^m$ genotypes for resistance to $m$ therapeutic agents:
        - $m=2$ loci (1 phage locus and 1 antibiotic locus), creating 4 bacterial genotypes:
          - `S_S` (WT, sensitive to both)
          - `R_S` (phage-resistant, antibiotic-sensitive)
          - `S_R` (phage-sensitive, antibiotic-resistant)
          - `R_R` (double-resistant)
        - Includes mutation rates between genotypes.
        """,
        "type": "single",
        "parameters": {
            "t_end": 72.0,
            "dt": 0.5,
            "extinction_threshold": 1.0,
            "solver_method": "BDF",
            "track_nutrients": True,
            "initial_S": 1.0,
            "monod_constant": 0.3,
            "recycle_fraction": 0.0,
            "s_in": 0.0,
            "s_out": 0.0,
            "immunity_enabled": False,
            "debris_enabled": False,
            "strains": [
                {
                    "name": "Sensitive (S_S)",
                    "initial_B": 1e7,
                    "growth_rate": 1.2,
                    "bacteria_to_resource_ratio": 1e9,
                    "death_rate_B": 0.0,
                    "dormancy_enabled": False,
                },
                {
                    "name": "Phage-Resist (R_S)",
                    "initial_B": 1e1,  # seeded at mutation rate
                    "growth_rate": 1.0,  # fitness cost
                    "bacteria_to_resource_ratio": 1e9,
                    "death_rate_B": 0.0,
                    "dormancy_enabled": False,
                },
                {
                    "name": "Abx-Resist (S_R)",
                    "initial_B": 1e1,  # seeded
                    "growth_rate": 1.1,  # minor fitness cost
                    "bacteria_to_resource_ratio": 1e9,
                    "death_rate_B": 0.0,
                    "dormancy_enabled": False,
                },
                {
                    "name": "Double-Resist (R_R)",
                    "initial_B": 0.0,
                    "growth_rate": 0.8,
                    "bacteria_to_resource_ratio": 1e9,
                    "death_rate_B": 0.0,
                    "dormancy_enabled": False,
                },
            ],
            "phages": [
                {
                    "name": "Phage 0",
                    "initial_P": 1e7,
                    # Phage adsorbs S_S and S_R at 1e-8; R_S and R_R at 1e-11
                    "adsorption_rates": [1e-8, 1e-11, 1e-8, 1e-11],
                    "adsorption_rates_dormant": 0.0,
                    "burst_sizes": 50.0,
                    "latent_periods": 0.5,
                    "phage_decay_rates": 0.1,
                    "pk_mode": "None",
                }
            ],
            "antibiotics": [
                {
                    "name": "Cipro",
                    "Vc": 250.0,
                    "k_elim": 0.3,
                    "emax": 3.0,
                    # Sensitive (S_S, R_S) ec50 = 0.2; Resistant (S_R, R_R) ec50 = 2.0
                    "ec50": 0.2,
                    "hill": 1.5,
                }
            ],
            "doses": [
                {
                    "time": 24.0,
                    "amount": 1000.0,
                    "target_type": "antibiotic",
                    "target_idx": 0,
                    "route": "bolus",
                    "duration": 0.0,
                }
            ],
        },
        "script_code": """# Tutorial 09: Binary Resistance Genotypes
import numpy as np
import matplotlib.pyplot as plt
from pbisim import PBIModel, solve_ode, DoseSchedule, DoseEvent
from pbisim.strains.genotypes import BinaryResistanceGenotypes, BacterialStrain, PhageStrain, Antibiotic

phages = [PhageStrain(name="Phage 0", burst_size_s=50.0, latent_period_s=0.5)]
bacteria = BacterialStrain(base_growth_rate=1.2)
antibiotics = [Antibiotic(
    name="Cipro",
    emax_s=3.0, ec50_s=0.2,
    emax_r=0.3, ec50_r=2.0,
    hill=1.5, k_elim=0.3, Vc=250.0,
)]

brg = BinaryResistanceGenotypes.from_strains(phages, bacteria=bacteria, antibiotics=antibiotics)
cfg = brg.to_config(
    n_latent=5, n_depth=1,
    monod_constant=0.3,
    dose_schedule=DoseSchedule([DoseEvent(time=24.0, amount=1000.0, target="antibiotic")]),
)

# Genotypes: 00=S_S (WT), 10=R_S (phage-R), 01=S_R (abx-R), 11=R_R (double-R)
initial_B = np.array([1e7, 10.0, 10.0, 0.0])

model = PBIModel(cfg, initial_B=initial_B, initial_P=np.array([1e7]), initial_S=1.0)
result = solve_ode(model, t_end=72.0, dt=0.5)

fig, ax = plt.subplots(figsize=(8, 4))
labels = ["S_S", "R_S", "S_R", "R_R"]
for i, l in enumerate(labels):
    ax.semilogy(result.time, np.maximum(result.get(f"B{i}"), 1.0), label=l)
ax.set(xlabel="Time (h)", ylabel="CFU/mL", title="Genotype Dynamics Under Dosing")
ax.legend()
plt.tight_layout()
""",
    },
    {
        "id": "tutorial_10",
        "name": "Tutorial 10: Cross-Resistance & CS",
        "description": """
        **Cross-Resistance (CR) and Collateral Sensitivity (CS)**
        
        This preset configures the unified CR/CS framework:
        - When bacteria evolve resistance to Phage A, it causes collateral sensitivity to Phage B (increases adsorption).
        - When bacteria evolve resistance to the phage, it also causes collateral sensitivity to the antibiotic (halves EC50).
        - Solves the coupled system showing evolutionary steering.
        """,
        "type": "single",
        "parameters": {
            "t_end": 72.0,
            "dt": 0.5,
            "extinction_threshold": 1.0,
            "solver_method": "BDF",
            "track_nutrients": True,
            "initial_S": 1.0,
            "monod_constant": 0.3,
            "recycle_fraction": 0.0,
            "s_in": 0.0,
            "s_out": 0.0,
            "immunity_enabled": False,
            "debris_enabled": False,
            "strains": [
                {
                    "name": "WT",
                    "initial_B": 1e7,
                    "growth_rate": 1.2,
                    "bacteria_to_resource_ratio": 1e9,
                    "death_rate_B": 0.0,
                    "dormancy_enabled": False,
                },
                {
                    "name": "Phage-Resist (Mutant)",
                    "initial_B": 10.0,
                    "growth_rate": 1.0,
                    "bacteria_to_resource_ratio": 1e9,
                    "death_rate_B": 0.0,
                    "dormancy_enabled": False,
                },
            ],
            "phages": [
                {
                    "name": "Phage A",
                    "initial_P": 1e7,
                    "adsorption_rates": [1e-8, 1e-11],  # mutant resistant
                    "adsorption_rates_dormant": 0.0,
                    "burst_sizes": 50.0,
                    "latent_periods": 0.5,
                    "phage_decay_rates": 0.1,
                    "pk_mode": "None",
                },
                {
                    "name": "Phage B (CS)",
                    "initial_P": 0.0,
                    "adsorption_rates": [1e-9, 5e-9],  # mutant collaterally sensitive (higher adsorption!)
                    "adsorption_rates_dormant": 0.0,
                    "burst_sizes": 50.0,
                    "latent_periods": 0.5,
                    "phage_decay_rates": 0.1,
                    "pk_mode": "None",
                },
            ],
            "antibiotics": [],
            "doses": [
                {
                    "time": 24.0,
                    "amount": 1e8,
                    "target_type": "phage",
                    "target_idx": 1,  # dose Phage B later
                    "route": "bolus",
                    "duration": 0.0,
                }
            ],
        },
        "script_code": """# Tutorial 10: Collateral Sensitivity & Evolutionary Steering
import numpy as np
import matplotlib.pyplot as plt
from pbisim import ModelBuilder, PBIModel, solve_ode, DoseSchedule, DoseEvent
from pbisim.analysis.single import plot_simulation

cfg = (
    ModelBuilder(n_bacteria=2, n_phages=2)
    .with_growth_rates([1.2, 1.0], bacteria_to_resource_ratio=1e9)
    # Phage A: WT sensitive (1e-8), Mutant resistant (1e-11)
    # Phage B: WT sensitive (1e-9), Mutant collaterally sensitive (5e-9)
    .with_phage_params(
        adsorption_rates=[[1e-8, 1e-9],
                          [1e-11, 5e-9]],
        burst_sizes=50.0,
        latent_periods=0.5,
        phage_decay_rates=0.1
    )
    .with_dose_schedule(DoseSchedule([
        DoseEvent(time=24.0, amount=1e8, target="phage", index=1) # dose Phage B at t=24
    ]))
    .build()
)

model = PBIModel(cfg, initial_B=np.array([1e7, 10.0]), initial_P=np.array([1e7, 0.0]), initial_S=1.0)
result = solve_ode(model, t_end=72.0, dt=0.5)

fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(result.time, result.get("B0"), label="WT", color="C0")
ax.semilogy(result.time, result.get("B1"), label="Phage A-Resistant (CS to B)", color="C1")
ax.semilogy(result.time, result.get("P1"), label="Phage B (Dosed t=24)", color="C3", ls="--")
ax.set(xlabel="Time (h)", ylabel="Count/mL", title="Collateral Sensitivity Steering")
ax.legend()
plt.tight_layout()
""",
    },
    {
        "id": "tutorial_11",
        "name": "Tutorial 11: Virtual Populations & IIV",
        "description": """
        **Inter-Individual Variability (IIV) & Population Simulation**
        
        Runs a simulation across a population of virtual patients:
        - Models patient-to-patient variability using statistical distributions (LogNormal).
        - Generates 10 virtual patient configs using `VirtualPopulation`.
        - Runs simulations in parallel using `TrialRunner`.
        - Displays population statistics.
        """,
        "type": "script",
        "script_code": """# Tutorial 11: Inter-Individual Variability
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pbisim import (ModelBuilder, PBIModel, solve_ode,
                    IIVSpec, LogNormal, VirtualPopulation,
                    TrialRunner, default_metrics)

def make_model(config):
    # initial conditions for each virtual patient
    return PBIModel(config, initial_B=np.array([1e7]), initial_P=np.array([1e6]), initial_S=1.0)

# 1. Base model
base_builder = (
    ModelBuilder(n_bacteria=1, n_phages=1, n_latent=5)
    .with_growth_rates(1.2, bacteria_to_resource_ratio=1e9)
    .with_phage_params(adsorption_rates=1e-8, burst_sizes=50.0, latent_periods=0.5, phage_decay_rates=0.1)
    .with_nutrient(monod_constant=0.3)
)

# 2. Specify Inter-individual variability (IIV)
iiv_spec = IIVSpec({
    "growth_rates": (LogNormal(cv=0.20), "multiplicative"),  # 20% Coefficient of Variation
    "adsorption_rates": (LogNormal(cv=0.30), "multiplicative") # 30% CV
})

vp = VirtualPopulation(base_builder, iiv_spec, seed=42)
patient_configs = vp.generate_cohort(10) # 10 patients

# 3. Run trial
runner = TrialRunner(model_factory=make_model, n_jobs=-1)
trial_result = runner.run(patient_configs, t_end=36.0, dt=0.5)

# 4. View results
fig, ax = plt.subplots(figsize=(8, 4))
for p_id in range(10):
    res = trial_result.patient_results[p_id]
    ax.semilogy(res.time, res.sum_prefixes("B"), color="C0", alpha=0.5)
ax.set(xlabel="Time (h)", ylabel="CFU/mL", title="Bacterial Dynamics Across 10 Virtual Patients")
plt.tight_layout()
""",
    },
    {
        "id": "tutorial_12",
        "name": "Tutorial 12: In-Silico Clinical Trials",
        "description": """
        **Clinical Trial Design: Crossover & Parallel Studies**
        
        Simulates a multi-arm, matched-twins clinical trial:
        - Matched virtual cohorts (control vs treatment).
        - Implements `PretreatmentPhase` (runs patient-specific pre-growth to stationary phase).
        - Computes survival analysis / time-to-clearance metrics.
        - Generates outcomes dataframes for Kaplan-Meier analysis.
        """,
        "type": "script",
        "script_code": """# Tutorial 12: Clinical Trial Design
import numpy as np
import matplotlib.pyplot as plt
from pbisim import (ModelBuilder, PBIModel, solve_ode,
                    IIVSpec, LogNormal, VirtualPopulation,
                    TreatmentArm, ClinicalTrial, PretreatmentPhase,
                    DoseSchedule, DoseEvent)

def make_model(config):
    return PBIModel(config, initial_B=np.array([1e7]), initial_P=np.array([0.0]), initial_S=1.0)

# Base config
base_builder = (
    ModelBuilder(n_bacteria=1, n_phages=1)
    .with_growth_rates(1.2, bacteria_to_resource_ratio=1e9)
    .with_phage_params(adsorption_rates=1e-8, burst_sizes=50.0, latent_periods=0.5, phage_decay_rates=0.1)
)

# IIV
iiv_spec = IIVSpec({
    "growth_rates": (LogNormal(cv=0.15), "multiplicative"),
    "adsorption_rates": (LogNormal(cv=0.25), "multiplicative")
})
vp = VirtualPopulation(base_builder, iiv_spec, seed=42)

# Treatment arms
arm_control = TreatmentArm("Control", DoseSchedule([]))
arm_phage = TreatmentArm("Phage Dosed", DoseSchedule([DoseEvent(time=0.0, amount=1e8, target="phage")]))

# Pretreatment pre-growth phase to stationary phase (12 hours)
pre_phase = PretreatmentPhase(t_prerun=12.0)

# Create trial
trial = ClinicalTrial(
    virtual_pop=vp,
    arms=[arm_control, arm_phage],
    model_factory=make_model,
    pretreatment=pre_phase,
    n_jobs=-1
)
result = trial.run(n_patients=5, t_end=24.0, dt=0.5)

# Plot comparison
fig, ax = plt.subplots(figsize=(8, 4))
for p_id in range(5):
    res_ctrl = result["Control"].patient_results[p_id]
    res_phg = result["Phage Dosed"].patient_results[p_id]
    ax.semilogy(res_ctrl.time, res_ctrl.sum_prefixes("B"), color="grey", alpha=0.5, label="Control" if p_id==0 else "")
    ax.semilogy(res_phg.time, res_phg.sum_prefixes("B"), color="C0", alpha=0.8, label="Phage Dosed" if p_id==0 else "")
ax.set(xlabel="Time (h)", ylabel="CFU/mL", title="Clinical Trial Crossover Match (n=5)")
ax.legend()
plt.tight_layout()
""",
    },
    {
        "id": "tutorial_13",
        "name": "Tutorial 13: Advanced Features",
        "description": """
        **Advanced Physical & Biological Options**
        
        Demonstrates advanced options for high-fidelity modeling:
        - **Bacterial Debris & OD**: Tracks cell fragments that continue scattering light.
        - **Nutrient Inflow/Washout**: Simulates continuous-flow systems (e.g. chemostat or open-wound models).
        - **Extinction Thresholds**: Enforces physical boundaries to avoid sub-CFU population rebounds.
        """,
        "type": "single",
        "parameters": {
            "t_end": 48.0,
            "dt": 0.1,
            "extinction_threshold": 1.0,
            "solver_method": "BDF",
            "track_nutrients": True,
            "initial_S": 1.0,
            "monod_constant": 0.3,
            "recycle_fraction": 0.5,
            "s_in": 0.2,      # continuous nutrient inflow
            "s_out": 0.1,     # continuous washout (dilution)
            "immunity_enabled": False,
            "debris_enabled": True,
            "debris_u": 1.0,
            "debris_v": 0.5,  # cell fragments scatter less than intact cells
            "debris_kdis": 0.1,
            "od_to_cfu_conversion_factor": 1e8,
            "strains": [
                {
                    "name": "Chemostat Strain",
                    "initial_B": 1e7,
                    "growth_rate": 1.2,
                    "bacteria_to_resource_ratio": 1e9,
                    "death_rate_B": 0.05,
                    "dormancy_enabled": False,
                }
            ],
            "phages": [
                {
                    "name": "Washout Phage",
                    "initial_P": 0.0,
                    "adsorption_rates": 1e-8,
                    "adsorption_rates_dormant": 0.0,
                    "burst_sizes": 50.0,
                    "latent_periods": 0.5,
                    "phage_decay_rates": 0.1,
                    "pk_mode": "None",
                }
            ],
            "antibiotics": [],
            "doses": [
                {
                    "time": 6.0,
                    "amount": 1e8,
                    "target_type": "phage",
                    "target_idx": 0,
                    "route": "bolus",
                    "duration": 0.0,
                }
            ],
        },
        "script_code": """# Tutorial 13: Debris ODE and Nutrient Inflow/Washout
import numpy as np
import matplotlib.pyplot as plt
from pbisim import ModelBuilder, PBIModel, solve_ode, DoseSchedule, DoseEvent
from pbisim.analysis.single import plot_simulation

cfg = (
    ModelBuilder(n_bacteria=1, n_phages=1)
    .with_growth_rates(1.2, bacteria_to_resource_ratio=1e9)
    .with_phage_params(adsorption_rates=1e-8, burst_sizes=50.0, latent_periods=0.5, phage_decay_rates=0.1)
    # Enable Debris and OD conversion
    .with_od_debris(u=1.0, v=0.5, kdis=0.1, od_to_cfu_conversion_factor=1e8)
    # Enable continuous inflow and washout
    .with_nutrient(s_in=0.2, s_out=0.1, monod_constant=0.3, recycle_fraction=0.5)
    .with_dose_schedule(DoseSchedule([
        DoseEvent(time=6.0, amount=1e8, target="phage")
    ]))
    .build()
)

model = PBIModel(cfg, initial_B=np.array([1e7]), initial_P=np.array([0.0]), initial_S=1.0)
result = solve_ode(model, t_end=48.0, dt=0.1, extinction_threshold=1.0)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(result.time, result.get_od(), label="Optical Density (OD)", color="black", lw=2)
ax.set(xlabel="Time (h)", ylabel="OD (AU)", title="Simulated OD tracking bacterial lysis & debris")
ax.legend()
plt.tight_layout()
""",
    },
]
