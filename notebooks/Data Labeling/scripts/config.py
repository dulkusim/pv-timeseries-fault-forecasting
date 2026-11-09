
from dataclasses import dataclass


@dataclass(frozen=True)
class SiteConfig:
    """Configuration for one NIST photovoltaic installation."""

    name: str
    p_rated_dc_kw: float
    p_ac_max_kw: float
    gamma: float
    t_stc: float
    n_strings: int
    core_strings: tuple[int, ...]

    op_mask_columns: tuple[str, ...]
    op_mask_agg: str

    poa_columns: tuple[str, ...]
    poa_agg: str

    rtd_drop: tuple[int, ...]
    rtd_module_temp: tuple[int, ...]

    raw_file_pattern: str


CANOPY = SiteConfig(
    name="canopy",
    p_rated_dc_kw=243.0,
    p_ac_max_kw=260.0,
    gamma=-0.00485,
    t_stc=25.0,
    n_strings=7,
    core_strings=(2, 3, 4, 5, 6),
    op_mask_columns=(
        "Pyra1_Wm2_Avg",
        "Pyra3_Wm2_Avg",
    ),
    op_mask_agg="max",
    poa_columns=(
        "RefCell1_Wm2_Avg",
        "RefCell2_Wm2_Avg",
    ),
    poa_agg="mean",
    rtd_drop=(3, 5, 6, 7),
    rtd_module_temp=(1, 2, 4, 8, 9),
    raw_file_pattern="onemin-Canopy-{year}.csv",
)


GROUND = SiteConfig(
    name="ground",
    p_rated_dc_kw=271.0,
    p_ac_max_kw=260.0,
    gamma=-0.00485,
    t_stc=25.0,
    n_strings=7,
    core_strings=(1, 2, 3, 4, 5, 6, 7),
    op_mask_columns=("Pyra2_Wm2_Avg",),
    op_mask_agg="single",
    poa_columns=("RefCell1_Wm2_Avg",),
    poa_agg="single",
    rtd_drop=(2, 3, 8),
    rtd_module_temp=(1, 4, 5, 6, 7, 9),
    raw_file_pattern="onemin-Ground-{year}.csv",
)


ROOF = None


# --------------------------------------------------
# Shared labeling thresholds
# --------------------------------------------------

POA_MIN_OPERATING = 100.0
POA_MIN_PR = 200.0

PR_CRITICAL = 0.50
PR_SEVERE = 0.70
PR_UNDERPERFORMANCE = 0.85
PR_SUSPICIOUS_HIGH = 1.05

STRING_REL_DEV_THRESHOLD = 0.20
OPEN_CIRCUIT_RATIO = 0.10

INV_EFFICIENCY_MIN = 0.85

# Inverter shutdown
INV_SHUTDOWN_POWER = 1.0
INV_SHUTDOWN_POWER_COLUMN = "InvPDC_kW_Avg"

INV_NORMAL_STATUS_MIN = 1
INV_NORMAL_STATUS_MAX = 5

# Efficiency evaluation
INV_EFFICIENCY_MIN = 0.85
INV_EFFICIENCY_MIN_PDC_KW = 5.0
INV_MIN_LOADING_RATIO = 0.20
INV_EFFICIENCY_MAX = 1.0

# Inverter self-reported fault codes:
# A nonzero value indicates a reported fault.
INV_FAULT_FLAG_NONZERO = True

# String imbalance
STRING_REL_DEV_THRESHOLD = 0.20
OPEN_CIRCUIT_RATIO = 0.10