from dataclasses import dataclass

@dataclass(frozen=True)
class ETLSettings:
    # Don't change this unless you want different random numbers from Monte Carlo
    SEED = 12345
    CONFIG_PATH = "./etl_config.json"
    RAW_DATA_DB_PATH = "../data/ep_tpd_experiment_data.db"
    OUTPUT_TRANSFORM_DB_PATH = "../data/ep_tpd_transformed_data.db"
    PLOT_OUTPUT_PATH = "../.plots/"

    # If TRUE, will create plots for each trace. This is much slower, and only needs to be run once. (Default: True, Paper: False)
    CAV_YIG_DEBUG = False

    # If TRUE, will generate all traces for coupled mode (Default: True, Paper: False)
    COUPLED_DEBUG = False

    # if TRUE, will treat phi as a free parameter in the fit (Default: False, Paper: False)
    FREE_PHI = False

    # if TRUE, will treat J as a free parameter in the fit (Default: False, Paper: False)
    FREE_J = False

    # if TRUE, will normalize data to avoid ill-conditioned fits (Default: True, Paper: True)
    NORMALIZE_DATA = True

    # if TRUE, will inflate covariances by multiplying by normalized Chi_R^2 (how well the model fits the amplitude data) (Default: False, Paper: False)
    # We kept this False because most of the time Chi_R^2 normalized is < 1, which would shrink the covariance matrix.
    INFLATE_COV = False

    # Multiply all error bars by this factor (Default: 5.0, Paper: 5.0)
    ERROR_BAR_SCALE_FACTOR = 5.0

    # Theory size (WARNING: This gets stored in the DB, do not make this too large) (Default: 2500, Paper: 2500)
    THEORY_SIZE = 2500

    # Number of monte carlo shots (Default: 10**3, Paper: 10**4)
    MONTE_CARLO_SHOTS = 10**4

