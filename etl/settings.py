from dataclasses import dataclass


@dataclass(frozen=True)
class ETLSettings:
    NUM_SIMULATION_SHOTS = 1e4
    SEED = 12345
    CONFIG_PATH = "./etl_config.json"
    DB_PATH = "../data/ep_tpd_experiment_data.db"

    # If TRUE, will create plots for each trace. This is much slower, and only needs to be run once. (Default: True)
    CAV_YIG_DEBUG = True
    # If TRUE, will generate all traces for coupled mode (Default: True)
    COUPLED_DEBUG = True
    # if TRUE, will treat phi as a free parameter in the fit (Default: False, Paper: False)
    FREE_PHI = False
    # if TRUE, will treat J as a free parameter in the fit (Default: False, Paper: False)
    FREE_J = False
    # if TRUE, will normalize data to avoid ill-conditioned fits (Default: True, Paper: True)
    NORMALIZE_DATA = True
