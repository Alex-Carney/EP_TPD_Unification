import json
from dataclasses import dataclass

@dataclass
class ETLConfig:
    experiment_id: str
    # Experiment parameters, fit parameters
    phi_value: float
    J_value: float
    a0_guess: float

    readout_type: str
    colorplot_freq_min: float
    colorplot_freq_max: float
    cavity_freq_min: float
    cavity_freq_max: float
    yig_freq_min: float
    yig_freq_max: float
    amperage_min: float
    amperage_max: float

def load_all_configs(config_path: str) -> dict[str, ETLConfig]:
    with open(config_path, "r") as file:
        all_configs = json.load(file)
    return {name: load_config(config_path, name) for name in all_configs}

def load_config(config_path: str, config_name: str = "default") -> ETLConfig:
    """
    Load the configuration from a JSON file and return an AnalysisConfig
    dataclass for the given configuration name.
    """
    with open(config_path, "r") as file:
        all_configs = json.load(file)
    if config_name not in all_configs:
        raise KeyError(f"Configuration '{config_name}' not found in {config_path}.")
    cfg = all_configs[config_name]
    return ETLConfig(
        experiment_id=config_name,
        colorplot_freq_min=cfg["frequency_limits"]["colorplot"]["min"],
        colorplot_freq_max=cfg["frequency_limits"]["colorplot"]["max"],
        cavity_freq_min=cfg["frequency_limits"]["cavity"]["min"],
        cavity_freq_max=cfg["frequency_limits"]["cavity"]["max"],
        yig_freq_min=cfg["frequency_limits"]["yig"]["min"],
        yig_freq_max=cfg["frequency_limits"]["yig"]["max"],
        amperage_min=cfg["amperage_limits"]["min"],
        amperage_max=cfg["amperage_limits"]["max"],
        phi_value=cfg["phi_value"],
        readout_type=cfg["readout_type"],
        J_value=cfg["J_value"],
        a0_guess=cfg["a0_guess"]
    )
