from itertools import groupby
from operator import attrgetter

import numpy as np
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker, selectinload

from errors.data_load_error import DataLoadError
from etl.etl_config import ETLConfig, load_all_configs
from etl.model_fitting import CoupledFitOutcome
from etl.settings import ETLSettings
from models.experiment import Trace
from models.experiment.experiment import Experiment
import model_fitting as fit

"""
ETL (Extract, Transform, Load) pipeline

Extracts raw data from ep_tpd_merged
Transforms data by fitting Lorentzians and applying statistical methods
Loads data into CSVs for consumption in Figures/
"""
def main():

    # Part 1 : EXTRACTION -  Load configs, data from DB
    settings: ETLSettings = ETLSettings()
    expr_id_config_map: dict[str, ETLConfig] = load_all_configs(settings.CONFIG_PATH)
    engine = create_engine("sqlite:///" + settings.DB_PATH, future=True)
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)

    # Cross-reference experiment IDs from session with expr_id_config_map
    with SessionLocal() as session:
        db_experiment_ids = {exp.experiment_id for exp in session.query(Experiment).all()}
        config_experiment_ids = set(expr_id_config_map.keys())
        missing_configs = db_experiment_ids - config_experiment_ids
        if missing_configs:
            raise DataLoadError(
                f"Found {len(missing_configs)} experiments without config entries: {missing_configs}"
            )
        missing_experiments = config_experiment_ids - db_experiment_ids
        if missing_experiments:
            raise DataLoadError(
                f"Found {len(missing_experiments)} config entries without experiments: {missing_experiments}",
            )

    with SessionLocal() as session:
        experiments_with_data = (
            session.query(Experiment)
            .options(
                selectinload(Experiment.traces)          # grab all traces
                .selectinload(Trace.raw_data)            # and their samples
            )
            .all()
        )
        for exp in experiments_with_data:
            expr_config: ETLConfig = expr_id_config_map[exp.experiment_id]
            ind_name = exp.independent_variable          # "set_voltage" or "set_amperage"
            keyfunc  = attrgetter(ind_name)              # fast attribute lookup

            # Group the traces by the independent variable value
            sorted_traces = sorted(exp.traces, key=keyfunc)
            for iv_value, traces_iter in groupby(sorted_traces, key=keyfunc):
                traces = list(traces_iter)               # the 4 traces for this iv_value
                print(f"{exp.experiment_id=}  {iv_value=}  traces={len(traces)}")

                # pick out the cavity trace & fit
                cavity_trace = next(t for t in traces if t.readout_type == "cavity")
                cavity_freqs  = np.asarray([rd.frequency_hz for rd in cavity_trace.raw_data])
                cavity_power = np.asarray([rd.power_dBm    for rd in cavity_trace.raw_data])
                cavity_fit: fit.FitOutcome = fit.fit_cavity_trace(cavity_freqs, cavity_power)

                # pick out the YIG trace & fit
                yig_trace = next(t for t in traces if t.readout_type == "yig")
                yig_freqs  = np.asarray([rd.frequency_hz for rd in yig_trace.raw_data])
                yig_power =np.asarray([rd.power_dBm    for rd in yig_trace.raw_data])
                yig_fit: fit.FitOutcome = fit.fit_yig_trace(yig_freqs, yig_power)

                # Get composite parameters and their errors
                delta_f = cavity_fit.f0 - yig_fit.f0
                delta_f_err = np.sqrt(cavity_fit.f0_err**2 + yig_fit.f0_err**2)
                delta_kappa = cavity_fit.kappa - yig_fit.kappa
                delta_kappa_err = np.sqrt(cavity_fit.kappa_err**2 + yig_fit.kappa_err**2)

                # Fit coupled trace based on coupled model
                data_trace = next(t for t in traces if t.readout_type == expr_config.readout_type)
                data_freqs  = [rd.frequency_hz for rd in data_trace.raw_data]
                data_power = [rd.power_dBm    for rd in data_trace.raw_data]
                data_power_normalized: np.ndarray[float] = np.asarray(data_power) - np.min(data_power) if settings.NORMALIZE_DATA else np.asarray(data_power)
                coupled_fit: CoupledFitOutcome = fit.fit_coupled_trace_fixed_J(
                    data_freqs, data_power_normalized,
                    J = expr_config.J_value, f_c=cavity_fit.f0, kappa_c=cavity_fit.kappa,
                    delta_f=delta_f, delta_kappa=delta_kappa,
                    phi=expr_config.phi_value, n_starts=100,
                    current=iv_value, maxfev=100000, a0_guess=4.4625e+15,
                    fit_phi=settings.FREE_PHI, vertical_offset=False
                ) if not settings.FREE_J else fit.fit_coupled_trace(
                    data_freqs, data_power_normalized,
                    f_c=cavity_fit.f0, kappa_c=cavity_fit.kappa,
                    delta_f=delta_f, delta_kappa=delta_kappa,
                    phi=expr_config.phi_value, n_starts=100,
                    current=iv_value, maxfev=100000, a0_guess=4.4625e+15,
                    fit_phi=settings.FREE_PHI, vertical_offset=False,
                    J_bounds=(expr_config.J_value * 0.5, expr_config.J_value * 1.5), initial_J=expr_config.J_value
                )

                # Based on coupled fit, where are the PEAKS?

                # Propagation of uncertainty, must use Monte Carlo

    return locals()


if __name__ == "__main__":
    globals().update(main())
