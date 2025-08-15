import os
from itertools import groupby
from operator import attrgetter
import numpy as np
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import sessionmaker, selectinload
import fitting.model_fitting as fit
import fitting.peak_fitting as peaks
import fitting.plot_fitting as plot_fit
import simulate_theory as theory
from errors.data_integrity_error import DataIntegrityError
from errors.data_load_error import DataLoadError
from etl.etl_config import ETLConfig, load_all_configs
from etl.settings import ETLSettings
from fitting.model_fitting import CoupledFitOutcome
from fitting.transition_fitting import EP_location, TPD_location, instability_location
from models.analysis import TheoryDataPoint
from models.analysis.analyzed_experiment import AnalyzedExperiment
from models.analysis.analyzed_trace import AnalyzedAggregateTrace
from models.experiment import Trace
from models.experiment.experiment import Experiment

"""
ETL (Extract, Transform, Load) pipeline

Extracts raw data from ep_tpd_merged
Transforms data by fitting Lorentzians and applying statistical methods
Loads data into CSVs for consumption in Figures/
"""
def main():
    # Part 1: EXTRACTION - Load configs, data from DB
    settings: ETLSettings = ETLSettings()
    np.random.seed(settings.SEED)
    expr_id_config_map: dict[str, ETLConfig] = load_all_configs(settings.CONFIG_PATH)
    # Engines for old data and transformed data
    raw_data_engine = create_engine("sqlite:///" + settings.RAW_DATA_DB_PATH, future=True)
    RawDataSessionLocal = sessionmaker(bind=raw_data_engine, expire_on_commit=False, future=True)
    transformed_data_engine = create_engine("sqlite:///" + settings.OUTPUT_TRANSFORM_DB_PATH, future=True)
    TransformedDataSessionLocal = sessionmaker(bind=transformed_data_engine, expire_on_commit=False, future=True)

    # [CHECK]: Cross-reference experiment IDs from session with expr_id_config_map
    with RawDataSessionLocal() as raw_data_session:
        db_experiment_ids = {exp.experiment_id for exp in raw_data_session.query(Experiment).all()}
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

    with RawDataSessionLocal() as raw_data_session:
        experiments_with_data = (
            raw_data_session.query(Experiment)
            .options(
                selectinload(Experiment.traces)          # grab all traces
                .selectinload(Trace.raw_data)            # and their samples
            )
            .all()
        )
        analyzed_exps: list[AnalyzedExperiment] = []
        for exp in experiments_with_data:
            expr_config: ETLConfig = expr_id_config_map[exp.experiment_id]
            exp_traces: list[AnalyzedAggregateTrace] = []
            ind_name = exp.independent_variable          # "set_voltage" or "set_amperage"
            keyfunc  = attrgetter(ind_name)
            # Group the traces by the independent variable value
            sorted_traces = sorted(exp.traces, key=keyfunc)
            for iv_value, traces_iter in groupby(sorted_traces, key=keyfunc):
                traces = list(traces_iter)               # the 4 traces for this iv_value
                print(f"{exp.experiment_id=}  {iv_value=}  traces={len(traces)}")
                if not len(traces) == 4:
                    raise DataIntegrityError(
                        f"Expected 4 traces for {exp.experiment_id} at {iv_value}, found {len(traces)}."
                    )

                trace_set_voltage = traces[0].set_voltage if traces else None
                trace_set_amperage = traces[0].set_amperage if traces else None

                ###
                # STEP 1 - FIT THE INDIVIDUAL MODES
                ###
                # pick out the cavity trace and fit
                cavity_trace = next(t for t in traces if t.readout_type == "cavity")
                cavity_freqs  = np.asarray([rd.frequency_hz for rd in cavity_trace.raw_data if expr_config.cavity_freq_min <= rd.frequency_hz <= expr_config.cavity_freq_max])
                cavity_power = np.asarray([rd.power_dBm    for rd in cavity_trace.raw_data if expr_config.cavity_freq_min <= rd.frequency_hz <= expr_config.cavity_freq_max])

                cavity_fit: fit.FitOutcome = fit.fit_cavity_trace(cavity_freqs, cavity_power)

                # pick out the YIG trace & fit
                yig_trace = next(t for t in traces if t.readout_type == "yig")
                yig_freqs  = np.asarray([rd.frequency_hz for rd in yig_trace.raw_data if expr_config.yig_freq_min <= rd.frequency_hz <= expr_config.yig_freq_max])
                yig_power =np.asarray([rd.power_dBm    for rd in yig_trace.raw_data if expr_config.yig_freq_min <= rd.frequency_hz <= expr_config.yig_freq_max])
                yig_fit: fit.FitOutcome = fit.fit_yig_trace(yig_freqs, yig_power)

                # Get composite parameters and their errors
                delta_f = cavity_fit.f0 - yig_fit.f0
                delta_f_err = np.sqrt(cavity_fit.f0_err**2 + yig_fit.f0_err**2)

                delta_kappa = (cavity_fit.kappa - yig_fit.kappa) / 2
                delta_kappa_err = np.sqrt(cavity_fit.kappa_err**2 + yig_fit.kappa_err**2) / 2

                ###
                # STEP 2 - FIT THE COUPLED TRANSMISSION
                ###
                # Fit coupled trace based on coupled model
                data_trace = next(t for t in traces if t.readout_type == expr_config.readout_type)
                data_freqs  = np.asarray([rd.frequency_hz for rd in data_trace.raw_data if expr_config.colorplot_freq_min <= rd.frequency_hz <= expr_config.colorplot_freq_max])
                data_power = np.asarray([rd.power_dBm    for rd in data_trace.raw_data if expr_config.colorplot_freq_min <= rd.frequency_hz <= expr_config.colorplot_freq_max])
                data_power_normalized: np.ndarray[float] = data_power - np.min(data_power) if settings.NORMALIZE_DATA else data_power
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

                ###
                # STEP 3 - GET THE PEAK LOCATIONS WITH MONTE CARLO UNCERTAINTY PROPAGATION
                ###
                # Based on coupled fit, where are the PEAKS?
                peak_locations = peaks.peak_location(
                    J=coupled_fit.J, f_c=cavity_fit.f0, kappa_c=cavity_fit.kappa,
                    delta_f=delta_f, delta_kappa=delta_kappa, phi=expr_config.phi_value)

                # Propagation of uncertainty for peak locations, must use Monte Carlo
                means = dict(J=coupled_fit.J, phi=coupled_fit.phi, kappa_c=cavity_fit.kappa,
                    f_c=cavity_fit.f0, delta_f=delta_f, delta_kappa=delta_kappa)
                sigmas = dict(
                    J=coupled_fit.J_err, phi=coupled_fit.phi_err if coupled_fit.phi != 0 else 0, kappa_c=cavity_fit.kappa_err,
                    f_c=cavity_fit.f0_err, delta_f=delta_f_err, delta_kappa=delta_kappa_err)
                peak_stats = peaks.peak_location_mc(means, sigmas, n_draws=settings.MONTE_CARLO_SHOTS)
                means = [peak_stats["mean"] for peak_stats in peak_stats.values()]
                maxes = [peak_stats["max"] for peak_stats in peak_stats.values()]
                mins = [peak_stats["min"] for peak_stats in peak_stats.values()]

                ###
                # STEP 4 - LOAD INTO MODELS
                ###
                exp_traces.append(AnalyzedAggregateTrace(
                    Delta_f_Hz=delta_f, Delta_f_Hz_err=delta_f_err, Delta_kappa_Hz=delta_kappa,
                    Delta_kappa_Hz_err=delta_kappa_err, kappa_c_Hz=cavity_fit.kappa, kappa_c_Hz_err=cavity_fit.kappa_err,
                    phi_rad=coupled_fit.phi, phi_rad_err=coupled_fit.phi_err,
                    f_c_Hz=cavity_fit.f0, f_c_Hz_err=cavity_fit.f0_err, J_Hz=coupled_fit.J,
                    J_Hz_err=coupled_fit.J_err, a0_value=coupled_fit.a, a0_err=coupled_fit.a_err,
                    nu_minus_mean_data_Hz=means[0], nu_minus_err_low_data_Hz=mins[0],
                    nu_minus_err_high_data_Hz=maxes[0], nu_plus_mean_data_Hz=means[1],
                    nu_plus_err_low_data_Hz=mins[1], nu_plus_err_high_data_Hz=maxes[1],
                    set_voltage=trace_set_voltage, set_amperage=trace_set_amperage,
                ))

                ###
                # (OPTIONAL) STEP 5 - PLOT THE FITS
                ###
                if settings.CAV_YIG_DEBUG:
                    debug_folder = os.path.join(settings.PLOT_OUTPUT_PATH, f"YIG_CAV_DEBUG_{exp.experiment_id}")
                    os.makedirs(debug_folder, exist_ok=True)
                    plot_fit.plot_trace_with_model(cavity_freqs, cavity_power, cavity_fit,
                                                   save_path=os.path.join(debug_folder, f"cav_{iv_value}.png"),)
                    plot_fit.plot_trace_with_model(yig_freqs, yig_power, yig_fit,
                                                    save_path=os.path.join(debug_folder, f"yig_{iv_value}.png"),)

                if settings.COUPLED_DEBUG:
                    debug_folder = os.path.join(settings.PLOT_OUTPUT_PATH, f"COUPLED_DEBUG_{exp.experiment_id}")
                    os.makedirs(debug_folder, exist_ok=True)
                    extra_info_dict = {
                        "f0_c": cavity_fit.f0, "f0_err_c": cavity_fit.f0_err, "kappa_c": cavity_fit.kappa,
                        "kappa_err_c": cavity_fit.kappa_err, "Delta_f": delta_f, "Delta_f_err": delta_f_err,
                        "Delta_kappa": delta_kappa, "Delta_kappa_err": delta_kappa_err,
                    }
                    plot_fit.plot_coupled_trace_with_model(data_freqs, data_power_normalized, coupled_fit, title=f"Coupled Fit for {exp.experiment_id} at {iv_value}",
                                                           save_path=os.path.join(debug_folder, f"coupled_{iv_value}.png"), voltage=iv_value, found_peaks=peak_locations,
                                                           found_peaks_maxima=maxes, found_peaks_minima=mins, extra_info=extra_info_dict)

            # Now that all the Traces are analyzed, find Experiment-wide results
            f_c_vals = [t.f_c_Hz for t in exp_traces]
            kappa_c_vals = [t.kappa_c_Hz for t in exp_traces]
            J_vals = [t.J_Hz for t in exp_traces]
            phi_vals = [t.phi_rad for t in exp_traces]
            Delta_f_vals = [t.Delta_f_Hz for t in exp_traces]
            Delta_kappa_vals = [t.Delta_kappa_Hz for t in exp_traces]

            # Get EP, TPD, and Instability locations
            J_val = np.mean(J_vals)
            phi_val = np.mean(phi_vals)
            kappa_c_val = np.mean(kappa_c_vals)

            # Get the Theory results
            theory_df, theory_dk, theory_results_nu_plus, theory_results_nu_minus = theory.simulate_theory(
                phi=expr_config.phi_value, df=Delta_f_vals, dk=Delta_kappa_vals,
                J_avg=J_val, fc_avg=np.mean(f_c_vals), kc_avg=kappa_c_val,
                settings=settings
            )

            cleaned_nu_plus = []
            for np_val, nm_val in zip(theory_results_nu_plus, theory_results_nu_minus):
                if np_val is None or (isinstance(np_val, float) and np.isnan(np_val)):
                    cleaned_nu_plus.append(nm_val)
                else:
                    cleaned_nu_plus.append(np_val)

            # Now zip with the cleaned nu_plus
            theory_data_points: list[TheoryDataPoint] = [
                TheoryDataPoint(
                    analyzed_experiment_pk=exp.id,
                    Delta_f=df_val,
                    Delta_kappa=dk_val,
                    nu_plus=np_val,
                    nu_minus=nm_val
                )
                for df_val, dk_val, np_val, nm_val in zip(
                    theory_df, theory_dk, cleaned_nu_plus, theory_results_nu_minus
                )
            ]

            analyzed_expr = AnalyzedExperiment(
                analyzed_experiment_id=exp.experiment_id,
                independent_variable=ind_name,
                J_avg=J_val, J_std=np.std(J_vals), kappa_c_avg=kappa_c_val,
                kappa_c_std=np.std(kappa_c_vals), f_c_avg=np.mean(f_c_vals), f_c_std=np.std(f_c_vals),
                phi_avg=phi_val, phi_std=np.std(phi_vals),
                Delta_kappa_avg=np.mean(Delta_kappa_vals), Delta_kappa_std=np.std(Delta_kappa_vals),
                Delta_kappa_min=np.min(Delta_kappa_vals), Delta_kappa_max=np.max(Delta_kappa_vals),
                Delta_f_min=np.min(Delta_f_vals), Delta_f_max=np.max(Delta_f_vals),
                Delta_f_avg=np.mean(Delta_f_vals), Delta_f_std=np.std(Delta_f_vals),
                EP_location=EP_location(phi_val, J_val),
                TPD_location=TPD_location(phi_val, kappa_c_val, J_val),
                Instability_location=instability_location(phi_val, kappa_c_val, J_val),
            )
            analyzed_expr.analyzed_aggregate_traces = exp_traces
            analyzed_expr.theory_data_points = theory_data_points

            analyzed_exps.append(analyzed_expr)

    # Final step - Create a NEW session to create the transformed database
    with TransformedDataSessionLocal() as transformed_data_session:
        # Create the tables if they don't exist
        AnalyzedExperiment.metadata.create_all(transformed_data_engine)
        AnalyzedAggregateTrace.metadata.create_all(transformed_data_engine)
        TheoryDataPoint.metadata.create_all(transformed_data_engine)

        # Wipe everything first
        transformed_data_session.execute(delete(TheoryDataPoint))
        transformed_data_session.execute(delete(AnalyzedAggregateTrace))
        transformed_data_session.execute(delete(AnalyzedExperiment))
        transformed_data_session.commit()

        # Add all analyzed experiments to the session
        transformed_data_session.add_all(analyzed_exps)

        # Commit the session to save changes
        transformed_data_session.commit()

    return locals()


if __name__ == "__main__":
    globals().update(main())
