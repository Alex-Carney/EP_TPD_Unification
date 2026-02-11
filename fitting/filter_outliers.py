from etl.settings import ETLSettings
from models.analysis import AnalyzedAggregateTrace
import numpy as np

def filter_outliers(traces: list[AnalyzedAggregateTrace], settings: ETLSettings) -> list[AnalyzedAggregateTrace]:
    """
    Filters traces based on Modified Z-Score of Delta_f and Delta_kappa.
    Returns a new list of traces with outliers removed.
    """
    if len(traces) < 3:
        return traces  # Not enough data to determine outliers statistically

    # Extract the relevant arrays
    df = np.array([t.Delta_f_Hz for t in traces])
    dk = np.array([t.Delta_kappa_Hz for t in traces])

    def get_modified_z_score(data):
        median = np.median(data)
        # Calculate Median Absolute Deviation (MAD)
        deviation = np.abs(data - median)
        mad = np.median(deviation)

        if mad == 0:
            return np.zeros_like(data)  # Avoid division by zero if all values are identical

        # 0.6745 is the consistency constant for normal distributions
        modified_z_score = 0.6745 * deviation / mad
        return modified_z_score

    # Calculate scores for both dimensions
    z_scores_df = get_modified_z_score(df)
    z_scores_dk = get_modified_z_score(dk)

    # Filter: Keep trace if BOTH scores are within threshold
    clean_traces = []
    for i, trace in enumerate(traces):
        # We check if the point is an outlier in EITHER dimension
        if z_scores_df[i] <= settings.OUTLIER_THRESHOLD and z_scores_dk[i] <= settings.OUTLIER_THRESHOLD:
            clean_traces.append(trace)
        else:
            print(f"[OUTLIER REMOVED] Exp: {trace.experiment_id} | Voltage: {trace.set_voltage} | "
                  f"Df Z-Score: {z_scores_df[i]:.2f} | Dk Z-Score: {z_scores_dk[i]:.2f}")

    return clean_traces