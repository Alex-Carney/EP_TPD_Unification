from sqlalchemy import (
    Column, Integer, Float, ForeignKey
)
from sqlalchemy.orm import relationship
from models.analysis_base import AnalysisBase


class AnalyzedAggregateTrace(AnalysisBase):
    __tablename__ = "analyzed_aggregate_trace"
    id = Column(Integer, primary_key=True, autoincrement=True)

    analyzed_experiment_pk = Column(Integer, ForeignKey("analyzed_experiment.id"), index=True, nullable=False)
    analyzed_experiment = relationship("AnalyzedExperiment", back_populates="analyzed_aggregate_traces")

    set_voltage = Column(Float)
    set_amperage = Column(Float)

    # Analyzed parameters
    Delta_f_Hz = Column(Float)
    Delta_f_Hz_err = Column(Float)
    Delta_kappa_Hz = Column(Float)
    Delta_kappa_Hz_err = Column(Float)
    kappa_c_Hz = Column(Float)
    kappa_c_Hz_err = Column(Float)
    f_c_Hz = Column(Float)
    f_c_Hz_err = Column(Float)
    J_Hz = Column(Float)
    J_Hz_err = Column(Float)
    a0_value = Column(Float)
    a0_err = Column(Float)
    phi_rad = Column(Float)
    phi_rad_err = Column(Float)

    # Peak fitted parameters
    nu_minus_mean_data_Hz = Column(Float)
    nu_minus_err_low_data_Hz = Column(Float)
    nu_minus_err_high_data_Hz = Column(Float)

    nu_plus_mean_data_Hz = Column(Float)
    nu_plus_err_low_data_Hz = Column(Float)
    nu_plus_err_high_data_Hz = Column(Float)

