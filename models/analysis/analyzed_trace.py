from sqlalchemy import (
    Column, Integer, Float, String, ForeignKey
)
from sqlalchemy.orm import relationship
from models.base import Base
class AnalyzedTrace(Base):
    __tablename__ = "analyzed_trace"
    id = Column(Integer, primary_key=True, autoincrement=True)

    analyzed_experiment_pk = Column(Integer, ForeignKey("experiment.id"), index=True, nullable=False)
    analyzed_experiment = relationship("AnalyzedExperiment", back_populates="analyzed_traces")

    set_voltage = Column(Float)
    set_amperage = Column(Float)
    readout_type = Column(String)

    # Analyzed parameters
    Delta_f_Hz = Column(Float)
    Delta_f_Hz_err = Column(Float)
    Delta_kappa_Hz = Column(Float)
    Delta_kappa_Hz_err = Column(Float)
    kappa_c_Hz = Column(Float)
    kappa_c_Hz_err = Column(Float)
    f_c_Hz = Column(Float)
    f_c_Hz_err = Column(Float)

