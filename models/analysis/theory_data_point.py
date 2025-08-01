from sqlalchemy import (
    Column, Integer, Float, ForeignKey
)
from sqlalchemy.orm import relationship
from models.analysis_base import AnalysisBase


class TheoryDataPoint(AnalysisBase):
    __tablename__ = "theory_data_point"

    id = Column(Integer, primary_key=True, autoincrement=True)

    analyzed_experiment_pk = Column(Integer, ForeignKey("analyzed_experiment.id"), index=True, nullable=False)
    analyzed_experiment = relationship("AnalyzedExperiment", back_populates="theory_data_points")

    Delta_f = Column(Float)
    Delta_kappa = Column(Float)
    nu_plus = Column(Float)
    nu_minus = Column(Float)
