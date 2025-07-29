from sqlalchemy import (
    Column, Integer, Float, String, ForeignKey
)
from sqlalchemy.orm import relationship
from models.base import Base
class AnalyzedPeaks(Base):
    __tablename__ = "analyzed_peaks"
    id = Column(Integer, primary_key=True, autoincrement=True)

    analyzed_experiment_pk = Column(Integer, ForeignKey("experiment.id"), index=True, nullable=False)
    analyzed_experiment = relationship("AnalyzedExperiment", back_populates="analyzed_peaks")

    Delta_kappa = Column(Float)
    Delta_f = Column(Float)
    peak_mean = Column(Float)
    peak_err_low = Column(Float)
    peak_err_high = Column(Float)