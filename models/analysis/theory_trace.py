from sqlalchemy import (
    Column, Integer, Float, String, ForeignKey
)
from sqlalchemy.orm import relationship
from models.base import Base
class TheoryTrace(Base):
    __tablename__ = "theory_trace"

    id = Column(Integer, primary_key=True, autoincrement=True)

    analyzed_experiment_pk = Column(Integer, ForeignKey("experiment.id"), index=True, nullable=False)
    analyzed_experiment = relationship("AnalyzedExperiment", back_populates="theory_traces")

    Delta_f = Column(Float)
    Delta_kappa = Column(Float)
    nu_plus = Column(Float)
    nu_minus = Column(Float)
