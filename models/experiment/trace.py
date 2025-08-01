from sqlalchemy import (
    Column, Integer, Float, String, ForeignKey
)
from sqlalchemy.orm import relationship
from models.base import Base
class Trace(Base):
    __tablename__ = "trace"
    id = Column(Integer, primary_key=True, autoincrement=True)

    experiment_pk = Column(Integer, ForeignKey("experiment.id"), index=True, nullable=False)
    experiment = relationship("Experiment", back_populates="traces")

    set_voltage = Column(Float)
    set_amperage = Column(Float)
    # This is the independent variable for phi/2 TPDs, thus it is included in the trace
    delta_kappa = Column(Float)
    # Also directly calculated in the experimental sweep for phi/2 TPDs, included here
    delta_f = Column(Float)
    readout_type = Column(String)

    raw_data = relationship("RawData", back_populates="trace", cascade="all, delete-orphan")