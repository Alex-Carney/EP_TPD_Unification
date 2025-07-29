from sqlalchemy import (
    Column, Integer, Float, String, ForeignKey
)
from sqlalchemy.orm import relationship
from models.base import Base
class RawData(Base):
    __tablename__ = "raw_data"
    id = Column(Integer, primary_key=True, autoincrement=True)

    trace_pk = Column(Integer, ForeignKey("trace.id"), index=True, nullable=False)
    trace = relationship("Trace", back_populates="raw_data")

    # Direct join back to experiment
    experiment_pk = Column(Integer, ForeignKey("experiment.id"), index=True)

    frequency_hz = Column(Float)
    power_dBm = Column(Float)
