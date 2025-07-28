from sqlalchemy import (
    Column, Integer, Float, String, ForeignKey
)
from sqlalchemy.orm import relationship
from models.base import Base
class RawData(Base):
    __tablename__ = "raw_data"
    id = Column(Integer, primary_key=True, autoincrement=True)

    experiment_pk = Column(Integer, ForeignKey("experiment.id"), index=True, nullable=False)
    experiment = relationship("Experiment", back_populates="raw_data")

    frequency_hz = Column(Float)
    power_dBm = Column(Float)
    I_trace = Column(Float)
    Q_trace = Column(Float)
    readout_type = Column(String)

    omega_C = Column(Float)
    omega_Y = Column(Float)
    kappa_C = Column(Float)
    kappa_Y = Column(Float)
    Delta = Column(Float)
    K = Column(Float)

    source_row_id = Column(Integer)   # optional
    source_db = Column(String)        # optional