from sqlalchemy import (
    Column, Integer, Float, String
)
from sqlalchemy.orm import relationship
from models.base import Base

class Experiment(Base):
    __tablename__ = "experiment"
    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String, unique=True, index=True, nullable=False)

    set_loop_phase_deg = Column(Float)
    set_loop_att = Column(Float)
    set_loopback_att = Column(Float)
    set_yig_fb_phase_deg = Column(Float)
    set_yig_fb_att = Column(Float)
    set_cavity_fb_phase_deg = Column(Float)
    set_cavity_fb_att = Column(Float)

    set_voltage = Column(Float)
    set_amperage = Column(Float)

    raw_data = relationship("RawData", back_populates="experiment", cascade="all, delete-orphan")