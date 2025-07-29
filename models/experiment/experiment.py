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

    independent_variable = Column(String, nullable=False)

    traces = relationship("Trace", back_populates="experiment", cascade="all, delete-orphan")