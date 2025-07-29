from sqlalchemy import (
    Column, Integer, Float, String
)
from sqlalchemy.orm import relationship
from models.base import Base

class AnalyzedExperiment(Base):
    __tablename__ = "analyzed_experiment"
    id = Column(Integer, primary_key=True, autoincrement=True)
    analyzed_experiment_id = Column(String, unique=True, index=True, nullable=False)

    set_loop_phase_deg = Column(Float)
    set_loop_att = Column(Float)
    set_loopback_att = Column(Float)
    set_yig_fb_phase_deg = Column(Float)
    set_yig_fb_att = Column(Float)
    set_cavity_fb_phase_deg = Column(Float)
    set_cavity_fb_att = Column(Float)

    # Analyzed
    independent_variable = Column(String, nullable=False)
    EP_location = Column(Float)
    TPD_location = Column(Float)
    Instability_location = Column(Float)

    J_avg = Column(Float)
    J_std = Column(Float)
    kappa_c_avg = Column(Float)
    kappa_c_std = Column(Float)
    f_c_avg = Column(Float)
    f_c_std = Column(Float)
    phi_avg = Column(Float)
    phi_std = Column(Float)
    Delta_kappa_min = Column(Float)
    Delta_kappa_max = Column(Float)
    Delta_f_min = Column(Float)
    Delta_f_max = Column(Float)

    # Sometimes applicable, nullable
    Delta_kappa_avg = Column(Float, nullable=True)
    Delta_kappa_std = Column(Float, nullable=True)
    Delta_f_avg = Column(Float, nullable=True)
    Delta_f_std = Column(Float, nullable=True)

    # Relationships
    analyzed_traces = relationship("AnalyzedTrace", back_populates="analyzed_experiment", cascade="all, delete-orphan")
    analyzed_peaks = relationship("AnalyzedPeaks", back_populates="analyzed_experiment", cascade="all, delete-orphan")
    theory_traces = relationship("TheoryTrace", back_populates="analyzed_experiment", cascade="all, delete-orphan")