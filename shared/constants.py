"""
LunarSight Constants
=====================
Physical, astronomical, and geotechnical constants used across the pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# ============================================================
# Lunar Physical Constants
# ============================================================

@dataclass(frozen=True)
class LunarConstants:
    """Immutable container for Moon physical parameters."""

    # Geometry
    radius_km: float = 1737.4            # Mean radius (km)
    radius_m: float = 1_737_400.0        # Mean radius (m)
    equatorial_radius_km: float = 1738.1

    # Gravity
    surface_gravity_ms2: float = 1.62    # Surface gravitational acceleration (m/s²)
    gravitational_param_km3s2: float = 4902.8  # GM (km³/s²)

    # Orbital / Illumination
    synodic_period_hours: float = 708.734  # Synodic period (~29.53 days in hours)
    synodic_period_seconds: float = 708.734 * 3600.0
    obliquity_deg: float = 1.5427         # Axial tilt to ecliptic (degrees)

    # Thermal
    solar_constant_wm2: float = 1361.0   # Solar irradiance at 1 AU (W/m²)
    surface_temp_day_k: float = 400.0    # Approximate dayside temp (K)
    surface_temp_night_k: float = 100.0  # Approximate nightside temp (K)
    psr_temp_k: float = 40.0            # Permanently shadowed region temp (K)

    @property
    def radius_m_float(self) -> float:
        return self.radius_km * 1000.0

    @property
    def synodic_period_days(self) -> float:
        return self.synodic_period_hours / 24.0


LUNAR = LunarConstants()


# ============================================================
# Regolith Geotechnical Properties (Apollo Data)
# ============================================================

@dataclass(frozen=True)
class RegolithProperties:
    """Lunar regolith soil mechanics parameters from Apollo data.

    These are used in Mohr-Coulomb and Janosi-Hanamoto models for
    terramechanics-based traversability analysis.
    """

    cohesion_pa: float = 170.0                    # c — Soil cohesion (Pa)
    friction_angle_deg: float = 33.0              # φ — Internal friction angle (°)
    friction_angle_rad: float = field(init=False)  # φ in radians (computed)
    shear_deformation_modulus_m: float = 0.018    # K — Shear deformation modulus (m)
    bulk_density_kgm3: float = 1500.0             # ρ — Bulk density (kg/m³)

    # Additional properties
    bearing_capacity_kpa: float = 10.0            # Approximate bearing capacity
    grain_size_mm: float = 0.07                   # Median grain size
    porosity: float = 0.44                        # Average porosity

    def __post_init__(self):
        # frozen=True requires object.__setattr__ for computed fields
        object.__setattr__(
            self,
            'friction_angle_rad',
            math.radians(self.friction_angle_deg),
        )


REGOLITH_DEFAULTS = RegolithProperties()


# ============================================================
# Polarimetric Thresholds (Sinha et al.)
# ============================================================

@dataclass(frozen=True)
class PolarimetricThresholds:
    """Default ice detection thresholds from Sinha et al.

    These can be overridden by mission_config.yaml values.
    """

    ice_cpr_min: float = 1.0       # CPR > 1 → volume scattering (ice signature)
    ice_dop_max: float = 0.13      # DOP < 0.13 → filters rocky dihedral returns
    rock_m_min: float = 0.7        # High m → surface/double-bounce (rock)
    ice_sband_cpr_min: float = 1.0
    ice_lband_cpr_min: float = 1.0


POLARIMETRIC_DEFAULTS = PolarimetricThresholds()


# ============================================================
# SAR / Radar Constants
# ============================================================

@dataclass(frozen=True)
class SARConstants:
    """Chandrayaan-2 DFSAR instrument parameters."""

    l_band_wavelength_m: float = 0.24       # L-band wavelength (24 cm)
    s_band_wavelength_m: float = 0.096      # S-band wavelength (9.6 cm)
    l_band_frequency_ghz: float = 1.25      # L-band center frequency
    s_band_frequency_ghz: float = 3.125     # S-band center frequency
    swath_width_km: float = 6.0             # Nominal swath width
    resolution_m: float = 2.0              # Spatial resolution (full-pol mode)


SAR = SARConstants()


# ============================================================
# Numerical Constants
# ============================================================

EPS = 1e-10           # Small epsilon to avoid division by zero
NAN_FILL = -9999.0    # Fill value for invalid/missing data
