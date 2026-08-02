"""
Polarimetric Thresholds
========================
Configurable diagnostic flags for ice and rock classification based on
polarimetric parameters (CPR, DOP, m-χ).

Default thresholds follow Sinha et al. criteria for Chandrayaan-2 DFSAR
south polar observations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import yaml

from shared.constants import EPS, POLARIMETRIC_DEFAULTS

logger = logging.getLogger(__name__)


@dataclass
class ThresholdConfig:
    """Configurable threshold parameters for ice/rock classification.

    These can be loaded from mission_config.yaml or set programmatically.
    """

    ice_cpr_min: float = 1.0         # CPR > threshold → ice candidate
    ice_dop_max: float = 0.13        # DOP < threshold → ice (low polarization)
    ice_lband_cpr_min: float = 1.0   # L-band specific CPR threshold
    ice_sband_cpr_min: float = 1.0   # S-band specific CPR threshold
    rock_m_min: float = 0.7          # High m → surface/double-bounce (rock)

    @classmethod
    def from_config(cls, config: dict) -> "ThresholdConfig":
        """Create from mission_config.yaml thresholds section."""
        thresholds = config.get("thresholds", {})
        return cls(
            ice_cpr_min=thresholds.get("ice_cpr_min", 1.0),
            ice_dop_max=thresholds.get("ice_dop_max", 0.13),
            ice_lband_cpr_min=thresholds.get("ice_lband_cpr_min", 1.0),
            ice_sband_cpr_min=thresholds.get("ice_sband_cpr_min", 1.0),
            rock_m_min=thresholds.get("rock_m_min", 0.7),
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ThresholdConfig":
        """Load from a YAML file."""
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls.from_config(config)


def classify_ice_candidates(
    l_cpr: np.ndarray,
    dop: np.ndarray,
    s_cpr: Optional[np.ndarray] = None,
    thresholds: Optional[ThresholdConfig] = None,
) -> np.ndarray:
    """Classify pixels as ice candidates using polarimetric criteria.

    Ice candidate criteria (Sinha et al.):
        - L-band CPR > threshold (volume scattering)
        - DOP < threshold (low degree of polarization)
        - S-band CPR > threshold (if available, confirms dual-wavelength consistency)

    Args:
        l_cpr: L-band CPR array (H×W).
        dop: Degree of polarization array (H×W).
        s_cpr: S-band CPR array (H×W), optional.
        thresholds: Threshold configuration. Uses defaults if None.

    Returns:
        Boolean array: True where pixel is classified as ice candidate.
    """
    if thresholds is None:
        thresholds = ThresholdConfig()

    # Primary criteria: CPR > threshold AND DOP < threshold
    ice_flag = (l_cpr > thresholds.ice_lband_cpr_min) & (dop < thresholds.ice_dop_max)

    # Secondary: S-band consistency (if available)
    if s_cpr is not None:
        s_band_consistent = s_cpr > thresholds.ice_sband_cpr_min
        ice_flag = ice_flag & s_band_consistent

    # Mask NaN regions
    nan_mask = ~np.isfinite(l_cpr) | ~np.isfinite(dop)
    ice_flag[nan_mask] = False

    count = np.sum(ice_flag)
    total = np.sum(~nan_mask)
    logger.info(
        f"Ice candidates: {count} / {total} pixels "
        f"({count / max(total, 1) * 100:.2f}%)"
    )

    return ice_flag


def classify_rock(
    m: np.ndarray,
    dominant_mechanism: np.ndarray,
    thresholds: Optional[ThresholdConfig] = None,
) -> np.ndarray:
    """Classify pixels as rock based on polarimetric parameters.

    Rock criteria:
        - High degree of polarization (m > threshold)
        - Dominant odd-bounce (surface) or double-bounce scattering

    Args:
        m: Degree of polarization array (H×W).
        dominant_mechanism: Scattering mechanism (0=surface, 1=volume, 2=dihedral).
        thresholds: Threshold configuration.

    Returns:
        Boolean array: True where pixel is classified as rock.
    """
    if thresholds is None:
        thresholds = ThresholdConfig()

    # Rock: high polarization AND surface/double-bounce dominant
    high_m = m > thresholds.rock_m_min
    surface_or_dihedral = (dominant_mechanism == 0) | (dominant_mechanism == 2)

    rock_flag = high_m & surface_or_dihedral

    # Mask NaN
    nan_mask = ~np.isfinite(m)
    rock_flag[nan_mask] = False

    count = np.sum(rock_flag)
    total = np.sum(~nan_mask)
    logger.info(
        f"Rock candidates: {count} / {total} pixels "
        f"({count / max(total, 1) * 100:.2f}%)"
    )

    return rock_flag


def generate_diagnostic_flags(
    l_cpr: np.ndarray,
    dop: np.ndarray,
    m: np.ndarray,
    dominant_mechanism: np.ndarray,
    s_cpr: Optional[np.ndarray] = None,
    thresholds: Optional[ThresholdConfig] = None,
) -> dict[str, np.ndarray]:
    """Generate all diagnostic classification flags.

    Args:
        l_cpr: L-band CPR (H×W).
        dop: Degree of polarization (H×W).
        m: m from m-χ decomposition (H×W).
        dominant_mechanism: Scattering mechanism map (H×W).
        s_cpr: S-band CPR (H×W), optional.
        thresholds: Threshold configuration.

    Returns:
        Dict with boolean arrays:
            - 'ice_flag': Ice candidate pixels
            - 'rock_flag': Rock candidate pixels
            - 'ambiguous_flag': Pixels that are neither ice nor rock
    """
    ice = classify_ice_candidates(l_cpr, dop, s_cpr, thresholds)
    rock = classify_rock(m, dominant_mechanism, thresholds)

    # Resolve conflicts: if both ice and rock, mark as ambiguous
    conflict = ice & rock
    ambiguous = ~ice & ~rock

    if np.any(conflict):
        logger.warning(
            f"Found {np.sum(conflict)} pixels classified as both ice and rock. "
            "Marking as ambiguous."
        )
        ice[conflict] = False
        rock[conflict] = False
        ambiguous[conflict] = True

    return {
        'ice_flag': ice,
        'rock_flag': rock,
        'ambiguous_flag': ambiguous,
    }
