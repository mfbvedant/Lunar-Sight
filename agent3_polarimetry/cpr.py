"""
Circular Polarization Ratio (CPR)
==================================
Compute the CPR from Stokes parameters. CPR > 1 is a key diagnostic
for volume scattering associated with buried water ice deposits.

Reference:
    Sinha, R.K. et al. (2024). "Ice detection in permanently shadowed
    regions using Chandrayaan-2 DFSAR polarimetric data."
"""

from __future__ import annotations

import numpy as np

from shared.constants import EPS


def compute_cpr(
    s1: np.ndarray,
    s4: np.ndarray,
) -> np.ndarray:
    """Compute the Circular Polarization Ratio (CPR).

    CPR is defined as the ratio of the same-sense circular (SC) to
    opposite-sense circular (OC) power:

        CPR = SC / OC = (S₁ - S₄) / (S₁ + S₄)

    Where:
        - S₁ = total power
        - S₄ = Stokes V parameter (signed circular polarization)
        - SC = (S₁ - S₄) / 2 = same-sense circular power
        - OC = (S₁ + S₄) / 2 = opposite-sense circular power

    Interpretation:
        - CPR < 1: surface scattering dominant (typical terrain)
        - CPR ≈ 1: mixed scattering
        - CPR > 1: volume/subsurface scattering (ice signature)

    Args:
        s1: Stokes S₁ parameter (total power), shape (H×W).
        s4: Stokes S₄ parameter, shape (H×W).

    Returns:
        CPR array (H×W). Masked to NaN where denominator ≈ 0.
    """
    denominator = s1 + s4
    cpr = np.full_like(s1, np.nan, dtype=np.float64)

    # Avoid division by zero
    valid = np.abs(denominator) > EPS
    cpr[valid] = (s1[valid] - s4[valid]) / denominator[valid]

    # Clip extreme values (numerical artifacts)
    cpr = np.clip(cpr, -10.0, 10.0)

    return cpr


def compute_cpr_from_complex(
    e_h: np.ndarray,
    e_v: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """Compute CPR directly from complex E-field channels.

    Convenience function that computes Stokes internally.

    Args:
        e_h: Complex horizontal E-field (H×W).
        e_v: Complex vertical E-field (H×W).
        window_size: Spatial averaging window size.

    Returns:
        CPR array (H×W).
    """
    from agent3_polarimetry.stokes import compute_stokes_from_complex_channels

    s1, _, _, s4 = compute_stokes_from_complex_channels(e_h, e_v, window_size)
    return compute_cpr(s1, s4)


def compute_sc_oc_power(
    s1: np.ndarray,
    s4: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Same-sense Circular (SC) and Opposite-sense Circular (OC) power.

    Args:
        s1: Total power (Stokes S₁).
        s4: Stokes S₄ parameter.

    Returns:
        Tuple of (SC, OC) power arrays.
    """
    sc = (s1 - s4) / 2.0
    oc = (s1 + s4) / 2.0
    return sc, oc


def cpr_statistics(cpr: np.ndarray) -> dict[str, float]:
    """Compute summary statistics for a CPR map.

    Args:
        cpr: CPR array.

    Returns:
        Dict of statistics including percentage of ice-candidate pixels.
    """
    valid = cpr[np.isfinite(cpr)]
    if len(valid) == 0:
        return {"valid_pixels": 0}

    return {
        "valid_pixels": int(len(valid)),
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "std": float(np.std(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "pct_above_1": float(np.sum(valid > 1.0) / len(valid) * 100),
        "pct_above_1_5": float(np.sum(valid > 1.5) / len(valid) * 100),
    }
