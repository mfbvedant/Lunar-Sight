"""
m-χ Decomposition
==================
Compute the m-chi polarimetric decomposition from Stokes parameters.

The m-χ decomposition separates radar backscatter into three scattering
mechanisms using the degree of polarization (m) and Poincaré ellipticity
angle (χ):
    - Red: Even/double-bounce scattering (urban, rocky dihedral)
    - Green: Volumetric scattering (rough surfaces, buried material)
    - Blue: Odd/surface scattering (smooth surfaces)

Reference:
    Raney, R.K. (2007). "Hybrid-polarity SAR architecture."
    IEEE TGRS, 45(11), 3397-3404.
"""

from __future__ import annotations

import numpy as np

from shared.constants import EPS


def compute_mchi(
    s1: np.ndarray,
    s2: np.ndarray,
    s3: np.ndarray,
    s4: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute the full m-χ decomposition from Stokes parameters.

    Args:
        s1: Stokes S₁ (total power), shape (H×W).
        s2: Stokes S₂, shape (H×W).
        s3: Stokes S₃, shape (H×W).
        s4: Stokes S₄, shape (H×W).

    Returns:
        Dict with keys:
            - 'm': degree of polarization [0, 1]
            - 'chi': signed ellipticity angle (radians)
            - 'sin2chi': sin(2χ) for direct use in RGB
            - 'R': red channel (even/double-bounce)
            - 'G': green channel (volumetric)
            - 'B': blue channel (odd/surface)
    """
    # Degree of polarization: m = √(S₂² + S₃² + S₄²) / S₁
    polarized_power = np.sqrt(s2 ** 2 + s3 ** 2 + s4 ** 2)
    m = np.full_like(s1, 0.0, dtype=np.float64)
    valid = np.abs(s1) > EPS
    m[valid] = polarized_power[valid] / s1[valid]
    m = np.clip(m, 0.0, 1.0)

    # Poincaré ellipticity: sin(2χ) = -S₄ / (m · S₁)
    denom = m * s1
    sin2chi = np.full_like(s1, 0.0, dtype=np.float64)
    valid_chi = np.abs(denom) > EPS
    sin2chi[valid_chi] = -s4[valid_chi] / denom[valid_chi]
    sin2chi = np.clip(sin2chi, -1.0, 1.0)

    # Chi angle
    chi = 0.5 * np.arcsin(sin2chi)

    # RGB scattering power decomposition
    # R = √(m · S₁ · (1 + sin2χ) / 2)   — even/double-bounce
    # G = √(S₁ · (1 - m))                — volume
    # B = √(m · S₁ · (1 - sin2χ) / 2)    — odd/surface
    ms1 = m * s1

    r_arg = ms1 * (1.0 + sin2chi) / 2.0
    g_arg = s1 * (1.0 - m)
    b_arg = ms1 * (1.0 - sin2chi) / 2.0

    # Ensure non-negative before sqrt
    r_arg = np.maximum(r_arg, 0.0)
    g_arg = np.maximum(g_arg, 0.0)
    b_arg = np.maximum(b_arg, 0.0)

    R = np.sqrt(r_arg)
    G = np.sqrt(g_arg)
    B = np.sqrt(b_arg)

    # Handle NaN/Inf from edge cases
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    G = np.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)
    B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        'm': m,
        'chi': chi,
        'sin2chi': sin2chi,
        'R': R,
        'G': G,
        'B': B,
    }


def compute_degree_of_polarization(
    s1: np.ndarray,
    s2: np.ndarray,
    s3: np.ndarray,
    s4: np.ndarray,
) -> np.ndarray:
    """Compute only the degree of polarization (DOP / m).

    DOP = √(S₂² + S₃² + S₄²) / S₁

    Values:
        - m = 0: completely unpolarized
        - m = 1: completely polarized
        - Low m (< 0.13): ice candidate (per Sinha et al.)

    Args:
        s1, s2, s3, s4: Stokes parameters (H×W each).

    Returns:
        DOP array in [0, 1].
    """
    m = np.sqrt(s2 ** 2 + s3 ** 2 + s4 ** 2)
    valid = np.abs(s1) > EPS
    result = np.full_like(s1, 0.0, dtype=np.float64)
    result[valid] = m[valid] / s1[valid]
    return np.clip(result, 0.0, 1.0)


def dominant_scattering_mechanism(
    R: np.ndarray,
    G: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """Classify each pixel by its dominant scattering mechanism.

    Args:
        R: Even/double-bounce power.
        G: Volume scattering power.
        B: Surface/odd-bounce power.

    Returns:
        Classification array: 0=surface, 1=volume, 2=double-bounce.
    """
    stack = np.stack([B, G, R], axis=0)  # B=0, G=1, R=2
    return np.argmax(stack, axis=0).astype(np.uint8)


def mchi_statistics(mchi_result: dict[str, np.ndarray]) -> dict[str, dict]:
    """Compute summary statistics for m-χ decomposition results.

    Args:
        mchi_result: Dict from compute_mchi().

    Returns:
        Nested dict with per-component statistics.
    """
    stats = {}
    for key in ['m', 'R', 'G', 'B']:
        arr = mchi_result[key]
        valid = arr[np.isfinite(arr)]
        if len(valid) > 0:
            stats[key] = {
                'mean': float(np.mean(valid)),
                'std': float(np.std(valid)),
                'min': float(np.min(valid)),
                'max': float(np.max(valid)),
                'median': float(np.median(valid)),
            }
    return stats
