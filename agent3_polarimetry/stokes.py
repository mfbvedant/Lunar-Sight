"""
Stokes Parameters
==================
Compute the four Stokes parameters (S₁, S₂, S₃, S₄) from a 2×2 covariance
matrix in the circular polarization transmit basis (as used by DFSAR).

Reference:
    Raney, R.K. et al. (2012). "The m-chi decomposition of hybrid
    dual-pol SAR data." IEEE GRSL.
"""

from __future__ import annotations

import numpy as np

from shared.constants import EPS


def compute_stokes_from_covariance(
    c11: np.ndarray,
    c12: np.ndarray,
    c21: np.ndarray,
    c22: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Stokes parameters from C₂ covariance matrix elements.

    The covariance matrix C₂ for hybrid (circular TX) dual-pol data is:

        C₂ = [ ⟨|E_H|²⟩     ⟨E_H · E_V*⟩ ]
             [ ⟨E_V · E_H*⟩   ⟨|E_V|²⟩   ]

    Where E_H and E_V are the horizontal and vertical received electric fields.

    The Stokes parameters are derived as:
        S₁ = C₁₁ + C₂₂ = ⟨|E_H|²⟩ + ⟨|E_V|²⟩  (total power)
        S₂ = C₁₁ - C₂₂ = ⟨|E_H|²⟩ - ⟨|E_V|²⟩
        S₃ = 2 · Re(C₁₂) = 2 · Re(⟨E_H · E_V*⟩)
        S₄ = -2 · Im(C₁₂) = -2 · Im(⟨E_H · E_V*⟩)

    Args:
        c11: ⟨|E_H|²⟩ — real-valued (H×W).
        c12: ⟨E_H · E_V*⟩ — complex-valued (H×W).
        c21: ⟨E_V · E_H*⟩ — complex-valued (H×W). Should be conj(c12).
        c22: ⟨|E_V|²⟩ — real-valued (H×W).

    Returns:
        Tuple of (S1, S2, S3, S4) — all real-valued H×W arrays.
    """
    # Ensure real-valued diagonal elements
    c11_real = np.real(c11).astype(np.float64)
    c22_real = np.real(c22).astype(np.float64)

    s1 = c11_real + c22_real                  # Total power
    s2 = c11_real - c22_real                  # HH - VV difference
    s3 = 2.0 * np.real(c12).astype(np.float64)   # 2·Re(C₁₂)
    s4 = -2.0 * np.imag(c12).astype(np.float64)  # -2·Im(C₁₂)

    return s1, s2, s3, s4


def compute_stokes_from_complex_channels(
    e_h: np.ndarray,
    e_v: np.ndarray,
    window_size: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Stokes parameters directly from complex E-field channels.

    First computes the C₂ covariance matrix with spatial averaging,
    then derives Stokes parameters.

    Args:
        e_h: Complex horizontal E-field (H×W), dtype=complex64/128.
        e_v: Complex vertical E-field (H×W), dtype=complex64/128.
        window_size: Size of spatial averaging window for ⟨·⟩ operator.

    Returns:
        Tuple of (S1, S2, S3, S4).
    """
    from scipy.ndimage import uniform_filter

    # Compute covariance matrix elements with spatial averaging
    c11 = uniform_filter(np.abs(e_h) ** 2, size=window_size)
    c22 = uniform_filter(np.abs(e_v) ** 2, size=window_size)

    # Cross-terms need real/imag averaged separately
    cross = e_h * np.conj(e_v)
    c12_real = uniform_filter(np.real(cross), size=window_size)
    c12_imag = uniform_filter(np.imag(cross), size=window_size)
    c12 = c12_real + 1j * c12_imag

    c21 = np.conj(c12)

    return compute_stokes_from_covariance(c11, c12, c21, c22)


def validate_stokes(
    s1: np.ndarray,
    s2: np.ndarray,
    s3: np.ndarray,
    s4: np.ndarray,
) -> dict[str, float]:
    """Validate Stokes parameters against physical constraints.

    Physical constraint: S₁² ≥ S₂² + S₃² + S₄² (for partially polarized light).

    Returns:
        Dict with validation statistics.
    """
    valid_mask = np.isfinite(s1)
    s1_v = s1[valid_mask]
    s2_v = s2[valid_mask]
    s3_v = s3[valid_mask]
    s4_v = s4[valid_mask]

    lhs = s1_v ** 2
    rhs = s2_v ** 2 + s3_v ** 2 + s4_v ** 2

    # Physical constraint violation
    violations = np.sum(lhs < rhs - EPS)
    total = len(s1_v)

    stats = {
        "total_pixels": int(total),
        "constraint_violations": int(violations),
        "violation_rate": float(violations / max(total, 1)),
        "s1_mean": float(np.mean(s1_v)),
        "s1_std": float(np.std(s1_v)),
        "s1_min": float(np.min(s1_v)),
        "s1_max": float(np.max(s1_v)),
    }

    return stats
