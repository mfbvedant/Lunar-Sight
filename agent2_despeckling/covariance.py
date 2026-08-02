"""
Covariance Matrix Computation
===============================
Compute C₂ (dual-pol) or C₃ (quad-pol) covariance matrices from raw
complex scattering vectors with spatial multi-look averaging.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def compute_c2_matrix(
    e_h: np.ndarray,
    e_v: np.ndarray,
    window_size: int = 5,
) -> dict[str, np.ndarray]:
    """Compute the 2×2 covariance matrix C₂ from dual-pol data.

    C₂ elements with spatial averaging (multi-look):
        C₁₁ = ⟨|E_H|²⟩
        C₁₂ = ⟨E_H · E_V*⟩
        C₂₁ = ⟨E_V · E_H*⟩ = C₁₂*
        C₂₂ = ⟨|E_V|²⟩

    The expectation ⟨·⟩ is computed via a uniform spatial averaging window.

    Args:
        e_h: Complex H-pol E-field (H×W), dtype=complex64/128.
        e_v: Complex V-pol E-field (H×W), dtype=complex64/128.
        window_size: Size of the spatial averaging window.

    Returns:
        Dict with keys 'C11', 'C12', 'C21', 'C22' — each an (H×W) array.
        C11 and C22 are real; C12 and C21 are complex.
    """
    # Diagonal elements (real-valued)
    c11 = uniform_filter(np.abs(e_h) ** 2, size=window_size)
    c22 = uniform_filter(np.abs(e_v) ** 2, size=window_size)

    # Off-diagonal (complex): average real and imag parts separately
    cross = e_h * np.conj(e_v)
    c12_real = uniform_filter(np.real(cross), size=window_size)
    c12_imag = uniform_filter(np.imag(cross), size=window_size)
    c12 = c12_real + 1j * c12_imag
    c21 = np.conj(c12)

    return {
        'C11': c11.astype(np.float64),
        'C12': c12.astype(np.complex128),
        'C21': c21.astype(np.complex128),
        'C22': c22.astype(np.float64),
    }


def compute_c3_matrix(
    e_hh: np.ndarray,
    e_hv: np.ndarray,
    e_vv: np.ndarray,
    window_size: int = 5,
) -> dict[str, np.ndarray]:
    """Compute the 3×3 covariance matrix C₃ from quad-pol data.

    Using the Pauli basis: k = [S_HH + S_VV, S_HH - S_VV, 2·S_HV]^T / √2

    For simplicity, we use the lexicographic basis:
        k = [S_HH, √2·S_HV, S_VV]^T

    C₃ = ⟨k · k†⟩

    Args:
        e_hh: Complex HH channel (H×W).
        e_hv: Complex HV channel (H×W). (= VH for monostatic).
        e_vv: Complex VV channel (H×W).
        window_size: Spatial averaging window size.

    Returns:
        Dict with keys 'Cij' for i,j in {1,2,3} — 9 elements of the 3×3 matrix.
    """
    sqrt2 = np.sqrt(2.0)
    k1 = e_hh
    k2 = sqrt2 * e_hv
    k3 = e_vv

    channels = [k1, k2, k3]
    result = {}

    for i in range(3):
        for j in range(3):
            cross = channels[i] * np.conj(channels[j])
            if i == j:
                # Diagonal: real-valued
                cij = uniform_filter(np.real(cross), size=window_size)
                result[f'C{i+1}{j+1}'] = cij.astype(np.float64)
            else:
                # Off-diagonal: complex
                cij_real = uniform_filter(np.real(cross), size=window_size)
                cij_imag = uniform_filter(np.imag(cross), size=window_size)
                result[f'C{i+1}{j+1}'] = (cij_real + 1j * cij_imag).astype(np.complex128)

    return result


def covariance_to_tensor(
    cov: dict[str, np.ndarray],
    matrix_size: int = 2,
) -> np.ndarray:
    """Pack a covariance matrix dict into a real-valued tensor.

    For a C₂ matrix (4 unique real values per pixel):
        Channel 0: Re(C₁₁) = C₁₁ (real)
        Channel 1: Re(C₁₂)
        Channel 2: Im(C₁₂)
        Channel 3: Re(C₂₂) = C₂₂ (real)

    For a C₃ matrix (9 unique real values per pixel):
        Channels 0-8: packed upper-triangle real/imag values.

    Args:
        cov: Covariance dict from compute_c2/c3_matrix.
        matrix_size: 2 for C₂, 3 for C₃.

    Returns:
        Real-valued tensor of shape (C, H, W).
    """
    if matrix_size == 2:
        channels = [
            np.real(cov['C11']),
            np.real(cov['C12']),
            np.imag(cov['C12']),
            np.real(cov['C22']),
        ]
    elif matrix_size == 3:
        # Pack upper triangle: C11, Re(C12), Im(C12), Re(C13), Im(C13),
        #                       C22, Re(C23), Im(C23), C33
        channels = [
            np.real(cov['C11']),
            np.real(cov['C12']), np.imag(cov['C12']),
            np.real(cov['C13']), np.imag(cov['C13']),
            np.real(cov['C22']),
            np.real(cov['C23']), np.imag(cov['C23']),
            np.real(cov['C33']),
        ]
    else:
        raise ValueError(f"Unsupported matrix size: {matrix_size}")

    return np.stack(channels, axis=0).astype(np.float32)


def tensor_to_covariance(
    tensor: np.ndarray,
    matrix_size: int = 2,
) -> dict[str, np.ndarray]:
    """Unpack a real-valued tensor back into a covariance matrix dict.

    Inverse of covariance_to_tensor().

    Args:
        tensor: Real-valued tensor (C, H, W).
        matrix_size: 2 for C₂, 3 for C₃.

    Returns:
        Covariance dict.
    """
    if matrix_size == 2:
        return {
            'C11': tensor[0],
            'C12': tensor[1] + 1j * tensor[2],
            'C21': tensor[1] - 1j * tensor[2],
            'C22': tensor[3],
        }
    elif matrix_size == 3:
        return {
            'C11': tensor[0],
            'C12': tensor[1] + 1j * tensor[2],
            'C13': tensor[3] + 1j * tensor[4],
            'C21': tensor[1] - 1j * tensor[2],
            'C22': tensor[5],
            'C23': tensor[6] + 1j * tensor[7],
            'C31': tensor[3] - 1j * tensor[4],
            'C32': tensor[6] - 1j * tensor[7],
            'C33': tensor[8],
        }
    else:
        raise ValueError(f"Unsupported matrix size: {matrix_size}")
