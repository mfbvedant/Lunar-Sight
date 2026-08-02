"""
Tests for Stokes Parameters
==============================
Verify Stokes computation against analytically known polarization states.
"""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent3_polarimetry.stokes import (
    compute_stokes_from_covariance,
    validate_stokes,
)


class TestStokesParameters:
    """Test Stokes parameter computation."""

    def test_horizontally_polarized(self):
        """Fully H-polarized: E_H=1, E_V=0 → S₁=1, S₂=1, S₃=0, S₄=0."""
        c11 = np.ones((10, 10))      # |E_H|² = 1
        c22 = np.zeros((10, 10))     # |E_V|² = 0
        c12 = np.zeros((10, 10), dtype=complex)
        c21 = np.zeros((10, 10), dtype=complex)

        s1, s2, s3, s4 = compute_stokes_from_covariance(c11, c12, c21, c22)

        np.testing.assert_allclose(s1, 1.0, atol=1e-10)
        np.testing.assert_allclose(s2, 1.0, atol=1e-10)
        np.testing.assert_allclose(s3, 0.0, atol=1e-10)
        np.testing.assert_allclose(s4, 0.0, atol=1e-10)

    def test_vertically_polarized(self):
        """Fully V-polarized: E_H=0, E_V=1 → S₁=1, S₂=-1, S₃=0, S₄=0."""
        c11 = np.zeros((10, 10))
        c22 = np.ones((10, 10))
        c12 = np.zeros((10, 10), dtype=complex)
        c21 = np.zeros((10, 10), dtype=complex)

        s1, s2, s3, s4 = compute_stokes_from_covariance(c11, c12, c21, c22)

        np.testing.assert_allclose(s1, 1.0, atol=1e-10)
        np.testing.assert_allclose(s2, -1.0, atol=1e-10)
        np.testing.assert_allclose(s3, 0.0, atol=1e-10)
        np.testing.assert_allclose(s4, 0.0, atol=1e-10)

    def test_45_degree_linear(self):
        """45° linear: E_H=E_V=1/√2 → S₁=1, S₂=0, S₃=1, S₄=0."""
        val = 0.5  # |1/√2|² = 0.5
        c11 = np.full((10, 10), val)
        c22 = np.full((10, 10), val)
        # Cross term: E_H * E_V* = (1/√2)(1/√2) = 0.5 (real)
        c12 = np.full((10, 10), 0.5 + 0j)
        c21 = np.full((10, 10), 0.5 + 0j)

        s1, s2, s3, s4 = compute_stokes_from_covariance(c11, c12, c21, c22)

        np.testing.assert_allclose(s1, 1.0, atol=1e-10)
        np.testing.assert_allclose(s2, 0.0, atol=1e-10)
        np.testing.assert_allclose(s3, 1.0, atol=1e-10)
        np.testing.assert_allclose(s4, 0.0, atol=1e-10)

    def test_right_circular_polarized(self):
        """Right circular: E_H=1/√2, E_V=j/√2
        C₁₂ = E_H * E_V* = (1/√2)(-j/√2) = -j/2
        S₃ = 2·Re(-j/2) = 0
        S₄ = -2·Im(-j/2) = -2·(-0.5) = 1
        """
        c11 = np.full((10, 10), 0.5)
        c22 = np.full((10, 10), 0.5)
        c12 = np.full((10, 10), -0.5j)
        c21 = np.full((10, 10), 0.5j)

        s1, s2, s3, s4 = compute_stokes_from_covariance(c11, c12, c21, c22)

        np.testing.assert_allclose(s1, 1.0, atol=1e-10)
        np.testing.assert_allclose(s2, 0.0, atol=1e-10)
        np.testing.assert_allclose(s3, 0.0, atol=1e-10)
        np.testing.assert_allclose(s4, 1.0, atol=1e-10)

    def test_unpolarized(self):
        """Unpolarized: C₁₁=C₂₂=0.5, C₁₂=0 → S₂=S₃=S₄=0."""
        c11 = np.full((10, 10), 0.5)
        c22 = np.full((10, 10), 0.5)
        c12 = np.zeros((10, 10), dtype=complex)
        c21 = np.zeros((10, 10), dtype=complex)

        s1, s2, s3, s4 = compute_stokes_from_covariance(c11, c12, c21, c22)

        np.testing.assert_allclose(s1, 1.0, atol=1e-10)
        np.testing.assert_allclose(s2, 0.0, atol=1e-10)
        np.testing.assert_allclose(s3, 0.0, atol=1e-10)
        np.testing.assert_allclose(s4, 0.0, atol=1e-10)

    def test_physical_constraint(self):
        """Verify S₁² ≥ S₂² + S₃² + S₄² for valid Stokes."""
        c11 = np.full((10, 10), 1.0)
        c22 = np.full((10, 10), 0.3)
        c12 = np.full((10, 10), 0.2 + 0.1j)
        c21 = np.conj(c12)

        s1, s2, s3, s4 = compute_stokes_from_covariance(c11, c12, c21, c22)
        stats = validate_stokes(s1, s2, s3, s4)

        # Should have zero violations for physically valid covariance
        assert stats["violation_rate"] == 0.0

    def test_output_shapes(self):
        """Stokes arrays should match input shape."""
        shape = (32, 64)
        c11 = np.random.rand(*shape)
        c22 = np.random.rand(*shape)
        c12 = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        c21 = np.conj(c12)

        s1, s2, s3, s4 = compute_stokes_from_covariance(c11, c12, c21, c22)

        assert s1.shape == shape
        assert s2.shape == shape
        assert s3.shape == shape
        assert s4.shape == shape

    def test_s1_non_negative(self):
        """Total power S₁ should always be non-negative."""
        c11 = np.random.rand(50, 50) * 2
        c22 = np.random.rand(50, 50) * 2
        c12 = np.zeros((50, 50), dtype=complex)
        c21 = np.zeros((50, 50), dtype=complex)

        s1, _, _, _ = compute_stokes_from_covariance(c11, c12, c21, c22)

        assert np.all(s1 >= 0)
