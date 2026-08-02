"""
Tests for m-χ Decomposition
==============================
Verify m-chi against synthetic covariance for pure scattering mechanisms.
"""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent3_polarimetry.mchi import (
    compute_mchi,
    compute_degree_of_polarization,
    dominant_scattering_mechanism,
)
from agent3_polarimetry.cpr import compute_cpr


class TestMchiDecomposition:
    """Test m-χ decomposition against known scattering mechanisms."""

    def test_fully_polarized_dop_equals_one(self):
        """For a fully polarized signal, m (DOP) should be 1.0.

        Fully H-polarized: S₁=1, S₂=1, S₃=0, S₄=0.
        m = √(1 + 0 + 0) / 1 = 1.
        """
        s1 = np.full((10, 10), 1.0)
        s2 = np.full((10, 10), 1.0)
        s3 = np.zeros((10, 10))
        s4 = np.zeros((10, 10))

        result = compute_mchi(s1, s2, s3, s4)
        np.testing.assert_allclose(result['m'], 1.0, atol=1e-10)

    def test_unpolarized_dop_equals_zero(self):
        """Unpolarized: S₂=S₃=S₄=0 → m=0."""
        s1 = np.full((10, 10), 1.0)
        s2 = np.zeros((10, 10))
        s3 = np.zeros((10, 10))
        s4 = np.zeros((10, 10))

        result = compute_mchi(s1, s2, s3, s4)
        np.testing.assert_allclose(result['m'], 0.0, atol=1e-10)

    def test_pure_volume_dominates_green(self):
        """Unpolarized signal → G (volume) should dominate.

        When m=0: R=0, G=√S₁, B=0.
        """
        s1 = np.full((10, 10), 4.0)
        s2 = np.zeros((10, 10))
        s3 = np.zeros((10, 10))
        s4 = np.zeros((10, 10))

        result = compute_mchi(s1, s2, s3, s4)

        np.testing.assert_allclose(result['R'], 0.0, atol=1e-10)
        np.testing.assert_allclose(result['G'], 2.0, atol=1e-10)  # √4 = 2
        np.testing.assert_allclose(result['B'], 0.0, atol=1e-10)

    def test_surface_scattering_dominates_blue(self):
        """Fully polarized with sin2χ=-1 → B (surface) dominates.

        sin2χ = -S₄/(m·S₁). For sin2χ = -1:
        Need S₄ = m·S₁ = S₁ (since m=1 for fully polarized).
        Set S₁=1, S₂=0, S₃=0, S₄=1.
        Then m = √(0+0+1)/1 = 1, sin2χ = -1/1 = -1.
        R = √(1·(1-1)/2) = 0
        B = √(1·(1+1)/2) = 1
        """
        s1 = np.full((10, 10), 1.0)
        s2 = np.zeros((10, 10))
        s3 = np.zeros((10, 10))
        s4 = np.full((10, 10), 1.0)

        result = compute_mchi(s1, s2, s3, s4)

        np.testing.assert_allclose(result['R'], 0.0, atol=1e-10)
        np.testing.assert_allclose(result['B'], 1.0, atol=1e-10)

    def test_double_bounce_dominates_red(self):
        """Fully polarized with sin2χ=+1 → R (double-bounce) dominates.

        Need S₄ = -m·S₁. Set S₁=1, S₂=0, S₃=0, S₄=-1.
        m = 1, sin2χ = -(-1)/1 = 1.
        R = √(1·(1+1)/2) = 1
        B = √(1·(1-1)/2) = 0
        """
        s1 = np.full((10, 10), 1.0)
        s2 = np.zeros((10, 10))
        s3 = np.zeros((10, 10))
        s4 = np.full((10, 10), -1.0)

        result = compute_mchi(s1, s2, s3, s4)

        np.testing.assert_allclose(result['R'], 1.0, atol=1e-10)
        np.testing.assert_allclose(result['B'], 0.0, atol=1e-10)

    def test_dop_in_zero_one_range(self):
        """DOP should always be in [0, 1]."""
        s1 = np.random.rand(50, 50) * 10 + 0.1
        s2 = np.random.randn(50, 50)
        s3 = np.random.randn(50, 50)
        s4 = np.random.randn(50, 50)

        m = compute_degree_of_polarization(s1, s2, s3, s4)

        assert np.all(m >= 0.0)
        assert np.all(m <= 1.0)

    def test_rgb_non_negative(self):
        """R, G, B channels should all be non-negative."""
        s1 = np.random.rand(50, 50) * 5 + 0.1
        s2 = np.random.randn(50, 50) * 0.5
        s3 = np.random.randn(50, 50) * 0.5
        s4 = np.random.randn(50, 50) * 0.5

        result = compute_mchi(s1, s2, s3, s4)

        assert np.all(result['R'] >= 0)
        assert np.all(result['G'] >= 0)
        assert np.all(result['B'] >= 0)

    def test_output_shapes(self):
        """All outputs should have same shape as inputs."""
        shape = (32, 64)
        s1 = np.ones(shape)
        s2 = np.zeros(shape)
        s3 = np.zeros(shape)
        s4 = np.zeros(shape)

        result = compute_mchi(s1, s2, s3, s4)

        for key in ['m', 'chi', 'sin2chi', 'R', 'G', 'B']:
            assert result[key].shape == shape, f"{key} shape mismatch"


class TestDominantScattering:
    """Test dominant scattering mechanism classification."""

    def test_surface_dominant(self):
        """When B > R, G → classify as surface (0)."""
        R = np.full((5, 5), 0.1)
        G = np.full((5, 5), 0.2)
        B = np.full((5, 5), 0.9)

        result = dominant_scattering_mechanism(R, G, B)
        assert np.all(result == 0)  # surface

    def test_volume_dominant(self):
        """When G > R, B → classify as volume (1)."""
        R = np.full((5, 5), 0.1)
        G = np.full((5, 5), 0.9)
        B = np.full((5, 5), 0.2)

        result = dominant_scattering_mechanism(R, G, B)
        assert np.all(result == 1)  # volume

    def test_dihedral_dominant(self):
        """When R > G, B → classify as dihedral (2)."""
        R = np.full((5, 5), 0.9)
        G = np.full((5, 5), 0.2)
        B = np.full((5, 5), 0.1)

        result = dominant_scattering_mechanism(R, G, B)
        assert np.all(result == 2)  # dihedral


class TestCPR:
    """Test CPR computation."""

    def test_cpr_surface_scatter_below_one(self):
        """Surface scattering: SC < OC → CPR < 1."""
        s1 = np.full((10, 10), 2.0)
        s4 = np.full((10, 10), 0.5)  # OC > SC

        cpr = compute_cpr(s1, s4)
        assert np.all(cpr < 1.0)

    def test_cpr_volume_scatter_above_one(self):
        """Volume scattering: SC > OC → CPR > 1."""
        s1 = np.full((10, 10), 2.0)
        s4 = np.full((10, 10), -0.5)  # SC > OC

        cpr = compute_cpr(s1, s4)
        assert np.all(cpr > 1.0)

    def test_cpr_equal_sc_oc_is_one(self):
        """When SC = OC → CPR = 1."""
        s1 = np.full((10, 10), 2.0)
        s4 = np.zeros((10, 10))  # SC = OC = S1/2

        cpr = compute_cpr(s1, s4)
        np.testing.assert_allclose(cpr, 1.0, atol=1e-10)

    def test_cpr_handles_zero_denominator(self):
        """CPR should be NaN where S₁ + S₄ ≈ 0."""
        s1 = np.full((10, 10), 1.0)
        s4 = np.full((10, 10), -1.0)  # denominator = 0

        cpr = compute_cpr(s1, s4)
        assert np.all(np.isnan(cpr))
