"""
Tests for Terramechanics
==========================
Verify soil mechanics models against known physical cases.
"""

import math
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent5_pathfinding.terramechanics import (
    mohr_coulomb,
    janosi_hanamoto,
    compute_slip_ratio,
    compute_drawbar_pull,
    is_traversable,
)


class TestMohrCoulomb:

    def test_zero_normal_stress(self):
        """At zero normal stress, shear = cohesion only."""
        tau = mohr_coulomb(sigma=0.0, c=170.0, phi_deg=33.0)
        assert abs(tau - 170.0) < 1e-10

    def test_increases_with_stress(self):
        """Shear strength increases with normal stress."""
        tau1 = mohr_coulomb(sigma=100.0)
        tau2 = mohr_coulomb(sigma=200.0)
        assert tau2 > tau1

    def test_known_value(self):
        """Verify against hand-calculated value."""
        # τ = 170 + 1000 * tan(33°) = 170 + 649.4 = 819.4
        tau = mohr_coulomb(sigma=1000.0, c=170.0, phi_deg=33.0)
        expected = 170.0 + 1000.0 * math.tan(math.radians(33.0))
        assert abs(tau - expected) < 0.1

    def test_zero_friction(self):
        """With zero friction angle, shear = cohesion regardless of stress."""
        tau = mohr_coulomb(sigma=5000.0, c=170.0, phi_deg=0.0)
        assert abs(tau - 170.0) < 1e-10


class TestJanosiHanamoto:

    def test_zero_displacement(self):
        """At zero displacement, shear = 0."""
        tau = janosi_hanamoto(sigma=1000.0, j=0.0)
        assert abs(tau) < 1e-10

    def test_large_displacement_converges(self):
        """At large displacement, shear approaches τ_max."""
        tau_max = mohr_coulomb(sigma=1000.0)
        tau_large = janosi_hanamoto(sigma=1000.0, j=1.0)  # 1m >> K=0.018m
        assert abs(tau_large - tau_max) / tau_max < 0.01  # Within 1%

    def test_monotonically_increasing(self):
        """Shear increases monotonically with displacement."""
        displacements = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
        shears = [janosi_hanamoto(sigma=500.0, j=j) for j in displacements]
        for i in range(1, len(shears)):
            assert shears[i] > shears[i - 1]


class TestSlipRatio:

    def test_flat_terrain_low_slip(self):
        """Flat terrain should have very low (near zero) slip."""
        slip = compute_slip_ratio(slope_deg=0.0)
        assert slip < 0.05

    def test_steep_terrain_high_slip(self):
        """Steep terrain should have higher slip."""
        slip_flat = compute_slip_ratio(slope_deg=0.0)
        slip_steep = compute_slip_ratio(slope_deg=8.0)
        assert slip_steep > slip_flat

    def test_slip_bounded(self):
        """Slip should be in [0, 1]."""
        for angle in range(0, 90):
            slip = compute_slip_ratio(slope_deg=float(angle))
            assert 0.0 <= slip <= 1.0

    def test_slip_roughly_mass_independent(self):
        """In the simplified model, slip ≈ sin(θ)/[cos(θ)·tan(φ)] — mass cancels.

        Both resistance and traction scale linearly with mass, so the
        slip ratio should be roughly stable across rover masses.
        """
        slip_light = compute_slip_ratio(slope_deg=5.0, mass_kg=10.0)
        slip_heavy = compute_slip_ratio(slope_deg=5.0, mass_kg=50.0)
        # Slip should be in the same ballpark (within 3x)
        assert slip_heavy < 3.0 * slip_light
        assert slip_light < 3.0 * slip_heavy


class TestTraversability:

    def test_flat_traversable(self):
        assert is_traversable(0.0) == True

    def test_over_max_slope_not_traversable(self):
        assert is_traversable(15.0, max_slope_deg=10.0) == False

    def test_moderate_slope_traversable(self):
        assert is_traversable(5.0, max_slope_deg=10.0) == True


class TestDrawbarPull:

    def test_flat_positive(self):
        """Drawbar pull on flat terrain should be positive."""
        dbp = compute_drawbar_pull(0.0)
        assert dbp > 0

    def test_steep_lower_than_flat(self):
        """Drawbar pull decreases with slope."""
        dbp_flat = compute_drawbar_pull(0.0)
        dbp_steep = compute_drawbar_pull(8.0)
        assert dbp_steep < dbp_flat
