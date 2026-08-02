"""
Tests for Horn's Slope Algorithm
==================================
Verify slope and aspect computation against analytically known terrain geometries.
"""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent1_ingestion.horn_slope import horn_slope, compute_hillshade


class TestHornSlope:
    """Test Horn's algorithm against known geometry."""

    def test_flat_plane_zero_slope(self):
        """A perfectly flat DEM should have zero slope everywhere."""
        dem = np.full((50, 50), 100.0)  # Constant elevation
        slope, aspect = horn_slope(dem, dx=10.0)

        np.testing.assert_allclose(slope, 0.0, atol=1e-10)

    def test_flat_plane_negative_one_aspect(self):
        """Flat areas should get aspect = -1 (undefined direction)."""
        dem = np.full((50, 50), 100.0)
        slope, aspect = horn_slope(dem, dx=10.0)

        assert np.all(aspect == -1.0)

    def test_45_degree_ramp_east(self):
        """A ramp rising 1m per 1m eastward = 45° slope.

        DEM: elevation increases linearly in X (columns).
        """
        rows, cols = 50, 50
        dx = 1.0
        dem = np.zeros((rows, cols))
        for j in range(cols):
            dem[:, j] = j * dx  # Rise = dx per pixel → 45°

        slope, aspect = horn_slope(dem, dx=dx)

        # Interior pixels (away from edges) should be ~45°
        interior_slope = slope[5:-5, 5:-5]
        np.testing.assert_allclose(interior_slope, 45.0, atol=0.5)

    def test_30_degree_ramp_south(self):
        """A ramp with tan(30°) = 0.577 rise per run → 30° slope.

        DEM: elevation increases in Y (rows, going south).
        """
        rows, cols = 50, 50
        dx = 10.0
        rise_per_pixel = dx * np.tan(np.radians(30))
        dem = np.zeros((rows, cols))
        for i in range(rows):
            dem[i, :] = i * rise_per_pixel

        slope, aspect = horn_slope(dem, dx=dx)

        interior_slope = slope[5:-5, 5:-5]
        np.testing.assert_allclose(interior_slope, 30.0, atol=1.0)

    def test_steep_slope(self):
        """Verify steep slopes (near vertical)."""
        rows, cols = 50, 50
        dx = 1.0
        rise_per_pixel = dx * np.tan(np.radians(80))
        dem = np.zeros((rows, cols))
        for j in range(cols):
            dem[:, j] = j * rise_per_pixel

        slope, _ = horn_slope(dem, dx=dx)

        interior = slope[5:-5, 5:-5]
        np.testing.assert_allclose(interior, 80.0, atol=2.0)

    def test_symmetric_cone(self):
        """A conical peak should have radially symmetric slope."""
        size = 101
        center = size // 2
        dem = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                dem[i, j] = max(0, 50 - dist)  # Cone peak at center

        slope, _ = horn_slope(dem, dx=1.0)

        # Slope should be roughly uniform away from peak and base
        ring = slope[30:40, center]
        assert np.std(ring) < 5.0  # Slope values should be consistent

    def test_output_shapes(self):
        """Slope and aspect should have same shape as input DEM."""
        dem = np.random.randn(100, 80)
        slope, aspect = horn_slope(dem, dx=5.0)

        assert slope.shape == dem.shape
        assert aspect.shape == dem.shape

    def test_slope_non_negative(self):
        """Slope should always be non-negative."""
        dem = np.random.randn(100, 100) * 50
        slope, _ = horn_slope(dem, dx=10.0)

        assert np.all(slope >= 0)

    def test_aspect_range(self):
        """Aspect should be in [0, 360) or -1 for flat areas."""
        dem = np.random.randn(100, 100) * 50
        _, aspect = horn_slope(dem, dx=10.0)

        non_flat = aspect[aspect >= 0]
        assert np.all(non_flat >= 0)
        assert np.all(non_flat < 360)

    def test_nonsquare_pixels(self):
        """Verify correct handling of non-square pixels (dx ≠ dy)."""
        rows, cols = 50, 50
        dx, dy = 5.0, 10.0
        dem = np.zeros((rows, cols))
        for j in range(cols):
            dem[:, j] = j * dx * np.tan(np.radians(30))

        slope, _ = horn_slope(dem, dx=dx, dy=dy)

        # Should still compute reasonable slopes
        interior = slope[5:-5, 5:-5]
        assert np.all(interior > 0)
        assert np.all(interior < 90)


class TestHillshade:
    """Test hillshade computation."""

    def test_hillshade_range(self):
        """Hillshade values should be in [0, 255]."""
        dem = np.random.randn(50, 50) * 20
        hs = compute_hillshade(dem, dx=10.0)

        assert hs.dtype == np.uint8
        assert np.all(hs >= 0)
        assert np.all(hs <= 255)

    def test_flat_terrain_uniform(self):
        """Flat terrain should have uniform hillshade."""
        dem = np.full((50, 50), 100.0)
        hs = compute_hillshade(dem, dx=10.0)

        # All values should be the same (within integer rounding)
        assert np.std(hs.astype(float)) < 2.0

    def test_output_shape(self):
        """Hillshade should have same shape as DEM."""
        dem = np.random.randn(80, 60) * 10
        hs = compute_hillshade(dem, dx=5.0)

        assert hs.shape == dem.shape
