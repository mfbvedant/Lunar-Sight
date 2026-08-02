"""
Tests for A* Pathfinder
=========================
Verify pathfinding on simple grids with known optimal solutions.
"""

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent5_pathfinding.kinodynamic_astar import kinodynamic_astar


class TestAStarBasic:

    def test_flat_grid_finds_path(self):
        """On flat terrain, A* should find a path."""
        slope = np.zeros((20, 20))
        path = kinodynamic_astar(
            slope, start=(0, 0), goal=(19, 19),
            pixel_size_m=1.0, max_iterations=100_000,
            battery_capacity_wh=500.0,
        )
        assert path is not None
        assert len(path) > 0
        # End point should be near goal
        assert path[-1]['y'] == 19
        assert path[-1]['x'] == 19

    def test_impassable_wall_avoids(self):
        """A* should avoid impassable terrain (slope > max)."""
        slope = np.zeros((10, 20))
        # Create a wall in the middle
        slope[:, 10] = 50.0  # Impassable slope
        # Leave a gap at row 5
        slope[5, 10] = 0.0

        path = kinodynamic_astar(
            slope, start=(5, 0), goal=(5, 19),
            pixel_size_m=1.0, max_iterations=100_000,
            battery_capacity_wh=500.0,
        )
        assert path is not None
        # Path must go through the gap at (5, 10)
        gap_visited = any(wp['y'] == 5 and wp['x'] == 10 for wp in path)
        assert gap_visited

    def test_no_path_returns_none(self):
        """When goal is completely walled off, return None."""
        slope = np.zeros((10, 10))
        # Complete wall
        slope[4, :] = 50.0
        slope[5, :] = 50.0

        path = kinodynamic_astar(
            slope, start=(0, 5), goal=(9, 5),
            pixel_size_m=10.0, max_iterations=50_000,
        )
        assert path is None

    def test_start_equals_goal(self):
        """If start == goal, should return a trivial path."""
        slope = np.zeros((10, 10))
        path = kinodynamic_astar(
            slope, start=(5, 5), goal=(5, 5),
            pixel_size_m=10.0,
        )
        assert path is not None
        assert len(path) >= 1

    def test_battery_constraint(self):
        """With very low battery, long paths should fail."""
        slope = np.zeros((50, 50))
        path = kinodynamic_astar(
            slope, start=(0, 0), goal=(49, 49),
            pixel_size_m=100.0,  # Large pixels = long distance
            battery_capacity_wh=0.001,  # Almost no battery
            max_iterations=50_000,
        )
        # Should fail due to battery constraint
        assert path is None

    def test_path_costs_monotonic(self):
        """Path costs should be monotonically increasing."""
        slope = np.random.rand(20, 20) * 3  # Gentle slopes
        path = kinodynamic_astar(
            slope, start=(0, 0), goal=(19, 19),
            pixel_size_m=10.0, max_iterations=100_000,
        )
        if path is not None and len(path) > 1:
            costs = [wp['cost'] for wp in path]
            for i in range(1, len(costs)):
                assert costs[i] >= costs[i-1]
