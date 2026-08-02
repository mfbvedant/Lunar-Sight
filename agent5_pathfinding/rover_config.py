"""
Rover Configuration
====================
Configurable dataclass for rover parameters. Defaults are Pragyan-class (ISRO).
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml


@dataclass
class RoverConfig:
    """Configurable rover physical parameters.

    All values can be overridden via mission_config.yaml.
    """

    name: str = "Generic Pragyan-class"
    mass_kg: float = 27.0
    wheel_count: int = 6
    wheel_radius_m: float = 0.105
    wheel_width_m: float = 0.14
    max_slope_deg: float = 10.0
    battery_capacity_wh: float = 50.0
    power_draw_nominal_w: float = 15.0
    power_draw_max_w: float = 50.0
    thermal_min_temp_k: float = 173.0
    thermal_max_temp_k: float = 323.0
    max_velocity_ms: float = 0.01   # ~1 cm/s
    suspension_type: str = "rocker_bogie"

    @classmethod
    def from_config(cls, config: dict) -> "RoverConfig":
        rover = config.get("rover", {})
        return cls(**{k: rover[k] for k in rover if k in cls.__dataclass_fields__})

    @classmethod
    def from_yaml(cls, path: str) -> "RoverConfig":
        with open(path) as f:
            return cls.from_config(yaml.safe_load(f))

    @property
    def weight_n(self) -> float:
        """Rover weight on the Moon (N)."""
        return self.mass_kg * 1.62

    @property
    def contact_area_m2(self) -> float:
        """Approximate ground contact area of all wheels."""
        single_wheel = self.wheel_width_m * 0.1  # ~10% of diameter sinks
        return single_wheel * self.wheel_count
