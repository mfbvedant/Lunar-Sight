"""
LunarSight Visualization
=========================
Shared plotting utilities for DEM rendering, CPR maps, polarimetric RGB
composites, path overlays, and general map visualization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _get_plt():
    """Lazy import matplotlib to avoid import errors when not installed."""
    import matplotlib.pyplot as plt
    return plt


def _get_colors():
    """Lazy import matplotlib colors."""
    import matplotlib.colors as mcolors
    return mcolors


# ============================================================
# DEM Visualization
# ============================================================

def plot_dem(
    dem: np.ndarray,
    title: str = "Digital Elevation Model",
    cmap: str = "terrain",
    colorbar_label: str = "Elevation (m)",
    save_path: Optional[str | Path] = None,
    figsize: tuple[int, int] = (10, 8),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Plot a DEM elevation map.

    Args:
        dem: 2D array of elevation values.
        title: Plot title.
        cmap: Matplotlib colormap name.
        colorbar_label: Label for the colorbar.
        save_path: If provided, save figure to this path.
        figsize: Figure size (width, height).
        vmin: Minimum value for colormap scaling.
        vmax: Maximum value for colormap scaling.
    """
    plt = _get_plt()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(dem, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(colorbar_label)
    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
        logger.info(f"Saved DEM plot → {save_path}")

    plt.show()


def plot_slope_map(
    slope: np.ndarray,
    title: str = "Slope Map (Horn's Algorithm)",
    cmap: str = "YlOrRd",
    save_path: Optional[str | Path] = None,
    figsize: tuple[int, int] = (10, 8),
    max_slope_deg: float = 45.0,
) -> None:
    """Plot a slope map derived from DEM.

    Args:
        slope: 2D array of slope values in degrees.
        title: Plot title.
        cmap: Colormap — warm colors for steep slopes.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
        max_slope_deg: Maximum slope for colormap clipping.
    """
    plt = _get_plt()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(slope, cmap=cmap, vmin=0, vmax=max_slope_deg)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Slope (degrees)")
    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
        logger.info(f"Saved slope plot → {save_path}")

    plt.show()


# ============================================================
# Radar / CPR Maps
# ============================================================

def plot_cpr_map(
    cpr: np.ndarray,
    title: str = "Circular Polarization Ratio (CPR)",
    cmap: str = "coolwarm",
    save_path: Optional[str | Path] = None,
    figsize: tuple[int, int] = (10, 8),
    cpr_threshold: float = 1.0,
) -> None:
    """Plot a CPR map with threshold overlay.

    CPR > 1 indicates potential ice deposits (volume scattering).

    Args:
        cpr: 2D CPR array.
        title: Plot title.
        cmap: Colormap.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
        cpr_threshold: Threshold line for ice detection.
    """
    plt = _get_plt()
    colors = _get_colors()

    fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))

    # Left: CPR heatmap
    ax1 = axes[0]
    # Center colormap around threshold
    norm = colors.TwoSlopeNorm(vmin=0, vcenter=cpr_threshold, vmax=3.0)
    im = ax1.imshow(cpr, cmap=cmap, norm=norm)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label("CPR")

    # Right: Binary threshold
    ax2 = axes[1]
    ice_mask = cpr > cpr_threshold
    ax2.imshow(ice_mask, cmap="Blues", vmin=0, vmax=1)
    ax2.set_title(f"CPR > {cpr_threshold} (Ice Candidates)", fontsize=14)

    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
        logger.info(f"Saved CPR plot → {save_path}")

    plt.show()


# ============================================================
# Polarimetric RGB (m-χ Decomposition)
# ============================================================

def plot_mchi_rgb(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    title: str = "m-χ Decomposition (R=Dbl, G=Vol, B=Srf)",
    save_path: Optional[str | Path] = None,
    figsize: tuple[int, int] = (10, 8),
    percentile_clip: float = 2.0,
) -> None:
    """Plot m-χ decomposition as an RGB composite.

    R = double-bounce / even scatter, G = volume scatter, B = surface / odd scatter.

    Args:
        red: Red channel array (double-bounce).
        green: Green channel array (volume).
        blue: Blue channel array (surface).
        title: Plot title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
        percentile_clip: Percentile for contrast stretching.
    """
    plt = _get_plt()

    def _normalize(arr: np.ndarray, pclip: float) -> np.ndarray:
        """Percentile-clip and normalize to [0, 1]."""
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            return np.zeros_like(arr)
        lo = np.percentile(valid, pclip)
        hi = np.percentile(valid, 100 - pclip)
        clipped = np.clip(arr, lo, hi)
        rng = hi - lo
        if rng < 1e-10:
            return np.zeros_like(arr)
        return (clipped - lo) / rng

    r = _normalize(red, percentile_clip)
    g = _normalize(green, percentile_clip)
    b = _normalize(blue, percentile_clip)

    # Handle NaN → 0
    r = np.nan_to_num(r, nan=0.0)
    g = np.nan_to_num(g, nan=0.0)
    b = np.nan_to_num(b, nan=0.0)

    rgb = np.stack([r, g, b], axis=-1)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(rgb)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
        logger.info(f"Saved m-χ RGB plot → {save_path}")

    plt.show()


# ============================================================
# Ice Mask Overlay
# ============================================================

def plot_ice_mask_overlay(
    dem: np.ndarray,
    ice_mask: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    title: str = "Ice Detection Overlay",
    save_path: Optional[str | Path] = None,
    figsize: tuple[int, int] = (12, 8),
) -> None:
    """Plot ice mask overlaid on DEM.

    Args:
        dem: 2D DEM array.
        ice_mask: Binary ice mask (0=rock, 1=ice).
        confidence: Optional confidence map [0, 1].
        title: Plot title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
    """
    plt = _get_plt()
    colors = _get_colors()

    ncols = 2 if confidence is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(figsize[0] * ncols / 2, figsize[1]))
    if ncols == 1:
        axes = [axes]

    # DEM with ice overlay
    ax = axes[0]
    ax.imshow(dem, cmap="gray", alpha=0.7)
    ice_cmap = colors.ListedColormap(['none', '#00BFFF'])  # Transparent + cyan
    ax.imshow(ice_mask, cmap=ice_cmap, alpha=0.5, vmin=0, vmax=1)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Confidence heatmap
    if confidence is not None:
        ax2 = axes[1]
        im = ax2.imshow(confidence, cmap="plasma", vmin=0, vmax=1)
        ax2.set_title("Confidence Map", fontsize=14, fontweight='bold')
        fig.colorbar(im, ax=ax2, shrink=0.8)

    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
        logger.info(f"Saved ice overlay plot → {save_path}")

    plt.show()


# ============================================================
# Traverse Path Visualization
# ============================================================

def plot_traverse_path(
    dem: np.ndarray,
    ice_mask: np.ndarray,
    waypoints: list[dict],
    title: str = "Optimal Rover Traverse",
    save_path: Optional[str | Path] = None,
    figsize: tuple[int, int] = (12, 10),
    color_by: str = "slip",
) -> None:
    """Plot the planned rover traverse on DEM with ice overlay.

    Args:
        dem: 2D DEM array.
        ice_mask: Binary ice mask.
        waypoints: List of waypoint dicts with keys: x, y, slip, battery, etc.
        title: Plot title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
        color_by: Which field to color path segments by ("slip", "battery").
    """
    plt = _get_plt()
    colors = _get_colors()

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Background: DEM
    ax.imshow(dem, cmap="gray", alpha=0.6)

    # Overlay: Ice mask
    ice_cmap = colors.ListedColormap(['none', '#00BFFF80'])
    ax.imshow(ice_mask, cmap=ice_cmap, alpha=0.4, vmin=0, vmax=1)

    if waypoints:
        xs = [wp["x"] for wp in waypoints]
        ys = [wp["y"] for wp in waypoints]
        values = [wp.get(color_by, 0.0) for wp in waypoints]

        scatter = ax.scatter(
            xs, ys,
            c=values, cmap="RdYlGn_r", s=8,
            edgecolors='none', zorder=5,
        )
        ax.plot(xs, ys, 'w-', linewidth=0.5, alpha=0.6, zorder=4)
        fig.colorbar(scatter, ax=ax, shrink=0.8, label=color_by.capitalize())

        # Mark start and end
        ax.plot(xs[0], ys[0], 'g^', markersize=12, zorder=10, label="Start")
        ax.plot(xs[-1], ys[-1], 'r*', markersize=14, zorder=10, label="Target")
        ax.legend(loc='upper right', fontsize=10)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
        logger.info(f"Saved traverse plot → {save_path}")

    plt.show()


# ============================================================
# Multi-Panel Summary
# ============================================================

def plot_pipeline_summary(
    dem: np.ndarray,
    slope: np.ndarray,
    cpr: np.ndarray,
    ice_mask: np.ndarray,
    waypoints: Optional[list[dict]] = None,
    save_path: Optional[str | Path] = None,
    figsize: tuple[int, int] = (20, 12),
) -> None:
    """Plot a 2×2 (or 2×3) summary of the full pipeline outputs.

    Args:
        dem: DEM elevation array.
        slope: Slope array in degrees.
        cpr: CPR array (any band).
        ice_mask: Binary ice mask.
        waypoints: Optional traverse waypoints.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
    """
    plt = _get_plt()
    colors = _get_colors()

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # (0,0) DEM
    ax = axes[0, 0]
    ax.imshow(dem, cmap="terrain")
    ax.set_title("DEM Elevation", fontweight='bold')

    # (0,1) Slope
    ax = axes[0, 1]
    ax.imshow(slope, cmap="YlOrRd", vmin=0, vmax=30)
    ax.set_title("Terrain Slope (°)", fontweight='bold')

    # (1,0) CPR
    ax = axes[1, 0]
    norm = colors.TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=3.0)
    ax.imshow(cpr, cmap="coolwarm", norm=norm)
    ax.set_title("CPR (ice: >1)", fontweight='bold')

    # (1,1) Ice + Path
    ax = axes[1, 1]
    ax.imshow(dem, cmap="gray", alpha=0.6)
    ice_cmap = colors.ListedColormap(['none', '#00BFFF'])
    ax.imshow(ice_mask, cmap=ice_cmap, alpha=0.5, vmin=0, vmax=1)
    if waypoints:
        xs = [wp["x"] for wp in waypoints]
        ys = [wp["y"] for wp in waypoints]
        ax.plot(xs, ys, 'lime', linewidth=2, zorder=5)
        ax.plot(xs[0], ys[0], 'g^', markersize=10, zorder=10)
        ax.plot(xs[-1], ys[-1], 'r*', markersize=12, zorder=10)
    ax.set_title("Ice Map + Traverse", fontweight='bold')

    plt.suptitle("LunarSight Pipeline Summary", fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
        logger.info(f"Saved pipeline summary → {save_path}")

    plt.show()
