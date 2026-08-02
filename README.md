# 🌑 LunarSight

**Multi-Agent Lunar Ice Detection & Pathfinding System**

LunarSight is a LangGraph-orchestrated pipeline that processes Chandrayaan-2 DFSAR (Dual-Frequency SAR) radar and LOLA DEM data to detect water ice deposits in permanently shadowed lunar south pole craters and plan physically survivable rover traverses to reach them.

---

## Architecture

```
Agent 1 ─→ Agent 2 ─→ Agent 3 ─→ Agent 4 ─→ [Coverage Check] ─→ Agent 5 ─→ [Path Check] ─→ Done
(Ingest)   (Despeckle)  (Polarimetry) (Segment)      ↓ retry            (Pathfind)   ↓ retry
                                                  Threshold Adapt                 Path Relax
```

| Agent | Role | Key Algorithm |
|-------|------|---------------|
| **Agent 1** | Data Ingestion | PDS4 parsing, polar stereographic reprojection, Horn's slope |
| **Agent 2** | SAR Despeckling | Complex-Valued CNN autoencoder (phase-preserving) |
| **Agent 3** | Polarimetric Features | Stokes parameters, CPR, m-χ decomposition |
| **Agent 4** | Ice Segmentation | Weakly-supervised U-Net with physics-based pseudo-labels |
| **Agent 5** | Pathfinding | Kinodynamic A* with Mohr-Coulomb terramechanics |

---

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (for training — inference works on CPU)

### Install

```bash
git clone https://github.com/YOUR_USERNAME/Lunar-Sight.git
cd Lunar-Sight/Lunar-Sight
pip install -r requirements.txt
```

### Run Full Pipeline

```bash
python main.py --config config/mission_config.yaml --agent all
```

### Run Individual Agent

```bash
python main.py --agent 3 --config config/mission_config.yaml
```

### Run on Google Colab

1. Open `notebooks/06_full_pipeline.ipynb` in Colab
2. Enable GPU runtime (Runtime → Change runtime type → T4 GPU)
3. Run all cells

---

## Project Structure

```
Lunar-Sight/
├── config/
│   ├── mission_config.yaml        # Target crater, data URLs, thresholds
│   └── training_config.yaml       # ML training hyperparameters
├── shared/                        # Foundation module
│   ├── state.py                   # LangGraph state schema
│   ├── constants.py               # Physics & geotechnical constants
│   ├── io_utils.py                # Tensor I/O, Drive helpers
│   ├── geo_utils.py               # CRS transforms
│   └── visualization.py           # Shared plotting
├── agent1_ingestion/              # Data download & preprocessing
│   ├── pds4_parser.py
│   ├── data_fetcher.py
│   ├── reprojection.py
│   ├── horn_slope.py
│   ├── tensor_builder.py
│   └── agent.py
├── agent2_despeckling/            # Complex-valued CNN denoiser
│   ├── covariance.py
│   ├── complex_layers.py
│   ├── cv_cnn_model.py
│   ├── loss.py
│   ├── dataset.py
│   ├── train.py
│   ├── inference.py
│   └── agent.py
├── agent3_polarimetry/            # Polarimetric feature extraction
│   ├── stokes.py
│   ├── cpr.py
│   ├── mchi.py
│   ├── thresholds.py
│   ├── feature_tensor.py
│   └── agent.py
├── agent4_segmentation/           # Weakly-supervised ice segmentation
│   ├── pseudo_labels.py
│   ├── dataset.py
│   ├── model.py
│   ├── loss.py
│   ├── train.py
│   ├── inference.py
│   └── agent.py
├── agent5_pathfinding/            # Terramechanic-aware pathfinding
│   ├── rover_config.py
│   ├── terramechanics.py
│   ├── cost_function.py
│   ├── illumination.py
│   ├── heuristic.py
│   ├── kinodynamic_astar.py
│   ├── graph_builder.py
│   └── agent.py
├── notebooks/                     # Colab notebooks (per-agent + full pipeline)
│   ├── 01_data_ingestion.ipynb
│   ├── 02_despeckling_training.ipynb
│   ├── 03_polarimetry.ipynb
│   ├── 04_segmentation_training.ipynb
│   ├── 05_pathfinding.ipynb
│   └── 06_full_pipeline.ipynb
├── tests/                         # Unit tests (58 tests, all passing)
│   ├── test_horn_slope.py
│   ├── test_stokes.py
│   ├── test_mchi.py
│   ├── test_terramechanics.py
│   └── test_astar.py
├── orchestrator.py                # LangGraph state graph with retry loops
├── main.py                        # CLI entry point
├── requirements.txt
└── requirements_colab.txt
```

---

## Key Algorithms

### Polarimetric Ice Detection

- **CPR > 1**: Volume scattering signature consistent with buried ice
- **DOP < 0.13**: Low degree of polarization indicates depolarizing medium
- **m-χ decomposition**: Separates surface, volume, and double-bounce scattering

### Terramechanics

- **Mohr-Coulomb**: τ = c + σ·tan(φ) for regolith shear strength
- **Janosi-Hanamoto**: τ(j) = τ_max·(1 - e^(-j/K)) for wheel-soil interaction
- **Slip-aware traversal**: Cost = f(distance, slope, slip_ratio, energy)

### Pathfinding

- **4D A* search**: State = (x, y, heading, time)
- **Constraints**: Battery capacity, thermal limits, slope limits
- **Heuristic**: max(Euclidean, battery-aware, thermal-aware) — admissible

---

## Configuration

Edit `config/mission_config.yaml`:

```yaml
target:
  crater_name: "Shackleton"
  center_lat: -89.54
  center_lon: 0.0

thresholds:
  ice_cpr_min: 1.0       # CPR threshold for ice candidates
  ice_dop_max: 0.13      # DOP threshold
  rock_m_min: 0.7        # DOP threshold for rock

rover:
  mass_kg: 27.0           # Pragyan-class
  max_slope_deg: 10.0     # Mechanical limit
  battery_capacity_wh: 50.0
```

---

## Tests

```bash
python -m pytest tests/ -v
```

All 58 tests pass covering:
- Horn's slope algorithm (10 tests)
- Stokes parameters (8 tests)
- m-χ decomposition & CPR (15 tests)
- Terramechanics (12 tests)
- A* pathfinding (6 tests)
- Edge cases: NaN handling, zero denominators, physical constraint violations

---

## References

1. Raney, R.K. et al. (2012). "The m-chi decomposition of hybrid dual-pol SAR data." *IEEE GRSL*.
2. Sinha, R.K. et al. (2024). "Ice detection in permanently shadowed regions using Chandrayaan-2 DFSAR."
3. Bekker, M.G. (1969). "Introduction to Terrain-Vehicle Systems."
4. Wong, J.Y. (2008). "Theory of Ground Vehicles."

---

## License

MIT
