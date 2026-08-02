# LunarSight — Detailed Implementation Plan

## Confirmed Parameters

| Parameter | Decision |
|---|---|
| Data | Real DFSAR data (user has access) |
| Colab Tier | **Free** — T4 GPU, 12hr sessions, aggressive checkpointing |
| Rover | Generic configurable params (Pragyan-class defaults) |
| Crater | **Configurable** — Faustini as default, any crater supported |
| Priority | Build for real data (no synthetic generator) |

---

## Project Structure

```
Lunar-Sight/
├── config/
│   ├── mission_config.yaml          # Crater coords, rover params, thresholds
│   └── training_config.yaml         # Hyperparams, batch sizes, epochs
│
├── shared/
│   ├── __init__.py
│   ├── state.py                     # Global StateGraph schema (TypedDict)
│   ├── io_utils.py                  # Tensor save/load, Drive mount helpers
│   ├── geo_utils.py                 # CRS transforms, bounding box utilities
│   ├── constants.py                 # Lunar radius, regolith params, physics constants
│   └── visualization.py            # Shared plotting: DEM, CPR maps, paths
│
├── agent1_ingestion/
│   ├── __init__.py
│   ├── pds4_parser.py               # PDS4 XML label reader
│   ├── data_fetcher.py              # PRADAN + NASA PDS download client
│   ├── reprojection.py              # GDAL/Rasterio polar stereographic warp
│   ├── horn_slope.py                # Horn's algorithm (slope + aspect)
│   ├── tensor_builder.py            # Multi-channel co-registered tensor output
│   └── agent.py                     # Agent 1 node function (LangGraph compatible)
│
├── agent2_despeckling/
│   ├── __init__.py
│   ├── covariance.py                # C₂/C₃ matrix computation from raw complex data
│   ├── complex_layers.py            # Complex Conv2d, ComplexBatchNorm, ComplexReLU
│   ├── cv_cnn_model.py              # CV-CNN autoencoder architecture
│   ├── loss.py                      # Complex Frobenius loss + phase coherence loss
│   ├── dataset.py                   # Patch-based dataset with complex tensor I/O
│   ├── train.py                     # Training loop with checkpointing
│   ├── inference.py                 # Run despeckle on full scene
│   └── agent.py                     # Agent 2 node function
│
├── agent3_polarimetry/
│   ├── __init__.py
│   ├── stokes.py                    # S₁–S₄ computation from covariance
│   ├── cpr.py                       # Circular Polarization Ratio
│   ├── mchi.py                      # m-χ decomposition (m, χ, RGB components)
│   ├── thresholds.py                # Sinha et al. diagnostic flags (configurable)
│   ├── feature_tensor.py            # Stack into Polarimetric Feature Tensor
│   └── agent.py                     # Agent 3 node function
│
├── agent4_segmentation/
│   ├── __init__.py
│   ├── pseudo_labels.py             # Physics-based seed generator
│   ├── dataset.py                   # Multi-channel dataset with NaN masking
│   ├── model.py                     # SMP U-Net wrapper with custom input channels
│   ├── loss.py                      # NaN-masked cross-entropy + optional CRF
│   ├── train.py                     # Training loop with self-training iterations
│   ├── inference.py                 # Full-scene inference → Binary Ice Mask
│   └── agent.py                     # Agent 4 node function
│
├── agent5_pathfinding/
│   ├── __init__.py
│   ├── terramechanics.py            # Mohr-Coulomb + Janosi-Hanamoto equations
│   ├── rover_config.py              # Configurable rover parameters class
│   ├── cost_function.py             # Slip-aware g(n) cost computation
│   ├── illumination.py              # Solar vector / shadow model (708h cycle)
│   ├── heuristic.py                 # h(n) with battery + thermal depletion
│   ├── kinodynamic_astar.py         # 4D A* search (x, y, θ, t)
│   ├── graph_builder.py             # DEM → searchable graph discretization
│   └── agent.py                     # Agent 5 node function
│
├── orchestrator/
│   ├── __init__.py
│   ├── supervisor.py                # Supervisor routing logic
│   ├── state_graph.py               # LangGraph StateGraph definition
│   ├── conflict_resolution.py       # Cyclical retry logic (Agent 5 → Agent 4)
│   └── runner.py                    # CLI entry point to run full pipeline
│
├── notebooks/                       # Google Colab notebooks
│   ├── 01_data_ingestion.ipynb
│   ├── 02_despeckling_training.ipynb
│   ├── 03_polarimetry.ipynb
│   ├── 04_segmentation_training.ipynb
│   ├── 05_pathfinding.ipynb
│   └── 06_full_pipeline.ipynb
│
├── tests/
│   ├── test_horn_slope.py
│   ├── test_stokes.py
│   ├── test_mchi.py
│   ├── test_terramechanics.py
│   └── test_astar.py
│
├── requirements.txt
├── requirements_colab.txt           # Colab-specific deps (lighter, no GDAL build)
├── README.md
└── LICENSE
```

---

## Agent-by-Agent Implementation Plans

---

### Agent 1 — Data Ingestion & Topographic Synthesis

#### Purpose
Download + co-register DFSAR radar + LOLA DEM → unified tensor with slope/aspect.

#### Modules

##### `pds4_parser.py`
- Parse PDS4 `.xml` label files to extract:
  - Array dimensions, data type, byte order
  - Polarization channel mapping (HH, HV, VH, VV)
  - Acquisition geometry metadata
  - Calibration/scaling factors
- Return a structured `ProductMetadata` dataclass

##### `data_fetcher.py`
- `DFSARFetcher` class: authenticated download from PRADAN
  - Accept configurable crater bounding box from `mission_config.yaml`
  - Download L-band + S-band `.img` complex arrays
  - Verify checksums, handle partial downloads
- `LOLAFetcher` class: download from NASA PDS Geosciences Node
  - Fetch DEM tiles covering the bounding box
  - Support both 118m global DEM and higher-res SLDEM2015
- Both fetchers return local file paths

##### `reprojection.py`
- Reproject all rasters into **Lunar Polar Stereographic CRS**
  - Central body: Moon (R = 1737.4 km)
  - Standard parallel: 90°S for south pole targets
- Use GDAL `Warp` with configurable resampling (bilinear default)
- Ensure pixel alignment: same resolution, same grid origin
- Output: co-registered `.tif` files

##### `horn_slope.py`
- Vectorized NumPy implementation of Horn's 3×3 algorithm
- Input: 2D DEM array + pixel size (Δx, Δy in meters)
- Compute:
  - East-west gradient: `dz_dx` using weighted kernel
  - North-south gradient: `dz_dy` using weighted kernel
  - Slope angle: `θ = arctan(√(dz_dx² + dz_dy²))` in degrees
  - Aspect: `arctan2(dz_dy, dz_dx)` converted to compass bearing
- Handle edges with numpy padding (reflect mode)
- Output: slope array + aspect array (same shape as DEM)

##### `tensor_builder.py`
- Stack into multi-channel tensor:
  - Channel 0-1: L-band complex (real, imag)
  - Channel 2-3: S-band complex (real, imag)
  - Channel 4: LOLA DEM elevation
  - Channel 5: Horn's slope (degrees)
  - Channel 6: Horn's aspect (degrees)
- Save as `.npy` or `.h5` with metadata dict
- Write file path to state

##### `agent.py`
- LangGraph-compatible node function signature:
  - Input: `state` dict with `target_bbox`, `mission_config`
  - Execute: fetch → reproject → slope → tensor
  - Output: update state with `raw_tensor_path`, `agent1_status`

#### Testing
- `test_horn_slope.py`: Unit test against known slope values (flat plane → 0°, 45° ramp → 45°)
- Validate reprojection preserves spatial extent

---

### Agent 2 — Complex-Valued SAR Despeckling

#### Purpose
Remove speckle noise from complex radar data while preserving polarimetric phase.

#### Modules

##### `covariance.py`
- Compute covariance matrices from raw complex scattering vectors
- **Dual-pol (C₂)**: 2×2 matrix per pixel
  ```
  C₂[i,j] = [ ⟨|E_HH|²⟩   ⟨E_HH·E_HV*⟩ ]
             [ ⟨E_HV·E_HH*⟩  ⟨|E_HV|²⟩   ]
  ```
- **Quad-pol (C₃)**: 3×3 matrix per pixel (if available)
- Use spatial averaging window (e.g., 5×5) for expectation ⟨·⟩
- Output: complex-valued covariance tensor

##### `complex_layers.py`
- Custom PyTorch modules operating on `torch.complex64`:
  - `ComplexConv2d`: convolves real and imaginary parts following complex multiplication rules
  - `ComplexBatchNorm`: normalize real and imaginary independently
  - `ComplexReLU`: apply ReLU to magnitude, preserve phase
  - `ComplexDropout`: dropout with complex masking
- Leverage ComplexPyTorch library where possible, custom-build where needed

##### `cv_cnn_model.py`
- **Architecture: Complex Denoising Autoencoder**
  - Encoder: 4-5 complex conv blocks (ComplexConv2d → ComplexBN → ComplexReLU → pool)
  - Bottleneck: complex dense features
  - Decoder: 4-5 complex transpose conv blocks with skip connections
  - Output: same shape as input (clean covariance matrix)
- Input channels: depends on C₂ (4 real values) or C₃ (9 real values)
- All operations in complex domain

##### `loss.py`
- **Complex Frobenius Loss**: `‖C_clean - C_pred‖_F` in complex domain
- **Phase Coherence Loss**: penalize degradation of off-diagonal phase angles
  - `L_phase = 1 - cos(∠C_clean_offdiag - ∠C_pred_offdiag)`
- Combined: `L = α·L_frobenius + β·L_phase`

##### `dataset.py`
- Patch-based loading from co-registered tensor
- Extract overlapping tiles (e.g., 128×128 with 32px stride)
- Convert to `torch.complex64` tensors
- Data augmentation: random rotations (90° increments), flips

##### `train.py`
- Training loop with **Colab Free tier optimizations**:
  - **Mixed precision** where compatible with complex ops
  - **Batch size**: start at 8, adjust based on T4 16GB VRAM
  - **Checkpoint every 5 epochs** to Google Drive
  - **Resume from checkpoint** if session disconnects
  - **Early stopping** on validation loss
  - **tqdm progress bars** for Colab display
  - Epoch time estimate + session time tracking
- Log metrics to a simple CSV on Drive

##### `inference.py`
- Load trained weights
- Tile-based inference on full scene (avoids OOM)
  - Overlapping tiles with Hann window blending
- Output: despeckled covariance tensor (same format as input)
- Phase integrity check: verify `∠(C_out_offdiag)` ≈ `∠(C_in_offdiag)` within tolerance

#### Colab Free Tier Training Strategy

> [!WARNING]
> **Colab Free Tier Constraints:**
> - T4 GPU (16 GB VRAM)
> - 12-hour session limit (often disconnects earlier)
> - No background execution
> - Must checkpoint aggressively

| Strategy | Implementation |
|---|---|
| Checkpointing | Save model + optimizer + epoch + best_loss every 5 epochs to `/content/drive/MyDrive/LunarSight/checkpoints/` |
| Resume | On notebook restart, auto-detect latest checkpoint and resume |
| Batch size | 8 (128×128 patches) — fits in 16GB VRAM with complex64 |
| Training time | ~20-30 min per epoch, plan for 50-100 epochs across multiple sessions |
| Monitoring | Print loss every batch, save loss curve plot to Drive every epoch |

---

### Agent 3 — Polarimetric Feature Extraction

#### Purpose
Convert despeckled covariance matrices into physically interpretable features.

#### Modules

##### `stokes.py`
- Compute Stokes parameters from C₂ matrix (circular TX basis):
  - `S₁ = ⟨|E_H|²⟩ + ⟨|E_V|²⟩` (total power)
  - `S₂ = ⟨|E_H|²⟩ - ⟨|E_V|²⟩`
  - `S₃ = 2·Re(⟨E_H·E_V*⟩)`
  - `S₄ = -2·Im(⟨E_H·E_V*⟩)`
- All vectorized NumPy, pixel-wise
- Output: 4 arrays (S₁, S₂, S₃, S₄)

##### `cpr.py`
- Circular Polarization Ratio: `CPR = (S₁ - S₄) / (S₁ + S₄)`
- Compute for both L-band and S-band separately
- Handle division-by-zero (mask where S₁ + S₄ ≈ 0)
- Output: L_CPR array, S_CPR array

##### `mchi.py`
- Degree of Polarization: `m = √(S₂² + S₃² + S₄²) / S₁`
- Poincaré ellipticity: `sin(2χ) = -S₄ / (m·S₁)`
- RGB scattering decomposition:
  - `R = √(m·S₁·(1 + sin2χ) / 2)` — even/double-bounce
  - `G = √(S₁·(1 - m))` — volumetric
  - `B = √(m·S₁·(1 - sin2χ) / 2)` — odd/surface
- Handle NaN/Inf from edge cases (zero power pixels, PSR noise floor)
- Output: m array, chi array, R/G/B arrays

##### `thresholds.py`
- **Configurable** diagnostic flags (loaded from `mission_config.yaml`):
  ```yaml
  thresholds:
    ice_cpr_min: 1.0        # CPR > 1
    ice_dop_max: 0.13       # DOP < 0.13
    rock_m_min: 0.7         # high m → surface/double-bounce
  ```
- Apply pixel-wise Boolean logic:
  - Ice candidate: `L_CPR > ice_cpr_min AND DOP < ice_dop_max`
  - Rock candidate: `m > rock_m_min AND dominant_odd_bounce`
- Output: binary flag arrays

##### `feature_tensor.py`
- Stack all computed layers into Polarimetric Feature Tensor:
  - Channels: L_CPR, S_CPR, m, R, G, B, ice_flag, rock_flag
  - + optional: raw S₁ (total power) for each band
- Save as `.npy` with metadata JSON sidecar
- Total: 8-12 channels depending on configuration

#### Testing
- `test_stokes.py`: verify against analytically known polarization states
- `test_mchi.py`: test with synthetic covariance for pure surface/volume/dihedral

---

### Agent 4 — Weakly-Supervised Semantic Segmentation

#### Purpose
Train U-Net to classify ice vs rock using sparse physics-based pseudo-labels.

#### Modules

##### `pseudo_labels.py`
- Input: Polarimetric Feature Tensor from Agent 3
- Generate sparse label map:
  - `1` (ICE) where: L-band CPR > 1 AND S-band CPR > 1 AND DOP < 0.13
  - `0` (ROCK) where: sunlit equatorial pixels with high odd-bounce
  - `NaN` (UNLABELED) everywhere else
- Report statistics: % ice seeds, % rock seeds, % unlabeled
- Save label map alongside feature tensor

##### `dataset.py`
- `LunarSegDataset(torch.utils.data.Dataset)`:
  - Load feature tensor + DEM/slope as input (N channels)
  - Load pseudo-label map as target
  - Tile into patches (e.g., 256×256) with configurable stride
  - Return: `(input_tensor, label_tensor, valid_mask)`
    - `valid_mask`: True where label is 0 or 1, False where NaN
- Augmentations: random flips, 90° rotations

##### `model.py`
- Wrapper around SMP library:
  ```
  Model = smp.Unet(
      encoder_name="efficientnet-b3",    # or resnet34
      encoder_weights="imagenet",
      in_channels=N,                     # matches feature tensor channels + topo
      classes=2,                         # ice vs rock (or 1 for binary)
      activation=None                    # raw logits
  )
  ```
- Custom first conv layer to accept N>3 input channels
  - Strategy: initialize first 3 channels from ImageNet weights, rest with Kaiming init
- Output: per-pixel logits

##### `loss.py`
- **NaN-masked Binary Cross-Entropy**:
  - Compute BCE only on pixels where `valid_mask == True`
  - Ignore all NaN/unlabeled pixels
  - Weighted: balance ice vs rock class frequency in seeds
- Optional **Dice Loss** component for boundary quality
- Combined: `L = α·BCE_masked + β·Dice_masked`

##### `train.py`
- **Colab Free optimized training loop**:
  - Batch size: 4-8 (256×256 patches, T4 16GB)
  - Optimizer: AdamW, lr=1e-4 with cosine annealing
  - Epochs: 50-100 (across multiple Colab sessions)
  - **Self-training iteration** (every 20 epochs):
    1. Run inference on unlabeled pixels
    2. Add high-confidence predictions (>0.95) as new pseudo-labels
    3. Retrain with expanded label set
  - Checkpoint: model + optimizer + scheduler + epoch + best metrics → Drive
  - Resume: auto-detect and load latest checkpoint

##### `inference.py`
- Load trained model
- Tile-based inference (avoid OOM on full scene)
- Output:
  - **Binary Ice Mask**: argmax of softmax (0=rock, 1=ice)
  - **Confidence Map**: max softmax probability per pixel
- Save both as georeferenced `.tif` (inheriting CRS from Agent 1)

#### Colab Free Tier Training Strategy

| Strategy | Implementation |
|---|---|
| Checkpointing | Every 5 epochs + after each self-training cycle |
| Batch size | 4 (256×256 multi-channel patches) — ~12GB VRAM usage |
| Self-training | 3 rounds of pseudo-label expansion (epochs 20, 40, 60) |
| Total training | ~50-80 epochs, split across 2-3 Colab sessions |
| Session management | Print estimated remaining time, warn at 10hr mark |

---

### Agent 5 — Terramechanic-Aware Spatiotemporal Pathfinding

#### Purpose
Plan physically survivable rover traverse from crater rim to ice deposit.

#### Modules

##### `rover_config.py`
- Configurable dataclass loaded from `mission_config.yaml`:
  ```yaml
  rover:
    mass_kg: 27.0                    # Pragyan-class default
    wheel_count: 6
    wheel_radius_m: 0.105
    wheel_width_m: 0.14
    max_slope_deg: 10.0
    battery_capacity_wh: 50.0
    power_draw_nominal_w: 15.0
    power_draw_max_w: 50.0
    thermal_min_temp_k: 173.0        # min operating temp
    max_velocity_ms: 0.01            # ~1 cm/s
    suspension_type: "rocker_bogie"  # or "active"
  ```
- All values configurable, Pragyan-like defaults

##### `terramechanics.py`
- **Lunar regolith parameters** (from Apollo data):
  ```yaml
  regolith:
    cohesion_pa: 170.0               # c (Pascals)
    friction_angle_deg: 33.0         # φ
    shear_deformation_modulus_m: 0.018  # K
    bulk_density_kgm3: 1500.0
  ```
- `mohr_coulomb(sigma, c, phi)` → τ_max (max shear stress)
- `janosi_hanamoto(sigma, c, phi, j, K)` → τ(j) (shear at displacement j)
- `compute_slip_ratio(slope_deg, rover_config, regolith)` → predicted slip %
- `compute_drawbar_pull(slope_deg, rover_config, regolith)` → force (N)
- `is_traversable(slope_deg, rover_config, regolith)` → Boolean

##### `cost_function.py`
- **Slip-aware edge cost** `g(n)`:
  - Base cost: Euclidean distance between nodes
  - Slip penalty: multiply by `1 / (1 - slip_ratio)` — higher slip = higher cost
  - Slope penalty: exponential near max traversable angle
  - **Infinite cost** if `is_traversable() == False`
- Energy cost component: power draw increases with slip

##### `illumination.py`
- Simple shadow model for south polar regions:
  - Input: DEM + sun elevation angle (varies with time)
  - Ray-casting or simplified horizon model
  - Output: illumination map at time `t`
- Sun position computed from lunar synodic period (~708 hrs)
- Configurable: can accept external illumination model or use built-in approximation

##### `heuristic.py`
- **A* heuristic** `h(n)`:
  - Euclidean distance to target (admissible)
  - Battery depletion estimate based on remaining distance + average slip
  - Thermal budget: time in shadow × cooling rate vs. thermal_min_temp
  - **Return trip awareness**: h(n) includes cost of returning to illuminated rim
- Must remain admissible (never overestimate) for A* optimality

##### `kinodynamic_astar.py`
- **4D Search Space**: (x_pixel, y_pixel, heading_θ, time_t)
- State: `(x, y, θ, t, battery_remaining, temperature)`
- Node expansion:
  1. For each 8-connected neighbor (or 16 for smoother paths):
     - Compute terrain cost via `cost_function.py`
     - Update time based on distance / (velocity × (1-slip))
     - Update battery: `battery -= power_draw × Δt`
     - Update temperature based on illumination at (x,y,t)
  2. Prune nodes where:
     - `battery_remaining < h_return_cost` (can't get back)
     - `temperature < thermal_min` (too cold)
     - `is_traversable == False` (soil failure)
- Open/closed sets using priority queue (heapq)
- Output: ordered list of `(x, y, θ, t, battery, slip_ratio)` waypoints

##### `graph_builder.py`
- Convert DEM + slope arrays into graph nodes
- Each pixel = one node with attributes: elevation, slope, aspect
- Adjacency: 8-connected (or 16 for diagonal smoothness)
- Pre-filter: mark permanently impassable nodes (slope > absolute max)
- Configurable resolution: can subsample DEM for faster initial planning

#### Testing
- `test_terramechanics.py`:
  - Verify Mohr-Coulomb against known failure cases
  - Verify Janosi-Hanamoto converges to τ_max at large displacement
  - Verify flat terrain has zero slip, steep terrain approaches 100%
- `test_astar.py`:
  - Simple grid with known optimal path
  - Grid with impassable zones → verify avoidance

---

### Orchestrator — LangGraph StateGraph

#### Modules

##### `state.py` (in `shared/`)
- Central state schema:
  ```
  LunarSightState(TypedDict):
      # Mission config
      target_bbox: tuple[float, float, float, float]  # lat_min, lat_max, lon_min, lon_max
      rover_config_path: str
      
      # Agent 1 outputs
      raw_tensor_path: str
      dem_path: str
      slope_path: str
      agent1_status: str  # "pending" | "success" | "error"
      
      # Agent 2 outputs
      despeckled_tensor_path: str
      agent2_status: str
      
      # Agent 3 outputs
      polarimetric_tensor_path: str
      agent3_status: str
      
      # Agent 4 outputs
      ice_mask_path: str
      confidence_map_path: str
      agent4_status: str
      agent4_confidence_threshold: float  # adjustable by supervisor
      
      # Agent 5 outputs
      traverse_path: list[dict]  # waypoints
      path_failure: bool
      agent5_status: str
      
      # System
      retry_count: int
      max_retries: int
      error_log: list[str]
  ```

##### `supervisor.py`
- Lightweight routing logic (NOT an LLM — deterministic):
  - Read state → determine which agent to trigger next
  - Decision tree:
    1. If `agent1_status != success` → run Agent 1
    2. If `agent2_status != success` → run Agent 2
    3. If `agent3_status != success` → run Agent 3
    4. If `agent4_status != success` → run Agent 4
    5. If `agent5_status != success` → run Agent 5
    6. If `path_failure == True AND retry_count < max_retries`:
       - Lower `agent4_confidence_threshold` by 0.1
       - Reset `agent4_status` and `agent5_status`
       - Route back to Agent 4 (cyclical resolution)
    7. If all success → END

##### `state_graph.py`
- Build LangGraph `StateGraph`:
  ```
  graph = StateGraph(LunarSightState)
  graph.add_node("agent1", agent1_node)
  graph.add_node("agent2", agent2_node)
  graph.add_node("agent3", agent3_node)
  graph.add_node("agent4", agent4_node)
  graph.add_node("agent5", agent5_node)
  graph.add_node("supervisor", supervisor_node)
  
  # Forward edges
  graph.add_edge("agent1", "supervisor")
  graph.add_edge("agent2", "supervisor")
  graph.add_edge("agent3", "supervisor")
  graph.add_edge("agent4", "supervisor")
  graph.add_edge("agent5", "supervisor")
  
  # Conditional routing from supervisor
  graph.add_conditional_edges("supervisor", route_decision)
  
  graph.set_entry_point("supervisor")
  ```

##### `conflict_resolution.py`
- Handles cyclical retry scenarios:
  - **Scenario 1**: Agent 5 path_failure → lower confidence → retry Agent 4+5
  - **Scenario 2**: Agent 5 path_failure after 3 retries → select different ice deposit
  - **Scenario 3**: Global failure → report to user with diagnostic state dump
- Max retry depth: configurable (default 3)

##### `runner.py`
- CLI entry point:
  ```
  python -m orchestrator.runner --config config/mission_config.yaml
  ```
- Initializes state from config, compiles graph, runs to completion
- Prints progress, timing, and final results

---

## Google Colab Notebook Plans

### Design Principles for Colab Free Tier

> [!CAUTION]
> **Colab Free disconnects without warning.** Every notebook MUST:
> 1. Mount Google Drive at the very top
> 2. Define a `CHECKPOINT_DIR` on Drive
> 3. Auto-resume from latest checkpoint on re-run
> 4. Save intermediate outputs (not just final) to Drive
> 5. Print session time elapsed and warn at 10-hour mark

### Common Notebook Header (all notebooks share this)
```
Cell 1: Mount Drive, clone/pull repo, install deps
Cell 2: Load configs, set paths, detect GPU
Cell 3: Check for existing checkpoints → resume or fresh start
```

---

### Notebook 01 — Data Ingestion (CPU, ~30 min)
| Cell | Action |
|---|---|
| 1 | Mount Drive, install GDAL + Rasterio |
| 2 | Upload DFSAR data from local or load from Drive |
| 3 | Parse PDS4 labels, validate data integrity |
| 4 | Reproject to polar stereographic CRS |
| 5 | Compute Horn's slope + aspect |
| 6 | Build + save co-registered tensor to Drive |
| 7 | Visualize: DEM, slope map, radar amplitude |

---

### Notebook 02 — Despeckling Training (GPU, ~3-6 hrs across sessions)
| Cell | Action |
|---|---|
| 1 | Mount Drive, install PyTorch + ComplexPyTorch |
| 2 | Load co-registered tensor, compute covariance matrices |
| 3 | Create patch dataset (128×128, stride 64) |
| 4 | Initialize CV-CNN model (or resume from checkpoint) |
| 5 | **Training loop** — checkpoint every 5 epochs to Drive |
| 6 | Plot training/validation loss curves |
| 7 | Run inference on full scene |
| 8 | Save despeckled tensor to Drive |
| 9 | Visualize: before/after comparison, phase integrity check |

**Session strategy**: Aim for 15-20 epochs per session. 3-4 sessions to reach convergence.

---

### Notebook 03 — Polarimetry (CPU, ~15 min)
| Cell | Action |
|---|---|
| 1 | Mount Drive, load despeckled tensor |
| 2 | Compute Stokes parameters |
| 3 | Compute CPR (L-band + S-band) |
| 4 | Compute m-χ decomposition + RGB channels |
| 5 | Apply Sinha et al. thresholds |
| 6 | Build + save Polarimetric Feature Tensor |
| 7 | Visualize: CPR map, m-χ RGB composite, threshold overlay |

---

### Notebook 04 — Segmentation Training (GPU, ~4-8 hrs across sessions)
| Cell | Action |
|---|---|
| 1 | Mount Drive, install SMP |
| 2 | Load Polarimetric Feature Tensor + DEM/slope |
| 3 | Generate pseudo-labels, print seed statistics |
| 4 | Create dataset (256×256 patches) |
| 5 | Initialize U-Net (or resume from checkpoint) |
| 6 | **Training loop** — checkpoint every 5 epochs |
| 7 | **Self-training round 1** (epoch 20): expand labels, continue |
| 8 | **Self-training round 2** (epoch 40): expand again, continue |
| 9 | Run full-scene inference → Binary Ice Mask + confidence |
| 10 | Save masks to Drive |
| 11 | Visualize: ice map overlaid on DEM, confidence heatmap |

**Session strategy**: ~20 epochs per session. Self-training rounds are natural session boundaries.

---

### Notebook 05 — Pathfinding (CPU, ~30 min)
| Cell | Action |
|---|---|
| 1 | Mount Drive, load Ice Mask + DEM + slope |
| 2 | Configure rover parameters |
| 3 | Build terrain graph, pre-filter impassable zones |
| 4 | Visualize cost map (traversability heatmap) |
| 5 | Select ice target (highest confidence reachable deposit) |
| 6 | Run kinodynamic A* pathfinding |
| 7 | Visualize: optimal path on DEM with slip ratio coloring |
| 8 | Output: waypoint table (lat, lon, time, battery, slip) |

---

### Notebook 06 — Full Pipeline Integration (CPU, ~10 min)
| Cell | Action |
|---|---|
| 1 | Mount Drive, load all pre-trained weights |
| 2 | Initialize LangGraph StateGraph |
| 3 | Set mission config (crater coordinates, rover params) |
| 4 | Run full pipeline with live state visualization |
| 5 | Demonstrate cyclical conflict resolution (force a failure) |
| 6 | Display final output: ice map + safe traverse |
| 7 | Export mission package (all maps, paths, metadata) |

---

## Configuration System

### `mission_config.yaml` (example)
```yaml
target:
  crater_name: "Faustini"
  bbox:
    lat_min: -88.0
    lat_max: -86.5
    lon_min: 76.0
    lon_max: 78.0
  # Easy to swap to another crater:
  # crater_name: "Shoemaker"
  # bbox: { lat_min: -89.0, lat_max: -87.5, lon_min: 40.0, lon_max: 42.0 }

thresholds:
  ice_cpr_min: 1.0
  ice_dop_max: 0.13
  confidence_initial: 0.5
  confidence_step_down: 0.1

rover:
  mass_kg: 27.0
  wheel_count: 6
  wheel_radius_m: 0.105
  max_slope_deg: 10.0
  battery_capacity_wh: 50.0
  thermal_min_temp_k: 173.0

orchestrator:
  max_retries: 3
```

---

## Development Phases & Order

### Phase A — Foundation (I build locally, no Colab needed)
1. `shared/` — state schema, constants, utilities
2. `config/` — YAML configs with defaults
3. `agent1_ingestion/` — complete (all modules)
4. `agent3_polarimetry/` — complete (all modules)
5. Unit tests for Horn's algorithm, Stokes, m-χ

### Phase B — ML Architectures (I build locally, you train on Colab)
6. `agent2_despeckling/` — CV-CNN architecture + training loop
7. `agent4_segmentation/` — U-Net wrapper + pseudo-labels + training loop
8. Both with full checkpoint/resume support

### Phase C — Pathfinding (I build locally)
9. `agent5_pathfinding/` — terramechanics + A* (complete)
10. Unit tests for soil mechanics + pathfinding

### Phase D — Orchestration (I build locally)
11. `orchestrator/` — LangGraph StateGraph + supervisor + conflict resolution
12. Integration test: run full pipeline with dummy data

### Phase E — Colab Notebooks (I build locally)
13. All 6 notebooks with Colab Free optimizations
14. Drive mount, checkpoint logic, session management
15. Visualization cells for each stage

### Phase F — Polish
16. README.md with full setup instructions
17. `requirements.txt` + `requirements_colab.txt`
18. Documentation for each agent's API

---

## Verification Plan

### Automated Tests
- `pytest tests/` — unit tests for all deterministic modules
- Horn's slope verified against known geometry
- Stokes/CPR/m-χ verified against analytical cases
- Terramechanics verified against published soil parameters
- A* verified on simple grids with known solutions

### Colab Verification
- Each notebook runs end-to-end on Colab Free
- Checkpoint/resume tested by intentional kernel restart
- GPU memory usage profiled (must stay under 15GB for T4)

### Integration Verification
- Full pipeline runs with real DFSAR data
- Cyclical conflict resolution demonstrated
- Output paths are physically plausible (no traversal of impassable terrain)

---

## Dependencies

### `requirements.txt` (local development)
```
numpy>=1.24
scipy>=1.11
rasterio>=1.3
gdal>=3.6
pyyaml>=6.0
torch>=2.0
segmentation-models-pytorch>=0.3
langgraph>=0.1
matplotlib>=3.7
h5py>=3.9
tqdm>=4.65
```

### `requirements_colab.txt` (lighter, Colab-specific)
```
segmentation-models-pytorch>=0.3
complexpytorch>=0.4
langgraph>=0.1
pyyaml>=6.0
h5py>=3.9
```
*(PyTorch, NumPy, SciPy, matplotlib already in Colab runtime)*

---

> [!IMPORTANT]
> **Ready to start building when you give the green light.** I'll follow the Phase A → F order above. All code will be written in the project at `c:\Users\VEDAN\Lunar-Sight\Lunar-Sight\`.
