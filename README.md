# Traveling Salesman Problem — Ant Colony Optimization

An implementation of three Ant Colony Optimization (ACO) variants applied to the Traveling Salesman Problem (TSP) on a real-world dataset of 225 Indian cities.

<p align="center">
  <img src="./assets/path_trace.gif"><br>
  <i>Optimal tour traced across 225 cities</i>
</p>

---

## Overview

This project implements and compares three ACO variants for solving the TSP:

| Variant | Pheromone Strategy | Behavior |
|---|---|---|
| **ACS** | Constant deposit per ant | Uniform exploration |
| **Elitist** | Bonus deposit on global best | Faster convergence toward quality solutions |
| **MaxMin** | Bounded pheromone, two-phase deposit | Balances exploration and exploitation |

---

## Dataset

225 cities located in the North-West and Central Indian subcontinent.

```
tsp dataset/
├── location_ll.txt          # Latitude/longitude of each city (tab-separated)
├── names.txt                # City names
├── distance.txt             # Euclidean distance matrix (225×225)
├── road_distance.txt        # Road distance matrix (225×225)
└── travel_time.txt          # Travel time matrix (225×225)
```

The algorithm uses `location_ll.txt` for geographic coordinates. Distances are computed as Euclidean distances over the scaled coordinate space.

---

## Requirements

```
Python >= 3.8
numpy
matplotlib
scikit-learn
tqdm
tkinter  (standard library, required for turtle graphics)
```

Install dependencies:

```bash
pip install numpy matplotlib scikit-learn tqdm
```

---

## Usage

### 1. Run the visualization on the full 225-city dataset

```bash
python show.py
```

Runs ACS with 15 ants for 50 iterations. Prints runtime and best distance to the console, then opens an interactive map showing the optimal tour traced over a geographic background.

**Console output:**
```
Runtime:  <seconds> s
Minimum distance:  <distance>
```

**Visualization:**
- Green dot — starting city
- Blue dots — intermediate cities
- Red dot — last city before returning to start
- Lines — optimal path edges

Click anywhere on the map window to close it.

---

### 2. Run the benchmark comparison (all three variants)

```bash
python aco_tsp.py
```

Runs 10 problem sizes (10 to 100 cities), 20 random trials each, across all three variants. Results are written to `out.csv`.

**Output file format (`out.csv`):**

```
Iteration,ACS_time,ACS_dist,Elitist_time,Elitist_dist,MaxMin_time,MaxMin_dist
1,0.0221,2423.48,0.0227,2423.48,0.0222,2423.48
...
```

Where `Iteration` corresponds to problem size = `Iteration × 10` cities.

---

### 3. Programmatic usage

```python
from aco_tsp import SolveTSPUsingACO

nodes = [(lat1, lon1), (lat2, lon2), ..., (latN, lonN)]

model = SolveTSPUsingACO(
    mode='ACS',          # 'ACS', 'Elitist', or 'MaxMin'
    colony_size=15,
    steps=50,
    nodes=nodes
)

runtime, best_distance = model.run()
model.plot(save=True, name='tour.png')
```

---

## Parameters Reference

### `SolveTSPUsingACO` constructor

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mode` | `str` | `'ACS'` | Algorithm variant: `'ACS'`, `'Elitist'`, or `'MaxMin'` |
| `colony_size` | `int` | `10` | Number of ants per iteration |
| `steps` | `int` | `100` | Number of iterations |
| `nodes` | `list` | — | **Required.** List of `(x, y)` or `(lat, lon)` tuples |
| `labels` | `list` | `None` | Node labels for plot annotations (defaults to `1..N`) |
| `alpha` | `float` | `1.0` | Pheromone influence exponent |
| `beta` | `float` | `3.0` | Heuristic (distance) influence exponent |
| `rho` | `float` | `0.1` | Pheromone evaporation rate per iteration (0–1) |
| `pheromone_deposit_weight` | `float` | `1.0` | Scalar multiplier on pheromone deposits |
| `initial_pheromone` | `float` | `1.0` | Starting pheromone level on all edges |
| `elitist_weight` | `float` | `1.0` | *(Elitist only)* Extra deposit multiplier for global best tour |
| `min_scaling_factor` | `float` | `0.001` | *(MaxMin only)* `min_pheromone = max_pheromone × min_scaling_factor` |

### Tuning guidance

- **`alpha` / `beta`**: Higher `alpha` increases exploitation of learned pheromone trails. Higher `beta` biases ants toward shorter immediate edges (greedy). Typical starting point: `alpha=1`, `beta=3`.
- **`rho`**: Higher values cause faster forgetting — useful when the search stagnates. Lower values preserve learned paths longer.
- **`colony_size`**: More ants improves solution quality at the cost of runtime. For 225 cities, 15–30 ants is a reasonable range.
- **`steps`**: Diminishing returns above ~200 iterations for this dataset size.

---

## `plot()` method

```python
model.plot(
    line_width=1,         # Tour edge line width
    point_radius=1.41,    # City marker radius
    annotation_size=8,    # Label font size
    dpi=120,              # Output image DPI
    save=True,            # Save to file if True, show interactively if False
    name=None             # Output filename; defaults to '<mode>_tour.png'
)
```

---

## Project Structure

```
TSP_ACO/
├── aco_tsp.py               # Core ACO implementation (all three variants)
├── show.py                  # Dataset loader, runner, and turtle visualization
├── result.csv               # Pre-computed benchmark results
├── assets/
│   ├── map.png              # Geographic background for visualization
│   └── path_trace.gif       # Animated example output
├── tsp dataset/
│   ├── location_ll.txt
│   ├── names.txt
│   ├── distance.txt
│   ├── road_distance.txt
│   └── travel_time.txt
└── references/
    ├── paper1.pdf
    ├── paper2.pdf
    └── paper3.pdf
```

---

## Results

Pre-computed benchmark results are available in `result.csv`. The benchmark covers problem sizes from 10 to 100 cities (20 random trials per size) using `colony_size=5` and `steps=50`.

Key observations:
- For small instances (10 cities), all three variants produce identical or near-identical results.
- MaxMin shows better distance quality on larger instances due to bounded pheromone exploration.
- Elitist converges faster on well-structured problem instances.

---

## References

Academic references are included in the `references/` directory.
