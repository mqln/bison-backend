# Bison Simulation Backend

High-performance Python backend for Alaska bison population dynamics simulation.

## Features

- FastAPI REST API
- NumPy/SciPy vectorized grid operations (~165ms per step on 1966×1966 grid)
- FFT-based migration using scipy.signal.fftconvolve
- Realistic population dynamics calibrated to Yellowstone data (5-7% annual growth)
- Adaptive release area based on local habitat quality

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Run server
python server.py
```

Server runs on http://localhost:3001

## API Endpoints

- `GET /api/health` - Health check
- `GET /api/grid-info` - Get grid dimensions
- `GET /api/initial-biomass` - Get biomass map for visualization
- `POST /api/start` - Start simulation with config
- `POST /api/step` - Advance simulation by N years
- `POST /api/stop` - Stop simulation
- `GET /api/status` - Get current status

## Deployment (Cloud Run)

```bash
# Build and deploy
gcloud run deploy bison-backend \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

## Configuration

Key parameters in `config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_growth_rate` | 10% | Calibrated to Yellowstone λ=1.07-1.08 |
| `starvation_threshold` | 20% | Food satisfaction below which population declines |
| `utilization_factor` | 50% | Fraction of digestible biomass harvestable |
| `diffusion_rate` | 15% | Annual population spread rate |

## Data

The `data/` folder contains `combined_digestible_biomass_1km.tif` - satellite-derived biomass data for Alaska processed with plant-type-specific digestibility factors:

- Forbs: 80% digestibility
- Graminoids: 50% digestibility
- Deciduous shrubs: 30% digestibility

## Architecture

```
server.py      - FastAPI server with endpoints
simulation.py  - Core simulation logic (NumPy/SciPy)
config.py      - Configuration dataclasses
data/          - GeoTIFF biomass data
```
