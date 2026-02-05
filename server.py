"""
FastAPI server for bison simulation.

High-performance backend using NumPy for vectorized grid operations.
"""

import os
import time
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
import rasterio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import SimulationConfig, InitializationConfig, VIS_OUTPUT_SIZE
from simulation import Simulation, SimulationState


# --- Data Loading ---

def load_geotiff(path: str) -> tuple[np.ndarray, float]:
    """Load GeoTIFF and return data array and cell size in km."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        # Get cell size from transform (assuming equal x/y resolution)
        cell_size_m = abs(src.transform[0])
        cell_size_km = cell_size_m / 1000.0

        # Handle nodata
        if src.nodata is not None:
            data[data == src.nodata] = 0.0

        # Ensure non-negative
        data = np.clip(data, 0, None)

    return data, cell_size_km


def downsample(data: np.ndarray, target_size: int, preserve_sum: bool = False) -> np.ndarray:
    """
    Downsample array to target size.

    Args:
        data: Input array
        target_size: Target dimension size
        preserve_sum: If True, preserve the sum (for population data).
                      If False, preserve the mean (for biomass/density data).
    """
    if data.shape[0] == target_size and data.shape[1] == target_size:
        return data

    h, w = data.shape

    # Use block-based downsampling for accuracy
    block_h = h // target_size
    block_w = w // target_size

    # Trim to exact multiple
    trimmed = data[:block_h * target_size, :block_w * target_size]

    # Reshape into blocks and aggregate
    reshaped = trimmed.reshape(target_size, block_h, target_size, block_w)

    if preserve_sum:
        # Sum the blocks (for population - total count matters)
        result = reshaped.sum(axis=(1, 3))
    else:
        # Average the blocks (for biomass/density - concentration matters)
        result = reshaped.mean(axis=(1, 3))

    return result.astype(np.float32)


# --- Global State ---

class AppState:
    """Application state container."""

    def __init__(self):
        self.biomass_data: Optional[np.ndarray] = None
        self.cell_size_km: float = 1.0
        self.simulation: Optional[Simulation] = None
        self.is_running: bool = False

    def load_data(self, geotiff_path: str):
        """Load biomass data from GeoTIFF."""
        print(f"Loading biomass data from: {geotiff_path}")
        start = time.perf_counter()

        self.biomass_data, self.cell_size_km = load_geotiff(geotiff_path)

        elapsed = time.perf_counter() - start
        print(f"Loaded {self.biomass_data.shape[0]}x{self.biomass_data.shape[1]} grid in {elapsed:.2f}s")
        print(f"Cell size: {self.cell_size_km:.3f} km")


app_state = AppState()


# --- Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data on startup."""
    # Find the GeoTIFF file - check multiple locations for dev vs production
    possible_paths = [
        # Cloud Run / Docker path
        Path("/app/data/combined_digestible_biomass_1km.tif"),
        # Local path relative to python_backend
        Path(__file__).parent / "data" / "combined_digestible_biomass_1km.tif",
        # Local dev path (parent/public/data)
        Path(__file__).parent.parent / "public" / "data" / "combined_digestible_biomass_1km.tif",
    ]

    geotiff_path = None
    for path in possible_paths:
        if path.exists():
            geotiff_path = path
            break

    if geotiff_path:
        app_state.load_data(str(geotiff_path))
    else:
        print(f"Warning: GeoTIFF not found in any of: {[str(p) for p in possible_paths]}")

    yield


# --- FastAPI App ---

app = FastAPI(
    title="Bison Simulation API",
    description="High-performance bison population dynamics simulation",
    version="2.0.0",
    lifespan=lifespan
)

# CORS for frontend - allow all origins for Cloud Run deployment
# In production, you could restrict this to your Firebase domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response Models ---

class StartRequest(BaseModel):
    """Request to start a new simulation."""
    config: Optional[dict] = None
    startCoordinates: Optional[dict] = None


class GridInfoResponse(BaseModel):
    """Grid information response."""
    width: int
    height: int
    cellSizeKm: float
    totalCells: int


class StateResponse(BaseModel):
    """Simulation state response (downsampled for visualization)."""
    sessionId: str
    year: int
    state: dict
    metadata: dict
    stepTimeMs: Optional[float] = None


# --- Endpoints ---

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "backend": "python",
        "dataLoaded": app_state.biomass_data is not None
    }


@app.get("/api/grid-info", response_model=GridInfoResponse)
async def get_grid_info():
    """Get information about the simulation grid."""
    if app_state.biomass_data is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    height, width = app_state.biomass_data.shape
    return GridInfoResponse(
        width=width,
        height=height,
        cellSizeKm=app_state.cell_size_km,
        totalCells=width * height
    )


@app.get("/api/initial-biomass")
async def get_initial_biomass():
    """Get initial biomass map (downsampled for visualization)."""
    if app_state.biomass_data is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    # Downsample for visualization
    downsampled = downsample(app_state.biomass_data, VIS_OUTPUT_SIZE)
    vis_cell_size = app_state.cell_size_km * (app_state.biomass_data.shape[1] / VIS_OUTPUT_SIZE)

    return {
        "biomass": downsampled.tolist(),
        "metadata": {
            "width": VIS_OUTPUT_SIZE,
            "height": VIS_OUTPUT_SIZE,
            "cellSizeKm": vis_cell_size,
            "fullResolution": {
                "width": app_state.biomass_data.shape[1],
                "height": app_state.biomass_data.shape[0],
                "cellSizeKm": app_state.cell_size_km
            }
        }
    }


@app.post("/api/start")
async def start_simulation(request: StartRequest):
    """Start a new simulation session."""
    if app_state.biomass_data is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    # Parse configuration
    config = SimulationConfig()

    if request.config:
        init_config = request.config.get("initialization", {})
        if "totalPopulation" in init_config:
            config.initialization.total_population = init_config["totalPopulation"]
        if "releaseRadiusCells" in init_config:
            config.initialization.release_radius_cells = init_config["releaseRadiusCells"]

    # Parse start coordinates
    start_location = None
    if request.startCoordinates:
        row = request.startCoordinates.get("row", 0)
        col = request.startCoordinates.get("col", 0)

        # Scale from visualization to full resolution
        # IMPORTANT: Must use the same block size as downsampling to avoid offset
        h, w = app_state.biomass_data.shape
        block_h = h // VIS_OUTPUT_SIZE
        block_w = w // VIS_OUTPUT_SIZE
        # Map vis coordinate to center of corresponding block in full resolution
        full_row = int(row * block_h + block_h // 2)
        full_col = int(col * block_w + block_w // 2)

        # Auto-scale release radius based on population AND local biomass quality
        # Calculate local carrying capacity to determine how large the release area needs to be
        from config import BiomassConfig, BisonConfig
        biomass_cfg = BiomassConfig()
        bison_cfg = BisonConfig()

        # Sample biomass in a region around start location
        sample_radius = 50
        r_min, r_max = max(0, full_row - sample_radius), min(h, full_row + sample_radius)
        c_min, c_max = max(0, full_col - sample_radius), min(w, full_col + sample_radius)
        local_biomass = app_state.biomass_data[r_min:r_max, c_min:c_max]

        # Calculate average carrying capacity per cell in this area
        land_mask = local_biomass > 0
        if land_mask.any():
            avg_biomass = local_biomass[land_mask].mean()
            # carrying capacity = (biomass * digestibility * utilization) / annual_intake
            avg_cc_per_cell = (avg_biomass * biomass_cfg.digestibility_factor *
                              biomass_cfg.utilization_factor) / bison_cfg.annual_intake_tonnes
        else:
            avg_cc_per_cell = 0.1  # Fallback

        # Calculate release radius to fit population at ~80% of carrying capacity
        # This allows room for growth without immediate die-off
        # In low-quality areas, we spread further to access more total CC
        target_ratio = 0.8  # Start at 80% of carrying capacity (aggressive spreading)
        total_pop = config.initialization.total_population
        cells_needed = total_pop / (avg_cc_per_cell * target_ratio) if avg_cc_per_cell > 0 else 2000

        # Apply a minimum based on population alone (ensures spreading for large populations)
        min_cells_for_pop = total_pop * 3  # At least 3 cells per bison to allow migration
        cells_needed = max(cells_needed, min_cells_for_pop)

        auto_radius = max(30, int(np.sqrt(cells_needed / 3.14)))
        config.initialization.release_radius_cells = min(300, auto_radius)

        print(f"Auto-scaled release radius to {config.initialization.release_radius_cells} for {total_pop} bison")
        print(f"  Local avg biomass: {avg_biomass:.1f}, avg CC/cell: {avg_cc_per_cell:.2f}, cells needed: {cells_needed:.0f}")

        # Validate location is on land
        if app_state.biomass_data[full_row, full_col] <= 0:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid start location",
                    "message": "Cannot start in water. Please select a location on land (green areas)."
                }
            )

        start_location = (full_row, full_col)
        print(f"Start location: row={full_row}, col={full_col}")

    # Create and initialize simulation
    app_state.simulation = Simulation(config)
    state = app_state.simulation.initialize(
        biomass=app_state.biomass_data.copy(),
        cell_size_km=app_state.cell_size_km,
        start_location=start_location
    )
    app_state.is_running = True

    print(f"Simulation started with {config.initialization.total_population} bison")
    print(f"  Grid size: {state.biomass.shape}")
    print(f"  Initial population: {state.population.sum():.0f}")

    # Prepare response (downsampled)
    return _create_state_response(state, "session_" + str(int(time.time())))


class StepRequest(BaseModel):
    """Request for simulation step."""
    years: int = 1


@app.post("/api/step")
async def simulation_step(request: Optional[StepRequest] = None):
    """Advance simulation by specified number of years (default 1)."""
    if app_state.simulation is None or not app_state.is_running:
        raise HTTPException(status_code=400, detail="No active simulation. Call /start first.")

    years = request.years if request else 1
    years = max(1, min(years, 50))  # Clamp between 1 and 50

    total_time_ms = 0
    state = None
    for _ in range(years):
        state, step_time_ms = app_state.simulation.step()
        total_time_ms += step_time_ms

    stats = app_state.simulation.get_statistics()
    print(f"Year {state.year}: pop={stats['total_population']:.0f}, "
          f"cells={stats['occupied_cells']}, time={total_time_ms:.0f}ms ({years} yrs)")

    return _create_state_response(state, "session", total_time_ms)


@app.post("/api/stop")
async def stop_simulation():
    """Stop the current simulation."""
    if app_state.simulation is None:
        return {"message": "No active simulation"}

    year = app_state.simulation.state.year if app_state.simulation.state else 0
    app_state.is_running = False

    print(f"Simulation stopped at year {year}")

    return {
        "message": "Simulation stopped",
        "finalYear": year
    }


@app.get("/api/status")
async def get_status():
    """Get current simulation status."""
    if app_state.simulation is None or app_state.simulation.state is None:
        return {"sessionId": None, "isRunning": False}

    return {
        "sessionId": "session",
        "currentYear": app_state.simulation.state.year,
        "isRunning": app_state.is_running,
        "statistics": app_state.simulation.get_statistics()
    }


def _create_state_response(state: SimulationState, session_id: str, step_time_ms: Optional[float] = None) -> dict:
    """Create API response with downsampled state."""
    # Downsample all grids for visualization
    # Population uses preserve_sum=True to keep total count accurate
    biomass_ds = downsample(state.biomass, VIS_OUTPUT_SIZE, preserve_sum=False)
    population_ds = downsample(state.population, VIS_OUTPUT_SIZE, preserve_sum=True)
    food_sat_ds = downsample(state.food_satisfaction, VIS_OUTPUT_SIZE, preserve_sum=False)
    carrying_cap_ds = downsample(state.carrying_capacity, VIS_OUTPUT_SIZE, preserve_sum=False)

    vis_cell_size = state.cell_size_km * (state.biomass.shape[1] / VIS_OUTPUT_SIZE)

    response = {
        "sessionId": session_id,
        "year": state.year,
        "state": {
            "biomass": biomass_ds.tolist(),
            "population": population_ds.tolist(),
            "foodSatisfaction": food_sat_ds.tolist(),
            "carryingCapacity": carrying_cap_ds.tolist()
        },
        "metadata": {
            "width": VIS_OUTPUT_SIZE,
            "height": VIS_OUTPUT_SIZE,
            "cellSizeKm": vis_cell_size
        }
    }

    if step_time_ms is not None:
        response["stepTimeMs"] = step_time_ms

    return response


# --- Main ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001)
