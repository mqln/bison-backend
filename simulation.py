"""
High-performance bison population simulation using NumPy.

All grid operations are vectorized for maximum performance.
Migration uses scipy convolution instead of cell-by-cell iteration.
"""

import numpy as np
from scipy import ndimage, signal
from numba import jit, prange
from typing import Tuple, Optional
from dataclasses import dataclass
import time

from config import (
    SimulationConfig, BiomassConfig, BisonConfig,
    MigrationConfig, InitializationConfig
)


@dataclass
class SimulationState:
    """Current state of the simulation."""
    year: int
    biomass: np.ndarray
    max_biomass: np.ndarray
    population: np.ndarray
    food_satisfaction: np.ndarray
    carrying_capacity: np.ndarray
    cell_size_km: float


class BiomassService:
    """Vectorized biomass calculations using NumPy."""

    @staticmethod
    def calculate_digestible(biomass: np.ndarray, config: BiomassConfig) -> np.ndarray:
        """Calculate digestible biomass (vectorized)."""
        return biomass * config.digestibility_factor

    @staticmethod
    def calculate_sustainable_harvest(digestible: np.ndarray, config: BiomassConfig) -> np.ndarray:
        """Calculate sustainable harvest amount (vectorized)."""
        return digestible * config.utilization_factor

    @staticmethod
    def calculate_regrowth(current: np.ndarray, max_biomass: np.ndarray, config: BiomassConfig) -> np.ndarray:
        """Calculate annual regrowth (vectorized)."""
        deficit = max_biomass - current
        return deficit * config.annual_growth_factor

    @staticmethod
    def update_biomass(
        current: np.ndarray,
        max_biomass: np.ndarray,
        consumed: np.ndarray,
        config: BiomassConfig
    ) -> np.ndarray:
        """
        Update biomass after consumption and regrowth.

        All operations are vectorized - no loops!
        """
        regrowth = BiomassService.calculate_regrowth(current, max_biomass, config)
        # In-place operations for memory efficiency
        result = current + regrowth - consumed
        np.clip(result, 0, None, out=result)
        return result


class BisonService:
    """Vectorized bison population calculations."""

    @staticmethod
    def calculate_food_demand(population: np.ndarray, config: BisonConfig) -> np.ndarray:
        """Calculate food demand per cell (vectorized)."""
        return population * config.annual_intake_tonnes

    @staticmethod
    def calculate_carrying_capacity(sustainable_harvest: np.ndarray, config: BisonConfig) -> np.ndarray:
        """Calculate carrying capacity based on available food (vectorized)."""
        return sustainable_harvest / config.annual_intake_tonnes

    @staticmethod
    def calculate_food_satisfaction(demand: np.ndarray, consumed: np.ndarray) -> np.ndarray:
        """
        Calculate food satisfaction ratio (vectorized).

        Returns 1.0 where demand is 0 (no population = satisfied).
        """
        satisfaction = np.ones_like(demand)
        mask = demand > 0
        satisfaction[mask] = np.clip(consumed[mask] / demand[mask], 0, 1)
        return satisfaction

    @staticmethod
    def initialize_population(
        shape: Tuple[int, int],
        config: InitializationConfig,
        land_mask: np.ndarray,
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Initialize population at specified location.

        Distributes total_population across cells within release_radius,
        weighted toward the center, only on valid land cells.
        """
        population = np.zeros(shape, dtype=np.float32)

        if config.center is None:
            return population

        center_row, center_col = config.center
        radius = config.release_radius_cells

        # Create coordinate grids
        rows = np.arange(max(0, center_row - radius), min(shape[0], center_row + radius + 1))
        cols = np.arange(max(0, center_col - radius), min(shape[1], center_col + radius + 1))
        row_grid, col_grid = np.meshgrid(rows, cols, indexing='ij')

        # Calculate distances from center
        distances = np.sqrt((row_grid - center_row)**2 + (col_grid - center_col)**2)

        # Create mask for valid cells (within radius and on land)
        within_radius = distances <= radius
        land_values = land_mask[row_grid, col_grid]
        valid_mask = within_radius & (land_values > 0)

        if not np.any(valid_mask):
            return population

        # Weight by inverse distance AND biomass quality
        # This ensures bison prefer the best habitat within the release area
        weights = np.zeros_like(distances)
        distance_weight = 1.0 / (1.0 + distances[valid_mask])
        biomass_weight = np.sqrt(land_values[valid_mask])  # sqrt to reduce extreme bias
        weights[valid_mask] = distance_weight * biomass_weight
        weights /= weights.sum()

        # Distribute population according to weights
        expected_counts = weights * config.total_population

        # Use Poisson distribution for realistic variation
        actual_counts = rng.poisson(expected_counts)
        actual_counts[~valid_mask] = 0

        # Adjust to hit target population
        total = actual_counts.sum()
        if total > 0:
            actual_counts = (actual_counts * config.total_population / total).astype(np.float32)

        # Place in population grid
        population[row_grid, col_grid] = actual_counts

        return population

    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def update_population_numba(
        population: np.ndarray,
        carrying_capacity: np.ndarray,
        food_satisfaction: np.ndarray,
        max_growth_rate: float,
        starvation_threshold: float,
        min_viable_density: float,
        pioneer_bonus: float
    ) -> np.ndarray:
        """
        Update population using logistic growth model.

        This is JIT-compiled with Numba for maximum performance.
        Uses parallel loops for multi-core execution.
        """
        height, width = population.shape
        result = np.empty_like(population)
        epsilon = 1e-10

        for row in prange(height):
            for col in range(width):
                pop = population[row, col]
                capacity = carrying_capacity[row, col]
                satisfaction = food_satisfaction[row, col]

                if pop < epsilon:
                    result[row, col] = 0.0
                    continue

                # Check if population is viable
                is_viable = pop >= min_viable_density

                # Calculate capacity ratio (how full is this cell relative to carrying capacity)
                capacity_ratio = pop / (capacity + epsilon)

                # Calculate growth factor based on food satisfaction
                # Uses logistic growth: r * (1 - N/K) where N/K is capacity_ratio
                if satisfaction > starvation_threshold:
                    if is_viable:
                        # Standard logistic growth with food modifier
                        # Small pioneer bonus when well below capacity
                        effective_rate = max_growth_rate
                        if capacity_ratio < 0.5:
                            effective_rate += pioneer_bonus
                        growth_factor = effective_rate * satisfaction * (1 - capacity_ratio)
                    else:
                        # Allee effect: sparse populations decline
                        growth_factor = -0.2 * (min_viable_density - pop) / min_viable_density
                else:
                    # Starvation - population decline proportional to food deficit
                    growth_factor = -max_growth_rate * (1 - satisfaction / starvation_threshold)

                # Clip growth factor to realistic bounds
                # Max ~20% growth (Yukon data), max 30% decline per year
                growth_factor = max(-0.3, min(0.20, growth_factor))

                # Update population
                new_pop = pop * (1 + growth_factor)
                result[row, col] = max(0.0, min(1e6, new_pop))

        return result

    @staticmethod
    def update_population(
        population: np.ndarray,
        carrying_capacity: np.ndarray,
        food_satisfaction: np.ndarray,
        config: BisonConfig
    ) -> np.ndarray:
        """Update population (wrapper for Numba function)."""
        return BisonService.update_population_numba(
            population.astype(np.float64),
            carrying_capacity.astype(np.float64),
            food_satisfaction.astype(np.float64),
            config.max_growth_rate,
            config.starvation_threshold,
            config.min_viable_density,
            config.pioneer_bonus
        ).astype(np.float32)


class MigrationService:
    """
    Vectorized migration using convolution.

    This is the KEY optimization - instead of 24 passes over the grid,
    we use a single convolution operation.
    """

    @staticmethod
    def create_migration_kernel(config: MigrationConfig, cell_size_km: float) -> np.ndarray:
        """
        Create a migration kernel for convolution.

        The kernel represents the probability of movement from the center
        cell to surrounding cells, based on distance.
        """
        # Kernel size based on migration distance (capped for performance)
        cells_per_year = config.annual_migration_km / cell_size_km
        kernel_radius = min(15, max(2, int(cells_per_year / 10)))  # Proportional kernel
        kernel_size = 2 * kernel_radius + 1

        # Create distance-based kernel
        y, x = np.ogrid[-kernel_radius:kernel_radius+1, -kernel_radius:kernel_radius+1]
        distances = np.sqrt(x*x + y*y)

        # Weight by inverse distance with steeper decay (keeps population more concentrated)
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        mask = (distances > 0) & (distances <= kernel_radius)
        kernel[mask] = 1.0 / (1.0 + distances[mask])**1.5  # Moderate decay: herd cohesion + frontier spread

        # Center cell keeps most population (1 - diffusion_rate)
        kernel[kernel_radius, kernel_radius] = 0

        # Normalize outward movement
        if kernel.sum() > 0:
            kernel = kernel / kernel.sum() * config.diffusion_rate

        # Center retains the rest
        kernel[kernel_radius, kernel_radius] = 1.0 - config.diffusion_rate

        return kernel

    @staticmethod
    def calculate_attractiveness(
        carrying_capacity: np.ndarray,
        config: MigrationConfig,
        rng: np.random.Generator
    ) -> np.ndarray:
        """Calculate cell attractiveness for migration (vectorized)."""
        base = carrying_capacity * config.food_preference_weight
        noise = rng.normal(0, config.movement_noise, base.shape).astype(np.float32)
        return np.clip(base + noise, 0, None)

    @staticmethod
    def create_directional_kernels(config: MigrationConfig, cell_size_km: float) -> dict:
        """Create directional migration kernels for attractiveness-biased movement."""
        cells_per_year = config.annual_migration_km / cell_size_km
        radius = min(15, max(2, int(cells_per_year / 10)))

        kernels = {}

        # Create kernels biased in each direction
        for direction, (dy, dx) in [('up', (-1, 0)), ('down', (1, 0)),
                                     ('left', (0, -1)), ('right', (0, 1))]:
            size = 2 * radius + 1
            kernel = np.zeros((size, size), dtype=np.float32)
            center = radius

            for i in range(size):
                for j in range(size):
                    if i == center and j == center:
                        continue
                    dist = np.sqrt((i - center)**2 + (j - center)**2)
                    if dist <= radius:
                        # Weight by alignment with direction
                        dir_alignment = (i - center) * dy + (j - center) * dx
                        if dir_alignment > 0:  # Only cells in the target direction
                            kernel[i, j] = dir_alignment / (1 + dist)

            if kernel.sum() > 0:
                kernel /= kernel.sum()
            kernels[direction] = kernel

        return kernels

    @staticmethod
    def migrate(
        population: np.ndarray,
        attractiveness: np.ndarray,
        land_mask: np.ndarray,
        config: MigrationConfig,
        cell_size_km: float,
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Perform migration using convolution - optimized version.

        Uses directional convolutions based on attractiveness gradients,
        avoiding expensive interpolation.
        """
        if population.sum() < 0.1:
            return population.copy()

        # Create base diffusion kernel
        kernel = MigrationService.create_migration_kernel(config, cell_size_km)

        # Base diffusion using FFT convolution (50x faster than ndimage for large grids)
        diffused = signal.fftconvolve(population, kernel, mode='same')

        # Calculate attractiveness gradients (which direction is better)
        # Use simple diff for speed (Sobel is slower)
        grad_y = np.diff(attractiveness, axis=0, prepend=attractiveness[:1, :])
        grad_x = np.diff(attractiveness, axis=1, prepend=attractiveness[:, :1])

        # Directional bias based on gradient
        # Shift population toward higher attractiveness using roll operations
        bias_strength = config.food_preference_weight * config.diffusion_rate * 0.5

        # Create shifted versions
        shift_up = np.roll(diffused, -1, axis=0)
        shift_down = np.roll(diffused, 1, axis=0)
        shift_left = np.roll(diffused, -1, axis=1)
        shift_right = np.roll(diffused, 1, axis=1)

        # Weight shifts by gradient direction
        # Positive grad_y means attractiveness increases downward
        up_weight = np.clip(-grad_y * bias_strength, 0, 0.2)
        down_weight = np.clip(grad_y * bias_strength, 0, 0.2)
        left_weight = np.clip(-grad_x * bias_strength, 0, 0.2)
        right_weight = np.clip(grad_x * bias_strength, 0, 0.2)

        # Apply directional shifts
        total_shift = up_weight + down_weight + left_weight + right_weight
        stay_weight = 1.0 - total_shift

        result = (
            stay_weight * diffused +
            up_weight * shift_up +
            down_weight * shift_down +
            left_weight * shift_left +
            right_weight * shift_right
        ).astype(np.float32)

        # Mask out water
        result[land_mask <= 0] = 0.0

        # Zero out edge artifacts from rolling
        result[0, :] = 0
        result[-1, :] = 0
        result[:, 0] = 0
        result[:, -1] = 0

        # Small stochastic variation
        pop_mask = result > 0.1
        if pop_mask.any():
            result[pop_mask] *= (1 + rng.normal(0, 0.02, pop_mask.sum()).astype(np.float32))

        # Ensure non-negative
        np.clip(result, 0, None, out=result)

        # Conserve total population
        total_before = population.sum()
        total_after = result.sum()
        if total_after > 0:
            result *= total_before / total_after

        return result


class Simulation:
    """Main simulation orchestrator."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.state: Optional[SimulationState] = None
        self._migration_kernel: Optional[np.ndarray] = None

    def initialize(
        self,
        biomass: np.ndarray,
        cell_size_km: float,
        start_location: Optional[Tuple[int, int]] = None
    ) -> SimulationState:
        """Initialize simulation with biomass data and starting location."""

        # Set up max biomass
        max_biomass = biomass * self.config.biomass.max_biomass_scaling

        # Initialize population at start location
        init_config = InitializationConfig(
            total_population=self.config.initialization.total_population,
            release_radius_cells=self.config.initialization.release_radius_cells,
            center=start_location
        )

        population = BisonService.initialize_population(
            biomass.shape,
            init_config,
            land_mask=biomass,  # biomass > 0 = land
            rng=self.rng
        )

        # Calculate initial metrics
        digestible = BiomassService.calculate_digestible(biomass, self.config.biomass)
        sustainable_harvest = BiomassService.calculate_sustainable_harvest(digestible, self.config.biomass)
        carrying_capacity = BisonService.calculate_carrying_capacity(sustainable_harvest, self.config.bison)
        food_demand = BisonService.calculate_food_demand(population, self.config.bison)
        consumed = np.minimum(sustainable_harvest, food_demand)
        food_satisfaction = BisonService.calculate_food_satisfaction(food_demand, consumed)

        self.state = SimulationState(
            year=0,
            biomass=biomass.astype(np.float32),
            max_biomass=max_biomass.astype(np.float32),
            population=population.astype(np.float32),
            food_satisfaction=food_satisfaction.astype(np.float32),
            carrying_capacity=carrying_capacity.astype(np.float32),
            cell_size_km=cell_size_km
        )

        # Pre-compute migration kernel
        self._migration_kernel = MigrationService.create_migration_kernel(
            self.config.migration,
            cell_size_km
        )

        return self.state

    def step(self) -> Tuple[SimulationState, float]:
        """
        Run one simulation step (one year).

        Returns the new state and the step time in milliseconds.
        """
        if self.state is None:
            raise RuntimeError("Simulation not initialized. Call initialize() first.")

        start_time = time.perf_counter()

        state = self.state
        biomass_config = self.config.biomass
        bison_config = self.config.bison
        migration_config = self.config.migration

        # Calculate food availability
        digestible = BiomassService.calculate_digestible(state.biomass, biomass_config)
        sustainable_harvest = BiomassService.calculate_sustainable_harvest(digestible, biomass_config)

        # Calculate food demand and consumption
        food_demand = BisonService.calculate_food_demand(state.population, bison_config)
        consumed = np.minimum(sustainable_harvest, food_demand)
        food_satisfaction = BisonService.calculate_food_satisfaction(food_demand, consumed)

        # Calculate carrying capacity
        carrying_capacity = BisonService.calculate_carrying_capacity(sustainable_harvest, bison_config)

        # Update biomass
        new_biomass = BiomassService.update_biomass(
            state.biomass,
            state.max_biomass,
            consumed,
            biomass_config
        )

        # Migration
        attractiveness = MigrationService.calculate_attractiveness(
            carrying_capacity,
            migration_config,
            self.rng
        )
        migrated_population = MigrationService.migrate(
            state.population,
            attractiveness,
            land_mask=state.max_biomass,  # max_biomass > 0 = land
            config=migration_config,
            cell_size_km=state.cell_size_km,
            rng=self.rng
        )

        # Update population
        new_population = BisonService.update_population(
            migrated_population,
            carrying_capacity,
            food_satisfaction,
            bison_config
        )

        # Update state
        self.state = SimulationState(
            year=state.year + 1,
            biomass=new_biomass,
            max_biomass=state.max_biomass,
            population=new_population,
            food_satisfaction=food_satisfaction,
            carrying_capacity=carrying_capacity,
            cell_size_km=state.cell_size_km
        )

        step_time_ms = (time.perf_counter() - start_time) * 1000

        return self.state, step_time_ms

    def get_statistics(self) -> dict:
        """Get current simulation statistics."""
        if self.state is None:
            return {}

        pop = self.state.population
        return {
            "total_population": float(pop.sum()),
            "occupied_cells": int((pop > 0.1).sum()),
            "max_density": float(pop.max()),
            "mean_satisfaction": float(self.state.food_satisfaction[pop > 0.1].mean()) if (pop > 0.1).any() else 0.0
        }
