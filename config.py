"""
Simulation configuration with scientifically-based parameters.

Sources:
- Meagher (1986) Bison bison, Mammalian Species No. 266
- Plumb & Dodd (1993) Foraging ecology of bison and cattle
- USDA Wildlife Services Bison Management Guidelines
- Poquérusse et al. (2024) - Digestibility factors
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np


@dataclass
class BiomassConfig:
    """Configuration for biomass dynamics."""
    # Data is already digestibility-weighted from GeoTIFF processing
    digestibility_factor: float = 1.0
    # Annual regrowth rate (fraction of deficit recovered per year)
    annual_growth_factor: float = 0.4
    # Fraction of digestible biomass that can be sustainably harvested
    # Increased to allow reasonable carrying capacity (50% is typical for managed grazing)
    utilization_factor: float = 0.5
    # Scaling factor for maximum biomass
    max_biomass_scaling: float = 1.0


@dataclass
class BisonConfig:
    """Configuration for bison population dynamics."""
    # Average adult body mass in kg (males ~900kg, females ~500kg)
    body_mass_kg: float = 700.0
    # Daily dry matter intake as fraction of body mass (1.5-2.5%)
    daily_intake_rate: float = 0.02
    # Maximum intrinsic growth rate (r_max)
    # Yukon reintroduction data shows ~20%/year in good habitat
    # Yellowstone studies show 7-8% in saturated habitat
    max_growth_rate: float = 0.20
    # Food satisfaction threshold below which population declines
    # Lowered to be more forgiving - decline only when very hungry
    starvation_threshold: float = 0.2
    # Minimum viable population density (Allee effect) - lowered to allow sparse colonization
    min_viable_density: float = 0.05
    # Growth bonus for pioneer populations colonizing new areas
    # Modest bonus for populations below carrying capacity
    pioneer_bonus: float = 0.05

    @property
    def annual_intake_tonnes(self) -> float:
        """Annual food intake per bison in tonnes."""
        return (self.body_mass_kg * self.daily_intake_rate * 365) / 1000


@dataclass
class MigrationConfig:
    """Configuration for bison migration behavior."""
    # Typical annual migration distance in km
    annual_migration_km: float = 200.0  # Bison seasonal migrations cover 100-300km
    # Base diffusion rate for population spread (fraction that moves per year)
    diffusion_rate: float = 0.35  # Higher rate for realistic frontier spread
    # Weight for food availability in attractiveness calculation
    food_preference_weight: float = 1.0
    # Random movement noise factor
    movement_noise: float = 0.05
    # Whether boundaries wrap (False = edges are barriers)
    wrap_boundaries: bool = False


@dataclass
class InitializationConfig:
    """Configuration for initial population placement."""
    # Total number of bison to introduce
    total_population: int = 50
    # Radius of release area in cells
    release_radius_cells: int = 5
    # Center coordinates (row, col) - None means use provided coordinates
    center: Optional[Tuple[int, int]] = None


@dataclass
class SimulationConfig:
    """Complete simulation configuration."""
    biomass: BiomassConfig = field(default_factory=BiomassConfig)
    bison: BisonConfig = field(default_factory=BisonConfig)
    migration: MigrationConfig = field(default_factory=MigrationConfig)
    initialization: InitializationConfig = field(default_factory=InitializationConfig)
    seed: Optional[int] = 42


# Visualization output size (downsampled for browser)
VIS_OUTPUT_SIZE = 400
