# PyPSA-ZA Network File Structure Documentation

## Overview

PyPSA-ZA produces network files in NetCDF (*.nc) format that contain comprehensive power system data for South Africa. These files are generated primarily by the `add_electricity.py` script and contain time-series data, component parameters, and network topology information suitable for power system optimization studies.

## File Generation Workflow

The network file is created through the following pipeline:

1. **Base Network Creation** (`base_network.py`)
   - Creates network topology (buses and transmission links)
   - Sets up temporal structure (snapshots and investment periods)
   - Establishes spatial regions

2. **Electricity System Addition** (`add_electricity.py`)
   - Adds load profiles and generators (fixed and extendable)
   - Incorporates storage units
   - Applies operational constraints and availability factors
   - Processes renewable energy profiles

3. **Network Preparation** (`prepare_and_solve_network.py`)
   - Applies time aggregation if specified
   - Sets capacity limits and constraints
   - Prepares for optimization

## Core Data Structure

### Dimensions

The network file uses several key dimensions:

- **`snapshots`** (87,600): Hourly time steps across all investment periods
- **`investment_periods`** (10): Planning years (e.g., 2025, 2030, ..., 2050)
- **Component indices**: Individual components like generators, loads, buses, storage units

### Coordinate Systems

#### Temporal Coordinates
```
snapshots: 0, 1, 2, ..., 87599 (indexed time steps)
investment_periods: 2025, 2030, 2035, 2040, 2045, 2050
```

#### Spatial Coordinates
```
buses_i: Individual bus identifiers (e.g., 'RSA' for single-node model)
generators_i: Generator names (e.g., 'Arnot*', 'Koeberg*')
```

## Component Categories

### 1. Buses (`buses_*`)

**Static Attributes:**
- `buses_v_nom`: Nominal voltage (kV)
- `buses_x`, `buses_y`: Geographic coordinates (longitude, latitude)
- `buses_POP_2016`: Population data for load disaggregation
- `buses_GVA_2016`: Gross Value Added for economic weighting

### 2. Loads (`loads_*`)

**Static Attributes:**
- `loads_bus`: Bus connection

**Time-Series Data:**
- `loads_t_p_set`: Hourly load demand (MW) across all snapshots

### 3. Generators (`generators_*`)

**Static Attributes:**
- `generators_bus`: Bus connection
- `generators_p_nom`: Installed capacity (MW)
- `generators_p_nom_extendable`: Boolean indicating if capacity can be expanded
- `generators_carrier`: Technology type (e.g., 'coal', 'solar_pv', 'wind_onshore')
- `generators_marginal_cost`: Variable O&M cost (ZAR/MWh)
- `generators_capital_cost`: Annualized capital cost (ZAR/MW/year)
- `generators_efficiency`: Energy conversion efficiency (per unit)
- `generators_build_year`: Year of commissioning
- `generators_lifetime`: Operational lifetime (years)

**Ramping and Unit Commitment:**
- `generators_ramp_limit_up/down`: Ramping rates (%/hour)
- `generators_ramp_limit_start_up/shut_down`: Startup/shutdown ramping
- `generators_start_up_cost`: Startup cost (ZAR)
- `generators_min_up_time/min_down_time`: Minimum online/offline time (hours)

**Time-Series Data:**
- `generators_t_p_max_pu`: Maximum capacity factor (0-1) for each snapshot
- `generators_t_p_min_pu`: Minimum capacity factor (0-1) for each snapshot
- `generators_t_ramp_limit_up/down`: Time-varying ramping limits

### 4. Storage Units (`storage_units_*`)

**Static Attributes:**
- `storage_units_bus`: Bus connection
- `storage_units_p_nom`: Power capacity (MW)
- `storage_units_carrier`: Storage technology type
- `storage_units_marginal_cost`: Variable cost (ZAR/MWh)
- `storage_units_capital_cost`: Annualized capital cost (ZAR/MW/year)
- `storage_units_max_hours`: Energy-to-power ratio (hours)
- `storage_units_efficiency_store/dispatch`: Round-trip efficiency components
- `storage_units_cyclic_state_of_charge`: Boolean for cyclic operation

### 5. Temporal Structure

**Snapshot Weightings:**
- `snapshots_objective`: Weighting for objective function
- `snapshots_generators`: Weighting for generator constraints
- `snapshots_period`: Investment period assignment
- `snapshots_timestep`: Actual datetime stamps

**Investment Period Weightings:**
- `investment_periods_objective`: Present value weighting
- `investment_periods_years`: Duration of each period

## Technology Classification

### Generator Carriers

Based on the code analysis, generators are classified by carrier:

**Conventional Technologies:**
- `coal`: Coal-fired power plants
- `nuclear`: Nuclear power plants
- `ocgt_gas`, `ocgt_diesel`: Open cycle gas turbines
- `ccgt_steam`: Combined cycle gas turbine steam component
- `hydro`: Hydroelectric power

**Renewable Technologies:**
- `solar_pv`: Solar photovoltaic (utility-scale)
- `solar_pv_rooftop`: Rooftop solar PV
- `wind_onshore`: Onshore wind
- `wind_offshore`: Offshore wind
- `biomass`: Biomass power plants

**Emerging Technologies:**
- `rmippp`: Risk Mitigation Independent Power Producer Programme

### Storage Carriers
- `battery`: Battery energy storage systems
- `PHS`: Pumped hydro storage

## Data Processing Pipeline

### 1. Load Data Processing
```python
# Load profiles are normalized and scaled by annual demand
profile_load = normed(load["system_energy"])
load = profile_load * annual_demand
```

### 2. Generator Availability Factors

**Fixed Generators:**
- Historical availability from Eskom data
- Plant-specific maintenance schedules
- Forced outage rates

**Renewable Generators:**
- Weather-dependent capacity factors
- Resource availability profiles
- Degradation adjustments

### 3. Multi-Investment Period Handling

The model supports capacity expansion planning across multiple periods:
- Components have `build_year` and `lifetime` attributes
- Time-series data spans all investment periods
- Capacity availability tracked by investment period

## Key Configuration Sources

### Excel Input Files

1. **`fixed_technologies.xlsx`**
   - Existing power plant database
   - Technical parameters and locations
   - Grouped by conventional, renewables, storage

2. **`extendable_technologies.xlsx`**
   - Future technology options
   - Cost trajectories by year
   - Build limits and constraints

3. **`annual_load.xlsx`**
   - Load growth trajectories
   - Demand scenarios

4. **`plant_availability.xlsx`**
   - Availability factors by technology
   - Maintenance schedules
   - Outage profiles

## Usage for Power System Modeling

### Data Access Patterns

```python
import xarray as xr

# Load network file
ds = xr.open_dataset('network.nc')

# Access generator data
generators = ds.sel(generators_i=slice(None))
load_data = ds['loads_t_p_set']

# Time-series analysis
capacity_factors = ds['generators_t_p_max_pu']
```

### Integration Considerations

**For NREL Sienna:**
1. **Component Mapping**: PyPSA generators/storage → Sienna ThermalGen/RenewableGen/Storage
2. **Time Series**: Hourly data → Sienna TimeSeriesData
3. **Network Topology**: Bus/branch model → Sienna System
4. **Multi-Period**: Investment periods → Sienna planning horizon

## File Characteristics

- **Format**: NetCDF4 with compression
- **Size**: Typically 100-500 MB depending on temporal resolution
- **Encoding**: UTF-8 for string data, float64 for numerical data
- **Chunking**: Optimized for time-series access patterns

## Quality Assurance

The data includes several validation mechanisms:
- Capacity factor bounds checking (0 ≤ p_max_pu ≤ 1)
- Energy balance validation
- Ramp rate feasibility checks
- Investment period consistency

This structure provides a comprehensive foundation for power system optimization studies, supporting both operational dispatch and capacity expansion planning for the South African electricity system.