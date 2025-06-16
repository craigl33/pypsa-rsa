# PyPSA-RSA vs PyPSA-EUR Version Comparison

## Major Package Updates

| Package | PyPSA-RSA (old) | PyPSA-EUR (new) | Change Impact |
|---------|------------------|------------------|---------------|
| **Python** | 3.10.10 | 3.12.10 | ⬆️ Major - Better performance, new features |
| **numpy** | 1.23.5 | 1.26.4 | ⬆️ Critical - Fixes compatibility issues |
| **pandas** | 2.0.1 | 2.2.3 | ⬆️ Critical - Fixes Excel parsing bugs |
| **pypsa** | 0.22.1 | 0.34.1 | ⬆️ Major - Many new features and bug fixes |
| **matplotlib** | 3.5.2 | 3.9.1 | ⬆️ Major - Better plots and compatibility |
| **scipy** | 1.10.1 | 1.15.2 | ⬆️ Major - Performance improvements |

## Scientific Computing Stack

| Package | PyPSA-RSA (old) | PyPSA-EUR (new) | Notes |
|---------|------------------|------------------|-------|
| **networkx** | 3.1 | 3.4.2 | Network analysis improvements |
| **xarray** | 2023.4.2 | 2025.4.0 | Major version jump - better performance |
| **netcdf4** | 1.6.3 | 1.7.2 | Climate data handling improvements |
| **numexpr** | 2.8.4 | 2.10.2 | Faster numerical expressions |
| **dask** | 2023.4.1 | 2025.4.1 | Parallel computing improvements |

## Geographic and GIS Stack

| Package | PyPSA-RSA (old) | PyPSA-EUR (new) | Notes |
|---------|------------------|------------------|-------|
| **geopandas** | 0.13.0 | 1.0.1 | ⬆️ Major version - API improvements |
| **shapely** | 2.0.1 | 2.0.7 | Geometry processing fixes |
| **cartopy** | 0.21.1 | 0.24.0 | Map plotting improvements |
| **fiona** | 1.9.3 | 1.10.1 | File I/O improvements |
| **pyproj** | 3.5.0 | 3.7.1 | Coordinate system improvements |
| **rasterio** | 1.3.6 | 1.4.3 | Raster data handling |

## Energy System Modeling

| Package | PyPSA-RSA (old) | PyPSA-EUR (new) | Notes |
|---------|------------------|------------------|-------|
| **atlite** | 0.2.11 | 0.4.1 | ⬆️ Major - Weather data processing |
| **powerplantmatching** | 0.5.6 | 0.7.1 | Power plant database improvements |
| **linopy** | 0.1.5 | 0.5.5 | ⬆️ Major - Optimization modeling |
| **country_converter** | 1.0.0 | 1.2 | Country code handling |

## Workflow Management

| Package | PyPSA-RSA (old) | PyPSA-EUR (new) | Notes |
|---------|------------------|------------------|-------|
| **snakemake-minimal** | 7.25.3 | 9.3.4 | ⬆️ Major version - New features |
| **ruamel.yaml** | 0.17.24 | Latest | YAML processing |

## Development Tools

| Package | PyPSA-RSA (old) | PyPSA-EUR (new) | Notes |
|---------|------------------|------------------|-------|
| **jupyterlab** | 3.6.3 | 4.4.2 | ⬆️ Major - New interface |
| **ipython** | 8.13.2 | 9.2.0 | Interactive Python improvements |
| **pre-commit** | 3.3.1 | 4.2.0 | Code quality tools |

## Solvers and Optimization

| Package | PyPSA-RSA (old) | PyPSA-EUR (new) | Notes |
|---------|------------------|------------------|-------|
| **glpk** | 5.0 | 5.0 | ✅ Same (stable) |
| **ipopt** | 3.13.2 | 3.14.17 | Optimization improvements |
| **highspy** | Not included | 1.10.0 | ⬆️ New - Fast modern solver |
| **coin-or-cbc** | 2.10.10 | 2.10.12 | MILP solver updates |

## Key Benefits of Updating

### 🔧 **Compatibility Fixes**
- Resolves pandas Excel reading issues
- Fixes numpy array conversion problems
- Better Python 3.12 support

### 🚀 **Performance Improvements**
- Faster optimization with HiGHS solver
- Better parallel processing with updated dask
- Improved geographic operations

### 📊 **New Features**
- Modern plotting capabilities
- Enhanced workflow management
- Better data handling

### 🛡️ **Stability**
- Tested package combinations
- Known working versions
- Reduced compatibility issues

## Installation Command

```bash
# Remove old environment
conda env remove -n pypsa-rsa --yes

# Create new environment with updated versions
conda env create -f environment-updated.yaml

# Activate and test
conda activate pypsa-rsa
python -c "import pypsa; print(f'PyPSA {pypsa.__version__} ready!')"
```

## Breaking Changes to Watch For

### **pandas 2.0.1 → 2.2.3**
- Improved Excel reading (fixes your current issue!)
- Some deprecated functions removed
- Better performance

### **numpy 1.23.5 → 1.26.4**
- Fixes array conversion issues
- Some deprecated aliases removed
- Better error messages

### **geopandas 0.13.0 → 1.0.1**
- Major API stabilization
- Better performance
- Some method names changed

### **snakemake 7.x → 9.x**
- New plugin system
- Updated CLI interface
- Better resource handling

Most changes are backward compatible, but test your workflow after updating!