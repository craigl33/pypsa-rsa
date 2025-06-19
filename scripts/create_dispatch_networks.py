# scripts/create_dispatch_networks.py

"""
Create individual yearly dispatch networks from capacity expansion results.

This script takes the optimized capacity expansion network and creates separate
yearly networks for operational dispatch optimization.
"""

import pypsa
import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from _helpers import single_year_network_copy

def create_dispatch_networks(capacity_network_path, output_paths):
   """
   Updated version that handles TSAM-clustered capacity networks
   """
   
   logging.info(f"Loading capacity expansion network: {capacity_network_path}")
   n_capacity = pypsa.Network(capacity_network_path)
   
   # Check if this is a multi-investment period network
   if not n_capacity.multi_invest:
       logging.error("Network is not a multi-investment period model")
       raise ValueError("Dispatch networks can only be created from multi-investment models")
   
   logging.info(f"Found {len(n_capacity.investment_periods)} investment periods: {n_capacity.investment_periods}")
   
   # Create dispatch network for each investment period
   for year in n_capacity.investment_periods:
       logging.info(f"Creating dispatch network for year {year}")
       
       # Get snapshots for this year (full 8760 hours)
       year_snapshots = pd.date_range(
           start=f"{year}-01-01 00:00", 
           end=f"{year}-12-31 23:00", 
           freq="H"
       )
       # Remove leap day
       year_snapshots = year_snapshots[~((year_snapshots.month == 2) & (year_snapshots.day == 29))]
       
       # Create single-year network copy with TSAM reconstruction if needed
       n_dispatch = single_year_network_copy_with_tsam_reconstruction(
           n_capacity, 
           snapshots=year_snapshots, 
           investment_periods=[year]
       )
       
       # Fix capacities at optimized values for dispatch
       fix_optimal_capacities(n_dispatch, n_capacity, year)
       
       # Update time series for full year resolution
       update_dispatch_time_series(n_dispatch, n_capacity, year)
       
       # Set network name
       n_dispatch.name = f"Dispatch_{year}"
       
       # Determine output path for this year
       output_path = None
       for path in output_paths:
           if f"dispatch-{year}.nc" in path:
               output_path = path
               break
       
       if output_path is None:
           logging.warning(f"No output path found for year {year}")
           continue
           
       # Create output directory
       os.makedirs(os.path.dirname(output_path), exist_ok=True)
       
       # Export dispatch network
       logging.info(f"Exporting dispatch network to: {output_path}")
       n_dispatch.export_to_netcdf(output_path)
   
   logging.info("Dispatch network creation complete")
def fix_optimal_capacities(n_dispatch, n_capacity, year):
    """
    Fix all capacities at their optimized values from capacity expansion
    """
    
    # Fix generator capacities
    if hasattr(n_capacity, 'generators') and 'p_nom_opt' in n_capacity.generators.columns:
        # Get active generators for this year
        active_gens = n_capacity.get_active_assets('Generator', year)
        active_gen_idx = n_capacity.generators.index[active_gens]
        
        for gen in active_gen_idx:
            if gen in n_dispatch.generators.index:
                # Set p_nom to optimized value and make non-extendable
                n_dispatch.generators.loc[gen, 'p_nom'] = n_capacity.generators.loc[gen, 'p_nom_opt']
                n_dispatch.generators.loc[gen, 'p_nom_extendable'] = False
                n_dispatch.generators.loc[gen, 'p_nom_min'] = n_capacity.generators.loc[gen, 'p_nom_opt']
                n_dispatch.generators.loc[gen, 'p_nom_max'] = n_capacity.generators.loc[gen, 'p_nom_opt']
    
    # Fix storage unit capacities
    if hasattr(n_capacity, 'storage_units') and 'p_nom_opt' in n_capacity.storage_units.columns:
        active_storage = n_capacity.get_active_assets('StorageUnit', year)
        active_storage_idx = n_capacity.storage_units.index[active_storage]
        
        for storage in active_storage_idx:
            if storage in n_dispatch.storage_units.index:
                n_dispatch.storage_units.loc[storage, 'p_nom'] = n_capacity.storage_units.loc[storage, 'p_nom_opt']
                n_dispatch.storage_units.loc[storage, 'p_nom_extendable'] = False
                n_dispatch.storage_units.loc[storage, 'p_nom_min'] = n_capacity.storage_units.loc[storage, 'p_nom_opt']
                n_dispatch.storage_units.loc[storage, 'p_nom_max'] = n_capacity.storage_units.loc[storage, 'p_nom_opt']
    
    # Fix line capacities (if extendable)
    if hasattr(n_capacity, 'lines') and 's_nom_opt' in n_capacity.lines.columns:
        for line in n_capacity.lines.index:
            if line in n_dispatch.lines.index:
                n_dispatch.lines.loc[line, 's_nom'] = n_capacity.lines.loc[line, 's_nom_opt']
                n_dispatch.lines.loc[line, 's_nom_extendable'] = False
    
    # Fix link capacities (if extendable)
    if hasattr(n_capacity, 'links') and 'p_nom_opt' in n_capacity.links.columns:
        for link in n_capacity.links.index:
            if link in n_dispatch.links.index:
                n_dispatch.links.loc[link, 'p_nom'] = n_capacity.links.loc[link, 'p_nom_opt']
                n_dispatch.links.loc[link, 'p_nom_extendable'] = False

def update_dispatch_time_series(n_dispatch, n_capacity, year):
    """
    Update time series data for full year dispatch optimization
    
    This function reconstructs full 8760-hour time series from the clustered
    capacity expansion time series, or uses original full time series if available.
    """
    
    logging.info(f"Updating time series for dispatch year {year}")
    
    # If capacity expansion used TSAM clustering, we need to reconstruct full time series
    # For now, we'll use a simple approach - you may want to enhance this
    
    year_snapshots = n_dispatch.snapshots
    
    # Load time series
    if not n_capacity.loads_t.p_set.empty:
        if n_capacity.multi_invest:
            # Extract year data from multi-investment snapshots
            capacity_year_data = n_capacity.loads_t.p_set.loc[year]
            
            # Simple reconstruction: repeat/interpolate to full year
            if len(capacity_year_data) != len(year_snapshots):
                # Use interpolation to go from clustered to full time series
                full_load = reconstruct_time_series(capacity_year_data, year_snapshots)
            else:
                full_load = capacity_year_data.copy()
                full_load.index = year_snapshots
        else:
            # Single year capacity expansion
            full_load = n_capacity.loads_t.p_set.copy()
            full_load.index = year_snapshots
        
        n_dispatch.loads_t.p_set = full_load
    
    # Generator availability time series
    if not n_capacity.generators_t.p_max_pu.empty:
        if n_capacity.multi_invest:
            capacity_gen_data = n_capacity.generators_t.p_max_pu.loc[year]
            
            if len(capacity_gen_data) != len(year_snapshots):
                full_gen_pu = reconstruct_time_series(capacity_gen_data, year_snapshots)
            else:
                full_gen_pu = capacity_gen_data.copy()
                full_gen_pu.index = year_snapshots
        else:
            full_gen_pu = n_capacity.generators_t.p_max_pu.copy()
            full_gen_pu.index = year_snapshots
        
        n_dispatch.generators_t.p_max_pu = full_gen_pu
    
    # Storage inflow time series
    if not n_capacity.storage_units_t.inflow.empty:
        if n_capacity.multi_invest:
            capacity_inflow_data = n_capacity.storage_units_t.inflow.loc[year]
            
            if len(capacity_inflow_data) != len(year_snapshots):
                full_inflow = reconstruct_time_series(capacity_inflow_data, year_snapshots)
            else:
                full_inflow = capacity_inflow_data.copy()
                full_inflow.index = year_snapshots
        else:
            full_inflow = n_capacity.storage_units_t.inflow.copy()
            full_inflow.index = year_snapshots
        
        n_dispatch.storage_units_t.inflow = full_inflow
    
    logging.info(f"Time series updated for {len(year_snapshots)} snapshots")

def reconstruct_time_series(clustered_data, target_snapshots):
    """
    Reconstruct full time series from clustered TSAM data
    Maybe to delete?
    
    This is a simplified reconstruction - you may want to implement
    a more sophisticated method that uses the original TSAM mapping.
    """
    
    if clustered_data.empty:
        return pd.DataFrame(index=target_snapshots, columns=clustered_data.columns)
    
    # Simple approach: interpolate clustered data to full time series
    # Create mapping from clustered time points to full year
    clustered_length = len(clustered_data)
    target_length = len(target_snapshots)
    
    if clustered_length == target_length:
        # No clustering was used
        result = clustered_data.copy()
        result.index = target_snapshots
        return result
    
    # Create interpolation indices
    clustered_indices = np.linspace(0, target_length - 1, clustered_length)
    target_indices = np.arange(target_length)
    
    # Interpolate each column
    result = pd.DataFrame(index=target_snapshots, columns=clustered_data.columns)
    
    for col in clustered_data.columns:
        result[col] = np.interp(target_indices, clustered_indices, clustered_data[col].values)
    
    return result

def reconstruct_full_chronology_from_tsam(n_clustered, target_snapshots=None):
    """
    Reconstruct full chronological time series from TSAM-clustered network
    
    Parameters:
    -----------
    n_clustered : pypsa.Network
        TSAM-clustered network with stored time mapping
    target_snapshots : pd.DatetimeIndex, optional
        Target snapshots to reconstruct. If None, uses original snapshots
    
    Returns:
    --------
    dict: Dictionary with reconstructed time series data
    """
    
    if not hasattr(n_clustered, '_tsam_time_mapping'):
        raise ValueError("Network doesn't have TSAM time mapping. Use apply_tsam_to_pypsa_network_with_mapping()")
    
    time_mapping = n_clustered._tsam_time_mapping
    original_snapshots = n_clustered._tsam_original_snapshots
    
    if target_snapshots is None:
        target_snapshots = original_snapshots
    
    print(f"🔄 Reconstructing full chronology from TSAM data...")
    print(f"   Target snapshots: {len(target_snapshots)}")
    
    reconstructed_data = {}
    
    # Reconstruct time series for each component
    for component_name in ['loads_t', 'generators_t', 'storage_units_t']:
        if hasattr(n_clustered, component_name):
            component_t = getattr(n_clustered, component_name)
            reconstructed_data[component_name] = {}
            
            for attr_name in ['p_set', 'p_max_pu', 'p_min_pu', 'inflow']:
                if hasattr(component_t, attr_name):
                    clustered_ts = getattr(component_t, attr_name)
                    
                    if not clustered_ts.empty:
                        # Reconstruct using time mapping
                        reconstructed_ts = pd.DataFrame(
                            index=target_snapshots,
                            columns=clustered_ts.columns
                        )
                        
                        for target_time in target_snapshots:
                            if target_time in time_mapping:
                                representative_time = time_mapping[target_time]
                                if representative_time in clustered_ts.index:
                                    reconstructed_ts.loc[target_time] = clustered_ts.loc[representative_time]
                        
                        # Fill any missing values
                        reconstructed_ts = reconstructed_ts.fillna(method='ffill').fillna(method='bfill').fillna(0)
                        
                        reconstructed_data[component_name][attr_name] = reconstructed_ts
    
    print(f"✅ Chronological reconstruction complete")
    return reconstructed_data


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake(
            "create_dispatch_networks",
            **{
                "scenario": "TEST",
                "dispatch_years": [2025, 2030, 2040, 2050],

            }
        )
    
    logging.basicConfig(level=logging.INFO)
    
    # Create dispatch networks
    create_dispatch_networks(
        capacity_network_path=snakemake.input.capacity_network,
        output_paths=snakemake.output
    )