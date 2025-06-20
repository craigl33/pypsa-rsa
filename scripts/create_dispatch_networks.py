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
from _helpers import load_scenario_definition

from prepare_and_solve_network import add_missing_carriers

def create_dispatch_networks(capacity_network_path, base_network_path, output_paths):
   """
   Updated version that handles TSAM-clustered capacity networks
   """
   
   logging.info(f"Loading capacity expansion network: {capacity_network_path}")
   scenario_setup = load_scenario_definition(snakemake)
   n_capacity = pypsa.Network(capacity_network_path)
   
   logging.info(f"Found {len(n_capacity.investment_periods)} investment periods: {n_capacity.investment_periods}")
   
   # Create dispatch network for each investment period
   for year in n_capacity.investment_periods:
       
       logging.info(f"Creating dispatch network for year {year}")
       n_dispatch = pypsa.Network(base_network_path)
       add_missing_carriers(n_dispatch)
    
       
       # Get snapshots for this year (full 8760 hours)
       year_snapshots = pd.date_range(
           start=f"{year}-01-01 00:00", 
           end=f"{year}-12-31 23:00", 
           freq="H"
       )
       # Remove leap day
       year_snapshots = year_snapshots[~((year_snapshots.month == 2) & (year_snapshots.day == 29))]


       # Fix capacities at optimized values for dispatch
       fix_optimal_capacities(n_dispatch, n_capacity, year)
    
       
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
        base_network_path=snakemake.input.network,
        output_paths=snakemake.output
    )