# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Enhanced solve_network_dispatch.py with Rolling Horizon Sequential Dispatch

Supports both single dispatch optimization and rolling horizon sequential dispatch
for PLEXOS-style production cost modeling.
"""

import logging
import pandas as pd
import numpy as np
import pypsa
from xarray import DataArray
import os
from pathlib import Path
from tqdm import tqdm


logger = logging.getLogger(__name__)

from _helpers import (
    add_missing_carriers,
    convert_cost_units,
    load_disaggregate, 
    map_component_parameters, 
    read_and_filter_generators,
    remove_leap_day,
    drop_non_pypsa_attrs,
    normed,
    get_start_year,
    get_snapshots,
    get_investment_periods,
    adjust_by_p_max_pu,
    apply_default_attr,
    single_year_network_copy,
    load_scenario_definition
)

from prepare_and_solve_network import (
    set_operational_limits,
    ccgt_steam_constraints,
    solve_network
)

from pypsa.descriptors import get_switchable_as_dense as get_as_dense, get_activity_mask

# =============================================================================
# ROLLING HORIZON SEQUENTIAL DISPATCH IMPLEMENTATION
# =============================================================================

def run_sequential_dispatch(n, scenario_setup, snakemake):
    """
    Main sequential dispatch function - runs daily rolling horizon optimization
    
    Parameters:
    -----------
    n_capacity : pypsa.Network
        Network with optimized capacities from capacity expansion
    dispatch_years : list
        Years to run sequential dispatch for
    scenario_setup : dict
        Scenario configuration
    snakemake : object
        Snakemake configuration object
        
    Returns:
    --------
    dict : Sequential dispatch results by year
    """
    
    all_results = {}
    year = n.snapshots[0][0]

   
    logger.info(f"🚀 Starting sequential dispatch for {year}")
    
    # Run sequential daily optimization with rolling horizon
    year_results = run_daily_rolling_horizon(n, year, scenario_setup, snakemake)
    all_results[year] = year_results
    
    logger.info(f"✅ Completed sequential dispatch for {year}")
    
    return all_results

def create_yearly_dispatch_network(n_capacity, year, scenario_setup):
    """
    Create yearly dispatch network from capacity expansion results
    
    Parameters:
    -----------
    n_capacity : pypsa.Network
        Network with capacity expansion results
    year : int
        Target year for dispatch
    scenario_setup : dict
        Scenario configuration
        
    Returns:
    --------
    pypsa.Network : Yearly dispatch network
    """
    
    logger.info(f"Creating yearly dispatch network for {year}")
    
    # Create year snapshots (hourly for full year)
    year_start = pd.Timestamp(f'{year}-01-01 00:00:00')
    year_end = pd.Timestamp(f'{year}-12-31 23:00:00') 
    year_snapshots = pd.date_range(year_start, year_end, freq='H')
    
    # Remove leap day for consistency
    year_snapshots = year_snapshots[~((year_snapshots.month == 2) & (year_snapshots.day == 29))]
    
    logger.info(f"Created {len(year_snapshots)} hourly snapshots for {year}")
    
    # Use existing infrastructure to create yearly network
    if hasattr(n_capacity, '_tsam_original_snapshots'):
        # Handle TSAM-clustered source network
        logger.info("Source network is TSAM-clustered, reconstructing full chronology")
        n_dispatch = single_year_network_copy_with_tsam_reconstruction(
            n_capacity, snapshots=year_snapshots
        )
    else:
        # Regular network copy
        logger.info("Creating dispatch network from regular capacity network")
        n_dispatch = single_year_network_copy(
            n_capacity, snapshots=year_snapshots
        )
    
    # Apply capacity optimization results to dispatch network
    apply_capacity_results_to_dispatch(n_dispatch, n_capacity, year)
    
    logger.info(f"✅ Yearly dispatch network created with {len(n_dispatch.generators)} generators")
    
    return n_dispatch

def run_daily_rolling_horizon(n, year, scenario_setup, snakemake):
    """
    Core rolling horizon implementation - daily optimization with lookahead
    
    Parameters:
    -----------
    n : pypsa.Network
        Yearly dispatch network
    year : int
        Dispatch year
    scenario_setup : dict
        Scenario configuration
    snakemake : object
        Snakemake configuration
        
    Returns:
    --------
    dict : Daily dispatch results
    """
    
    # Get configuration
    config = snakemake.config.get("sequential_dispatch", {})
    lookahead_hours = config.get("rolling_horizon", {}).get("lookahead_hours", 24)
    
    logger.info(f"Starting rolling horizon dispatch with {lookahead_hours}h lookahead")
    
    daily_results = {}
    storage_states = initialize_storage_states(n)


    
    # Get daily date range for the year
    start_date = pd.Timestamp(f'{year}-01-01')
    horizon_length = config.get("horizon_length", 365) - 1 # length of horizon, accounting for the fact that the start date is a full step already
    end_date = start_date + pd.Timedelta(f'{horizon_length}D')
    
    
    daily_dates = pd.date_range(start_date, end_date, freq='D')
    
    total_days = len(daily_dates)
    logger.info(f"Processing {total_days} days for {year}")
    
    # Initialize progress bar
    progress_bar = tqdm(
        daily_dates, 
        desc=f"Sequential Dispatch {year}", 
        unit="day",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    successful_days = 0
    for current_day in progress_bar:
        
        day_str = current_day.strftime('%Y-%m-%d')
        day_str = current_day.strftime('%Y-%m-%d')
        
        try:
            # Create daily network with lookahead window
            daily_network = create_daily_network_with_lookahead(
                n, current_day, lookahead_hours, scenario_setup
            )
            
            # Set storage initial conditions from previous day
            set_storage_initial_conditions(daily_network, storage_states)
            
            # Optimize daily network
            solve_daily_dispatch(daily_network, scenario_setup, snakemake)
            
            # Extract and store results for actual day (not lookahead period)
            daily_results[current_day] = extract_daily_results(
                daily_network, current_day, lookahead_hours
            )
            
            # Update storage states for next day
            storage_states = extract_end_of_day_storage_states(
                daily_network, current_day
            )

            successful_days += 1
            
        except Exception as e:
            logger.error(f"❌ Error optimizing day {day_str}: {e}")
            # Continue with next day but log the error
            continue

    
    progress_bar.close()
    logger.info(f"✅ Completed: {successful_days}/{len(daily_dates)} successful days")
    
    
    return daily_results

def create_daily_network_with_lookahead(n, current_day, lookahead_hours, scenario_setup):
    """
    Create daily network with lookahead window for optimization
    
    Parameters:
    -----------
    n : pypsa.Network
        Source yearly network
    current_day : pd.Timestamp
        Current day to optimize
    lookahead_hours : int
        Hours of lookahead for optimization
    scenario_setup : dict
        Scenario configuration
        
    Returns:
    --------
    pypsa.Network : Daily network with lookahead
    """
    
    # Define time window: current day + lookahead
    year = current_day.year
    day_start = current_day
    day_end = current_day + pd.Timedelta(days=1)
    lookahead_end = day_end + pd.Timedelta(hours=lookahead_hours)

    
    # Create snapshots for optimization window matching the snapshot format of MuiltiIndex (year, datetime)
    optimization_snapshots = pd.date_range(day_start, lookahead_end, freq='H')[:-1]

    if isinstance(n.snapshots, pd.MultiIndex):
        optimization_snapshots = pd.MultiIndex.from_arrays([[year] * len(optimization_snapshots), optimization_snapshots], names=['year', 'datetime']) 
        optimization_snapshots.name = 'snapshot'

        day_end = (year, day_end)
        lookahead_end = (year, lookahead_end)
    


    # Ensure snapshots exist in source network
    available_snapshots = optimization_snapshots.intersection(n.snapshots)
    if len(available_snapshots) < len(optimization_snapshots):
        # Truncate if we're at year end
        optimization_snapshots = available_snapshots
    
    # Create daily network copy
    daily_network = n.copy(with_time=False)
    daily_network.set_snapshots(optimization_snapshots)
    
    # Copy time-series data for optimization window
    copy_timeseries_for_window(daily_network, n, optimization_snapshots)
    
    # Set snapshot weightings (only weight actual day, not lookahead)
    actual_day_snapshots = optimization_snapshots[optimization_snapshots < day_end]
    daily_weightings = pd.Series(0.0, index=optimization_snapshots)
    daily_weightings.loc[actual_day_snapshots] = 1.0  # Weight actual day only
    
    # Set weightings for all required attributes
    if hasattr(daily_network, 'snapshot_weightings'):
        if isinstance(daily_network.snapshot_weightings, pd.DataFrame):
            daily_network.snapshot_weightings = pd.DataFrame(
                index=optimization_snapshots,
                data={
                    'objective': daily_weightings,
                    'generators': daily_weightings,
                    'stores': daily_weightings
                }
            )
        else:
            daily_network.snapshot_weightings = daily_weightings
    else:
        daily_network.snapshot_weightings = daily_weightings
    
    return daily_network

def copy_timeseries_for_window(daily_network, source_network, optimization_snapshots):
    """
    Copy time-series data for optimization window from source network
    
    Parameters:
    -----------
    daily_network : pypsa.Network
        Target daily network
    source_network : pypsa.Network
        Source yearly network
    optimization_snapshots : pd.DatetimeIndex
        Time window for optimization
    """
    
    # Copy load profiles
    if not source_network.loads_t.p_set.empty:
        daily_network.loads_t.p_set = source_network.loads_t.p_set.loc[optimization_snapshots].copy()
    
    # Copy generator availability profiles
    if not source_network.generators_t.p_max_pu.empty:
        daily_network.generators_t.p_max_pu = source_network.generators_t.p_max_pu.loc[optimization_snapshots].copy()
    
    if not source_network.generators_t.p_min_pu.empty:
        daily_network.generators_t.p_min_pu = source_network.generators_t.p_min_pu.loc[optimization_snapshots].copy()
    
    # Copy storage inflow profiles if any
    if not source_network.storage_units_t.inflow.empty:
        daily_network.storage_units_t.inflow = source_network.storage_units_t.inflow.loc[optimization_snapshots].copy()
    
    # Copy any other time-series data
    for attr in ['marginal_cost']:
        if hasattr(source_network.generators_t, attr):
            ts_data = getattr(source_network.generators_t, attr)
            if not ts_data.empty:
                setattr(daily_network.generators_t, attr, ts_data.loc[optimization_snapshots].copy())

def solve_daily_dispatch(daily_network, scenario_setup, snakemake):
    """
    Solve daily dispatch using config-driven solver settings
    
    Parameters:
    -----------
    daily_network : pypsa.Network
        Daily network to optimize
    scenario_setup : dict
        Scenario configuration
    snakemake : object
        Snakemake configuration
    """
    
    # Get sequential dispatch configuration
    seq_config = snakemake.config.get("sequential_dispatch", {})
    opt_config = seq_config.get("optimization", {})
    rolling_config = seq_config.get("rolling_horizon", {})
    
    # Determine solver settings
    solver_name = opt_config.get("solver", "highs")
    solver_profile = opt_config.get("solver_profile", "highs-default")
    enable_uc = rolling_config.get("unit_commitment", True)
    
    # Get solver options from config
    solving_config = snakemake.config.get("solving", {})
    all_solver_options = solving_config.get("solver_options", {})
    
    if solver_profile not in all_solver_options:
        logger.warning(f"Solver profile '{solver_profile}' not found, using highs-default")
        solver_profile = "highs-default"
    
    solver_options = all_solver_options.get(solver_profile, {})
    
    def extra_functionality(n, snapshots):
        """Add daily dispatch constraints"""
        if enable_uc:
            try:
                committable_config = snakemake.config.get("electricity", {}).get("dispatch_committable_carriers", {})
                if committable_config:
                    p_max_pu = set_existing_committable(n, snapshots, scenario_setup, committable_config)
                    set_max_status(n, snapshots, p_max_pu)
            except Exception as e:
                logger.warning(f"Unit commitment constraints failed: {e}")
        
        try:
            set_operational_limits(n, snapshots, scenario_setup)
            ccgt_steam_constraints(n, snapshots, snakemake)
        except Exception as e:
            logger.warning(f"Operational constraints failed: {e}")
        
        try:
            param = load_extendable_parameters(n, scenario_setup, snakemake)
            annual_co2_constraints(n, snapshots, param, scenario_setup)
        except Exception as e:
            pass
    
    # Solve with specified solver configuration
    daily_network.optimize(
        snapshots=daily_network.snapshots,
        multi_investment_periods=False,
        solver_name=solver_name,
        solver_options=solver_options,
        extra_functionality=extra_functionality,
        linearized_unit_commitment=enable_uc
    )

def initialize_storage_states(n):
    """
    Initialize storage states for start of sequential dispatch
    
    Parameters:
    -----------
    n : pypsa.Network
        Network with storage units
        
    Returns:
    --------
    dict : Initial storage states
    """
    
    storage_states = {}
    
    if len(n.storage_units) > 0:
        logger.info(f"Initializing {len(n.storage_units)} storage units")
        
        for storage in n.storage_units.index:
            # Initialize at 50% state of charge
            p_nom = n.storage_units.loc[storage, 'p_nom']
            max_hours = n.storage_units.loc[storage, 'max_hours']
            max_energy = p_nom * max_hours
            storage_states[storage] = max_energy * 0.5
            
            logger.debug(f"  {storage}: Initial SOC = {storage_states[storage]:.1f} MWh")
    
    return storage_states

def set_storage_initial_conditions(daily_network, storage_states):
    """
    Set storage initial conditions from previous day results
    
    Parameters:
    -----------
    daily_network : pypsa.Network
        Daily network
    storage_states : dict
        Storage states from previous day
    """
    
    for storage, initial_soc in storage_states.items():
        if storage in daily_network.storage_units.index:
            daily_network.storage_units.loc[storage, 'state_of_charge_initial'] = initial_soc

def extract_daily_results(daily_network, current_day, lookahead_hours):
    """
    Extract results for actual day (excluding lookahead period)
    
    Parameters:
    -----------
    daily_network : pypsa.Network
        Optimized daily network
    current_day : pd.Timestamp
        Current day
    lookahead_hours : int
        Lookahead hours to exclude
        
    Returns:
    --------
    dict : Daily results
    """
    
    # Get snapshots for actual day only
    day_end = current_day + pd.Timedelta(days=1)
    if isinstance(daily_network.snapshots, pd.MultiIndex):
        day_end = (current_day.year, day_end)
        
    actual_day_snapshots = daily_network.snapshots[
        daily_network.snapshots < day_end
    ]
    
    # Extract relevant results
    daily_results = {
        'snapshots': actual_day_snapshots,
        'objective_value': getattr(daily_network, 'objective', 0),
        'status': getattr(daily_network, 'optimization_status', 'unknown')
    }
    
    # Extract generator dispatch
    if not daily_network.generators_t.p.empty:
        daily_results['generators_p'] = daily_network.generators_t.p.loc[actual_day_snapshots].copy()
    
    # Extract storage dispatch and SOC
    if not daily_network.storage_units_t.p.empty:
        daily_results['storage_p'] = daily_network.storage_units_t.p.loc[actual_day_snapshots].copy()
    
    if not daily_network.storage_units_t.state_of_charge.empty:
        daily_results['storage_soc'] = daily_network.storage_units_t.state_of_charge.loc[actual_day_snapshots].copy()
    
    # Extract load (for validation)
    if not daily_network.loads_t.p.empty:
        daily_results['loads_p'] = daily_network.loads_t.p.loc[actual_day_snapshots].copy()
    
    # Extract marginal prices
    if not daily_network.buses_t.marginal_price.empty:
        daily_results['marginal_prices'] = daily_network.buses_t.marginal_price.loc[actual_day_snapshots].copy()
    
    return daily_results

def extract_end_of_day_storage_states(daily_network, current_day):
    """
    Extract storage states at end of day for next day initialization
    
    Parameters:
    -----------
    daily_network : pypsa.Network
        Optimized daily network
    current_day : pd.Timestamp
        Current day
        
    Returns:
    --------
    dict : Storage states at end of day
    """
    
    
    day_end_snapshot = current_day + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
    if isinstance(daily_network.snapshots, pd.MultiIndex):
        day_end_snapshot = (current_day.year, day_end_snapshot)
    
    storage_states = {}
    if len(daily_network.storage_units) > 0 and not daily_network.storage_units_t.state_of_charge.empty:
        for storage in daily_network.storage_units.index:
            if storage in daily_network.storage_units_t.state_of_charge.columns:
                try:
                    storage_states[storage] = daily_network.storage_units_t.state_of_charge.loc[day_end_snapshot, storage]
                except KeyError:
                    # If exact snapshot not available, use last available
                    available_snapshots = daily_network.storage_units_t.state_of_charge.index
                    last_snapshot = available_snapshots[available_snapshots <= day_end_snapshot][-1]
                    storage_states[storage] = daily_network.storage_units_t.state_of_charge.loc[last_snapshot, storage]
    
    return storage_states

def apply_capacity_results_to_dispatch(n_dispatch, n_capacity, year):
    """
    Apply optimized capacities from capacity expansion to dispatch network
    
    Parameters:
    -----------
    n_dispatch : pypsa.Network
        Dispatch network to modify
    n_capacity : pypsa.Network
        Capacity expansion network with results
    year : int
        Target year
    """
    
    logger.info("Applying capacity expansion results to dispatch network")
    
    # Apply generator capacities
    generators_updated = 0
    for gen in n_dispatch.generators.index:
        if gen in n_capacity.generators.index:
            # Use optimized capacity if available, otherwise use original
            if 'p_nom_opt' in n_capacity.generators.columns:
                optimized_capacity = n_capacity.generators.loc[gen, 'p_nom_opt']
                n_dispatch.generators.loc[gen, 'p_nom'] = optimized_capacity
                generators_updated += 1
            
            # Check if generator is active in this year (for multi-investment)
            if hasattr(n_capacity, 'investment_periods'):
                build_year = n_capacity.generators.loc[gen, 'build_year']
                lifetime = n_capacity.generators.loc[gen, 'lifetime']
                if year < build_year or year > build_year + lifetime:
                    n_dispatch.generators.loc[gen, 'p_nom'] = 0
                    logger.debug(f"Generator {gen} inactive in {year} (build: {build_year}, lifetime: {lifetime})")
    
    # Apply storage unit capacities
    storage_updated = 0
    for storage in n_dispatch.storage_units.index:
        if storage in n_capacity.storage_units.index:
            if 'p_nom_opt' in n_capacity.storage_units.columns:
                optimized_capacity = n_capacity.storage_units.loc[storage, 'p_nom_opt']
                n_dispatch.storage_units.loc[storage, 'p_nom'] = optimized_capacity
                storage_updated += 1
    
    logger.info(f"Updated {generators_updated} generators and {storage_updated} storage units with optimized capacities")

# =============================================================================
# MAIN SOLVE_NETWORK_DISPATCH FUNCTION
# =============================================================================

def solve_network_dispatch(n, sns, enable_unit_commitment=False, sequential_mode=False):
    """
    Main solve function supporting both single and sequential dispatch
    
    Parameters:
    -----------
    n : pypsa.Network
        Network to optimize
    sns : pd.DatetimeIndex
        Snapshots to optimize
    enable_unit_commitment : bool
        Enable unit commitment constraints
    sequential_mode : bool
        Flag indicating if called from sequential dispatch (prevents recursion)
        
    Returns:
    --------
    dict or None : Results (for sequential mode) or None (for regular mode)
    """
    
    if sequential_mode:
        logger.info("🔄 Sequential dispatch enabled - running rolling horizon optimization")
        return run_sequential_dispatch_workflow(n, sns, snakemake)
    else:
        # Run single optimization (existing functionality)
        logger.info("🔧 Running single dispatch optimization")
        return run_single_optimization(n, sns, enable_unit_commitment)

def run_sequential_dispatch_workflow(n, sns, snakemake):
    """
    Main workflow for sequential dispatch
    
    Parameters:
    -----------
    n : pypsa.Network
        Network (should be capacity expansion results)
    sns : pd.DatetimeIndex
        Snapshots
    snakemake : object
        Snakemake configuration
        
    Returns:
    --------
    dict : Sequential dispatch results
    """
    
    scenario_setup = load_scenario_definition(snakemake)
    
    # Determine dispatch years
    if hasattr(n, 'investment_periods'):
        dispatch_years = n.investment_periods
        logger.info(f"Multi-investment network: dispatching years {dispatch_years}")
    else:
        dispatch_years = [sns[0].year]
        logger.info(f"Single-year network: dispatching year {dispatch_years[0]}")
    
    # Run sequential dispatch
    results = run_sequential_dispatch(n, scenario_setup, snakemake)
    
    # Export results
    export_sequential_results(results, snakemake)
    
    return results

def run_single_optimization(n, sns, enable_unit_commitment):
    """
    Run single optimization (original functionality)
    
    Parameters:
    -----------
    n : pypsa.Network
        Network to optimize
    sns : pd.DatetimeIndex
        Snapshots
    enable_unit_commitment : bool
        Enable unit commitment
    export_to_Sienna : bool
        Export to Sienna format
    """
    
    def extra_functionality(n, snapshots):
        """Add constraints for dispatch optimization"""

        if enable_unit_commitment:
            config = snakemake.config["electricity"]["dispatch_committable_carriers"]
            p_max_pu = set_existing_committable(n, snapshots, scenario_setup, config)
            set_max_status(n, snapshots, p_max_pu)

        set_operational_limits(n, snapshots, scenario_setup)
        ccgt_steam_constraints(n, snapshots, snakemake)
        
        param = load_extendable_parameters(n, scenario_setup, snakemake)
        annual_co2_constraints(n, snapshots, param, scenario_setup)
    
        solver_config = snakemake.config["solving"]
        solver_name = solver_config['solver']["name"]
        solver_options = solver_config["solver_options"][solver_config['solver'].get("options", {})]

        logger.info(f"Solving with {solver_name}")
        
        n.optimize(
            snapshots=sns,
            multi_investment_periods=n.multi_invest,
            solver_name=solver_name,
            solver_options=solver_options,
            extra_functionality=extra_functionality,
            linearized_unit_commitment=enable_unit_commitment
        )

        return n.results
    
def export_sequential_results(results, snakemake):
    """
    Export sequential dispatch results to files
    
    Parameters:
    -----------
    results : dict
        Sequential dispatch results by year
    snakemake : object
        Snakemake configuration with output paths
    """
    
    logger.info("📤 Exporting sequential dispatch results")
    
    # Create output directory
    output_dir = Path(snakemake.output.dispatch_results).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Aggregate results across all years
    aggregated_results = aggregate_sequential_results(results)
    
    # Export aggregated network (if possible)
    try:
        if hasattr(snakemake, 'output') and len(snakemake.output) > 0:
            # Save aggregated results as netcdf
            aggregated_results.to_netcdf(snakemake.output[0])
            logger.info(f"✅ Exported aggregated results to {snakemake.output[0]}")
    except Exception as e:
        logger.warning(f"Could not export aggregated results: {e}")
    
    # Export daily summaries
    try:
        daily_summaries = create_daily_summaries(results)
        summary_path = output_dir / "daily_summaries.csv"
        daily_summaries.to_csv(summary_path)
        logger.info(f"✅ Exported daily summaries to {summary_path}")
    except Exception as e:
        logger.warning(f"Could not export daily summaries: {e}")

def aggregate_sequential_results(results):
    """
    Aggregate sequential dispatch results into a single network-like structure
    
    Parameters:
    -----------
    results : dict
        Sequential dispatch results by year
        
    Returns:
    --------
    pypsa.Network or dict : Aggregated results
    """
    
    # For now, return a summary dictionary
    # In the future, this could reconstruct a full network
    
    summary = {
        'years_processed': list(results.keys()),
        'total_days': sum(len(year_results) for year_results in results.values()),
        'successful_optimizations': 0,
        'failed_optimizations': 0
    }
    
    for year, year_results in results.items():
        for day, day_results in year_results.items():
            if day_results.get('status') == 'optimal':
                summary['successful_optimizations'] += 1
            else:
                summary['failed_optimizations'] += 1
    
    return summary

def create_daily_summaries(results):
    """
    Create daily summary statistics from sequential dispatch results
    
    Parameters:
    -----------
    results : dict
        Sequential dispatch results
        
    Returns:
    --------
    pd.DataFrame : Daily summaries
    """
    
    summaries = []
    
    for year, year_results in results.items():
        for day, day_results in year_results.items():
            summary = {
                'year': year,
                'date': day,
                'status': day_results.get('status', 'unknown'),
                'objective_value': day_results.get('objective_value', np.nan)
            }
            
            # Add generation summaries
            if 'generators_p' in day_results:
                gen_p = day_results['generators_p']
                summary['total_generation'] = gen_p.sum().sum()
                summary['peak_generation'] = gen_p.sum(axis=1).max()
            
            # Add load summaries
            if 'loads_p' in day_results:
                load_p = day_results['loads_p']
                summary['total_load'] = load_p.sum().sum()
                summary['peak_load'] = load_p.sum(axis=1).max()
            
            summaries.append(summary)
    
    return pd.DataFrame(summaries)

# =============================================================================
# EXISTING FUNCTIONS (keep your original implementations)
# =============================================================================

def get_min_stable_level(n, model_file, model_setup, existing_carriers, extended_carriers):
    
    existing_param = pd.read_excel(
        model_file, 
        sheet_name="fixed_conventional",
        na_values=["-"],
        index_col=[0,1]
    ).loc[model_setup["fixed_conventional"]]
    
    existing_gens = n.generators.query("carrier in @existing_carriers & p_nom_extendable == False").index
    existing_msl= existing_param.loc[existing_gens, "Min Stable Level (%)"].rename("p_min_pu")
    
    extended_param = pd.read_excel(
        model_file, 
        sheet_name = "extendable_parameters",
        index_col = [0,2,1],
    ).sort_index().loc[model_setup["extendable_parameters"]]

    extended_gens = n.generators.query("carrier in @extended_carriers & p_nom_extendable").index
    extended_msl = pd.Series(index=extended_gens, name = "p_min_pu")
    for g in extended_gens:
        carrier = g.split("-")[1]
        y = int(g.split("-")[2])
        if y in extended_param.columns:
            extended_msl[g] = extended_param.loc[("min_stable_level", carrier), y].astype(float)
        else:
            interp_data = extended_param.loc[("min_stable_level", carrier), :].drop(["unit", "source"]).astype(float)
            interp_data = interp_data.append(pd.Series(index=[y], data=[np.nan])).interpolate()
            extended_msl[g] = interp_data.loc[y]

    return existing_msl, extended_msl


def set_max_status(n, sns, p_max_pu):

    # init period = 100h to let model stabilise status
    if sns[0] == n.snapshots[0]:
        init_periods=100
        n.generators_t.p_max_pu.loc[
            sns[:init_periods], p_max_pu.columns
        ] = p_max_pu.loc[sns[:init_periods], :].values
        
        n.generators_t.p_min_pu.loc[:,p_max_pu.columns] = get_as_dense(n, "Generator", "p_min_pu").loc[:,p_max_pu.columns]
        n.generators_t.p_min_pu.loc[
            sns[:init_periods], p_max_pu.columns
        ] = 0
        sns = sns[init_periods:]

    active = get_activity_mask(n, "Generator", sns, p_max_pu.columns)
    active.rename_axis("Generator-com", axis = 1, inplace = True)
    p_max_pu = p_max_pu.loc[sns, active.any(axis=0)]
    p_max_pu = p_max_pu.loc[sns, (p_max_pu != 1).any(axis=0)]

    status = n.model.variables["Generator-status"].sel({"Generator-com":p_max_pu.columns})
    lhs = status.sel(snapshot=sns)
    if p_max_pu.columns.name != "Generator-com":
        p_max_pu.columns.name = "Generator-com"
    rhs = DataArray(p_max_pu)
    
    n.model.add_constraints(lhs, "<=", rhs, name="max_status")

def set_upper_combined_status_bus(n, sns, p_max_pu):

    active = get_activity_mask(n, "Generator", sns, p_max_pu.columns)
    active.rename_axis("Generator-com", axis = 1, inplace = True)
    p_max_pu = p_max_pu.loc[sns, active.any(axis=0)]
    p_max_pu = p_max_pu.loc[:, (p_max_pu != 1).any(axis=0)]

    for bus_i in n.buses.index:
        bus_gens = n.generators.query("bus == @bus_i").index.intersection(p_max_pu.columns)
        if len(bus_gens) >= 0: 
            p_nom = n.generators.loc[bus_gens, "p_nom"]
            p_nom.name = "Generator-com"
            status = n.model.variables["Generator-status"].sel({"snapshot":sns, "Generator-com":bus_gens})

            p_nom_df = pd.DataFrame(index = sns, columns = p_nom.index)        
            p_nom_df.loc[:] = p_nom.values
            p_nom_df.rename_axis("Generator-com", axis = 1, inplace = True)

            active.columns.name = "Generator-com"
            lhs = (DataArray(p_nom_df) * status).sum("Generator-com")
            rhs = (p_nom * p_max_pu[bus_gens]).sum(axis=1)
            
            n.model.add_constraints(lhs, "<=", rhs, name=f"{bus_i}-max_status")


def set_upper_avg_status_over_sns(n, sns, p_max_pu):
    
    active = get_activity_mask(n, "Generator", sns, p_max_pu.columns)
    active.rename_axis("Generator-com", axis = 1, inplace = True)
    p_max_pu = p_max_pu.loc[sns, active.any(axis=0)]
    p_max_pu = p_max_pu.loc[:, (p_max_pu != 1).any(axis=0)]

    weightings = pd.DataFrame(index = sns, columns = p_max_pu.columns)
    weight_values = n.snapshot_weightings.generators.loc[sns].values.reshape(-1, 1)
    weightings.loc[:] = weight_values
    weightings.rename_axis("Generator-com", axis = 1, inplace = True)

    status = n.model.variables["Generator-status"].sel({"Generator-com":p_max_pu.columns, "snapshot":sns})
    lhs = (status * weightings).sum("snapshot")
    if p_max_pu.columns.name != "Generator-com":
        p_max_pu.columns.name = "Generator-com"
    rhs = (weightings * p_max_pu).sum()

    n.model.add_constraints(lhs, "<=", rhs, name="upper_avg_status_sns")

def set_max_status4(n, sns, p_max_pu):
    
    # init period = 100h to let model stabilise status
    # if sns[0] == n.snapshots[0]:
    #     init_periods=100
    #     n.generators_t.p_max_pu.loc[
    #         sns[:init_periods], p_max_pu.columns
    #     ] = p_max_pu.loc[sns[:init_periods], :].values
        
    #     n.generators_t.p_min_pu.loc[:,p_max_pu.columns] = get_as_dense(n, "Generator", "p_min_pu").loc[:,p_max_pu.columns]
    #     n.generators_t.p_min_pu.loc[
    #         sns[:init_periods], p_max_pu.columns
    #     ] = 0
    #     sns = sns[init_periods:]

    active = get_activity_mask(n, "Generator", sns, p_max_pu.columns)
    p_max_pu = p_max_pu.loc[sns, active.any(axis=0)]

    active.columns.name = "Generator-com"
    status = n.model.variables["Generator-status"].sel({"Generator-com":p_max_pu.columns})
    lhs = status.sel(snapshot=sns).groupby("snapshot.week").sum()
    if p_max_pu.columns.name != "Generator-com":
        p_max_pu.columns.name = "Generator-com"
    rhs = p_max_pu.groupby(p_max_pu.index.isocalendar().week).sum()
    
    n.model.add_constraints(lhs, "<=", rhs, name="max_status")

def set_existing_committable(n, sns, model_file, model_setup, config):

    existing_carriers = config['existing']
    existing_gen = n.generators.query("carrier in @existing_carriers & p_nom_extendable == False").index.to_list()

    extended_carriers = config['extended']
    extended_gen = n.generators.query("carrier in @extended_carriers & p_nom_extendable").index.to_list()
    
    n.generators.loc[existing_gen + extended_gen, "committable"] = True

    p_max_pu = get_as_dense(n, "Generator", "p_max_pu", sns)[existing_gen + extended_gen].copy()
    n.generators_t.p_max_pu.loc[:, existing_gen + extended_gen] = 1
    n.generators.loc[existing_gen + extended_gen, "p_max_pu"] = 1

    existing_msl, extended_msl = get_min_stable_level(n, model_file, model_setup, existing_carriers, extended_carriers)

    n.generators.loc[existing_gen, "p_min_pu"] = existing_msl
    n.generators.loc[extended_gen, "p_min_pu"] = extended_msl

    return p_max_pu

# Add other existing functions as needed...

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake(
            'solve_network_dispatch', 
            **{
                'scenario':'TEST',
                'year':2030
            }
        )

    # Load network (could be capacity expansion results or dispatch network)
    n = pypsa.Network(snakemake.input.dispatch_network)
    scenario_setup = load_scenario_definition(snakemake)
    config = snakemake.config.get("sequential_dispatch", {})

    # Check if we should run sequential dispatch instead
    sequential_mode = config.get("enable", False)
    enable_unit_commitment = config.get("enable_unit_commitment", False)
    
    # Run dispatch (will automatically choose sequential or single based on config)
    solve_network_dispatch(n, n.snapshots, enable_unit_commitment=enable_unit_commitment, sequential_mode=sequential_mode)

    logger.info("✅ Dispatch optimization completed")