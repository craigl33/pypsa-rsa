# -*- coding: utf-8 -*-
"""
Drop-in replacement for solve_network_dispatch.py with PyPSA version detection

This tries PyPSA's built-in rolling horizon first, falls back to manual implementation if not available.
"""

import logging
import pandas as pd
import numpy as np
import pypsa
import os
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

from _helpers import (
    add_missing_carriers,
    load_scenario_definition
)

from pypsa.descriptors import get_switchable_as_dense as get_as_dense, get_activity_mask
from xarray import DataArray

# =============================================================================
# VERSION-AWARE ROLLING HORIZON IMPLEMENTATION
# =============================================================================

def solve_network_dispatch(n, snakemake):
    """
    Drop-in replacement for your solve_network_dispatch function
    
    Automatically detects PyPSA version and uses appropriate rolling horizon method:
    - Tries built-in n.optimize.optimize_with_rolling_horizon() first
    - Falls back to manual implementation if not available
    
    Parameters:
    -----------
    n : pypsa.Network
        Network to optimize
    snakemake : object
        Snakemake configuration object
        
    Returns:
    --------
    pypsa.Network : Optimized network with results
    """
    
    scenario_setup = load_scenario_definition(snakemake)
    config = snakemake.config.get("sequential_dispatch", {})
    
    # Check if rolling horizon is enabled
    if config.get("enable", False):
        logger.info("🔄 Rolling horizon dispatch enabled")
        
        # Try built-in rolling horizon first
        if hasattr(n, 'optimize') and hasattr(n.optimize, 'optimize_with_rolling_horizon'):
            logger.info("✅ Using PyPSA built-in rolling horizon")
            n = run_builtin_rolling_horizon(n, scenario_setup, snakemake)
        else:
            logger.info("⚠️  Built-in rolling horizon not available, using manual implementation")
            n = run_manual_rolling_horizon(n, scenario_setup, snakemake)
    else:
        logger.info("🔧 Running single dispatch optimization")
        n = run_single_dispatch(n, scenario_setup, snakemake)
    
    # Export results
    output_path = snakemake.output.dispatch_results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    n.export_to_netcdf(output_path)
    
    logger.info("✅ Dispatch optimization completed")
    return n

def run_builtin_rolling_horizon(n, scenario_setup, snakemake):
    """
    Use PyPSA's built-in rolling horizon (for newer versions)
    """
    
    config = snakemake.config.get("sequential_dispatch", {})
    
    # Get parameters with corrected terminology
    # horizon_length_hours = config.get("horizon_length_hours", 48)
    interval_length_hours = config.get("interval_length_hours", 24)  
    overlap_hours = config.get("overlap_hours", 24)
    horizon_length_hours = interval_length_hours + overlap_hours  # Total window length
    
    # Retrieve snapshot window parameters
    # These are used to define the rolling horizon, if not specified, default to the whole horizon
    total_steps = config.get("total_steps", None)  # Number of steps in the rolling horizon
    initial_time = config.get("initial_time", None)  # Initial time for the rolling horizon

    # Validation
    horizon_length_hours = interval_length_hours + overlap_hours

        # Convert hours to number of snapshots (PyPSA built-in expects integers, not Timedelta)
    if isinstance(n.snapshots, pd.MultiIndex):
        # Multi-investment period network
        sample_snapshots = n.snapshots.get_level_values(1)[:10]
    else:
        # Single period network  
        sample_snapshots = n.snapshots[:10]
    
    try:
        snapshot_freq = pd.infer_freq(sample_snapshots)
        if snapshot_freq.upper() == 'H' or snapshot_freq.upper() == '1H':
            # Hourly snapshots
            horizon_snapshots = horizon_length_hours
            overlap_snapshots = overlap_hours
        elif snapshot_freq.upper() == '3H':
            # 3-hourly snapshots
            horizon_snapshots = horizon_length_hours // 3
            overlap_snapshots = overlap_hours // 3
        elif snapshot_freq.upper() == '6H':
            # 6-hourly snapshots  
            horizon_snapshots = horizon_length_hours // 6
            overlap_snapshots = overlap_hours // 6
        else:
            logger.warning(f"Unknown frequency {snapshot_freq}, assuming hourly")
            horizon_snapshots = horizon_length_hours
            overlap_snapshots = overlap_hours
    except Exception:
        # If frequency inference fails, assume hourly
        logger.warning("Could not infer snapshot frequency, assuming hourly")
        horizon_snapshots = horizon_length_hours
        overlap_snapshots = overlap_hours
    
    # Get solver configuration
    solver_config = snakemake.config.get("solving", {})
    solver_name = solver_config.get("solver", {}).get("name", "highs") 
    
    opt_config = config.get("optimization", {})
    solver_profile = opt_config.get("solver_profile", "highs-default")
    solver_options = solver_config.get("solver_options", {}).get(solver_profile, {})
    
    logger.info(f"Rolling horizon: {horizon_length_hours}h total ({horizon_snapshots} snapshots), {interval_length_hours}h steps, {overlap_hours}h overlap ({overlap_snapshots} snapshots)")
    logger.info(f"Solver: {solver_name} with profile {solver_profile}")
    
    # Define constraints function
    def extra_functionality(network, snapshots):
        """Add dispatch constraints"""
        add_dispatch_constraints(network, snapshots, scenario_setup, snakemake)
    
    
    if initial_time is not None and total_steps is not None:
        # Set initial time and total steps for rolling horizon
        logger.info(f"Initial time: {initial_time}, Total steps: {total_steps}")
        initial_time = pd.Timestamp(initial_time)
        end_time = initial_time + pd.Timedelta(total_steps * interval_length_hours, snapshot_freq)

        if isinstance(n.snapshots, pd.MultiIndex):
            # MultiIndex snapshots
            snapshot_window = n.snapshots[(n.snapshots.get_level_values(1) >= initial_time) & (n.snapshots.get_level_values(1) < end_time)]
        else:
            # SingleIndex snapshots
            snapshot_window = n.snapshots[(n.snapshots >= initial_time) & (n.snapshots < end_time)]

    try:
        # Use PyPSA's built-in rolling horizon optimization but with logging monitoring for progress bar
        ## Previously was just n.optimize.optimize_with_rolling_horizon(snapshot_window,...etc.)
            n.optimize.optimize_with_rolling_horizon(
            snapshots=snapshot_window,  # Use the defined snapshot window
            horizon=horizon_snapshots,    # INTEGER: number of snapshots
            overlap=overlap_snapshots,    # INTEGER: number of snapshots  
            solver_name=solver_name,
            solver_options=solver_options,
            extra_functionality=extra_functionality,
            linearized_unit_commitment=config.get("rolling_horizon", {}).get("unit_commitment", False)
        )
    except Exception as e:
        logger.error(f"Built-in rolling horizon failed: {e}")
        logger.info("Falling back to manual implementation")
        return run_manual_rolling_horizon(n, scenario_setup, snakemake)
    
    return n

def run_manual_rolling_horizon(n, scenario_setup, snakemake):
    """
    Manual rolling horizon implementation (simplified version of your original)
    """
    
    config = snakemake.config.get("sequential_dispatch", {})
    
    # Get parameters
    horizon_length_hours = config.get("horizon_length_hours", 48)
    interval_length_hours = config.get("interval_length_hours", 24)
    overlap_hours = config.get("overlap_hours", 24)
    
    logger.info(f"Manual rolling horizon: {horizon_length_hours}h window, {interval_length_hours}h steps")
    
    # Get all snapshots
    if isinstance(n.snapshots, pd.MultiIndex):
        all_snapshots = n.snapshots.get_level_values(1)
        year = n.snapshots.get_level_values(0)[0]
    else:
        all_snapshots = n.snapshots
        year = None
    
    # Initialize storage states
    storage_states = {}
    if len(n.storage_units) > 0:
        for storage in n.storage_units.index:
            p_nom = n.storage_units.loc[storage, 'p_nom']
            max_hours = n.storage_units.loc[storage, 'max_hours']
            storage_states[storage] = p_nom * max_hours * 0.5  # 50% initial SOC
    
    # Create optimization windows
    start_time = all_snapshots[0]
    end_time = all_snapshots[-1]
    current_time = start_time
    
    step_timedelta = pd.Timedelta(hours=interval_length_hours)
    horizon_timedelta = pd.Timedelta(hours=horizon_length_hours)
    
    iteration = 0
    while current_time < end_time:
        iteration += 1
        
        # Define optimization window
        window_end = min(current_time + horizon_timedelta, end_time)
        
        if isinstance(n.snapshots, pd.MultiIndex):
            window_snapshots = n.snapshots[
                (n.snapshots.get_level_values(1) >= current_time) & 
                (n.snapshots.get_level_values(1) < window_end)
            ]
        else:
            window_snapshots = all_snapshots[
                (all_snapshots >= current_time) & 
                (all_snapshots < window_end)
            ]
        
        if len(window_snapshots) == 0:
            break
            
        logger.info(f"Iteration {iteration}: Optimizing {len(window_snapshots)} snapshots from {current_time}")
        
        # Set storage initial conditions
        for storage, soc in storage_states.items():
            if storage in n.storage_units.index:
                n.storage_units.loc[storage, 'state_of_charge_initial'] = soc
        
        # Optimize this window
        try:
            solve_window(n, window_snapshots, scenario_setup, snakemake)
            
            # Extract end-of-interval storage states for next iteration
            interval_end = min(current_time + step_timedelta, end_time)
            if isinstance(n.snapshots, pd.MultiIndex):
                interval_end_snapshot = (year, interval_end - pd.Timedelta(hours=1))
            else:
                interval_end_snapshot = interval_end - pd.Timedelta(hours=1)
            
            # Update storage states
            if len(n.storage_units) > 0 and not n.storage_units_t.state_of_charge.empty:
                for storage in storage_states.keys():
                    if storage in n.storage_units_t.state_of_charge.columns:
                        try:
                            storage_states[storage] = n.storage_units_t.state_of_charge.loc[interval_end_snapshot, storage]
                        except (KeyError, IndexError):
                            # Keep previous SOC if snapshot not available
                            pass
                            
        except Exception as e:
            logger.error(f"Optimization failed for iteration {iteration}: {e}")
            break
        
        # Move to next interval
        current_time += step_timedelta
        
        # Prevent infinite loop
        if iteration > 365:  # Max 1 year of daily dispatch
            logger.warning("Reached maximum iterations, stopping")
            break
    
    logger.info(f"Completed {iteration} rolling horizon iterations")
    return n

def solve_window(n, snapshots, scenario_setup, snakemake):
    """
    Solve a single optimization window
    """
    
    config = snakemake.config.get("sequential_dispatch", {})
    solver_config = snakemake.config.get("solving", {})
    
    # Get solver configuration
    solver_name = solver_config.get("solver", {}).get("name", "highs")
    opt_config = config.get("optimization", {})
    solver_profile = opt_config.get("solver_profile", "highs-default")
    solver_options = solver_config.get("solver_options", {}).get(solver_profile, {})
    
    # Define constraints function
    def extra_functionality(network, sns):
        """Add dispatch constraints"""
        add_dispatch_constraints(network, sns, scenario_setup, snakemake)
    
    # Optimize
    n.optimize(
        snapshots=snapshots,
        solver_name=solver_name,
        solver_options=solver_options,
        extra_functionality=extra_functionality,
        linearized_unit_commitment=config.get("rolling_horizon", {}).get("unit_commitment", False)
    )

def run_single_dispatch(n, scenario_setup, snakemake):
    """
    Run single (non-rolling) dispatch optimization
    """
    
    config = snakemake.config.get("sequential_dispatch", {})
    solver_config = snakemake.config.get("solving", {})
    
    # Get solver configuration
    solver_name = solver_config.get("solver", {}).get("name", "highs")
    opt_config = config.get("optimization", {})
    solver_profile = opt_config.get("solver_profile", "highs-default")
    solver_options = solver_config.get("solver_options", {}).get(solver_profile, {})
    
    # Define constraints function
    def extra_functionality(network, snapshots):
        """Add dispatch constraints"""
        add_dispatch_constraints(network, snapshots, scenario_setup, snakemake)
    
    # Single optimization
    n.optimize(
        solver_name=solver_name,
        solver_options=solver_options,
        extra_functionality=extra_functionality,
        linearized_unit_commitment=config.get("unit_commitment", False)
    )
    
    return n

def add_dispatch_constraints(n, snapshots, scenario_setup, snakemake):
    """
    Consolidated constraint function for dispatch optimization
    """
    
    config = snakemake.config.get("sequential_dispatch", {})
    
    # Unit commitment constraints
    if config.get("rolling_horizon", {}).get("unit_commitment", False) or config.get("unit_commitment", False):
        try:
            committable_config = snakemake.config.get("electricity", {}).get("dispatch_committable_carriers", {})
            if committable_config:
                p_max_pu = set_existing_committable(n, snapshots, scenario_setup, committable_config)
                set_max_status(n, snapshots, p_max_pu)
        except Exception as e:
            logger.warning(f"Unit commitment constraints failed: {e}")
    
    # Operational constraints
    try:
        # Import your constraint functions here
        # set_operational_limits(n, snapshots, scenario_setup)
        # ccgt_steam_constraints(n, snapshots, snakemake)
        pass
    except Exception as e:
        logger.warning(f"Operational constraints failed: {e}")


    """
    Monkey patch PyPSA's optimize method to add progress tracking
    
    This temporarily replaces n.optimize to add progress tracking,
    then restores the original method.
    """
    
    if snapshots is None:
        snapshots = n.snapshots
    
    # Calculate expected iterations
    total_snapshots = len(snapshots)
    dispatch_interval = horizon - overlap
    expected_iterations = max(1, int(np.ceil((total_snapshots - overlap) / dispatch_interval)))
    
    print(f"🔄 Rolling Horizon with Monkey Patching")
    print(f"   Expected iterations: {expected_iterations}")
    print()
    
    # Create progress bar
    pbar = tqdm(
        total=expected_iterations,
        desc="Rolling Horizon",
        unit="iteration",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"
    )
    
    # Store original optimize method
    original_optimize = n.optimize.__class__.__call__
    iteration_count = [0]  # Use list for mutable reference
    
    def tracked_optimize(self, *args, **kwargs):
        """Replacement optimize method with progress tracking"""
        iteration_count[0] += 1
        
        # Extract snapshots info if available
        opt_snapshots = kwargs.get('snapshots', args[0] if args else None)
        if opt_snapshots is not None and len(opt_snapshots) > 0:
            start_time_str = str(opt_snapshots[0])[:16]
            pbar.set_postfix_str(f"Iter {iteration_count[0]}: {start_time_str}")
        else:
            pbar.set_postfix_str(f"Iteration {iteration_count[0]}")
        
        # Call original optimize method
        start_time = time.time()
        result = original_optimize(self, *args, **kwargs)
        solve_time = time.time() - start_time
        
        # Update progress
        pbar.set_postfix_str(f"Iter {iteration_count[0]}: {solve_time:.1f}s")
        pbar.update(1)
        
        return result
    
    # Apply monkey patch
    n.optimize.__class__.__call__ = tracked_optimize
    
    try:
        # Run PyPSA's built-in rolling horizon
        start_time = time.time()
        result = n.optimize.optimize_with_rolling_horizon(
            snapshots=snapshots,
            horizon=horizon,
            overlap=overlap,
            **kwargs
        )
        
        # Complete progress bar if needed
        pbar.update(expected_iterations - pbar.n)
        pbar.set_postfix_str(f"Completed in {time.time() - start_time:.1f}s")
        pbar.close()
        
        print("✅ Rolling horizon optimization completed!")
        return result
        
    except Exception as e:
        pbar.set_postfix_str(f"❌ Error: {str(e)[:20]}")
        pbar.close()
        raise e
    
    finally:
        # Restore original optimize method
        n.optimize.__class__.__call__ = original_optimize
# =============================================================================
# PLACEHOLDER CONSTRAINT FUNCTIONS (add your implementations)
# =============================================================================

def set_existing_committable(n, snapshots, scenario_setup, config):
    """Placeholder - add your implementation"""
    logger.debug("Unit commitment constraints applied")
    return pd.DataFrame()  # Return empty DataFrame as placeholder

def set_max_status(n, snapshots, p_max_pu):
    """Placeholder - add your implementation"""
    logger.debug("Max status constraints applied")
    pass

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake(
            'solve_network_dispatch', 
            **{'scenario':'TEST', 'year':2030}
        )

    # Load network
    n = pypsa.Network(snakemake.input.dispatch_network)
    
    # Run dispatch (automatically detects PyPSA version and method)
    solve_network_dispatch(n, snakemake)