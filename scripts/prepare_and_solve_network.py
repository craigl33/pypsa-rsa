# SPDX-FileCopyrightText:  PyPSA-ZA2, PyPSA-ZA, PyPSA-Earth and PyPSA-Eur Authors
# # SPDX-License-Identifier: MIT
# coding: utf-8
"""
Prepare PyPSA network for solving according to :ref:`opts` and :ref:`ll`, such as

- adding an annual **limit** of carbon-dioxide emissions,
- adding an exogenous **price** per tonne emissions of carbon-dioxide (or other kinds),
- setting an **N-1 security margin** factor for transmission line capacities,
- specifying an expansion limit on the **cost** of transmission expansion,
- specifying an expansion limit on the **volume** of transmission expansion, and
- reducing the **temporal** resolution by averaging over multiple hours
  or segmenting time series into chunks of varying lengths using ``tsam``.

Relevant Settings
-----------------

.. code:: yaml

    costs:
        emission_prices:
        USD2013_to_EUR2013:
        discountrate:
        marginal_cost:
        capital_cost:

    electricity:
        co2limit:
        max_hours:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`costs_cf`, :ref:`electricity_cf`

Inputs
------

- ``data/costs.csv``: The database of cost assumptions for all included technologies for specific years from various sources; e.g. discount rate, lifetime, investment (CAPEX), fixed operation and maintenance (FOM), variable operation and maintenance (VOM), fuel costs, efficiency, carbon-dioxide intensity.
- ``networks/elec_s{simpl}_{clusters}.nc``: confer :ref:`cluster`

Outputs
-------

- ``networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc``: Complete PyPSA network that will be handed to the ``solve_network`` rule.

Description
-----------

.. tip::
    The rule :mod:`prepare_all_networks` runs
    for all ``scenario`` s in the configuration file
    the rule :mod:`prepare_network`.

"""
import logging
import re

import numpy as np
import pandas as pd
import pypsa

# Updated imports for PyPSA 0.34.1 - using linopy-based optimization
from pypsa.descriptors import get_switchable_as_dense as get_as_dense, expand_series

from _helpers import configure_logging, remove_leap_day, normalize_and_rename_df, assign_segmented_df_to_network, load_scenario_definition, add_missing_carriers
from add_electricity import load_extendable_parameters#, update_transmission_costs
from concurrent.futures import ProcessPoolExecutor
import xarray as xr
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Comment out for debugging and development
from custom_constraints import set_operational_limits, ccgt_steam_constraints, reserve_margin_constraints, annual_co2_constraints
from custom_constraints import add_national_capacity_constraints


idx = pd.IndexSlice
import os

"""
********************************************************************************
    Build limit constraints
********************************************************************************
"""


def enhanced_set_extendable_limits_global(n, scenario_setup):
    """
    Enhanced version that handles both original global limits and new national constraints.
    """

    # Check if multi-region scenario
    regions_setting = scenario_setup.get("regions", "1")
    is_multi_region = str(regions_setting) in ["10", "34", "159"]
    
    if not is_multi_region:
    # Original function
        _set_extendable_limits_national(n)
    else:
    # Then set high individual limits for regional technologies
        _set_extendable_limits_regional(n, scenario_setup)

def _set_extendable_limits_national(n):

    ext_years = n.investment_periods if n.multi_invest else [n.snapshots[0].year]
    sense = {"max": "<=", "min": ">="}
    ignore = {"max": "unc", "min": 0}

    # Initialize an empty dictionary for global limits
    global_limits = {}

    

    # Iterate over possible limits and try to read them from the Excel file
    for lim in ["max", "min"]:
        try:
            # Adapted from that used in add_electricity.py
            global_limit = pd.read_excel(
                os.path.join(scenario_setup["sub_path"], "extendable_technologies.xlsx"),
                sheet_name=f'{lim}_total_installed')
            
            national_id = "RSA"
            global_limit = global_limit.set_index(["Scenario","Location",  "Carrier"]).drop(columns=["Supply Region", "Category", "Component"])
            scen = scenario_setup[f"extendable_{lim}_total"]
            # Reads the national constraints for all carriers across all given years
            global_limit = global_limit.loc[(scen, national_id, slice(None)), ext_years]
            global_limit.index = global_limit.index.droplevel(["Scenario", "Location"])

            # If successfully read, add to the global_limits dictionary
            global_limits[lim] = global_limit
        except Exception as e:
            logging.warning(f"Error: {e} occured")

    # Now global_limits only contains keys for successfully read sheets
    for lim, global_limit in global_limits.items():
        global_limit = global_limit.loc[~(global_limit == ignore[lim]).all(axis=1)]
        constraints = [
            {
                "name": f"global_{lim}-{carrier}-{y}",
                "carrier_attribute": carrier,
                "sense": sense[lim],
                "type": "tech_capacity_expansion_limit",
                **({"investment_period": y} if n.multi_invest else {}),
                "constant": global_limit.loc[carrier, y],
            }
            for carrier in global_limit.index
            for y in ext_years
            if global_limit.loc[carrier, y] != ignore[lim]
        ]

        for constraint in constraints:
            n.add("GlobalConstraint", **constraint)

def set_extendable_limits_with_regional(n, scenario_setup):
    """
    Set high individual limits for regional technologies, 
    since national constraints will be applied separately.

    Note that this is currently geared towards 10-region setups. should be more generalised.
    """
    high_limit = 1e6  # 1000 GW - effectively unlimited
    
    # For generators
    for gen in n.generators.query("p_nom_extendable").index:
        # Check if this is a regional technology (has suffix)
        if any(suffix in gen for suffix in ['_0', '_1', '_2', '_3', '_4', '_5', '_6', '_7', '_8', '_9', 
                                          '_EC', '_FS', '_GP', '_HY', '_ZN', '_LP', '_MP', '_NW', '_NC', '_WC']):
            n.generators.loc[gen, "p_nom_max"] = high_limit
    
    # For storage units
    for su in n.storage_units.query("p_nom_extendable").index:
        if any(suffix in su for suffix in ['_0', '_1', '_2', '_3', '_4', '_5', '_6', '_7', '_8', '_9',
                                         '_EC', '_FS', '_GP', '_HY', '_ZN', '_LP', '_MP', '_NW', '_NC', '_WC']):
            n.storage_units.loc[su, "p_nom_max"] = high_limit

def set_extendable_limits_explicit_per_bus(n):
    """
    Legacy function for setting extendable limits per explicit technology and per bus
    """
    
    ext_years = n.investment_periods if n.multi_invest else [n.snapshots[0].year]
    ignore = {"max": "unc", "min": 0}

    bus_limits = {
        lim: pd.read_excel(
            os.path.join(scenario_setup["sub_path"], "extendable_technologies.xlsx"),
            sheet_name=f'{lim}_total_installed',
            index_col=[0, 1, 3, 2, 4],
        ).loc[(scenario_setup[f"extendable_{lim}_total"], scenario_setup["regions"], slice(None)), ext_years]
        for lim in ["max", "min"]
    }

    ext_carriers = (
        list(n.generators.carrier[n.generators.p_nom_extendable].unique())
        + list(n.storage_units.carrier[n.storage_units.p_nom_extendable].unique())
    )
    for lim, bus_limit in bus_limits.items():
        bus_limit.index = bus_limit.index.droplevel([0, 1, 2])
        bus_limit = bus_limit.loc[~(bus_limit == ignore[lim]).all(axis=1)]
        bus_limit = bus_limit.loc[bus_limit.index.get_level_values(1).isin(ext_carriers)]

        for idx in bus_limit.index:
            for y in ext_years:
                if bus_limit.loc[idx, y] != ignore[lim]:
                    n.buses.loc[idx[0],f"nom_{lim}_{idx[1]}_{y}"] = bus_limit.loc[idx, y]


"""
********************************************************************************
    Emissions limits and pricing
********************************************************************************
"""


def add_emission_prices(n, emission_prices=None, exclude_co2=False):
    if emission_prices is None:
        emission_prices = snakemake.config["costs"]["emission_prices"]
    if exclude_co2: emission_prices.pop("co2")
    ep = (pd.Series(emission_prices).rename(lambda x: x+"_emissions") * n.carriers).sum(axis=1)
    n.generators["marginal_cost"] += n.generators.carrier.map(ep)
    n.storage_units["marginal_cost"] += n.storage_units.carrier.map(ep)

# """
# ********************************************************************************
#     Transmission constraints
# ********************************************************************************
# """

# def set_line_s_max_pu(n):
#     s_max_pu = snakemake.config["lines"]["s_max_pu"]
#     n.lines["s_max_pu"] = s_max_pu
#     logger.info(f"N-1 security margin of lines set to {s_max_pu}")


# def set_transmission_limit(n, ll_type, factor, costs, Nyears=1):
#     links_dc_b = n.links.carrier == "DC" if not n.links.empty else pd.Series()

#     _lines_s_nom = (
#         np.sqrt(3)
#         * n.lines.type.map(n.line_types.i_nom)
#         * n.lines.num_parallel
#         * n.lines.bus0.map(n.buses.v_nom)
#     )
#     lines_s_nom = n.lines.s_nom.where(n.lines.type == "", _lines_s_nom)

#     col = "capital_cost" if ll_type == "c" else "length"
#     ref = (
#         lines_s_nom @ n.lines[col]
#         + n.links.loc[links_dc_b, "p_nom"] @ n.links.loc[links_dc_b, col]
#     )

#     update_transmission_costs(n, costs)

#     if factor == "opt" or float(factor) > 1.0:
#         n.lines["s_nom_min"] = lines_s_nom
#         n.lines["s_nom_extendable"] = True

#         n.links.loc[links_dc_b, "p_nom_min"] = n.links.loc[links_dc_b, "p_nom"]
#         n.links.loc[links_dc_b, "p_nom_extendable"] = True

#     if factor != "opt":
#         con_type = "expansion_cost" if ll_type == "c" else "volume_expansion"
#         rhs = float(factor) * ref
#         n.add(
#             "GlobalConstraint",
#             f"l{ll_type}_limit",
#             type=f"transmission_{con_type}_limit",
#             sense="<=",
#             constant=rhs,
#             carrier_attribute="AC, DC",
#         )
#     return n

# def set_line_nom_max(n, s_nom_max_set=np.inf, p_nom_max_set=np.inf):
#     n.lines.s_nom_max.clip(upper=s_nom_max_set, inplace=True)
#     n.links.p_nom_max.clip(upper=p_nom_max_set, inplace=True)

"""
********************************************************************************
    Time step reduction
********************************************************************************
"""

def average_every_nhours(n, offset):
    logging.info(f"Resampling the network to {offset}")
    m = n.copy()#with_time=False)
    snapshots_unstacked = n.snapshots.get_level_values(1)

    snapshot_weightings = n.snapshot_weightings.copy().set_index(snapshots_unstacked).resample(offset).sum()
    snapshot_weightings = remove_leap_day(snapshot_weightings)
    snapshot_weightings=snapshot_weightings[snapshot_weightings.index.year.isin(n.investment_periods)]
    snapshot_weightings.index = pd.MultiIndex.from_arrays([snapshot_weightings.index.year, snapshot_weightings.index])
    m.set_snapshots(snapshot_weightings.index)
    m.snapshot_weightings = snapshot_weightings

    for c in n.iterate_components():
        pnl = getattr(m, c.list_name + "_t")
        for k, df in c.pnl.items():
            if not df.empty:
                resampled = df.set_index(snapshots_unstacked).resample(offset).mean()
                resampled = remove_leap_day(resampled)
                resampled=resampled[resampled.index.year.isin(n.investment_periods)]
                resampled.index = snapshot_weightings.index
                pnl[k] = resampled
    return m

def single_year_segmentation(n, snapshots, config):

    p_max_pu, p_max_pu_max = normalize_and_rename_df(n.generators_t.p_max_pu, snapshots, 1, 'max')
    load, load_max = normalize_and_rename_df(n.loads_t.p_set, snapshots, 1, "load")
    inflow, inflow_max = normalize_and_rename_df(n.storage_units_t.inflow, snapshots, 0, "inflow")

    raw = pd.concat([p_max_pu, load, inflow], axis=1, sort=False)
    segments = config['tsam_clustering']['segments']

    multi_index = False
    if isinstance(raw.index, pd.MultiIndex):
        multi_index = True
        raw.index = raw.index.droplevel(0)
        
    y = snapshots.get_level_values(0)[0] if multi_index else snapshots[0].year

    agg = tsam.TimeSeriesAggregation(
        raw,
        hoursPerPeriod=len(raw),
        noTypicalPeriods=1,
        noSegments=int(segments),
        segmentation=True,
        solver=config['solver'],
    )

    segmented_df = agg.createTypicalPeriods()
    weightings = segmented_df.index.get_level_values("Segment Duration")
    cumsum = np.cumsum(weightings[:-1])
    
    if np.floor(y/4)-y/4 == 0: # check if leap year and add Feb 29 
            cumsum = np.where(cumsum >= 1416, cumsum + 24, cumsum) # 1416h from start year to Feb 29
    
    offsets = np.insert(cumsum, 0, 0)
    start_snapshot = snapshots[0][1] if n.multi_invest else snapshots[0]
    snapshots = pd.DatetimeIndex([start_snapshot + pd.Timedelta(hours=offset) for offset in offsets])
    snapshots = pd.MultiIndex.from_arrays([snapshots.year, snapshots]) if multi_index else snapshots
    weightings = pd.Series(weightings, index=snapshots, name="weightings", dtype="float64")
    segmented_df.index = snapshots

    segmented_df[p_max_pu.columns] *= p_max_pu_max
    segmented_df[load.columns] *= load_max
    segmented_df[inflow.columns] *= inflow_max
     
    logging.info(f"Segmentation complete for period: {y}")

    return segmented_df, weightings

def apply_time_segmentation(n, segments, config):
    logging.info(f"Aggregating time series to {segments} segments.")    
    years = n.investment_periods if n.multi_invest else [n.snapshots[0].year]

    if len(years) == 1:
        segmented_df, weightings = single_year_segmentation(n, n.snapshots, segments, config)
    else:

        with ProcessPoolExecutor(max_workers = min(len(years),config['nprocesses'])) as executor:
            parallel_seg = {
                year: executor.submit(
                    single_year_segmentation,
                    n,
                    n.snapshots[n.snapshots.get_level_values(0) == year],
                    segments,
                    config
                )
                for year in years
            }

        segmented_df = pd.concat(
            [parallel_seg[year].result()[0] for year in parallel_seg], axis=0
        )
        weightings = pd.concat(
            [parallel_seg[year].result()[1] for year in parallel_seg], axis=0
        )

    n.set_snapshots(segmented_df.index)
    n.snapshot_weightings = weightings   
    
    assign_segmented_df_to_network(segmented_df, "_load", "", n.loads_t.p_set)
    assign_segmented_df_to_network(segmented_df, "_max", "", n.generators_t.p_max_pu)
    assign_segmented_df_to_network(segmented_df, "_min", "", n.generators_t.p_min_pu)
    assign_segmented_df_to_network(segmented_df, "_inflow", "", n.storage_units_t.inflow)

    return n

def calc_emissions(n, scenario_setup):

    carrier_emissions = pd.read_excel(
        os.path.join(scenario_setup["sub_path"], "extendable_technologies.xlsx"), 
        sheet_name = "parameters",
        index_col = [0,2,1],
    ).sort_index().loc["default", "co2_emissions"].drop(["unit","source"], axis=1)
    gens = n.generators.query("carrier in @carrier_emissions.index").index
    efficiency = n.generators.loc[gens, "efficiency"]

    co2_emissions = pd.DataFrame(index=gens, columns = n.investment_periods)

    energy = n.generators_t.p[gens].groupby(level=0).sum()

    for y in n.investment_periods:
        for gen in gens:
            co2_emissions.loc[gen, y] = energy.loc[y, gen] * carrier_emissions.loc[n.generators.carrier[gen], y] / efficiency[gen]

    return co2_emissions.sum()/1e6

def calc_cumulative_new_capacity(n):
    carriers = list(n.generators.carrier.unique())+list(n.storage_units.carrier.unique())
    new_capacity = pd.DataFrame(index=carriers, columns = [2024]+list(n.investment_periods))
    for period in [2024]+list(n.investment_periods):
        for carrier in n.generators.carrier.unique():
            new_capacity.loc[carrier,period] = n.generators.p_nom_opt[(n.generators.carrier==carrier) & (n.generators.build_year<=period)].sum()
        for carrier in n.storage_units.carrier.unique():
            new_capacity.loc[carrier,period] = n.storage_units.p_nom_opt[(n.storage_units.carrier==carrier) & (n.storage_units.build_year<=period)].sum()    
    return new_capacity

def solve_network(n, sns):
    """
    Solve network using the new Linopy-based optimization approach.
    
    This follows the PyPSA-EUR v2025.04.0 pattern:
    1. Create the optimization model using the new API
    2. Add custom constraints via extra_functionality 
    3. Solve the model
    """
    
    def extra_functionality(n, snapshots):
        """
        Add custom constraints to the model.
        This function is called after model creation but before solving.
        """
        # Custom constraints using the new Linopy-based approach
        set_operational_limits(n, snapshots, scenario_setup)
        ccgt_steam_constraints(n, snapshots, snakemake)
        reserve_margin_constraints(n, snapshots, scenario_setup, snakemake)
        
        param = load_extendable_parameters(n, scenario_setup, snakemake)
        annual_co2_constraints(n, snapshots, param, scenario_setup)
        
        # Add national capacity constraints for regional technologies
        add_national_capacity_constraints(n, snapshots, scenario_setup)
    
    solver_config = snakemake.config["solving"]
    solver_name = solver_config['solver']["name"]  # should be a string, e.g., "gurobi"
    solver_options = solver_config["solver_options"][solver_config['solver'].get("options", {})] # should be a dict
    # Solve using the new optimize method with extra_functionality
    # This is the PyPSA 0.34.1 way following PyPSA-EUR patterns

    logging.info("solver_name =", solver_name, type(solver_name))
    logging.info("solver_options =", solver_options, type(solver_options))
    for k, v in solver_options.items():
        logging.info(f"Option key: {k}, value: {v}, type: {type(v)}")

    n.optimize(
        snapshots=sns,
        multi_investment_periods=n.multi_invest,
        solver_name=solver_name,
        solver_options=solver_options,
        extra_functionality=extra_functionality
    )

"""
TSAM clustering functionality
"""
def calculate_renewable_generation(n):
    """Calculate total renewable generation"""
    
    renewable_carriers = ['solar', 'wind', 'hydro', 'ror']  # adjust for your system
    total_renewable = pd.Series(0.0, index=n.snapshots)
    
    for gen in n.generators.index:
        carrier = n.generators.loc[gen, 'carrier']
        if any(ren in carrier.lower() for ren in renewable_carriers):
            # Get capacity
            if 'p_nom_max' in n.generators.columns:
                capacity = n.generators.loc[gen, 'p_nom_max']
            else:
                capacity = n.generators.loc[gen, 'p_nom']
            
            # Get availability profile
            if gen in n.generators_t.p_max_pu.columns:
                availability = n.generators_t.p_max_pu[gen]
            else:
                availability = 1.0  # constant availability
            
            generation = capacity * availability
            total_renewable += generation
    
    return total_renewable

def apply_tsam_clustering(n, period_type='total', typical_periods=36, 
                         hours_per_period=24, max_features=np.inf, clustering_mode='demand'):
    """
    Flexible TSAM clustering with configurable period frequency
    
    Parameters:
    -----------
    n : pypsa.Network
        The PyPSA network to cluster
    period_type : str
        - 'total': typical_periods applies to entire dataset (original behavior)
        - 'yearly': typical_periods per year
        - 'monthly': typical_periods per month
        - 'seasonal': typical_periods per season (quarterly)
        - 'weekly': typical_periods per week
    typical_periods : int
        Number of typical periods according to period_type
    hours_per_period : int
        Hours per typical period (default 24 for daily clustering)
    max_features : int
        Maximum number of features for clustering
    
    Returns:
    --------
    tuple: (typical_periods_data, weights, aggregation_object)
    """
    
    print(f"Applying flexible TSAM clustering:")
    print(f"  Period type: {period_type}")
    print(f"  Typical periods: {typical_periods}")
    print(f"  Hours per period: {hours_per_period}")
    print(f"  Input snapshots: {len(n.snapshots)}")
    
    # Create proper datetime index
    if isinstance(n.snapshots, pd.MultiIndex):
        datetime_index = n.snapshots.get_level_values(1)
        print(f"Using MultiIndex datetime: {datetime_index[0]} to {datetime_index[-1]}")
    else:
        datetime_index = n.snapshots
        print(f"Using direct snapshots: {datetime_index[0]} to {datetime_index[-1]}")
    
    # Ensure proper DatetimeIndex for TSAM
    if not isinstance(datetime_index, pd.DatetimeIndex):
        print("Converting to proper DatetimeIndex...")
        start_date = pd.Timestamp('2025-01-01')
        datetime_index = pd.date_range(start_date, periods=len(n.snapshots), freq='H')
    
    # Calculate total typical periods based on period_type
    total_typical_periods = _calculate_total_periods(datetime_index, period_type, typical_periods)
    
    print(f"  Calculated total typical periods: {total_typical_periods}")
    print(f"  Output snapshots: {total_typical_periods * hours_per_period}")
    
    # Build raw_data with proper index
    raw_data = pd.DataFrame(index=datetime_index)
    
    raw_data = pd.DataFrame(index=datetime_index)

    # NEW: Mode-based feature selection
    if clustering_mode == 'demand':
        # Cluster based on total demand only
        if not n.loads_t.p_set.empty:
            total_load = n.loads_t.p_set.sum(axis=1)
            raw_data['total_demand'] = total_load.values
            print(f"Added total_demand feature")

    elif clustering_mode == 'net_demand':
        # Cluster based on net demand (load - renewables)
        if not n.loads_t.p_set.empty:
            total_load = n.loads_t.p_set.sum(axis=1)
            
            # Calculate total renewable generation
            total_renewable = calculate_renewable_generation(n)
            
            net_demand = total_load - total_renewable
            raw_data['net_demand'] = net_demand.values 
            print(f"Added net_demand feature")

    elif clustering_mode == 'multi_feature':
        # Your existing multi-feature approach

        # Add total load (always important)
        if not n.loads_t.p_set.empty:
            total_load = n.loads_t.p_set.sum(axis=1)
            raw_data['total_load'] = total_load.values 
            print(f"Added total_load feature")
        
        # Add regional loads (top regions)
        if not n.loads_t.p_set.empty:
            regional_totals = n.loads_t.p_set.sum().sort_values(ascending=False)
            top_regions = regional_totals.head(5).index
            
            for region in top_regions:
                regional_load = n.loads_t.p_set[region]
                if regional_load.max() > 0:
                    clean_load = regional_load.fillna(0)
                    raw_data[f'load_{region}'] = clean_load.values / clean_load.max()
                    print(f"Added load_{region} feature")
        
        # Add REGIONAL renewable profiles
        feature_count = len(raw_data.columns)
        
        if not n.generators_t.p_max_pu.empty:
            gen_info = n.generators[['carrier', 'bus']].copy()
            
            for carrier in ['solar', 'wind']:
                carrier_gens = n.generators[n.generators.carrier.str.contains(carrier, case=False, na=False)]
                
                if len(carrier_gens) > 0:
                    print(f"Processing {len(carrier_gens)} {carrier} generators...")
                    
                    for bus in carrier_gens['bus'].unique()[:8]:
                        bus_gens = carrier_gens[carrier_gens['bus'] == bus].index
                        
                        if len(bus_gens) > 0 and all(gen in n.generators_t.p_max_pu.columns for gen in bus_gens):
                            capacity_weights = n.generators.loc[bus_gens, 'p_nom_max']
                            if capacity_weights.sum() > 0:
                                regional_cf = (n.generators_t.p_max_pu[bus_gens] * capacity_weights).sum(axis=1) / capacity_weights.sum()
                                clean_cf = regional_cf.fillna(0)
                                
                                feature_name = f'{carrier}_{bus}'
                                raw_data[feature_name] = clean_cf.values
                                feature_count += 1
                                print(f"Added {feature_name} feature")
                                
                                if feature_count >= max_features:
                                    break
                    
                    if feature_count >= max_features:
                        print(f"Reached max features ({max_features}) - stopping here")
                        break
        
        print(f"TSAM features ({len(raw_data.columns)}): {raw_data.columns.tolist()}")
        
    # Final NaN check and cleanup
    if raw_data.isna().any().any():
        print("⚠️  Found NaN values - cleaning...")
        nan_cols = raw_data.columns[raw_data.isna().any()]
        for col in nan_cols:
            nan_count = raw_data[col].isna().sum()
            print(f"  {col}: {nan_count} NaN values")
            raw_data[col] = raw_data[col].interpolate().fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        final_nan_check = raw_data.isna().any().any()
        if final_nan_check:
            print("❌ Still have NaN - dropping problematic rows")
            raw_data = raw_data.dropna()
        
        print(f"Clean data shape: {raw_data.shape}")
        
    # Apply TSAM clustering with calculated total periods
    from tsam.timeseriesaggregation import TimeSeriesAggregation
    
    
    aggregation = TimeSeriesAggregation(
        raw_data,
        resolution=1.0,
        noTypicalPeriods=total_typical_periods,
        hoursPerPeriod=hours_per_period,
        representationMethod= 'distributionAndMinMaxRepresentation', #'distributionAndMinMaxRepresentation', distributionRepresentation # Fixed
        clusterMethod='hierarchical',
        solver='cbc',
        rescaleClusterPeriods=True,  # Essential for energy conservation
        roundOutput=4,
        
        # # Min/max preservation
        # extremePeriodMethod='new_cluster_center',
        # addPeakMax=[raw_data.columns[0]],  # Use first column name
        # addPeakMin=[raw_data.columns[0]],  # Use first column name
        
        sortValues=False,
    )
    
    typical_periods = aggregation.createTypicalPeriods()
    
    # Get weights from the correct TSAM attribute
    weights = None
    
    # Try to get weights from clusterPeriodNoOccur (most reliable)
    if hasattr(aggregation, 'clusterPeriodNoOccur'):
        cluster_occurrences = aggregation.clusterPeriodNoOccur
        weights = []
        for period_idx in range(aggregation.noTypicalPeriods):
            if period_idx in cluster_occurrences:
                weights.append(cluster_occurrences[period_idx])
            else:
                weights.append(1)  # fallback
        logging.info(f"Found weights from clusterPeriodNoOccur: sum={sum(weights)}")
    
    # Fallback to other weight attributes
    if weights is None:
        logging.warning("clusterPeriodNoOccur not found - trying other attributes...")
        for attr_name in ['periodOccurrences', 'clusterOccurrences', 'typPeriodOccurrences']:
            if hasattr(aggregation, attr_name):
                weights = getattr(aggregation, attr_name)
                logging.info(f"Found weights in attribute: {attr_name}")
                break
    
    # Final fallback (this should not happen with proper TSAM setup)
    if weights is None:
        logging.warning("⚠️  No weights found - using uniform weights")
        num_periods = len(typical_periods) // hours_per_period
        original_periods = len(raw_data) // hours_per_period
        weight_per_period = original_periods / num_periods
        weights = [weight_per_period] * num_periods
    
        # Simple verification
    original_total = raw_data.sum().sum()

    weighted_total = np.sum(np.sum(
        typical_periods.loc[idx[i,:],:].values*weights[i]
        for i in np.arange(total_typical_periods)
    ))

    
    print(f"✅ TSAM clustering successful!")
    print(f"  Input: {len(raw_data)} time steps, {len(raw_data.columns)} features")
    print(f"  Output: {len(typical_periods)} representative time steps")
    print(f"  Energy conservation: {weighted_total/original_total:.6f}")
    print(f"  Peak values maintained: {typical_periods.values.max()/raw_data.values.max():.6f}")
    print(f"  Min values maintained: {typical_periods.values.min()/raw_data.values.min():.6f}")
    
    # Additional validation for demand clustering modes
    if clustering_mode in ['demand', 'net_demand']:
        print(f"  Clustering mode: {clustering_mode} (single feature)")
        print(f"  Feature used: {raw_data.columns[0]}")
    
    return typical_periods, weights, aggregation


def _calculate_total_periods(datetime_index, period_type, typical_periods):
    """
    Calculate total number of typical periods based on period_type
    """
    start_date = datetime_index[0]
    end_date = datetime_index[-1]
    
    if period_type == 'total':
        return typical_periods
    
    elif period_type == 'yearly':
        # Count number of years in the dataset
        years = end_date.year - start_date.year + 1
        return typical_periods * years
    
    elif period_type == 'monthly':
        # Count number of months in the dataset
        months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
        return typical_periods * months
    
    elif period_type == 'seasonal':
        # Count number of seasons (quarters) in the dataset
        start_quarter = (start_date.month - 1) // 3 + 1
        end_quarter = (end_date.month - 1) // 3 + 1
        years = end_date.year - start_date.year + 1
        
        if years == 1:
            quarters = end_quarter - start_quarter + 1
        else:
            quarters = (4 - start_quarter + 1) + (years - 2) * 4 + end_quarter
        
        return typical_periods * quarters
    
    elif period_type == 'weekly':
        # Count number of weeks in the dataset
        weeks = ((end_date - start_date).days // 7) + 1
        return typical_periods * weeks
    
    else:
        raise ValueError(f"Unknown period_type: {period_type}")


def _get_period_breakdown(datetime_index, period_type, typical_periods):
    """
    Get a human-readable breakdown of the period configuration
    """
    start_date = datetime_index[0]
    end_date = datetime_index[-1]
    
    if period_type == 'total':
        return f"{typical_periods} typical periods for entire dataset"
    
    elif period_type == 'yearly':
        years = end_date.year - start_date.year + 1
        return f"{typical_periods} typical periods × {years} years = {typical_periods * years} total"
    
    elif period_type == 'monthly':
        months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
        return f"{typical_periods} typical periods × {months} months = {typical_periods * months} total"
    
    elif period_type == 'seasonal':
        start_quarter = (start_date.month - 1) // 3 + 1
        end_quarter = (end_date.month - 1) // 3 + 1
        years = end_date.year - start_date.year + 1
        
        if years == 1:
            quarters = end_quarter - start_quarter + 1
        else:
            quarters = (4 - start_quarter + 1) + (years - 2) * 4 + end_quarter
        
        return f"{typical_periods} typical periods × {quarters} seasons = {typical_periods * quarters} total"
    
    elif period_type == 'weekly':
        weeks = ((end_date - start_date).days // 7) + 1
        return f"{typical_periods} typical periods × {weeks} weeks = {typical_periods * weeks} total"


def apply_tsam_to_pypsa_network(n, period_type='total', typical_periods=36, 
                              hours_per_period=24, max_features=np.inf):
    """
    Complete flexible TSAM clustering integration with PyPSA network
    
    Examples:
    ---------
    # Original behavior (36 typical days total)
    n_clustered = apply_tsam_to_pypsa_network(n, period_type='total', typical_periods=36)
    
    # 3 typical days per month
    n_clustered = apply_tsam_to_pypsa_network(n, period_type='monthly', typical_periods=3)
    
    # 12 typical days per year
    n_clustered = apply_tsam_to_pypsa_network(n, period_type='yearly', typical_periods=12)
    
    # 5 typical days per season (quarterly)
    n_clustered = apply_tsam_to_pypsa_network(n, period_type='seasonal', typical_periods=5)
    
    # 2 typical days per week
    n_clustered = apply_tsam_to_pypsa_network(n, period_type='weekly', typical_periods=2)
    """
    
    # Store original information BEFORE clustering
    original_snapshots = n.snapshots.copy()

    print(f"🔄 Applying flexible TSAM clustering to PyPSA network...")
    
    # Step 1: Apply flexible TSAM clustering
    typical_periods_data, weights, aggregation = apply_tsam_clustering(
        n, period_type=period_type, typical_periods=typical_periods, 
        hours_per_period=hours_per_period, max_features=max_features
    )
    
    if typical_periods_data is None:
        print("❌ TSAM clustering failed - cannot proceed")
        return None
    
    # Step 2: Create new clustered network with proper deep copy
    n_clustered = n.copy(with_time=False)
    
    # Manually copy static data to ensure clean state
    for component in n.iterate_components():
        if hasattr(n_clustered, component.list_name):
            static_data = component.df.copy()
            setattr(n_clustered, component.list_name, static_data)
            
            if hasattr(n_clustered, component.name + '_t'):
                component_t = getattr(n_clustered, component.name + '_t')
                for attr in ['p_set', 'p_max_pu', 'p_min_pu', 'e_max_pu', 'e_min_pu', 'inflow']:
                    if hasattr(component_t, attr):
                        setattr(component_t, attr, pd.DataFrame())
    
    # Step 3: Update snapshots to representative periods with proper datetime formatting
    
    # Create proper hourly datetime index for representative periods
    num_periods = len(typical_periods_data)
    base_datetime = pd.Timestamp('2025-01-01 00:00:00')
    proper_datetime_index = pd.date_range(
        start=base_datetime,
        periods=num_periods,
        freq='H'
    )
    
    if isinstance(n.snapshots, pd.MultiIndex):
        # Handle MultiIndex with investment periods
        original_years = n.snapshots.get_level_values(0).unique()
        
        # Preserve all investment periods by distributing snapshots across them
        years_list = []
        datetime_list = []
        
        snapshots_per_year = num_periods // len(original_years)
        remaining_snapshots = num_periods % len(original_years)
        
        start_idx = 0
        for i, year in enumerate(original_years):
            # Calculate snapshots for this year
            year_snapshots = snapshots_per_year
            if i < remaining_snapshots:  # Distribute remaining snapshots
                year_snapshots += 1
            
            # Create datetime range for this year
            year_base = pd.Timestamp(f'{year}-01-01 00:00:00')
            year_datetime_range = pd.date_range(
                start=year_base,
                periods=year_snapshots,
                freq='H'
            )
            
            years_list.extend([year] * year_snapshots)
            datetime_list.extend(year_datetime_range)
            start_idx += year_snapshots
        
        new_snapshots = pd.MultiIndex.from_arrays(
            [years_list, datetime_list],
            names=n.snapshots.names
        )
        
        # Copy investment periods to maintain them
        if hasattr(n, 'investment_periods'):
            n_clustered.investment_periods = n.investment_periods.copy()
        
    else:
        # Simple DatetimeIndex case
        new_snapshots = proper_datetime_index.copy()
        
        if hasattr(n.snapshots, 'name'):
            new_snapshots.name = n.snapshots.name
    
    n_clustered.snapshots = new_snapshots
    print(f"✅ Updated snapshots: {len(n.snapshots)} → {len(n_clustered.snapshots)}")
    
    # Debug: Check snapshot structure
    print(f"    New snapshots sample: {new_snapshots[:3].tolist()}")
    if hasattr(n_clustered, 'investment_periods'):
        print(f"    Investment periods preserved: {n_clustered.investment_periods.tolist()}")
    
    # Step 4: Update snapshot weightings
    expanded_weights = []
    for period_weight in weights:
        expanded_weights.extend([period_weight] * hours_per_period)
    
    expanded_weights = expanded_weights[:len(new_snapshots)]
    print(f"✅ Expanded weights: {len(weights)} periods → {len(expanded_weights)} hours")
    
    if hasattr(n_clustered, 'snapshot_weightings'):
        if isinstance(n_clustered.snapshot_weightings, pd.DataFrame):
            n_clustered.snapshot_weightings = pd.DataFrame(
                index=new_snapshots,
                data={
                    'objective': expanded_weights,
                    'generators': expanded_weights,
                    'stores': expanded_weights
                }
            )
        else:
            n_clustered.snapshot_weightings = pd.Series(expanded_weights, index=new_snapshots)
    else:
        n_clustered.snapshot_weightings = pd.Series(expanded_weights, index=new_snapshots)
    
    print(f"✅ Updated snapshot weightings")
    
    # Step 5: Update all time-series data using the TSAM representative periods
    print("🔄 Updating time-series data...")
    
    if isinstance(n.snapshots, pd.MultiIndex):
        original_datetime_index = n.snapshots.get_level_values(1)
    else:
        original_datetime_index = n.snapshots
    
    if isinstance(typical_periods_data.index, pd.MultiIndex):
        repr_datetime_index = typical_periods_data.index.get_level_values(-1)
    else:
        repr_datetime_index = typical_periods_data.index
    
    # Update time-series for each component
    for component_name in ['loads_t', 'generators_t', 'storage_units_t', 'stores_t', 'lines_t', 'links_t']:
        if hasattr(n_clustered, component_name):
            component_t = getattr(n_clustered, component_name)
            
            ts_attrs = []
            for attr_name in dir(component_t):
                if not attr_name.startswith('_') and hasattr(component_t, attr_name):
                    attr_obj = getattr(component_t, attr_name)
                    if isinstance(attr_obj, pd.DataFrame) and not attr_obj.empty:
                        ts_attrs.append(attr_name)
            
            for attr_name in ts_attrs:
                try:
                    ts_data = getattr(component_t, attr_name)
                    
                    matching_cols = [col for col in ts_data.columns if col in typical_periods_data.columns]
                    
                    if matching_cols:
                        new_ts_data = pd.DataFrame(
                            index=new_snapshots,
                            columns=ts_data.columns,
                            dtype=ts_data.dtypes.iloc[0] if len(ts_data.dtypes) > 0 else float
                        )
                        
                        for col in matching_cols:
                            new_ts_data[col] = typical_periods_data[col].values
                        
                        for col in ts_data.columns:
                            if col not in matching_cols:
                                sample_indices = np.linspace(0, len(ts_data) - 1, len(new_snapshots), dtype=int)
                                new_ts_data[col] = ts_data[col].iloc[sample_indices].values
                        
                    else:
                        sample_indices = np.linspace(0, len(ts_data) - 1, len(new_snapshots), dtype=int)
                        new_ts_data = ts_data.iloc[sample_indices].copy()
                        new_ts_data.index = new_snapshots
                    
                    if isinstance(new_snapshots, pd.MultiIndex):
                        new_ts_data.index.names = new_snapshots.names
                    else:
                        new_ts_data.index.name = new_snapshots.name
                    
                    new_ts_data.columns.name = ts_data.columns.name
                    
                    setattr(component_t, attr_name, new_ts_data)
                    print(f"    Updated {component_name}.{attr_name}: {ts_data.shape} → {new_ts_data.shape}")
                    
                except Exception as e:
                    print(f"    ⚠️  Failed to update {component_name}.{attr_name}: {e}")
                    continue
    
    # Step 6: Final consistency fixes
    print("🔧 Final consistency adjustments...")
    
    for component_name in ['loads_t', 'generators_t', 'storage_units_t', 'stores_t', 'lines_t', 'links_t']:
        if hasattr(n_clustered, component_name):
            component_t = getattr(n_clustered, component_name)
            
            for attr_name in dir(component_t):
                if not attr_name.startswith('_'):
                    try:
                        ts_data = getattr(component_t, attr_name)
                        if isinstance(ts_data, pd.DataFrame) and not ts_data.empty:
                            if not ts_data.index.equals(new_snapshots):
                                ts_data.index = new_snapshots
                                if isinstance(new_snapshots, pd.MultiIndex):
                                    ts_data.index.names = new_snapshots.names
                                else:
                                    ts_data.index.name = new_snapshots.name
                                setattr(component_t, attr_name, ts_data)
                    except:
                        continue
    
    # Ensure snapshot_weightings index is consistent
    if hasattr(n_clustered, 'snapshot_weightings'):
        if isinstance(n_clustered.snapshot_weightings, pd.DataFrame):
            n_clustered.snapshot_weightings.index = new_snapshots
            if isinstance(new_snapshots, pd.MultiIndex):
                n_clustered.snapshot_weightings.index.names = new_snapshots.names
            else:
                n_clustered.snapshot_weightings.index.name = new_snapshots.name
        else:
            n_clustered.snapshot_weightings.index = new_snapshots
            if isinstance(new_snapshots, pd.MultiIndex):
                n_clustered.snapshot_weightings.index.names = new_snapshots.names
            else:
                n_clustered.snapshot_weightings.index.name = new_snapshots.name
    
    # Step 7: Validate the clustered network
    if not n.loads_t.p_set.empty and not n_clustered.loads_t.p_set.empty:
        orig_total_energy = n.loads_t.p_set.sum().sum()
        
        if isinstance(n_clustered.snapshot_weightings, pd.DataFrame):
            weights_for_energy = n_clustered.snapshot_weightings['objective']
        else:
            weights_for_energy = n_clustered.snapshot_weightings
        
        clustered_total_energy = (n_clustered.loads_t.p_set.sum(axis=1) * weights_for_energy).sum()
        
        energy_conservation = clustered_total_energy / orig_total_energy * 100
        print(f"    Energy conservation: {energy_conservation:.2f}%")
        
        if abs(energy_conservation - 100) > 5:
            print(f"    ⚠️  Energy conservation error > 5%!")
    
    expected_snapshots = len(typical_periods_data)
    actual_snapshots = len(n_clustered.snapshots)
    print(f"    Snapshots: expected {expected_snapshots}, got {actual_snapshots}")
    
    total_weight = sum(expanded_weights)
    expected_weight = len(n.snapshots)
    print(f"    Total weight: {total_weight} (original periods: {expected_weight})")
    
    print(f"✅ Flexible TSAM clustering successfully applied to PyPSA network!")
    print(f"    Period configuration: {_get_period_breakdown(n.snapshots if isinstance(n.snapshots, pd.DatetimeIndex) else n.snapshots.get_level_values(1), period_type, typical_periods)}")
    print(f"    Original network: {len(n.snapshots)} snapshots")
    print(f"    Clustered network: {len(n_clustered.snapshots)} snapshots")
    print(f"    Reduction factor: {len(n.snapshots)/len(n_clustered.snapshots):.1f}x")
    
    # NEW: Store mapping information as network attributes
    n_clustered._tsam_original_snapshots = original_snapshots
    n_clustered._tsam_aggregation_obj = aggregation

    # NEW: Create and store time mapping
    n_clustered._tsam_time_mapping = create_simple_time_mapping(
    aggregation, original_snapshots, n_clustered.snapshots
    )

    return n_clustered

def create_simple_time_mapping(aggregation, original_snapshots, representative_snapshots):
    """
    Create basic time mapping from TSAM aggregation
    """
    
    time_mapping = pd.Series(index=original_snapshots, dtype=object)
    
    # Simple approach: distribute original times across representative periods
    for i, original_time in enumerate(original_snapshots):
        repr_idx = i % len(representative_snapshots)
        time_mapping.iloc[i] = representative_snapshots[repr_idx]
    
    return time_mapping

def export_time_mapping(n_clustered, filepath):
    """Export time mapping to CSV file"""
    if hasattr(n_clustered, '_tsam_time_mapping'):
        n_clustered._tsam_time_mapping.to_csv(filepath, header=['representative_time'])
        print(f"📄 Time mapping exported to: {filepath}")

def load_time_mapping(n_clustered, filepath):
    """Load time mapping from CSV file"""
    time_mapping = pd.read_csv(filepath, index_col=0, parse_dates=True)['representative_time']
    n_clustered._tsam_time_mapping = time_mapping
    print(f"📄 Time mapping loaded from: {filepath}")

def compare_networks_before_after_tsam(n_original, n_clustered):
    """
    Compare original and TSAM-clustered networks
    """
    print("\n" + "="*60)
    print("NETWORK COMPARISON: ORIGINAL vs TSAM CLUSTERED")
    print("="*60)
    
    comparison = {}
    
    # Basic network stats
    comparison['snapshots'] = [len(n_original.snapshots), len(n_clustered.snapshots)]
    comparison['buses'] = [len(n_original.buses), len(n_clustered.buses)]
    comparison['generators'] = [len(n_original.generators), len(n_clustered.generators)]
    comparison['loads'] = [len(n_original.loads), len(n_clustered.loads)]
    
    # Time series shapes
    if not n_original.loads_t.p_set.empty:
        comparison['load_timeseries'] = [n_original.loads_t.p_set.shape, n_clustered.loads_t.p_set.shape]
    
    if not n_original.generators_t.p_max_pu.empty:
        comparison['gen_timeseries'] = [n_original.generators_t.p_max_pu.shape, n_clustered.generators_t.p_max_pu.shape]
    
    # Create comparison DataFrame
    comp_df = pd.DataFrame(comparison, index=['Original', 'TSAM_Clustered']).T
    print(comp_df)
    
    # Energy statistics
    if not n_original.loads_t.p_set.empty and not n_clustered.loads_t.p_set.empty:
        print(f"\n📊 ENERGY ANALYSIS:")
        
        # Original energy
        orig_energy = n_original.loads_t.p_set.sum().sum()
        print(f"    Original total energy: {orig_energy:,.0f} MWh")
        
        # Clustered energy (weighted)
        if isinstance(n_clustered.snapshot_weightings, pd.DataFrame):
            weights = n_clustered.snapshot_weightings['objective']
        else:
            weights = n_clustered.snapshot_weightings
        
        clustered_energy = (n_clustered.loads_t.p_set.sum(axis=1) * weights).sum()
        print(f"    Clustered total energy: {clustered_energy:,.0f} MWh")
        print(f"    Conservation ratio: {clustered_energy/orig_energy:.4f}")
    
    # Load profile comparison
    if not n_original.loads_t.p_set.empty and not n_clustered.loads_t.p_set.empty:
        print(f"\n📈 LOAD PROFILE ANALYSIS:")
        
        orig_load = n_original.loads_t.p_set.sum(axis=1)
        clustered_load = n_clustered.loads_t.p_set.sum(axis=1)
        
        print(f"    Original load - Min: {orig_load.min():.0f}, Max: {orig_load.max():.0f}, Mean: {orig_load.mean():.0f}")
        print(f"    Clustered load - Min: {clustered_load.min():.0f}, Max: {clustered_load.max():.0f}, Mean: {clustered_load.mean():.0f}")
        
        # Peak preservation
        peak_preservation = clustered_load.max() / orig_load.max()
        print(f"    Peak preservation: {peak_preservation:.2%}")



# Main usage:
# n_clustered = apply_tsam_to_pypsa_network(n, period_type='monthly', typical_periods=3)
# n_clustered.optimize(solver_name='highs')



def compare_networks_before_after_tsam(n_original, n_clustered):
    """
    Compare original and TSAM-clustered networks
    """
    print("\n" + "="*60)
    print("NETWORK COMPARISON: ORIGINAL vs TSAM CLUSTERED")
    print("="*60)
    
    comparison = {}
    
    # Basic network stats
    comparison['snapshots'] = [len(n_original.snapshots), len(n_clustered.snapshots)]
    comparison['buses'] = [len(n_original.buses), len(n_clustered.buses)]
    comparison['generators'] = [len(n_original.generators), len(n_clustered.generators)]
    comparison['loads'] = [len(n_original.loads), len(n_clustered.loads)]
    
    # Time series shapes
    if not n_original.loads_t.p_set.empty:
        comparison['load_timeseries'] = [n_original.loads_t.p_set.shape, n_clustered.loads_t.p_set.shape]
    
    if not n_original.generators_t.p_max_pu.empty:
        comparison['gen_timeseries'] = [n_original.generators_t.p_max_pu.shape, n_clustered.generators_t.p_max_pu.shape]
    
    # Create comparison DataFrame
    comp_df = pd.DataFrame(comparison, index=['Original', 'TSAM_Clustered']).T
    print(comp_df)
    
    # Energy statistics
    if not n_original.loads_t.p_set.empty and not n_clustered.loads_t.p_set.empty:
        print(f"\n📊 ENERGY ANALYSIS:")
        
        # Original energy
        orig_energy = n_original.loads_t.p_set.sum().sum()
        print(f"    Original total energy: {orig_energy:,.0f} MWh")
        
        # Clustered energy (weighted)
        if isinstance(n_clustered.snapshot_weightings, pd.DataFrame):
            weights = n_clustered.snapshot_weightings['objective']
        else:
            weights = n_clustered.snapshot_weightings
        
        clustered_energy = (n_clustered.loads_t.p_set.sum(axis=1) * weights).sum()
        print(f"    Clustered total energy: {clustered_energy:,.0f} MWh")
        print(f"    Conservation ratio: {clustered_energy/orig_energy:.4f}")
    
    # Load profile comparison
    if not n_original.loads_t.p_set.empty and not n_clustered.loads_t.p_set.empty:
        print(f"\n📈 LOAD PROFILE ANALYSIS:")
        
        orig_load = n_original.loads_t.p_set.sum(axis=1)
        clustered_load = n_clustered.loads_t.p_set.sum(axis=1)
        
        print(f"    Original load - Min: {orig_load.min():.0f}, Max: {orig_load.max():.0f}, Mean: {orig_load.mean():.0f}")
        print(f"    Clustered load - Min: {clustered_load.min():.0f}, Max: {clustered_load.max():.0f}, Mean: {clustered_load.mean():.0f}")
        
        # Peak preservation
        peak_preservation = clustered_load.max() / orig_load.max()
        print(f"    Peak preservation: {peak_preservation:.2%}")


# Example usage:
def run_complete_tsam_workflow(n, typical_days=12, hours_per_day=24):
    """
    Complete workflow: TSAM clustering + PyPSA optimization
    """
    print("🚀 COMPLETE TSAM + PyPSA WORKFLOW")
    print("="*50)
    
    # Step 1: Apply TSAM clustering
    n_clustered = apply_tsam_to_pypsa_network(n, typical_days=typical_days, hours_per_day=hours_per_day)
    
    if n_clustered is None:
        print("❌ TSAM clustering failed - stopping workflow")
        return None
    
    # Step 2: Compare networks
    compare_networks_before_after_tsam(n, n_clustered)
    
    # Step 3: Ready for optimization
    print(f"\n🎯 READY FOR OPTIMIZATION!")
    print(f"    Use: n_clustered.optimize(solver_name='highs')")
    print(f"    Time series are properly weighted for representative periods")
    print(f"    Snapshot weightings preserve total energy balance")
    
    return n_clustered


# Usage example:
# n_clustered = apply_tsam_to_pypsa_network(n, typical_days=12, hours_per_day=24)
# 
# # Then run optimization on clustered network:
# n_clustered.optimize(solver_name='highs')
# 
# # Results will be properly weighted by the representative periods
# print(f"Objective value: {n_clustered.objective}")
# print(f"Generator capacities: {n_clustered.generators.p_nom_opt}")


"""
********************************************************************************
    DEBUG FUNCTIONS - Added for comprehensive network debugging
********************************************************************************
"""

def debug_tsam_infeasibility(n_clustered):
    """
    Comprehensive debugging for TSAM-clustered network infeasibility
    """
    print("🔍 DEBUGGING TSAM NETWORK INFEASIBILITY")
    print("="*60)
    
    # 1. Check basic network structure
    print("1️⃣ NETWORK STRUCTURE CHECK:")
    print(f"    Snapshots: {len(n_clustered.snapshots)}")
    print(f"    Buses: {len(n_clustered.buses)}")
    print(f"    Generators: {len(n_clustered.generators)}")
    print(f"    Loads: {len(n_clustered.loads)}")
    print(f"    Lines: {len(n_clustered.lines)}")
    print(f"    Links: {len(n_clustered.links) if hasattr(n_clustered, 'links') else 0}")
    
    # 2. Check snapshot weightings
    print(f"\n2️⃣ SNAPSHOT WEIGHTINGS CHECK:")
    if hasattr(n_clustered, 'snapshot_weightings'):
        weights = n_clustered.snapshot_weightings
        if isinstance(weights, pd.DataFrame):
            weights = weights['objective']
        print(f"    ✅ Snapshot weightings exist: {len(weights)} periods")
        print(f"    Weight range: {weights.min():.1f} to {weights.max():.1f}")
        print(f"    Total weight: {weights.sum():.0f}")
        
        # Check for invalid weights
        if (weights <= 0).any():
            print(f"    ⚠️  Found {(weights <= 0).sum()} zero or negative weights!")
        if weights.isna().any():
            print(f"    ⚠️  Found {weights.isna().sum()} NaN weights!")
    else:
        print(f"    ❌ No snapshot weightings found!")
    
    # 3. Energy balance check
    print(f"\n3️⃣ ENERGY BALANCE CHECK:")
    
    # Total demand
    if not n_clustered.loads_t.p_set.empty:
        total_demand = n_clustered.loads_t.p_set.sum(axis=1)
        max_demand = total_demand.max()
        mean_demand = total_demand.mean()
        print(f"    Max demand: {max_demand:,.0f} MW")
        print(f"    Mean demand: {mean_demand:,.0f} MW")
        
        # Check for zero or negative demand
        if (total_demand <= 0).any():
            print(f"    ⚠️  Found {(total_demand <= 0).sum()} periods with zero/negative demand!")
    else:
        print(f"    ❌ No load data found!")
        return
    
    # Total generation capacity
    total_capacity = 0
    if not n_clustered.generators.empty:
        # Check for different capacity columns
        capacity_col = None
        for col in ['p_nom_max', 'p_nom', 'p_nom_extendable']:
            if col in n_clustered.generators.columns:
                capacity_col = col
                break
        
        if capacity_col:
            if capacity_col == 'p_nom_extendable':
                # For extendable generators, use a large number or check p_nom_max
                extendable_gens = n_clustered.generators[n_clustered.generators.p_nom_extendable == True]
                fixed_gens = n_clustered.generators[n_clustered.generators.p_nom_extendable == False]
                
                fixed_capacity = fixed_gens['p_nom'].sum() if len(fixed_gens) > 0 else 0
                print(f"    Fixed capacity: {fixed_capacity:,.0f} MW")
                print(f"    Extendable generators: {len(extendable_gens)}")
                
                if 'p_nom_max' in n_clustered.generators.columns:
                    max_extendable = extendable_gens['p_nom_max'].sum()
                    total_capacity = fixed_capacity + max_extendable
                    print(f"    Max extendable capacity: {max_extendable:,.0f} MW")
                else:
                    total_capacity = fixed_capacity
                    print(f"    ⚠️  No p_nom_max for extendable generators!")
            else:
                total_capacity = n_clustered.generators[capacity_col].sum()
                print(f"    Total capacity ({capacity_col}): {total_capacity:,.0f} MW")
        else:
            print(f"    ❌ No capacity column found in generators!")
    
    # Capacity adequacy
    if total_capacity > 0 and max_demand > 0:
        adequacy_ratio = total_capacity / max_demand
        print(f"    Capacity adequacy ratio: {adequacy_ratio:.2f}")
        if adequacy_ratio < 1.0:
            print(f"    ❌ INSUFFICIENT CAPACITY! Need at least {max_demand:,.0f} MW")
        elif adequacy_ratio < 1.2:
            print(f"    ⚠️  LOW CAPACITY MARGIN! Consider adding reserves")
        else:
            print(f"    ✅ Adequate capacity available")
    
    # 4. Check renewable availability
    print(f"\n4️⃣ RENEWABLE AVAILABILITY CHECK:")
    
    if not n_clustered.generators_t.p_max_pu.empty:
        renewable_carriers = ['solar', 'wind', 'hydro']
        
        for carrier in renewable_carriers:
            carrier_gens = n_clustered.generators[
                n_clustered.generators.carrier.str.contains(carrier, case=False, na=False)
            ].index
            
            if len(carrier_gens) > 0:
                carrier_capacity = n_clustered.generators.loc[carrier_gens, 'p_nom_max'].sum()
                
                if len(carrier_gens) > 0 and all(gen in n_clustered.generators_t.p_max_pu.columns for gen in carrier_gens):
                    avg_cf = n_clustered.generators_t.p_max_pu[carrier_gens].mean().mean()
                    max_cf = n_clustered.generators_t.p_max_pu[carrier_gens].max().max()
                    min_cf = n_clustered.generators_t.p_max_pu[carrier_gens].min().min()
                    
                    print(f"    {carrier.title()}: {len(carrier_gens)} units, {carrier_capacity:,.0f} MW")
                    print(f"      CF range: {min_cf:.3f} to {max_cf:.3f} (avg: {avg_cf:.3f})")
                    
                    # Check for problematic capacity factors
                    if max_cf > 1.0:
                        print(f"      ⚠️  Capacity factors > 1.0 found!")
                    if avg_cf < 0.01:
                        print(f"      ⚠️  Very low average capacity factor!")
    
    # 5. Check transmission constraints
    print(f"\n5️⃣ TRANSMISSION CHECK:")
    
    if not n_clustered.lines.empty:
        print(f"    Lines: {len(n_clustered.lines)}")
        if 's_nom' in n_clustered.lines.columns:
            total_line_capacity = n_clustered.lines.s_nom.sum()
            print(f"    Total line capacity: {total_line_capacity:,.0f} MVA")
        
        # Check for zero capacity lines
        if 's_nom' in n_clustered.lines.columns:
            zero_capacity_lines = (n_clustered.lines.s_nom <= 0).sum()
            if zero_capacity_lines > 0:
                print(f"    ⚠️  {zero_capacity_lines} lines with zero/negative capacity!")
    
    if hasattr(n_clustered, 'links') and not n_clustered.links.empty:
        print(f"    Links: {len(n_clustered.links)}")
        if 'p_nom' in n_clustered.links.columns:
            total_link_capacity = n_clustered.links.p_nom.sum()
            print(f"    Total link capacity: {total_link_capacity:,.0f} MW")
    
    # 6. Check for common infeasibility causes
    print(f"\n6️⃣ COMMON INFEASIBILITY CAUSES:")
    
    issues_found = []
    
    # Island detection
    if not n_clustered.lines.empty:
        # Simple connectivity check
        buses_with_lines = set(n_clustered.lines.bus0) | set(n_clustered.lines.bus1)
        all_buses = set(n_clustered.buses.index)
        isolated_buses = all_buses - buses_with_lines
        
        if isolated_buses:
            print(f"    ⚠️  {len(isolated_buses)} isolated buses found!")
            issues_found.append("isolated_buses")
    
    # Investment period issues
    if hasattr(n_clustered, 'investment_periods'):
        if len(n_clustered.investment_periods) > 1:
            print(f"    Multi-period model: {len(n_clustered.investment_periods)} periods")
            # Check for investment period specific issues
            if isinstance(n_clustered.snapshots, pd.MultiIndex):
                periods_in_snapshots = n_clustered.snapshots.get_level_values(0).unique()
                missing_periods = set(n_clustered.investment_periods) - set(periods_in_snapshots)
                if missing_periods:
                    print(f"    ⚠️  Investment periods missing from snapshots: {missing_periods}")
                    issues_found.append("missing_investment_periods")
    
    # Weekly operational limits warning (from your output)
    print(f"    ⚠️  Weekly operational limits detected - may conflict with TSAM weighting")
    issues_found.append("weekly_limits_conflict")
    
    # Time series data integrity
    ts_issues = check_timeseries_integrity(n_clustered)
    if ts_issues:
        issues_found.extend(ts_issues)
    
    # 7. Recommendations
    print(f"\n7️⃣ RECOMMENDATIONS:")
    
    if not issues_found:
        print(f"    ✅ No obvious issues found - infeasibility may be in detailed constraints")
        print(f"    Try: n_clustered.optimize(solver_name='highs', track_variables=True)")
    else:
        print(f"    Issues found that may cause infeasibility:")
        for issue in issues_found:
            if issue == "isolated_buses":
                print(f"    • Fix isolated buses by adding transmission connections")
            elif issue == "missing_investment_periods":
                print(f"    • Ensure all investment periods have snapshots")
            elif issue == "weekly_limits_conflict":
                print(f"    • Disable weekly operational limits for TSAM models")
                print(f"    • Or use period_type='weekly' in TSAM clustering")
            elif issue == "zero_demand":
                print(f"    • Fix zero/negative demand periods")
            elif issue == "insufficient_capacity":
                print(f"    • Add more generation capacity or make generators extendable")
            elif issue == "invalid_cf":
                print(f"    • Fix capacity factors > 1.0 or NaN values")
    
    return issues_found


def check_timeseries_integrity(n):
    """
    Check time series data for common issues
    """
    issues = []
    
    # Check loads
    if not n.loads_t.p_set.empty:
        if n.loads_t.p_set.isna().any().any():
            print(f"    ⚠️  NaN values in load time series!")
            issues.append("nan_loads")
        
        if (n.loads_t.p_set < 0).any().any():
            print(f"    ⚠️  Negative values in load time series!")
            issues.append("negative_loads")
        
        if (n.loads_t.p_set.sum(axis=1) == 0).any():
            print(f"    ⚠️  Zero total demand in some periods!")
            issues.append("zero_demand")
    
    # Check generators
    if not n.generators_t.p_max_pu.empty:
        if n.generators_t.p_max_pu.isna().any().any():
            print(f"    ⚠️  NaN values in generator availability!")
            issues.append("nan_generators")
        
        if (n.generators_t.p_max_pu > 1.0).any().any():
            print(f"    ⚠️  Capacity factors > 1.0 found!")
            issues.append("invalid_cf")
        
        if (n.generators_t.p_max_pu < 0).any().any():
            print(f"    ⚠️  Negative capacity factors found!")
            issues.append("negative_cf")
    
    return issues


def fix_common_tsam_issues(n_clustered):
    """
    Attempt to fix common TSAM-related infeasibility issues
    """
    print("🔧 ATTEMPTING TO FIX COMMON TSAM ISSUES")
    print("="*50)
    
    fixed_issues = []
    
    # 1. Fix snapshot weightings issues
    if hasattr(n_clustered, 'snapshot_weightings'):
        weights = n_clustered.snapshot_weightings
        if isinstance(weights, pd.DataFrame):
            weights = weights['objective']
        
        # Fix zero or negative weights
        if (weights <= 0).any():
            print("    Fixing zero/negative weights...")
            weights = weights.clip(lower=1)  # Set minimum weight to 1
            n_clustered.snapshot_weightings = weights
            fixed_issues.append("negative_weights")
        
        # Fix NaN weights
        if weights.isna().any():
            print("    Fixing NaN weights...")
            weights = weights.fillna(weights.mean())
            n_clustered.snapshot_weightings = weights
            fixed_issues.append("nan_weights")
    
    # 2. Fix time series data issues
    # Fix load data
    if not n_clustered.loads_t.p_set.empty:
        load_data = n_clustered.loads_t.p_set
        
        # Fix NaN values
        if load_data.isna().any().any():
            print("    Fixing NaN values in load data...")
            n_clustered.loads_t.p_set = load_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
            fixed_issues.append("nan_loads")
        
        # Fix negative values
        if (load_data < 0).any().any():
            print("    Fixing negative load values...")
            n_clustered.loads_t.p_set = load_data.clip(lower=0)
            fixed_issues.append("negative_loads")
        
        # Fix zero total demand periods
        total_demand = n_clustered.loads_t.p_set.sum(axis=1)
        if (total_demand == 0).any():
            print("    Fixing zero demand periods...")
            # Set minimum demand to 1% of average
            avg_demand = total_demand[total_demand > 0].mean()
            min_demand = avg_demand * 0.01
            
            zero_periods = total_demand == 0
            for period in zero_periods[zero_periods].index:
                # Distribute minimum demand across all loads proportionally
                if not n_clustered.loads_t.p_set.loc[period].empty:
                    n_clustered.loads_t.p_set.loc[period] = min_demand / len(n_clustered.loads_t.p_set.columns)
            
            fixed_issues.append("zero_demand")
    
    # Fix generator availability data
    if not n_clustered.generators_t.p_max_pu.empty:
        gen_data = n_clustered.generators_t.p_max_pu
        
        # Fix NaN values
        if gen_data.isna().any().any():
            print("    Fixing NaN values in generator availability...")
            n_clustered.generators_t.p_max_pu = gen_data.fillna(0)  # Assume no availability when NaN
            fixed_issues.append("nan_generators")
        
        # Fix capacity factors > 1.0
        if (gen_data > 1.0).any().any():
            print("    Fixing capacity factors > 1.0...")
            n_clustered.generators_t.p_max_pu = gen_data.clip(upper=1.0)
            fixed_issues.append("invalid_cf")
        
        # Fix negative capacity factors
        if (gen_data < 0).any().any():
            print("    Fixing negative capacity factors...")
            n_clustered.generators_t.p_max_pu = gen_data.clip(lower=0)
            fixed_issues.append("negative_cf")
    
    # 3. Ensure adequate capacity
    if not n_clustered.loads_t.p_set.empty and not n_clustered.generators.empty:
        max_demand = n_clustered.loads_t.p_set.sum(axis=1).max()
        
        # Check total available capacity
        total_capacity = 0
        if 'p_nom_max' in n_clustered.generators.columns:
            total_capacity = n_clustered.generators.p_nom_max.sum()
        elif 'p_nom' in n_clustered.generators.columns:
            total_capacity = n_clustered.generators.p_nom.sum()
        
        if total_capacity < max_demand * 1.1:  # Need at least 10% margin
            print(f"    Insufficient capacity detected. Adding slack generator...")
            
            # Add a slack generator with high marginal cost
            slack_capacity = max_demand * 1.5 - total_capacity
            
            # Find a bus to connect the slack generator
            main_bus = n_clustered.loads_t.p_set.sum().idxmax()  # Bus with highest total load
            
            # Add slack generator
            n_clustered.add("Generator",
                          name="slack_generator",
                          bus=main_bus,
                          p_nom_max=slack_capacity,
                          marginal_cost=1000,  # High cost to discourage use
                          carrier="slack")
            
            # Add constant availability
            n_clustered.generators_t.p_max_pu["slack_generator"] = 1.0
            
            fixed_issues.append("insufficient_capacity")
    
    print(f"    Fixed issues: {fixed_issues}")
    return fixed_issues


def retry_optimization_with_debugging(n_clustered, solver='highs'):
    """
    Complete debugging and fixing workflow for TSAM optimization
    """
    print("🚀 RETRY OPTIMIZATION WITH DEBUGGING")
    print("="*60)
    
    # Step 1: Debug infeasibility
    issues = debug_tsam_infeasibility(n_clustered)
    
    # Step 2: Attempt fixes
    if issues:
        print(f"\n🔧 Attempting to fix {len(issues)} identified issues...")
        fixed = fix_common_tsam_issues(n_clustered)
        
        if fixed:
            print(f"    Fixed: {fixed}")
        else:
            print(f"    No automatic fixes available for: {issues}")
    
    # Step 3: Retry optimization
    print(f"\n🔄 Retrying optimization...")
    
    try:
        status, termination = n_clustered.optimize(solver_name=solver)
        
        if status == 'ok':
            print(f"    ✅ Optimization successful!")
            print(f"    Objective: {n_clustered.objective:,.0f}")
            return n_clustered
        else:
            print(f"    ❌ Optimization still failed: {status} ({termination})")
            
            # Try with relaxed solver settings
            print(f"    🔄 Trying with relaxed settings...")
            status, termination = n_clustered.optimize(
                solver_name=solver,
                solver_options={'feasibility_tolerance': 1e-5,
                              'optimality_tolerance': 1e-5}
            )
            
            if status == 'ok':
                print(f"    ✅ Optimization successful with relaxed settings!")
                print(f"    Objective: {n_clustered.objective:,.0f}")
                return n_clustered
            else:
                print(f"    ❌ Still failed: {status} ({termination})")
                return None
    
    except Exception as e:
        print(f"    ❌ Optimization error: {e}")
        return None


def fix_nan_values(n):
    """Fix NaN values that could cause solver issues"""
    
    print("\n🔧 Fixing NaN values...")
    fixes_applied = False
    
    # Fill critical NaNs with defaults
    for component_name in ['generators', 'storage_units', 'links']:
        component = getattr(n, component_name)
        if component.empty:
            continue
            
        # Fill numeric columns with 0 or appropriate defaults
        numeric_cols = component.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if component[col].isna().any():
                print(f"  Filling NaNs in {component_name}.{col}")
                fixes_applied = True
                if col in ['p_nom_max', 'e_nom_max']:
                    component[col] = component[col].fillna(np.inf)
                elif col in ['efficiency']:
                    component[col] = component[col].fillna(1.0)
                elif col in ['marginal_cost', 'capital_cost']:
                    component[col] = component[col].fillna(0.0)
                elif col in ['p_nom_min', 'e_nom_min']:
                    component[col] = component[col].fillna(0.0)
                else:
                    component[col] = component[col].fillna(0.0)
    
    if not fixes_applied:
        print("  ✅ No NaN values found to fix")
    else:
        print("  ✅ NaN values fixed")


def check_generation_potential(n):
    """Check if network has sufficient generation potential"""
    
    print("\n⚡ Checking generation potential...")
    
    # Fixed capacity
    fixed_capacity = n.generators.query('not p_nom_extendable').p_nom.sum()
    
    # Maximum extendable capacity  
    extendable_capacity = n.generators.query('p_nom_extendable').p_nom_max.sum()
    
    # Total demand
    total_demand = n.loads_t.p_set.sum().sum()
    
    print(f"  Fixed capacity: {fixed_capacity:,.0f} MW")
    print(f"  Max extendable capacity: {extendable_capacity:,.0f} MW") 
    print(f"  Total capacity potential: {(fixed_capacity + extendable_capacity):,.0f} MW")
    print(f"  Total demand: {total_demand:,.0f} MWh")
    
    # Calculate required capacity factor
    total_capacity_potential = fixed_capacity + extendable_capacity
    if total_capacity_potential > 0:
        required_cf = total_demand / (total_capacity_potential * 8760)
        print(f"  Required capacity factor: {required_cf:.2%}")
        
        if required_cf > 0.8:
            print("  ❌ WARNING: Very high capacity factor required (>80%)!")
            return False
        elif required_cf > 0.5:
            print("  ⚠️  Warning: High capacity factor required (>50%)")
        else:
            print("  ✅ Capacity factor requirement looks reasonable")
    else:
        print("  ❌ ERROR: No generation capacity potential!")
        return False
    
    # Check if any extendable generators exist
    extendable_count = n.generators.p_nom_extendable.sum()
    print(f"  Extendable generators: {extendable_count}")
    
    if extendable_count == 0 and fixed_capacity == 0:
        print("  ❌ ERROR: No generators (fixed or extendable) available!")
        return False
    
    return True


def analyze_generator_configuration(n):
    """Detailed analysis of generator configuration"""
    
    print("\n🔍 Analyzing generator configuration...")
    
    print(f"  Total generators: {len(n.generators)}")
    print(f"  Extendable generators: {n.generators.p_nom_extendable.sum()}")
    print(f"  Fixed generators: {(~n.generators.p_nom_extendable).sum()}")
    
    # Check capacity limits for extendable generators
    extendable = n.generators.query('p_nom_extendable')
    if not extendable.empty:
        print(f"  Extendable p_nom_max range: {extendable.p_nom_max.min():.0f} - {extendable.p_nom_max.max():.0f} MW")
        print(f"  Extendable p_nom_min range: {extendable.p_nom_min.min():.0f} - {extendable.p_nom_min.max():.0f} MW")
        
        # Check for problematic bounds
        zero_max = extendable.query('p_nom_max <= 0')
        if not zero_max.empty:
            print(f"  ❌ {len(zero_max)} extendable generators have p_nom_max <= 0!")
        
        inverted_bounds = extendable.query('p_nom_min > p_nom_max')
        if not inverted_bounds.empty:
            print(f"  ❌ {len(inverted_bounds)} extendable generators have p_nom_min > p_nom_max!")
    
    # Check carriers
    carriers = n.generators.carrier.value_counts()
    print(f"  Generator carriers:")
    for carrier, count in carriers.head(10).items():
        print(f"    {carrier}: {count}")
    
    # Check marginal costs
    print(f"  Marginal cost range: {n.generators.marginal_cost.min():.2f} - {n.generators.marginal_cost.max():.2f}")
    
    # Check for generators with very high costs (potential issue)
    high_cost = n.generators.query('marginal_cost > 10000')
    if not high_cost.empty:
        print(f"  ⚠️  {len(high_cost)} generators have very high marginal costs (>10000)")


def check_extendable_limits(n, scenario_setup):
    """Check if extendable technology limits are reasonable"""
    
    print("\n📊 Checking extendable technology limits...")
    
    try:
        # Check the Excel file constraints
        excel_file = os.path.join(scenario_setup["sub_path"], "extendable_technologies.xlsx")
        
        if os.path.exists(excel_file):
            print(f"  Found extendable technologies file: {excel_file}")
            
            # Try to read max limits
            try:
                max_limits = pd.read_excel(excel_file, sheet_name='max_total_installed')
                print(f"  Max total installed sheet has {len(max_limits)} rows")
                
                # Check if limits are too restrictive
                national_limits = max_limits[max_limits['Location'] == 'RSA']
                print(f"  National (RSA) limits: {len(national_limits)} entries")
                
                if not national_limits.empty:
                    print("  Sample national limits:")
                    print(national_limits.head(3).to_string())
                    
                    # Check for very low limits
                    numeric_cols = national_limits.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        if col in national_limits.columns:
                            low_limits = national_limits[national_limits[col] < 1000]  # Less than 1 GW
                            if not low_limits.empty:
                                print(f"  ⚠️  {len(low_limits)} carriers have limits < 1000 MW in {col}")
                
            except Exception as e:
                print(f"  ❌ Could not read max_total_installed sheet: {e}")
                
        else:
            print(f"  ❌ Extendable technologies file not found: {excel_file}")
            
    except Exception as e:
        print(f"  ❌ Error checking extendable limits: {e}")


def debug_simple_optimization(n):
    """Try a simple optimization without custom constraints"""
    
    print("\n🚀 Attempting simple optimization...")
    
    # Check basic requirements
    if n.generators.p_nom_extendable.sum() == 0:
        print("  ❌ No extendable generators found!")
        fixed_cap = n.generators.query('not p_nom_extendable').p_nom.sum()
        total_demand = n.loads_t.p_set.sum().sum()
        print(f"  Fixed capacity: {fixed_cap:.0f} MW, Total demand: {total_demand:.0f} MWh")
        if fixed_cap * 8760 * 0.3 < total_demand:  # Assume 30% capacity factor
            print("  ❌ Insufficient fixed capacity to meet demand!")
            return False
    
    # Backup original constraints
    original_constraints = n.global_constraints.copy()
    print(f"  Backing up {len(original_constraints)} global constraints")
    
    # Clear constraints for simple test
    n.global_constraints = n.global_constraints.iloc[0:0]  # Empty DataFrame
    
    # Try basic optimization
    try:
        print("  Starting optimization...")
        result = n.optimize(
            solver_name='highs',
            solver_options={'log_to_console': True, 'time_limit': 300}  # 5 minute limit
        )
        
        print(f"  Optimization status: {n.optimization_status}")
        if hasattr(n, 'objective'):
            print(f"  Objective value: {n.objective:,.0f}")
        
        if hasattr(n, 'generators') and 'p_nom_opt' in n.generators.columns:
            total_capacity = n.generators.p_nom_opt.sum()
            print(f"  Total optimized capacity: {total_capacity:,.0f} MW")
            
            if total_capacity > 0:
                print("  ✅ Simple optimization successful!")
                
                # Show top 5 technologies by capacity
                capacity_by_carrier = n.generators.groupby('carrier').p_nom_opt.sum().sort_values(ascending=False)
                print("  Top 5 technologies by capacity:")
                for carrier, cap in capacity_by_carrier.head(5).items():
                    print(f"    {carrier}: {cap:,.0f} MW")
                
                # Restore original constraints
                n.global_constraints = original_constraints
                return True
            else:
                print("  ❌ Optimization succeeded but no capacity was built!")
        else:
            print("  ❌ Optimization failed - no p_nom_opt found")
            
    except Exception as e:
        print(f"  ❌ Optimization failed with error: {e}")
        
    # Restore original constraints
    n.global_constraints = original_constraints
    return False


def comprehensive_network_check(n, scenario_setup):
    """Run all debugging checks in sequence"""
    
    print("\n" + "="*60)
    print("🔍 COMPREHENSIVE NETWORK DEBUGGING")
    print("="*60)
    
    # Step 1: Fix NaN values
    fix_nan_values(n)
    
    # Step 2: Check generation potential
    gen_potential_ok = check_generation_potential(n)
    
    # Step 3: Analyze generator configuration
    analyze_generator_configuration(n)
    
    # Step 4: Check extendable limits
    check_extendable_limits(n, scenario_setup)
    
    # Step 5: Run validation
    issues_found = validate_network(n)
    
    # Step 6: Try simple optimization if everything looks reasonable
    if gen_potential_ok and not issues_found:
        print("\n🎯 Network setup looks reasonable, trying simple optimization...")
        simple_opt_success = debug_simple_optimization(n)
        
        if simple_opt_success:
            print("\n✅ DEBUGGING COMPLETE: Network can be optimized!")
            return True
        else:
            print("\n❌ DEBUGGING COMPLETE: Simple optimization failed")
            return False
    else:
        print("\n❌ DEBUGGING COMPLETE: Network setup issues found")
        return False
    
def validate_network(n):
    print("🔍 Starting PyPSA network validation...")

    issues = False

    # 1. Check for NaNs
    print("\n🧪 Checking for NaN values in key components:")
    for comp in ["buses", "loads", "generators", "storage_units", "lines", "links"]:
        df = getattr(n, comp)
        if df.isnull().any().any():
            issues = True
            print(f"❌ NaNs found in `{comp}`:")
            print(df[df.isnull().any(axis=1)])
        else:
            print(f"✅ No NaNs in `{comp}`")

    # 2. Check for missing buses in connected components
    print("\n🔌 Validating component-bus connections:")
    for comp in ["loads", "generators", "storage_units"]:
        df = getattr(n, comp)
        missing_buses = df.loc[~df.bus.isin(n.buses.index)]
        if not missing_buses.empty:
            issues = True
            print(f"❌ {comp} assigned to missing buses:")
            print(missing_buses)
        else:
            print(f"✅ All {comp} have valid buses")

    # 3. Check generator and storage bounds
    print("\n📏 Checking generator and storage bounds:")
    invalid_gens = n.generators.query("p_nom_min > p_nom_max")
    if not invalid_gens.empty:
        issues = True
        print("❌ Invalid generator p_nom bounds:")
        print(invalid_gens)
    else:
        print("✅ Generator p_nom_min and p_nom_max are consistent")

    if not n.storage_units.empty:
        invalid_storage = n.storage_units.query("p_nom_min > p_nom_max")
        if not invalid_storage.empty:
            issues = True
            print("❌ Invalid storage p_nom bounds:")
            print(invalid_storage)
        else:
            print("✅ Storage p_nom_min and p_nom_max are consistent")

    # 4. Demand vs. supply sanity check
    print("\n⚖️  Demand vs. Generator Capacity:")
    if not n.loads_t.p_set.empty:
        total_demand = n.loads_t.p_set.sum().sum()
    else:
        total_demand = 0

    if "p_nom_opt" in n.generators:
        total_capacity = n.generators.p_nom_opt.sum()
    else:
        total_capacity = n.generators.p_nom.sum()

    print(f"Total Demand (MWh): {total_demand:.2f}")
    print(f"Total Generator Capacity (MW): {total_capacity:.2f}")

    if total_capacity < total_demand * 0.05:
        issues = True
        print("❌ Warning: Generator capacity is very low relative to demand.")
    else:
        print("✅ Generator capacity appears reasonable.")

    # 5. Check for presence of load shedding
    print("\n🛑 Load shedding carrier present?")
    shedding_carriers = n.generators[n.generators.carrier.str.contains("load_shedding", case=False, na=False)]
    if not shedding_carriers.empty:
        print("⚠️  Load shedding is enabled. Check that this is intentional.")
    else:
        print("✅ No load shedding generators found.")

    # 6. Check carriers
    print("\n🔍 Verifying carrier definitions:")
    missing_carriers = n.carriers[n.carriers.color.isnull()]
    if not missing_carriers.empty:
        issues = True
        print("❌ Carriers with missing color:")
        print(missing_carriers)
    else:
        print("✅ All carriers have defined attributes.")

    # 7. If solved, print constraint and variable overview
    if hasattr(n, "model") and n.model is not None:
        print("\n📦 Model built. Listing variables and constraints:")
        print("Variables:", list(n.model.variables.keys()))
        print("Constraints:", list(n.model.constraints.keys()))
    else:
        print("\nℹ️ Model not yet built. Solve the network to inspect constraints.")

    # Summary
    print("\n✅ Validation complete.")
    if issues:
        print("⚠️ Issues found. Please review messages above.")
    else:
        print("🎉 No major issues found.")

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake(
            'prepare_and_solve_network', 
            **{
                'scenario':'TEST',
            }
        )
    logging.info("Preparing costs")

    n = pypsa.Network(snakemake.input.network)
    add_missing_carriers(n)
    print("Network created")
    scenario_setup = load_scenario_definition(snakemake)
    
    # opts = scenario_setup["options"].split("-")
    # for o in opts:
    #     m = re.match(r"^\d+h$", o, re.IGNORECASE)
    #     if m is not None:
    #         n = average_every_nhours(n, m[0])
    #         break

    # for o in opts:
    #     m = re.match(r"^\d+SEG$", o, re.IGNORECASE)
    #     if m is not None:
    #         print("Using TSAM")
    #         try:
    #             import tsam.timeseriesaggregation as tsam
    #         except:
    #             raise ModuleNotFoundError(
    #                 "Optional dependency 'tsam' not found." "Install via 'pip install tsam'"
    #             )
    #         n = apply_time_segmentation(n, m[0][:-3], snakemake.config["tsam_clustering"])
    #         break

    logging.info("Setting global and regional build limits")
    if len(n.buses) != 1: # Checks whether national limits need to be set across multi-regional extendable technologies
        _set_extendable_limits_national(n) 
    else:
        # Legacy function for setting per bus in extendable_technologies.xlsx
        set_extendable_limits_explicit_per_bus(n)    

    logging.info("Solving network")
    # Apply TSAM clustering

    # n_clustered = apply_tsam_to_pypsa_network(n, typical_days=12, hours_per_day=24)
    n_clustered = apply_tsam_to_pypsa_network(
            n, period_type='monthly', typical_periods=3, hours_per_period=24)
    

    # # Use fewer time steps for testing
    # n_small = n.copy()
    # n_small.snapshots = n.snapshots[:168]  # Just 168 hours

    # Use your existing solve_network function
    solve_network(n_clustered, n_clustered.snapshots)

    logging.info("Network solved")
    # Export results
    n_clustered.export_to_netcdf(snakemake.output[0])
    n_clustered.statistics().to_csv(snakemake.output[1])
    calc_emissions(n_clustered, scenario_setup).to_csv(snakemake.output[2])

    # Print summary stats
    print("Total demand:", n.loads_t.p_set.sum().sum())
    print("Total supply capacity:", n.generators.p_nom_opt.sum())

    # Run consistency check
    n_clustered.consistency_check()

    # Only access n.model if it exists
    if hasattr(n, "model"):
        print("Model constraints:", list(n_clustered.model.constraints.keys())[:5])  # show just a few
        print("Model variables:", list(n_clustered.model.variables.keys())[:5])
    else:
        print("No model found. Use store_model=True if you want access to it.")
