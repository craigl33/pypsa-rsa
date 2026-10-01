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

def single_year_segmentation(n, snapshots, segments, config):

    p_max_pu, p_max_pu_max = normalize_and_rename_df(n.generators_t.p_max_pu, snapshots, 1, 'max')
    load, load_max = normalize_and_rename_df(n.loads_t.p_set, snapshots, 1, "load")
    inflow, inflow_max = normalize_and_rename_df(n.storage_units_t.inflow, snapshots, 0, "inflow")

    raw = pd.concat([p_max_pu, load, inflow], axis=1, sort=False)

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
    
    # Get solver configuration
    solver_name = snakemake.config["solving"]["solver"].pop("name")
    solver_options = snakemake.config["solving"]["solver"].copy()
    
    # Solve using the new optimize method with extra_functionality
    # This is the PyPSA 0.34.1 way following PyPSA-EUR patterns
    n.optimize(
        snapshots=sns,
        multi_investment_periods=n.multi_invest,
        solver_name=solver_name,
        solver_options=solver_options,
        extra_functionality=extra_functionality
    )

"""
********************************************************************************
    DEBUG FUNCTIONS - Added for comprehensive network debugging
********************************************************************************
"""

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

def validate_network2(n):
    print("Running basic network validation checks...")

    # 1. Check for empty components
    empty_components = [c for c in n.iterate_components() if c.df.empty]
    if empty_components:
        print(f"❗ Empty components found: {[c.name for c in empty_components]}")
    else:
        print("✅ No empty components.")

    # 2. Check missing time series for loads/generators
    def check_missing_profiles(df, kind):
        if 'p_set' in df and df['p_set'].isnull().all().all():
            print(f"❗ {kind} has completely missing p_set time series.")

    check_missing_profiles(n.loads_t.p_set, "Loads")
    check_missing_profiles(n.generators_t.p_max_pu, "Generators (p_max_pu)")
    check_missing_profiles(n.generators_t.p_min_pu, "Generators (p_min_pu)")
    
    # 3. Check buses with no connections
    isolated_buses = n.buses.index.difference(
        pd.concat([n.lines.bus0, n.lines.bus1, n.links.bus0, n.links.bus1, 
                n.loads.bus, n.generators.bus, n.storage_units.bus])
    )
    if not isolated_buses.empty:
        print(f"❗ Isolated buses with no connections: {list(isolated_buses)}")
    else:
        print("✅ All buses are connected.")

    # 4. Check carrier emission attribute (for cost/emission constraints)
    if n.carriers.empty:
        print("❗ No carriers defined.")
    elif "co2_emissions" not in n.carriers.columns:
        print("❗ `co2_emissions` column missing in carriers.")
    else:
        print("✅ Carrier emission data present.")

    # 5. Check for bounds on p_nom (capacity) that might be too strict
    if "p_nom_max" in n.generators.columns:
        too_low = n.generators.query("p_nom_max <= 0")
        if not too_low.empty:
            print(f"❗ Generators with p_nom_max <= 0:\n{too_low[['p_nom_max']]}")
    
    if "p_nom_min" in n.generators.columns:
        too_high = n.generators.query("p_nom_min > p_nom_max")
        if not too_high.empty:
            print("❗ Some generators have p_nom_min > p_nom_max")

    # 6. Storage energy capacity checks
    if not n.storage_units.empty:
        missing_e_nom = n.storage_units.query("e_nom_extendable & e_nom_max <= 0")
        if not missing_e_nom.empty:
            print("❗ Storage units with e_nom_max <= 0 despite being extendable.")

    print("🔍 Validation complete.")

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

    n = pypsa.Network(snakemake.input[0])
    add_missing_carriers(n)
    print("Network created")
    scenario_setup = load_scenario_definition(snakemake)
    
    opts = scenario_setup["options"].split("-")
    for o in opts:
        m = re.match(r"^\d+h$", o, re.IGNORECASE)
        if m is not None:
            n = average_every_nhours(n, m[0])
            break

    for o in opts:
        m = re.match(r"^\d+SEG$", o, re.IGNORECASE)
        if m is not None:
            print("Using TSAM")
            try:
                import tsam.timeseriesaggregation as tsam
            except:
                raise ModuleNotFoundError(
                    "Optional dependency 'tsam' not found." "Install via 'pip install tsam'"
                )
            n = apply_time_segmentation(n, m[0][:-3], snakemake.config["tsam_clustering"])
            break

    logging.info("Setting global and regional build limits")
    if len(n.buses) != 1: # Checks whether national limits need to be set across multi-regional extendable technologies
        _set_extendable_limits_national(n)
    else:
        # Legacy function for setting per bus in extendable_technologies.xlsx
        set_extendable_limits_explicit_per_bus(n)

    comprehensive_network_check(n, scenario_setup)