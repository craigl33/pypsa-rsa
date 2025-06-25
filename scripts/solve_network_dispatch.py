# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Solves linear optimal dispatch in hourly resolution using the capacities of
previous capacity expansion in rule :mod:`solve_network`.
"""


import logging
import pandas as pd
import numpy as np
import pypsa
from xarray import DataArray
#from _helpers import configure_logging, update_config_with_sector_opts
#from solve_network import prepare_network, solve_network
from pypsa.descriptors import get_switchable_as_dense as get_as_dense, get_activity_mask
from scripts.export_to_sienna import export_pypsa_to_sienna

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
    apply_default_attr
)

from prepare_and_solve_network import (
    set_operational_limits,
    ccgt_steam_constraints,
    solve_network
    # rmippp_constraints,
)

import os

def get_min_stable_level(n, model_setup, existing_carriers, extended_carriers):
    
    
    existing_param = pd.read_excel(
        os.path.join(model_setup["sub_path"], "fixed_technologies.xlsx"), 
        sheet_name="conventional",
        na_values=["-"],
    )
    existing_param=existing_param.set_index(["Scenario", "Power Station Name"]).loc[model_setup["fixed_conventional"]]
    
    existing_gens = n.generators.query("carrier in @existing_carriers & p_nom_extendable == False").index
    existing_msl= existing_param.loc[existing_gens, "Min Stable Level (%)"].rename("p_min_pu")
    
    extended_param = pd.read_excel(
        os.path.join(model_setup["sub_path"], "extendable_technologies.xlsx"), 
        sheet_name = "parameters"
    )

    extended_param = extended_param.set_index( ["scenario", "parameter", "carrier"]
    ).sort_index()
    
    ## Extended paramaeters are actually either default or different for AMBITIONS
    try:
        extended_param = extended_param.loc[model_setup["extendable_techs"]]
    except KeyError:
        logger.info("No extendable technologies defined for this scenario. Using default settings")
        extended_param = extended_param.loc["default"]

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
    """
    Set a constraint on the status of generators to not exceed the maximum availability
    given by p_max_pu. The first 100 hours of the simulation are considered as an
    initialization period, where the generators are fully available. This is done to
    avoid the model from crashing if the generators are not available at all.
    """
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

def set_max_status(n, sns, p_max_pu):
    
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

def set_existing_committable(n, sns, model_setup, config):

    existing_carriers = config['existing']
    existing_gen = n.generators.query("carrier in @existing_carriers & p_nom_extendable == False").index.to_list()

    extended_carriers = config['extended']
    extended_gen = n.generators.query("carrier in @extended_carriers & p_nom_extendable").index.to_list()
    
    n.generators.loc[existing_gen + extended_gen, "committable"] = True

    p_max_pu = get_as_dense(n, "Generator", "p_max_pu", sns)[existing_gen + extended_gen].copy()
    n.generators_t.p_max_pu.loc[:, existing_gen + extended_gen] = 1
    n.generators.loc[existing_gen + extended_gen, "p_max_pu"] = 1

    existing_msl, extended_msl = get_min_stable_level(n, model_setup, existing_carriers, extended_carriers)

    n.generators.loc[existing_gen, "p_min_pu"] = existing_msl
    n.generators.loc[extended_gen, "p_min_pu"] = extended_msl

    return p_max_pu

from prepare_and_solve_network import (
    set_operational_limits,
    ccgt_steam_constraints,
    reserve_margin_constraints,
    load_extendable_parameters,
    annual_co2_constraints,
    add_national_capacity_constraints,
)

from _helpers import load_scenario_definition

def add_operational_constraints(n, snapshots):
    """
    Add operational constraints to the network model for dispatch optimization.

    This function applies a series of constraints to the network model using Linopy.
    It includes constraints specific to dispatch optimization such as unit commitment,
    operational limits, and CO2 constraints. These constraints are tailored to ensure
    the network model adheres to both technical and regulatory requirements during 
    the optimization process.

    Parameters:
    n : pypsa.Network
        The PyPSA network object containing all components and their attributes.
    snapshots : list or pd.Index
        The time snapshots for which the constraints are to be applied.

    Custom Constraints:
    - Unit Commitment: Applied to generators marked as committable.
    - Operational Limits: Sets operational boundaries for the network.
    - CCGT Steam Constraints: Ensures compliance with CCGT operational parameters.
    - Annual CO2 Constraints: Enforces CO2 emission limits annually.

    Note:
    Reserve margin constraints are not included as they are not needed for dispatch.
    """

    if hasattr(n, 'generators') and 'committable' in n.generators.columns:
        committable_gens = n.generators[n.generators.committable == True].index
        if len(committable_gens) > 0:
            # Add your unit commitment constraints here
            set_max_status(n, snapshots, p_max_pu)  # from your dispatch script
            

    set_operational_limits(n, snapshots, scenario_setup)
    ccgt_steam_constraints(n, snapshots, snakemake)
    # reserve_margin_constraints(n, snapshots, scenario_setup, snakemake) # Not needed for dispatch
    
    param = load_extendable_parameters(n, scenario_setup, snakemake)
    annual_co2_constraints(n, snapshots, param, scenario_setup)

def solve_network_dispatch(n, sns, enable_unit_commitment=False, export_to_Sienna=False):
    """
    Solve network using the new Linopy-based optimization approach.
    
    This follows the PyPSA-EUR v2025.04.0 pattern:
    1. Create the optimization model using the new API
    2. Add custom constraints via extra_functionality 
    3. Solve the model
    """
    
    def extra_functionality(n, snapshots):
        """
        Add custom constraints to the model specifically for dispatch optimisation.
        This function is called after model creation but before solving.
        """
        # Custom constraints using the new Linopy-based approach
        # DISPATCH-SPECIFIC: Add unit commitment constraints if needed
        if hasattr(n, 'generators') and 'committable' in n.generators.columns:
            committable_gens = n.generators[n.generators.committable == True].index
            if len(committable_gens) > 0:
                # Add your unit commitment constraints here
                set_max_status(n, snapshots, p_max_pu)  # from your dispatch script
                

        set_operational_limits(n, snapshots, scenario_setup)
        ccgt_steam_constraints(n, snapshots, snakemake)
        # reserve_margin_constraints(n, snapshots, scenario_setup, snakemake) # Not needed for dispatch
        
        param = load_extendable_parameters(n, scenario_setup, snakemake)
        annual_co2_constraints(n, snapshots, param, scenario_setup)
        
        # Add national capacity constraints for regional technologies
        # add_national_capacity_constraints(n, snapshots, scenario_setup) # Not needed for dispatch
    
    if enable_unit_commitment:
        # Set up committable generators (from your existing code)
        config = snakemake.config["electricity"]["dispatch_committable_carriers"]
        p_max_pu = set_existing_committable(n, sns, model_setup, config)
        
        # Store for use in extra_functionality
        n._dispatch_constraints = {'unit_commitment': True, 'p_max_pu': p_max_pu}

    if not export_to_Sienna:

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
            extra_functionality=extra_functionality,
            linearized_unit_commitment=True  # Add this for dispatch with UC
        )
    else:
        logging.info("Exporting to Sienna format is not yet implemented.")
        raise NotImplementedError

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        # snakemake = mock_snakemake(
        #     'solve_network_dispatch', 
        #     **{
        #         'model_file':'TEST',
        #         'regions':'10',
        #         'resarea':'redz',
        #         'll':'copt',
        #         'opts':'LC',
        #         'years':'2024',
        #     }
        # )

        snakemake = mock_snakemake(
            'solve_network_dispatch', 
            **{
                'scenario':'TEST',
                'year':2030
            }
        )

    n = pypsa.Network(snakemake.input.dispatch_network)

    scenario_setup = load_scenario_definition(snakemake)
    
    config = snakemake.config["electricity"]["dispatch_committable_carriers"]
    p_max_pu = set_existing_committable(n=n, sns=n.snapshots, model_setup=scenario_setup, config=config)
    n_dispatch = pypsa.Network(snakemake.input.dispatch_network)
    
    solve_network_dispatch(n_dispatch, n_dispatch.snapshots)
    n.export_to_netcdf(snakemake.dispatch_results)
    logging.info(f"Exported to {snakemake.dispatch_results}")
