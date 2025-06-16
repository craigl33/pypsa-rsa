import logging
import pandas as pd
import pypsa
from pypsa.descriptors import get_switchable_as_dense as get_as_dense, expand_series, get_activity_mask
import os
import numpy as np



from _helpers import get_investment_periods
# from add_electricity import load_costs, update_transmission_costs

import xarray as xr
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Comment out for debugging and development

idx = pd.IndexSlice
logger = logging.getLogger(__name__)


"""
********************************************************************************
    Operational limits
********************************************************************************
"""

def calc_max_gen_potential(n, sns, gens, incl_pu, weightings, active, cf_limit, extendable=False):
    """
    Calculate maximum generation potential for capacity factor constraints.
    Updated for PyPSA 0.34.1 Linopy approach.
    """
    suffix = "" if extendable == False else "-ext"
    p_max_pu = get_as_dense(n, 'Generator', "p_max_pu", sns)[gens] if incl_pu else pd.DataFrame(1, index=sns, columns=gens)
    p_max_pu.columns.name = f'Generator{suffix}'
    
    if n.multi_invest:
        cf_limit_h = pd.DataFrame(0, index=sns, columns=gens)
        for y in cf_limit_h.index.get_level_values(0).unique():
            cf_limit_h.loc[y] = cf_limit[y]
    else:
        cf_limit_h = pd.DataFrame(cf_limit[sns[0].year], index=sns, columns=gens) * weightings[gens]
    
    if not extendable:
        return cf_limit_h[gens] * active[gens] * p_max_pu * weightings[gens] * n.generators.loc[gens, "p_nom"]
    
    # Updated for Linopy: Access capacity variables through the model
    p_nom_var_name = f"Generator-p_nom" if not extendable else f"Generator-p_nom"
    if p_nom_var_name not in n.model.variables:
        logging.warning(f"Variable {p_nom_var_name} not found in model variables")
        return 0
    
    p_nom = n.model.variables[p_nom_var_name]
    if extendable:
        # Select only extendable generators
        try:
            p_nom = p_nom.sel({f"Generator-ext": gens})
        except KeyError:
            # Fallback if dimension naming is different
            p_nom = p_nom.sel(Generator=gens) if "Generator" in p_nom.dims else p_nom
    
    potential = xr.DataArray(cf_limit_h[gens] * active[gens] * p_max_pu * weightings[gens])
    
    # Ensure dimension alignment for Linopy
    if extendable and "Generator" in potential.dims:
        potential = potential.rename({"Generator": "Generator-ext"})
    
    return (potential * p_nom).sum(f'Generator{suffix}')

def group_and_sum(data, groupby_func):
    grouped_data = data.groupby(groupby_func).sum()
    return grouped_data.sum(axis=1) if len(grouped_data) > 1 else grouped_data
    
def apply_operational_constraints(n, sns, **kwargs):
    """
    Apply operational constraints using the new Linopy approach.
    Updated for PyPSA 0.34.1.
    """
    # Check if model is created
    if not hasattr(n, 'model') or n.model is None:
        logging.warning("Network model not created yet. Skipping operational constraints.")
        return
    
    energy_unit_conversion = {"GW":1e3, "GJ": 1/3.6, "TJ": 1000/3.6, "PJ": 1e6/3.6, "GWh": 1e3, "TWh": 1e6}
    apply_to = kwargs["apply_to"]

    carrier = [c.strip() for c in kwargs["carrier"].split("+")]

    bus = kwargs["bus"]
    period = kwargs["period"]

    type_ = "energy_power" if kwargs["type"] in ["primary_energy", "output_energy", "output_power"] else "capacity_factor"

    if (period  == "week") & (max(n.snapshot_weightings["generators"])>1):
        logger.warning(
            "Applying weekly operational limits and time segmentation should be used with caution as the snapshot weightings might not align with the weekly grouping."
        )
    incl_pu = kwargs["incl_pu"]
    limit = kwargs["limit"]

    sense = "<=" if limit == "max" else ">="

    cf_limit = 0 * kwargs["values"] if type_ == "energy_power" else kwargs["values"]
    en_pow_limit = 0 * kwargs["values"] if type_ == "capacity_factor" else kwargs["values"]

    if ((kwargs["type"] in ["primary_energy", "output_energy"]) & (kwargs["units"] != "MWh")) or ((kwargs["type"] == "output_power") & (kwargs["units"] != "MW")):
        en_pow_limit *= energy_unit_conversion[kwargs["units"]]

    years = get_investment_periods(sns, n.multi_invest)

    filtered_gens = n.generators.query("carrier in @carrier") if len(carrier)>1 else n.generators.query("carrier == @carrier")
    if bus != "global":
        filtered_gens = filtered_gens.query("bus == @bus")
    fix_i = filtered_gens.query("not p_nom_extendable").index if apply_to in ["fixed", "all"] else []
    ext_i = filtered_gens.query("p_nom_extendable").index if apply_to in ["extendable", "all"] else []
    filtered_gens = filtered_gens.loc[list(fix_i) + list(ext_i)]

    if len(filtered_gens) == 0:
        return

    # Check if Generator-p variable exists
    if "Generator-p" not in n.model.variables:
        logging.warning("Generator-p variable not found in model. Skipping operational constraints.")
        return

    efficiency = get_as_dense(n, "Generator", "efficiency", inds=filtered_gens.index) if kwargs["type"] == "primary_energy" else pd.DataFrame(1, index=n.snapshots, columns = filtered_gens.index)
    weightings = (1/efficiency).multiply(n.snapshot_weightings.generators, axis=0)

    # if only extendable generators only select snapshots where generators are active
    min_year = n.generators.loc[filtered_gens.index, "build_year"].min()
    sns_active = sns[sns.get_level_values(0) >= min_year] if n.multi_invest else sns[sns.year >= min_year]
    
    # Access generator dispatch variables using Linopy approach
    gen_p_var = n.model.variables['Generator-p']
    
    # Select relevant generators and snapshots
    try:
        gen_p_filtered = gen_p_var.sel(Generator=filtered_gens.index, snapshot=sns_active)
        
        # Apply weightings and sum over generators
        weightings_aligned = xr.DataArray(
            weightings.loc[sns_active, filtered_gens.index],
            dims=['snapshot', 'Generator'],
            coords={'snapshot': sns_active, 'Generator': filtered_gens.index}
        )
        
        act_gen = (gen_p_filtered * weightings_aligned).sum('Generator')
        act_gen_pow = gen_p_filtered.sum('Generator')
        
    except (KeyError, ValueError) as e:
        logging.warning(f"Error accessing generator variables: {e}. Skipping constraint.")
        return

    timestep = "timestep" if n.multi_invest else "snapshot"
    groupby_dict = {
        "year": f"{timestep}.year",
        "month": f"{timestep}.month",
        "week": f"{timestep}.week",
        "hour": None
    }

    active = get_activity_mask(n, "Generator", sns).astype(int)
    if type_ != "energy_power":
        max_gen_fix = calc_max_gen_potential(n, sns, fix_i, incl_pu, weightings, active, cf_limit, extendable=False) if len(fix_i)>0 else 0
        max_gen_ext = calc_max_gen_potential(n, sns, ext_i, incl_pu, weightings, active, cf_limit, extendable=True) if len(ext_i)>0 else 0

    if groupby := groupby_dict[period]:
        for y in years:
            year_sns = sns_active[sns_active.get_level_values(0)==y] if n.multi_invest else sns_active
            if len(year_sns) > 0:
                if type_ == "capacity_factor":
                    lhs = (act_gen - max_gen_ext) 
                    if (isinstance(max_gen_fix, int)) | (isinstance(max_gen_fix, float)):
                        rhs = max_gen_fix
                    else:
                        rhs = max_gen_fix.loc[y] if n.multi_invest else max_gen_fix.loc[year_sns]
                else:
                    lhs = act_gen
                    rhs = en_pow_limit[y]
                    
                lhs = lhs.sel(snapshot=year_sns)
                lhs_p = lhs.sum() if period == "year" else lhs.groupby(groupby).sum()

                rhs_p = rhs if isinstance(rhs, int) else rhs.sum().sum()
                
                # Add constraint using Linopy method
                n.model.add_constraints(lhs_p, sense, rhs_p, name=f'{limit}-{kwargs["carrier"]}-{period}-{kwargs["apply_to"][:3]}-{y}')

    else:
        lhs = (act_gen - max_gen_ext).sel(snapshot = sns_active) if type_ == "capacity_factor" else act_gen_pow.sel(snapshot = sns_active)
        if kwargs["type"] == "output_energy":
            logging.warning("Energy limits are not yet implemented for hourly operational limits.")
            return
    
        if type_ == "capacity_factor":
            rhs = (
                max_gen_fix
                if isinstance(max_gen_fix, int)
                else xr.DataArray(
                    max_gen_fix.loc[sns_active].sum(axis=1)
                ).rename({"dim_0": "snapshot"})
            )
        else:
            rhs = pd.Series(index = sns)
            for y in years:
                rhs.loc[y] = en_pow_limit[y]
                
        n.model.add_constraints(lhs, sense, rhs, name = f'{limit}-{kwargs["carrier"]}-hour-{kwargs["apply_to"][:3]}')

def set_operational_limits(n, sns, scenario_setup):
    """
    Set operational limits from Excel configuration.
    Updated for PyPSA 0.34.1.
    """
    if not hasattr(n, 'model') or n.model is None:
        logging.warning("Network model not created yet. Skipping operational limits.")
        return

    op_limits = pd.read_excel(
        os.path.join(scenario_setup["sub_path"], "operational_constraints.xlsx"),
        sheet_name='operational_constraints',
        index_col=list(range(9)),
    )
    
    if scenario_setup["operational_limits"] not in op_limits.index.get_level_values(0).unique():
        return
    op_limits = op_limits.loc[scenario_setup["operational_limits"]]

    #drop rows where all NaN
    op_limits = op_limits.loc[~(op_limits.isna().all(axis=1))]
    for idx, row in op_limits.iterrows():
        apply_operational_constraints(
            n, sns, 
            bus = idx[0], carrier = idx[1], 
            type = idx[2], values = row, 
            period = idx[3], incl_pu = idx[4],
            limit = idx[5], apply_to = idx[6],
            units = idx[7],
        )


def ccgt_steam_constraints(n, sns, snakemake):
    """
    CCGT steam constraints using Linopy approach.
    Updated for PyPSA 0.34.1.
    """
    # Check if model is created
    if not hasattr(n, 'model') or n.model is None:
        logging.warning("Network model not created yet. Skipping CCGT steam constraints.")
        return
    
    # Check if Generator-p variable exists
    if "Generator-p" not in n.model.variables:
        logging.warning("Generator-p variable not found in model. Skipping CCGT steam constraints.")
        return
    
    # At each bus HRSG power is limited by what OCGT power production
    config = snakemake.config["electricity"]["conventional_generators"]
    p_nom_ratio = config["ccgt_st_to_gt_ratio"]
    ocgt_carriers = n.generators.carrier[(n.generators.carrier.str.contains("ocgt"))].unique()

    # remove ocgt_diesel_emg from ocgt_carriers
    ocgt_carriers = [c for c in ocgt_carriers if c in config["allowable_ocgt_st_carriers"]]

    gen_p_var = n.model.variables['Generator-p']
    
    for bus in n.buses.index:
        ocgt_gens = n.generators.query("bus == @bus & carrier in @ocgt_carriers").index
        ccgt_hrsg = n.generators.query("bus == @bus & carrier == 'ccgt_steam'").index
        
        if len(ocgt_gens) == 0 or len(ccgt_hrsg) == 0:
            continue
        
        try:
            # Access generator variables for specific generators and snapshots
            ccgt_p = gen_p_var.sel(Generator=ccgt_hrsg, snapshot=sns)
            ocgt_p = gen_p_var.sel(Generator=ocgt_gens, snapshot=sns)
            
            # Create constraint: CCGT steam <= ratio * OCGT gas
            lhs = (ccgt_p.sum("Generator") - p_nom_ratio * ocgt_p.sum("Generator"))
            rhs = 0
            
            n.model.add_constraints(lhs, "<=", rhs, name=f'ccgt_steam_limit-{bus}')
            
        except (KeyError, ValueError) as e:
            logging.warning(f"Error creating CCGT steam constraint for bus {bus}: {e}")
            continue

"""
********************************************************************************
    Reserve margin
********************************************************************************
"""
def check_active(n, c, y, list):
    """Helper function to check active assets for multi-investment periods."""
    active = n.df(c).index[n.get_active_assets(c, y)] if n.multi_invest else list
    return list.intersection(active)

def reserve_margin_constraints(n, sns, scenario_setup, snakemake):
    """
    Reserve margin constraints using Linopy approach.
    Updated for PyPSA 0.34.1.
    """
    # Check if model is created
    if not hasattr(n, 'model') or n.model is None:
        logging.warning("Network model not created yet. Skipping reserve margin constraints.")
        return
    
    ###################################################################################
    # Reserve margin above maximum peak demand in each year
    # The sum of res_margin_carriers multiplied by their assumed contribution factors 
    # must be higher than the maximum peak demand in each year by the reserve_margin value

    res_margin = pd.read_excel(
        os.path.join(scenario_setup["sub_path"], "reserve_margin.xlsx"), 
        sheet_name="reserve_margin",
        index_col=[0,1]).loc[scenario_setup["reserve_margin"]].drop("unit", axis=1)

    capacity_credit = pd.read_excel(
            os.path.join(scenario_setup["sub_path"], "reserve_margin.xlsx"), 
            sheet_name="capacity_credits",
            index_col=[0])[scenario_setup["capacity_credits"]]

    res_mrgn_active = res_margin.loc["reserve_margin_active"]
    res_mrgn = res_margin.loc["reserve_margin"]

    peak = n.loads_t.p_set.loc[sns].sum(axis=1).groupby(sns.get_level_values(0)).max() if n.multi_invest else n.loads_t.p_set.loc[sns].sum(axis=1).max()
    peak = peak if n.multi_invest else pd.Series(peak, index = sns.year.unique())

    for y in peak.index:
        if res_mrgn_active[y]:    
            fix_cap = 0
            lhs = 0
            
            for c in ["Generator", "StorageUnit"]:
                # Fixed capacity contribution
                fix_i = n.df(c).query("not p_nom_extendable & carrier in @capacity_credit.index").index
                fix_i = check_active(n, c, y, fix_i)

                fix_cap += (
                    n.df(c).loc[fix_i, "carrier"].map(capacity_credit)
                    * n.df(c).loc[fix_i, "p_nom"]
                ).sum()
            
                # Extendable capacity contribution
                ext_i = n.df(c).query("p_nom_extendable & carrier in @capacity_credit.index").index
                ext_i = check_active(n, c, y, ext_i)
                
                if len(ext_i) > 0:
                    # Access capacity variables using Linopy approach
                    var_name = f"{c}-p_nom"
                    if var_name in n.model.variables:
                        try:
                            p_nom_var = n.model.variables[var_name]
                            
                            # Select extendable components
                            if f"{c}-ext" in p_nom_var.dims:
                                p_nom_ext = p_nom_var.sel({f"{c}-ext": ext_i})
                                dim_name = f"{c}-ext"
                            elif c in p_nom_var.dims:
                                p_nom_ext = p_nom_var.sel({c: ext_i})
                                dim_name = c
                            else:
                                logging.warning(f"Cannot find appropriate dimension for {c} in {var_name}")
                                continue
                            
                            # Create capacity credit array
                            capacity_credit_array = xr.DataArray(
                                n.df(c).loc[ext_i, "carrier"].map(capacity_credit),
                                dims=[dim_name],
                                coords={dim_name: ext_i}
                            )
                            
                            lhs += (p_nom_ext * capacity_credit_array).sum(dim_name)
                            
                        except (KeyError, ValueError) as e:
                            logging.warning(f"Error accessing {var_name} for reserve margin: {e}")
                            continue

            rhs = peak.loc[y] * (1 + res_mrgn[y]) - fix_cap 

            if lhs != 0:  # Only add constraint if there are extendable assets
                n.model.add_constraints(lhs, ">=", rhs, name=f"reserve_margin_{y}")    

def annual_co2_constraints(n, sns, param, scenario_setup):
    """
    Annual CO2 constraints using Linopy approach.
    Updated for PyPSA 0.34.1.
    """
    # Check if model is created
    if not hasattr(n, 'model') or n.model is None:
        logging.warning("Network model not created yet. Skipping CO2 constraints.")
        return

    if scenario_setup["co2_constraints"] in ["None", "none", ""]:
        return

    # Check if Generator-p variable exists
    if "Generator-p" not in n.model.variables:
        logging.warning("Generator-p variable not found in model. Skipping CO2 constraints.")
        return

    gen_emissions = param.loc["co2_emissions"].drop("unit", axis=1)
    gen_emissions = gen_emissions[gen_emissions.mean(axis=1) > 0]

    # Get relevant generators
    relevant_gens = n.generators.query("carrier in @gen_emissions.index").index
    if len(relevant_gens) == 0:
        logging.info("No generators with CO2 emissions found. Skipping CO2 constraints.")
        return

    try:
        # Access generator dispatch variables
        gen_p = n.model.variables['Generator-p']
        gen_p_filtered = gen_p.sel(Generator=relevant_gens)

        # Create emissions coefficient array
        co2_emissions = xr.DataArray(coords=gen_p_filtered.coords, dims=gen_p_filtered.dims)

        for gen in relevant_gens:
            carrier = n.generators.loc[gen, "carrier"]
            efficiency = n.generators.loc[gen, "efficiency"]
            
            if n.multi_invest:
                for y in n.investment_periods:
                    # Select snapshots for this investment period
                    period_snapshots = sns[sns.get_level_values(0) == y]
                    if len(period_snapshots) > 0:
                        emission_factor = gen_emissions.loc[carrier, y] / efficiency
                        co2_emissions.loc[dict(snapshot=period_snapshots, Generator=gen)] = emission_factor
            else:
                # Single investment period
                y = sns[0].year if hasattr(sns[0], 'year') else list(gen_emissions.columns)[0]
                emission_factor = gen_emissions.loc[carrier, y] / efficiency
                co2_emissions.loc[dict(Generator=gen)] = emission_factor

        # Calculate total emissions
        total_emissions = (gen_p_filtered * co2_emissions).sum("Generator")
        
        if n.multi_invest:
            # Group by investment period
            lhs = total_emissions.groupby("snapshot.year").sum()
        else:
            # Sum over all snapshots for annual constraint
            lhs = total_emissions.sum("snapshot")

        # Read annual limits
        annual_limits = pd.read_excel(
            os.path.join(scenario_setup["sub_path"], "carbon_constraints.xlsx"), 
            sheet_name="annual_carbon_constraint",
            index_col=[0]).loc[scenario_setup["co2_constraints"]]

        # Unit conversion
        conv = 1
        unit = annual_limits.get("unit", "t")
        if isinstance(unit, str):
            if unit.startswith("Mt"):
                conv = 1e6
            elif unit.startswith("Gt"):
                conv = 1e9

        if n.multi_invest:
            rhs = (annual_limits.loc[n.investment_periods] * conv)
            rhs.index.name = "year"  # Match the groupby dimension
        else:
            # Single period
            y = sns[0].year if hasattr(sns[0], 'year') else list(annual_limits.index)[0]
            rhs = annual_limits.loc[y] * conv

        n.model.add_constraints(lhs, "<=", rhs, name='annual_co2_limits')
        
    except (KeyError, ValueError) as e:
        logging.warning(f"Error creating CO2 constraints: {e}")
        return
    
def add_national_capacity_constraints(n, sns, scenario_setup):
    """
    Add custom constraints that enforce national capacity limits
    across all regional variants of each technology.
    
    This function should be called during the optimization setup
    (e.g., in the extra_functionality of prepare_and_solve_network.py).
    """
    
    if not hasattr(n, 'model') or n.model is None:
        logging.warning("Network model not created yet. Skipping national capacity constraints.")
        return
    
    # Add constraints for both generators and storage units
    for component_type in ["Generator", "StorageUnit"]:
        add_national_constraints_for_component(n, sns, component_type, scenario_setup)


def add_national_constraints_for_component(n, sns, component_type, scenario_setup):
    """
    Add national constraints for a specific component type.
    """
    
    constraint_key = f"national_constraints_{component_type}"
    
    # Get stored national constraints
    if hasattr(scenario_setup, '_national_constraints') and constraint_key in scenario_setup._national_constraints:
        national_constraints = scenario_setup._national_constraints[constraint_key]
    elif constraint_key in globals():
        national_constraints = globals()[constraint_key]
    else:
        logging.warning(f"No national constraints found for {component_type}")
        return
    
    # Get component dataframe
    if component_type == "Generator":
        components = n.generators
        p_nom_var_name = "Generator-p_nom"
    elif component_type == "StorageUnit":
        components = n.storage_units
        p_nom_var_name = "StorageUnit-p_nom"
    else:
        return
    
    # Check if the variable exists in the model
    if p_nom_var_name not in n.model.variables:
        logging.warning(f"{p_nom_var_name} variable not found in model")
        return
    
    p_nom_var = n.model.variables[p_nom_var_name]
    
    # Group regional technologies by base carrier
    carrier_groups = group_regional_technologies_by_base_carrier(components)
    
    # Add constraints for each constraint type and carrier
    for constraint_type in national_constraints.index.get_level_values(0).unique():
        constraint_data = national_constraints.xs(constraint_type, level=0)
        
        for carrier in constraint_data.index:
            if carrier in carrier_groups:
                add_carrier_national_constraint(
                    n, p_nom_var, carrier_groups[carrier], 
                    constraint_data.loc[carrier], constraint_type, carrier, component_type
                )
