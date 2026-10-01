import pandas as pd
import numpy as np
import logging
import os

# Regional mapping for 10-region South Africa model
REGIONAL_MAPPING = {
    'Eastern Cape': {'code': 'EC', 'index': 0},
    'Free State': {'code': 'FS', 'index': 1},
    'Gauteng': {'code': 'GP', 'index': 2},
    'Hydra Central': {'code': 'HY', 'index': 3},
    'KwaZulu Natal': {'code': 'ZN', 'index': 4},
    'Limpopo': {'code': 'LP', 'index': 5},
    'Mpumalanga': {'code': 'MP', 'index': 6},
    'North West': {'code': 'NW', 'index': 7},
    'Northern Cape': {'code': 'NC', 'index': 8},
    'Western Cape': {'code': 'WC', 'index': 9}
}

def define_extendable_tech_with_national_constraints(carriers, years, type_, ext_param):
    """
    Create regional extendable technologies with equal individual limits,
    then apply national constraints separately.
    
    This approach:
    1. Creates equal extendable capacity for each carrier in each region
    2. Stores national constraint data for later use in custom constraints
    3. Lets the optimizer determine optimal regional distribution within national limits
    """
    
    # Check if multi-region scenario
    regions_setting = scenario_setup.get("regions", "1")
    is_multi_region = str(regions_setting) in ["10", "34", "159"]
    
    if not is_multi_region:
        # Single region - use original approach
        return define_extendable_tech_original(carriers, years, type_, ext_param)
    
    logging.info(f"Creating regional extendable technologies with national constraints for {regions_setting} regions")
    
    # Read national constraint data
    national_constraints = read_national_constraint_data(scenario_setup, years, type_)
    
    if national_constraints.empty:
        logging.warning(f"No national constraint data found for {type_}")
        return []
    
    # Store national constraints for later use in custom constraints
    store_national_constraints(national_constraints, type_, scenario_setup)
    
    # Create regional technologies with high individual limits
    regional_tech_list = create_regional_technologies_with_equal_limits(
        carriers, years, type_, national_constraints
    )
    
    logging.info(f"Created {len(regional_tech_list)} regional {type_} technologies with national constraints")
    return regional_tech_list


def read_national_constraint_data(scenario_setup, years, type_):
    """
    Read all national constraint data (max_total, min_total, max_annual, min_annual).
    """
    
    excel_file = os.path.join(scenario_setup["sub_path"], "extendable_technologies.xlsx")
    constraint_data = {}
    
    # Different constraint types to read
    constraint_sheets = {
        'max_total_installed': 'max_total',
        'min_total_installed': 'min_total', 
        'max_annual_installed': 'max_annual',
        'min_annual_installed': 'min_annual'
    }
    
    for sheet_name, constraint_type in constraint_sheets.items():
        try:
            # Try different national identifiers
            for national_id in ["1", "RSA", "SA", "national"]:
                try:
                    data = pd.read_excel(
                        excel_file,
                        sheet_name=sheet_name,
                        index_col=[0,1,3,2,4],
                    ).loc[(scenario_setup[f"extendable_{constraint_type}"], national_id, type_, slice(None)), years]
                    
                    # Clean up data
                    data.replace("unc", np.inf, inplace=True)
                    data.index = data.index.droplevel([0, 1, 2])
                    data = data.loc[~(data==0).all(axis=1)]
                    
                    constraint_data[constraint_type] = data
                    logging.info(f"Loaded {constraint_type} constraints for {type_}")
                    break
                    
                except KeyError:
                    continue
            else:
                logging.warning(f"Could not find {constraint_type} data for {type_}")
                
        except Exception as e:
            logging.warning(f"Error reading {sheet_name}: {e}")
    
    # Combine all constraint data
    if constraint_data:
        combined_data = pd.concat(constraint_data, names=['constraint_type', 'carrier'])
        return combined_data
    else:
        return pd.DataFrame()


def create_regional_technologies_with_equal_limits(carriers, years, type_, national_constraints):
    """
    Create regional extendable technologies with high individual limits.
    The national constraints will be applied separately.
    """
    
    # Configuration
    use_regional_codes = snakemake.config.get("electricity", {}).get("use_regional_codes", False)
    
    # Get eligible carriers
    if type_ == "Generator":
        eligible_carriers = carriers['extendable']['conventional'] + carriers['extendable']['renewables']
    elif type_ == "StorageUnit":
        eligible_carriers = carriers['extendable']['storage']
    else:
        return []
    
    # Get regions from network
    if 'n' in globals():
        regions = [bus for bus in n.buses.index if bus in REGIONAL_MAPPING]
    else:
        regions = list(REGIONAL_MAPPING.keys())
    
    # Find which carriers have national constraints
    carriers_with_constraints = set()
    if 'max_total' in national_constraints.index.get_level_values(0):
        carriers_with_constraints.update(
            national_constraints.xs('max_total', level=0).index
        )
    
    # Filter for carriers that are both eligible and have constraints
    constrained_carriers = [c for c in eligible_carriers if c in carriers_with_constraints]
    
    if not constrained_carriers:
        logging.warning(f"No {type_} carriers found with national constraints")
        return []
    
    # Create regional technology list
    regional_tech_list = []
    
    for carrier in constrained_carriers:
        for year in years:
            for region in regions:
                # Create regional identifier
                region_info = REGIONAL_MAPPING[region]
                suffix = region_info['code'] if use_regional_codes else str(region_info['index'])
                
                regional_tech_id = f"{region}-{carrier}_{suffix}-{year}"
                regional_tech_list.append(regional_tech_id)
    
    return regional_tech_list


def store_national_constraints(national_constraints, type_, scenario_setup):
    """
    Store national constraints in a format that can be used by custom constraint functions.
    """
    
    # Store in scenario_setup or global variable for later access
    constraint_key = f"national_constraints_{type_}"
    
    if hasattr(scenario_setup, '_national_constraints'):
        scenario_setup._national_constraints[constraint_key] = national_constraints
    else:
        scenario_setup._national_constraints = {constraint_key: national_constraints}
    
    # Also store globally for access in constraint functions
    globals()[constraint_key] = national_constraints


def set_extendable_limits_with_high_individual_limits(n, scenario_setup):
    """
    Set high individual limits for regional technologies, 
    since national constraints will be applied separately.
    """
    
    # Set very high individual limits for all extendable components
    high_limit = 1e6  # 1000 GW - effectively unlimited
    
    # For generators
    for gen in n.generators.query("p_nom_extendable").index:
        n.generators.loc[gen, "p_nom_max"] = high_limit
    
    # For storage units
    for su in n.storage_units.query("p_nom_extendable").index:
        n.storage_units.loc[su, "p_nom_max"] = high_limit


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


def group_regional_technologies_by_base_carrier(components):
    """
    Group regional technology variants by their base carrier name.
    
    E.g., solar_pv_0, solar_pv_1, ..., solar_pv_9 all belong to base carrier 'solar_pv'
    """
    
    carrier_groups = {}
    
    for idx, row in components.iterrows():
        carrier = row['carrier']
        
        # Extract base carrier name (remove regional suffixes)
        base_carrier = extract_base_carrier_name(carrier)
        
        if base_carrier not in carrier_groups:
            carrier_groups[base_carrier] = []
        
        carrier_groups[base_carrier].append(idx)
    
    return carrier_groups


def extract_base_carrier_name(regional_carrier):
    """
    Extract base carrier name from regional carrier.
    
    E.g., 'solar_pv_0' -> 'solar_pv'
         'wind_EC' -> 'wind'
         'battery_4h_5' -> 'battery_4h'
    """
    
    # Remove numerical suffixes (0-9)
    if regional_carrier.endswith(tuple(f'_{i}' for i in range(10))):
        return regional_carrier[:-2]
    
    # Remove regional code suffixes (EC, FS, GP, etc.)
    regional_codes = ['EC', 'FS', 'GP', 'HY', 'ZN', 'LP', 'MP', 'NW', 'NC', 'WC']
    for code in regional_codes:
        if regional_carrier.endswith(f'_{code}'):
            return regional_carrier[:-3]
    
    # If no suffix found, return as is
    return regional_carrier


def add_carrier_national_constraint(n, p_nom_var, component_list, constraint_data, 
                                  constraint_type, carrier, component_type):
    """
    Add a national constraint for a specific carrier.
    """
    
    try:
        # Select the relevant components
        if component_type == "Generator":
            components_var = p_nom_var.sel(Generator=component_list)
            sum_var = components_var.sum("Generator")
        elif component_type == "StorageUnit":
            components_var = p_nom_var.sel(StorageUnit=component_list)
            sum_var = components_var.sum("StorageUnit")
        
        # Add constraints for each year
        if n.multi_invest:
            for year in n.investment_periods:
                if year in constraint_data.index:
                    limit = constraint_data[year]
                    
                    if constraint_type == "max_total":
                        n.model.add_constraints(
                            sum_var, "<=", limit, 
                            name=f"national_max_total_{carrier}_{year}_{component_type}"
                        )
                    elif constraint_type == "min_total":
                        n.model.add_constraints(
                            sum_var, ">=", limit,
                            name=f"national_min_total_{carrier}_{year}_{component_type}"
                        )
                    # Annual constraints would need different handling (sum of new capacity in year)
                    
        else:
            # Single investment period
            year = sns[0].year if hasattr(sns[0], 'year') else list(constraint_data.index)[0]
            if year in constraint_data.index:
                limit = constraint_data[year]
                
                if constraint_type == "max_total":
                    n.model.add_constraints(
                        sum_var, "<=", limit,
                        name=f"national_max_total_{carrier}_{component_type}"
                    )
                elif constraint_type == "min_total":
                    n.model.add_constraints(
                        sum_var, ">=", limit,
                        name=f"national_min_total_{carrier}_{component_type}"
                    )
        
        logging.info(f"Added national {constraint_type} constraint for {carrier} {component_type}: {len(component_list)} regional components")
        
    except Exception as e:
        logging.warning(f"Error adding national constraint for {carrier}: {e}")


# Integration functions for existing codebase

def enhanced_set_extendable_limits_global(n, scenario_setup):
    """
    Enhanced version that handles both original global limits and new national constraints.
    """
    
    # First, apply original global limits (if any)
    try:
        set_extendable_limits_global_original(n)
    except:
        pass  # Original function might not exist or might fail
    
    # Then set high individual limits for regional technologies
    set_extendable_limits_with_high_individual_limits(n, scenario_setup)


def define_extendable_tech_original(carriers, years, type_, ext_param):
    """
    Original single-region function (unchanged).
    """
    
    try:
        ext_max_build = pd.read_excel(
            os.path.join(scenario_setup["sub_path"],"extendable_technologies.xlsx"), 
            sheet_name='max_total_installed',
            index_col=[0,1,3,2,4],
        ).loc[(scenario_setup["extendable_max_total"], scenario_setup["regions"], type_, slice(None)), years]
        
        ext_max_build.replace("unc", np.inf, inplace=True)
        ext_max_build.index = ext_max_build.index.droplevel([0, 1, 2])
        ext_max_build = ext_max_build.loc[~(ext_max_build==0).all(axis=1)]

        if type_ == "Generator":
            eligible_carriers = carriers['extendable']['conventional'] + carriers['extendable']['renewables']
        elif type_ == "StorageUnit":
            eligible_carriers = carriers['extendable']['storage']

        idx = [(bus, c) for (bus, c) in ext_max_build.index if c in eligible_carriers]
        ext_max_build = ext_max_build.loc[idx]

        return (
            ext_max_build[ext_max_build != 0].stack().index.to_series().apply(lambda x: "-".join([x[0], x[1], str(x[2])]))
        ).values
        
    except Exception as e:
        logging.error(f"Error in original define_extendable_tech: {e}")
        return []


# Configuration example
CONFIG_EXAMPLE = """
# Add to config.yaml:
electricity:
  use_regional_codes: false  # Use _0, _1, _2... instead of _EC, _FS, _GP...
  
# The national constraints will be read from the extendable_technologies.xlsx file
# and applied automatically during optimization
"""

# Usage instructions for integration
INTEGRATION_INSTRUCTIONS = """
INTEGRATION STEPS:

1. Replace define_extendable_tech() in add_electricity.py with:
   define_extendable_tech_with_national_constraints()

2. In prepare_and_solve_network.py, add to the extra_functionality function:
   
   def extra_functionality(n, snapshots):
       # ... existing constraints ...
       
       # Add national capacity constraints
       add_national_capacity_constraints(n, snapshots, scenario_setup)

3. Make sure your extendable_technologies.xlsx has data for region "1" 
   (representing national totals)

4. The system will automatically:
   - Create regional variants of each technology (carrier_0, carrier_1, etc.)
   - Set high individual limits for each regional variant
   - Apply national total constraints across all regional variants
   - Let the optimizer decide optimal regional distribution
"""
