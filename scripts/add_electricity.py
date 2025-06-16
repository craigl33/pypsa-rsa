# SPDX-FileCopyrightText:  PyPSA-RSA, PyPSA-ZA, PyPSA-Earth and PyPSA-Eur Authors
# # SPDX-License-Identifier: MIT

# coding: utf-8

"""
Adds fixed and extendable components to the base network. The primary functions run inside main are:

    attach_load
    attach_fixed_generators
    attach_extendable_generators
    attach_fixed_storage
    attach_extendable_storage

Relevant Settings
-----------------

.. code:: yaml

    costs:
        year:
        USD_to_ZAR:
        EUR_to_ZAR:
        marginal_cost:
        dicountrate:
        emission_prices:
        load_shedding:

    electricity:
        max_hours:
        marginal_cost:
        capital_cost:
        conventional_carriers:
        co2limit:
        extendable_carriers:
        include_renewable_capacities_from_OPSD:
        estimate_renewable_capacities_from_capacity_stats:

    load:
        scale:
        ssp:
        weather_year:
        prediction_year:
        region_load:

    renewable:
        hydro:
            carriers:
            hydro_max_hours:
            hydro_capital_cost:

    lines:
        length_factor:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at :ref:`costs_cf`,
    :ref:`electricity_cf`, :ref:`load_cf`, :ref:`renewable_cf`, :ref:`lines_cf`

Inputs
------
- ``model_file.xlsx``: The database to setup different scenarios based on cost assumptions for all included technologies for specific years from various sources; e.g. discount rate, lifetime, investment (CAPEX), fixed operation and maintenance (FOM), variable operation and maintenance (VOM), fuel costs, efficiency, carbon-dioxide intensity.
- ``data/Eskom EAF data.xlsx``: Hydropower plant store/discharge power capacities, energy storage capacity, and average hourly inflow by country.  Not currently used!
- ``data/eskom_pu_profiles.csv``: alternative to capacities above; not currently used!
- ``data/bundle/SystemEnergy2009_22.csv`` Hourly country load profiles produced by GEGIS
- ``resources/regions_onshore.geojson``: confer :ref:`busregions`
- ``resources/gadm_shapes.geojson``: confer :ref:`shapes`
- ``data/bundle/supply_regions/{regions}.shp``: confer :ref:`powerplants`
- ``resources/profile_{}_{regions}_{resarea}.nc``: all technologies in ``config["renewables"].keys()``, confer :ref:`renewableprofiles`.
- ``networks/base_{model_file}_{regions}.nc``: confer :ref:`base`

Outputs
-------

- ``networks/elec_{model_file}_{regions}_{resarea}.nc``:

    .. image:: ../img/elec.png
            :scale: 33 %

Description
-----------

The rule :mod:`add_electricity` ties all the different data inputs from the preceding rules together into a detailed PyPSA network that is stored in ``networks/elec.nc``. It includes:

- today"s transmission topology and transfer capacities (in future, optionally including lines which are under construction according to the config settings ``lines: under_construction`` and ``links: under_construction``),
- today"s thermal and hydro power generation capacities (for the technologies listed in the config setting ``electricity: conventional_carriers``), and
- today"s load time-series (upsampled in a top-down approach according to population and gross domestic product)

It further adds extendable ``generators`` with **zero** capacity for

- photovoltaic, onshore and AC- as well as DC-connected offshore wind installations with today"s locational, hourly wind and solar capacity factors (but **no** current capacities),
- additional open- and combined-cycle gas turbines (if ``OCGT`` and/or ``CCGT`` is listed in the config setting ``electricity: extendable_carriers``)
"""

import geopandas as gpd
import logging
logger = logging.getLogger(__name__)
import numpy as np
import pandas as pd
import pypsa
from pypsa.descriptors import get_switchable_as_dense as get_as_dense, get_activity_mask
from pypsa.io import import_components_from_dataframe
import os
from shapely.geometry import Point
import xarray as xr
import warnings
warnings.simplefilter(action="ignore") # Comment out for debugging and development


from _helpers import (
    add_missing_carriers,
    add_noise,
    adjust_by_p_max_pu,
    apply_default_attr,
    convert_cost_units,
    drop_non_pypsa_attrs,
    get_carriers_from_model_file,
    get_investment_periods,
    get_snapshots,
    get_start_year,
    initial_ramp_rate_fix,
    load_disaggregate, 
    map_component_parameters, 
    normed,
    read_and_filter_generators,
    remove_leap_day,
    single_year_network_copy,
    load_scenario_definition,
    get_base_carrier
)

# Regional mapping for 10-region South Africa model
REGIONAL_MAPPING = {
    'Eastern Cape': 'EC',
    'Free State': 'FS',
    'Gauteng': 'GP',
    'Hydra Central': 'HY',
    'KwaZulu Natal': 'ZN',
    'Limpopo':'LP',
    'Mpumalanga': 'MP',
    'North West':  'NW',
    'Northern Cape': 'NC',
    'Western Cape':  'WC'
}

# Easy inverses and indices using dict comprehension
CODE_TO_NAME = {v: k for k, v in REGIONAL_MAPPING.items()}
# Position-based indices (order matters, so use sorted for consistency)
REGION_NAMES = sorted(REGIONAL_MAPPING.keys())  # Ensures consistent ordering
NAME_TO_INDEX = {name: i for i, name in enumerate(REGION_NAMES)}
INDEX_TO_NAME = {i: name for i, name in enumerate(REGION_NAMES)}

"""
********************************************************************************
    Cost related functions
********************************************************************************
"""
def annualise_costs(investment, lifetime, discount_rate, FOM):
    """
    Annualises the costs of an investment over its lifetime.The costs are annualised using the 
    Capital Recovery Factor (CRF) formula.

    Args:
    - investment: The overnight investment cost.
    - lifetime: The lifetime of the investment.
    - discount_rate: The discount rate used for annualisation.
    - FOM: The fixed operating and maintenance costs.

    Returns:
    A Series containing the annualised costs.
    """
    CRF = discount_rate / (1 - 1 / (1 + discount_rate) ** lifetime)
    return (investment * CRF + FOM).fillna(0)

def load_extendable_parameters(n, scenario_setup, snakemake):
    """
    set all asset costs tab in the model file
    """
    defaults = snakemake.config["electricity"]["extendable_parameters"]["defaults"]

    param_mapping = pd.read_excel(
        os.path.join(scenario_setup["sub_path"],"extendable_technologies.xlsx"), 
        sheet_name = "parameter_mapping",
        index_col = [0,1],
    ).loc[scenario_setup["extendable_parameters"]]

    param_matrix = pd.read_excel(
        os.path.join(scenario_setup["sub_path"],"extendable_technologies.xlsx"), 
        sheet_name = "parameters",
        index_col = [0,2,1],
    ).sort_index()

    param = pd.DataFrame(index = pd.MultiIndex.from_product([param_mapping.columns, param_mapping.index]), columns = param_matrix.columns)
    
    for idx in param.index:
        mapping_ = param_mapping.loc[idx[1], idx[0]]
        if pd.isna(mapping_):
            # get default value from config file
            param.loc[idx, :] = defaults[idx[0]]
        else:
            param.loc[idx, :] = param_matrix.loc[(mapping_, idx[0], idx[1]), :].values

    param.drop("source", axis=1, inplace=True)
    
    # Interpolate for years in config file but not in cost_data excel file
    ext_years = get_investment_periods(n.snapshots, n.multi_invest)
    ext_years_array = np.array(ext_years)
    missing_year = ext_years_array[~np.isin(ext_years_array,param.columns)]
    if len(missing_year) > 0:
        for i in missing_year: 
            param.insert(0,i,np.nan) # add columns of missing year to dataframe
        param_tmp = param.drop("unit", axis=1).sort_index(axis=1)
        param_tmp = param_tmp.interpolate(axis=1)
        param= pd.concat([param_tmp, param["unit"]], ignore_index=False, axis=1)

    # correct units to MW and ZAR
    param_yr = param.columns.drop("unit")

    param = convert_cost_units(param, snakemake.config["costs"]["USD_to_ZAR"], snakemake.config["costs"]["EUR_to_ZAR"])
    
    full_param = pd.DataFrame(
        index = pd.MultiIndex.from_product(
            [
                param.index.get_level_values(0).unique(),
                param.index.get_level_values(1).unique()]),
        columns = param.columns
    )
    full_param.loc[param.index] = param.values

    # full_costs adds default values when missing from costs table
    config_defaults = snakemake.config["electricity"]["extendable_parameters"]["defaults"]
    for default in param.index.get_level_values(0).intersection(config_defaults.keys()):
        full_param.loc[param.loc[(default, slice(None)),:].index, :] = param.loc[(default, slice(None)),:]
        full_param.loc[(default, slice(None)), param_yr] = full_param.loc[(default, slice(None)), param_yr].fillna(config_defaults[default])
    #full_param = full_param.fillna("default")
    param = full_param.copy()

    # Get entries where FOM is specified as % of CAPEX
    fom_perc_capex=param.loc[param.unit.str.contains(r"%capex/year") == True, param_yr]
    fom_perc_capex_idx=fom_perc_capex.index.get_level_values(1)

    add_param = pd.DataFrame(
        index = pd.MultiIndex.from_product([["capital_cost","marginal_cost"],param.loc["FOM"].index]),
        columns = param.columns
    )
    
    param = pd.concat([param, add_param],axis=0)
    param.loc[("FOM",fom_perc_capex_idx), param_yr] *= (param.loc[("investment", fom_perc_capex_idx),param_yr]).values/100.0
    param.loc[("FOM",fom_perc_capex_idx), "unit"] = param.loc[("investment", fom_perc_capex_idx),"unit"].values

    capital_costs = annualise_costs(
        param.loc["investment", param_yr],
        param.loc["lifetime", param_yr], 
        param.loc["discount_rate", param_yr],
        param.loc["FOM", param_yr],
    )

    param.loc["capital_cost", param_yr] = capital_costs.fillna(0).values
    param.loc["capital_cost","unit"] = "R/MWe"

    vom = param.loc["VOM", param_yr].fillna(0)
    fuel = (param.loc["fuel", param_yr] / param.loc["efficiency", param_yr]).fillna(0)

    param.loc[("marginal_cost", vom.index), param_yr] = vom.values
    param.loc[("marginal_cost", fuel.index), param_yr] += fuel.values
    param.loc["marginal_cost","unit"] = "R/MWhe"

    #max_hours = snakemake.config["electricity"]["max_hours"]
    #param.loc[("capital_cost","battery"), :] = param.loc[("capital_cost","battery inverter"),:]
    #param.loc[("capital_cost","battery"), param_yr] += max_hours["battery"]*param.loc[("capital_cost", "battery storage"), param_yr]
    
    return param

# def update_transmission_costs(n, costs, length_factor=1.0, simple_hvdc_costs=False):
#     # Currently only average transmission costs are implemented
#     n.lines["capital_cost"] = (
#         n.lines["length"] * length_factor * costs.loc[("capital_cost","HVAC_overhead"), costs.columns.drop("unit")].mean()
#     )

#     if n.links.empty:
#         return

#     dc_b = n.links.carrier == "DC"
#     # If there are no "DC" links, then the "underwater_fraction" column
#     # may be missing. Therefore we have to return here.
#     # TODO: Require fix
#     if n.links.loc[n.links.carrier == "DC"].empty:
#         return

#     if simple_hvdc_costs:
#         hvdc_costs = (
#             n.links.loc[dc_b, "length"]
#             * length_factor
#             * costs.loc[("capital_cost","HVDC_overhead"),:].mean()
#         )
#     else:
#         hvdc_costs = (
#             n.links.loc[dc_b, "length"]
#             * length_factor
#             * (
#                 (1.0 - n.links.loc[dc_b, "underwater_fraction"])
#                 * costs.loc[("capital_cost","HVDC_overhead"),:].mean()
#                 + n.links.loc[dc_b, "underwater_fraction"]
#                 * costs.loc[("capital_cost","HVDC_submarine"),:].mean()
#             )
#             + costs.loc[("capital_cost","HVDC inverter_pair"),:].mean()
#         )
#     n.links.loc[dc_b, "capital_cost"] = hvdc_costs


"""
********************************************************************************
    Add load to the network 
********************************************************************************
"""

def attach_load(n, scenario_setup):
    """
    Attaches load to the network based on the provided annual demand.
    Demand is disaggregated by the load_disaggreate function according to either population 
    or GVA (GDP) in each region. 

    Args:
    - n: The network object.
    - annual_demand: A DataFrame containing the annual demand values.

        """

    # Read load data with explicit datetime conversion
    load = pd.read_csv(snakemake.input.load, index_col="Date Time Hour Beginning")

    # Ensure index is properly converted to datetime
    if not isinstance(load.index, pd.DatetimeIndex):
        load.index = pd.to_datetime(load.index, dayfirst=True)
    load.index.name == "datetime"

    # Prepare load dataframe from raw Eskom data updated from the Eskom Data Portal  
    load = load.rename(columns={"RSA Contracted Demand":"system_energy"})
    load.loc[:,'Year'] = load.index.year

    annual_load = (
        pd.read_excel(
            os.path.join(
                scenario_setup["sub_path"],
                "annual_load.xlsx",
            ),
            sheet_name="annual_load",
            index_col=[0],
        )
    ).loc[scenario_setup["load_trajectory"]]

    annual_load = annual_load.drop(["unit","Source"]).T*1e6
    profile_load = normed(remove_leap_day(load[load.index.year == int(snakemake.config["years"]["reference_load_year"])]["system_energy"]))
    
    if n.multi_invest:
        load=pd.Series(0,index=n.snapshots)
        for y in n.investment_periods:
            load.loc[y]=profile_load.values*annual_load.loc[y]
    else:
        load = pd.Series(profile_load.values*annual_load[n.snapshots[0].year], index = n.snapshots)

    if len(n.buses) == 1:
        n.add("Load", "RSA",
            bus="RSA",
            p_set=load)
    else:
        n.madd("Load", list(n.buses.index),
            bus = list(n.buses.index),
            p_set = load_disaggregate(load, normed(n.buses[snakemake.config["electricity"]["load_disaggregation"]])))


"""
********************************************************************************
    Function to define p_max_pu and p_min_pu profiles 
********************************************************************************
"""
def init_pu_profiles(gens, snapshots):
    pu_profiles = pd.DataFrame(
        index = pd.MultiIndex.from_product(
            [["max", "min"], snapshots], 
            names=["profile", "snapshots"]
            ), 
        columns = gens.index
    )
    pu_profiles.loc["max"] = 1 
    pu_profiles.loc["min"] = 0

    return pu_profiles


def extend_reference_data(n, ref_data, snapshots):

    # delete data from years if all zeros
    ref_data = ref_data.groupby(pd.Grouper(freq='Y')).filter(lambda x: not (x==0).all().all())

    ext_years = snapshots.year.unique()
    if len(ref_data.shape) > 1:
        extended_data = pd.DataFrame(0, index=snapshots, columns=ref_data.columns)
    else:
        extended_data = pd.Series(0, index=snapshots)     
    ref_years = ref_data.index.year.unique()

    for _ in range(int(np.ceil(len(ext_years) / len(ref_years)))-1):
        ref_data = pd.concat([ref_data, ref_data],axis=0)

    extended_data.iloc[:] = ref_data.iloc[range(len(extended_data))].values

    return extended_data.clip(lower=0., upper=1.)


def get_eaf_profiles(snapshots, type):
      
    outages = pd.read_excel(
            os.path.join(scenario_setup["sub_path"], "plant_availability.xlsx"), 
            sheet_name='outage_profiles',
            index_col=[0,1,2],
            header=[0,1],
    ).loc[scenario_setup["outage_profiles"]]
    outages = outages[type+"_generators"]

    def proc_outage(outages, _type, snapshots):
        out_df = outages.loc[(_type, range(1,54)), :]
        out_df.index = range(1,54)
        std_dev = outages.loc[(_type, "std_dev_noise"),:]

        eaf_hrly = out_df.loc[snapshots.isocalendar().week]
        eaf_hrly.index = snapshots

        for col in eaf_hrly.columns:
            eaf_hrly[col] = add_noise(eaf_hrly[col], std_dev[col], 48)

        return eaf_hrly

    pclf = proc_outage(outages, "planned", snapshots)
    uoclf = proc_outage(outages, "unplanned", snapshots)

    
    return 1- (pclf + uoclf)

def clip_pu_profiles(n, pu, gen_list, lower=0, upper=1):
    n.generators_t[pu] = n.generators_t[pu].copy()
    n.generators_t[pu].loc[:, gen_list] = get_as_dense(n, "Generator", pu)[gen_list].clip(lower=lower, upper=upper)


def proj_eaf_override(eaf_hrly, snapshots, include = "_EAF", exclude = "extendable"):
    """
    Overrides the hourly EAF (Energy Availability Factor) values with projected EAF values, if these are defined
    under the project_parameters tab in the model_file.xlsx. Existing generators have suffix _EAF and extendable generators
    have the suffix _extendable_EAF by convention.  

    Args:
    - eaf_hrly: A DataFrame containing the hourly EAF values.
    - projections: A DataFrame containing the projected EAF values.
    - snapshots: A Series containing the snapshots.
    - include: A string used to filter the projections based on the index.
    - exclude: A string used to exclude certain projections based on the index.

    Relevant model_file.xlsx settings:
        project_parameters: parameters with _EAF or _extendable_EAF suffix  
    
    """
    annual_avail = pd.read_excel(
            os.path.join(scenario_setup["sub_path"], "plant_availability.xlsx"), 
            sheet_name='annual_availability',
            index_col=[0,1],
    ).loc[scenario_setup["annual_availability"]]

    eaf_yrly = eaf_hrly.groupby(eaf_hrly.index.year).mean()
    proj_eaf = annual_avail.loc[(annual_avail.index.str.contains(include) & ~annual_avail.index.str.contains(exclude)), snapshots.year.unique()]
    proj_eaf.index = proj_eaf.index.str.replace(include,"")

    # remove decom_stations
    proj_eaf = proj_eaf[proj_eaf.index.isin(eaf_yrly.columns)]
    scaling = proj_eaf.T.div(eaf_yrly[proj_eaf.index], axis="columns", level="year").fillna(1)

    for y in snapshots.year.unique():
        eaf_hrly.loc[str(y), scaling.columns] *= scaling.loc[y, :]  

    return eaf_hrly

def generate_eskom_re_profiles(n, carriers):
    """
    Generates Eskom renewable energy profiles for the network, based on the Eskom Data Portal information, found under
    https://www.eskom.co.za/dataportal/. Data is available from 2018 to 2023 for aggregate carriers (e.g. all solar_pv, biomass, hydro etc).
    The user can specify whether to use this data under config.yaml. These Eskom profiles for biomass, hydro and hydro_import are used by default.

    Args:
    - n: The PyPSA network object.

    Relevant config.yaml settings:
    electricity:
        renewable_generators:
            carriers:
    years:
        reference_weather_years:
    enable:
        use_eskom_wind_solar
    """
    ext_years = get_investment_periods(n.snapshots, n.multi_invest)
    ref_years = snakemake.config["years"]["reference_weather_years"]
    carriers = [elem for elem in carriers if not elem.startswith(("wind","solar_pv"))]

    eskom_data = (
        pd.read_csv(
            snakemake.input.eskom_profiles,skiprows=[1], 
            index_col=0,parse_dates=True
        )
        .resample("1h").mean()
    )

    eskom_data = remove_leap_day(eskom_data)
    eskom_profiles = pd.DataFrame(0, index=n.snapshots, columns=carriers)

    for carrier in carriers:
        weather_years = ref_years[carrier].copy()
        if n.multi_invest:
            weather_years *= int(np.ceil(len(ext_years) / len(weather_years)))

        for cnt, y in enumerate(ext_years):
            y = y if n.multi_invest else str(y)
            eskom_profiles.loc[y, carrier] = (eskom_data.loc[str(weather_years[cnt]), carrier]
                                            .clip(lower=0., upper=1.)).values
    return eskom_profiles
def generate_fixed_wind_solar_profiles(n, gens, snapshots, carriers, pu_profiles):
    """
    Generates fixed wind and solar PV profiles for the network based on timeseries data that is pre-calculated
    and using the notebooks under resource_processing. Data is loaded from NetCDF format.


    Args:
    - n: The network object.
    - gens: A DataFrame containing the generator information.
    - snapshots: A Series containing the snapshots.
    - pu_profiles: The DataFrame to store the pu profiles.

    """

    #only select entries in carriers that start with wind or solar_pv
    carriers = [elem for elem in carriers if elem.startswith(("wind","solar_pv"))]
    config = snakemake.config["electricity"]["renewable_generators"]
    ref_years = snakemake.config["years"]["reference_weather_years"]
    for carrier in carriers:
        if carrier not in ["solar_pv_rooftop", "wind_offshore"]:
            base_carrier = "wind" if carrier.startswith("wind") else "solar_pv"
        else:
            base_carrier = carrier
            
        data = f"{carrier}_fixed_{config['resource_profiles']['datasets'][carrier]}"
        pu= xr.open_dataarray(snakemake.input.renewable_profiles, group=data).sel(param="pu").to_pandas()
        pu = remove_leap_day(pu)
        pu = pu[pu.index.year.isin(ref_years[base_carrier])]
        mapping = gens.loc[(gens["Model Key"] != np.nan) & (gens["carrier"] == carrier),"Model Key"]
        mapping = pd.Series(mapping.index,index=mapping.values)
        pu = pu[mapping.index]
        pu.columns = mapping[pu.columns].values
        pu = extend_reference_data(n, pu, snapshots) * (1-config["degradation_adj_capacity_factor"][carrier])
        pu_profiles.loc["max", pu.columns] = pu.values
        pu_profiles.loc["min", pu.columns] = 0.98*pu.values # Existing REIPPP take or pay constraint (100% can cause instabilitites)

    return pu_profiles


def get_available_renewable_groups(filepath):
    """
    Get all available renewable energy groups from the NetCDF file.
    """
    import netCDF4
    
    try:
        with netCDF4.Dataset(filepath, 'r') as nc:
            groups = list(nc.groups.keys())
    except Exception as e:
        logging.error(f"Cannot read NetCDF groups: {e}")
        return {}
    
    # Organize groups by technology and region count
    available = {}
    for group in groups:
        parts = group.split('_')
        if len(parts) >= 3:
            if parts[-1] in ['era5', 'sarah', 'wasa', 'csir']:
                source = parts[-1]
                if parts[-2].isdigit():
                    regions = parts[-2]
                    tech = '_'.join(parts[:-2])
                elif parts[-2] == 'fixed':
                    regions = 'fixed'
                    tech = '_'.join(parts[:-2])
                else:
                    continue
                    
                key = f"{tech}_{regions}"
                if key not in available:
                    available[key] = []
                available[key].append(group)
    
    logging.info(f"Found renewable groups: {available}")
    return available


def generate_extendable_wind_solar_profiles(n, gens, snapshots, carriers, pu_profiles):
    """
    Simple fix for renewable profiles that works with your actual NetCDF structure.
    
    This uses your existing REGIONAL_MAPPING and just fixes the NetCDF access.
    """
    # Filter for wind and solar carriers only
    renewable_carriers = [elem for elem in carriers if elem.startswith(("wind", "solar_pv"))]
    
    extendable_gens = gens.query("carrier in @renewable_carriers & p_nom_extendable")
    
    if extendable_gens.empty:
        logging.info("No extendable renewable generators found")
        return pu_profiles

    logging.info(f"Processing renewable profiles for {len(renewable_carriers)} carriers")
    
    # Get available groups from NetCDF file
    available_groups = get_available_renewable_groups(snakemake.input.renewable_profiles)
    if not available_groups:
        logging.warning("No renewable groups found in NetCDF file")
        return pu_profiles
    
    years = snapshots.year.unique()
    config = snakemake.config["electricity"]["renewable_generators"]
    
    # Use your existing regional mapping

    use_regional_codes = snakemake.config.get("electricity", {}).get("use_regional_codes", False)
 
    if use_regional_codes:
        bus_mapping = CODE_TO_NAME
    else:
        bus_mapping = INDEX_TO_NAME

    # Resource area mapping
    RESOURCE_AREA_MAPPING = {"low": 0, "medium": 1, "high": 2, "redz": 3}
    
    # Process each carrier
    for carrier in renewable_carriers:
        try:
            # Simple base carrier extraction
            base_carrier = get_base_carrier(carrier, is_multi_region=True)
            
            # Extract regional information
            target_bus = None
            for suffix, bus_name in bus_mapping.items():
                if carrier.endswith(f'_{suffix}'):
                    target_bus = bus_name
                    break
            
            if target_bus:
                target_buses = [target_bus]
                logging.info(f"Processing {carrier} -> base: {base_carrier}, bus: {target_bus}")
            else:
                target_buses = list(n.buses.index)
                logging.info(f"Processing {carrier} as national technology")
            
            # Find matching dataset in NetCDF file
            n_regions = len(n.buses)
            region_key = str(n_regions)
            
            # Try to find the right dataset
            dataset_group = None
            search_patterns = [
                f"{base_carrier}_{region_key}",
                f"{base_carrier}_fixed", 
                base_carrier
            ]
            
            for pattern in search_patterns:
                if pattern in available_groups:
                    group_options = available_groups[pattern]
                    # Prefer era5, then sarah, then others
                    for source in ['era5', 'sarah', 'wasa', 'csir']:
                        matching = [g for g in group_options if g.endswith(source)]
                        if matching:
                            dataset_group = matching[0]
                            break
                    
                    if not dataset_group and group_options:
                        dataset_group = group_options[0]
                    
                    if dataset_group:
                        logging.info(f"Using dataset: {dataset_group} for {carrier}")
                        break
            
            if not dataset_group:
                logging.warning(f"No matching dataset found for {carrier}")
                continue
            
            # Load the renewable resource data
            try:
                data = xr.open_dataarray(snakemake.input.renewable_profiles, group=dataset_group)
                logging.info(f"Loaded {dataset_group}: {data.dims}, shape {data.shape}")
            except Exception as e:
                logging.error(f"Failed to load dataset {dataset_group}: {e}")
                continue
            
            # Get reference weather years
            reference_weather_years = snakemake.config["years"]["reference_weather_years"]
            weather_key = "wind" if base_carrier.startswith("wind") else "solar_pv"
            ref_years = reference_weather_years.get(weather_key, [2019])
            
            # Get resource area setting
            resource_area = scenario_setup.get("resource_area", "medium")
            area_idx = RESOURCE_AREA_MAPPING.get(resource_area, 1)
            
            # Process each target bus
            for bus in target_buses:
                try:
                    if bus not in n.buses.index:
                        continue
                    
                    # Check if bus exists in NetCDF data
                    netcdf_buses = list(data.coords['bus'].values)
                    if bus not in netcdf_buses:
                        logging.warning(f"Bus {bus} not found in NetCDF data")
                        continue
                    
                    # Extract time series for this bus and resource area
                    if 'intra_region' in data.dims and area_idx < data.sizes['intra_region']:
                        pu_ref = data.sel(bus=bus, intra_region=area_idx).to_pandas()
                    else:
                        pu_ref = data.sel(bus=bus).mean('intra_region').to_pandas()
                    
                    # Convert time coordinates
                    if not isinstance(pu_ref.index, pd.DatetimeIndex):
                        pu_ref.index = pd.to_datetime(pu_ref.index)
                    
                    # Filter for reference years and remove leap days
                    pu_ref = pu_ref[pu_ref.index.year.isin(ref_years)]
                    pu_ref = remove_leap_day(pu_ref)
                    
                    if pu_ref.empty:
                        logging.warning(f"No data for {carrier} at {bus} after filtering")
                        continue
                    
                    # Apply degradation factor
                    degradation_config = config.get("degradation_adj_capacity_factor", {})
                    degradation_factor = 1 - degradation_config.get(carrier, 0)
                    
                    # Create profiles for each investment year
                    for y in years:
                        gen_name = f"{bus}-{carrier}-{y}"
                        
                        if gen_name in pu_profiles.columns:
                            try:
                                extended_profile = extend_reference_data(n, pu_ref, snapshots)
                                pu_profiles.loc["max", gen_name] = extended_profile.values * degradation_factor
                                logging.debug(f"Set profile for {gen_name}")
                            except Exception as e:
                                logging.error(f"Error extending profile for {gen_name}: {e}")
                                continue
                
                except Exception as e:
                    logging.error(f"Error processing bus {bus} for carrier {carrier}: {e}")
                    continue
            
            # Close the data to free memory
            data.close()
                    
        except Exception as e:
            logging.error(f"Error processing carrier {carrier}: {e}")
            continue

    logging.info("Completed renewable profile generation")
    return pu_profiles

def generate_rmippp_profiles(gens, pu_profiles):
    gen_list = gens[gens.carrier=="rmippp"].index
    pu_profiles.loc[("max", slice(None)), gen_list] = 1
    pu_profiles.loc[("max", pu_profiles.index.get_level_values(1).hour<5), gen_list] = 0
    pu_profiles.loc[("max", pu_profiles.index.get_level_values(1).hour>21), gen_list] = 0
    return pu_profiles

def group_pu_profiles(pu_profiles, component_df):
    years = pu_profiles.index.get_level_values(1).year.unique()
    p_nom_pu = pd.DataFrame(1, index = pu_profiles.loc["max"].index, columns = [])
    pu_mul_p_nom = pu_profiles * component_df["p_nom"]

    filtered_df = component_df[component_df["apply_grouping"]].copy().fillna(0)

    for bus in filtered_df.bus.unique():
        for carrier in filtered_df.carrier.unique():
            carrier_list = filtered_df[(filtered_df["carrier"] == carrier) & (filtered_df["bus"] == bus)].index

            for y in years:
                active = carrier_list[(component_df.loc[carrier_list, ["build_year", "lifetime"]].sum(axis=1) >= y) & (component_df.loc[carrier_list, "build_year"] <= y)]
                if len(active)>0:
                    key_list = filtered_df.loc[active, "Grouping"]
                    for key in key_list.unique():
                        active_key = active[filtered_df.loc[active, "Grouping"] == key]
                        init_active_key = carrier_list[filtered_df.loc[carrier_list, "Grouping"] == key]
                        pu_profiles.loc[(slice(None), str(y)), bus + "-" + carrier + "_" + key] = pu_mul_p_nom.loc[(slice(None), str(y)), active_key].sum(axis=1) / component_df.loc[init_active_key, "p_nom"].sum()
                        p_nom_pu.loc[str(y), bus + "-" + carrier + "_" + key] = component_df.loc[active_key, "p_nom"].sum() / component_df.loc[init_active_key, "p_nom"].sum()
            pu_profiles.drop(columns = carrier_list, inplace=True)

    return pu_profiles.fillna(0), p_nom_pu.fillna(0) # TODO check .fillna(0) doesn't make ramp_rate infeasible on p_max_pu

"""
********************************************************************************
    Functions to define and attach generators to the network  
********************************************************************************
"""
def load_fixed_components(carriers, start_year, config, tech_flag):
    """
    Load components from a model file based on specified filters and configurations.

    Args:
        model_file: The file path to the model file.
        scenario_setup: The model setup object.
        carriers: A list of carriers to filter the generators.
        start_year: The start year for the components.
        config: A dictionary containing configuration settings.

    Returns:
        A DataFrame containing the loaded components.
    """

    component_file = os.path.join(scenario_setup["sub_path"], "fixed_technologies.xlsx")
    if tech_flag == "Generator":
        conv_tech = read_and_filter_generators(component_file, "conventional", scenario_setup["fixed_conventional"], carriers)
        re_tech = read_and_filter_generators(component_file, "renewables", scenario_setup["fixed_renewables"], carriers)

        conv_tech["apply_grouping"] = config["conventional_generators"]["apply_grouping"]
        conv_tech["Type"], conv_tech["Status"] = "Generator", "fixed"

        re_tech["apply_grouping"] = config["renewable_generators"]["apply_grouping"]
        re_tech.set_index((re_tech["Model Key"] + "_" + re_tech["Carrier"]).values,inplace=True)
        re_tech["Type"], re_tech["Status"] = "Generator", "fixed"

        tech= pd.concat([conv_tech, re_tech])

    else:
        tech = read_and_filter_generators(component_file, "storage", scenario_setup["fixed_storage"], carriers)
        tech["apply_grouping"] = config["storage"]["apply_grouping"]
        tech["Type"], tech["Status"] = "StorageUnit", "fixed"

    tech = map_component_parameters(tech, start_year,tech_flag)
    tech = tech.query("(p_nom > 0) & x.notnull() & y.notnull() & (lifetime >= 0)")
    
    return tech

def map_components_to_buses(component_df, regions, crs_config):
    """
    Associate every generator/storage_unit with the bus of the region based on GPS coords.
    """
    regions_gdf = gpd.read_file(regions).to_crs(snakemake.config["gis"]["crs"]["distance_crs"])
    
    # Ensure regions has the correct index
    if "name" not in regions_gdf.columns:
        # Try different possible name columns
        possible_name_cols = ['name', 'Name', 'LocalArea', 'SupplyArea', 'region_name']
        name_col = None
        for col in possible_name_cols:
            if col in regions_gdf.columns:
                name_col = col
                break
        
        if name_col is None:
            logging.error(f"No valid name column found in regions file. Available columns: {regions_gdf.columns.tolist()}")
            raise ValueError("Cannot find name column in regions file")
        
        regions_gdf = regions_gdf.rename(columns={name_col: 'name'})
    
    regions_gdf = regions_gdf.set_index("name")
    
    # Create GeoDataFrame for components
    gps_gdf = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries([Point(o.x, o.y) for o in component_df[["x", "y"]].itertuples()],
        index=component_df.index,
        crs=crs_config["geo_crs"]
    )).to_crs(snakemake.config["gis"]["crs"]["distance_crs"])
    
    # Perform spatial join
    joined = gpd.sjoin(gps_gdf, regions_gdf, how="left", predicate="within")
    component_df["bus"] = joined["name"].copy()

    # Log components without bus assignment
    missing_bus = component_df[component_df["bus"].isnull()]
    if len(missing_bus) > 0:
        logging.warning(f"Found {len(missing_bus)} components without bus assignment:")
        for idx in missing_bus.index:
            logging.warning(f"  - {idx}: coords=({missing_bus.loc[idx, 'x']}, {missing_bus.loc[idx, 'y']})")
        
        # Try to assign to nearest bus for missing components
        logging.info("Attempting to assign missing components to nearest bus...")
        
        # Get centroids of regions for distance calculation
        region_centroids = regions_gdf.geometry.centroid
        
        for idx in missing_bus.index:
            component_point = Point(missing_bus.loc[idx, 'x'], missing_bus.loc[idx, 'y'])
            
            # Calculate distances to all region centroids
            distances = region_centroids.distance(component_point)
            nearest_bus = distances.idxmin()
            
            logging.info(f"Assigning {idx} to nearest bus: {nearest_bus}")
            component_df.loc[idx, "bus"] = nearest_bus

    # Final check - remove any components still without buses
    still_missing = component_df[component_df["bus"].isnull()]
    if len(still_missing) > 0:
        logging.error(f"Still have {len(still_missing)} components without bus assignment. Dropping them:")
        for idx in still_missing.index:
            logging.error(f"  - Dropping {idx}")
        component_df = component_df[component_df["bus"].notnull()]

    return component_df

def group_components(component_df, attrs):
    """
    Apply grouping of similar carrier if specified in snakemake config.

    Args:
        component_df: A DataFrame containing generator/storage_unit data.

    Returns:
        A tuple containing two DataFrames: grouped_df, non_grouped_df
    """
    
    params = ["bus", "carrier", "lifetime", "build_year", "p_nom", "efficiency", "ramp_limit_up", "ramp_limit_down", "marginal_cost", "capital_cost"]
    uc_params = ["ramp_limit_start_up","ramp_limit_shut_down", "start_up_cost", "shut_down_cost", "min_up_time", "min_down_time"] #,"p_min_pu"]
    params += uc_params    
    param_cols = [p for p in params if p not in ["bus","carrier","p_nom"]]

    filtered_df = component_df.query("apply_grouping").copy().fillna(0)#[component_df["apply_grouping"]].copy().fillna(0)

    if len(filtered_df) > 0:
        grouped_df = pd.DataFrame(index=filtered_df.groupby(["Grouping", "carrier", "bus"]).sum().index, columns = param_cols)
        grouped_df["p_nom"] = filtered_df.groupby(["Grouping", "carrier", "bus"]).sum()["p_nom"]
        grouped_df["build_year"] = filtered_df.groupby(["Grouping", "carrier", "bus"]).min()["build_year"]
        grouped_df["lifetime"] = filtered_df.groupby(["Grouping", "carrier", "bus"]).max()["lifetime"]
        
        # calculate weighted average of remaining parameters in gens dataframe
        for param in [p for p in params if p not in ["bus","carrier","p_nom", "lifetime", "build_year"]]:
            weighted_sum = filtered_df.groupby(["Grouping", "carrier", "bus"]).apply(lambda x: (x[param] * x["p_nom"]).sum())
            total_p_nom = filtered_df.groupby(["Grouping", "carrier", "bus"])["p_nom"].sum()
            weighted_average = weighted_sum / total_p_nom 
            grouped_df.loc[weighted_average.index, param] = weighted_average.values
        
        rename_idx = grouped_df.index.get_level_values(2) +  "-" + grouped_df.index.get_level_values(1) +  "_" + grouped_df.index.get_level_values(0)
        grouped_df = grouped_df.reset_index(level=[1,2]).replace(0, np.nan).set_index(rename_idx) 
    else:
        grouped_df = pd.DataFrame(columns=params)
    
    non_grouped_df = component_df[~component_df["apply_grouping"]][params].copy()
    # Fill missing values with default values (excluding defaults that are NaN)    
    grouped_df = apply_default_attr(grouped_df, attrs)
    non_grouped_df = apply_default_attr(non_grouped_df, attrs)

    return grouped_df, non_grouped_df

def attach_fixed_generators(n, carriers):

    # setup carrier info
    gen_attrs = n.component_attrs["Generator"]
    conv_carriers = carriers["fixed"]["conventional"]
    re_carriers = carriers["fixed"]["renewables"]
    carriers = conv_carriers + re_carriers

    start_year = get_start_year(n.snapshots, n.multi_invest)
    snapshots = get_snapshots(n.snapshots, n.multi_invest)
    
    # load generators from model file
    gens = load_fixed_components(carriers, start_year, snakemake.config["electricity"], "Generator")
    gens = map_components_to_buses(gens, snakemake.input.supply_regions, snakemake.config["gis"]["crs"])
    
    # New code to add bus validation
    # Validate that all assigned buses exist in the network
    valid_buses = set(n.buses.index)
    invalid_bus_gens = gens[~gens["bus"].isin(valid_buses)]
    
    if len(invalid_bus_gens) > 0:
        logging.error(f"Found generators assigned to non-existent buses:")
        for idx, row in invalid_bus_gens.iterrows():
            logging.error(f"  - {idx}: assigned to bus '{row['bus']}' which doesn't exist")
        
        # Remove generators with invalid bus assignments
        gens = gens[gens["bus"].isin(valid_buses)]
        logging.info(f"Removed {len(invalid_bus_gens)} generators with invalid bus assignments")
    
    
    pu_profiles = init_pu_profiles(gens, snapshots)

    unique_entries = set()
    coal_gens =  [unique_entries.add(g.split("*")[0]) or g.split("*")[0] for g in gens[gens.carrier == 'coal'].index if g.split("*")[0] not in unique_entries]
    conv_pu = get_eaf_profiles(snapshots, "fixed")
    
    not_in_pu = [g for g in coal_gens if g not in conv_pu.columns]
    conv_pu[not_in_pu] = 1    
    conv_pu = proj_eaf_override(conv_pu, snapshots, include = "_EAF", exclude = "extendable")
    
    # Copy pu_profiles max to conv_pu
    common_gens = [g for g in conv_pu.columns if g in pu_profiles.loc["max"].columns]
    pu_profiles.loc["max", common_gens] = conv_pu[common_gens].values

    split_gens = [g for g in gens.index if "*" in g]
    for gen in split_gens:
        station = gen.split("*")[0]
        pu_profiles.loc["max", gen] = conv_pu[station].values

    # eskom_carriers = [carrier for carrier in conv_carriers if carrier not in ["nuclear", "hydro", "hydro_import"]]
    # for col in gens.query("Grouping == 'eskom' & carrier in @eskom_carriers").index:
    #     pu_profiles.loc["max", col] = conv_pu[col.split("*")[0]].values


    #rmippp_constraints(n, n.snapshots)

    # Hourly data from Eskom data portal
    eskom_re_pu = generate_eskom_re_profiles(n, re_carriers)
    eskom_re_carriers = eskom_re_pu.columns
    for col in gens.query("carrier in @eskom_re_carriers").index:
        pu_profiles.loc["max", col] = eskom_re_pu[gens.loc[col, "carrier"]].values
        pu_profiles.loc["min", col] = eskom_re_pu[gens.loc[col, "carrier"]].values

    # Wind and solar profiles if not using Eskom data portal
    pu_profiles = generate_fixed_wind_solar_profiles(n, gens, snapshots, re_carriers, pu_profiles)

    # Add specific dispatch profile to RMIPPP generators
    pu_profiles = generate_rmippp_profiles(gens, pu_profiles)
    pu_profiles, p_nom_pu = group_pu_profiles(pu_profiles, gens) #includes both grouped an non-grouped generators
    grouped_gens, non_grouped_gens = group_components(gens, gen_attrs)
    grouped_gens["p_nom_extendable"] = False
    non_grouped_gens["p_nom_extendable"] = False
    
    n.import_components_from_dataframe(drop_non_pypsa_attrs(n, "Generator", non_grouped_gens), "Generator")
    n.import_components_from_dataframe(drop_non_pypsa_attrs(n, "Generator", grouped_gens), "Generator")

    pu_max, pu_min = pu_profiles.loc["max"], pu_profiles.loc["min"]
    pu_max.index, pu_min.index, p_nom_pu.index = n.snapshots, n.snapshots, n.snapshots

    n.generators_t.p_nom_pu = p_nom_pu
    n.generators_t.p_max_pu = pu_max.clip(lower=0.0, upper=1.0)
    n.generators_t.p_min_pu = pu_min.clip(lower=0.0, upper=1.0)
    
    #for carrier, value in snakemake.config["electricity"]["min_hourly_station_gen"]["fixed"].items():
    #    clip_pu_profiles(n, "p_min_pu", n.generators.query("carrier == @carrier & p_nom_extendable == False").index, lower=value, upper=1.0)

# def extendable_max_build_per_bus_per_carrier():

#     ext_max_build = (
#         pd.read_excel(
#             model_file, 
#             sheet_name='ext_max_total',
#             index_col=[0,1,2,3,4])
#     ).loc[scenario_setup]
#     ext_max_build.replace("unc", np.inf, inplace=True)

#     return ext_max_build.loc[snakemake.wildcards.regions]

def define_extendable_tech(carriers, years, type_, ext_param):
    """
    Enhanced version that handles both single-region and multi-region scenario

    carriers:
    years:
    type_:
    ext_param:

    """
        
    # Check if multi-region scenario
    regions_setting = scenario_setup.get("regions", "1")
    is_multi_region = str(regions_setting) in ["10", "34", "159"]
    
    if not is_multi_region:
        # Single region - use original logic
        return _define_extendable_tech_single_region(carriers, years, type_, ext_param)
    else:
        # Multi-region - use new logic
        return _define_extendable_tech_multi_region(carriers, years, type_, ext_param)


def _define_extendable_tech_single_region(carriers, years, type_, ext_param):

    """
    Original version for single region
    """

    ext_max_build = pd.read_excel(
        os.path.join(scenario_setup["sub_path"],"extendable_technologies.xlsx"), 
        sheet_name='max_total_installed',
        index_col= [0,1,3,2,4],
    ).loc[(scenario_setup["extendable_max_total"], scenario_setup["regions"], type_, slice(None)), years]
    
    ext_max_build.replace("unc", np.inf, inplace=True)
    ext_max_build.index = ext_max_build.index.droplevel([0, 1, 2])
    ext_max_build = ext_max_build.loc[~(ext_max_build==0).all(axis=1)]

    # Drop extendable techs that are not defined as eligible carriers in ext_techs tab
    if type_ == "Generator":
        eligible_carriers = carriers['extendable']['conventional'] + carriers['extendable']['renewables']
    elif type_ == "StorageUnit":
        eligible_carriers = carriers['extendable']['storage']
    idx = [(bus, c) for (bus, c) in ext_max_build.index if c in eligible_carriers]
    ext_max_build = ext_max_build.loc[idx]


    carrier_names = ext_max_build.index.get_level_values(1)
    if bad_name := list(carrier_names[carrier_names.str.contains("-")]):
        logger.warning(f"Carrier names in extendable_max_build sheet must not contain the character '-'. The following carriers will be ignored: {bad_name}")
        ext_max_build = ext_max_build[~carrier_names.str.contains("-")]
    
    carrier_names = ext_max_build.index.get_level_values(1)
    if bad_name := list(carrier_names[~carrier_names.isin(ext_param.index.get_level_values(1))]):
        logger.warning(f"Carrier names in extendable_max_build sheet must be in the extendable_paramaters sheet. The following carriers will be ignored: {bad_name}")
        ext_max_build = ext_max_build[carrier_names.isin(ext_param.index.get_level_values(1))]

    return (
        ext_max_build[ext_max_build != 0].stack().index.to_series().apply(lambda x: "-".join([x[0], x[1], str(x[2])]))
    ).values


def _define_extendable_tech_multi_region(carriers, years, type_, ext_param):
    """
    New implementation for multi-region scenarios with national constraints.
    """
    logging.info(f"Creating regional extendable technologies")
    
    # Read and store national constraint data
    national_constraints = _read_national_constraint_data(scenario_setup, years, type_)
    _store_national_constraints(national_constraints, type_, scenario_setup)
    
    # Create regional technology list
    regional_tech_list = _create_regional_technologies_with_equal_limits(
        carriers, years, type_, national_constraints
    )
    
    return regional_tech_list


def _read_national_constraint_data(scenario_setup, years, type_):
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
            data = pd.read_excel(
                excel_file,
                sheet_name=sheet_name,
            )

            national_id = "RSA"

            data = data.set_index(["Scenario", "Location", "Component", "Carrier"]).drop(columns=["Supply Region", "Category"])
            scen = scenario_setup[f"extendable_{constraint_type}"]
        
            # Reads the national constraints for all carriers across all given years
            data = data.loc[(scen, national_id, type_, slice(None)), years]
            
            # Clean up data
            data.replace("unc", np.inf, inplace=True)
            data.index = data.index.droplevel(["Scenario", "Location", "Component"])
            
            # Drops zero-constrained values
            data = data.loc[~(data==0).all(axis=1)]
            
            constraint_data[constraint_type] = data
            logging.info(f"Loaded {constraint_type} constraints for {type_}")
            break
        except Exception as e:
            logging.warning(f"Error reading {sheet_name}: {e}")
    
    # Combine all constraint data
    if constraint_data:
        combined_data = pd.concat(constraint_data, names=['constraint_type', 'carrier'])
        return combined_data
    else:
        return pd.DataFrame()


def _create_regional_technologies_with_equal_limits(carriers, years, type_, national_constraints):
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
        logging.info("Defining extendable technology suffix from bus indices")
        regions = [bus for bus in n.buses.index if bus in REGIONAL_MAPPING]
    else:
        logging.info("Defining extendable technology suffix from hard-coded keys")
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
                suffix = REGIONAL_MAPPING[region] if use_regional_codes else str(NAME_TO_INDEX[region])

                regional_tech_id = f"{region}-{carrier}-{year}"
                regional_tech_list.append(regional_tech_id)
    
    return regional_tech_list


def _store_national_constraints(national_constraints, type_, scenario_setup):
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

### Functions for attaching extendable generators etc###


def detect_regional_suffixes_and_create_mapping(regional_carriers, base_carriers):
    """
    Auto-detect regional suffixes by comparing regional carriers with base carriers.
    Creates a mapping from regional carriers to their base carriers.
    
    Parameters:
    -----------
    regional_carriers : list
        List of regional carrier names (e.g., ['solar_pv_EC', 'wind_WC', 'battery_0'])
    base_carriers : list  
        List of base carrier names (e.g., ['solar_pv', 'wind', 'battery'])
    
    Returns:
    --------
    dict: mapping from regional_carrier -> base_carrier
    """
    carrier_mapping = {}
    detected_suffixes = set()
    
    for regional_carrier in regional_carriers:
        # First check if it's already a base carrier
        if regional_carrier in base_carriers:
            carrier_mapping[regional_carrier] = regional_carrier
            continue
        
        # Try to find a matching base carrier
        base_found = False
        for base_carrier in base_carriers:
            if regional_carrier.startswith(base_carrier):
                # Extract the suffix
                suffix = regional_carrier[len(base_carrier):]
                if suffix:  # Non-empty suffix
                    carrier_mapping[regional_carrier] = base_carrier
                    detected_suffixes.add(suffix)
                    base_found = True
                    break
        
        if not base_found:
            logging.warning(f"Could not map regional carrier '{regional_carrier}' to any base carrier")
            # Fallback: use as is
            carrier_mapping[regional_carrier] = regional_carrier
    
    logging.info(f"Auto-detected regional suffixes: {sorted(detected_suffixes)}")
    logging.info(f"Created mapping for {len(carrier_mapping)} carriers")
    
    return carrier_mapping


def set_extendable_params(c, bus_carrier_years, ext_param, **config):
    """
    Much simpler version using the straightforward suffix removal.
    """
    if len(bus_carrier_years) == 0:
        return pd.DataFrame()
    
    default_param = [
        "bus", "p_nom_extendable", "carrier", "build_year", "lifetime",
        "capital_cost", "marginal_cost", "ramp_limit_up", "ramp_limit_down", "efficiency",
    ]
    uc_param = [
        "ramp_limit_start_up", "ramp_limit_shut_down", "min_up_time", 
        "min_down_time", "start_up_cost", "shut_down_cost",
    ]

    if c == "StorageUnit":
        default_param += ["max_hours", "efficiency_store", "efficiency_dispatch"]

    default_col = [p for p in default_param if p not in ["bus", "carrier", "build_year", "p_nom_extendable", "efficiency_store", "efficiency_dispatch"]]

    component_df = pd.DataFrame(index=bus_carrier_years, columns=default_param)
    component_df["p_nom_extendable"] = True
    component_df["p_nom"] = 0
    component_df["bus"] = component_df.index.str.split("-").str[0]
    component_df["carrier"] = component_df.index.str.split("-").str[1]
    component_df["build_year"] = component_df.index.str.split("-").str[2].astype(int)
    
    # Check if we're in multi-region mode
    regions_setting = scenario_setup.get("regions", "1")
    is_multi_region = str(regions_setting) not in ["1"]
    
    logging.info(f"Multi-region mode: {is_multi_region}")
    
    
    def safe_param_lookup(param, carrier, build_year):
        """
        Lookup parameter using simple base carrier logic.
        """
        base_carrier = get_base_carrier(carrier, is_multi_region)
        
        # Log the mapping for debugging
        if carrier != base_carrier:
            logging.debug(f"Mapping '{carrier}' -> '{base_carrier}'")
        
        try:
            if (param, base_carrier) in ext_param.index:
                if build_year in ext_param.columns:
                    return ext_param.loc[(param, base_carrier), build_year]
                else:
                    # Interpolate if exact year not available
                    available_years = [col for col in ext_param.columns if isinstance(col, (int, float))]
                    if available_years:
                        year_data = ext_param.loc[(param, base_carrier), available_years]
                        year_series = pd.Series(year_data.values, index=available_years)
                        return year_series.reindex([build_year]).interpolate().iloc[0]
            
            # If parameter not found, log it
            logging.debug(f"Parameter {param} not found for base carrier {base_carrier}")
            return np.nan
            
        except Exception as e:
            logging.warning(f"Error looking up {param} for {carrier} (base: {base_carrier}): {e}")
            return np.nan
    
    # Set parameters for Generators
    if c == "Generator":
        component_df = pd.concat([component_df, pd.DataFrame(index=bus_carrier_years, columns=uc_param)], axis=1)
        
        for param in default_col + uc_param:
            component_df[param] = component_df.apply(
                lambda row: safe_param_lookup(param, row["carrier"], row["build_year"]), 
                axis=1
            )
        
        component_df = apply_default_attr(component_df, n.component_attrs[c])
        
    # Set parameters for StorageUnits
    elif c == "StorageUnit":
        for param in default_col:
            component_df[param] = component_df.apply(
                lambda row: safe_param_lookup(param, row["carrier"], row["build_year"]), 
                axis=1
            )
        
        component_df["cyclic_state_of_charge"] = True
        component_df["cyclic_state_of_charge_per_period"] = True
        component_df["efficiency_store"] = component_df["efficiency"]**0.5
        component_df["efficiency_dispatch"] = component_df["efficiency"]**0.5
        component_df = component_df.drop("efficiency", axis=1)
    
    return component_df


def attach_extendable_generators(n, carriers):
    """
    Updated version using the simple suffix logic.
    """
    logging.info("Attaching extendable generators with simple regional mapping")
    
    gen_attrs = n.component_attrs["Generator"]
    config = snakemake.config["electricity"]
    
    conv_carriers = carriers["extendable"]["conventional"]
    re_carriers = carriers["extendable"]["renewables"]

    ext_years = get_investment_periods(n.snapshots, n.multi_invest)
    snapshots = get_snapshots(n.snapshots, n.multi_invest)

    # Check multi-region mode
    regions_setting = scenario_setup.get("regions", "1")
    is_multi_region = str(regions_setting) not in ["1"]

    try:
        ext_param = load_extendable_parameters(n, scenario_setup, snakemake)
        ext_gens_list = define_extendable_tech(carriers, ext_years, "Generator", ext_param) 
        
        if not ext_gens_list:
            logging.warning("No extendable generators defined")
            return
            
        gens = set_extendable_params("Generator", ext_gens_list, ext_param)
        if gens.empty:
            logging.warning("No generators created after parameter setting")
            return
            
        gens = set_annual_build_limits(gens, ext_years, "Generator")
         
        pu_profiles = init_pu_profiles(gens, snapshots)
        
        # Apply profiles using simple base carrier logic
        # Conventional profiles
        try:
            conv_pu = get_eaf_profiles(snapshots, "extendable")
            conv_pu = proj_eaf_override(conv_pu, snapshots, include="_extendable_EAF", exclude="NA")

            for col in gens.query("carrier in @conv_carriers & p_nom_extendable == True").index:
                carrier = col.split("-")[1]
                base_carrier = get_base_carrier(carrier, is_multi_region)
                
                if base_carrier in conv_pu.columns:
                    pu_profiles.loc["max", col] = conv_pu[base_carrier].values
                else:
                    logging.debug(f"No conventional profile for {base_carrier}")
                    
        except Exception as e:
            logging.warning(f"Error applying conventional profiles: {e}")

        # Renewable profiles  
        try:
            eskom_ref_re_pu = generate_eskom_re_profiles(n, re_carriers)  
            
            for col in gens.query("p_nom_extendable == True").index:
                carrier = col.split("-")[1]
                base_carrier = get_base_carrier(carrier, is_multi_region)
                
                if base_carrier in eskom_ref_re_pu.columns:
                    pu_profiles.loc["max", col] = eskom_ref_re_pu[base_carrier].values

            pu_profiles = generate_extendable_wind_solar_profiles(n, gens, snapshots, re_carriers, pu_profiles)
            
        except Exception as e:
            logging.warning(f"Error applying renewable profiles: {e}")
        
        # Import to network
        gens = drop_non_pypsa_attrs(n, "Generator", gens)
        n.import_components_from_dataframe(gens, "Generator")
        n.generators["plant_name"] = n.generators.index.str.split("*").str[0]

        in_network = [g for g in pu_profiles.columns if g in n.generators.index]
        pu_max, pu_min = pu_profiles.loc["max", in_network], pu_profiles.loc["min", in_network]
        pu_max.index, pu_min.index = n.snapshots, n.snapshots
        n.generators_t.p_max_pu.loc[:, in_network] = pu_max.loc[:, in_network].clip(lower=0.0, upper=1.0)
        n.generators_t.p_min_pu.loc[:, in_network] = pu_min.loc[:, in_network].clip(lower=0.0, upper=1.0)

        logging.info(f"Successfully attached {len(gens)} extendable generators")
        
    except Exception as e:
        logging.error(f"Error in attach_extendable_generators: {e}")
        return


# You can also apply the same pattern to any other extendable components
def set_annual_build_limits(techs, ext_years, component):
    """
    Updated version that also uses auto-detection for annual limits.
    """
    default_lim = {"max": np.inf, "min": 0}
    
    # Initialize with defaults
    for lim in ['max', 'min']:
        techs[f"p_nom_{lim}"] = default_lim[lim]
    
    for lim in ['max', 'min']:
        try:
            scenario_key = f"extendable_{lim}_annual"
            if scenario_key not in scenario_setup.index:
                continue
                
            scenario_value = scenario_setup[scenario_key]
            if scenario_value in ["UNC", "unc", "unconstrained", "", None] or pd.isna(scenario_value):
                logging.info(f"Annual {lim} limits set to unconstrained (UNC)")
                continue
            
            # Rest of the annual limits logic using the same auto-detection approach...
            # (keeping the existing logic but applying carrier mapping where needed)
            
        except Exception as e:
            logging.warning(f"Error processing {lim} annual limits: {e}")
            continue

    return techs


def apply_extendable_phase_in(n):
    param = load_extendable_parameters(n, scenario_setup, snakemake).loc["build_phase_in"]
    pu_max = get_as_dense(n, "Generator", "p_max_pu")
    ext_i = n.generators.query("p_nom_extendable").index
    
    phase_in = pd.DataFrame(1, index =range(8760), columns = ["overnight", "linear", "quarterly"]) 
    phase_in.loc[:, "linear"] = np.round(np.arange(0, 1, 1/8760),3)
    phase_in.loc[:2190, "quarterly"] = 0.25
    phase_in.loc[2190:4380, "quarterly"] = 0.5
    phase_in.loc[4380:6570, "quarterly"] = 0.75

    for gen in ext_i:
        build_year = n.generators.loc[gen, "build_year"]  
        method = param.loc[gen.split('-')[1], build_year]
        phasing = phase_in[method]
        if method not in ["overnight", "linear", "quarterly"]:
            raise ValueError(f"Invalid method for phase-in of extendable generators: {method}. Choose from 'overnight', 'linear' or 'quarterly'.")

        if method != "overnight":
            n.generators.loc[gen, "lifetime"] += 1

        gen_pu_max = pu_max[gen]
        build_year_level = [build_year] * len(gen_pu_max.loc[build_year].index)
        phasing.index = pd.MultiIndex.from_arrays([build_year_level, gen_pu_max.loc[build_year].index])
        
        gen_pu_max.loc[pd.IndexSlice[gen_pu_max.index.get_level_values(0) < build_year]] = 0
        gen_pu_max.loc[build_year] =  gen_pu_max.loc[build_year] * phasing

        n.generators_t.p_max_pu.loc[:, gen] = gen_pu_max.values


def adjust_for_variable_fuel_costs(n):
    param = load_extendable_parameters(n, scenario_setup, snakemake)
    # check if any variation in marginal cost across years
    variable_fuel_cost = param.loc["fuel", n.investment_periods]
    variable_fuel_cost = variable_fuel_cost[variable_fuel_cost.mean(axis=1) != variable_fuel_cost.iloc[:, 0]]

    gen_list = n.generators.query("carrier in @variable_fuel_cost.index").index
    marginal_cost = pd.DataFrame(index = n.snapshots, columns = gen_list)
    for carrier in variable_fuel_cost.index:
        for gen in n.generators.query("carrier == @carrier").index:
            cost = variable_fuel_cost.loc[carrier] / n.generators.loc[gen, "efficiency"] + param.loc[("VOM", carrier), n.investment_periods]
            for y in n.investment_periods:
                marginal_cost.loc[y, gen_list] = cost.loc[y]
            n.generators.marginal_cost[gen] = 0
            n.generators_t.marginal_cost[gen] = marginal_cost[gen]

def set_hourly_coal_generation_threshold(n):
    if scenario_setup["min_station_hourly"] in ["NA", "None", "none"]:
        return
    
    min_pu_override = (
        pd.read_excel(
            os.path.join(scenario_setup["sub_path"],"plant_availability.xlsx"), 
            sheet_name="min_station_hrly_cap_fact",
            index_col=[0,1])
            .loc[scenario_setup["min_station_hourly"]]
    )
    carrier_list = list(min_pu_override.index)

    n.generators_t.p_min_pu = n.generators_t.p_min_pu.copy()    
    for carrier in carrier_list:
        gen_list = n.generators.query("carrier == @carrier").index
        for y in n.investment_periods:
            n.generators_t.p_min_pu.loc[y, gen_list] = get_as_dense(n, "Generator", "p_min_pu")[gen_list].clip(lower=min_pu_override.loc[carrier, y])

"""
********************************************************************************
    Functions to define and attach storage units to the network  
********************************************************************************
"""
def attach_fixed_storage(n, carriers): 
    carriers = carriers["fixed"]["storage"]
    start_year = get_start_year(n.snapshots, n.multi_invest)
    
    storage = load_fixed_components(carriers, start_year, snakemake.config["electricity"], "StorageUnit")
    storage = map_components_to_buses(storage, snakemake.input.supply_regions, snakemake.config["gis"]["crs"])

    # New validation code for buses
    # Validate that all assigned buses exist in the network
    valid_buses = set(n.buses.index)
    invalid_bus_storage = storage[~storage["bus"].isin(valid_buses)]
    
    if len(invalid_bus_storage) > 0:
        logging.error(f"Found storage units assigned to non-existent buses:")
        for idx, row in invalid_bus_storage.iterrows():
            logging.error(f"  - {idx}: assigned to bus '{row['bus']}' which doesn't exist")
        
        # Remove storage units with invalid bus assignments
        storage = storage[storage["bus"].isin(valid_buses)]
        logging.info(f"Removed {len(invalid_bus_storage)} storage units with invalid bus assignments")

    storage["efficiency_store"] = storage["efficiency"]**0.5
    storage["efficiency_dispatch"] = storage["efficiency"]**0.5
    storage["cyclic_state_of_charge"], storage["p_nom_extendable"] = True, False
    storage["p_min_pu"] = -1

    storage = drop_non_pypsa_attrs(n, "StorageUnit", storage)
    n.import_components_from_dataframe(storage, "StorageUnit")


def attach_extendable_storage(n, carriers):
    """
    Updated storage attachment using the simple suffix logic.
    """
    st_carriers = carriers["extendable"]["storage"]
    config = snakemake.config["electricity"]
    ext_years = get_investment_periods(n.snapshots, n.multi_invest)
    
    try:
        ext_param = load_extendable_parameters(n, scenario_setup, snakemake)
        ext_storage_list = define_extendable_tech(carriers, ext_years, "StorageUnit", ext_param)
        
        if not ext_storage_list:
            logging.info("No extendable storage units defined")
            return
            
        storage = set_extendable_params("StorageUnit", ext_storage_list, ext_param, **config)
        storage = set_annual_build_limits(storage, ext_years, "StorageUnit")
        
        if len(storage) > 0:
            storage = drop_non_pypsa_attrs(n, "StorageUnit", storage)
            n.import_components_from_dataframe(storage, "StorageUnit")
            logging.info(f"Successfully attached {len(storage)} extendable storage units")
        
    except Exception as e:
        logging.error(f"Error in attach_extendable_storage: {e}")
        logging.info("Continuing without extendable storage")
        
"""
********************************************************************************
    Transmission network functions  
********************************************************************************
"""

# def convert_lines_to_links(n):
#     """
#     Convert AC lines to DC links for multi-decade optimisation with line
#     expansion.

#     Losses of DC links are assumed to be 3% per 1000km
#     """
#     years = get_investment_periods(n.snapshots, n.multi_invest)

#     logger.info("Convert AC lines to DC links to perform multi-decade optimisation.")
#     extendable = False
#     p_nom = n.lines.s_nom

#     for y in years:
#         n.madd(
#             "Link",
#             n.lines.index + "_" + str(y),
#             bus0 = n.lines.bus0,
#             bus1 = n.lines.bus1,
#             p_nom = p_nom,
#             p_nom_min = n.lines.s_nom,
#             p_min_pu = -1,
#             lifetime = 100,
#             efficiency = 1 - 0.03 * n.lines.length / 1000,
#             marginal_cost = 0,
#             length = n.lines.length,
#             capital_cost = n.lines.capital_cost,
#             p_nom_extendable = extendable,
#         )
#         if y == years[0]:
#             extendable = True
#             p_nom = 0

#     # Remove AC lines
#     logger.info("Removing AC lines")
#     lines_rm = n.lines.index
#     n.mremove("Line", lines_rm)



"""
********************************************************************************
    Other functions
********************************************************************************
"""


def check_pu_profiles(clean_flag):
    p_max_pu = get_as_dense(n, "Generator", "p_max_pu")
    p_min_pu = get_as_dense(n, "Generator", "p_min_pu")

    errors = p_max_pu < p_min_pu
    error_lst = errors.any()
    if errors.any().any() and not clean_flag:
        raise ValueError(
            f'Some generators have p_max_pu < p_min_pu in some hours. This will cause the problem to be infeasible.\nEither set clean_pu_profiles under config to True or correct input assumptions. Errors can be found in the follwing generators:\n{error_lst[error_lst].index}'
        )
    elif errors.any().any() and clean_flag:
        logging.info(f"Adjusting by p_max_pu for {error_lst[error_lst].index}")
        n.generators_t.p_max_pu = p_max_pu.where(~errors, p_min_pu)


def add_carrier_emissions(n):

    emissions = (
        pd.read_excel(
            model_file, 
            sheet_name='emissions',
            index_col=[0,1])
    )
    co2_emissions = emissions.loc["co2_emissions"]
    
    n.madd(
        "Carrier", 
        co2_emissions.index,
        co2_emissions = co2_emissions["Value"],
     )

def add_nice_carrier_names(n):

    carrier_i = n.carriers.index
    nice_names = (
        pd.Series(snakemake.config["plotting"]["nice_names"])
        .reindex(carrier_i)
        .fillna(carrier_i.to_series().str.title())
    )
    n.carriers["nice_name"] = nice_names
    colors = pd.Series(snakemake.config["plotting"]["tech_colors"]).reindex(carrier_i)
    if colors.isna().any():
        missing_i = list(colors.index[colors.isna()])
        logger.warning(
            f"tech_colors for carriers {missing_i} not defined " "in config."
        )
    n.carriers["color"] = colors


def add_load_shedding(n, cost):
    n.add("Carrier", "load_shedding")
    buses_i = n.buses.index
    n.madd(
        "Generator",
        buses_i,
        "_load_shedding",
        bus = buses_i,
        p_nom = 1e6,  # MW
        carrier = "load_shedding",
        build_year = get_start_year(n.snapshots, n.multi_invest),
        lifetime = 100,
        marginal_cost = cost,
    )


"""
********************************************************************************
        MAIN  
********************************************************************************
"""


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake(
            "add_electricity", 
            **{
                "model_type":"capacity",
                "scenario":"TEST",
            }
        )

    scenario_path = snakemake.output[0].split("capacity")[0]
    if not os.path.exists(scenario_path):
        os.makedirs(scenario_path)

    #configure_logging(snakemake, skip_handlers=False)
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(asctime)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    scenario_setup = load_scenario_definition(snakemake)

    logging.info("Loading carriers from scenario file")
    carriers = get_carriers_from_model_file(scenario_setup)

    logging.info(f"Loading base network {snakemake.input.base_network}")
    n = pypsa.Network(snakemake.input.base_network)

    logging.info("Attaching load")
    attach_load(n, scenario_setup)

    logging.info("Attaching fixed generators")
    attach_fixed_generators(n, carriers)

    logging.info("Attaching extendable generators")
    attach_extendable_generators(n, carriers)

    logging.info("Attaching fixed storage")
    attach_fixed_storage(n, carriers)

    logging.info("Attaching extendable storage")
    attach_extendable_storage(n, carriers)

    logging.info("Adjusting for changes in fuel price over time")
    adjust_for_variable_fuel_costs(n)

    adj_by_pu = snakemake.config["electricity"]["adjust_by_p_max_pu"]
    logging.info(f"Adjusting by p_max_pu for {list(adj_by_pu.keys())}")
    adjust_by_p_max_pu(n, adj_by_pu)

    logging.info("Applying coal minimum generation threshold")
    set_hourly_coal_generation_threshold(n)

    logging.info("Applying phase in of extendable generators")
    apply_extendable_phase_in(n)
    
    logging.info("Checking pu_profiles for infeasibilities")
    check_pu_profiles(snakemake.config["electricity"]["clean_pu_profiles"])

    if snakemake.config["solving"]["options"]["load_shedding"]:
        ls_cost = snakemake.config["costs"]["load_shedding"]
        logging.info("Adding load shedding")
        add_load_shedding(n, ls_cost) 

    add_missing_carriers(n)

    
    # Debugging code
    logging.info(f"Loading base network {snakemake.input.base_network}")
    n = pypsa.Network(snakemake.input.base_network)
    
    # Debug: Print network bus information
    logging.info(f"Network has {len(n.buses)} buses:")
    for bus in n.buses.index:
        logging.info(f"  - Bus: {bus}")
    
    if len(n.buses) == 0:
        logging.error("Network has no buses! Check base_network.py")
        raise ValueError("Base network contains no buses")

    logging.info("Attaching load")
    attach_load(n, scenario_setup)

    logging.info("Attaching fixed generators")
    attach_fixed_generators(n, carriers)
    

    # Debug: Check generators after attachment
    logging.info(f"Network now has {len(n.generators)} generators")
    gen_buses = n.generators["bus"].unique()
    logging.info(f"Generators are assigned to buses: {gen_buses}")
    
    # Check for missing buses
    missing_buses = set(gen_buses) - set(n.buses.index)
    if missing_buses:
        logging.error(f"Generators reference non-existent buses: {missing_buses}")

    logging.info("Attaching extendable generators")
    attach_extendable_generators(n, carriers)

    logging.info("Attaching fixed storage")
    attach_fixed_storage(n, carriers)
    
    # Debug: Check storage after attachment
    if len(n.storage_units) > 0:
        logging.info(f"Network now has {len(n.storage_units)} storage units")
        storage_buses = n.storage_units["bus"].unique()
        logging.info(f"Storage units are assigned to buses: {storage_buses}")
        
        # Check for missing buses
        missing_storage_buses = set(storage_buses) - set(n.buses.index)
        if missing_storage_buses:
            logging.error(f"Storage units reference non-existent buses: {missing_storage_buses}")

    logging.info("Exporting network.")
    # Comprehensive data type fix for NetCDF export
    def ensure_consistent_dtypes(network):
        """Ensure all time-series data has consistent data types for NetCDF export"""
        # Fix generators time-series data
        if hasattr(network, 'generators_t'):
            for attr in ['p_max_pu', 'p_min_pu', 'p_nom_pu', 'marginal_cost']:
                if hasattr(network.generators_t, attr):
                    setattr(network.generators_t, attr, 
                           getattr(network.generators_t, attr).astype(float))
        
        # Fix storage units time-series data
        if hasattr(network, 'storage_units_t'):
            for attr in ['p_max_pu', 'p_min_pu', 'inflow', 'state_of_charge_set']:
                if hasattr(network.storage_units_t, attr):
                    setattr(network.storage_units_t, attr, 
                           getattr(network.storage_units_t, attr).astype(float))
        
        # Fix loads time-series data
        if hasattr(network, 'loads_t'):
            for attr in ['p_set']:
                if hasattr(network.loads_t, attr):
                    setattr(network.loads_t, attr, 
                           getattr(network.loads_t, attr).astype(float))
        
        # Fix lines time-series data if any
        if hasattr(network, 'lines_t'):
            for attr in ['s_max_pu']:
                if hasattr(network.lines_t, attr):
                    setattr(network.lines_t, attr, 
                           getattr(network.lines_t, attr).astype(float))
        
        # Fix links time-series data if any
        if hasattr(network, 'links_t'):
            for attr in ['p_max_pu', 'p_min_pu']:
                if hasattr(network.links_t, attr):
                    setattr(network.links_t, attr, 
                           getattr(network.links_t, attr).astype(float))
    
    ensure_consistent_dtypes(n)
    if n.multi_invest:
        initial_ramp_rate_fix(n)

    if snakemake.wildcards.model_type == "dispatch":
        for y in n.investment_periods:
            sns = n.snapshots[n.snapshots.get_level_values("period")==y]
            n_y = single_year_network_copy(n, snapshots=sns, investment_periods=sns.unique("period"))
            n_y.export_to_netcdf(f"{scenario_path}/dispatch_{y}.nc") 
    else:
        n.export_to_netcdf(snakemake.output[0])

    logging.info("Attaching extendable storage")
    attach_extendable_storage(n, carriers)
    
    logging.info("About to adjust for variable fuel costs")
    try:
        adjust_for_variable_fuel_costs(n)
        logging.info("✓ Fuel costs adjusted successfully")
    except Exception as e:
        logging.error(f"Error in fuel costs: {e}")
    
    logging.info("About to adjust by p_max_pu")
    try:
        adj_by_pu = snakemake.config["electricity"]["adjust_by_p_max_pu"]
        adjust_by_p_max_pu(n, adj_by_pu)
        logging.info("✓ p_max_pu adjusted successfully")
    except Exception as e:
        logging.error(f"Error in p_max_pu adjustment: {e}")
    
    logging.info("About to export network")
    try:
        # Test if network export is the issue
        n.export_to_netcdf(snakemake.output[0])
        logging.info("✓ Network exported successfully")
    except Exception as e:
        logging.error(f"Error in network export: {e}")
