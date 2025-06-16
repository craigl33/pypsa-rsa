configfile: "config.yaml"

from os.path import normpath, exists, isdir
import pandas as pd
import os
from pathlib import Path
import xarray as xr
from tsam.timeseriesaggregation import TimeSeriesAggregation


wildcard_constraints:
    model_type="[a-zA-Z0-9_]+",
    scenarios_to_run="[-a-zA-Z0-9_]+",

# Read scenarios
scenarios_folder = Path(config['scenarios']['path']).expanduser() / config['scenarios']['folder']
scenarios_file = scenarios_folder / config['scenarios']['setup']
print(f"Looking for scenarios file: {scenarios_file}")


scenarios = pd.read_excel(
    scenarios_file,
    sheet_name="scenario_definition",
    engine="openpyxl"
)
# Set first column as index manually
scenarios = scenarios.set_index(scenarios.columns[0])
scenarios.index = scenarios.index.astype(str)

scenarios_to_run = scenarios[scenarios["run_scenario"] == True]  # Use True instead of 1
print(f"Successfully loaded {len(scenarios)} scenarios")
print(f"Columns: {scenarios.columns.tolist()}")

gis_path = Path(config["gis"]["path"]).expanduser().resolve()
config["gis"]["path"] = str(gis_path)
print(f"Resolved GIS path to: {config['gis']['path']}")


rule all:
    input:
        "results/solve_all_scenarios_complete"

rule solve_all_scenarios:
    input:
        expand(
            "results/" + config["scenarios"]["folder"] + "/network/capacity-{scenario}.nc",
            scenario=scenarios_to_run.index,
        )
    output:
        touch("results/solve_all_scenarios_complete")
		
rule build_topology:
    input:
        supply_regions=config["gis"]["path"] + "/supply_regions/rsa_supply_regions.gpkg",
        existing_lines=config["gis"]["path"] + "/transmission_grid/eskom_gcca_2022/Existing_Lines.shp",
        planned_lines=config["gis"]["path"] + "/transmission_grid/tdp_digitised/TDP_2023_32.shp",
        gdp_pop_data=config["gis"]["path"] + "/CSIR/Mesozones/Mesozones.shp",        
    output:
        buses="resources/"+config["scenarios"]["folder"]+"/buses-{scenario}.geojson",
        lines="resources/"+config["scenarios"]["folder"]+"/lines-{scenario}.geojson",
    threads: 1
    script: "scripts/build_topology.py"

rule base_network:
    input:
        buses="resources/" + config["scenarios"]["folder"] + "/buses-{scenario}.geojson",
        lines="resources/" + config["scenarios"]["folder"] + "/lines-{scenario}.geojson",
    output: "networks/" + config["scenarios"]["folder"] + "/base/{model_type}-{scenario}.nc",
    threads: 1
    resources: mem_mb=1000
    script: "scripts/base_network.py"

rule add_electricity:
    input:
        base_network="networks/" + config["scenarios"]["folder"] + "/base/{model_type}-{scenario}.nc",
        supply_regions="resources/" + config["scenarios"]["folder"] + "/buses-{scenario}.geojson",
        load="data/eskom_data.csv",
        eskom_profiles="data/eskom_pu_profiles.csv",
        renewable_profiles="pre_processing/resource_processing/renewable_profiles_updated.nc",
    output: "networks/"+ config["scenarios"]["folder"] + "/elec/{model_type}-{scenario}.nc",
    script: "scripts/add_electricity.py"


rule prepare_and_solve_network:
    input:
        network="networks/"+ config["scenarios"]["folder"] + "/elec/capacity-{scenario}.nc",
    output: 
        network="results/" + config["scenarios"]["folder"] + "/network/capacity-{scenario}.nc",
        network_stats="results/" + config["scenarios"]["folder"] +"/network_stats/{scenario}.csv",
        emissions="results/" + config["scenarios"]["folder"] +"/emissions/{scenario}.csv",
    resources:
        solver_slots=1
    script:
        "scripts/prepare_and_solve_network.py"

rule cluster_time:
    input:
        demand="resources/load_data/{year}.csv",  # Adjust to match your data source
    output:
        "resources/time_clustering/{year}.nc",
    params:
        periods=lambda wildcards: config["time"]["clustering"].get("snapshots", 8760),
        method=lambda wildcards: config["time"]["clustering"].get("how", "representative"),
    run:
        print("Running TSAM for time series aggregation...")
        df = pd.read_csv(input.demand, index_col=0, parse_dates=True)

        tsam_model = TimeSeriesAggregation(
            raw=df,
            noTypicalPeriods=params.periods,
            hoursPerPeriod=24,
            clusterMethod='hierarchical',
            representationMethod=params.method,
            standardize=True
        )

        typical_periods = tsam_model.createTypicalPeriods()

        ds = xr.Dataset({
            'typical_periods': (['period', 'hour', 'feature'], typical_periods.values),
        })
        ds.coords['period'] = range(typical_periods.shape[0])
        ds.coords['hour'] = range(typical_periods.shape[1])
        ds.coords['feature'] = typical_periods.columns

        ds.to_netcdf(output[0])

rule solve_network_dispatch:
    input:
        dispatch_network="networks/elec/{scenario}/dispatch-{year}.nc",
        optimised_network_stats="networks/network_stats/{scenario}.csv",
    output: "networks/dispatch/{scenario}/dispatch_{year}.nc",
    script:
        "scripts/solve_network_dispatch.py"