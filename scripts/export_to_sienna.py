# -*- coding: utf-8 -*-
"""
PyPSA to PowerSystems.jl 4.6.2 Compatible CSV Export Implementation

This module provides comprehensive functionality to export PyPSA dispatch networks 
to PowerSystems.jl-compatible CSV format for use with the Sienna ecosystem.

The implementation creates all necessary CSV files, configuration files, and Julia
import scripts required for PowerSystems.jl 4.6.2 compatibility.

Key Features:
- Unified generator export in single gen.csv file
- Proper PowerSystems.jl field mappings and data types
- Comprehensive time series data export
- Automatic generator type classification via fuel/prime_mover mapping
- Full PowerSystems.jl 4.6.2 API compatibility

Example Usage:
    ```python
    from export_to_sienna import export_pypsa_to_sienna
    
    # Export PyPSA network to Sienna format
    export_results = export_pypsa_to_sienna(
        network=pypsa_network,
        scenario_setup=scenario_config,
        output_dir="/path/to/output",
        include_time_series=True
    )
    ```
"""

import pypsa
import pandas as pd
import numpy as np
import os
import yaml
import json
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional, Tuple

from _helpers import load_scenario_definition

logger = logging.getLogger(__name__)


class PyPSAToSiennaExporter:
    """
    Exports PyPSA dispatch networks to PowerSystems.jl 4.6.2 compatible CSV format.
    
    This class handles the complete conversion process from PyPSA network data structures
    to the CSV files, configuration files, and import scripts required by PowerSystems.jl.
    
    The exporter creates all necessary files for PowerSystems.jl including:
    - Component CSV files (bus.csv, gen.csv, load.csv, branch.csv, etc.)
    - Configuration files (user_descriptors.yaml, generator_mapping.yaml)
    - Time series data files and metadata
    - Julia import script for PowerSystems.jl 4.6.2
    
    Attributes:
        network (pypsa.Network): The PyPSA network to export
        scenario_setup (dict): Scenario configuration parameters
        base_power (float): Base power for per-unit calculations (MVA)
        generator_details (dict): Additional generator data from scenario files
        time_series_metadata (list): Metadata for time series data linking
        network_summary (dict): Summary of network components and structure
    """
    
    def __init__(self, network: pypsa.Network, scenario_setup: dict):
        """
        Initialize the PyPSA to Sienna exporter.
        
        Args:
            network (pypsa.Network): The solved PyPSA dispatch network to export
            scenario_setup (dict): Scenario configuration from load_scenario_definition()
                Must contain paths to additional data files and scenario parameters
        """
        self.network = network
        self.scenario_setup = scenario_setup
        self.base_power = 100.0  # MVA base for PowerSystems.jl (standard)
        
        # Time series storage
        self.time_series_data = {}
        self.time_series_metadata = []
        
        # Load additional generator data from scenario files if available
        self.generator_details = self._load_generator_details()
        
        # Analyze network structure
        self._analyze_network()
    
    def _load_generator_details(self) -> Dict[str, pd.DataFrame]:
        """
        Load detailed generator data from scenario Excel files.
        
        Attempts to load additional generator information from the scenario's
        fixed_technologies.xlsx and extendable_technologies.xlsx files to
        enhance the exported data with detailed operational parameters.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing loaded generator details
                with keys 'conventional', 'renewables', and 'extendable_params'
        """
        generator_details = {}
        
        try:
            # Load fixed technologies data
            fixed_tech_file = os.path.join(self.scenario_setup.get("sub_path", ""), "fixed_technologies.xlsx")
            if os.path.exists(fixed_tech_file):
                try:
                    # Load conventional generators
                    conv_gens = pd.read_excel(fixed_tech_file, sheet_name="conventional", index_col=0)
                    generator_details['conventional'] = conv_gens
                    logger.info(f"Loaded {len(conv_gens)} conventional generator details")
                except Exception as e:
                    logger.warning(f"Could not load conventional generators: {e}")
                
                try:
                    # Load renewable generators
                    renew_gens = pd.read_excel(fixed_tech_file, sheet_name="renewables", index_col=0)
                    generator_details['renewables'] = renew_gens
                    logger.info(f"Loaded {len(renew_gens)} renewable generator details")
                except Exception as e:
                    logger.warning(f"Could not load renewable generators: {e}")
            
            # Load extendable technologies data
            ext_tech_file = os.path.join(self.scenario_setup.get("sub_path", ""), "extendable_technologies.xlsx")
            if os.path.exists(ext_tech_file):
                try:
                    # Load parameters sheet
                    ext_params = pd.read_excel(ext_tech_file, sheet_name="parameters", index_col=[0, 1, 2])
                    generator_details['extendable_params'] = ext_params
                    logger.info(f"Loaded extendable technology parameters")
                except Exception as e:
                    logger.warning(f"Could not load extendable parameters: {e}")
        
        except Exception as e:
            logger.warning(f"Could not load generator detail files: {e}")
        
        return generator_details
    
    def _get_generator_detail(self, gen_name: str, field: str, default_value=None):
        """
        Retrieve detailed information about a specific generator.
        
        Searches through the loaded generator details to find additional
        information about a generator that may not be present in the PyPSA
        network data structures.
        
        Args:
            gen_name (str): Name of the generator to look up
            field (str): Field name to retrieve (e.g., 'startup_cost', 'heat_rate')
            default_value: Default value to return if field is not found
            
        Returns:
            The requested field value if found, otherwise the default_value
        """
        try:
            # Check conventional generator data
            if 'conventional' in self.generator_details:
                conv_data = self.generator_details['conventional']
                if gen_name in conv_data.index and field in conv_data.columns:
                    value = conv_data.loc[gen_name, field]
                    if pd.notna(value):
                        return value
            
            # Check renewable generator data
            if 'renewables' in self.generator_details:
                renew_data = self.generator_details['renewables']
                if gen_name in renew_data.index and field in renew_data.columns:
                    value = renew_data.loc[gen_name, field]
                    if pd.notna(value):
                        return value
            
            # Check extendable parameters for regional generators
            if 'extendable_params' in self.generator_details:
                ext_data = self.generator_details['extendable_params']
                # Extract carrier from generator name for extendable generators
                if '-' in gen_name:
                    carrier = gen_name.split('-')[1]
                    try:
                        if (field, carrier, 2030) in ext_data.index:
                            value = ext_data.loc[(field, carrier, 2030), '2030']
                            if pd.notna(value):
                                return value
                    except:
                        pass
        
        except Exception as e:
            logger.debug(f"Error getting detail for {gen_name}.{field}: {e}")
        
        return default_value
    
    def _analyze_network(self):
        """
        Analyze the PyPSA network to understand structure and available data.
        
        Examines the network to determine which components are present,
        their quantities, and available data fields. This information is
        used to guide the export process and provide user feedback.
        
        Populates self.network_summary with component counts and available fields.
        """
        self.network_summary = {}
        
        # Analyze static components
        for component_name in ['Bus', 'Generator', 'Load', 'Line', 'Transformer', 'Link', 'StorageUnit']:
            if hasattr(self.network, component_name.lower() + 's'):
                component_df = getattr(self.network, component_name.lower() + 's')
            elif hasattr(self.network, component_name.lower() + 'es'):
                component_df = getattr(self.network, component_name.lower() + 'es')
            else:
                component_df = pd.DataFrame()
            
            if not component_df.empty:
                self.network_summary[component_name] = {
                    'count': len(component_df),
                    'columns': list(component_df.columns)
                }
        
        # Analyze time series data availability
        self.network_summary['time_series'] = {}
        for component_name in self.network_summary.keys():
            if component_name != 'time_series':
                ts_attr = getattr(self.network, component_name.lower() + 's_t', None)
                if ts_attr is None:
                    ts_attr = getattr(self.network, component_name.lower() + 'es_t', None)
                    
                if ts_attr is not None:
                    ts_data = {}
                    for attr_name in dir(ts_attr):
                        if not attr_name.startswith('_'):
                            attr_data = getattr(ts_attr, attr_name)
                            if hasattr(attr_data, 'shape') and not attr_data.empty:
                                ts_data[attr_name] = attr_data.shape
                    if ts_data:
                        self.network_summary['time_series'][component_name] = ts_data
        
        logger.info(f"Network analysis complete: {self.network_summary}")
    
    def export_to_csv(self, output_dir: str, include_time_series: bool = True) -> Dict[str, str]:
        """
        Main export function to convert PyPSA network to PowerSystems.jl format.
        
        Creates all necessary CSV files, configuration files, and import scripts
        required for PowerSystems.jl 4.6.2 compatibility.
        
        Args:
            output_dir (str): Directory path where exported files will be created
            include_time_series (bool): Whether to export time series data files
                Defaults to True. Set to False for static network export only.
                
        Returns:
            Dict[str, str]: Dictionary mapping file types to their created file paths
                Includes paths to all CSV files, configuration files, and Julia script
        """
        logger.info("Starting PyPSA to PowerSystems.jl CSV export...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        files_created = {}
        
        # Export static component data
        static_files = self._export_static_components(output_path)
        files_created.update(static_files)
        
        # Export time series data
        if include_time_series:
            ts_files = self._export_time_series_data(output_path)
            files_created.update(ts_files)
        
        # Create configuration files
        config_files = self._create_configuration_files(output_path)
        files_created.update(config_files)
        
        # Create Julia import script
        julia_script = self._create_julia_import_script(output_path)
        files_created['julia_script'] = str(julia_script)
        
        logger.info(f"Export complete. Created {len(files_created)} files.")
        return files_created
    
    def _export_static_components(self, output_path: Path) -> Dict[str, str]:
        """
        Export all static component data in PowerSystems.jl 4.6.2 format.
        
        Creates the core CSV files required by PowerSystems.jl with proper
        field mappings and data types. Components are exported in dependency
        order (buses first, then components that reference buses).
        
        Args:
            output_path (Path): Directory where CSV files will be created
            
        Returns:
            Dict[str, str]: Mapping of component types to created file paths
        """
        files_created = {}
        
        # Export buses first (other components reference them)
        if 'Bus' in self.network_summary:
            bus_file = self._export_buses(output_path)
            files_created.update(bus_file)
        
        # Export ALL generators in a SINGLE gen.csv file (PowerSystems.jl requirement)
        if 'Generator' in self.network_summary:
            gen_files = self._export_generators_unified(output_path)
            files_created.update(gen_files)
        
        # Export loads
        if 'Load' in self.network_summary:
            load_files = self._export_loads(output_path)
            files_created.update(load_files)
        
        # Export branches (lines + transformers)
        branch_files = self._export_branches(output_path)
        files_created.update(branch_files)
        
        # Export DC branches (links) if they exist
        if 'Link' in self.network_summary:
            dc_files = self._export_dc_branches(output_path)
            files_created.update(dc_files)
        
        # Export storage
        if 'StorageUnit' in self.network_summary:
            storage_files = self._export_storage(output_path)
            files_created.update(storage_files)
        
        return files_created
    
    def _export_buses(self, output_path: Path) -> Dict[str, str]:
        """
        Export bus data in PowerSystems.jl 4.6.2 compatible format.
        
        Creates bus.csv with all required fields for PowerSystems.jl including
        voltage levels, bus types, geographic coordinates, and area/zone assignments.
        
        Args:
            output_path (Path): Directory where bus.csv will be created
            
        Returns:
            Dict[str, str]: Mapping with 'bus' key pointing to created file path
        """
        buses_df = self.network.buses.copy()
        
        if buses_df.empty:
            logger.warning("No buses to export")
            return {}
        
        # Create PowerSystems.jl compatible bus format
        sienna_buses = pd.DataFrame(index=buses_df.index)
        
        # Required fields for PowerSystems.jl 4.6.2
        sienna_buses['name'] = buses_df.index
        sienna_buses['base_voltage'] = buses_df['v_nom']  # kV
        sienna_buses['bus_type'] = 'PQ'  # Default, will be updated
        sienna_buses['voltage'] = 1.0  # p.u. initial voltage
        sienna_buses['angle'] = 0.0  # Initial voltage angle (radians)
        
        # Optional geographic coordinates
        if 'x' in buses_df.columns and 'y' in buses_df.columns:
            sienna_buses['longitude'] = buses_df['x']
            sienna_buses['latitude'] = buses_df['y']
        else:
            sienna_buses['longitude'] = 0.0
            sienna_buses['latitude'] = 0.0
        
        # Set proper bus types based on connected components
        self._set_bus_types(sienna_buses)
        
        # Add areas/zones for PowerSystems.jl aggregation
        if 'carrier' in buses_df.columns:
            sienna_buses['area'] = buses_df['carrier']
            sienna_buses['zone'] = buses_df['carrier']
        else:
            sienna_buses['area'] = 1
            sienna_buses['zone'] = 1
        
        # Export to CSV
        bus_file = output_path / 'bus.csv'
        sienna_buses.to_csv(bus_file, index=False)
        
        logger.info(f"Exported {len(sienna_buses)} buses to {bus_file}")
        return {'bus': str(bus_file)}
    
    def _set_bus_types(self, sienna_buses: pd.DataFrame):
        """
        Set appropriate bus types for PowerSystems.jl based on connected components.
        
        Assigns bus types following power system conventions:
        - REF: Reference/slack bus (typically the largest generator bus)
        - PV: Generator buses (voltage controlled)
        - PQ: Load buses (power specified)
        
        Args:
            sienna_buses (pd.DataFrame): Bus dataframe to modify in-place
        """
        # Initialize all buses as PQ (load buses)
        sienna_buses['bus_type'] = 'PQ'
        
        # Set one slack bus (REF) - choose bus with largest generator
        if hasattr(self.network, 'generators') and not self.network.generators.empty:
            gen_capacity_by_bus = self.network.generators.groupby('bus')['p_nom'].sum()
            slack_bus = gen_capacity_by_bus.idxmax()
            sienna_buses.loc[sienna_buses['name'] == slack_bus, 'bus_type'] = 'REF'
        else:
            # If no generators, set first bus as slack
            sienna_buses.iloc[0, sienna_buses.columns.get_loc('bus_type')] = 'REF'
        
        # Set PV buses (generator buses that are not slack)
        if hasattr(self.network, 'generators') and not self.network.generators.empty:
            gen_buses = set(self.network.generators['bus'].unique())
            slack_bus_name = sienna_buses[sienna_buses['bus_type'] == 'REF']['name'].iloc[0]
            pv_buses = gen_buses - {slack_bus_name}
            
            for bus in pv_buses:
                sienna_buses.loc[sienna_buses['name'] == bus, 'bus_type'] = 'PV'

    def _export_generators_unified(self, output_path: Path) -> Dict[str, str]:
        """
        Export ALL generators in a single gen.csv file for PowerSystems.jl 4.6.2.
        
        Creates a unified generator file with all thermal and renewable generators
        together, using fuel and type fields to enable PowerSystems.jl's automatic
        generator type classification system.
        
        FIXED: Eliminates pandas SettingWithCopyWarning by proper DataFrame construction.
        
        Args:
            output_path (Path): Directory where gen.csv will be created
            
        Returns:
            Dict[str, str]: Mapping with 'gen' key pointing to created file path
        """
        generators_df = self.network.generators.copy()
        
        if generators_df.empty:
            logger.warning("No generators to export")
            return {}
        
        # Create unified generator CSV with all required PowerSystems.jl fields
        # Use proper DataFrame construction to avoid pandas warnings
        sienna_gen = pd.DataFrame(index=generators_df.index)
        
        # Core required fields
        sienna_gen['name'] = generators_df.index.values
        sienna_gen['bus'] = generators_df['bus'].values
        
        # Power limits (MW)
        sienna_gen['active_power'] = generators_df.get('p_nom', 0.0).values
        sienna_gen['max_active_power'] = generators_df['p_nom'].values
        
        # Calculate minimum power from p_min_pu
        p_min_pu = generators_df.get('p_min_pu', 0.0).copy()
        if hasattr(self.network, 'generators_t') and hasattr(self.network.generators_t, 'p_min_pu'):
            # Use time series min if available (take mean)
            ts_p_min = self.network.generators_t.p_min_pu
            for gen in generators_df.index:
                if gen in ts_p_min.columns and not ts_p_min[gen].empty:
                    p_min_pu.loc[gen] = ts_p_min[gen].mean()
        
        sienna_gen['min_active_power'] = (generators_df['p_nom'] * p_min_pu).values
        
        # Reactive power limits (MVAr) - PowerSystems.jl requirement
        sienna_gen['max_reactive_power'] = (generators_df['p_nom'] * 0.3).values  # Default 0.3 power factor
        sienna_gen['min_reactive_power'] = (-generators_df['p_nom'] * 0.3).values
        
        # Cost information (currency/MWh)
        sienna_gen['variable'] = generators_df.get('marginal_cost', 0.0).values
        sienna_gen['startup'] = 0.0  # Default startup cost
        sienna_gen['shutdown'] = 0.0  # Default shutdown cost
        
        # Fuel and prime_mover mapping for PowerSystems.jl generator classification
        fuel_mapping = self._get_fuel_mapping()
        prime_mover_mapping = self._get_prime_mover_mapping()
        
        # Create lists to avoid pandas slice warnings
        fuel_values = []
        type_values = []
        
        for gen in generators_df.index:
            carrier = generators_df.loc[gen, 'carrier']
            fuel = fuel_mapping.get(carrier.lower(), 'OTHER')
            prime_mover = prime_mover_mapping.get(carrier.lower(), 'OT')
            
            fuel_values.append(fuel)
            type_values.append(prime_mover)
        
        sienna_gen['fuel'] = fuel_values
        sienna_gen['type'] = type_values
        
        # Operational parameters
        sienna_gen['ramp_30'] = generators_df['p_nom'].values  # Default: full capacity in 30 min
        sienna_gen['ramp_10'] = (generators_df['p_nom'] * 0.33).values  # Default: 1/3 capacity in 10 min
        
        # Unit commitment parameters (for thermal generators)
        sienna_gen['min_up_time'] = 1.0  # hours
        sienna_gen['min_down_time'] = 1.0  # hours
        
        # Status and availability
        sienna_gen['available'] = True
        sienna_gen['status'] = 1  # 1 = online, 0 = offline
        
        # Fill any NaN values with appropriate defaults
        sienna_gen = sienna_gen.fillna({
            'active_power': 0.0,
            'max_active_power': 0.0,
            'min_active_power': 0.0,
            'max_reactive_power': 0.0,
            'min_reactive_power': 0.0,
            'variable': 0.0,
            'startup': 0.0,
            'shutdown': 0.0,
            'fuel': 'OTHER',
            'type': 'OT',
            'ramp_30': 1000.0,
            'ramp_10': 333.0,
            'min_up_time': 1.0,
            'min_down_time': 1.0,
            'available': True,
            'status': 1
        })
        
        # Export to CSV with correct filename for PowerSystems.jl
        gen_file = output_path / 'gen.csv'  # Must be 'gen.csv' for PowerSystems.jl
        sienna_gen.to_csv(gen_file, index=False)
        
        logger.info(f"Exported {len(sienna_gen)} generators to {gen_file}")
        return {'gen': str(gen_file)}
    
    def _get_fuel_mapping(self) -> Dict[str, str]:
        """
        Get fuel type mapping from PyPSA carriers to PowerSystems.jl fuel types.
        
        Maps PyPSA generator carrier names to standardized PowerSystems.jl
        fuel type enumerations used for generator classification.
        
        UPDATED: Includes South African specific carriers like 'rmippp' and 'bioenergy'.

        TODO: In future iterations, this may be best housed in an external CSV
        
        Returns:
            Dict[str, str]: Mapping from PyPSA carrier names to PowerSystems.jl fuel types
        """
        return {
            'gas': 'NATURAL_GAS',
            'ccgt': 'NATURAL_GAS',
            'ccgt_gas': 'NATURAL_GAS',
            'ocgt': 'NATURAL_GAS',
            'ocgt_gas': 'NATURAL_GAS',
            'ocgt_diesel': 'DIESEL',
            'coal': 'COAL',
            'lignite': 'COAL',
            'oil': 'DIESEL',
            'nuclear': 'NUCLEAR',
            'biomass': 'BIOMASS',
            'waste': 'BIOMASS',
            'wind': 'WIND',
            'wind_onshore': 'WIND',
            'wind_offshore': 'WIND',
            'solar': 'SOLAR',
            'solar_pv': 'SOLAR',
            'pv': 'SOLAR',
            'hydro': 'HYDRO',
            'ror': 'HYDRO',
            'geothermal': 'GEOTHERMAL',
            # South African specific carriers
            'rmippp': 'BIOMASS',  # Renewable Energy IPP Program (often biomass/waste)
            'bioenergy': 'BIOMASS'  # Bioenergy plants
        }

    def _get_prime_mover_mapping(self) -> Dict[str, str]:
        """
        Get prime mover type mapping from PyPSA carriers to PowerSystems.jl types.
        
        Maps PyPSA generator carrier names to PowerSystems.jl prime mover type
        enumerations used for detailed generator technology classification.
        
        UPDATED: Includes South African specific carriers like 'rmippp' and 'bioenergy'.
        
        Returns:
            Dict[str, str]: Mapping from PyPSA carrier names to PowerSystems.jl prime mover types
        """
        return {
            'gas': 'CC',  # Combined Cycle
            'ccgt': 'CC',
            'ccgt_gas': 'CC',
            'ocgt': 'CT',  # Combustion Turbine
            'ocgt_gas': 'CT',
            'ocgt_diesel': 'CT',
            'coal': 'ST',  # Steam Turbine
            'lignite': 'ST',
            'oil': 'IC',  # Internal Combustion
            'nuclear': 'ST',
            'biomass': 'ST',
            'waste': 'ST',
            'wind': 'WT',  # Wind Turbine
            'wind_onshore': 'WT',
            'wind_offshore': 'WT',
            'solar': 'PV',  # Photovoltaic
            'solar_pv': 'PV',
            'pv': 'PV',
            'hydro': 'HY',  # Hydro
            'ror': 'HY',
            'geothermal': 'ST',
            # South African specific carriers
            'rmippp': 'ST',  # Steam turbine for renewable IPP program
            'bioenergy': 'ST'  # Steam turbine for bioenergy
        }


    def _export_loads(self, output_path: Path) -> Dict[str, str]:
        """
        Export electrical loads in PowerSystems.jl 4.6.2 format.
        
        Creates load.csv with load specifications including maximum power
        levels and power factor assumptions.
        
        Args:
            output_path (Path): Directory where load.csv will be created
            
        Returns:
            Dict[str, str]: Mapping with 'load' key pointing to created file path
        """
        loads_df = self.network.loads.copy()
        
        if loads_df.empty:
            logger.warning("No loads to export")
            return {}
        
        sienna_loads = pd.DataFrame(index=loads_df.index)
        sienna_loads['name'] = loads_df.index
        sienna_loads['bus'] = loads_df['bus']
        
        # Get load values - use max from time series if available
        max_active_power = loads_df['p_set'].copy()
        if hasattr(self.network, 'loads_t') and hasattr(self.network.loads_t, 'p_set'):
            ts_loads = self.network.loads_t.p_set
            for load in loads_df.index:
                if load in ts_loads.columns and not ts_loads[load].empty:
                    max_active_power.loc[load] = ts_loads[load].max()
        
        sienna_loads['max_active_power'] = max_active_power
        sienna_loads['max_reactive_power'] = max_active_power * 0.3  # Assume 0.3 power factor
        sienna_loads['available'] = True
        sienna_loads['status'] = 1
        
        # Export to CSV
        load_file = output_path / 'load.csv'
        sienna_loads.to_csv(load_file, index=False)
        
        logger.info(f"Exported {len(sienna_loads)} loads to {load_file}")
        return {'load': str(load_file)}
    
    def _export_branches(self, output_path: Path) -> Dict[str, str]:
        """
        Export transmission lines and transformers in PowerSystems.jl format.
        
        Combines PyPSA lines and transformers into a single branch.csv file
        with proper electrical parameters and ratings.
        
        Args:
            output_path (Path): Directory where branch.csv will be created
            
        Returns:
            Dict[str, str]: Mapping with 'branch' key pointing to created file path
        """
        branches = []
        
        # Add lines
        if hasattr(self.network, 'lines') and not self.network.lines.empty:
            lines_df = self.network.lines.copy()
            lines_df['component_type'] = 'Line'
            branches.append(lines_df)
        
        # Add transformers  
        if hasattr(self.network, 'transformers') and not self.network.transformers.empty:
            transformers_df = self.network.transformers.copy()
            transformers_df['component_type'] = 'Transformer'
            branches.append(transformers_df)
        
        if not branches:
            logger.info("No branches to export")
            return {}
        
        # Combine all branches
        all_branches = pd.concat(branches, ignore_index=False)
        
        # Create PowerSystems.jl branch format
        sienna_branches = pd.DataFrame(index=all_branches.index)
        sienna_branches['name'] = all_branches.index
        sienna_branches['connection_points_from'] = all_branches['bus0']
        sienna_branches['connection_points_to'] = all_branches['bus1']
        sienna_branches['r'] = all_branches['r']  # p.u. resistance
        sienna_branches['x'] = all_branches['x']  # p.u. reactance
        sienna_branches['b'] = all_branches.get('b', 0.0)  # p.u. susceptance
        sienna_branches['rate'] = all_branches['s_nom']  # MVA rating
        sienna_branches['tap'] = all_branches.get('tap_ratio', 1.0)  # Transformer tap ratio
        sienna_branches['shift'] = 0.0  # Phase shift angle (radians)
        sienna_branches['available'] = True
        sienna_branches['status'] = 1
        
        # Export to CSV
        branch_file = output_path / 'branch.csv'
        sienna_branches.to_csv(branch_file, index=False)
        
        logger.info(f"Exported {len(sienna_branches)} branches to {branch_file}")
        return {'branch': str(branch_file)}
    
    def _export_dc_branches(self, output_path: Path) -> Dict[str, str]:
        """
        Export DC transmission links in PowerSystems.jl format.
        
        Converts PyPSA links to PowerSystems.jl DC branch representation
        with power limits and loss characteristics.
        
        Args:
            output_path (Path): Directory where dc_branch.csv will be created
            
        Returns:
            Dict[str, str]: Mapping with 'dc_branch' key pointing to created file path
        """
        links_df = self.network.links.copy()
        
        if links_df.empty:
            logger.info("No DC branches to export")
            return {}
        
        sienna_dc = pd.DataFrame(index=links_df.index)
        sienna_dc['name'] = links_df.index
        sienna_dc['connection_points_from'] = links_df['bus0']
        sienna_dc['connection_points_to'] = links_df['bus1']
        sienna_dc['active_power_limits_from'] = links_df['p_nom']  # MW
        sienna_dc['active_power_limits_to'] = links_df['p_nom']    # MW
        sienna_dc['loss'] = 1.0 - links_df.get('efficiency', 1.0)  # Loss factor
        sienna_dc['available'] = True
        sienna_dc['status'] = 1
        
        # Export to CSV
        dc_file = output_path / 'dc_branch.csv'
        sienna_dc.to_csv(dc_file, index=False)
        
        logger.info(f"Exported {len(sienna_dc)} DC branches to {dc_file}")
        return {'dc_branch': str(dc_file)}
    
    def _export_storage(self, output_path: Path) -> Dict[str, str]:
        """
        Export energy storage systems in PowerSystems.jl format.
        
        Converts PyPSA storage units to PowerSystems.jl storage representation
        with energy capacity, power limits, and efficiency parameters.
        
        Args:
            output_path (Path): Directory where storage.csv will be created
            
        Returns:
            Dict[str, str]: Mapping with 'storage' key pointing to created file path
        """
        storage_df = self.network.storage_units.copy()
        
        if storage_df.empty:
            logger.info("No storage units to export")
            return {}
        
        sienna_storage = pd.DataFrame(index=storage_df.index)
        sienna_storage['name'] = storage_df.index
        sienna_storage['bus'] = storage_df['bus']
        
        # Calculate energy capacity from power rating and duration
        max_hours = storage_df.get('max_hours', 6.0)
        energy_capacity = storage_df['p_nom'] * max_hours
        sienna_storage['energy_capacity'] = energy_capacity  # MWh
        
        sienna_storage['input_active_power_limits'] = storage_df['p_nom']   # MW charging
        sienna_storage['output_active_power_limits'] = storage_df['p_nom']  # MW discharging
        sienna_storage['efficiency_in'] = storage_df.get('efficiency_store', 0.95)
        sienna_storage['efficiency_out'] = storage_df.get('efficiency_dispatch', 0.95)
        
        # Initial energy state
        initial_soc = storage_df.get('state_of_charge_initial', 0.5)
        sienna_storage['initial_energy'] = initial_soc * energy_capacity
        
        sienna_storage['available'] = True
        sienna_storage['status'] = 1
        
        # Export to CSV
        storage_file = output_path / 'storage.csv'
        sienna_storage.to_csv(storage_file, index=False)
        
        logger.info(f"Exported {len(sienna_storage)} storage units to {storage_file}")
        return {'storage': str(storage_file)}
    
    def _export_time_series_data(self, output_path: Path) -> Dict[str, str]:
        """
        Export time series data in PowerSystems.jl 4.6.2 format.
        
        Creates time series CSV files and associated metadata for load profiles,
        renewable availability factors, and other time-varying parameters.
        
        Args:
            output_path (Path): Directory where time series files will be created
            
        Returns:
            Dict[str, str]: Mapping of time series types to created file paths
        """
        if 'time_series' not in self.network_summary:
            logger.info("No time series data found")
            return {}
        
        ts_dir = output_path / 'timeseries_data'
        ts_dir.mkdir(exist_ok=True)
        
        files_created = {}
        
        # Export load time series
        if 'Load' in self.network_summary.get('time_series', {}):
            load_ts_files = self._export_load_time_series(ts_dir)
            files_created.update(load_ts_files)
        
        # Export renewable availability time series
        if 'Generator' in self.network_summary.get('time_series', {}):
            gen_ts_files = self._export_generator_time_series(ts_dir)
            files_created.update(gen_ts_files)
        
        return files_created
    
    def _export_load_time_series(self, ts_dir: Path) -> Dict[str, str]:
        """
        Export load time series data in PowerSystems.jl format.
        
        Creates CSV file with hourly load profiles and adds appropriate
        metadata entries for PowerSystems.jl time series linking.
        
        Args:
            ts_dir (Path): Directory where load time series file will be created
            
        Returns:
            Dict[str, str]: Mapping with 'load_timeseries' key pointing to created file path
        """
        if not hasattr(self.network, 'loads_t') or not hasattr(self.network.loads_t, 'p_set'):
            return {}
        
        load_ts = self.network.loads_t.p_set
        if load_ts.empty:
            return {}
        
        # Convert to PowerSystems.jl time series format
        sienna_load_ts = load_ts.copy()
        
        # Ensure datetime index is in ISO format
        if hasattr(sienna_load_ts.index, 'strftime'):
            sienna_load_ts.index = sienna_load_ts.index.strftime('%Y-%m-%dT%H:%M:%S')
        sienna_load_ts.index.name = 'DateTime'
        
        # Export to CSV
        load_ts_file = ts_dir / 'load_timeseries.csv'
        sienna_load_ts.to_csv(load_ts_file)
        
        # Add to time series metadata with correct format
        for load_name in sienna_load_ts.columns:
            self.time_series_metadata.append({
                'simulation': 'DA',  # Day-ahead
                'category': 'ElectricLoad',
                'component': load_name,
                'label': 'max_active_power',
                'data_file': 'timeseries_data/load_timeseries.csv',
                'data_column': load_name,
                'scaling_factor_multiplier': 'get_max_active_power',
                'normalization_factor': 1.0
            })
        
        logger.info(f"Exported load time series ({len(sienna_load_ts)} time steps, {len(sienna_load_ts.columns)} loads)")
        return {'load_timeseries': str(load_ts_file)}
    
    def _export_generator_time_series(self, ts_dir: Path) -> Dict[str, str]:
        """
        Export generator time series data in PowerSystems.jl format.
        
        Creates CSV files for renewable availability factors and other
        generator time-varying parameters.
        
        Args:
            ts_dir (Path): Directory where generator time series files will be created
            
        Returns:
            Dict[str, str]: Mapping of generator time series types to created file paths
        """
        files_created = {}
        
        # Export renewable availability (p_max_pu)
        if hasattr(self.network, 'generators_t') and hasattr(self.network.generators_t, 'p_max_pu'):
            gen_availability = self.network.generators_t.p_max_pu
            if not gen_availability.empty:
                renewable_file = self._export_renewable_availability(ts_dir, gen_availability)
                files_created.update(renewable_file)
        
        return files_created
    
    def _export_renewable_availability(self, ts_dir: Path, gen_availability: pd.DataFrame) -> Dict[str, str]:
        """
        Export renewable generator availability factors.
        
        Filters for renewable generators and exports their capacity factors
        with proper PowerSystems.jl time series metadata.
        
        Args:
            ts_dir (Path): Directory where renewable availability file will be created
            gen_availability (pd.DataFrame): Generator availability time series data
            
        Returns:
            Dict[str, str]: Mapping with 'renewable_availability' key pointing to created file path
        """
        # Filter for renewable generators
        renewable_carriers = ['wind', 'solar', 'pv', 'onshore', 'offshore', 'hydro', 'ror', 'biomass']
        renewable_gens = []
        
        for gen_name in gen_availability.columns:
            if gen_name in self.network.generators.index:
                gen_carrier = self.network.generators.loc[gen_name, 'carrier']
                if any(carrier in gen_carrier.lower() for carrier in renewable_carriers):
                    renewable_gens.append(gen_name)
        
        if not renewable_gens:
            logger.info("No renewable generators found for availability export")
            return {}
        
        renewable_availability = gen_availability[renewable_gens].copy()
        
        # Ensure datetime formatting is ISO compliant
        if hasattr(renewable_availability.index, 'strftime'):
            renewable_availability.index = renewable_availability.index.strftime('%Y-%m-%dT%H:%M:%S')
        renewable_availability.index.name = 'DateTime'
        
        # Export to CSV
        gen_ts_file = ts_dir / 'renewable_availability.csv'
        renewable_availability.to_csv(gen_ts_file)
        
        # Add to time series metadata with correct PowerSystems.jl format
        for gen_name in renewable_availability.columns:
            self.time_series_metadata.append({
                'simulation': 'DA',  # Day-ahead
                'category': 'RenewableGen',
                'component': gen_name,
                'label': 'max_active_power',
                'data_file': 'timeseries_data/renewable_availability.csv',
                'data_column': gen_name,
                'scaling_factor_multiplier': 'get_max_active_power',
                'normalization_factor': 1.0
            })
        
        logger.info(f"Exported renewable availability ({len(renewable_availability)} time steps, {len(renewable_gens)} generators)")
        return {'renewable_availability': str(gen_ts_file)}
    
    def _create_configuration_files(self, output_path: Path) -> Dict[str, str]:
        """
        Create PowerSystems.jl 4.6.2 compatible configuration files.
        
        Generates the YAML and JSON configuration files required by PowerSystems.jl
        for proper data parsing and component type mapping.
        
        Args:
            output_path (Path): Directory where configuration files will be created
            
        Returns:
            Dict[str, str]: Mapping of configuration file types to created file paths
        """
        files_created = {}
        
        # Create user_descriptors.yaml
        descriptors_file = self._create_user_descriptors(output_path)
        files_created['user_descriptors'] = str(descriptors_file)
        
        # Create timeseries_metadata.json
        if self.time_series_metadata:
            ts_metadata_file = self._create_timeseries_metadata(output_path)
            files_created['timeseries_metadata'] = str(ts_metadata_file)
        
        # Create generator_mapping.yaml
        gen_mapping_file = self._create_generator_mapping(output_path)
        files_created['generator_mapping'] = str(gen_mapping_file)
        
        return files_created
        
    def _create_user_descriptors(self, output_path: Path) -> Path:
        """
        Create user_descriptors.yaml in PowerSystems.jl 4.6.2 format.

        Returns:
            Path: Path to the created user_descriptors.yaml file
        """

        def _flatten_fields(field_dict):
            return [{k: v['name']} for k, v in field_dict.items()]

        user_descriptors_raw = {
            'bus': {
                'fields': {
                    'name': {'name': 'name'},
                    'base_voltage': {'name': 'base_voltage'},
                    'bus_type': {'name': 'bus_type'},
                    'area': {'name': 'area'},
                    'zone': {'name': 'zone'},
                    'longitude': {'name': 'longitude'},
                    'latitude': {'name': 'latitude'},
                    'voltage': {'name': 'voltage'},
                    'angle': {'name': 'angle'}
                }
            },
            'gen': {
                'fields': {
                    'name': {'name': 'name'},
                    'bus': {'name': 'bus'},
                    'fuel': {'name': 'fuel'},
                    'type': {'name': 'type'},
                    'active_power': {'name': 'active_power'},
                    'max_active_power': {'name': 'max_active_power'},
                    'min_active_power': {'name': 'min_active_power'},
                    'max_reactive_power': {'name': 'max_reactive_power'},
                    'min_reactive_power': {'name': 'min_reactive_power'},
                    'variable': {'name': 'variable'},
                    'startup': {'name': 'startup'},
                    'shutdown': {'name': 'shutdown'},
                    'ramp_30': {'name': 'ramp_30'},
                    'ramp_10': {'name': 'ramp_10'},
                    'min_up_time': {'name': 'min_up_time'},
                    'min_down_time': {'name': 'min_down_time'},
                    'available': {'name': 'available'},
                    'status': {'name': 'status'}
                }
            },
            'load': {
                'fields': {
                    'name': {'name': 'name'},
                    'bus': {'name': 'bus'},
                    'max_active_power': {'name': 'max_active_power'},
                    'max_reactive_power': {'name': 'max_reactive_power'},
                    'available': {'name': 'available'},
                    'status': {'name': 'status'}
                }
            },
            'branch': {
                'fields': {
                    'name': {'name': 'name'},
                    'connection_points_from': {'name': 'connection_points_from'},
                    'connection_points_to': {'name': 'connection_points_to'},
                    'r': {'name': 'r'},
                    'x': {'name': 'x'},
                    'b': {'name': 'b'},
                    'rate': {'name': 'rate'},
                    'tap': {'name': 'tap'},
                    'shift': {'name': 'shift'},
                    'available': {'name': 'available'},
                    'status': {'name': 'status'}
                }
            },
            'dc_branch': {
                'fields': {
                    'name': {'name': 'name'},
                    'connection_points_from': {'name': 'connection_points_from'},
                    'connection_points_to': {'name': 'connection_points_to'},
                    'active_power_limits_from': {'name': 'active_power_limits_from'},
                    'active_power_limits_to': {'name': 'active_power_limits_to'},
                    'loss': {'name': 'loss'},
                    'available': {'name': 'available'},
                    'status': {'name': 'status'}
                }
            },
            'storage': {
                'fields': {
                    'name': {'name': 'name'},
                    'bus': {'name': 'bus'},
                    'energy_capacity': {'name': 'energy_capacity'},
                    'input_active_power_limits': {'name': 'input_active_power_limits'},
                    'output_active_power_limits': {'name': 'output_active_power_limits'},
                    'efficiency_in': {'name': 'efficiency_in'},
                    'efficiency_out': {'name': 'efficiency_out'},
                    'initial_energy': {'name': 'initial_energy'},
                    'available': {'name': 'available'},
                    'status': {'name': 'status'}
                }
            }
        }

        # Flatten fields to match PowerSystems.jl expectations
        user_descriptors = {
            section: {"fields": _flatten_fields(section_data["fields"])}
            for section, section_data in user_descriptors_raw.items()
        }

        descriptors_file = output_path / 'user_descriptors.yaml'
        with open(descriptors_file, 'w') as f:
            yaml.dump(user_descriptors, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Created user descriptors: {descriptors_file}")
        return descriptors_file

    
    def _create_timeseries_metadata(self, output_path: Path) -> Path:
        """
        Create timeseries_metadata.json for PowerSystems.jl time series linking.
        
        Generates the metadata file that links time series CSV files to
        specific components and their time-varying attributes.
        
        Args:
            output_path (Path): Directory where timeseries_metadata.json will be created
            
        Returns:
            Path: Path to the created timeseries_metadata.json file
        """
        
        ts_metadata_file = output_path / 'timeseries_metadata.json'
        with open(ts_metadata_file, 'w') as f:
            json.dump(self.time_series_metadata, f, indent=2)
        
        logger.info(f"Created time series metadata: {ts_metadata_file}")
        return ts_metadata_file
    
    def _create_generator_mapping(self, output_path: Path) -> Path:
        """
        Create generator_mapping.yaml for PowerSystems.jl generator classification.
        
        Defines how fuel and prime mover type combinations map to specific
        PowerSystems.jl generator types (ThermalStandard, RenewableDispatch, etc.).
        
        Args:
            output_path (Path): Directory where generator_mapping.yaml will be created
            
        Returns:
            Path: Path to the created generator_mapping.yaml file
        """
        
        # PowerSystems.jl generator mapping format
        generator_mapping = {
            'ThermalStandard': {
                'fuel': ['COAL', 'NATURAL_GAS', 'DIESEL', 'NUCLEAR', 'BIOMASS'],
                'type': ['ST', 'CC', 'CT', 'IC']
            },
            'RenewableDispatch': {
                'fuel': ['WIND', 'SOLAR', 'HYDRO', 'GEOTHERMAL'],
                'type': ['WT', 'PV', 'HY']
            }
        }
        
        mapping_file = output_path / 'generator_mapping.yaml'
        with open(mapping_file, 'w') as f:
            yaml.dump(generator_mapping, f, default_flow_style=False)
        
        logger.info(f"Created generator mapping: {mapping_file}")
        return mapping_file
    
    def _create_julia_import_script(self, output_path: Path) -> Path:
        """
        Create Julia script for importing data into PowerSystems.jl 4.6.2.
        
        Generates a complete Julia script that demonstrates how to load the
        exported CSV data into PowerSystems.jl and run a basic economic dispatch.
        
        Args:
            output_path (Path): Directory where Julia script will be created
            
        Returns:
            Path: Path to the created Julia import script
        """
        
        julia_code = f'''#!/usr/bin/env julia

"""
Julia script to import PyPSA-exported data into PowerSystems.jl 4.6.2

This script demonstrates how to load the exported CSV data into PowerSystems.jl
and perform basic system validation and economic dispatch modeling.

Usage:
    julia import_to_powersystems.jl

Requirements:
    - PowerSystems.jl v4.6.2+
    - PowerSimulations.jl
    - HiGHS.jl (or other compatible solver)
"""

using PowerSystems
using PowerSimulations
using Dates
using TimeSeries
using HiGHS

# Configuration
data_dir = "{output_path.absolute()}"
base_power = {self.base_power}  # MVA
user_descriptors = joinpath(data_dir, "user_descriptors.yaml")

# Optional files (check if they exist)
timeseries_metadata_file = joinpath(data_dir, "timeseries_metadata.json")
generator_mapping_file = joinpath(data_dir, "generator_mapping.yaml")

println("🔧 PowerSystems.jl 4.6.2 Data Import")
println("Loading PyPSA data from: ", data_dir)
println("Files in directory:")
for file in readdir(data_dir)
    println("  ✓ ", file)
end

# Create PowerSystemTableData with proper PowerSystems.jl 4.6.2 API
try
    println("\\n📊 Creating PowerSystemTableData...")
    
    # Check which optional files exist
    has_timeseries = isfile(timeseries_metadata_file)
    has_gen_mapping = isfile(generator_mapping_file)
    
    if has_timeseries && has_gen_mapping
        println("   Using full configuration with time series and generator mapping")
        data = PowerSystemTableData(
            data_dir,
            base_power,
            user_descriptors;
            timeseries_metadata_file = timeseries_metadata_file,
            generator_mapping_file = generator_mapping_file
        )
    elseif has_timeseries
        println("   Using time series metadata only")
        data = PowerSystemTableData(
            data_dir,
            base_power, 
            user_descriptors;
            timeseries_metadata_file = timeseries_metadata_file
        )
    elseif has_gen_mapping
        println("   Using generator mapping only")
        data = PowerSystemTableData(
            data_dir,
            base_power,
            user_descriptors;
            generator_mapping_file = generator_mapping_file
        )
    else
        println("   Using basic configuration without optional files")
        data = PowerSystemTableData(
            data_dir,
            base_power,
            user_descriptors
        )
    end
    
    println("✅ PowerSystemTableData created successfully")
    
    # Create System with proper PowerSystems.jl 4.6.2 API
    println("\\n🏗️  Creating PowerSystems.jl System...")
    sys = System(data; time_series_in_memory = true)
    println("✅ PowerSystems.jl System created successfully")
    
    # Print detailed system summary using PowerSystems.jl 4.6.2 API
    println("\\n📋 === System Summary ===")
    println("Base Power: ", get_base_power(sys), " MVA")
    println("Buses: ", length(get_components(Bus, sys)))
    
    # Count generators by type
    thermal_gens = get_components(ThermalStandard, sys)
    renewable_gens = get_components(RenewableDispatch, sys)
    
    println("Thermal Generators: ", length(thermal_gens))
    if length(thermal_gens) > 0
        sample_thermal = first(thermal_gens)
        println("  📋 Sample thermal: ", get_name(sample_thermal))
        println("     Max power: ", get_max_active_power(sample_thermal), " MW")
        println("     Fuel: ", get_fuel(sample_thermal))
        println("     Prime mover: ", get_prime_mover(sample_thermal))
    end
    
    println("Renewable Generators: ", length(renewable_gens))
    if length(renewable_gens) > 0
        sample_renewable = first(renewable_gens)
        println("  🌱 Sample renewable: ", get_name(sample_renewable))
        println("     Max power: ", get_max_active_power(sample_renewable), " MW")
        println("     Prime mover: ", get_prime_mover(sample_renewable))
    end
    
    println("Loads: ", length(get_components(ElectricLoad, sys)))
    println("Branches: ", length(get_components(ACBranch, sys)))
    println("Storage: ", length(get_components(GenericBattery, sys)))
    
    # Check time series data using PowerSystems.jl 4.6.2 API
    ts_count = length(get_time_series_multiple(sys))
    if ts_count > 0
        println("Time series: ", ts_count, " time series found")
        
        # Get time series summary
        first_ts = first(get_time_series_multiple(sys))
        ts_component = get_component(sys, first_ts[1])
        ts_data = first_ts[2]
        
        println("  📈 Sample TS component: ", get_name(ts_component))
        println("     TS type: ", typeof(ts_data))
        println("     TS length: ", length(ts_data))
    else
        println("Time series: No time series data found")
    end
    
    # System validation using PowerSystems.jl 4.6.2 API
    println("\\n🔍 === System Validation ===")
    try
        total_demand = sum(get_max_active_power(load) for load in get_components(ElectricLoad, sys))
        total_generation = sum(get_max_active_power(gen) for gen in get_components(Generator, sys))
        
        println("Total demand: ", round(total_demand, digits=1), " MW")
        println("Total generation capacity: ", round(total_generation, digits=1), " MW")
        
        if total_generation > 0
            reserve_margin = (total_generation - total_demand) / total_demand * 100
            println("Reserve margin: ", round(reserve_margin, digits=1), "%")
            
            if total_generation >= total_demand
                println("✅ System has adequate generation capacity")
            else
                println("⚠️  Warning: Generation capacity may be insufficient")
            end
        end
        
    catch e
        println("⚠️  Could not validate system: ", e)
    end
    
    # Save system for later use
    sys_file = joinpath(data_dir, "pypsa_system.json")
    to_json(sys, sys_file)
    println("\\n💾 System saved to: ", sys_file)
    
    # Example Economic Dispatch using PowerSystems.jl 4.6.2 API
    if ts_count > 0
        println("\\n⚡ === Running Example Economic Dispatch ===")
        
        try
            # Set up Economic Dispatch problem with PowerSystems.jl 4.6.2
            template = ProblemTemplate(NetworkModel(DCPPowerModel))
            
            # Add device models
            set_device_model!(template, ThermalStandard, ThermalStandardUnitCommitment)
            set_device_model!(template, RenewableDispatch, RenewableFullDispatch)
            set_device_model!(template, PowerLoad, StaticPowerLoad)
            set_device_model!(template, Line, StaticBranch)
            
            # Create Decision Model
            decision_model = DecisionModel(
                template,
                sys;
                name = "PyPSA_ED",
                optimizer = HiGHS.Optimizer,
                store_variable_names = true
            )
            
            println("✅ Economic Dispatch model created")
            
            # Build and solve
            build!(decision_model)
            solve!(decision_model)
            
            if get_optimizer_stats(decision_model).termination_status == MOI.OPTIMAL
                println("✅ Economic Dispatch solved successfully")
                
                # Get and save results
                results = ProblemResults(decision_model)
                results_dir = joinpath(data_dir, "dispatch_results")
                mkpath(results_dir)
                
                # Export key results
                write_results(results, results_dir)
                
                println("✅ Results exported to: ", results_dir)
                
                # Show basic results summary
                obj_value = get_objective_value(decision_model)
                println("💰 Total cost: ", round(obj_value, digits=2), " currency units")
                
            else
                status = get_optimizer_stats(decision_model).termination_status
                println("❌ Economic Dispatch failed with status: ", status)
            end
            
        catch e
            println("❌ Economic Dispatch failed: ", e)
            println("Stack trace:")
            showerror(stdout, e, catch_backtrace())
        end
    end
    
    println("\\n🎉 === Import Complete ===")
    return sys
    
catch e
    println("❌ Error creating PowerSystems.jl system: ", e)
    println("Stack trace:")
    showerror(stdout, e, catch_backtrace())
    rethrow(e)
end
'''
        
        julia_file = output_path / 'import_to_powersystems.jl'
        with open(julia_file, 'w') as f:
            f.write(julia_code)
        
        # Make executable
        julia_file.chmod(0o755)
        
        logger.info(f"Created Julia import script: {julia_file}")
        return julia_file


def export_pypsa_to_sienna(network: pypsa.Network, 
                          scenario_setup: dict,
                          output_dir: str,
                          include_time_series: bool = True) -> Dict[str, str]:
    """
    Export PyPSA network to PowerSystems.jl 4.6.2 compatible format.
    
    This is the main entry point for converting PyPSA dispatch network data
    to the CSV files and configuration required by PowerSystems.jl.
    
    Args:
        network (pypsa.Network): The solved PyPSA dispatch network to export
        scenario_setup (dict): Scenario configuration from load_scenario_definition()
        output_dir (str): Directory path where exported files will be created
        include_time_series (bool): Whether to export time series data files.
            Defaults to True. Set to False for static network export only.
            
    Returns:
        Dict[str, str]: Comprehensive results dictionary containing:
            - File paths for all created CSV and configuration files
            - Export summary with component counts and validation status
            - Import instructions for using the exported data in Julia
            
    Example:
        ```python
        export_results = export_pypsa_to_sienna(
            network=solved_pypsa_network,
            scenario_setup=scenario_config,
            output_dir="/path/to/sienna_export",
            include_time_series=True
        )
        print(f"Created {len(export_results)} files")
        ```
    """
    
    # Validate network before export
    validation = validate_pypsa_network(network)
    if not validation['is_valid']:
        logger.warning("Network validation issues found:")
        for issue in validation['issues']:
            logger.warning(f"  - {issue}")
    
    # Create exporter and perform export
    exporter = PyPSAToSiennaExporter(network, scenario_setup)
    export_results = exporter.export_to_csv(output_dir, include_time_series)
    
    # Add comprehensive summary
    export_results['export_summary'] = {
        'network_components': exporter.network_summary,
        'generator_details_loaded': bool(exporter.generator_details),
        'time_series_exported': include_time_series and len(exporter.time_series_metadata) > 0,
        'validation_status': validation,
        'powersystems_compatibility': 'PowerSystems.jl 4.6.2+'
    }
    
    # Add import instructions
    export_results['import_instructions'] = [
        "PowerSystems.jl 4.6.2 Import Instructions:",
        "1. Navigate to the export directory",
        "2. Ensure Julia has required packages installed:",
        "   julia> using Pkg; Pkg.add([\"PowerSystems\", \"PowerSimulations\", \"HiGHS\"])",
        "3. Run the generated import script:",
        "   julia import_to_powersystems.jl",
        "4. Or import manually in Julia:",
        "   using PowerSystems",
        f"   data = PowerSystemTableData(\"{output_dir}\", 100.0, \"user_descriptors.yaml\")",
        "   sys = System(data; time_series_in_memory=true)"
    ]
    
    return export_results

def validate_pypsa_network(network) -> Dict[str, Any]:
    """
    Validate PyPSA network for compatibility with Sienna export.
    
    Performs comprehensive checks to ensure the network contains all
    required data and references are consistent before attempting export.
    
    UPDATED: Includes South African specific carriers in known types.
    
    Args:
        network (pypsa.Network): The PyPSA network to validate
        
    Returns:
        Dict[str, Any]: Validation results containing:
            - 'is_valid' (bool): True if network passes all checks
            - 'issues' (List[str]): List of validation issues found
            
    Example:
        ```python
        validation = validate_pypsa_network(network)
        if not validation['is_valid']:
            for issue in validation['issues']:
                print(f"Warning: {issue}")
        ```
    """
    issues = []
    
    # Check for required components
    if network.buses.empty:
        issues.append("No buses found in network")
    
    if network.generators.empty and network.loads.empty:
        issues.append("No generators or loads found in network")
    
    # Check bus references consistency
    all_buses = set(network.buses.index)
    
    if not network.generators.empty:
        gen_buses = set(network.generators['bus'])
        missing_gen_buses = gen_buses - all_buses
        if missing_gen_buses:
            issues.append(f"Generators reference missing buses: {missing_gen_buses}")
    
    if not network.loads.empty:
        load_buses = set(network.loads['bus'])
        missing_load_buses = load_buses - all_buses
        if missing_load_buses:
            issues.append(f"Loads reference missing buses: {missing_load_buses}")
    
    # Check for required data fields
    required_gen_fields = ['bus', 'p_nom', 'carrier']
    if not network.generators.empty:
        missing_gen_fields = set(required_gen_fields) - set(network.generators.columns)
        if missing_gen_fields:
            issues.append(f"Generators missing required fields: {missing_gen_fields}")
    
    required_bus_fields = ['v_nom']
    missing_bus_fields = set(required_bus_fields) - set(network.buses.columns)
    if missing_bus_fields:
        issues.append(f"Buses missing required fields: {missing_bus_fields}")
    
    # Check generator carriers for known mappings - UPDATED with South African carriers
    if not network.generators.empty:
        unknown_carriers = []
        known_fuel_types = {
            'gas', 'ccgt', 'ccgt_gas', 'ocgt', 'ocgt_gas', 'ocgt_diesel',
            'coal', 'lignite', 'oil', 'nuclear', 'biomass', 'waste',
            'wind', 'wind_onshore', 'wind_offshore', 'solar', 'solar_pv', 'pv',
            'hydro', 'ror', 'geothermal',
            # South African specific carriers
            'rmippp', 'bioenergy'
        }
        
        for carrier in network.generators['carrier'].unique():
            if not any(known in carrier.lower() for known in known_fuel_types):
                unknown_carriers.append(carrier)
        
        if unknown_carriers:
            issues.append(f"Unknown generator carriers (may need mapping): {unknown_carriers}")
    
    # Check for time series data consistency if present
    if hasattr(network, 'generators_t') and hasattr(network.generators_t, 'p_max_pu'):
        p_max_pu = network.generators_t.p_max_pu
        if not p_max_pu.empty:
            # Check for generators in time series that don't exist in static data
            missing_gens = set(p_max_pu.columns) - set(network.generators.index)
            if missing_gens:
                issues.append(f"Time series references missing generators: {list(missing_gens)[:5]}...")
    
    is_valid = len(issues) == 0
    return {'is_valid': is_valid, 'issues': issues}

if __name__ == "__main__":
    """
    Example usage and integration with PyPSA-ZA workflow.
    
    This section demonstrates how to use the exporter within the PyPSA-ZA
    snakemake workflow or as a standalone script.
    """
    logging.basicConfig(level=logging.INFO)
    
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            'export_to_sienna', 
            **{
                'scenario':'TEST',
                'year':2030
            }
        )

    # Load network and scenario configuration
    n = pypsa.Network(snakemake.input.dispatch_network)
    scenario_setup = load_scenario_definition(snakemake)
    export_folder = snakemake.output.sienna_export_dir

    logger.info(f"Starting export to PowerSystems.jl 4.6.2 format...")
    logger.info(f"Network: {len(n.buses)} buses, {len(n.generators)} generators, {len(n.loads)} loads")
    
    # Perform export with comprehensive data
    export_results = export_pypsa_to_sienna(
        network=n,
        scenario_setup=scenario_setup,
        output_dir=export_folder,
        include_time_series=True
    )
    
    # Display detailed results
    logger.info("=== Export Results ===")
    logger.info("Files created:")
    for file_type, file_path in export_results.items():
        if file_type not in ['import_instructions', 'export_summary']:
            logger.info(f"  ✓ {file_type}: {file_path}")
    
    if 'export_summary' in export_results:
        summary = export_results['export_summary']
        logger.info(f"Network components exported: {summary['network_components']}")
        logger.info(f"Generator details loaded: {summary['generator_details_loaded']}")
        logger.info(f"Time series exported: {summary['time_series_exported']}")
        logger.info(f"PowerSystems.jl compatibility: {summary['powersystems_compatibility']}")
    
    logger.info("\nTo import in Julia:")
    for instruction in export_results['import_instructions']:
        logger.info(f"  {instruction}")
    
    logger.info(f"\nExport completed successfully to {export_folder}")