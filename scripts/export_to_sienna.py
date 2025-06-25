# -*- coding: utf-8 -*-
"""
Complete PyPSA to Sienna CSV Export Implementation
Updated for PowerSystems.jl v4.0+ and PowerSimulations.jl

This implementation exports PyPSA dispatch networks to PowerSystems.jl-compatible
CSV format using PowerSystemTableData parsing.
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
    Complete exporter for PyPSA dispatch networks to Sienna-compatible CSV format.
    
    Supports all PowerSystems.jl component types and time series data.
    """
    
    def __init__(self, network: pypsa.Network, scenario_setup: dict):
        """
        Initialize exporter with PyPSA network and scenario configuration.
        
        Parameters:
        -----------
        network : pypsa.Network
            The solved PyPSA dispatch network
        scenario_setup : dict
            Scenario configuration from load_scenario_definition()
        """
        self.network = network
        self.scenario_setup = scenario_setup
        self.base_power = 100.0  # MVA base for PowerSystems.jl (standard)
        
        # Time series storage
        self.time_series_data = {}
        self.time_series_metadata = []
        
        # Component mapping to Sienna types
        self._initialize_component_mapping()
        
        # Analyze network
        self._analyze_network()
    
    def _initialize_component_mapping(self):
        """Initialize PyPSA to PowerSystems.jl component mapping."""
        self.component_mapping = {
            'Bus': {
                'csv_name': 'bus.csv',
                'sienna_type': 'Bus',
                'required_fields': ['name', 'base_voltage'],
                'field_mapping': {
                    'name': 'name',
                    'v_nom': 'base_voltage',  # kV -> kV
                    'x': 'longitude',
                    'y': 'latitude',
                    'carrier': 'area'  # Map carrier to area for grouping
                }
            },
            
            'Generator': {
                'csv_name': 'generator.csv', 
                'sienna_type': 'ThermalStandard',  # Default, will split by carrier
                'required_fields': ['name', 'bus', 'max_active_power'],
                'field_mapping': {
                    'name': 'name',
                    'bus': 'bus',
                    'p_nom': 'max_active_power',  # MW
                    'p_min_pu': 'min_active_power_factor',  # fraction of p_nom
                    'marginal_cost': 'variable_cost',  # €/MWh
                    'carrier': 'fuel',
                    'efficiency': 'efficiency'
                }
            },
            
            'Load': {
                'csv_name': 'load.csv',
                'sienna_type': 'PowerLoad', 
                'required_fields': ['name', 'bus', 'max_active_power'],
                'field_mapping': {
                    'name': 'name',
                    'bus': 'bus', 
                    'p_set': 'max_active_power'  # MW (will use time series)
                }
            },
            
            'Line': {
                'csv_name': 'branch.csv',
                'sienna_type': 'Line',
                'required_fields': ['name', 'connection_points_from', 'connection_points_to', 'r', 'x'],
                'field_mapping': {
                    'name': 'name',
                    'bus0': 'connection_points_from',
                    'bus1': 'connection_points_to', 
                    'r': 'r',  # p.u.
                    'x': 'x',  # p.u.
                    'b': 'b',  # p.u.
                    's_nom': 'rate'  # MVA
                }
            },
            
            'Transformer': {
                'csv_name': 'branch.csv',  # Combined with lines
                'sienna_type': 'Transformer2W',
                'required_fields': ['name', 'connection_points_from', 'connection_points_to', 'r', 'x'],
                'field_mapping': {
                    'name': 'name',
                    'bus0': 'connection_points_from',
                    'bus1': 'connection_points_to',
                    'r': 'r',  # p.u.
                    'x': 'x',  # p.u. 
                    'tap_ratio': 'tap',
                    's_nom': 'rate'  # MVA
                }
            },
            
            'Link': {
                'csv_name': 'dc_branch.csv',
                'sienna_type': 'TwoTerminalHVDCLine',
                'required_fields': ['name', 'connection_points_from', 'connection_points_to'],
                'field_mapping': {
                    'name': 'name',
                    'bus0': 'connection_points_from',
                    'bus1': 'connection_points_to',
                    'p_nom': 'active_power_limits_from',
                    'efficiency': 'loss'
                }
            },
            
            'StorageUnit': {
                'csv_name': 'storage.csv',
                'sienna_type': 'GenericBattery',
                'required_fields': ['name', 'bus', 'energy_capacity', 'input_active_power_limits', 'output_active_power_limits'],
                'field_mapping': {
                    'name': 'name',
                    'bus': 'bus',
                    'p_nom': 'output_active_power_limits',
                    'p_nom': 'input_active_power_limits',  # Assume symmetric
                    'max_hours': 'energy_capacity',  # Will convert to MWh
                    'efficiency_store': 'efficiency_in',
                    'efficiency_dispatch': 'efficiency_out',
                    'state_of_charge_initial': 'initial_energy'
                }
            }
        }
    
    def _analyze_network(self):
        """Analyze the PyPSA network to understand structure and data."""
        self.network_summary = {}
        
        # Analyze static components
        for component_name in ['Bus', 'Generator', 'Load', 'Line', 'Transformer', 'Link', 'StorageUnit']:
            if hasattr(self.network, component_name.lower() + 's'):
                component_df = getattr(self.network, component_name.lower() + 's')
                if not component_df.empty:
                    self.network_summary[component_name] = {
                        'count': len(component_df),
                        'columns': list(component_df.columns)
                    }
        
        # Analyze time series data
        self.network_summary['time_series'] = {}
        for component_name in self.network_summary.keys():
            if component_name != 'time_series':
                ts_attr = getattr(self.network, component_name.lower() + 's_t', None)
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
        Main export function - converts PyPSA network to Sienna CSV format.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save CSV files and configuration
        include_time_series : bool
            Whether to export time series data
            
        Returns:
        --------
        Dict[str, str]
            Dictionary with paths to created files
        """
        logger.info("Starting PyPSA to Sienna CSV export...")
        
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
        """Export all static component data to CSV files."""
        files_created = {}
        
        # Export buses first (other components reference them)
        if 'Bus' in self.network_summary:
            bus_file = self._export_buses(output_path)
            files_created.update(bus_file)
        
        # Export generators (split by type)
        if 'Generator' in self.network_summary:
            gen_files = self._export_generators(output_path)
            files_created.update(gen_files)
        
        # Export loads
        if 'Load' in self.network_summary:
            load_files = self._export_loads(output_path)
            files_created.update(load_files)
        
        # Export branches (lines + transformers)
        branch_files = self._export_branches(output_path)
        files_created.update(branch_files)
        
        # Export DC branches (links)
        if 'Link' in self.network_summary:
            dc_files = self._export_dc_branches(output_path)
            files_created.update(dc_files)
        
        # Export storage
        if 'StorageUnit' in self.network_summary:
            storage_files = self._export_storage(output_path)
            files_created.update(storage_files)
        
        return files_created
    
    def _export_buses(self, output_path: Path) -> Dict[str, str]:
        """Export bus data to bus.csv."""
        buses_df = self.network.buses.copy()
        
        # Convert to Sienna format
        sienna_buses = pd.DataFrame()
        sienna_buses['name'] = buses_df.index
        sienna_buses['base_voltage'] = buses_df['v_nom']  # kV
        sienna_buses['bus_type'] = 'PQ'  # Default, will be updated based on components
        
        # Add geographic coordinates if available
        if 'x' in buses_df.columns and 'y' in buses_df.columns:
            sienna_buses['longitude'] = buses_df['x']
            sienna_buses['latitude'] = buses_df['y']
        
        # Set bus types based on connected components
        self._set_bus_types(sienna_buses)
        
        # Add areas/zones if available
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
        """Set bus types based on connected components."""
        # Default all to PQ
        sienna_buses['bus_type'] = 'PQ'
        
        # Set slack bus (first generator bus or first bus)
        if hasattr(self.network, 'generators') and not self.network.generators.empty:
            slack_bus = self.network.generators.iloc[0]['bus']
            sienna_buses.loc[sienna_buses['name'] == slack_bus, 'bus_type'] = 'REF'
        else:
            sienna_buses.iloc[0, sienna_buses.columns.get_loc('bus_type')] = 'REF'
    
    def _export_generators(self, output_path: Path) -> Dict[str, str]:
        """Export generators split by type (thermal vs renewable)."""
        generators_df = self.network.generators.copy()
        files_created = {}
        
        # Define renewable carriers
        renewable_carriers = ['wind', 'solar', 'pv', 'onshore', 'offshore', 'hydro', 'ror']
        
        # Split generators by type
        is_renewable = generators_df['carrier'].str.lower().str.contains('|'.join(renewable_carriers), na=False)
        thermal_gens = generators_df[~is_renewable]
        renewable_gens = generators_df[is_renewable]
        
        # Export thermal generators
        if not thermal_gens.empty:
            thermal_file = self._export_thermal_generators(output_path, thermal_gens)
            files_created.update(thermal_file)
        
        # Export renewable generators
        if not renewable_gens.empty:
            renewable_file = self._export_renewable_generators(output_path, renewable_gens)
            files_created.update(renewable_file)
        
        return files_created
    
    def _export_thermal_generators(self, output_path: Path, thermal_gens: pd.DataFrame) -> Dict[str, str]:
        """Export thermal generators to thermal_generators.csv."""
        sienna_thermal = pd.DataFrame()
        
        # Required fields
        sienna_thermal['name'] = thermal_gens.index
        sienna_thermal['bus'] = thermal_gens['bus']
        sienna_thermal['fuel'] = thermal_gens['carrier']
        sienna_thermal['max_active_power'] = thermal_gens['p_nom']  # MW
        sienna_thermal['min_active_power'] = thermal_gens['p_nom'] * thermal_gens.get('p_min_pu', 0.0)  # MW
        
        # Optional fields
        sienna_thermal['max_reactive_power'] = thermal_gens.get('q_nom', thermal_gens['p_nom'] * 0.3)  # Assume 0.3 p.f.
        sienna_thermal['min_reactive_power'] = -sienna_thermal['max_reactive_power']
        sienna_thermal['variable_cost'] = thermal_gens.get('marginal_cost', 0.0)  # €/MWh
        sienna_thermal['startup_cost'] = thermal_gens.get('start_up_cost', 0.0)  # €
        sienna_thermal['shutdown_cost'] = 0.0  # €
        
        # Operational constraints
        sienna_thermal['ramp_up'] = thermal_gens.get('ramp_limit_up', thermal_gens['p_nom'])  # MW/h
        sienna_thermal['ramp_down'] = thermal_gens.get('ramp_limit_down', thermal_gens['p_nom'])  # MW/h
        sienna_thermal['min_up_time'] = thermal_gens.get('min_up_time', 1.0)  # hours
        sienna_thermal['min_down_time'] = thermal_gens.get('min_down_time', 1.0)  # hours
        
        # Prime mover mapping
        sienna_thermal['prime_mover_type'] = thermal_gens['carrier'].map(self._get_prime_mover_mapping())
        
        # Export to CSV
        thermal_file = output_path / 'thermal_generators.csv'
        sienna_thermal.to_csv(thermal_file, index=False)
        
        logger.info(f"Exported {len(sienna_thermal)} thermal generators to {thermal_file}")
        return {'thermal_generators': str(thermal_file)}
    
    def _export_renewable_generators(self, output_path: Path, renewable_gens: pd.DataFrame) -> Dict[str, str]:
        """Export renewable generators to renewable_generators.csv."""
        sienna_renewable = pd.DataFrame()
        
        # Required fields
        sienna_renewable['name'] = renewable_gens.index
        sienna_renewable['bus'] = renewable_gens['bus']
        sienna_renewable['max_active_power'] = renewable_gens['p_nom']  # MW
        sienna_renewable['prime_mover_type'] = renewable_gens['carrier'].map(self._get_renewable_prime_mover_mapping())
        
        # Optional fields
        sienna_renewable['max_reactive_power'] = renewable_gens.get('q_nom', 0.0)  # MW
        sienna_renewable['min_reactive_power'] = 0.0  # MW
        sienna_renewable['variable_cost'] = renewable_gens.get('marginal_cost', 0.0)  # €/MWh
        
        # Export to CSV
        renewable_file = output_path / 'renewable_generators.csv'
        sienna_renewable.to_csv(renewable_file, index=False)
        
        logger.info(f"Exported {len(sienna_renewable)} renewable generators to {renewable_file}")
        return {'renewable_generators': str(renewable_file)}
    
    def _get_prime_mover_mapping(self) -> Dict[str, str]:
        """Get mapping from PyPSA carriers to PowerSystems.jl prime mover types."""
        return {
            'gas': 'CC',  # Combined Cycle
            'ccgt': 'CC',
            'ocgt': 'CT',  # Combustion Turbine
            'coal': 'ST',  # Steam Turbine
            'lignite': 'ST',
            'oil': 'CT',
            'nuclear': 'ST',
            'biomass': 'ST',
            'waste': 'ST'
        }
    
    def _get_renewable_prime_mover_mapping(self) -> Dict[str, str]:
        """Get mapping from PyPSA renewable carriers to PowerSystems.jl prime mover types."""
        return {
            'wind': 'WT',  # Wind Turbine
            'onshore': 'WT',
            'offshore': 'WT', 
            'solar': 'PV',  # Photovoltaic
            'pv': 'PV',
            'hydro': 'HY',  # Hydro
            'ror': 'HY'
        }
    
    def _export_loads(self, output_path: Path) -> Dict[str, str]:
        """Export loads to load.csv."""
        loads_df = self.network.loads.copy()
        
        sienna_loads = pd.DataFrame()
        sienna_loads['name'] = loads_df.index
        sienna_loads['bus'] = loads_df['bus']
        sienna_loads['max_active_power'] = loads_df['p_set']  # MW (base value)
        sienna_loads['max_reactive_power'] = loads_df.get('q_set', loads_df['p_set'] * 0.3)  # Assume 0.3 p.f.
        sienna_loads['power_factor'] = 0.95  # Default
        
        # Export to CSV
        load_file = output_path / 'load.csv'
        sienna_loads.to_csv(load_file, index=False)
        
        logger.info(f"Exported {len(sienna_loads)} loads to {load_file}")
        return {'load': str(load_file)}
    
    def _export_branches(self, output_path: Path) -> Dict[str, str]:
        """Export lines and transformers to branch.csv."""
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
            return {}
        
        # Combine all branches
        all_branches = pd.concat(branches, ignore_index=False)
        
        sienna_branches = pd.DataFrame()
        sienna_branches['name'] = all_branches.index
        sienna_branches['connection_points_from'] = all_branches['bus0']
        sienna_branches['connection_points_to'] = all_branches['bus1']
        sienna_branches['r'] = all_branches['r']  # p.u.
        sienna_branches['x'] = all_branches['x']  # p.u.
        sienna_branches['b'] = all_branches.get('b', 0.0)  # p.u.
        sienna_branches['rate'] = all_branches['s_nom']  # MVA
        sienna_branches['angle_limits_min'] = -30.0  # degrees
        sienna_branches['angle_limits_max'] = 30.0   # degrees
        
        # Add transformer-specific fields
        if 'tap_ratio' in all_branches.columns:
            sienna_branches['tap'] = all_branches.get('tap_ratio', 1.0)
        
        # Export to CSV
        branch_file = output_path / 'branch.csv'
        sienna_branches.to_csv(branch_file, index=False)
        
        logger.info(f"Exported {len(sienna_branches)} branches to {branch_file}")
        return {'branch': str(branch_file)}
    
    def _export_dc_branches(self, output_path: Path) -> Dict[str, str]:
        """Export links to dc_branch.csv."""
        links_df = self.network.links.copy()
        
        sienna_dc = pd.DataFrame()
        sienna_dc['name'] = links_df.index
        sienna_dc['connection_points_from'] = links_df['bus0']
        sienna_dc['connection_points_to'] = links_df['bus1']
        sienna_dc['active_power_limits_from'] = links_df['p_nom']  # MW
        sienna_dc['active_power_limits_to'] = links_df['p_nom']    # MW
        sienna_dc['loss'] = 1.0 - links_df.get('efficiency', 1.0)  # Loss factor
        
        # Export to CSV
        dc_file = output_path / 'dc_branch.csv'
        sienna_dc.to_csv(dc_file, index=False)
        
        logger.info(f"Exported {len(sienna_dc)} DC branches to {dc_file}")
        return {'dc_branch': str(dc_file)}
    
    def _export_storage(self, output_path: Path) -> Dict[str, str]:
        """Export storage units to storage.csv."""
        storage_df = self.network.storage_units.copy()
        
        sienna_storage = pd.DataFrame()
        sienna_storage['name'] = storage_df.index
        sienna_storage['bus'] = storage_df['bus']
        sienna_storage['energy_capacity'] = storage_df['p_nom'] * storage_df.get('max_hours', 6.0)  # MWh
        sienna_storage['input_active_power_limits'] = storage_df['p_nom']   # MW
        sienna_storage['output_active_power_limits'] = storage_df['p_nom']  # MW
        sienna_storage['efficiency_in'] = storage_df.get('efficiency_store', 0.95)
        sienna_storage['efficiency_out'] = storage_df.get('efficiency_dispatch', 0.95)
        sienna_storage['initial_energy'] = storage_df.get('state_of_charge_initial', 0.5) * sienna_storage['energy_capacity']
        
        # Export to CSV
        storage_file = output_path / 'storage.csv'
        sienna_storage.to_csv(storage_file, index=False)
        
        logger.info(f"Exported {len(sienna_storage)} storage units to {storage_file}")
        return {'storage': str(storage_file)}
    
    def _export_time_series_data(self, output_path: Path) -> Dict[str, str]:
        """Export all time series data."""
        if 'time_series' not in self.network_summary:
            logger.info("No time series data found")
            return {}
        
        ts_dir = output_path / 'timeseries_data'
        ts_dir.mkdir(exist_ok=True)
        
        files_created = {}
        
        # Export load time series
        if 'Load' in self.network_summary['time_series']:
            load_ts_files = self._export_load_time_series(ts_dir)
            files_created.update(load_ts_files)
        
        # Export renewable availability time series
        if 'Generator' in self.network_summary['time_series']:
            gen_ts_files = self._export_generator_time_series(ts_dir)
            files_created.update(gen_ts_files)
        
        return files_created
    
    def _export_load_time_series(self, ts_dir: Path) -> Dict[str, str]:
        """Export load time series data."""
        if not hasattr(self.network, 'loads_t') or not hasattr(self.network.loads_t, 'p_set'):
            return {}
        
        load_ts = self.network.loads_t.p_set
        if load_ts.empty:
            return {}
        
        # Convert to Sienna format
        sienna_load_ts = load_ts.copy()
        sienna_load_ts.index.name = 'DateTime'
        
        # Export to CSV
        load_ts_file = ts_dir / 'load_timeseries.csv'
        sienna_load_ts.to_csv(load_ts_file)
        
        # Add to time series metadata
        for load_name in sienna_load_ts.columns:
            self.time_series_metadata.append({
                'simulation': 'simulation1',
                'category': 'ElectricLoad',
                'component': load_name,
                'label': 'max_active_power',
                'data_file': 'timeseries_data/load_timeseries.csv',
                'data_column': load_name,
                'scaling_factor': 1.0
            })
        
        logger.info(f"Exported load time series to {load_ts_file}")
        return {'load_timeseries': str(load_ts_file)}
    
    def _export_generator_time_series(self, ts_dir: Path) -> Dict[str, str]:
        """Export generator availability time series (for renewables)."""
        if not hasattr(self.network, 'generators_t') or not hasattr(self.network.generators_t, 'p_max_pu'):
            return {}
        
        gen_availability = self.network.generators_t.p_max_pu
        if gen_availability.empty:
            return {}
        
        # Filter for renewable generators only
        renewable_carriers = ['wind', 'solar', 'pv', 'onshore', 'offshore', 'hydro', 'ror']
        renewable_gens = []
        for gen_name in gen_availability.columns:
            gen_carrier = self.network.generators.loc[gen_name, 'carrier']
            if any(carrier in gen_carrier.lower() for carrier in renewable_carriers):
                renewable_gens.append(gen_name)
        
        if not renewable_gens:
            return {}
        
        renewable_availability = gen_availability[renewable_gens]
        renewable_availability.index.name = 'DateTime'
        
        # Export to CSV
        gen_ts_file = ts_dir / 'renewable_availability.csv'
        renewable_availability.to_csv(gen_ts_file)
        
        # Add to time series metadata
        for gen_name in renewable_availability.columns:
            self.time_series_metadata.append({
                'simulation': 'simulation1',
                'category': 'RenewableDispatch',
                'component': gen_name,
                'label': 'max_active_power',
                'data_file': 'timeseries_data/renewable_availability.csv',
                'data_column': gen_name,
                'scaling_factor': 'Max'  # Scale by max value in column
            })
        
        logger.info(f"Exported renewable availability to {gen_ts_file}")
        return {'renewable_availability': str(gen_ts_file)}
    
    def _create_configuration_files(self, output_path: Path) -> Dict[str, str]:
        """Create PowerSystems.jl configuration files."""
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
        """Create user_descriptors.yaml for custom field mapping."""
        
        user_descriptors = {
            'bus': {
                'name': {'custom_name': 'name'},
                'base_voltage': {'custom_name': 'base_voltage'},
                'bus_type': {'custom_name': 'bus_type'},
                'area': {'custom_name': 'area'},
                'zone': {'custom_name': 'zone'},
                'longitude': {'custom_name': 'longitude'},
                'latitude': {'custom_name': 'latitude'}
            },
            
            'generator': {
                'name': {'custom_name': 'name'},
                'bus': {'custom_name': 'bus'},
                'fuel': {'custom_name': 'fuel'},
                'max_active_power': {'custom_name': 'max_active_power'},
                'min_active_power': {'custom_name': 'min_active_power'},
                'max_reactive_power': {'custom_name': 'max_reactive_power'},
                'min_reactive_power': {'custom_name': 'min_reactive_power'},
                'variable_cost': {'custom_name': 'variable_cost'},
                'startup_cost': {'custom_name': 'startup_cost'},
                'shutdown_cost': {'custom_name': 'shutdown_cost'},
                'ramp_up': {'custom_name': 'ramp_up'},
                'ramp_down': {'custom_name': 'ramp_down'},
                'min_up_time': {'custom_name': 'min_up_time'},
                'min_down_time': {'custom_name': 'min_down_time'},
                'prime_mover_type': {'custom_name': 'prime_mover_type'}
            },
            
            'load': {
                'name': {'custom_name': 'name'},
                'bus': {'custom_name': 'bus'},
                'max_active_power': {'custom_name': 'max_active_power'},
                'max_reactive_power': {'custom_name': 'max_reactive_power'},
                'power_factor': {'custom_name': 'power_factor'}
            },
            
            'branch': {
                'name': {'custom_name': 'name'},
                'connection_points_from': {'custom_name': 'connection_points_from'},
                'connection_points_to': {'custom_name': 'connection_points_to'},
                'r': {'custom_name': 'r'},
                'x': {'custom_name': 'x'},
                'b': {'custom_name': 'b'},
                'rate': {'custom_name': 'rate'},
                'angle_limits_min': {'custom_name': 'angle_limits_min'},
                'angle_limits_max': {'custom_name': 'angle_limits_max'},
                'tap': {'custom_name': 'tap'}
            },
            
            'dc_branch': {
                'name': {'custom_name': 'name'},
                'connection_points_from': {'custom_name': 'connection_points_from'},
                'connection_points_to': {'custom_name': 'connection_points_to'},
                'active_power_limits_from': {'custom_name': 'active_power_limits_from'},
                'active_power_limits_to': {'custom_name': 'active_power_limits_to'},
                'loss': {'custom_name': 'loss'}
            },
            
            'storage': {
                'name': {'custom_name': 'name'},
                'bus': {'custom_name': 'bus'},
                'energy_capacity': {'custom_name': 'energy_capacity'},
                'input_active_power_limits': {'custom_name': 'input_active_power_limits'},
                'output_active_power_limits': {'custom_name': 'output_active_power_limits'},
                'efficiency_in': {'custom_name': 'efficiency_in'},
                'efficiency_out': {'custom_name': 'efficiency_out'},
                'initial_energy': {'custom_name': 'initial_energy'}
            }
        }
        
        descriptors_file = output_path / 'user_descriptors.yaml'
        with open(descriptors_file, 'w') as f:
            yaml.dump(user_descriptors, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Created user descriptors: {descriptors_file}")
        return descriptors_file
    
    def _create_timeseries_metadata(self, output_path: Path) -> Path:
        """Create timeseries_metadata.json for time series data linking."""
        
        ts_metadata_file = output_path / 'timeseries_metadata.json'
        with open(ts_metadata_file, 'w') as f:
            json.dump(self.time_series_metadata, f, indent=2)
        
        logger.info(f"Created time series metadata: {ts_metadata_file}")
        return ts_metadata_file
    
    def _create_generator_mapping(self, output_path: Path) -> Path:
        """Create generator_mapping.yaml for generator type mapping."""
        
        generator_mapping = {
            'ThermalStandard': {
                'fuel': ['gas', 'ccgt', 'ocgt', 'coal', 'lignite', 'oil', 'nuclear', 'biomass', 'waste'],
                'prime_mover_type': ['CC', 'CT', 'ST']
            },
            'RenewableDispatch': {
                'fuel': ['wind', 'solar', 'pv', 'hydro', 'ror'],
                'prime_mover_type': ['WT', 'PV', 'HY']
            }
        }
        
        mapping_file = output_path / 'generator_mapping.yaml'
        with open(mapping_file, 'w') as f:
            yaml.dump(generator_mapping, f, default_flow_style=False)
        
        logger.info(f"Created generator mapping: {mapping_file}")
        return mapping_file
    
    def _create_julia_import_script(self, output_path: Path) -> Path:
        """Create Julia script to import the data into PowerSystems.jl."""
        
        julia_code = f'''#!/usr/bin/env julia

"""
Julia script to import PyPSA-exported data into PowerSystems.jl
Generated automatically by PyPSA-to-Sienna exporter
"""

using PowerSystems
using PowerSimulations
using Dates
using TimeSeries

# Configuration
data_dir = "{output_path.absolute()}"
base_power = {self.base_power}  # MVA
user_descriptors = joinpath(data_dir, "user_descriptors.yaml")

# Optional files (check if they exist)
timeseries_metadata_file = joinpath(data_dir, "timeseries_metadata.json")
generator_mapping_file = joinpath(data_dir, "generator_mapping.yaml")

println("Loading PyPSA data from: ", data_dir)

# Create PowerSystemTableData
try
    if isfile(timeseries_metadata_file) && isfile(generator_mapping_file)
        data = PowerSystemTableData(
            data_dir,
            base_power,
            user_descriptors;
            timeseries_metadata_file = timeseries_metadata_file,
            generator_mapping_file = generator_mapping_file
        )
    elseif isfile(timeseries_metadata_file)
        data = PowerSystemTableData(
            data_dir,
            base_power, 
            user_descriptors;
            timeseries_metadata_file = timeseries_metadata_file
        )
    else
        data = PowerSystemTableData(
            data_dir,
            base_power,
            user_descriptors
        )
    end
    
    println("✓ PowerSystemTableData created successfully")
    
    # Create System
    sys = System(data, time_series_in_memory = true)
    println("✓ PowerSystems.jl System created successfully")
    
    # Print summary
    println("\\n=== System Summary ===")
    println("Buses: ", length(get_components(Bus, sys)))
    println("Generators: ", length(get_components(Generator, sys)))
    println("Loads: ", length(get_components(ElectricLoad, sys)))
    println("Branches: ", length(get_components(Branch, sys)))
    println("Storage: ", length(get_components(Storage, sys)))
    
    # Check time series data
    ts_count = length(get_time_series_multiple(sys))
    if ts_count > 0
        println("Time series: ", ts_count, " time series found")
        
        # Get time series summary
        ts_horizon = get_time_series_resolution(sys)
        println("Time series resolution: ", ts_horizon)
    else
        println("Time series: No time series data")
    end
    
    # Save system for later use
    sys_file = joinpath(data_dir, "pypsa_system.json")
    to_json(sys, sys_file)
    println("\\n✓ System saved to: ", sys_file)
    
    # Example: Create and run a basic Economic Dispatch
    println("\\n=== Running Example Economic Dispatch ===")
    
    # Define simulation dates (use first week of time series if available)
    if ts_count > 0
        # Use actual time series dates
        ts_data = first(get_time_series_multiple(sys))
        start_time = TimeSeries.timestamp(ts_data[2])[1]
        end_time = start_time + Day(7)  # One week
    else
        # Use default dates
        start_time = DateTime("2030-01-01T00:00:00")
        end_time = start_time + Day(7)
    end
    
    # Set up Economic Dispatch problem
    template = EconomicDispatchTemplate()
    
    # Create Decision Model
    decision_model = DecisionModel(
        template,
        sys;
        name = "PyPSA_ED",
        optimizer = HiGHS.Optimizer,
        optimizer_solve_log_print = true
    )
    
    println("✓ Economic Dispatch model created")
    
    # Solve
    try
        solve!(decision_model)
        println("✓ Economic Dispatch solved successfully")
        
        # Get results
        results = OptimizationProblemResults(decision_model)
        
        # Export results
        results_dir = joinpath(data_dir, "dispatch_results")
        mkpath(results_dir)
        
        # Export generator results
        gen_results = read_realized_variables(results, names = [:ActivePowerVariable])
        export_realized_results(results, results_dir)
        
        println("✓ Results exported to: ", results_dir)
        
    catch e
        println("✗ Economic Dispatch failed: ", e)
    end
    
    return sys
    
catch e
    println("✗ Error creating PowerSystems.jl system: ", e)
    throw(e)
end

println("\\n=== Import Complete ===")
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
    Main function to export PyPSA network to Sienna-compatible format.
    
    This function integrates with your existing solve_network_dispatch.py workflow.
    
    Parameters:
    -----------
    network : pypsa.Network
        Solved PyPSA dispatch network
    scenario_setup : dict
        Scenario configuration from PyPSA-RSA
    output_dir : str
        Output directory for exported files
    include_time_series : bool
        Whether to export time series data
        
    Returns:
    --------
    Dict[str, str]
        Dictionary with created file paths and import instructions
    """
    
    # Create exporter and export
    exporter = PyPSAToSiennaExporter(network, scenario_setup)
    export_results = exporter.export_to_csv(output_dir, include_time_series)
    
    # Add import instructions
    export_results['import_instructions'] = [
        "1. Navigate to the export directory",
        "2. Run: julia import_to_powersystems.jl",
        "3. Or import manually in Julia:",
        "   using PowerSystems",
        f"   data = PowerSystemTableData(\"{output_dir}\", 100.0, \"user_descriptors.yaml\")",
        "   sys = System(data, time_series_in_memory=true)"
    ]
    
    return export_results


def validate_pypsa_network(network: pypsa.Network) -> Dict[str, Any]:
    """
    Validate PyPSA network for Sienna export.
    
    Parameters:
    -----------
    network : pypsa.Network
        PyPSA network to validate
        
    Returns:
    --------
    Dict[str, Any]
        Validation results with is_valid flag and issues list
    """
    issues = []
    
    # Check for required components
    if network.buses.empty:
        issues.append("No buses found in network")
    
    if network.generators.empty and network.loads.empty:
        issues.append("No generators or loads found in network")
    
    # Check bus references
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
    
    # Check for required fields
    required_gen_fields = ['bus', 'p_nom', 'carrier']
    if not network.generators.empty:
        missing_gen_fields = set(required_gen_fields) - set(network.generators.columns)
        if missing_gen_fields:
            issues.append(f"Generators missing required fields: {missing_gen_fields}")
    
    required_bus_fields = ['v_nom']
    missing_bus_fields = set(required_bus_fields) - set(network.buses.columns)
    if missing_bus_fields:
        issues.append(f"Buses missing required fields: {missing_bus_fields}")
    
    # Check for solved optimization results (optional but recommended)
    if not hasattr(network, 'objective'):
        issues.append("Network does not appear to have been solved (no objective value)")
    
    is_valid = len(issues) == 0
    return {'is_valid': is_valid, 'issues': issues}


# Integration with solve_network_dispatch.py
def integrate_with_dispatch_solver():
    """
    Integration example for your solve_network_dispatch.py file.
    
    Add this to your solve_network_dispatch function where export_to_Sienna=True.
    """
    
    integration_code = '''
    # In your solve_network_dispatch.py, add this after solving:
    
    if export_to_Sienna:
        from export_to_sienna import export_pypsa_to_sienna
        
        logger.info("Exporting solved network to Sienna format...")
        
        # Export to Sienna CSV format
        export_results = export_pypsa_to_sienna(
            network=n,
            scenario_setup=scenario_setup,
            output_dir=sienna_output_dir,
            include_time_series=True
        )
        
        logger.info("Sienna export complete!")
        logger.info("Files created:")
        for file_type, file_path in export_results.items():
            if file_type != 'import_instructions':
                logger.info(f"  {file_type}: {file_path}")
        
        logger.info("\\nTo import in Julia:")
        for instruction in export_results['import_instructions']:
            logger.info(f"  {instruction}")
        
        return export_results
    '''
    
    return integration_code


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # This would be called from your solve_network_dispatch.py
    # when export_to_Sienna=True

    if 'snakemake' not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            'export_to_sienna', 
            **{
                'scenario':'TEST',
                'year':2030
            }
        )

    n = pypsa.Network(snakemake.input.dispatch_network)

    scenario_setup = load_scenario_definition(snakemake)
    export_folder = snakemake.output.sienna_export_dir

    logging.info(f"Exporting to Sienna format to {export_folder}")
    export_pypsa_to_sienna(
            network=n,
            scenario_setup=scenario_setup,
            output_dir=export_folder,
            include_time_series=True
    )
    
    logging.info(f"Exporting to Sienna format to {export_folder}")
