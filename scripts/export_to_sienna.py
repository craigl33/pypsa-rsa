"""
CSV-based PyPSA to Sienna Export Skeleton
Integration with PyPSA-RSA solve_network_dispatch.py

This skeleton provides the structure for exporting PyPSA dispatch networks
to Sienna-compatible CSV format that PowerSystems.jl can import.
"""

import pypsa
import pandas as pd
import numpy as np
import os
import yaml
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class PyPSAToSiennaCSVExporter:
    """
    Exports PyPSA dispatch networks to Sienna-compatible CSV format.
    
    This class handles the conversion from PyPSA network components to 
    the CSV table format that PowerSystems.jl can parse using its
    PowerSystemTableData functionality.
    
    Exports ALL PyPSA components systematically:
    - Static components (buses, generators, loads, etc.)
    - Time-varying data (p_max_pu, p_set, etc.) 
    - Constraints (GlobalConstraint, custom constraints)
    - Network metadata and configuration
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
        self.base_power = 100.0  # MVA base for PowerSystems.jl
        
        # Component analysis results
        self.component_inventory = {}
        self.time_series_inventory = {}
        self.constraint_inventory = {}
        
        # PyPSA to Sienna component mapping
        self.component_mappings = self._initialize_component_mappings()
        
        self._analyze_all_components()
    
    def export_to_sienna_csv(self, output_dir: str, 
                           include_time_series: bool = True) -> Dict[str, str]:
        """
        Main export function - converts PyPSA network to Sienna CSV format.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save CSV files
        include_time_series : bool
            Whether to export time series data
            
        Returns:
        --------
        Dict[str, str]
            Dictionary with paths to created files and import instructions
        """
        logger.info("Starting PyPSA to Sienna CSV export...")
        
        # Create output directory structure
        output_path = Path(output_dir)
        self._create_directory_structure(output_path)
        
        # Export static component data
        files_created = self._export_static_components(output_path)
        
        # Export time series data if requested
        if include_time_series:
            ts_files = self._export_time_series_data(output_path)
            files_created.update(ts_files)
        
        # Create PowerSystems.jl configuration files
        config_files = self._create_powersystems_config(output_path)
        files_created.update(config_files)
        
        # Create Julia import script
        import_script = self._create_julia_import_script(output_path)
        files_created['julia_import_script'] = str(import_script)
        
        logger.info(f"Export complete. Created {len(files_created)} files.")
        return files_created
    
    def _initialize_component_mappings(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize mappings from PyPSA components to Sienna equivalents.
        
        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Mapping configuration for each PyPSA component type
        """
        return {
            # Core network components
            'Bus': {
                'sienna_type': 'Bus',
                'required_fields': ['name', 'base_voltage'],
                'optional_fields': ['bus_type', 'area', 'zone', 'longitude', 'latitude', 
                                  'voltage_limits_min', 'voltage_limits_max'],
                'export_priority': 1  # Export first (other components reference buses)
            },
            
            'Carrier': {
                'sienna_type': 'Fuel',  # Map to fuel types in Sienna
                'required_fields': ['name'],
                'optional_fields': ['co2_emissions', 'fuel_cost'],
                'export_priority': 2
            },
            
            # Generation components
            'Generator': {
                'sienna_type': 'ThermalStandard',  # Default, will split by carrier
                'required_fields': ['name', 'bus', 'max_active_power'],
                'optional_fields': ['min_active_power', 'max_reactive_power', 'min_reactive_power',
                                  'ramp_up', 'ramp_down', 'start_up_cost', 'variable_cost',
                                  'min_up_time', 'min_down_time', 'fuel', 'prime_mover_type'],
                'export_priority': 4,
                'split_by_carrier': True  # Split thermal vs renewable
            },
            
            # Load components
            'Load': {
                'sienna_type': 'PowerLoad',
                'required_fields': ['name', 'bus', 'max_active_power'],
                'optional_fields': ['max_reactive_power', 'power_factor'],
                'export_priority': 3
            },
            
            # Transmission components
            'Line': {
                'sienna_type': 'Line',
                'required_fields': ['name', 'connection_points_from', 'connection_points_to', 'r', 'x'],
                'optional_fields': ['b', 'rate', 'angle_limits_min', 'angle_limits_max'],
                'export_priority': 5
            },
            
            'Transformer': {
                'sienna_type': 'Transformer2W',
                'required_fields': ['name', 'connection_points_from', 'connection_points_to', 'r', 'x'],
                'optional_fields': ['tap', 'rate', 'primary_shunt'],
                'export_priority': 5
            },
            
            'Link': {
                'sienna_type': 'TwoTerminalHVDCLine',
                'required_fields': ['name', 'connection_points_from', 'connection_points_to'],
                'optional_fields': ['active_power_limits_from', 'active_power_limits_to', 'loss'],
                'export_priority': 6
            },
            
            # Storage components
            'StorageUnit': {
                'sienna_type': 'GenericBattery',
                'required_fields': ['name', 'bus', 'energy_capacity', 'input_active_power_limits', 'output_active_power_limits'],
                'optional_fields': ['efficiency_in', 'efficiency_out', 'state_of_charge_limits'],
                'export_priority': 7
            },
            
            'Store': {
                'sienna_type': 'HydroEnergyReservoir', 
                'required_fields': ['name', 'bus', 'storage_capacity'],
                'optional_fields': ['inflow', 'initial_storage'],
                'export_priority': 7
            },
            
            # Network topology
            'SubNetwork': {
                'sienna_type': 'Area',  # Map to system areas
                'required_fields': ['name'],
                'optional_fields': ['carrier'],
                'export_priority': 8
            },
            
            # Other components
            'ShuntImpedance': {
                'sienna_type': 'FixedAdmittance',
                'required_fields': ['name', 'bus'],
                'optional_fields': ['g', 'b'],
                'export_priority': 9
            },
            
            # Constraints and metadata
            'GlobalConstraint': {
                'sienna_type': 'GlobalConstraint',  # Custom export format
                'required_fields': ['name', 'type', 'sense'],
                'optional_fields': ['constant', 'carrier_attribute'],
                'export_priority': 10
            },
            
            # Type definitions (export as reference data)
            'LineType': {
                'sienna_type': 'LineType',
                'required_fields': ['name'],
                'optional_fields': ['r', 'x', 'b', 'i_nom'],
                'export_priority': 11
            },
            
            'TransformerType': {
                'sienna_type': 'TransformerType', 
                'required_fields': ['name'],
                'optional_fields': ['r', 'x', 's_nom'],
                'export_priority': 11
            }
        }
    
    def _analyze_all_components(self):
        """
        Systematically analyze all components in the PyPSA network.
        
        This function inventories all static components, time series data,
        and constraints present in the network.
        """
        logger.info("Analyzing PyPSA network components...")
        
        # Analyze static components
        for component in self.network.iterate_components():
            component_name = component.name
            component_df = component.df
            
            if not component_df.empty:
                self.component_inventory[component_name] = {
                    'count': len(component_df),
                    'columns': list(component_df.columns),
                    'has_data': True,
                    'sample_data': component_df.head(2).to_dict() if len(component_df) > 0 else {}
                }
                
                logger.info(f"Found {len(component_df)} {component_name} components")
                
                # Analyze time-varying data for this component
                self._analyze_component_time_series(component)
            else:
                self.component_inventory[component_name] = {
                    'count': 0,
                    'has_data': False
                }
        
        # Analyze constraints
        self._analyze_constraints()
        
        # Log summary
        self._log_component_summary()
    
    def _analyze_component_time_series(self, component):
        """Analyze time-varying data for a specific component."""
        component_name = component.name
        
        # Check for time-varying attributes
        if hasattr(component, 'pnl'):
            ts_data = {}
            for attr_name, attr_data in component.pnl.items():
                if not attr_data.empty:
                    ts_data[attr_name] = {
                        'shape': attr_data.shape,
                        'columns': list(attr_data.columns),
                        'time_range': (str(attr_data.index[0]), str(attr_data.index[-1])) if len(attr_data) > 0 else None,
                        'has_data': True
                    }
                    logger.debug(f"Found time series {component_name}.{attr_name}: {attr_data.shape}")
            
            if ts_data:
                self.time_series_inventory[component_name] = ts_data
    
    def _analyze_constraints(self):
        """Analyze GlobalConstraints and any custom constraints."""
        
        # Global constraints
        if not self.network.global_constraints.empty:
            self.constraint_inventory['GlobalConstraint'] = {
                'count': len(self.network.global_constraints),
                'types': list(self.network.global_constraints['type'].unique()),
                'carriers': list(self.network.global_constraints['carrier_attribute'].unique()),
                'data': self.network.global_constraints.to_dict('records')
            }
            logger.info(f"Found {len(self.network.global_constraints)} global constraints")
        
        # Check for solved optimization model constraints
        if hasattr(self.network, 'model') and self.network.model is not None:
            # TODO: Extract custom constraints from the optimization model
            # This would include operational limits, reserve margins, etc.
            pass
    
    def _log_component_summary(self):
        """Log a summary of all components found in the network."""
        logger.info("=== PyPSA Network Component Summary ===")
        
        # Static components
        total_components = sum(info['count'] for info in self.component_inventory.values() if info.get('has_data', False))
        logger.info(f"Total static components: {total_components}")
        
        for comp_name, info in self.component_inventory.items():
            if info.get('has_data', False):
                logger.info(f"  {comp_name}: {info['count']} components")
        
        # Time series
        total_time_series = sum(len(ts_data) for ts_data in self.time_series_inventory.values())
        logger.info(f"Total time series attributes: {total_time_series}")
        
        for comp_name, ts_data in self.time_series_inventory.items():
            for attr_name, attr_info in ts_data.items():
                logger.info(f"  {comp_name}.{attr_name}: {attr_info['shape']}")
        
        # Constraints
        total_constraints = sum(info['count'] for info in self.constraint_inventory.values())
        logger.info(f"Total constraints: {total_constraints}")
        
        logger.info("=" * 40)
    
    def _create_directory_structure(self, output_path: Path):
        """Create the directory structure for Sienna CSV export."""
        # TODO: Create directories for:
        # - static_data/
        # - time_series_data/
        # - config/
        pass
    
    def _export_static_components(self, output_path: Path) -> Dict[str, str]:
        """
        Export all static component data to CSV files systematically.
        
        This function exports ALL PyPSA components, not just the common ones.
        Components are exported in priority order to handle dependencies.
        """
        files_created = {}
        
        # Sort components by export priority
        components_to_export = []
        for comp_name, comp_info in self.component_inventory.items():
            if comp_info.get('has_data', False):
                mapping = self.component_mappings.get(comp_name, {})
                priority = mapping.get('export_priority', 999)
                components_to_export.append((priority, comp_name, comp_info))
        
        components_to_export.sort(key=lambda x: x[0])  # Sort by priority
        
        logger.info(f"Exporting {len(components_to_export)} component types...")
        
        # Export each component type
        for priority, comp_name, comp_info in components_to_export:
            try:
                component_files = self._export_component_type(output_path, comp_name)
                files_created.update(component_files)
                logger.info(f"✓ Exported {comp_name} ({comp_info['count']} components)")
            except Exception as e:
                logger.error(f"✗ Failed to export {comp_name}: {e}")
                # Continue with other components
        
        return files_created
    
    def _export_component_type(self, output_path: Path, component_name: str) -> Dict[str, str]:
        """
        Export a specific PyPSA component type to CSV.
        
        Parameters:
        -----------
        output_path : Path
            Base output directory
        component_name : str
            PyPSA component name (e.g., 'Generator', 'Bus', etc.)
            
        Returns:
        --------
        Dict[str, str]
            Dictionary of files created for this component type
        """
        files_created = {}
        
        # Get component data from PyPSA network
        component_df = getattr(self.network, component_name.lower() + 's', pd.DataFrame())
        
        if component_df.empty:
            return files_created
        
        # Get mapping configuration
        mapping = self.component_mappings.get(component_name, {})
        
        # Handle special cases for components that need splitting
        if mapping.get('split_by_carrier', False) and component_name == 'Generator':
            files_created.update(self._export_generators_by_type(output_path, component_df))
        else:
            # Standard component export
            files_created.update(self._export_standard_component(output_path, component_name, component_df, mapping))
        
        return files_created
    
    def _export_standard_component(self, output_path: Path, component_name: str, 
                                 component_df: pd.DataFrame, mapping: Dict[str, Any]) -> Dict[str, str]:
        """Export a standard component to CSV with appropriate field mapping."""
        
        static_dir = output_path / "static_data"
        static_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert PyPSA component to Sienna format
        sienna_df = self._convert_component_to_sienna_format(component_df, component_name, mapping)
        
        # Determine output filename
        sienna_type = mapping.get('sienna_type', component_name.lower())
        filename = f"{sienna_type.lower()}.csv"
        filepath = static_dir / filename
        
        # Export to CSV
        sienna_df.to_csv(filepath, index=False)
        
        return {f"{component_name.lower()}_static": str(filepath)}
    
    def _export_generators_by_type(self, output_path: Path, generators_df: pd.DataFrame) -> Dict[str, str]:
        """
        Export generators split by type (thermal vs renewable).
        
        This handles the special case where PyPSA has one Generator component
        but Sienna has separate ThermalStandard and RenewableDispatch components.
        """
        files_created = {}
        static_dir = output_path / "static_data"
        
        # Split generators by carrier type
        renewable_carriers = ['wind', 'solar', 'hydro', 'pv', 'onshore', 'offshore']
        
        # Identify renewable generators
        is_renewable = generators_df['carrier'].str.lower().str.contains('|'.join(renewable_carriers), na=False)
        
        thermal_gens = generators_df[~is_renewable]
        renewable_gens = generators_df[is_renewable]
        
        # Export thermal generators
        if not thermal_gens.empty:
            thermal_df = self._convert_thermal_generators(thermal_gens)
            thermal_file = static_dir / "thermal_generators.csv"
            thermal_df.to_csv(thermal_file, index=False)
            files_created['thermal_generators'] = str(thermal_file)
            logger.info(f"  Exported {len(thermal_gens)} thermal generators")
        
        # Export renewable generators  
        if not renewable_gens.empty:
            renewable_df = self._convert_renewable_generators(renewable_gens)
            renewable_file = static_dir / "renewable_generators.csv"
            renewable_df.to_csv(renewable_file, index=False)
            files_created['renewable_generators'] = str(renewable_file)
            logger.info(f"  Exported {len(renewable_gens)} renewable generators")
        
        return files_created
    
    def _convert_component_to_sienna_format(self, component_df: pd.DataFrame, 
                                          component_name: str, mapping: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert a PyPSA component DataFrame to Sienna-compatible format.
        
        This is the main conversion function that handles field mapping,
        unit conversions, and data transformations.
        """
        sienna_df = pd.DataFrame()
        
        # Copy index as name (Sienna requirement)
        sienna_df['name'] = component_df.index
        
        # Map fields based on component type
        if component_name == 'Bus':
            sienna_df = self._convert_buses(component_df)
        elif component_name == 'Load':
            sienna_df = self._convert_loads(component_df)
        elif component_name == 'Line':
            sienna_df = self._convert_lines(component_df)
        elif component_name == 'Transformer':
            sienna_df = self._convert_transformers(component_df)
        elif component_name == 'Link':
            sienna_df = self._convert_links(component_df)
        elif component_name == 'StorageUnit':
            sienna_df = self._convert_storage_units(component_df)
        elif component_name == 'Store':
            sienna_df = self._convert_stores(component_df)
        elif component_name == 'GlobalConstraint':
            sienna_df = self._convert_global_constraints(component_df)
        else:
            # Generic conversion for other component types
            sienna_df = self._convert_generic_component(component_df, mapping)
        
        return sienna_df
    
    def _convert_thermal_generators(self, thermal_gens: pd.DataFrame) -> pd.DataFrame:
        """Convert thermal generators to Sienna ThermalStandard format."""
        # TODO: Implement thermal generator conversion
        # Required fields: name, bus, fuel, max_active_power, min_active_power
        # Optional: max_reactive_power, min_reactive_power, ramp_up, ramp_down,
        #          start_up_cost, variable_cost, min_up_time, min_down_time
        pass
    
    def _convert_renewable_generators(self, renewable_gens: pd.DataFrame) -> pd.DataFrame:
        """Convert renewable generators to Sienna RenewableDispatch format."""
        # TODO: Implement renewable generator conversion
        pass
    
    def _convert_buses(self, buses_df: pd.DataFrame) -> pd.DataFrame:
        """Convert PyPSA buses to Sienna Bus format."""
        # TODO: Implement bus conversion
        pass
    
    def _convert_loads(self, loads_df: pd.DataFrame) -> pd.DataFrame:
        """Convert PyPSA loads to Sienna PowerLoad format."""
        # TODO: Implement load conversion
        pass
    
    def _convert_lines(self, lines_df: pd.DataFrame) -> pd.DataFrame:
        """Convert PyPSA lines to Sienna Line format."""
        # TODO: Implement line conversion
        pass
    
    def _convert_transformers(self, transformers_df: pd.DataFrame) -> pd.DataFrame:
        """Convert PyPSA transformers to Sienna Transformer2W format."""
        # TODO: Implement transformer conversion
        pass
    
    def _convert_links(self, links_df: pd.DataFrame) -> pd.DataFrame:
        """Convert PyPSA links to Sienna TwoTerminalHVDCLine format."""
        # TODO: Implement link conversion
        pass
    
    def _convert_storage_units(self, storage_df: pd.DataFrame) -> pd.DataFrame:
        """Convert PyPSA storage units to Sienna GenericBattery format."""
        # TODO: Implement storage unit conversion
        pass
    
    def _convert_stores(self, stores_df: pd.DataFrame) -> pd.DataFrame:
        """Convert PyPSA stores to Sienna HydroEnergyReservoir format."""
        # TODO: Implement store conversion
        pass
    
    def _convert_global_constraints(self, constraints_df: pd.DataFrame) -> pd.DataFrame:
        """Convert PyPSA global constraints to exportable format."""
        # TODO: Implement constraint export
        # This might need special handling as Sienna may not have direct equivalent
        pass
    
    def _convert_generic_component(self, component_df: pd.DataFrame, mapping: Dict[str, Any]) -> pd.DataFrame:
        """Generic conversion for components without specific converters."""
        # TODO: Implement generic field mapping based on the mapping configuration
        pass
    
    def _export_buses(self, output_path: Path) -> Dict[str, str]:
        """Export bus data to bus.csv for PowerSystems.jl."""
        # TODO: Convert PyPSA buses to PowerSystems.jl bus format
        # Required columns: name, base_voltage, bus_type, area, zone
        # Optional: longitude, latitude, voltage_limits_min, voltage_limits_max
        pass
    
    def _export_thermal_generators(self, output_path: Path) -> Dict[str, str]:
        """Export thermal generators to thermal_generators.csv."""
        # TODO: Convert PyPSA thermal generators to PowerSystems.jl format
        # Required columns: name, bus, fuel, max_active_power, min_active_power
        # Optional: max_reactive_power, min_reactive_power, ramp_up, ramp_down,
        #          start_up_cost, variable_cost, min_up_time, min_down_time
        pass
    
    def _export_renewable_generators(self, output_path: Path) -> Dict[str, str]:
        """Export renewable generators to renewable_generators.csv."""
        # TODO: Convert PyPSA renewable generators to PowerSystems.jl format
        # Required columns: name, bus, prime_mover_type, max_active_power
        # Optional: max_reactive_power, variable_cost
        pass
    
    def _export_loads(self, output_path: Path) -> Dict[str, str]:
        """Export loads to loads.csv."""
        # TODO: Convert PyPSA loads to PowerSystems.jl format
        # Required columns: name, bus, max_active_power, max_reactive_power
        pass
    
    def _export_transmission(self, output_path: Path) -> Dict[str, str]:
        """Export transmission lines/links to branch.csv."""
        # TODO: Convert PyPSA lines and links to PowerSystems.jl branch format
        # Required columns: name, connection_points_from, connection_points_to,
        #                  r, x, b, rate
        pass
    
    def _export_storage(self, output_path: Path) -> Dict[str, str]:
        """Export storage units to storage.csv."""
        # TODO: Convert PyPSA storage units to PowerSystems.jl format
        # Required columns: name, bus, max_active_power, max_reactive_power,
        #                  storage_capacity, efficiency_in, efficiency_out
        pass
    
    def _export_time_series_data(self, output_path: Path) -> Dict[str, str]:
        """
        Export ALL time series data systematically.
        
        This function exports time-varying data for all components that have it,
        not just loads and renewable availability.
        """
        files_created = {}
        
        if not self.time_series_inventory:
            logger.info("No time series data found to export")
            return files_created
        
        time_series_dir = output_path / "time_series_data"
        time_series_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Exporting time series data...")
        
        # Export time series for each component type
        for component_name, ts_attributes in self.time_series_inventory.items():
            try:
                component_ts_files = self._export_component_time_series(
                    time_series_dir, component_name, ts_attributes
                )
                files_created.update(component_ts_files)
                logger.info(f"✓ Exported {len(ts_attributes)} time series for {component_name}")
            except Exception as e:
                logger.error(f"✗ Failed to export time series for {component_name}: {e}")
        
        return files_created
    
    def _export_component_time_series(self, ts_dir: Path, component_name: str, 
                                    ts_attributes: Dict[str, Any]) -> Dict[str, str]:
        """
        Export time series data for a specific component type.
        
        Parameters:
        -----------
        ts_dir : Path
            Time series output directory
        component_name : str
            Component name (e.g., 'Generator', 'Load')
        ts_attributes : Dict[str, Any]
            Time series attributes for this component
            
        Returns:
        --------
        Dict[str, str]
            Files created for this component's time series
        """
        files_created = {}
        
        # Get the component's time-varying data
        component = getattr(self.network, component_name.lower() + 's')
        component_t = getattr(self.network, component_name.lower() + 's_t')
        
        for attr_name, attr_info in ts_attributes.items():
            if not attr_info.get('has_data', False):
                continue
            
            # Get the time series data
            ts_data = getattr(component_t, attr_name)
            
            if ts_data.empty:
                continue
            
            # Create filename
            filename = f"{component_name.lower()}_{attr_name}.csv"
            filepath = ts_dir / filename
            
            # Convert to Sienna-compatible format
            sienna_ts_data = self._convert_time_series_to_sienna_format(
                ts_data, component_name, attr_name
            )
            
            # Export to CSV
            sienna_ts_data.to_csv(filepath)
            files_created[f"{component_name.lower()}_{attr_name}_ts"] = str(filepath)
            
            logger.debug(f"  Exported {attr_name}: {sienna_ts_data.shape}")
        
        return files_created
    
    def _convert_time_series_to_sienna_format(self, ts_data: pd.DataFrame, 
                                            component_name: str, attr_name: str) -> pd.DataFrame:
        """
        Convert PyPSA time series to Sienna-compatible CSV format.
        
        Sienna expects time series data in a specific format with timestamps
        and component columns.
        """
        # TODO: Implement time series format conversion
        # This should:
        # 1. Ensure proper timestamp format
        # 2. Handle multi-investment period data if present
        # 3. Add metadata columns if needed
        # 4. Handle unit conversions (MW to per-unit if needed)
        pass
    
    def _export_constraints_data(self, output_path: Path) -> Dict[str, str]:
        """
        Export constraint data including GlobalConstraints and custom constraints.
        
        This exports constraint definitions that can be used to reconstruct
        the optimization problem in Sienna.
        """
        files_created = {}
        
        if not self.constraint_inventory:
            logger.info("No constraints found to export")
            return files_created
        
        constraints_dir = output_path / "constraints_data"
        constraints_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Exporting constraint data...")
        
        # Export GlobalConstraints
        if 'GlobalConstraint' in self.constraint_inventory:
            gc_data = self.constraint_inventory['GlobalConstraint']
            gc_df = pd.DataFrame(gc_data['data'])
            
            gc_file = constraints_dir / "global_constraints.csv"
            gc_df.to_csv(gc_file, index=False)
            files_created['global_constraints'] = str(gc_file)
            
            logger.info(f"✓ Exported {len(gc_df)} global constraints")
        
        # Export custom constraints metadata
        # TODO: If we can extract custom constraints from the solved model,
        # export them here as well
        
        # Export constraint summary
        constraint_summary = {
            'total_constraints': sum(info['count'] for info in self.constraint_inventory.values()),
            'constraint_types': list(self.constraint_inventory.keys()),
            'export_timestamp': datetime.now().isoformat()
        }
        
        summary_file = constraints_dir / "constraint_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(constraint_summary, f, indent=2)
        files_created['constraint_summary'] = str(summary_file)
        
        return files_created
    
    def _export_load_time_series(self, output_path: Path) -> Dict[str, str]:
        """Export load time series data to CSV files."""
        # TODO: Export n.loads_t.p_set data in PowerSystems.jl format
        pass
    
    def _export_renewable_time_series(self, output_path: Path) -> Dict[str, str]:
        """Export renewable availability time series data."""
        # TODO: Export n.generators_t.p_max_pu data for renewables
        pass
    
    def _create_powersystems_config(self, output_path: Path) -> Dict[str, str]:
        """Create PowerSystems.jl configuration files."""
        config_files = {}
        
        # Create user_descriptors.yaml
        config_files['user_descriptors'] = self._create_user_descriptors(output_path)
        
        # Create time series metadata file
        config_files['timeseries_metadata'] = self._create_timeseries_metadata(output_path)
        
        return config_files
    
    def _create_user_descriptors(self, output_path: Path) -> str:
        """Create user_descriptors.yaml for PowerSystems.jl parsing."""
        # TODO: Create YAML file that maps CSV columns to PowerSystems.jl fields
        # This tells PowerSystems.jl how to interpret the CSV data
        pass
    
    def _create_timeseries_metadata(self, output_path: Path) -> str:
        """Create time series metadata file (JSON or CSV)."""
        # TODO: Create metadata file that links components to their time series
        pass
    
    def _create_julia_import_script(self, output_path: Path) -> Path:
        """Create Julia script to import the exported data into PowerSystems.jl."""
        # TODO: Create .jl script with PowerSystems.jl import commands
        pass
    
    def _has_load_time_series(self) -> bool:
        """Check if network has load time series data."""
        # TODO: Check if n.loads_t.p_set exists and is not empty
        pass
    
    def _has_renewable_time_series(self) -> bool:
        """Check if network has renewable time series data."""
        # TODO: Check if n.generators_t.p_max_pu exists for renewable generators
        pass


# Integration functions for solve_network_dispatch.py

def export_dispatch_network_to_sienna_csv(network: pypsa.Network, 
                                         scenario_setup: dict,
                                         output_dir: str,
                                         year: Optional[int] = None) -> Dict[str, str]:
    """
    Export a PyPSA dispatch network to Sienna CSV format.
    
    This function is designed to be called from solve_network_dispatch.py
    when export_to_Sienna=True.
    
    Parameters:
    -----------
    network : pypsa.Network
        Solved PyPSA dispatch network
    scenario_setup : dict
        Scenario configuration from PyPSA-RSA
    output_dir : str
        Base output directory 
    year : int, optional
        Dispatch year (for organizing outputs)
        
    Returns:
    --------
    Dict[str, str]
        Dictionary with created file paths and import instructions
    """
    
    # Create year-specific output directory if year is provided
    if year is not None:
        final_output_dir = os.path.join(output_dir, f"dispatch_{year}")
    else:
        final_output_dir = output_dir
    
    # Initialize exporter
    exporter = PyPSAToSiennaCSVExporter(network, scenario_setup)
    
    # Export to CSV format
    return exporter.export_to_sienna_csv(
        output_dir=final_output_dir,
        include_time_series=True
    )


def validate_sienna_export_requirements(network: pypsa.Network) -> Tuple[bool, List[str]]:
    """
    Validate that the PyPSA network has the required components for Sienna export.
    
    Parameters:
    -----------
    network : pypsa.Network
        PyPSA network to validate
        
    Returns:
    --------
    Tuple[bool, List[str]]
        (is_valid, list_of_issues)
    """
    issues = []
    
    # TODO: Check for required components and data
    # - Must have buses
    # - Must have at least one generator or load
    # - Check for valid bus assignments
    # - Validate solved optimization results exist
    
    is_valid = len(issues) == 0
    return is_valid, issues


def get_sienna_import_commands(export_results: Dict[str, str]) -> List[str]:
    """
    Generate Julia commands for importing the exported data into PowerSystems.jl.
    
    Parameters:
    -----------
    export_results : Dict[str, str]
        Results from export_dispatch_network_to_sienna_csv()
        
    Returns:
    --------
    List[str]
        List of Julia commands to run
    """
    commands = []
    
    # TODO: Generate Julia import commands based on what was exported
    # Example commands:
    # - using PowerSystems
    # - data = PowerSystemTableData("path/to/csv/directory", 100.0, "user_descriptors.yaml")
    # - sys = System(data)
    
    return commands


# Integration with your existing solve_network_dispatch.py
def integrate_with_solve_network_dispatch():
    """
    Integration points for your solve_network_dispatch.py file.
    
    This is a reference showing where to add the Sienna export functionality.
    """
    
    integration_points = {
        "imports": [
            "from sienna_csv_export import export_dispatch_network_to_sienna_csv",
            "from sienna_csv_export import validate_sienna_export_requirements",
            "from sienna_csv_export import get_sienna_import_commands"
        ],
        
        "function_signature_update": """
        def solve_network_dispatch(n, sns, enable_unit_commitment=False, 
                                  export_to_Sienna=False, sienna_output_dir=None):
        """,
        
        "export_logic": """
        if export_to_Sienna:
            if sienna_output_dir is None:
                raise ValueError("sienna_output_dir must be specified when export_to_Sienna=True")
            
            # Validate network is suitable for export
            is_valid, issues = validate_sienna_export_requirements(n)
            if not is_valid:
                logging.error(f"Network validation failed: {issues}")
                raise ValueError("Network cannot be exported to Sienna format")
            
            # Export to Sienna CSV format
            logging.info("Exporting network to Sienna CSV format...")
            export_results = export_dispatch_network_to_sienna_csv(
                network=n,
                scenario_setup=scenario_setup,
                output_dir=sienna_output_dir,
                year=wildcards.get('year')  # From snakemake wildcards
            )
            
            # Generate import commands
            import_commands = get_sienna_import_commands(export_results)
            
            logging.info("Sienna export complete!")
            logging.info("To import in Julia:")
            for cmd in import_commands:
                logging.info(f"  {cmd}")
            
            return export_results
        """,
        
        "snakemake_rule_update": """
        # In your Snakemake rule, you can now use:
        rule solve_network_dispatch:
            input:
                dispatch_network="networks/{folder}/elec/{scenario}/dispatch-{year}.nc"
            output:
                dispatch_results="results/{folder}/dispatch/{scenario}/dispatch_{year}.nc",
                sienna_export=directory("results/{folder}/sienna/{scenario}/dispatch_{year}/")  # Optional
            run:
                n = pypsa.Network(input.dispatch_network)
                
                # For Sienna export
                if config.get('export_to_sienna', False):
                    solve_network_dispatch(
                        n, n.snapshots, 
                        export_to_Sienna=True,
                        sienna_output_dir=output.sienna_export
                    )
                else:
                    # Normal dispatch solve
                    solve_network_dispatch(n, n.snapshots)
                    n.export_to_netcdf(output.dispatch_results)
        """
    }
    
    return integration_points


if __name__ == "__main__":
    # Example usage for testing
    logging.basicConfig(level=logging.INFO)
    
    # This would be called from your solve_network_dispatch.py
    # when export_to_Sienna=True
    
    # Example:
    # n = pypsa.Network("dispatch_network.nc")
    # scenario_setup = load_scenario_definition(snakemake)
    # 
    # results = export_dispatch_network_to_sienna_csv(
    #     network=n,
    #     scenario_setup=scenario_setup,
    #     output_dir="./sienna_export/",
    #     year=2030
    # )
    # 
    # print("Export complete!")
    # print(f"Files created: {list(results.keys())}")