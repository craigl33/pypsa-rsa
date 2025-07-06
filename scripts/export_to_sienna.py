# -*- coding: utf-8 -*-
"""
PyPSA to PowerSystems.jl 4.6.2 Compatible CSV Export Implementation - IMPROVED VERSION

This module provides comprehensive functionality to export PyPSA dispatch networks 
to PowerSystems.jl-compatible CSV format for use with the Sienna ecosystem.

IMPROVEMENTS:
- Enhanced validation to prevent downstream PowerSystems.jl errors
- Intelligent fuel/prime mover mapping with fallbacks
- Technology-specific operational parameters
- Comprehensive error detection and reporting
- No more silent 'OTHER' mappings that cause solver failures
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
    
    IMPROVED VERSION with enhanced validation and error handling.
    """
    
    def __init__(self, network: pypsa.Network, scenario_setup: dict):
        """Initialize the exporter with enhanced validation."""
        self.network = network
        self.scenario_setup = scenario_setup
        self.base_power = 100.0  # MVA base for PowerSystems.jl
        
        # Time series storage
        self.time_series_data = {}
        self.time_series_metadata = []
        
        # Load additional generator data
        self.generator_details = self._load_generator_details()
        
        # Analyze network structure
        self._analyze_network()
    
    def _load_generator_details(self) -> Dict[str, pd.DataFrame]:
        """Load detailed generator data from scenario Excel files."""
        generator_details = {}
        
        try:
            # Load fixed technologies data
            fixed_tech_file = os.path.join(self.scenario_setup.get("sub_path", ""), "fixed_technologies.xlsx")
            if os.path.exists(fixed_tech_file):
                try:
                    conv_gens = pd.read_excel(fixed_tech_file, sheet_name="conventional", index_col=0)
                    generator_details['conventional'] = conv_gens
                    logger.info(f"Loaded {len(conv_gens)} conventional generator details")
                except Exception as e:
                    logger.warning(f"Could not load conventional generators: {e}")
                
                try:
                    renew_gens = pd.read_excel(fixed_tech_file, sheet_name="renewables", index_col=0)
                    generator_details['renewables'] = renew_gens
                    logger.info(f"Loaded {len(renew_gens)} renewable generator details")
                except Exception as e:
                    logger.warning(f"Could not load renewable generators: {e}")
            
            # Load extendable technologies data
            ext_tech_file = os.path.join(self.scenario_setup.get("sub_path", ""), "extendable_technologies.xlsx")
            if os.path.exists(ext_tech_file):
                try:
                    ext_params = pd.read_excel(ext_tech_file, sheet_name="parameters", index_col=[0, 1, 2])
                    generator_details['extendable_params'] = ext_params
                    logger.info(f"Loaded extendable technology parameters")
                except Exception as e:
                    logger.warning(f"Could not load extendable parameters: {e}")
        
        except Exception as e:
            logger.warning(f"Could not load generator detail files: {e}")
        
        return generator_details
    
    def _get_generator_detail(self, gen_name: str, field: str, default_value=None):
        """Retrieve detailed information about a specific generator."""
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
            
            # Check extendable parameters
            if 'extendable_params' in self.generator_details:
                ext_data = self.generator_details['extendable_params']
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
        """Analyze the PyPSA network structure."""
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
        
        # Analyze time series data
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
        """Main export function with enhanced validation."""
        logger.info("Starting PyPSA to PowerSystems.jl CSV export...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        files_created = {}
        
        # Export static components
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
        """Export all static component data with improved AC/DC link handling."""
        files_created = {}
        
        # Export buses first
        if 'Bus' in self.network_summary:
            bus_file = self._export_buses(output_path)
            files_created.update(bus_file)
        
        # Export generators (IMPROVED VERSION)
        if 'Generator' in self.network_summary:
            gen_files = self._export_generators_unified(output_path)
            files_created.update(gen_files)
        
        # Export loads
        if 'Load' in self.network_summary:
            load_files = self._export_loads(output_path)
            files_created.update(load_files)
        
        # Export AC branches (including AC links) - UPDATED
        branch_files = self._export_branches(output_path)
        files_created.update(branch_files)
        
        # Export DC branches (only true DC links) - UPDATED  
        if 'Link' in self.network_summary:
            dc_files = self._export_dc_branches(output_path)
            files_created.update(dc_files)
        
        # Export storage
        if 'StorageUnit' in self.network_summary:
            storage_files = self._export_storage(output_path)
            files_created.update(storage_files)
        
        return files_created
    
    def _export_buses(self, output_path: Path) -> Dict[str, str]:
        """Export bus data in PowerSystems.jl format."""
        buses_df = self.network.buses.copy()
        
        if buses_df.empty:
            logger.warning("No buses to export")
            return {}
        
        sienna_buses = pd.DataFrame(index=buses_df.index)
        sienna_buses['name'] = buses_df.index
        sienna_buses['base_voltage'] = buses_df['v_nom']
        sienna_buses['bus_type'] = 'PQ'
        sienna_buses['voltage'] = 1.0
        sienna_buses['angle'] = 0.0
        
        if 'x' in buses_df.columns and 'y' in buses_df.columns:
            sienna_buses['longitude'] = buses_df['x']
            sienna_buses['latitude'] = buses_df['y']
        else:
            sienna_buses['longitude'] = 0.0
            sienna_buses['latitude'] = 0.0
        
        self._set_bus_types(sienna_buses)
        
        if 'carrier' in buses_df.columns:
            sienna_buses['area'] = buses_df['carrier']
            sienna_buses['zone'] = buses_df['carrier']
        else:
            sienna_buses['area'] = 1
            sienna_buses['zone'] = 1
        
        bus_file = output_path / 'bus.csv'
        sienna_buses.to_csv(bus_file, index=False)
        
        logger.info(f"Exported {len(sienna_buses)} buses to {bus_file}")
        return {'bus': str(bus_file)}
    
    def _set_bus_types(self, sienna_buses: pd.DataFrame):
        """Set appropriate bus types for PowerSystems.jl."""
        sienna_buses['bus_type'] = 'PQ'
        
        if hasattr(self.network, 'generators') and not self.network.generators.empty:
            gen_capacity_by_bus = self.network.generators.groupby('bus')['p_nom'].sum()
            slack_bus = gen_capacity_by_bus.idxmax()
            sienna_buses.loc[sienna_buses['name'] == slack_bus, 'bus_type'] = 'REF'
        else:
            sienna_buses.iloc[0, sienna_buses.columns.get_loc('bus_type')] = 'REF'
        
        if hasattr(self.network, 'generators') and not self.network.generators.empty:
            gen_buses = set(self.network.generators['bus'].unique())
            slack_bus_name = sienna_buses[sienna_buses['bus_type'] == 'REF']['name'].iloc[0]
            pv_buses = gen_buses - {slack_bus_name}
            
            for bus in pv_buses:
                sienna_buses.loc[sienna_buses['name'] == bus, 'bus_type'] = 'PV'

    def _export_generators_unified(self, output_path: Path) -> Dict[str, str]:
        """
        IMPROVED: Export ALL generators with enhanced validation and error handling.
        """
        generators_df = self.network.generators.copy()
        
        if generators_df.empty:
            logger.warning("No generators to export")
            return {}
        
        # ENHANCED VALIDATION - Check required fields first
        required_fields = ['bus', 'p_nom', 'carrier']
        missing_fields = set(required_fields) - set(generators_df.columns)
        if missing_fields:
            raise ValueError(f"Generators missing critical fields: {missing_fields}")
        
        # Check for invalid data
        invalid_p_nom = generators_df['p_nom'].isna() | (generators_df['p_nom'] <= 0)
        if invalid_p_nom.any():
            invalid_gens = generators_df[invalid_p_nom].index.tolist()
            logger.error(f"Generators with invalid p_nom values: {invalid_gens}")
            # Remove invalid generators rather than filling with defaults
            generators_df = generators_df[~invalid_p_nom]
            logger.warning(f"Removed {invalid_p_nom.sum()} generators with invalid capacity (zero or less)")
        
        # Check for missing bus references
        all_buses = set(self.network.buses.index)
        invalid_buses = ~generators_df['bus'].isin(all_buses)
        if invalid_buses.any():
            invalid_gens = generators_df[invalid_buses].index.tolist()
            raise ValueError(f"Generators reference non-existent buses: {invalid_gens}")
        
        # Create DataFrame with proper construction
        sienna_gen = pd.DataFrame(index=generators_df.index)
        
        # Core required fields with validation
        sienna_gen['name'] = generators_df.index.values
        sienna_gen['bus'] = generators_df['bus'].values
        
        # Power limits with validation
        p_nom = generators_df['p_nom'].values
        sienna_gen['active_power'] = p_nom
        sienna_gen['max_active_power'] = p_nom
        
        # Calculate minimum power with proper validation
        p_min_pu = generators_df.get('p_min_pu', pd.Series(0.0, index=generators_df.index))
        p_min_pu = p_min_pu.clip(0.0, 1.0)  # Ensure valid range
        
        # Handle time series minimum power if available
        if hasattr(self.network, 'generators_t') and hasattr(self.network.generators_t, 'p_min_pu'):
            ts_p_min = self.network.generators_t.p_min_pu
            for gen in generators_df.index:
                if gen in ts_p_min.columns and not ts_p_min[gen].empty:
                    mean_p_min = ts_p_min[gen].mean()
                    if pd.notna(mean_p_min) and 0 <= mean_p_min <= 1:
                        p_min_pu.loc[gen] = mean_p_min
        
        sienna_gen['min_active_power'] = (p_nom * p_min_pu.values).clip(0, p_nom)
        
        # Reactive power limits with realistic defaults
        power_factor = 0.9  # More realistic than 0.3
        reactive_capacity = p_nom * np.sqrt(1 - power_factor**2) / power_factor
        sienna_gen['max_reactive_power'] = reactive_capacity
        sienna_gen['min_reactive_power'] = -reactive_capacity
        
        # Cost information with validation
        marginal_cost = generators_df.get('marginal_cost', pd.Series(0.0, index=generators_df.index))
        marginal_cost = marginal_cost.fillna(0.0)
        negative_costs = marginal_cost < 0
        if negative_costs.any():
            logger.warning(f"Found {negative_costs.sum()} generators with negative marginal costs - setting to 0")
            marginal_cost[negative_costs] = 0.0
        
        sienna_gen['variable'] = marginal_cost.values
        
        # ENHANCED fuel and prime mover mapping
        fuel_mapping = self._get_fuel_mapping_with_validation(generators_df)
        prime_mover_mapping = self._get_prime_mover_mapping_with_validation(generators_df)
        
        # Apply mappings with validation
        fuel_values = []
        type_values = []
        startup_costs = []
        
        for gen in generators_df.index:
            carrier = generators_df.loc[gen, 'carrier'].lower()
            
            fuel = fuel_mapping.get(carrier, 'NATURAL_GAS')  # Conservative fallback
            prime_mover = prime_mover_mapping.get(carrier, 'ST')  # Conservative fallback
            
            fuel_values.append(fuel)
            type_values.append(prime_mover)
            
            # Set reasonable startup costs based on technology
            if fuel in ['WIND', 'SOLAR']:
                startup_cost = 0.0  # Renewables have no startup cost
            elif prime_mover == 'CC':
                startup_cost = 50.0  # Combined cycle
            elif prime_mover == 'CT':
                startup_cost = 25.0  # Combustion turbine
            elif prime_mover == 'ST':
                startup_cost = 100.0  # Steam turbine
            else:
                startup_cost = 30.0  # Default
            
            startup_costs.append(startup_cost)
        
        sienna_gen['fuel'] = fuel_values
        sienna_gen['type'] = type_values
        sienna_gen['startup'] = startup_costs
        sienna_gen['shutdown'] = [cost * 0.5 for cost in startup_costs]
        
        # Technology-specific operational parameters
        ramp_rates_30 = []
        ramp_rates_10 = []
        min_up_times = []
        min_down_times = []
        
        for i, (fuel, prime_mover) in enumerate(zip(fuel_values, type_values)):
            p_nom_gen = p_nom[i]
            
            if fuel in ['WIND', 'SOLAR', 'HYDRO']:
                # Renewables: fast ramping
                ramp_30 = p_nom_gen
                ramp_10 = p_nom_gen
                min_up = 0.0
                min_down = 0.0
            elif prime_mover == 'CT':
                # Gas turbines: medium ramping
                ramp_30 = p_nom_gen * 0.8
                ramp_10 = p_nom_gen * 0.3
                min_up = 1.0
                min_down = 1.0
            elif prime_mover == 'CC':
                # Combined cycle: slower ramping
                ramp_30 = p_nom_gen * 0.5
                ramp_10 = p_nom_gen * 0.2
                min_up = 2.0
                min_down = 2.0
            elif fuel == 'COAL':
                # Coal: slow ramping
                ramp_30 = p_nom_gen * 0.3
                ramp_10 = p_nom_gen * 0.1
                min_up = 4.0
                min_down = 4.0
            elif fuel == 'NUCLEAR':
                # Nuclear: very slow ramping
                ramp_30 = p_nom_gen * 0.1
                ramp_10 = p_nom_gen * 0.05
                min_up = 24.0
                min_down = 24.0
            else:
                # Default values
                ramp_30 = p_nom_gen * 0.5
                ramp_10 = p_nom_gen * 0.2
                min_up = 1.0
                min_down = 1.0
            
            ramp_rates_30.append(ramp_30)
            ramp_rates_10.append(ramp_10)
            min_up_times.append(min_up)
            min_down_times.append(min_down)
        
        sienna_gen['ramp_30'] = ramp_rates_30
        sienna_gen['ramp_10'] = ramp_rates_10
        sienna_gen['min_up_time'] = min_up_times
        sienna_gen['min_down_time'] = min_down_times
        
        # Status and availability
        sienna_gen['available'] = True
        sienna_gen['status'] = 1
        
        # FINAL VALIDATION - check for any remaining NaN values
        nan_columns = sienna_gen.columns[sienna_gen.isna().any()].tolist()
        if nan_columns:
            logger.error(f"Found NaN values in columns: {nan_columns}")
            for col in nan_columns:
                nan_count = sienna_gen[col].isna().sum()
                logger.error(f"  {col}: {nan_count} NaN values")
            raise ValueError("Cannot export generators with NaN values to PowerSystems.jl")
        
        # Export to CSV
        gen_file = output_path / 'gen.csv'
        sienna_gen.to_csv(gen_file, index=False)
        
        logger.info(f"Exported {len(sienna_gen)} generators to {gen_file}")
        
        # Log technology summary
        tech_summary = pd.DataFrame({
            'fuel': sienna_gen['fuel'],
            'type': sienna_gen['type'],
            'capacity_MW': sienna_gen['max_active_power']
        }).groupby(['fuel', 'type'])['capacity_MW'].sum().round(1)
        
        logger.info("Technology summary:")
        for (fuel, ptype), capacity in tech_summary.items():
            logger.info(f"  {fuel} {ptype}: {capacity} MW")
        
        return {'gen': str(gen_file)}
    
    def _get_fuel_mapping_with_validation(self, generators_df: pd.DataFrame) -> Dict[str, str]:
        """
        IMPROVED: Enhanced fuel mapping with validation and user feedback.
        """
        base_mapping = {
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
            'rmippp': 'BIOMASS',
            'bioenergy': 'BIOMASS',
            'hydro_import': 'HYDRO', 
            'sasol_coal': 'COAL',
            'sasol_gas':'NATURAL_GAS',
            'ccgt_steam':'NATURAL_GAS',
            'solar_pv_low': 'SOLAR',
            'solar_pv_rooftop':'SOLAR',
            'solar_csp': 'SOLAR'

        }
        
        # Check for unmapped carriers
        unique_carriers = set(generators_df['carrier'].str.lower().unique())
        unmapped_carriers = unique_carriers - set(base_mapping.keys())
        
        if unmapped_carriers:
            logger.warning(f"Found unmapped generator carriers: {unmapped_carriers}")
            
            # Provide intelligent suggestions
            for carrier in unmapped_carriers:
                suggestion = self._suggest_fuel_mapping(carrier)
                base_mapping[carrier] = suggestion
                logger.warning(f"  '{carrier}' -> mapping to '{suggestion}' (auto-suggested)")
        
        return base_mapping

    def _suggest_fuel_mapping(self, carrier: str) -> str:
        """
        IMPROVED: Intelligently suggest fuel mapping based on carrier name patterns.
        """
        carrier_lower = carrier.lower()
        
        if any(term in carrier_lower for term in ['gas', 'ccgt', 'ocgt', 'lng']):
            return 'NATURAL_GAS'
        elif any(term in carrier_lower for term in ['coal', 'lignite']):
            return 'COAL'
        elif any(term in carrier_lower for term in ['wind', 'onshore', 'offshore']):
            return 'WIND'
        elif any(term in carrier_lower for term in ['solar', 'pv', 'photovoltaic']):
            return 'SOLAR'
        elif any(term in carrier_lower for term in ['hydro', 'water', 'ror']):
            return 'HYDRO'
        elif any(term in carrier_lower for term in ['bio', 'waste', 'wood']):
            return 'BIOMASS'
        elif any(term in carrier_lower for term in ['nuclear', 'uranium']):
            return 'NUCLEAR'
        elif any(term in carrier_lower for term in ['oil', 'diesel', 'liquid']):
            return 'DIESEL'
        elif any(term in carrier_lower for term in ['geo', 'thermal']):
            return 'GEOTHERMAL'
        else:
            # Conservative fallback - use most common thermal type
            logger.error(f"Cannot determine appropriate fuel type for carrier '{carrier}'. Using NATURAL_GAS as fallback.")
            return 'NATURAL_GAS'

    def _get_prime_mover_mapping_with_validation(self, generators_df: pd.DataFrame) -> Dict[str, str]:
        """
        IMPROVED: Enhanced prime mover mapping with validation.
        """
        base_mapping = {
            'gas': 'CC',
            'ccgt': 'CC',
            'ccgt_gas': 'CC',
            'ocgt': 'CT',
            'ocgt_gas': 'CT',
            'ocgt_diesel': 'CT',
            'coal': 'ST',
            'lignite': 'ST',
            'oil': 'IC',
            'nuclear': 'ST',
            'biomass': 'ST',
            'waste': 'ST',
            'wind': 'WT',
            'wind_onshore': 'WT',
            'wind_offshore': 'WT',
            'solar': 'PV',
            'solar_pv': 'PV',
            'pv': 'PV',
            'hydro': 'HY',
            'ror': 'HY',
            'geothermal': 'ST',
            'rmippp': 'ST',
            'bioenergy': 'ST',
            'hydro_import': 'HY', 
            'sasol_coal': 'ST',
            'sasol_gas':'CT',
            'ccgt_steam':'CC',
            'solar_pv_low': 'PV',
            'solar_pv_rooftop':'PV',
            'solar_csp': 'ST'
        }
        
        # Check for unmapped carriers
        unique_carriers = set(generators_df['carrier'].str.lower().unique())
        unmapped_carriers = unique_carriers - set(base_mapping.keys())
        
        if unmapped_carriers:
            logger.warning(f"Found unmapped prime mover carriers: {unmapped_carriers}")
            for carrier in unmapped_carriers:
                suggestion = self._suggest_prime_mover_mapping(carrier)
                base_mapping[carrier] = suggestion
                logger.warning(f"  '{carrier}' -> mapping to '{suggestion}' (auto-suggested)")
        
        return base_mapping

    def _suggest_prime_mover_mapping(self, carrier: str) -> str:
        """Suggest prime mover mapping based on carrier patterns."""
        carrier_lower = carrier.lower()
        
        if any(term in carrier_lower for term in ['ccgt', 'combined']):
            return 'CC'
        elif any(term in carrier_lower for term in ['ocgt', 'turbine', 'gas']):
            return 'CT'
        elif any(term in carrier_lower for term in ['wind']):
            return 'WT'
        elif any(term in carrier_lower for term in ['solar', 'pv']):
            return 'PV'
        elif any(term in carrier_lower for term in ['hydro']):
            return 'HY'
        else:
            return 'ST'  # Default to steam turbine

    def _export_loads(self, output_path: Path) -> Dict[str, str]:
        """Export electrical loads in PowerSystems.jl format."""
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
        sienna_loads['max_reactive_power'] = max_active_power * 0.3
        sienna_loads['available'] = True
        sienna_loads['status'] = 1
        
        load_file = output_path / 'load.csv'
        sienna_loads.to_csv(load_file, index=False)
        
        logger.info(f"Exported {len(sienna_loads)} loads to {load_file}")
        return {'load': str(load_file)}
    
    # Replace these methods in your PyPSAToSiennaExporter class

    def _export_branches(self, output_path: Path) -> Dict[str, str]:
        """Export transmission lines, transformers, and AC links as AC branches."""
        branches = []
        
        # Add existing lines
        if hasattr(self.network, 'lines') and not self.network.lines.empty:
            lines_df = self.network.lines.copy()
            lines_df['component_type'] = 'Line'
            branches.append(lines_df)
        
        # Add transformers
        if hasattr(self.network, 'transformers') and not self.network.transformers.empty:
            transformers_df = self.network.transformers.copy()
            transformers_df['component_type'] = 'Transformer'
            branches.append(transformers_df)
        
        # Add AC links (filter out DC links)
        if hasattr(self.network, 'links') and not self.network.links.empty:
            ac_links = self._identify_ac_links()
            if not ac_links.empty:
                ac_links_df = ac_links.copy()
                ac_links_df['component_type'] = 'AC_Link'
                # Convert AC links to branch format
                ac_links_df = self._convert_ac_links_to_branches(ac_links_df)
                branches.append(ac_links_df)
        
        if not branches:
            logger.info("No AC branches to export")
            return {}
        
        all_branches = pd.concat(branches, ignore_index=False)
        
        sienna_branches = pd.DataFrame(index=all_branches.index)
        sienna_branches['name'] = all_branches.index
        sienna_branches['connection_points_from'] = all_branches['bus0']
        sienna_branches['connection_points_to'] = all_branches['bus1']
        
        # Handle different component types - improved parameter handling
        sienna_branches['r'] = all_branches.get('r', 0.01)  # Default small resistance for AC links
        sienna_branches['x'] = all_branches.get('x', 0.1)   # Default reactance for AC links
        sienna_branches['b'] = all_branches.get('b', 0.0)
        
        # For capacity limits - handle both s_nom and p_nom
        if 's_nom' in all_branches.columns:
            sienna_branches['rate'] = all_branches['s_nom']
        elif 'p_nom' in all_branches.columns:
            sienna_branches['rate'] = all_branches['p_nom']
        else:
            sienna_branches['rate'] = 1000.0  # Default rating
        
        sienna_branches['tap'] = all_branches.get('tap_ratio', 1.0)
        sienna_branches['shift'] = all_branches.get('phase_shift', 0.0)
        sienna_branches['available'] = True
        sienna_branches['status'] = 1
        
        branch_file = output_path / 'branch.csv'
        sienna_branches.to_csv(branch_file, index=False)
        
        # Log summary of what was exported
        component_summary = all_branches['component_type'].value_counts().to_dict()
        logger.info(f"Exported {len(sienna_branches)} AC branches to {branch_file}")
        logger.info(f"AC Branch breakdown: {component_summary}")
        
        return {'branch': str(branch_file)}

    def _identify_ac_links(self) -> pd.DataFrame:
        """Identify which links should be treated as AC transmission corridors."""
        links_df = self.network.links.copy()
        
        if links_df.empty:
            return pd.DataFrame()
        
        # Method 1: Check for AC/DC indicators in the carrier or name
        ac_indicators = ['ac', 'transmission', 'line', 'corridor', 'interconnector', 'interconnection']
        dc_indicators = ['dc', 'hvdc', 'converter', 'cable', 'subsea']
        
        # Default assumption: if efficiency is very high (>0.98), likely AC
        # If efficiency is lower, likely DC with converter losses
        is_ac_link = links_df.get('efficiency', 1.0) > 0.98
        
        # Override based on carrier information
        if 'carrier' in links_df.columns:
            for idx, carrier in links_df['carrier'].items():
                if pd.isna(carrier):
                    continue
                carrier_lower = str(carrier).lower()
                if any(indicator in carrier_lower for indicator in dc_indicators):
                    is_ac_link.loc[idx] = False
                    logger.debug(f"Link {idx}: carrier '{carrier}' -> DC")
                elif any(indicator in carrier_lower for indicator in ac_indicators):
                    is_ac_link.loc[idx] = True
                    logger.debug(f"Link {idx}: carrier '{carrier}' -> AC")
        
        # Override based on link name
        for idx in links_df.index:
            name_lower = str(idx).lower()
            if any(indicator in name_lower for indicator in dc_indicators):
                is_ac_link.loc[idx] = False
                logger.debug(f"Link {idx}: name -> DC")
            elif any(indicator in name_lower for indicator in ac_indicators):
                is_ac_link.loc[idx] = True
                logger.debug(f"Link {idx}: name -> AC")
        
        ac_links = links_df[is_ac_link]
        dc_links = links_df[~is_ac_link]
        
        logger.info(f"Link classification: {len(ac_links)} AC links, {len(dc_links)} DC links")
        
        if len(ac_links) > 0:
            logger.info(f"AC links: {list(ac_links.index)}")
        if len(dc_links) > 0:
            logger.info(f"DC links: {list(dc_links.index)}")
        
        # Store DC links for separate export
        self._dc_links = dc_links
        
        return ac_links

    def _convert_ac_links_to_branches(self, ac_links_df: pd.DataFrame) -> pd.DataFrame:
        """Convert AC links to branch format with appropriate electrical parameters."""
        
        # Estimate electrical parameters for AC links
        # If not provided, use typical values based on capacity and assume overhead lines
        
        if 'length' in ac_links_df.columns:
            # Use length if available
            lengths = ac_links_df['length']
            logger.info("Using provided link lengths for electrical parameter estimation")
        else:
            # Estimate length from capacity (very rough approximation)
            # Assume larger capacity links are shorter high-voltage lines
            p_nom = ac_links_df.get('p_nom', 1000.0)
            # Rough heuristic: 50km for large links, 200km for smaller ones
            lengths = pd.Series(
                np.where(p_nom > 2000, 50.0, 200.0), 
                index=ac_links_df.index
            )
            logger.warning("No link lengths provided - using estimated lengths based on capacity")
        
        # Typical parameters per km for overhead transmission lines
        # These are rough estimates - you may want to adjust based on your system
        voltage_level = 400  # Assume 400kV for transmission corridors
        
        if voltage_level >= 400:
            r_per_km = 0.025e-3  # Resistance in pu/km (400kV+)
            x_per_km = 0.25e-3   # Reactance in pu/km
            b_per_km = 4.0e-6    # Susceptance in pu/km
        else:
            r_per_km = 0.05e-3   # Higher resistance for lower voltage
            x_per_km = 0.35e-3   # Higher reactance
            b_per_km = 3.0e-6    # Lower susceptance
        
        # Apply electrical parameters if not already present
        if 'r' not in ac_links_df.columns:
            ac_links_df['r'] = lengths * r_per_km
            logger.info("Estimated resistance values for AC links")
        
        if 'x' not in ac_links_df.columns:
            ac_links_df['x'] = lengths * x_per_km
            logger.info("Estimated reactance values for AC links")
        
        if 'b' not in ac_links_df.columns:
            ac_links_df['b'] = lengths * b_per_km
            logger.info("Estimated susceptance values for AC links")
        
        # Convert power rating to apparent power rating if needed
        if 's_nom' not in ac_links_df.columns and 'p_nom' in ac_links_df.columns:
            # Assume power factor of 0.95 for transmission lines
            power_factor = 0.95
            ac_links_df['s_nom'] = ac_links_df['p_nom'] / power_factor
            logger.info(f"Converted p_nom to s_nom using power factor {power_factor}")
        
        # Log parameter summary
        logger.info("AC link electrical parameters:")
        logger.info(f"  R range: {ac_links_df['r'].min():.6f} - {ac_links_df['r'].max():.6f} pu")
        logger.info(f"  X range: {ac_links_df['x'].min():.6f} - {ac_links_df['x'].max():.6f} pu")
        logger.info(f"  B range: {ac_links_df['b'].min():.6f} - {ac_links_df['b'].max():.6f} pu")
        
        return ac_links_df

    def _export_dc_branches(self, output_path: Path) -> Dict[str, str]:
        """Export only true DC transmission links."""
        
        # Use the DC links identified in _identify_ac_links
        # This method will be called after _export_branches, so _dc_links should exist
        if not hasattr(self, '_dc_links'):
            # Fallback: if _identify_ac_links wasn't called, treat all links as DC
            logger.warning("DC links not pre-identified, falling back to all links")
            if hasattr(self.network, 'links') and not self.network.links.empty:
                self._dc_links = self.network.links.copy()
            else:
                self._dc_links = pd.DataFrame()
        
        if self._dc_links.empty:
            logger.info("No DC branches to export")
            return {}
        
        links_df = self._dc_links
        
        sienna_dc = pd.DataFrame(index=links_df.index)
        sienna_dc['name'] = links_df.index
        sienna_dc['connection_points_from'] = links_df['bus0']
        sienna_dc['connection_points_to'] = links_df['bus1']
        sienna_dc['active_power_limits_from'] = links_df['p_nom']
        sienna_dc['active_power_limits_to'] = links_df['p_nom']
        
        # Calculate losses more carefully
        efficiency = links_df.get('efficiency', 1.0)
        # Ensure efficiency is between 0 and 1
        efficiency = efficiency.clip(0.0, 1.0)
        loss_factor = 1.0 - efficiency
        
        sienna_dc['loss'] = loss_factor
        sienna_dc['available'] = True
        sienna_dc['status'] = 1
        
        dc_file = output_path / 'dc_branch.csv'
        sienna_dc.to_csv(dc_file, index=False)
        
        logger.info(f"Exported {len(sienna_dc)} DC branches to {dc_file}")
        if len(sienna_dc) > 0:
            avg_loss = loss_factor.mean() * 100
            logger.info(f"Average DC link losses: {avg_loss:.1f}%")
        
        return {'dc_branch': str(dc_file)}

    # Also add this method to help with configuration
    def configure_link_classification(self, ac_link_names: List[str] = None, 
                                    dc_link_names: List[str] = None,
                                    efficiency_threshold: float = 0.98):
        """
        Configure link classification parameters.
        
        Parameters:
        -----------
        ac_link_names : List[str], optional
            Explicit list of link names to treat as AC
        dc_link_names : List[str], optional  
            Explicit list of link names to treat as DC
        efficiency_threshold : float, default 0.98
            Efficiency threshold above which links are considered AC
        """
        self.ac_link_names = set(ac_link_names or [])
        self.dc_link_names = set(dc_link_names or [])
        self.efficiency_threshold = efficiency_threshold
        
        logger.info(f"Link classification configured:")
        logger.info(f"  Explicit AC links: {len(self.ac_link_names)}")
        logger.info(f"  Explicit DC links: {len(self.dc_link_names)}")
        logger.info(f"  Efficiency threshold: {efficiency_threshold}")

    # Modified _identify_ac_links to use configuration
    def _identify_ac_links_with_config(self) -> pd.DataFrame:
        """Enhanced version that uses configuration if available."""
        links_df = self.network.links.copy()
        
        if links_df.empty:
            return pd.DataFrame()
        
        # Start with efficiency-based classification
        efficiency_threshold = getattr(self, 'efficiency_threshold', 0.98)
        is_ac_link = links_df.get('efficiency', 1.0) > efficiency_threshold
        
        # Apply explicit classifications if configured
        if hasattr(self, 'ac_link_names'):
            for link_name in self.ac_link_names:
                if link_name in links_df.index:
                    is_ac_link.loc[link_name] = True
                    logger.info(f"Explicitly classified {link_name} as AC")
        
        if hasattr(self, 'dc_link_names'):
            for link_name in self.dc_link_names:
                if link_name in links_df.index:
                    is_ac_link.loc[link_name] = False
                    logger.info(f"Explicitly classified {link_name} as DC")
        
        # Apply automatic classification for remaining links
        ac_indicators = ['ac', 'transmission', 'line', 'corridor', 'interconnector', 'interconnection']
        dc_indicators = ['dc', 'hvdc', 'converter', 'cable', 'subsea']
        
        # Check carrier information
        if 'carrier' in links_df.columns:
            for idx, carrier in links_df['carrier'].items():
                if pd.isna(carrier):
                    continue
                carrier_lower = str(carrier).lower()
                if any(indicator in carrier_lower for indicator in dc_indicators):
                    is_ac_link.loc[idx] = False
                elif any(indicator in carrier_lower for indicator in ac_indicators):
                    is_ac_link.loc[idx] = True
        
        # Check link names
        for idx in links_df.index:
            name_lower = str(idx).lower()
            if any(indicator in name_lower for indicator in dc_indicators):
                is_ac_link.loc[idx] = False
            elif any(indicator in name_lower for indicator in ac_indicators):
                is_ac_link.loc[idx] = True
        
        ac_links = links_df[is_ac_link]
        dc_links = links_df[~is_ac_link]
        
        logger.info(f"Final link classification: {len(ac_links)} AC links, {len(dc_links)} DC links")
        
        self._dc_links = dc_links
        return ac_links
    
    def _export_storage(self, output_path: Path) -> Dict[str, str]:
        """Export energy storage systems."""
        storage_df = self.network.storage_units.copy()
        
        if storage_df.empty:
            logger.info("No storage units to export")
            return {}
        
        sienna_storage = pd.DataFrame(index=storage_df.index)
        sienna_storage['name'] = storage_df.index
        sienna_storage['bus'] = storage_df['bus']
        
        max_hours = storage_df.get('max_hours', 6.0)
        energy_capacity = storage_df['p_nom'] * max_hours
        sienna_storage['energy_capacity'] = energy_capacity
        
        sienna_storage['input_active_power_limits'] = storage_df['p_nom']
        sienna_storage['output_active_power_limits'] = storage_df['p_nom']
        sienna_storage['efficiency_in'] = storage_df.get('efficiency_store', 0.95)
        sienna_storage['efficiency_out'] = storage_df.get('efficiency_dispatch', 0.95)
        
        initial_soc = storage_df.get('state_of_charge_initial', 0.5)
        sienna_storage['initial_energy'] = initial_soc * energy_capacity
        
        sienna_storage['available'] = True
        sienna_storage['status'] = 1
        
        storage_file = output_path / 'storage.csv'
        sienna_storage.to_csv(storage_file, index=False)
        
        logger.info(f"Exported {len(sienna_storage)} storage units to {storage_file}")
        return {'storage': str(storage_file)}
    
    def _export_time_series_data(self, output_path: Path) -> Dict[str, str]:
        """Export time series data."""
        if 'time_series' not in self.network_summary:
            logger.info("No time series data found")
            return {}
        
        ts_dir = output_path / 'timeseries_data'
        ts_dir.mkdir(exist_ok=True)
        
        files_created = {}
        
        if 'Load' in self.network_summary.get('time_series', {}):
            load_ts_files = self._export_load_time_series(ts_dir)
            files_created.update(load_ts_files)
        
        if 'Generator' in self.network_summary.get('time_series', {}):
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
        
        sienna_load_ts = load_ts.copy()
        
        if hasattr(sienna_load_ts.index, 'strftime'):
            sienna_load_ts.index = sienna_load_ts.index.strftime('%Y-%m-%dT%H:%M:%S')
        sienna_load_ts.index.name = 'DateTime'
        
        load_ts_file = ts_dir / 'load_timeseries.csv'
        sienna_load_ts.to_csv(load_ts_file)
        
        for load_name in sienna_load_ts.columns:
            self.time_series_metadata.append({
                'simulation': 'DA',
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
        """Export generator time series data."""
        files_created = {}
        
        if hasattr(self.network, 'generators_t') and hasattr(self.network.generators_t, 'p_max_pu'):
            gen_availability = self.network.generators_t.p_max_pu
            if not gen_availability.empty:
                renewable_file = self._export_renewable_availability(ts_dir, gen_availability)
                files_created.update(renewable_file)
        
        return files_created
    
    def _export_renewable_availability(self, ts_dir: Path, gen_availability: pd.DataFrame) -> Dict[str, str]:
        """Export renewable generator availability factors."""
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
        
        if hasattr(renewable_availability.index, 'strftime'):
            renewable_availability.index = renewable_availability.index.strftime('%Y-%m-%dT%H:%M:%S')
        renewable_availability.index.name = 'DateTime'
        
        gen_ts_file = ts_dir / 'renewable_availability.csv'
        renewable_availability.to_csv(gen_ts_file)
        
        for gen_name in renewable_availability.columns:
            self.time_series_metadata.append({
                'simulation': 'DA',
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
        """Create PowerSystems.jl configuration files."""
        files_created = {}
        
        descriptors_file = self._create_user_descriptors(output_path)
        files_created['user_descriptors'] = str(descriptors_file)
        
        if self.time_series_metadata:
            ts_metadata_file = self._create_timeseries_metadata(output_path)
            files_created['timeseries_metadata'] = str(ts_metadata_file)
        
        gen_mapping_file = self._create_generator_mapping(output_path)
        files_created['generator_mapping'] = str(gen_mapping_file)
        
        return files_created
        
    def _create_user_descriptors(self, output_path: Path) -> Path:
        """Create user_descriptors.yaml."""
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
        """Create timeseries_metadata.json."""
        ts_metadata_file = output_path / 'timeseries_metadata.json'
        with open(ts_metadata_file, 'w') as f:
            json.dump(self.time_series_metadata, f, indent=2)
        
        logger.info(f"Created time series metadata: {ts_metadata_file}")
        return ts_metadata_file
    
    def _create_generator_mapping(self, output_path: Path) -> Path:
        """Create generator_mapping.yaml."""
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
        """Create Julia script for importing data into PowerSystems.jl."""
        julia_code = f'''#!/usr/bin/env julia

"""
IMPROVED Julia script to import PyPSA-exported data into PowerSystems.jl 4.6.2

This script includes enhanced validation and error handling.
"""

using PowerSystems
using PowerSimulations
using Dates
using TimeSeries
using HiGHS

# Configuration
data_dir = "{output_path.absolute()}"
base_power = {self.base_power}
user_descriptors = joinpath(data_dir, "user_descriptors.yaml")

timeseries_metadata_file = joinpath(data_dir, "timeseries_metadata.json")
generator_mapping_file = joinpath(data_dir, "generator_mapping.yaml")

println("🔧 PowerSystems.jl 4.6.2 Data Import (IMPROVED VERSION)")
println("Loading PyPSA data from: ", data_dir)

# Enhanced import with better error handling
try
    println("\\n📊 Creating PowerSystemTableData...")
    
    has_timeseries = isfile(timeseries_metadata_file)
    has_gen_mapping = isfile(generator_mapping_file)
    
    if has_timeseries && has_gen_mapping
        println("   Using full configuration")
        data = PowerSystemTableData(
            data_dir,
            base_power,
            user_descriptors;
            timeseries_metadata_file = timeseries_metadata_file,
            generator_mapping_file = generator_mapping_file
        )
    elseif has_timeseries
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
    
    println("✅ PowerSystemTableData created successfully")
    
    # Create System
    println("\\n🏗️  Creating PowerSystems.jl System...")
    sys = System(data; time_series_in_memory = true)
    println("✅ PowerSystems.jl System created successfully")
    
    # Enhanced validation
    println("\\n📋 === Enhanced System Summary ===")
    println("Base Power: ", get_base_power(sys), " MVA")
    println("Buses: ", length(get_components(Bus, sys)))
    
    thermal_gens = get_components(ThermalStandard, sys)
    renewable_gens = get_components(RenewableDispatch, sys)
    
    println("Thermal Generators: ", length(thermal_gens))
    println("Renewable Generators: ", length(renewable_gens))
    println("Loads: ", length(get_components(ElectricLoad, sys)))
    println("Branches: ", length(get_components(ACBranch, sys)))
    
    # Validation checks
    println("\\n🔍 === Enhanced Validation ===")
    total_demand = sum(get_max_active_power(load) for load in get_components(ElectricLoad, sys))
    total_generation = sum(get_max_active_power(gen) for gen in get_components(Generator, sys))
    
    println("Total demand: ", round(total_demand, digits=1), " MW")
    println("Total generation capacity: ", round(total_generation, digits=1), " MW")
    
    if total_generation >= total_demand
        reserve_margin = (total_generation - total_demand) / total_demand * 100
        println("Reserve margin: ", round(reserve_margin, digits=1), "%")
        println("✅ System has adequate generation capacity")
    else
        println("⚠️  Warning: Generation capacity insufficient!")
    end
    
    # Check for any generators with invalid parameters
    println("\\n🔍 Checking generator parameters...")
    invalid_gens = []
    for gen in get_components(Generator, sys)
        if get_max_active_power(gen) <= 0
            push!(invalid_gens, get_name(gen))
        end
    end
    
    if !isempty(invalid_gens)
        println("❌ Found generators with invalid parameters: ", invalid_gens)
    else
        println("✅ All generators have valid parameters")
    end
    
    # Save system
    sys_file = joinpath(data_dir, "pypsa_system_improved.json")
    to_json(sys, sys_file)
    println("\\n💾 System saved to: ", sys_file)
    
    println("\\n🎉 === Import Complete (IMPROVED VERSION) ===")
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
        
        julia_file.chmod(0o755)
        
        logger.info(f"Created improved Julia import script: {julia_file}")
        return julia_file

def export_pypsa_to_sienna(network: pypsa.Network, 
                                    scenario_setup: dict,
                                    output_dir: str,
                                    include_time_series: bool = True,
                                    ac_link_names: List[str] = None,
                                    dc_link_names: List[str] = None,
                                    efficiency_threshold: float = 0.98) -> Dict[str, str]:
    """
    Enhanced export function with link classification configuration.
    
    Parameters:
    -----------
    network : pypsa.Network
        The PyPSA network to export
    scenario_setup : dict
        Scenario configuration
    output_dir : str
        Output directory for CSV files
    include_time_series : bool, default True
        Whether to export time series data
    ac_link_names : List[str], optional
        Explicit list of link names to treat as AC transmission corridors
    dc_link_names : List[str], optional
        Explicit list of link names to treat as DC links
    efficiency_threshold : float, default 0.98
        Efficiency threshold above which links are considered AC
    
    Returns:
    --------
    Dict[str, str]
        Dictionary of created files
    """
    
    # Enhanced validation before export
    validation = validate_pypsa_network(network)
    if not validation['is_valid']:
        logger.error("Network validation failed:")
        for issue in validation['issues']:
            logger.error(f"  {issue}")
        raise ValueError("Network validation failed - cannot export to PowerSystems.jl")
    
    if validation.get('warnings'):
        logger.warning("Network validation warnings:")
        for warning in validation['warnings']:
            logger.warning(f"  {warning}")
    
    # Create exporter and configure link classification
    exporter = PyPSAToSiennaExporter(network, scenario_setup)
    
    # Configure link classification if parameters provided
    if ac_link_names or dc_link_names or efficiency_threshold != 0.98:
        exporter.configure_link_classification(
            ac_link_names=ac_link_names,
            dc_link_names=dc_link_names, 
            efficiency_threshold=efficiency_threshold
        )
        # Use the configured version of the method
        exporter._identify_ac_links = exporter._identify_ac_links_with_config
    
    # Perform export
    export_results = exporter.export_to_csv(output_dir, include_time_series)
    
    # Add comprehensive summary
    export_results['export_summary'] = {
        'network_components': exporter.network_summary,
        'generator_details_loaded': bool(exporter.generator_details),
        'time_series_exported': include_time_series and len(exporter.time_series_metadata) > 0,
        'validation_status': validation,
        'powersystems_compatibility': 'PowerSystems.jl 4.6.2+',
        'ac_dc_classification': {
            'ac_links': len(getattr(exporter, '_dc_links', pd.DataFrame())) if hasattr(network, 'links') else 0,
            'dc_links': len(network.links) - len(getattr(exporter, '_dc_links', pd.DataFrame())) if hasattr(network, 'links') else 0,
            'classification_method': 'configured' if (ac_link_names or dc_link_names) else 'automatic'
        },
        'improvements_applied': [
            'Enhanced fuel/prime mover mapping',
            'Comprehensive data validation', 
            'Technology-specific parameters',
            'AC/DC link classification',
            'Proper PowerSystems.jl branch modeling'
        ]
    }
    
    return export_results
# def export_pypsa_to_sienna(network: pypsa.Network, 
#                           scenario_setup: dict,
#                           output_dir: str,
#                           include_time_series: bool = True) -> Dict[str, str]:
#     """
#     IMPROVED: Export PyPSA network to PowerSystems.jl with enhanced validation.
#     """
#     # Enhanced validation before export
#     validation = validate_pypsa_network(network)
#     if not validation['is_valid']:
#         logger.error("Network validation failed:")
#         for issue in validation['issues']:
#             logger.error(f"  {issue}")
#         raise ValueError("Network validation failed - cannot export to PowerSystems.jl")
    
#     if validation.get('warnings'):
#         logger.warning("Network validation warnings:")
#         for warning in validation['warnings']:
#             logger.warning(f"  {warning}")
    
#     # Create exporter and perform export
#     exporter = PyPSAToSiennaExporter(network, scenario_setup)
#     export_results = exporter.export_to_csv(output_dir, include_time_series)
    
#     # Add comprehensive summary
#     export_results['export_summary'] = {
#         'network_components': exporter.network_summary,
#         'generator_details_loaded': bool(exporter.generator_details),
#         'time_series_exported': include_time_series and len(exporter.time_series_metadata) > 0,
#         'validation_status': validation,
#         'powersystems_compatibility': 'PowerSystems.jl 4.6.2+',
#         'improvements_applied': [
#             'Enhanced fuel/prime mover mapping',
#             'Comprehensive data validation',
#             'Technology-specific parameters',
#             'NaN value detection and handling',
#             'Invalid data removal vs filling'
#         ]
#     }
    
#     export_results['import_instructions'] = [
#         "IMPROVED PowerSystems.jl 4.6.2 Import Instructions:",
#         "1. Navigate to the export directory",
#         "2. Ensure Julia packages: julia> using Pkg; Pkg.add([\"PowerSystems\", \"PowerSimulations\", \"HiGHS\"])",
#         "3. Run: julia import_to_powersystems.jl",
#         "4. System includes enhanced validation and error detection"
#     ]
    
#     return export_results

def validate_pypsa_network(network) -> Dict[str, Any]:
    """
    IMPROVED: Enhanced validation with specific checks for PowerSystems.jl compatibility.
    """
    issues = []
    warnings = []
    
    # Critical checks (will cause export failure)
    if network.buses.empty:
        issues.append("CRITICAL: No buses found in network")
    
    if network.generators.empty and network.loads.empty:
        issues.append("CRITICAL: No generators or loads found in network")
    
    # Generator data validation
    if not network.generators.empty:
        # Check for required fields
        required_fields = ['bus', 'p_nom', 'carrier']
        missing_fields = set(required_fields) - set(network.generators.columns)
        if missing_fields:
            issues.append(f"CRITICAL: Generators missing required fields: {missing_fields}")
        
        # Check for invalid capacity values
        invalid_capacity = network.generators['p_nom'].isna() | (network.generators['p_nom'] < 0)
        if invalid_capacity.any():
            invalid_count = invalid_capacity.sum()
            invalid_gens = network.generators[invalid_capacity].index.tolist()[:5]
            issues.append(f"CRITICAL: {invalid_count} generators with invalid capacity: {invalid_gens}...")
        
        # Check for missing marginal costs
        if 'marginal_cost' not in network.generators.columns:
            warnings.append("No marginal_cost column - will use default values")
        else:
            missing_costs = network.generators['marginal_cost'].isna().sum()
            if missing_costs > 0:
                warnings.append(f"{missing_costs} generators missing marginal costs")
        
        # Check carrier mapping
        unknown_carriers = []
        known_carriers = {
            'gas', 'ccgt', 'ocgt', 'coal', 'nuclear', 'wind', 'solar', 'hydro',
            'biomass', 'oil', 'diesel', 'rmippp', 'bioenergy'
        }
        
        for carrier in network.generators['carrier'].unique():
            if not any(known in carrier.lower() for known in known_carriers):
                unknown_carriers.append(carrier)
        
        if unknown_carriers:
            warnings.append(f"Unknown carriers (will auto-map): {unknown_carriers}")
    
    # Bus reference validation
    all_buses = set(network.buses.index)
    
    if not network.generators.empty:
        gen_buses = set(network.generators['bus'])
        missing_buses = gen_buses - all_buses
        if missing_buses:
            issues.append(f"CRITICAL: Generators reference missing buses: {missing_buses}")
    
    # Time series validation
    if hasattr(network, 'generators_t'):
        for attr_name in ['p_max_pu', 'p_min_pu']:
            if hasattr(network.generators_t, attr_name):
                ts_data = getattr(network.generators_t, attr_name)
                if not ts_data.empty:
                    missing_gens = set(ts_data.columns) - set(network.generators.index)
                    if missing_gens:
                        warnings.append(f"Time series {attr_name} references unknown generators: {len(missing_gens)} found")
    
    is_valid = len(issues) == 0
    return {
        'is_valid': is_valid, 
        'issues': issues,
        'warnings': warnings,
        'generator_count': len(network.generators),
        'bus_count': len(network.buses)
    }

def export_with_validation(network, scenario_setup, output_dir):
    """
    IMPROVED: Export with comprehensive validation and error handling.
    """
    # Enhanced validation first
    validation = validate_pypsa_network(network)
    
    if not validation['is_valid']:
        logger.error("Network validation failed:")
        for issue in validation['issues']:
            logger.error(f"  {issue}")
        raise ValueError("Network validation failed - cannot export to PowerSystems.jl")
    
    if validation.get('warnings'):
        logger.warning("Network validation warnings:")
        for warning in validation['warnings']:
            logger.warning(f"  {warning}")
    
    # Proceed with improved export
    return export_pypsa_to_sienna(network, scenario_setup, output_dir, include_time_series=True)

if __name__ == "__main__":
    """
    IMPROVED: Example usage with enhanced validation and error handling.
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
    export_folder = Path(snakemake.output.sienna_export_dir).expanduser()
    

    logger.info(f"Starting IMPROVED export to PowerSystems.jl 4.6.2 format...")
    logger.info(f"Network: {len(n.buses)} buses, {len(n.generators)} generators, {len(n.loads)} loads")
    
    # Enhanced validation before export
    logger.info("Running enhanced network validation...")
    validation = validate_pypsa_network(n)
    
    if not validation['is_valid']:
        logger.error("❌ Network validation failed:")
        for issue in validation['issues']:
            logger.error(f"  {issue}")
        raise SystemExit("Cannot proceed with export due to validation errors")
    
    if validation.get('warnings'):
        logger.warning("⚠️  Network validation warnings:")
        for warning in validation['warnings']:
            logger.warning(f"  {warning}")
    
    logger.info("✅ Network validation passed")
    
    # Perform improved export
    try:
        export_results = export_pypsa_to_sienna(
            network=n,
            scenario_setup=scenario_setup,
            output_dir=export_folder,
            include_time_series=True
        )
        
        # Display detailed results
        logger.info("=== IMPROVED Export Results ===")
        logger.info("Files created:")
        for file_type, file_path in export_results.items():
            if file_type not in ['import_instructions', 'export_summary']:
                logger.info(f"  ✓ {file_type}: {file_path}")
        
        if 'export_summary' in export_results:
            summary = export_results['export_summary']
            logger.info(f"Improvements applied: {summary.get('improvements_applied', [])}")
            logger.info(f"PowerSystems.jl compatibility: {summary['powersystems_compatibility']}")
        
        logger.info("✅ IMPROVED export completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Export failed: {e}")
        raise