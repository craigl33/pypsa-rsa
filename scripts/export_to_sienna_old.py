"""
Enhanced PyPSA to Sienna CSV Exporter - Core Functionality Development
Step-by-step implementation to make the class fully functional
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
from typing import Dict, List, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class PyPSAToSiennaCSVExporter:
    """
    Enhanced PyPSA to Sienna CSV Exporter with full functionality.
    
    This class systematically converts PyPSA networks to Sienna-compatible CSV format
    that PowerSystems.jl can import using its PowerSystemTableData functionality.
    """
    
    def __init__(self, network: pypsa.Network, scenario_setup: dict):
        """
        Initialize exporter with comprehensive validation and setup.
        
        Parameters:
        -----------
        network : pypsa.Network
            The PyPSA network to export (solved or unsolved)
        scenario_setup : dict
            Scenario configuration from load_scenario_definition()
        """
        # Store inputs
        self.network = network
        self.scenario_setup = scenario_setup
        self.base_power = 100.0  # MVA base for PowerSystems.jl
        
        # Analysis results storage
        self.component_inventory = {}
        self.time_series_inventory = {}
        self.constraint_inventory = {}
        self.export_summary = {}
        
        # Configuration
        self.component_mappings = self._initialize_component_mappings()
        self.unit_conversions = self._initialize_unit_conversions()
        self.field_mappings = self._initialize_field_mappings()
        
        # Status tracking
        self.is_analyzed = False
        self.is_validated = False
        self.export_ready = False
        
        # Automatically analyze the network
        self._analyze_network_comprehensive()
        
        logger.info(f"PyPSAToSiennaCSVExporter initialized for network with {len(self.network.buses)} buses")
    
    def _initialize_component_mappings(self) -> Dict[str, Dict[str, Any]]:
        """
        Complete mapping from PyPSA components to Sienna equivalents.
        
        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Comprehensive mapping configuration for each PyPSA component type
        """
        return {
            # Core network components
            'Bus': {
                'sienna_type': 'Bus',
                'required_fields': ['name', 'base_voltage'],
                'optional_fields': ['bus_type', 'area', 'zone', 'longitude', 'latitude', 
                                  'voltage_limits_min', 'voltage_limits_max'],
                'export_priority': 1,
                'split_by_carrier': False,
                'energy_component': False
            },
            
            'Carrier': {
                'sienna_type': 'Fuel',
                'required_fields': ['name'],
                'optional_fields': ['co2_emissions', 'fuel_cost', 'units'],
                'export_priority': 2,
                'split_by_carrier': False,
                'energy_component': False
            },
            
            # Generation components
            'Generator': {
                'sienna_type': 'ThermalStandard',  # Default type only, will be split by carrier
                'required_fields': ['name', 'bus', 'max_active_power'],
                'optional_fields': ['min_active_power', 'max_reactive_power', 'min_reactive_power',
                                  'ramp_up', 'ramp_down', 'start_up_cost', 'variable_cost',
                                  'min_up_time', 'min_down_time', 'fuel', 'prime_mover_type',
                                  'efficiency', 'marginal_cost'],
                'export_priority': 4,
                'split_by_carrier': True,
                'energy_component': True
            },
            
            # Load components
            'Load': {
                'sienna_type': 'PowerLoad',
                'required_fields': ['name', 'bus', 'max_active_power'],
                'optional_fields': ['max_reactive_power', 'power_factor'],
                'export_priority': 3,
                'split_by_carrier': False,
                'energy_component': True
            },
            
            # Transmission components
            'Line': {
                'sienna_type': 'Line',
                'required_fields': ['name', 'connection_points_from', 'connection_points_to', 'r', 'x'],
                'optional_fields': ['b', 'rate', 'angle_limits_min', 'angle_limits_max', 'length'],
                'export_priority': 5,
                'split_by_carrier': False,
                'energy_component': False
            },
            
            'Link': {
                'sienna_type': 'TwoTerminalHVDCLine',
                'required_fields': ['name', 'connection_points_from', 'connection_points_to'],
                'optional_fields': ['active_power_limits_from', 'active_power_limits_to', 'loss', 'efficiency'],
                'export_priority': 6,
                'split_by_carrier': False,
                'energy_component': False
            },
            
            # Storage components
            'StorageUnit': {
                'sienna_type': 'GenericBattery',
                'required_fields': ['name', 'bus', 'energy_capacity', 'input_active_power_limits', 'output_active_power_limits'],
                'optional_fields': ['efficiency_in', 'efficiency_out', 'state_of_charge_limits'],
                'export_priority': 7,
                'split_by_carrier': True,
                'energy_component': True
            },
            
            'Store': {
                'sienna_type': 'HydroEnergyReservoir',
                'required_fields': ['name', 'bus', 'storage_capacity'],
                'optional_fields': ['inflow', 'initial_storage'],
                'export_priority': 7,
                'split_by_carrier': False,
                'energy_component': True
            },
            
            # Constraints and metadata
            'GlobalConstraint': {
                'sienna_type': 'GlobalConstraint',
                'required_fields': ['name', 'type', 'sense'],
                'optional_fields': ['constant', 'carrier_attribute'],
                'export_priority': 10,
                'split_by_carrier': False,
                'energy_component': False
            }
        }
    
    def _initialize_unit_conversions(self) -> Dict[str, Dict[str, float]]:
        """
        Initialize unit conversion factors for different quantities.
        
        Returns:
        --------
        Dict[str, Dict[str, float]]
            Unit conversion factors
        """
        return {
            'power': {
                'MW_to_MW': 1.0,
                'GW_to_MW': 1000.0,
                'kW_to_MW': 0.001,
                'W_to_MW': 1e-6
            },
            'energy': {
                'MWh_to_MWh': 1.0,
                'GWh_to_MWh': 1000.0,
                'kWh_to_MWh': 0.001,
                'Wh_to_MWh': 1e-6
            },
            'cost': {
                'EUR_to_USD': 1.1,  # Approximate, should be configurable
                'ZAR_to_USD': 0.055,  # Approximate, should be configurable
                'per_MWh_to_per_MW': 1.0  # This depends on time resolution
            },
            'voltage': {
                'kV_to_kV': 1.0,
                'V_to_kV': 0.001
            }
        }
    
    def _initialize_field_mappings(self) -> Dict[str, Dict[str, str]]:
        """
        Initialize field name mappings from PyPSA to Sienna.
        
        Returns:
        --------
        Dict[str, Dict[str, str]]
            Field name mappings for each component type
        """
        return {
            'Bus': {
                # PyPSA field -> Sienna field
                'v_nom': 'base_voltage',
                'x': 'longitude',
                'y': 'latitude',
                'carrier': 'area'  # Map carrier to area if needed
            },
            'Generator': {
                'p_nom_max': 'max_active_power',
                'p_nom_min': 'min_active_power', 
                'p_nom': 'max_active_power',  # For fixed generators
                'carrier': 'fuel',
                'efficiency': 'efficiency',
                'marginal_cost': 'variable_cost',
                'capital_cost': 'investment_cost',
                'ramp_limit_up': 'ramp_up',
                'ramp_limit_down': 'ramp_down',
                'min_up_time': 'min_up_time',
                'min_down_time': 'min_down_time',
                'start_up_cost': 'start_up_cost'
            },
            'Load': {
                'p_set': 'max_active_power',  # Peak load value
                'q_set': 'max_reactive_power'
            },
            'Line': {
                'bus0': 'connection_points_from',
                'bus1': 'connection_points_to',
                's_nom_max': 'rate',
                's_nom': 'rate'
            },
            'Link': {
                'bus0': 'connection_points_from',
                'bus1': 'connection_points_to',
                'p_nom_max': 'active_power_limits_from',
                'p_nom': 'active_power_limits_from',
                'efficiency': 'efficiency'
            },
            'StorageUnit': {
                'p_nom_max': 'input_active_power_limits',
                'p_nom': 'output_active_power_limits',
                'max_hours': 'energy_capacity_multiplier',
                'efficiency_store': 'efficiency_in',
                'efficiency_dispatch': 'efficiency_out'
            }
        }
    
    def _analyze_network_comprehensive(self):
        """
        Comprehensive analysis of the PyPSA network structure and data.
        
        This function systematically inventories all components, time series,
        and constraints in the network.
        """
        logger.info("Starting comprehensive network analysis...")
        
        # Reset analysis results
        self.component_inventory = {}
        self.time_series_inventory = {}
        self.constraint_inventory = {}
        
        # Analyze static components
        self._analyze_static_components()
        
        # Analyze time series data
        self._analyze_time_series_data()
        
        # Analyze constraints
        self._analyze_constraints()
        
        # Analyze network topology
        self._analyze_network_topology()
        
        # Validate for Sienna compatibility
        self._validate_sienna_compatibility()
        
        # Generate summary
        self._generate_analysis_summary()
        
        self.is_analyzed = True
        logger.info("Network analysis complete")
    
    def _analyze_static_components(self):
        """Analyze all static components in the network."""
        logger.info("Analyzing static components...")
        
        for component in self.network.iterate_components():
            component_name = component.name
            component_df = component.df
            
            if not component_df.empty:
                # Basic component information
                component_info = {
                    'count': len(component_df),
                    'columns': list(component_df.columns),
                    'has_data': True,
                    'dtypes': component_df.dtypes.to_dict(),
                    'sample_data': component_df.head(2).to_dict() if len(component_df) > 0 else {},
                    'missing_data': component_df.isnull().sum().to_dict(),
                    'value_ranges': {}
                }
                
                # Analyze numeric columns for ranges and validity
                numeric_cols = component_df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    series = component_df[col]
                    if not series.empty:
                        component_info['value_ranges'][col] = {
                            'min': float(series.min()),
                            'max': float(series.max()),
                            'mean': float(series.mean()),
                            'has_negative': bool((series < 0).any()),
                            'has_zero': bool((series == 0).any()),
                            'has_inf': bool(np.isinf(series).any()),
                            'has_nan': bool(series.isnull().any())
                        }
                
                # Check Sienna mapping compatibility
                if component_name in self.component_mappings:
                    mapping = self.component_mappings[component_name]
                    component_info['sienna_compatible'] = self._check_component_sienna_compatibility(component_df, mapping)
                else:
                    component_info['sienna_compatible'] = {
                        'has_mapping': False,
                        'missing_required_fields': [],
                        'available_optional_fields': []
                    }
                
                self.component_inventory[component_name] = component_info
                logger.info(f"  Analyzed {component_name}: {len(component_df)} components")
            else:
                self.component_inventory[component_name] = {
                    'count': 0,
                    'has_data': False,
                    'sienna_compatible': {'has_mapping': False}
                }
    
    def _check_component_sienna_compatibility(self, component_df: pd.DataFrame, mapping: Dict[str, Any]) -> Dict[str, Any]:
        """Check if a component is compatible with Sienna requirements."""
        compatibility = {
            'has_mapping': True,
            'missing_required_fields': [],
            'available_optional_fields': [],
            'field_mapping_issues': [],
            'data_quality_issues': []
        }
        
        # Get field mappings for this component type
        component_type = None
        for comp_type, comp_mapping in self.component_mappings.items():
            if comp_mapping == mapping:
                component_type = comp_type
                break
        
        field_mapping = self.field_mappings.get(component_type, {})
        
        # Check required fields
        required_fields = mapping.get('required_fields', [])
        for field in required_fields:
            # Check if field exists directly or can be mapped
            pypsa_field = None
            for pypsa_col, sienna_col in field_mapping.items():
                if sienna_col == field:
                    pypsa_field = pypsa_col
                    break
            
            if field == 'name':
                # Name is always the index in PyPSA
                continue
            elif pypsa_field and pypsa_field in component_df.columns:
                # Field can be mapped
                continue
            elif field in component_df.columns:
                # Field exists directly
                continue
            else:
                compatibility['missing_required_fields'].append(field)
        
        # Check optional fields
        optional_fields = mapping.get('optional_fields', [])
        for field in optional_fields:
            pypsa_field = None
            for pypsa_col, sienna_col in field_mapping.items():
                if sienna_col == field:
                    pypsa_field = pypsa_col
                    break
            
            if pypsa_field and pypsa_field in component_df.columns:
                compatibility['available_optional_fields'].append(field)
            elif field in component_df.columns:
                compatibility['available_optional_fields'].append(field)
        
        # Check for data quality issues
        for col in component_df.select_dtypes(include=[np.number]).columns:
            series = component_df[col]
            if series.isnull().any():
                compatibility['data_quality_issues'].append(f"NaN values in {col}")
            if np.isinf(series).any():
                compatibility['data_quality_issues'].append(f"Infinite values in {col}")
        
        return compatibility
    
    def _analyze_time_series_data(self):
        """Analyze time-varying data for all components."""
        logger.info("Analyzing time series data...")
        
        for component in self.network.iterate_components():
            component_name = component.name
            
            # Check for time-varying attributes
            if hasattr(component, 'pnl'):
                ts_data = {}
                for attr_name, attr_data in component.pnl.items():
                    if not attr_data.empty:
                        ts_info = {
                            'shape': attr_data.shape,
                            'columns': list(attr_data.columns),
                            'time_range': self._get_time_range(attr_data.index),
                            'has_data': True,
                            'data_quality': self._analyze_time_series_quality(attr_data),
                            'sienna_relevance': self._assess_sienna_time_series_relevance(component_name, attr_name)
                        }
                        ts_data[attr_name] = ts_info
                        logger.debug(f"  Found time series {component_name}.{attr_name}: {attr_data.shape}")
                
                if ts_data:
                    self.time_series_inventory[component_name] = ts_data
    
    def _get_time_range(self, time_index) -> Dict[str, str]:
        """Get formatted time range information."""
        if len(time_index) == 0:
            return {'start': None, 'end': None, 'periods': 0}
        
        if isinstance(time_index, pd.MultiIndex):
            # Handle multi-investment period case
            time_values = time_index.get_level_values(-1)  # Get the datetime level
            return {
                'start': str(time_values[0]),
                'end': str(time_values[-1]),
                'periods': len(time_index),
                'investment_periods': list(time_index.get_level_values(0).unique()) if len(time_index.names) > 1 else None
            }
        else:
            return {
                'start': str(time_index[0]),
                'end': str(time_index[-1]),
                'periods': len(time_index)
            }
    
    def _analyze_time_series_quality(self, ts_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze quality of time series data."""
        quality = {
            'has_nan': ts_data.isnull().any().any(),
            'has_inf': np.isinf(ts_data.select_dtypes(include=[np.number])).any().any(),
            'has_negative': (ts_data.select_dtypes(include=[np.number]) < 0).any().any(),
            'all_zero_columns': [],
            'constant_columns': [],
            'value_ranges': {}
        }
        
        # Check for problematic columns
        for col in ts_data.columns:
            series = ts_data[col]
            if pd.api.types.is_numeric_dtype(series):
                if series.nunique() == 1:
                    quality['constant_columns'].append(col)
                if (series == 0).all():
                    quality['all_zero_columns'].append(col)
                
                quality['value_ranges'][col] = {
                    'min': float(series.min()),
                    'max': float(series.max()),
                    'mean': float(series.mean())
                }
        
        return quality
    
    def _assess_sienna_time_series_relevance(self, component_name: str, attr_name: str) -> Dict[str, Any]:
        """Assess relevance of time series for Sienna export."""
        relevance = {
            'export_recommended': False,
            'sienna_equivalent': None,
            'export_type': 'time_series',
            'notes': []
        }
        
        # Define important time series for Sienna
        important_ts = {
            'Load': ['p_set', 'q_set'],
            'Generator': ['p_max_pu', 'p_min_pu', 'marginal_cost'],
            'StorageUnit': ['p_max_pu', 'p_min_pu', 'inflow', 'state_of_charge_set'],
            'Line': ['s_max_pu'],
            'Link': ['p_max_pu', 'p_min_pu']
        }
        
        if component_name in important_ts and attr_name in important_ts[component_name]:
            relevance['export_recommended'] = True
            relevance['sienna_equivalent'] = f"{component_name}_{attr_name}"
            
            if attr_name in ['p_set', 'q_set']:
                relevance['notes'].append("Load time series - essential for Sienna")
            elif attr_name in ['p_max_pu', 'p_min_pu']:
                relevance['notes'].append("Availability time series - important for renewables")
            elif attr_name == 'marginal_cost':
                relevance['notes'].append("Variable cost time series - for fuel price variations")
        else:
            relevance['notes'].append("Not typically exported to Sienna")
        
        return relevance
    
    def _analyze_constraints(self):
        """Analyze constraints in the network."""
        logger.info("Analyzing constraints...")
        
        # Global constraints
        if hasattr(self.network, 'global_constraints') and not self.network.global_constraints.empty:
            gc_data = {
                'count': len(self.network.global_constraints),
                'types': list(self.network.global_constraints['type'].unique()) if 'type' in self.network.global_constraints.columns else [],
                'carriers': list(self.network.global_constraints['carrier_attribute'].unique()) if 'carrier_attribute' in self.network.global_constraints.columns else [],
                'data': self.network.global_constraints.to_dict('records'),
                'sienna_compatibility': self._assess_constraint_sienna_compatibility()
            }
            self.constraint_inventory['GlobalConstraint'] = gc_data
            logger.info(f"  Found {len(self.network.global_constraints)} global constraints")
        
        # Check for optimization model constraints if solved
        if hasattr(self.network, 'model') and self.network.model is not None:
            self.constraint_inventory['OptimizationModel'] = {
                'has_solved_model': True,
                'variables': list(self.network.model.variables.keys()) if hasattr(self.network.model, 'variables') else [],
                'constraints': list(self.network.model.constraints.keys()) if hasattr(self.network.model, 'constraints') else [],
                'note': 'Custom constraints from solved model - may not be directly exportable'
            }
        else:
            self.constraint_inventory['OptimizationModel'] = {
                'has_solved_model': False,
                'note': 'No solved optimization model found'
            }
    
    def _assess_constraint_sienna_compatibility(self) -> Dict[str, Any]:
        """Assess how well constraints can be exported to Sienna."""
        compatibility = {
            'directly_exportable': [],
            'needs_conversion': [],
            'not_exportable': [],
            'notes': []
        }
        
        if hasattr(self.network, 'global_constraints'):
            for _, constraint in self.network.global_constraints.iterrows():
                constraint_type = constraint.get('type', 'unknown')
                
                # Categorize constraints based on Sienna compatibility
                if constraint_type in ['transmission_expansion_cost_limit', 'tech_capacity_expansion_limit']:
                    compatibility['directly_exportable'].append(constraint_type)
                elif constraint_type in ['co2_limit', 'emission_limit']:
                    compatibility['needs_conversion'].append(constraint_type)
                    compatibility['notes'].append(f"{constraint_type} may need custom Sienna implementation")
                else:
                    compatibility['not_exportable'].append(constraint_type)
                    compatibility['notes'].append(f"{constraint_type} has no direct Sienna equivalent")
        
        return compatibility
    
    def _analyze_network_topology(self):
        """Analyze network topology and connectivity."""
        logger.info("Analyzing network topology...")
        
        topology = {
            'bus_count': len(self.network.buses),
            'connectivity': {},
            'isolated_buses': [],
            'islands': [],
            'transmission_summary': {}
        }
        
        # Check bus connectivity
        if not self.network.lines.empty or (hasattr(self.network, 'links') and not self.network.links.empty):
            connected_buses = set()
            
            # Add buses connected by lines
            if not self.network.lines.empty:
                connected_buses.update(self.network.lines['bus0'])
                connected_buses.update(self.network.lines['bus1'])
            
            # Add buses connected by links
            if hasattr(self.network, 'links') and not self.network.links.empty:
                connected_buses.update(self.network.links['bus0'])
                connected_buses.update(self.network.links['bus1'])
            
            # Find isolated buses
            all_buses = set(self.network.buses.index)
            topology['isolated_buses'] = list(all_buses - connected_buses)
            topology['connectivity']['connected_buses'] = len(connected_buses)
            topology['connectivity']['isolated_buses'] = len(topology['isolated_buses'])
        else:
            topology['isolated_buses'] = list(self.network.buses.index)
            topology['connectivity']['connected_buses'] = 0
            topology['connectivity']['isolated_buses'] = len(self.network.buses)
        
        # Transmission summary
        if not self.network.lines.empty:
            topology['transmission_summary']['lines'] = {
                'count': len(self.network.lines),
                'total_capacity': float(self.network.lines['s_nom'].sum()) if 's_nom' in self.network.lines.columns else 0,
                'extendable_count': int(self.network.lines['s_nom_extendable'].sum()) if 's_nom_extendable' in self.network.lines.columns else 0
            }
        
        if hasattr(self.network, 'links') and not self.network.links.empty:
            topology['transmission_summary']['links'] = {
                'count': len(self.network.links),
                'total_capacity': float(self.network.links['p_nom'].sum()) if 'p_nom' in self.network.links.columns else 0,
                'extendable_count': int(self.network.links['p_nom_extendable'].sum()) if 'p_nom_extendable' in self.network.links.columns else 0
            }
        
        self.component_inventory['Topology'] = topology
    
    def _validate_sienna_compatibility(self):
        """Validate overall network compatibility with Sienna requirements."""
        logger.info("Validating Sienna compatibility...")
        
        validation = {
            'is_compatible': True,
            'critical_issues': [],
            'warnings': [],
            'recommendations': [],
            'component_compatibility': {},
            'overall_score': 0
        }
        
        # Check each component type for compatibility
        total_score = 0
        max_score = 0
        
        for comp_name, comp_info in self.component_inventory.items():
            if comp_info.get('has_data', False) and comp_name in self.component_mappings:
                comp_compat = comp_info.get('sienna_compatible', {})
                
                # Score component compatibility
                score = 0
                if comp_compat.get('has_mapping', False):
                    score += 20
                
                missing_required = len(comp_compat.get('missing_required_fields', []))
                if missing_required == 0:
                    score += 30
                elif missing_required <= 2:
                    score += 15
                    validation['warnings'].append(f"{comp_name} missing {missing_required} required fields")
                else:
                    validation['critical_issues'].append(f"{comp_name} missing {missing_required} required fields")
                
                available_optional = len(comp_compat.get('available_optional_fields', []))
                score += min(available_optional * 5, 25)  # Up to 25 points for optional fields
                
                data_issues = len(comp_compat.get('data_quality_issues', []))
                if data_issues == 0:
                    score += 25
                else:
                    score += max(0, 25 - data_issues * 5)
                    validation['warnings'].extend(comp_compat.get('data_quality_issues', []))
                
                validation['component_compatibility'][comp_name] = {
                    'score': score,
                    'max_score': 100,
                    'percentage': score,
                    'issues': comp_compat
                }
                
                total_score += score
                max_score += 100
        
        # Calculate overall compatibility score
        if max_score > 0:
            validation['overall_score'] = (total_score / max_score) * 100
        
        # Determine if network is compatible
        if validation['overall_score'] < 60:
            validation['is_compatible'] = False
            validation['critical_issues'].append("Overall compatibility score too low (<60%)")
        
        # Add recommendations based on issues found
        if validation['critical_issues']:
            validation['recommendations'].append("Fix critical issues before attempting export")
        if validation['warnings']:
            validation['recommendations'].append("Address warnings to improve export quality")
        if validation['overall_score'] < 80:
            validation['recommendations'].append("Consider data validation and cleanup")
        
        self.export_summary['validation'] = validation
        self.is_validated = True
        
        if validation['is_compatible']:
            self.export_ready = True
            logger.info(f"Network validation passed (score: {validation['overall_score']:.1f}%)")
        else:
            logger.warning(f"Network validation failed (score: {validation['overall_score']:.1f}%)")
    
    def _generate_analysis_summary(self):
        """Generate comprehensive analysis summary."""
        logger.info("Generating analysis summary...")
        
        summary = {
            'network_overview': {
                'total_buses': len(self.network.buses),
                'total_generators': len(self.network.generators),
                'total_loads': len(self.network.loads),
                'total_lines': len(self.network.lines),
                'total_links': len(getattr(self.network, 'links', pd.DataFrame())),
                'total_storage_units': len(getattr(self.network, 'storage_units', pd.DataFrame())),
                'has_time_series': bool(self.time_series_inventory),
                'has_constraints': bool(self.constraint_inventory),
                'is_solved': hasattr(self.network, 'model') and self.network.model is not None
            },
            'component_summary': {},
            'time_series_summary': {},
            'export_readiness': {
                'is_ready': self.export_ready,
                'compatibility_score': self.export_summary.get('validation', {}).get('overall_score', 0),
                'critical_issues_count': len(self.export_summary.get('validation', {}).get('critical_issues', [])),
                'warnings_count': len(self.export_summary.get('validation', {}).get('warnings', []))
            },
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Component summary
        for comp_name, comp_info in self.component_inventory.items():
            if comp_info.get('has_data', False):
                summary['component_summary'][comp_name] = {
                    'count': comp_info['count'],
                    'has_sienna_mapping': comp_name in self.component_mappings,
                    'compatibility_score': comp_info.get('sienna_compatible', {}).get('score', 0)
                }
        
        # Time series summary
        total_ts_attributes = sum(len(ts_data) for ts_data in self.time_series_inventory.values())
        summary['time_series_summary'] = {
            'total_components_with_ts': len(self.time_series_inventory),
            'total_ts_attributes': total_ts_attributes,
            'exportable_ts': sum(
                len([attr for attr, info in ts_data.items() 
                     if info.get('sienna_relevance', {}).get('export_recommended', False)])
                for ts_data in self.time_series_inventory.values()
            )
        }
        
        self.export_summary['analysis'] = summary
        logger.info("Analysis summary generated")
    
    # Public methods for accessing analysis results
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive analysis summary.
        
        Returns:
        --------
        Dict[str, Any]
            Complete analysis summary including compatibility assessment
        """
        if not self.is_analyzed:
            self._analyze_network_comprehensive()
        
        return self.export_summary.get('analysis', {})
    
    def get_component_inventory(self) -> Dict[str, Any]:
        """Get detailed component inventory."""
        return self.component_inventory
    
    def get_time_series_inventory(self) -> Dict[str, Any]:
        """Get detailed time series inventory."""
        return self.time_series_inventory
    
    def get_constraint_inventory(self) -> Dict[str, Any]:
        """Get detailed constraint inventory."""
        return self.constraint_inventory
    
    def get_validation_results(self) -> Dict[str, Any]:
        """Get Sienna compatibility validation results."""
        return self.export_summary.get('validation', {})
    
    def is_export_ready(self) -> bool:
        """Check if network is ready for export to Sienna."""
        return self.export_ready
    
    def print_analysis_report(self):
        """Print a comprehensive analysis report to console."""
        if not self.is_analyzed:
            logger.warning("Network not yet analyzed. Run analysis first.")
            return
        
        print("\n" + "="*80)
        print("PYPSA TO SIENNA EXPORT ANALYSIS REPORT")
        print("="*80)
        
        # Overview
        overview = self.export_summary.get('analysis', {}).get('network_overview', {})
        print(f"\n📊 NETWORK OVERVIEW:")
        print(f"    Buses: {overview.get('total_buses', 0)}")
        print(f"    Generators: {overview.get('total_generators', 0)}")
        print(f"    Loads: {overview.get('total_loads', 0)}")
        print(f"    Lines: {overview.get('total_lines', 0)}")
        print(f"    Links: {overview.get('total_links', 0)}")
        print(f"    Storage Units: {overview.get('total_storage_units', 0)}")
        print(f"    Has Time Series: {overview.get('has_time_series', False)}")
        print(f"    Has Constraints: {overview.get('has_constraints', False)}")
        print(f"    Is Solved: {overview.get('is_solved', False)}")
        
        # Component compatibility
        print(f"\n🔧 COMPONENT COMPATIBILITY:")
        validation = self.get_validation_results()
        comp_compat = validation.get('component_compatibility', {})
        
        for comp_name, compat in comp_compat.items():
            score = compat.get('percentage', 0)
            status = "✅" if score >= 80 else "⚠️" if score >= 60 else "❌"
            print(f"    {status} {comp_name}: {score:.0f}%")
            
            issues = compat.get('issues', {})
            missing_fields = issues.get('missing_required_fields', [])
            if missing_fields:
                print(f"        Missing required: {', '.join(missing_fields)}")
            
            data_issues = issues.get('data_quality_issues', [])
            if data_issues:
                print(f"        Data issues: {len(data_issues)} found")
        
        # Time series analysis
        ts_summary = self.export_summary.get('analysis', {}).get('time_series_summary', {})
        print(f"\n📈 TIME SERIES ANALYSIS:")
        print(f"    Components with time series: {ts_summary.get('total_components_with_ts', 0)}")
        print(f"    Total time series attributes: {ts_summary.get('total_ts_attributes', 0)}")
        print(f"    Recommended for export: {ts_summary.get('exportable_ts', 0)}")
        
        # Export readiness
        readiness = self.export_summary.get('analysis', {}).get('export_readiness', {})
        overall_score = readiness.get('compatibility_score', 0)
        is_ready = readiness.get('is_ready', False)
        
        print(f"\n🎯 EXPORT READINESS:")
        status_icon = "✅" if is_ready else "❌"
        print(f"    {status_icon} Overall Compatibility: {overall_score:.1f}%")
        print(f"    Critical Issues: {readiness.get('critical_issues_count', 0)}")
        print(f"    Warnings: {readiness.get('warnings_count', 0)}")
        
        # Issues and recommendations
        critical_issues = validation.get('critical_issues', [])
        warnings = validation.get('warnings', [])
        recommendations = validation.get('recommendations', [])
        
        if critical_issues:
            print(f"\n❌ CRITICAL ISSUES:")
            for issue in critical_issues:
                print(f"    • {issue}")
        
        if warnings:
            print(f"\n⚠️  WARNINGS:")
            for warning in warnings[:5]:  # Show first 5 warnings
                print(f"    • {warning}")
            if len(warnings) > 5:
                print(f"    ... and {len(warnings) - 5} more warnings")
        
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"    • {rec}")
        
        # Export capability summary
        print(f"\n🚀 EXPORT CAPABILITIES:")
        if is_ready:
            print(f"    ✅ Network is ready for Sienna export")
            print(f"    ✅ All critical requirements met")
            print(f"    ✅ Compatible components identified")
        else:
            print(f"    ❌ Network requires fixes before export")
            print(f"    🔧 Address critical issues first")
            print(f"    📋 Review compatibility requirements")
        
        print("="*80)
    
    def print_detailed_component_analysis(self, component_name: str):
        """Print detailed analysis for a specific component."""
        if component_name not in self.component_inventory:
            print(f"❌ Component '{component_name}' not found in inventory")
            return
        
        comp_info = self.component_inventory[component_name]
        
        print(f"\n" + "="*60)
        print(f"DETAILED ANALYSIS: {component_name}")
        print("="*60)
        
        # Basic info
        print(f"📊 BASIC INFORMATION:")
        print(f"    Count: {comp_info.get('count', 0)}")
        print(f"    Has Data: {comp_info.get('has_data', False)}")
        print(f"    Columns: {len(comp_info.get('columns', []))}")
        
        # Column details
        if comp_info.get('columns'):
            print(f"\n📋 COLUMNS:")
            for col in comp_info['columns']:
                dtype = comp_info.get('dtypes', {}).get(col, 'unknown')
                missing = comp_info.get('missing_data', {}).get(col, 0)
                print(f"    {col}: {dtype} (missing: {missing})")
        
        # Value ranges for numeric columns
        value_ranges = comp_info.get('value_ranges', {})
        if value_ranges:
            print(f"\n📈 VALUE RANGES:")
            for col, ranges in value_ranges.items():
                print(f"    {col}:")
                print(f"      Range: {ranges.get('min', 0):.2f} to {ranges.get('max', 0):.2f}")
                print(f"      Mean: {ranges.get('mean', 0):.2f}")
                issues = []
                if ranges.get('has_negative'):
                    issues.append("negative values")
                if ranges.get('has_inf'):
                    issues.append("infinite values")
                if ranges.get('has_nan'):
                    issues.append("NaN values")
                if issues:
                    print(f"      Issues: {', '.join(issues)}")
        
        # Sienna compatibility
        sienna_compat = comp_info.get('sienna_compatible', {})
        if sienna_compat.get('has_mapping'):
            print(f"\n🔧 SIENNA COMPATIBILITY:")
            print(f"    Has Mapping: ✅")
            
            missing_req = sienna_compat.get('missing_required_fields', [])
            if missing_req:
                print(f"    Missing Required Fields: {', '.join(missing_req)}")
            else:
                print(f"    Required Fields: ✅ All present")
            
            available_opt = sienna_compat.get('available_optional_fields', [])
            print(f"    Available Optional Fields: {len(available_opt)}")
            if available_opt:
                print(f"      {', '.join(available_opt)}")
            
            data_issues = sienna_compat.get('data_quality_issues', [])
            if data_issues:
                print(f"    Data Quality Issues:")
                for issue in data_issues:
                    print(f"      • {issue}")
        else:
            print(f"\n🔧 SIENNA COMPATIBILITY:")
            print(f"    Has Mapping: ❌ No mapping defined")
        
        # Time series info if available
        if component_name in self.time_series_inventory:
            ts_info = self.time_series_inventory[component_name]
            print(f"\n📈 TIME SERIES DATA:")
            for attr_name, attr_info in ts_info.items():
                shape = attr_info.get('shape', (0, 0))
                export_rec = attr_info.get('sienna_relevance', {}).get('export_recommended', False)
                status = "✅" if export_rec else "❌"
                print(f"    {status} {attr_name}: {shape[0]} timesteps × {shape[1]} components")
                
                quality = attr_info.get('data_quality', {})
                if quality.get('has_nan') or quality.get('has_inf'):
                    issues = []
                    if quality.get('has_nan'):
                        issues.append("NaN")
                    if quality.get('has_inf'):
                        issues.append("Inf")
                    print(f"      Issues: {', '.join(issues)}")
        
        print("="*60)
    
    def get_export_recommendations(self) -> Dict[str, List[str]]:
        """
        Get specific recommendations for improving export compatibility.
        
        Returns:
        --------
        Dict[str, List[str]]
            Categorized recommendations for improving compatibility
        """
        recommendations = {
            'critical_fixes': [],
            'data_quality': [],
            'optional_improvements': [],
            'time_series': [],
            'constraints': []
        }
        
        validation = self.get_validation_results()
        
        # Critical fixes
        for issue in validation.get('critical_issues', []):
            recommendations['critical_fixes'].append(issue)
        
        # Component-specific recommendations
        comp_compat = validation.get('component_compatibility', {})
        for comp_name, compat in comp_compat.items():
            issues = compat.get('issues', {})
            
            missing_fields = issues.get('missing_required_fields', [])
            if missing_fields:
                recommendations['critical_fixes'].append(
                    f"Add missing required fields for {comp_name}: {', '.join(missing_fields)}"
                )
            
            data_issues = issues.get('data_quality_issues', [])
            for issue in data_issues:
                recommendations['data_quality'].append(f"{comp_name}: {issue}")
        
        # Time series recommendations
        for comp_name, ts_data in self.time_series_inventory.items():
            for attr_name, attr_info in ts_data.items():
                quality = attr_info.get('data_quality', {})
                if quality.get('has_nan') or quality.get('has_inf'):
                    recommendations['time_series'].append(
                        f"Fix data quality issues in {comp_name}.{attr_name}"
                    )
                
                if not attr_info.get('sienna_relevance', {}).get('export_recommended', False):
                    recommendations['optional_improvements'].append(
                        f"Consider whether {comp_name}.{attr_name} should be exported"
                    )
        
        # Constraint recommendations
        constraint_compat = self.constraint_inventory.get('GlobalConstraint', {}).get('sienna_compatibility', {})
        not_exportable = constraint_compat.get('not_exportable', [])
        for constraint_type in not_exportable:
            recommendations['constraints'].append(
                f"Constraint type '{constraint_type}' cannot be directly exported to Sienna"
            )
        
        return recommendations
    
    # Methods for component-specific analysis
    
    def analyze_renewable_generators(self) -> Dict[str, Any]:
        """Analyze renewable generators specifically for Sienna export."""
        if 'Generator' not in self.component_inventory:
            return {'error': 'No generators found in network'}
        
        generators = self.network.generators
        analysis = {
            'total_generators': len(generators),
            'renewable_generators': {},
            'time_series_availability': {},
            'sienna_readiness': {}
        }
        
        # Identify renewable carriers
        renewable_carriers = ['solar', 'wind', 'hydro', 'biomass', 'geothermal']
        renewable_gens = generators[
            generators['carrier'].str.lower().str.contains('|'.join(renewable_carriers), na=False)
        ]
        
        analysis['renewable_generators'] = {
            'count': len(renewable_gens),
            'carriers': renewable_gens['carrier'].value_counts().to_dict(),
            'total_capacity': float(renewable_gens['p_nom_max'].sum()) if 'p_nom_max' in renewable_gens.columns else 0
        }
        
        # Check time series availability
        if 'Generator' in self.time_series_inventory:
            gen_ts = self.time_series_inventory['Generator']
            analysis['time_series_availability'] = {
                'p_max_pu': 'p_max_pu' in gen_ts,
                'p_min_pu': 'p_min_pu' in gen_ts,
                'marginal_cost': 'marginal_cost' in gen_ts
            }
        
        # Sienna readiness assessment
        analysis['sienna_readiness'] = {
            'has_capacity_data': 'p_nom_max' in renewable_gens.columns or 'p_nom' in renewable_gens.columns,
            'has_availability_profiles': 'p_max_pu' in self.time_series_inventory.get('Generator', {}),
            'ready_for_export': True
        }
        
        # Check for issues
        issues = []
        if analysis['renewable_generators']['count'] == 0:
            issues.append("No renewable generators found")
        if not analysis['time_series_availability']['p_max_pu']:
            issues.append("No renewable availability profiles found")
        if not analysis['sienna_readiness']['has_capacity_data']:
            issues.append("No capacity data found for renewables")
        
        analysis['sienna_readiness']['issues'] = issues
        analysis['sienna_readiness']['ready_for_export'] = len(issues) == 0
        
        return analysis
    
    def analyze_load_data(self) -> Dict[str, Any]:
        """Analyze load data for Sienna export."""
        if 'Load' not in self.component_inventory:
            return {'error': 'No loads found in network'}
        
        loads = self.network.loads
        analysis = {
            'total_loads': len(loads),
            'load_distribution': {},
            'time_series_data': {},
            'sienna_readiness': {}
        }
        
        # Load distribution by bus
        analysis['load_distribution'] = {
            'buses_with_loads': loads['bus'].nunique(),
            'loads_per_bus': loads['bus'].value_counts().to_dict(),
            'total_peak_load': 0
        }
        
        # Check time series data
        if 'Load' in self.time_series_inventory and 'p_set' in self.time_series_inventory['Load']:
            load_ts_info = self.time_series_inventory['Load']['p_set']
            load_ts = self.network.loads_t.p_set
            
            analysis['time_series_data'] = {
                'has_time_series': True,
                'shape': load_ts_info['shape'],
                'time_range': load_ts_info['time_range'],
                'total_energy': float(load_ts.sum().sum()),
                'peak_demand': float(load_ts.sum(axis=1).max()),
                'data_quality': load_ts_info['data_quality']
            }
            analysis['load_distribution']['total_peak_load'] = analysis['time_series_data']['peak_demand']
        else:
            analysis['time_series_data'] = {
                'has_time_series': False,
                'note': 'Static load values only'
            }
            if 'p_set' in loads.columns:
                analysis['load_distribution']['total_peak_load'] = float(loads['p_set'].sum())
        
        # Sienna readiness
        analysis['sienna_readiness'] = {
            'has_load_data': len(loads) > 0,
            'has_time_series': analysis['time_series_data']['has_time_series'],
            'has_bus_assignments': loads['bus'].notna().all(),
            'ready_for_export': True,
            'issues': []
        }
        
        # Check for issues
        if not analysis['sienna_readiness']['has_load_data']:
            analysis['sienna_readiness']['issues'].append("No load data found")
        if not analysis['sienna_readiness']['has_bus_assignments']:
            analysis['sienna_readiness']['issues'].append("Some loads not assigned to buses")
        if analysis['time_series_data'].get('data_quality', {}).get('has_nan', False):
            analysis['sienna_readiness']['issues'].append("NaN values in load time series")
        
        analysis['sienna_readiness']['ready_for_export'] = len(analysis['sienna_readiness']['issues']) == 0
        
        return analysis
    
    def analyze_transmission_network(self) -> Dict[str, Any]:
        """Analyze transmission network for Sienna export."""
        analysis = {
            'lines': {},
            'links': {},
            'topology': {},
            'sienna_readiness': {}
        }
        
        # Analyze lines
        if not self.network.lines.empty:
            lines = self.network.lines
            analysis['lines'] = {
                'count': len(lines),
                'has_impedance_data': all(col in lines.columns for col in ['r', 'x']),
                'has_capacity_data': 's_nom' in lines.columns,
                'extendable_count': int(lines.get('s_nom_extendable', pd.Series(False)).sum()),
                'total_capacity': float(lines.get('s_nom', pd.Series(0)).sum())
            }
        else:
            analysis['lines'] = {'count': 0, 'note': 'No AC lines found'}
        
        # Analyze links (DC lines)
        if hasattr(self.network, 'links') and not self.network.links.empty:
            links = self.network.links
            analysis['links'] = {
                'count': len(links),
                'has_capacity_data': 'p_nom' in links.columns,
                'extendable_count': int(links.get('p_nom_extendable', pd.Series(False)).sum()),
                'total_capacity': float(links.get('p_nom', pd.Series(0)).sum())
            }
        else:
            analysis['links'] = {'count': 0, 'note': 'No DC links found'}
        
        # Topology analysis from existing inventory
        topology_info = self.component_inventory.get('Topology', {})
        analysis['topology'] = {
            'total_buses': topology_info.get('bus_count', 0),
            'connected_buses': topology_info.get('connectivity', {}).get('connected_buses', 0),
            'isolated_buses': len(topology_info.get('isolated_buses', [])),
            'connectivity_ratio': 0
        }
        
        if analysis['topology']['total_buses'] > 0:
            analysis['topology']['connectivity_ratio'] = (
                analysis['topology']['connected_buses'] / analysis['topology']['total_buses']
            )
        
        # Sienna readiness
        analysis['sienna_readiness'] = {
            'has_transmission_data': analysis['lines']['count'] > 0 or analysis['links']['count'] > 0,
            'has_impedance_data': analysis['lines'].get('has_impedance_data', False),
            'network_connected': analysis['topology']['connectivity_ratio'] > 0.8,
            'ready_for_export': True,
            'issues': []
        }
        
        if not analysis['sienna_readiness']['has_transmission_data']:
            analysis['sienna_readiness']['issues'].append("No transmission lines or links found")
        if not analysis['sienna_readiness']['has_impedance_data'] and analysis['lines']['count'] > 0:
            analysis['sienna_readiness']['issues'].append("Missing impedance data (r, x) for AC lines")
        if analysis['topology']['isolated_buses'] > 0:
            analysis['sienna_readiness']['issues'].append(f"{analysis['topology']['isolated_buses']} isolated buses found")
        
        analysis['sienna_readiness']['ready_for_export'] = len(analysis['sienna_readiness']['issues']) == 0
        
        return analysis
    
    # Future methods for actual export functionality
    def export_to_sienna_csv(self, output_dir: str, include_time_series: bool = True) -> Dict[str, str]:
        """
        Placeholder for main export function.
        
        This will be implemented in the next development phase.
        """
        if not self.export_ready:
            raise ValueError("Network is not ready for export. Run validation and fix issues first.")
        
        logger.info("Export functionality will be implemented in next phase")
        return {"status": "not_implemented", "message": "Export functionality coming in next phase"}
    
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

# Utility functions for testing and development

def test_exporter_with_sample_network():
    """Create a sample network and test the exporter functionality."""
    print("🧪 Testing PyPSAToSiennaCSVExporter with sample network...")
    
    # Create a simple test network
    n = pypsa.Network()
    
    # Add buses
    n.add("Bus", "bus1", v_nom=380, x=0, y=0)
    n.add("Bus", "bus2", v_nom=380, x=1, y=1)
    
    # Add generators
    n.add("Generator", "gas1", bus="bus1", p_nom_max=1000, marginal_cost=50, carrier="gas")
    n.add("Generator", "solar1", bus="bus2", p_nom_max=500, marginal_cost=0, carrier="solar")
    
    # Add load
    n.add("Load", "load1", bus="bus1", p_set=800)
    n.add("Load", "load2", bus="bus2", p_set=400)
    
    # Add line
    n.add("Line", "line1", bus0="bus1", bus1="bus2", r=0.01, x=0.1, s_nom=1500)
    
    # Add some time series data
    import pandas as pd
    snapshots = pd.date_range("2024-01-01", periods=24, freq="H")
    n.set_snapshots(snapshots)
    
    # Load time series
    load_profile = np.sin(np.linspace(0, 2*np.pi, 24)) * 0.3 + 0.7
    n.loads_t.p_set["load1"] = 800 * load_profile
    n.loads_t.p_set["load2"] = 400 * load_profile
    
    # Solar availability
    solar_profile = np.maximum(0, np.sin(np.linspace(0, np.pi, 24)))
    n.generators_t.p_max_pu["solar1"] = solar_profile
    
    print("✅ Sample network created")
    
    # Test the exporter
    scenario_setup = {"test": "scenario"}
    exporter = PyPSAToSiennaCSVExporter(n, scenario_setup)
    
    print("✅ Exporter initialized")
    
    # Print analysis report
    exporter.print_analysis_report()
    
    # Test specific component analysis
    print("\n" + "="*60)
    exporter.print_detailed_component_analysis("Generator")
    
    # Test renewable analysis
    print("\n" + "="*60)
    renewable_analysis = exporter.analyze_renewable_generators()
    print("🔋 RENEWABLE ANALYSIS:")
    print(f"    Total renewables: {renewable_analysis['renewable_generators']['count']}")
    print(f"    Ready for export: {renewable_analysis['sienna_readiness']['ready_for_export']}")
    
    # Test load analysis
    load_analysis = exporter.analyze_load_data()
    print("\n⚡ LOAD ANALYSIS:")
    print(f"    Total loads: {load_analysis['total_loads']}")
    print(f"    Has time series: {load_analysis['time_series_data']['has_time_series']}")
    print(f"    Peak demand: {load_analysis['time_series_data'].get('peak_demand', 'N/A')} MW")
    
    # Test transmission analysis
    transmission_analysis = exporter.analyze_transmission_network()
    print("\n🔌 TRANSMISSION ANALYSIS:")
    print(f"    Lines: {transmission_analysis['lines']['count']}")
    print(f"    Connected buses: {transmission_analysis['topology']['connected_buses']}")
    print(f"    Ready for export: {transmission_analysis['sienna_readiness']['ready_for_export']}")
    
    # Get recommendations
    recommendations = exporter.get_export_recommendations()
    print("\n💡 EXPORT RECOMMENDATIONS:")
    for category, recs in recommendations.items():
        if recs:
            print(f"    {category.replace('_', ' ').title()}:")
            for rec in recs[:3]:  # Show first 3
                print(f"      • {rec}")
    
    print("\n🎉 Exporter testing complete!")
    return exporter

if __name__ == "__main__":
    # Example usage for testing
    logging.basicConfig(level=logging.INFO)
    
    # This would be called from your solve_network_dispatch.py
    # when export_to_Sienna=True
    
    # Example:
    n = pypsa.Network("dispatch_network.nc")
    scenario_setup = load_scenario_definition(snakemake)
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