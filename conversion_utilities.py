#!/usr/bin/env python3
"""
Additional utilities for PyPSA-ZA to Sienna conversion.

This module provides helper functions for data validation, preprocessing,
and post-processing of converted network files.
"""

import xarray as xr
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class PyPSAAnalyzer:
    """
    Utility class for analyzing PyPSA-ZA network files before conversion.
    """
    
    def __init__(self, network_file: str):
        """Initialize analyzer with PyPSA network file."""
        self.network_file = Path(network_file)
        self.ds = xr.open_dataset(network_file)
        
    def analyze_system_structure(self) -> Dict[str, Any]:
        """
        Analyze the structure and characteristics of the PyPSA system.
        
        Returns:
            Dictionary containing system analysis results
        """
        analysis = {
            'file_info': self._analyze_file_structure(),
            'temporal_structure': self._analyze_temporal_structure(),
            'spatial_structure': self._analyze_spatial_structure(),
            'generation_mix': self._analyze_generation_mix(),
            'load_characteristics': self._analyze_load_characteristics(),
            'storage_analysis': self._analyze_storage_systems(),
            'data_quality': self._check_data_quality()
        }
        
        return analysis
    
    def _analyze_file_structure(self) -> Dict[str, Any]:
        """Analyze basic file structure and metadata."""
        file_size_mb = self.network_file.stat().st_size / (1024 * 1024)
        
        return {
            'file_size_mb': round(file_size_mb, 2),
            'dimensions': dict(self.ds.dims),
            'coordinates': list(self.ds.coords.keys()),
            'data_variables': list(self.ds.data_vars.keys()),
            'total_variables': len(self.ds.data_vars),
            'encoding': {var: str(self.ds[var].dtype) for var in list(self.ds.data_vars.keys())[:10]}
        }
    
    def _analyze_temporal_structure(self) -> Dict[str, Any]:
        """Analyze temporal characteristics of the dataset."""
        num_snapshots = len(self.ds.coords['snapshots'])
        
        temporal_info = {
            'total_snapshots': num_snapshots,
            'time_resolution': 'hourly' if num_snapshots % 8760 == 0 else 'variable'
        }
        
        if 'investment_periods' in self.ds.coords:
            periods = self.ds.coords['investment_periods'].values
            temporal_info.update({
                'is_multi_period': True,
                'investment_periods': [int(p) for p in periods],
                'num_periods': len(periods),
                'snapshots_per_period': num_snapshots // len(periods),
                'planning_horizon': int(periods[-1] - periods[0])
            })
        else:
            temporal_info['is_multi_period'] = False
        
        if 'snapshots_timestep' in self.ds:
            start_time = pd.to_datetime(self.ds['snapshots_timestep'].values[0])
            end_time = pd.to_datetime(self.ds['snapshots_timestep'].values[-1])
            temporal_info.update({
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_years': (end_time - start_time).days / 365.25
            })
        
        return temporal_info
    
    def _analyze_spatial_structure(self) -> Dict[str, Any]:
        """Analyze spatial characteristics and network topology."""
        num_buses = len(self.ds.coords['buses_i'])
        
        spatial_info = {
            'num_buses': num_buses,
            'is_single_node': num_buses == 1,
            'bus_names': [str(bus) for bus in self.ds.coords['buses_i'].values]
        }
        
        if 'buses_x' in self.ds and 'buses_y' in self.ds:
            coords_x = self.ds['buses_x'].values
            coords_y = self.ds['buses_y'].values
            
            spatial_info.update({
                'has_coordinates': True,
                'coordinate_bounds': {
                    'x_min': float(coords_x.min()),
                    'x_max': float(coords_x.max()),
                    'y_min': float(coords_y.min()),
                    'y_max': float(coords_y.max())
                }
            })
        
        return spatial_info
    
    def _analyze_generation_mix(self) -> Dict[str, Any]:
        """Analyze generation technologies and capacity mix."""
        if 'generators_carrier' not in self.ds:
            return {'error': 'No generator data found'}
        
        carriers = self.ds['generators_carrier'].values
        capacities = self.ds['generators_p_nom'].values
        extendable = self.ds['generators_p_nom_extendable'].values
        
        # Group by carrier
        carrier_df = pd.DataFrame({
            'carrier': carriers,
            'capacity': capacities,
            'extendable': extendable
        })
        
        capacity_by_carrier = carrier_df.groupby('carrier')['capacity'].sum().sort_values(ascending=False)
        extendable_capacity = carrier_df[carrier_df['extendable']].groupby('carrier')['capacity'].sum()
        
        generation_mix = {
            'total_capacity_mw': float(capacity_by_carrier.sum()),
            'num_generators': len(carriers),
            'num_technologies': len(capacity_by_carrier),
            'technology_mix': capacity_by_carrier.to_dict(),
            'capacity_shares': (capacity_by_carrier / capacity_by_carrier.sum() * 100).round(2).to_dict(),
            'extendable_capacity': extendable_capacity.to_dict(),
            'largest_technology': capacity_by_carrier.index[0],
            'most_diverse_threshold': len([c for c in capacity_by_carrier if c > capacity_by_carrier.sum() * 0.05])
        }
        
        return generation_mix
    
    def _analyze_load_characteristics(self) -> Dict[str, Any]:
        """Analyze load profiles and characteristics."""
        if 'loads_t_p_set' not in self.ds:
            return {'error': 'No load data found'}
        
        load_data = self.ds['loads_t_p_set'].values
        
        if load_data.ndim > 1:
            # Sum across all loads if multiple
            total_load = load_data.sum(axis=1)
        else:
            total_load = load_data
        
        load_stats = {
            'peak_load_mw': float(total_load.max()),
            'min_load_mw': float(total_load.min()),
            'avg_load_mw': float(total_load.mean()),
            'load_factor': float(total_load.mean() / total_load.max()),
            'total_energy_gwh': float(total_load.sum() / 1000),  # Assuming hourly data
            'num_load_nodes': load_data.shape[1] if load_data.ndim > 1 else 1
        }
        
        # Seasonal analysis if we have enough data
        if len(total_load) >= 8760:
            # Assume first year for seasonal analysis
            yearly_load = total_load[:8760]
            monthly_avg = pd.Series(yearly_load).groupby(pd.date_range('2025-01-01', periods=8760, freq='H').month).mean()
            
            load_stats.update({
                'seasonal_variation': {
                    'summer_avg': float(monthly_avg[[12, 1, 2]].mean()),  # Dec, Jan, Feb
                    'winter_avg': float(monthly_avg[[6, 7, 8]].mean()),   # Jun, Jul, Aug
                    'peak_month': int(monthly_avg.idxmax()),
                    'low_month': int(monthly_avg.idxmin())
                }
            })
        
        return load_stats
    
    def _analyze_storage_systems(self) -> Dict[str, Any]:
        """Analyze storage unit characteristics."""
        if 'storage_units_carrier' not in self.ds:
            return {'has_storage': False}
        
        carriers = self.ds['storage_units_carrier'].values
        power_capacity = self.ds['storage_units_p_nom'].values
        energy_capacity = power_capacity * self.ds['storage_units_max_hours'].values
        
        storage_df = pd.DataFrame({
            'carrier': carriers,
            'power_mw': power_capacity,
            'energy_mwh': energy_capacity
        })
        
        storage_by_type = storage_df.groupby('carrier').agg({
            'power_mw': 'sum',
            'energy_mwh': 'sum'
        })
        
        storage_analysis = {
            'has_storage': True,
            'total_power_capacity_mw': float(storage_df['power_mw'].sum()),
            'total_energy_capacity_mwh': float(storage_df['energy_mwh'].sum()),
            'num_storage_units': len(carriers),
            'storage_technologies': storage_by_type.to_dict(),
            'avg_duration_hours': float(storage_df['energy_mwh'].sum() / storage_df['power_mw'].sum())
        }
        
        return storage_analysis
    
    def _check_data_quality(self) -> Dict[str, Any]:
        """Check for data quality issues."""
        quality_issues = {
            'missing_data': {},
            'invalid_values': {},
            'consistency_checks': {}
        }
        
        # Check for missing values
        for var in self.ds.data_vars:
            if self.ds[var].isnull().any():
                null_count = int(self.ds[var].isnull().sum())
                total_count = int(self.ds[var].size)
                quality_issues['missing_data'][var] = {
                    'null_count': null_count,
                    'null_percentage': round(null_count / total_count * 100, 2)
                }
        
        # Check for invalid capacity factors
        if 'generators_t_p_max_pu' in self.ds:
            cf_data = self.ds['generators_t_p_max_pu']
            invalid_cf = ((cf_data < 0) | (cf_data > 1)).any()
            if invalid_cf.any():
                quality_issues['invalid_values']['capacity_factors'] = 'Found capacity factors outside [0,1] range'
        
        # Check energy balance (rough check)
        if 'loads_t_p_set' in self.ds and 'generators_p_nom' in self.ds:
            total_gen_capacity = float(self.ds['generators_p_nom'].sum())
            peak_load = float(self.ds['loads_t_p_set'].max())
            
            if total_gen_capacity < peak_load:
                quality_issues['consistency_checks']['adequacy'] = f'Generation capacity ({total_gen_capacity:.0f} MW) < peak load ({peak_load:.0f} MW)'
        
        return quality_issues
    
    def generate_summary_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate a comprehensive summary report of the PyPSA system.
        
        Args:
            output_file: Optional file path to save the report
            
        Returns:
            Report as string
        """
        analysis = self.analyze_system_structure()
        
        report_lines = [
            "=" * 80,
            f"PyPSA-ZA System Analysis Report",
            f"File: {self.network_file.name}",
            "=" * 80,
            "",
            "## File Structure",
            f"Size: {analysis['file_info']['file_size_mb']} MB",
            f"Dimensions: {analysis['file_info']['dimensions']}",
            f"Total variables: {analysis['file_info']['total_variables']}",
            "",
            "## Temporal Structure", 
            f"Total snapshots: {analysis['temporal_structure']['total_snapshots']:,}",
            f"Multi-period: {analysis['temporal_structure']['is_multi_period']}",
        ]
        
        if analysis['temporal_structure']['is_multi_period']:
            report_lines.extend([
                f"Investment periods: {analysis['temporal_structure']['investment_periods']}",
                f"Planning horizon: {analysis['temporal_structure']['planning_horizon']} years"
            ])
        
        report_lines.extend([
            "",
            "## Spatial Structure",
            f"Number of buses: {analysis['spatial_structure']['num_buses']}",
            f"Single node model: {analysis['spatial_structure']['is_single_node']}",
            "",
            "## Generation Mix",
            f"Total capacity: {analysis['generation_mix']['total_capacity_mw']:,.0f} MW",
            f"Number of generators: {analysis['generation_mix']['num_generators']}",
            f"Technologies: {analysis['generation_mix']['num_technologies']}",
            f"Largest technology: {analysis['generation_mix']['largest_technology']}",
            "",
            "### Capacity by Technology:"
        ])
        
        for tech, capacity in analysis['generation_mix']['technology_mix'].items():
            share = analysis['generation_mix']['capacity_shares'][tech]
            report_lines.append(f"  {tech}: {capacity:,.0f} MW ({share:.1f}%)")
        
        report_lines.extend([
            "",
            "## Load Characteristics",
            f"Peak load: {analysis['load_characteristics']['peak_load_mw']:,.0f} MW",
            f"Average load: {analysis['load_characteristics']['avg_load_mw']:,.0f} MW",
            f"Load factor: {analysis['load_characteristics']['load_factor']:.2f}",
            f"Total energy: {analysis['load_characteristics']['total_energy_gwh']:,.0f} GWh",
        ])
        
        if analysis['storage_analysis']['has_storage']:
            report_lines.extend([
                "",
                "## Storage Systems",
                f"Total power capacity: {analysis['storage_analysis']['total_power_capacity_mw']:,.0f} MW",
                f"Total energy capacity: {analysis['storage_analysis']['total_energy_capacity_mwh']:,.0f} MWh",
                f"Average duration: {analysis['storage_analysis']['avg_duration_hours']:.1f} hours",
                f"Number of units: {analysis['storage_analysis']['num_storage_units']}"
            ])
        
        # Data quality section
        quality = analysis['data_quality']
        if quality['missing_data'] or quality['invalid_values'] or quality['consistency_checks']:
            report_lines.extend([
                "",
                "## Data Quality Issues"
            ])
            
            if quality['missing_data']:
                report_lines.append("### Missing Data:")
                for var, info in quality['missing_data'].items():
                    report_lines.append(f"  {var}: {info['null_count']} missing ({info['null_percentage']:.1f}%)")
            
            if quality['invalid_values']:
                report_lines.append("### Invalid Values:")
                for issue, description in quality['invalid_values'].items():
                    report_lines.append(f"  {issue}: {description}")
            
            if quality['consistency_checks']:
                report_lines.append("### Consistency Issues:")
                for check, issue in quality['consistency_checks'].items():
                    report_lines.append(f"  {check}: {issue}")
        
        report_lines.extend([
            "",
            "=" * 80,
            f"Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80
        ])
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_file}")
        
        return report


class SiennaValidator:
    """
    Utility class for validating converted Sienna data.
    """
    
    def __init__(self, sienna_dir: str):
        """Initialize validator with Sienna output directory."""
        self.sienna_dir = Path(sienna_dir)
        self.system_data = self._load_system_data()
        
    def _load_system_data(self) -> Dict[str, Any]:
        """Load Sienna system data."""
        system_file = self.sienna_dir / 'system.json'
        if not system_file.exists():
            raise FileNotFoundError(f"System file not found: {system_file}")
        
        with open(system_file, 'r') as f:
            return json.load(f)
    
    def validate_conversion(self, original_pypsa_file: str) -> Dict[str, Any]:
        """
        Validate conversion by comparing original PyPSA data with Sienna output.
        
        Args:
            original_pypsa_file: Path to original PyPSA network file
            
        Returns:
            Validation results dictionary
        """
        # Load original data
        original_ds = xr.open_dataset(original_pypsa_file)
        
        validation_results = {
            'capacity_validation': self._validate_capacities(original_ds),
            'component_count_validation': self._validate_component_counts(original_ds),
            'time_series_validation': self._validate_time_series(original_ds),
            'data_consistency': self._check_data_consistency()
        }
        
        # Overall validation status
        all_passed = all(
            result.get('passed', False) for result in validation_results.values()
        )
        validation_results['overall_passed'] = all_passed
        
        return validation_results
    
    def _validate_capacities(self, original_ds: xr.Dataset) -> Dict[str, Any]:
        """Validate that generation capacities match."""
        try:
            # Original total generation capacity
            original_gen_capacity = float(original_ds['generators_p_nom'].sum())
            
            # Sienna total generation capacity
            sienna_gen_capacity = 0
            for gen_type in ['thermal', 'renewable', 'hydro']:
                if gen_type in self.system_data['generators']:
                    sienna_gen_capacity += sum(
                        gen['rating'] for gen in self.system_data['generators'][gen_type]
                    )
            
            # Calculate error
            capacity_error = abs(original_gen_capacity - sienna_gen_capacity) / original_gen_capacity
            tolerance = 0.01  # 1% tolerance
            
            return {
                'passed': capacity_error < tolerance,
                'original_capacity_mw': original_gen_capacity,
                'sienna_capacity_mw': sienna_gen_capacity,
                'error_percentage': capacity_error * 100,
                'tolerance_percentage': tolerance * 100
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
    
    def _validate_component_counts(self, original_ds: xr.Dataset) -> Dict[str, Any]:
        """Validate component counts match."""
        try:
            # Original counts
            original_counts = {
                'generators': len(original_ds.coords['generators_i']),
                'buses': len(original_ds.coords['buses_i']),
                'loads': len(original_ds.coords['loads_i'])
            }
            
            if 'storage_units_i' in original_ds.coords:
                original_counts['storage'] = len(original_ds.coords['storage_units_i'])
            
            # Sienna counts
            sienna_counts = {
                'generators': sum(len(self.system_data['generators'].get(cat, [])) 
                                for cat in ['thermal', 'renewable', 'hydro']),
                'buses': len(self.system_data.get('buses', [])),
                'loads': len(self.system_data.get('loads', [])),
                'storage': len(self.system_data.get('storage', []))
            }
            
            # Check matches
            matches = {}
            for component, original_count in original_counts.items():
                sienna_count = sienna_counts.get(component, 0)
                matches[component] = original_count == sienna_count
            
            return {
                'passed': all(matches.values()),
                'original_counts': original_counts,
                'sienna_counts': sienna_counts,
                'matches': matches
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
    
    def _validate_time_series(self, original_ds: xr.Dataset) -> Dict[str, Any]:
        """Validate time series data consistency."""
        try:
            ts_dir = self.sienna_dir / 'time_series'
            if not ts_dir.exists():
                return {
                    'passed': False,
                    'error': 'Time series directory not found'
                }
            
            ts_files = list(ts_dir.glob('*.json'))
            
            # Check load time series
            load_ts_files = [f for f in ts_files if 'Load' in f.name and 'active_power' in f.name]
            
            validation_info = {
                'num_time_series_files': len(ts_files),
                'load_time_series_count': len(load_ts_files)
            }
            
            if load_ts_files:
                # Validate one load time series as example
                with open(load_ts_files[0], 'r') as f:
                    ts_data = json.load(f)
                
                original_load_length = len(original_ds.coords['snapshots'])
                sienna_ts_length = len(ts_data['data'])
                
                validation_info.update({
                    'original_snapshots': original_load_length,
                    'sienna_ts_length': sienna_ts_length,
                    'length_match': original_load_length == sienna_ts_length
                })
            
            return {
                'passed': True,
                **validation_info
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
    
    def _check_data_consistency(self) -> Dict[str, Any]:
        """Check internal data consistency in Sienna format."""
        try:
            consistency_checks = {}
            
            # Check bus references
            bus_names = {bus['name'] for bus in self.system_data.get('buses', [])}
            
            # Check generator bus references
            gen_bus_refs = set()
            for gen_type in ['thermal', 'renewable', 'hydro']:
                for gen in self.system_data['generators'].get(gen_type, []):
                    gen_bus_refs.add(gen['bus'])
            
            invalid_gen_buses = gen_bus_refs - bus_names
            consistency_checks['generator_bus_references'] = {
                'passed': len(invalid_gen_buses) == 0,
                'invalid_buses': list(invalid_gen_buses)
            }
            
            # Check load bus references
            load_bus_refs = {load['bus'] for load in self.system_data.get('loads', [])}
            invalid_load_buses = load_bus_refs - bus_names
            consistency_checks['load_bus_references'] = {
                'passed': len(invalid_load_buses) == 0,
                'invalid_buses': list(invalid_load_buses)
            }
            
            return {
                'passed': all(check['passed'] for check in consistency_checks.values()),
                'checks': consistency_checks
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }


class ConversionVisualizer:
    """
    Utility class for creating visualizations of PyPSA-ZA data and conversion results.
    """
    
    def __init__(self, pypsa_file: str, sienna_dir: Optional[str] = None):
        """Initialize visualizer."""
        self.pypsa_file = pypsa_file
        self.sienna_dir = Path(sienna_dir) if sienna_dir else None
        self.ds = xr.open_dataset(pypsa_file)
    
    def create_generation_mix_plot(self, output_file: str = 'generation_mix.png'):
        """Create generation mix visualization."""
        if 'generators_carrier' not in self.ds:
            logger.warning("No generator data found for visualization")
            return
        
        carriers = self.ds['generators_carrier'].values
        capacities = self.ds['generators_p_nom'].values
        
        # Group by carrier
        carrier_df = pd.DataFrame({
            'carrier': carriers,
            'capacity': capacities
        })
        
        capacity_by_carrier = carrier_df.groupby('carrier')['capacity'].sum().sort_values(ascending=False)
        
        # Create pie chart
        plt.figure(figsize=(12, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(capacity_by_carrier)))
        
        plt.pie(capacity_by_carrier.values, 
                labels=capacity_by_carrier.index,
                autopct='%1.1f%%',
                colors=colors,
                startangle=90)
        
        plt.title(f'Generation Mix by Technology\nTotal Capacity: {capacity_by_carrier.sum():,.0f} MW')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generation mix plot saved to {output_file}")
    
    def create_load_profile_plot(self, output_file: str = 'load_profile.png', num_days: int = 7):
        """Create load profile visualization."""
        if 'loads_t_p_set' not in self.ds:
            logger.warning("No load data found for visualization")
            return
        
        load_data = self.ds['loads_t_p_set'].values
        if load_data.ndim > 1:
            total_load = load_data.sum(axis=1)
        else:
            total_load = load_data
        
        # Plot first week
        hours_to_plot = min(num_days * 24, len(total_load))
        time_range = range(hours_to_plot)
        
        plt.figure(figsize=(14, 6))
        plt.plot(time_range, total_load[:hours_to_plot], linewidth=1.5)
        plt.title(f'Load Profile - First {num_days} Days')
        plt.xlabel('Hour')
        plt.ylabel('Load (MW)')
        plt.grid(True, alpha=0.3)
        
        # Add day markers
        for day in range(1, num_days + 1):
            plt.axvline(x=day * 24, color='red', linestyle='--', alpha=0.5)
            plt.text(day * 24, max(total_load[:hours_to_plot]) * 0.9, f'Day {day+1}', 
                    rotation=90, verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Load profile plot saved to {output_file}")
    
    def create_capacity_factor_heatmap(self, output_file: str = 'capacity_factors.png'):
        """Create capacity factor heatmap for renewable generators."""
        if 'generators_t_p_max_pu' not in self.ds:
            logger.warning("No capacity factor data found for visualization")
            return
        
        cf_data = self.ds['generators_t_p_max_pu']
        gen_names = self.ds.coords['generators_t_p_max_pu_i'].values
        
        # Select renewable generators (assuming they have variable capacity factors)
        renewable_gens = []
        for gen in gen_names:
            if gen in self.ds.coords['generators_i'].values:
                carrier = str(self.ds['generators_carrier'].sel(generators_i=gen).values)
                if carrier in ['solar_pv', 'wind_onshore', 'wind_offshore']:
                    renewable_gens.append(gen)
        
        if not renewable_gens:
            logger.warning("No renewable generators found for capacity factor visualization")
            return
        
        # Sample data for visualization (first month)
        sample_hours = min(24 * 30, len(self.ds.coords['snapshots']))  # First 30 days
        cf_sample = cf_data.sel(generators_t_p_max_pu_i=renewable_gens[:10]).values[:sample_hours, :]
        
        plt.figure(figsize=(14, 8))
        sns.heatmap(cf_sample.T, 
                   cmap='viridis',
                   cbar_kws={'label': 'Capacity Factor'},
                   yticklabels=renewable_gens[:10],
                   xticklabels=False)
        
        plt.title('Renewable Generation Capacity Factors (First 30 Days)')
        plt.xlabel('Hour')
        plt.ylabel('Generator')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Capacity factor heatmap saved to {output_file}")
    
    def create_validation_summary_plot(self, validation_results: Dict[str, Any], 
                                     output_file: str = 'validation_summary.png'):
        """Create validation results summary plot."""
        if not validation_results:
            logger.warning("No validation results provided")
            return
        
        # Extract validation status for each category
        categories = []
        statuses = []
        
        for category, result in validation_results.items():
            if category != 'overall_passed' and isinstance(result, dict):
                categories.append(category.replace('_', ' ').title())
                statuses.append('Pass' if result.get('passed', False) else 'Fail')
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        colors = ['green' if status == 'Pass' else 'red' for status in statuses]
        bars = plt.bar(categories, [1] * len(categories), color=colors, alpha=0.7)
        
        # Add status text on bars
        for bar, status in zip(bars, statuses):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                    status, ha='center', va='center', fontweight='bold', color='white')
        
        plt.title('Conversion Validation Results')
        plt.ylabel('Validation Status')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.2)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add overall status
        overall_status = 'PASSED' if validation_results.get('overall_passed', False) else 'FAILED'
        plt.text(0.5, 1.1, f'Overall: {overall_status}', 
                transform=plt.gca().transAxes, ha='center', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Validation summary plot saved to {output_file}")


def main():
    """Main function demonstrating utility usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='PyPSA-ZA Conversion Utilities')
    parser.add_argument('--analyze', type=str, help='Analyze PyPSA network file')
    parser.add_argument('--validate', nargs=2, help='Validate conversion: <original_pypsa> <sienna_dir>')
    parser.add_argument('--visualize', type=str, help='Create visualizations for PyPSA file')
    parser.add_argument('--output', '-o', type=str, default='.', help='Output directory')
    
    args = parser.parse_args()
    
    if args.analyze:
        print("Analyzing PyPSA network...")
        analyzer = PyPSAAnalyzer(args.analyze)
        report = analyzer.generate_summary_report(f"{args.output}/analysis_report.txt")
        print(report)
        
    elif args.validate:
        print("Validating conversion...")
        pypsa_file, sienna_dir = args.validate
        validator = SiennaValidator(sienna_dir)
        results = validator.validate_conversion(pypsa_file)
        
        print(f"Validation Results:")
        print(f"Overall Passed: {results['overall_passed']}")
        for category, result in results.items():
            if category != 'overall_passed':
                print(f"  {category}: {'PASS' if result.get('passed', False) else 'FAIL'}")
        
    elif args.visualize:
        print("Creating visualizations...")
        visualizer = ConversionVisualizer(args.visualize)
        visualizer.create_generation_mix_plot(f"{args.output}/generation_mix.png")
        visualizer.create_load_profile_plot(f"{args.output}/load_profile.png")
        visualizer.create_capacity_factor_heatmap(f"{args.output}/capacity_factors.png")
        print(f"Visualizations saved to {args.output}/")


if __name__ == "__main__":
    main()