#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone debug script for PyPSA-ZA spatial join issues.
Run this script independently to debug spatial join problems.

Usage:
    python debug_spatial_joins.py --rule build_topology --scenario CNS_G_RB_CB_10_7
    python debug_spatial_joins.py --rule add_electricity --scenario AMBITIONS_LC2 --model_type capacity
"""

import sys
import os
import logging
import argparse
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('spatial_debug.log')
    ]
)

# Add the scripts directory to Python path
sys.path.append(str(Path(__file__).parent))

# Import mock_snakemake from _helpers
try:
    from _helpers import mock_snakemake
except ImportError:
    logging.error("Could not import mock_snakemake. Make sure _helpers.py is in the same directory.")
    sys.exit(1)


class SpatialJoinDebugger:
    """Debug spatial joins in PyPSA-ZA workflows"""
    
    def __init__(self, rule_name, **wildcards):
        """Initialize debugger with snakemake rule and wildcards"""
        self.rule_name = rule_name
        self.wildcards = wildcards
        self.snakemake = mock_snakemake(rule_name, **wildcards)
        self.results = {}
        
    def debug_load_region_data(self):
        """Debug the load_region_data function from build_topology.py"""
        logging.info("\n" + "="*60)
        logging.info("DEBUGGING: load_region_data (build_topology.py)")
        logging.info("="*60)
        
        # Get model regions from scenario setup
        try:
            from _helpers import load_scenario_definition
            scenario_setup = load_scenario_definition(self.snakemake)
            model_regions = str(scenario_setup.loc["regions"])
        except Exception as e:
            logging.error(f"Could not load scenario setup: {e}")
            model_regions = self.wildcards.get('regions', '10')
        
        # Load regions
        logging.info(f"\n1. Loading regions (layer: {model_regions})...")
        try:
            regions = gpd.read_file(
                self.snakemake.input.supply_regions,
                layer=model_regions,
            )
            logging.info(f"   ✓ Loaded regions: shape={regions.shape}")
            logging.info(f"   Original CRS: {regions.crs}")
            logging.info(f"   Columns: {regions.columns.tolist()}")
            
            # Check for index columns
            possible_index_cols = ['name', 'Name', 'LocalArea', 'SupplyArea', 'region_name']
            found_index_cols = [col for col in possible_index_cols if col in regions.columns]
            logging.info(f"   Found index columns: {found_index_cols}")
            
            # Transform to distance CRS
            regions = regions.to_crs(self.snakemake.config["gis"]["crs"]["distance_crs"])
            logging.info(f"   Transformed CRS: {regions.crs}")
            logging.info(f"   Bounds: {regions.total_bounds}")
            
            # Set index
            if found_index_cols:
                regions = regions.set_index(found_index_cols[0])
                regions.index.name = 'name'
                logging.info(f"   Set index to: {found_index_cols[0]}")
                logging.info(f"   Sample index values: {regions.index[:5].tolist()}")
            
            self.results['regions'] = regions
            
        except Exception as e:
            logging.error(f"   ✗ Error loading regions: {e}")
            return
        
        # Load GDP/population data
        logging.info("\n2. Loading GDP/population data...")
        try:
            gdp_pop = gpd.read_file(self.snakemake.input.gdp_pop_data)
            logging.info(f"   ✓ Loaded GDP/pop data: shape={gdp_pop.shape}")
            logging.info(f"   Original CRS: {gdp_pop.crs}")
            logging.info(f"   Columns: {gdp_pop.columns.tolist()}")
            
            # Transform to same CRS as regions
            gdp_pop = gdp_pop.to_crs(regions.crs)
            logging.info(f"   Transformed CRS: {gdp_pop.crs}")
            logging.info(f"   Bounds: {gdp_pop.total_bounds}")
            
            self.results['gdp_pop'] = gdp_pop
            
        except Exception as e:
            logging.error(f"   ✗ Error loading GDP/pop data: {e}")
            return
        
        # Check spatial overlap
        logging.info("\n3. Checking spatial overlap...")
        regions_union = regions.unary_union
        gdp_pop_union = gdp_pop.unary_union
        
        if regions_union.intersects(gdp_pop_union):
            logging.info("   ✓ Geometries intersect!")
            intersection = regions_union.intersection(gdp_pop_union)
            logging.info(f"   Intersection area: {intersection.area:,.0f} sq units")
        else:
            logging.error("   ✗ No intersection found between regions and GDP/pop data!")
        
        # Test different spatial join methods
        logging.info("\n4. Testing spatial joins...")
        self._test_spatial_joins(gdp_pop, regions)
        
        # Create visualization
        self._create_spatial_plot(regions, gdp_pop, "load_region_data")
        
    def debug_map_components_to_buses(self):
        """Debug the map_components_to_buses function from add_electricity.py"""
        logging.info("\n" + "="*60)
        logging.info("DEBUGGING: map_components_to_buses (add_electricity.py)")
        logging.info("="*60)
        
        # First, we need to load some generators to test with
        # This mimics what happens in add_electricity.py
        try:
            from _helpers import load_scenario_definition, get_carriers_from_model_file
            from add_electricity import load_fixed_components
            
            scenario_setup = load_scenario_definition(self.snakemake)
            carriers = get_carriers_from_model_file(scenario_setup)
            
            # Load some generators
            conv_carriers = carriers["fixed"]["conventional"]
            if conv_carriers:
                logging.info(f"\n1. Loading fixed conventional generators...")
                gens = load_fixed_components(
                    conv_carriers[:3],  # Just load first 3 carriers for testing
                    2024,  # start_year
                    self.snakemake.config["electricity"],
                    "Generator"
                )
                logging.info(f"   ✓ Loaded {len(gens)} generators")
                logging.info(f"   Sample coordinates:")
                for idx in gens.index[:5]:
                    logging.info(f"   {idx}: x={gens.loc[idx, 'x']}, y={gens.loc[idx, 'y']}")
                
                self.results['generators'] = gens
            else:
                logging.warning("No conventional carriers found")
                return
                
        except Exception as e:
            logging.error(f"Could not load generators: {e}")
            return
        
        # Load regions for mapping
        try:
            regions_gdf = gpd.read_file(self.snakemake.input.supply_regions)
            regions_gdf = regions_gdf.to_crs(self.snakemake.config["gis"]["crs"]["distance_crs"])
            logging.info(f"\n2. Loaded regions: shape={regions_gdf.shape}")
            
            # Set proper index
            possible_name_cols = ['name', 'Name', 'LocalArea', 'SupplyArea', 'region_name']
            name_col = None
            for col in possible_name_cols:
                if col in regions_gdf.columns:
                    name_col = col
                    break
            
            if name_col:
                regions_gdf = regions_gdf.set_index(name_col)
                logging.info(f"   Set index to: {name_col}")
            
            self.results['regions_for_mapping'] = regions_gdf
            
        except Exception as e:
            logging.error(f"Could not load regions: {e}")
            return
        
        # Test different Point creation methods
        logging.info("\n3. Testing Point creation methods...")
        self._test_point_creation(gens, regions_gdf)
        
        # Create visualization
        self._create_component_mapping_plot(gens, regions_gdf)
        
    def _test_spatial_joins(self, gdp_pop, regions):
        """Test different spatial join methods"""
        
        # Reset index on regions for testing
        regions_reset = regions.reset_index()
        
        test_cases = [
            ("Default sjoin", 
             lambda: gpd.sjoin(gdp_pop, regions_reset, predicate="within")),
            
            ("Explicit left join", 
             lambda: gpd.sjoin(gdp_pop, regions_reset, how="left", predicate="within")),
            
            ("Explicit right join", 
             lambda: gpd.sjoin(gdp_pop, regions_reset, how="right", predicate="within")),
            
            ("With suffix parameters", 
             lambda: gpd.sjoin(gdp_pop, regions_reset, predicate="within", 
                              lsuffix="left", rsuffix="right")),
            
            ("Using intersects", 
             lambda: gpd.sjoin(gdp_pop, regions_reset, predicate="intersects")),
            
            ("Using overlay", 
             lambda: gpd.overlay(gdp_pop, regions_reset, how='intersection')),
        ]
        
        for test_name, test_func in test_cases:
            logging.info(f"\n   Testing: {test_name}")
            try:
                result = test_func()
                logging.info(f"   ✓ Success! Shape: {result.shape}")
                logging.info(f"   Columns: {result.columns.tolist()}")
                
                # Check for index columns
                index_cols = [col for col in result.columns if 'index' in col.lower()]
                logging.info(f"   Index columns: {index_cols}")
                
                # Check for the region name column
                if 'name' in result.columns:
                    logging.info(f"   ✓ 'name' column exists")
                if 'index_right' in result.columns:
                    logging.info(f"   ✓ 'index_right' column exists")
                    unique_regions = result['index_right'].nunique()
                    logging.info(f"   Unique regions in join: {unique_regions}")
                
                # Save first successful result
                if test_name == "Default sjoin":
                    self.results['joined'] = result
                    
            except Exception as e:
                logging.error(f"   ✗ Failed: {e}")
        
    def _test_point_creation(self, component_df, regions_gdf):
        """Test different ways of creating points from coordinates"""
        
        # Sample 5 components
        sample_df = component_df.sample(min(5, len(component_df)))
        
        logging.info("\n   Testing coordinate order...")
        
        # Test 1: Point(x, y) - standard order
        points_xy = []
        for idx, row in sample_df.iterrows():
            p = Point(row['x'], row['y'])
            points_xy.append(p)
            logging.info(f"   Point(x={row['x']}, y={row['y']}) -> {p}")
        
        # Test 2: Point(y, x) - swapped order
        points_yx = []
        for idx, row in sample_df.iterrows():
            p = Point(row['y'], row['x'])
            points_yx.append(p)
            logging.info(f"   Point(y={row['y']}, x={row['x']}) -> {p}")
        
        # Create GeoDataFrames
        gdf_xy = gpd.GeoDataFrame(geometry=points_xy, index=sample_df.index, 
                                  crs=self.snakemake.config["gis"]["crs"]["geo_crs"])
        gdf_yx = gpd.GeoDataFrame(geometry=points_yx, index=sample_df.index,
                                  crs=self.snakemake.config["gis"]["crs"]["geo_crs"])
        
        # Transform to regions CRS
        gdf_xy = gdf_xy.to_crs(regions_gdf.crs)
        gdf_yx = gdf_yx.to_crs(regions_gdf.crs)
        
        # Test which points fall within regions
        within_xy = gdf_xy.within(regions_gdf.unary_union).sum()
        within_yx = gdf_yx.within(regions_gdf.unary_union).sum()
        
        logging.info(f"\n   Results:")
        logging.info(f"   Points within regions using Point(x,y): {within_xy}/{len(gdf_xy)}")
        logging.info(f"   Points within regions using Point(y,x): {within_yx}/{len(gdf_yx)}")
        
        if within_xy > within_yx:
            logging.info("   ✓ Recommendation: Use Point(x, y)")
        else:
            logging.info("   ✓ Recommendation: Use Point(y, x)")
        
        self.results['point_test'] = {
            'xy_within': within_xy,
            'yx_within': within_yx,
            'total': len(gdf_xy)
        }
        
    def _create_spatial_plot(self, regions, gdp_pop, title_suffix):
        """Create visualization of spatial data"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
            
            # Plot 1: Regions
            regions.plot(ax=ax1, edgecolor='black', facecolor='lightblue', alpha=0.5)
            ax1.set_title(f"Regions ({len(regions)} total)")
            ax1.set_xlabel("X coordinate")
            ax1.set_ylabel("Y coordinate")
            
            # Plot 2: GDP/Pop data with regions
            regions.plot(ax=ax2, edgecolor='black', facecolor='none', linewidth=2)
            if 'gdp_pop' in self.results:
                gdp_pop.plot(ax=ax2, markersize=1, color='red', alpha=0.5)
            ax2.set_title("GDP/Population points overlaid on regions")
            ax2.set_xlabel("X coordinate")
            ax2.set_ylabel("Y coordinate")
            
            plt.tight_layout()
            filename = f'debug_spatial_{title_suffix}.png'
            plt.savefig(filename, dpi=150)
            logging.info(f"\n   ✓ Saved visualization to {filename}")
            plt.close()
            
        except Exception as e:
            logging.warning(f"   Could not create plot: {e}")
    
    def _create_component_mapping_plot(self, components, regions):
        """Create visualization of component mapping"""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            
            # Plot regions
            regions.plot(ax=ax, edgecolor='black', facecolor='lightgray', alpha=0.5)
            
            # Plot components
            if 'x' in components.columns and 'y' in components.columns:
                # Create points - test both orders
                points_xy = [Point(row.x, row.y) for _, row in components.iterrows()]
                gdf_xy = gpd.GeoDataFrame(geometry=points_xy, crs="EPSG:4326")
                gdf_xy = gdf_xy.to_crs(regions.crs)
                
                gdf_xy.plot(ax=ax, color='red', markersize=50, alpha=0.7, 
                           label=f'Components ({len(components)})')
            
            ax.set_title("Component locations vs Region boundaries")
            ax.set_xlabel("X coordinate")
            ax.set_ylabel("Y coordinate")
            ax.legend()
            
            plt.tight_layout()
            filename = 'debug_component_mapping.png'
            plt.savefig(filename, dpi=150)
            logging.info(f"\n   ✓ Saved component mapping plot to {filename}")
            plt.close()
            
        except Exception as e:
            logging.warning(f"   Could not create component plot: {e}")
    
    def check_geopandas_settings(self):
        """Check GeoPandas version and settings"""
        logging.info("\n" + "="*60)
        logging.info("GEOPANDAS ENVIRONMENT CHECK")
        logging.info("="*60)
        
        logging.info(f"\nGeoPandas version: {gpd.__version__}")
        logging.info(f"Shapely version: {gpd.io.shapely.__version__}")
        logging.info(f"Pandas version: {pd.__version__}")
        
        # Test simple spatial join behavior
        logging.info("\nTesting default spatial join behavior...")
        
        # Create test data
        points = gpd.GeoDataFrame(
            {'id': [1, 2, 3], 'value': ['a', 'b', 'c']},
            geometry=[Point(0, 0), Point(1, 1), Point(2, 2)]
        )
        
        polygons = gpd.GeoDataFrame(
            {'name': ['poly1', 'poly2'], 'data': [10, 20]},
            geometry=[Point(0, 0).buffer(1.5), Point(2, 2).buffer(1.5)]
        )
        polygons = polygons.set_index('name')
        
        # Test join
        result = gpd.sjoin(points, polygons, predicate='within')
        logging.info(f"Test join columns: {result.columns.tolist()}")
        logging.info(f"Has 'index_right': {'index_right' in result.columns}")
        
        self.results['geopandas_test'] = result
        
    def run_all_tests(self):
        """Run all debug tests"""
        self.check_geopandas_settings()
        
        if self.rule_name == 'build_topology':
            self.debug_load_region_data()
        elif self.rule_name == 'add_electricity':
            self.debug_map_components_to_buses()
        else:
            logging.warning(f"No specific debug tests for rule: {self.rule_name}")
        
        # Save results
        self._save_results()
        
    def _save_results(self):
        """Save debug results to file"""
        import pickle
        
        filename = f'debug_results_{self.rule_name}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self.results, f)
        
        logging.info(f"\n✓ Debug results saved to {filename}")
        logging.info("\nDEBUG COMPLETE - Check spatial_debug.log for full details")


def main():
    """Main function to run debug"""
    parser = argparse.ArgumentParser(description='Debug PyPSA-ZA spatial joins')
    parser.add_argument('--rule', type=str, required=True, 
                      help='Snakemake rule to debug (e.g., build_topology, add_electricity)')
    parser.add_argument('--scenario', type=str, default='CNS_G_RB_CB_10_7',
                      help='Scenario name')
    parser.add_argument('--model_type', type=str, default='capacity',
                      help='Model type (capacity/dispatch)')
    parser.add_argument('--regions', type=str, default='10',
                      help='Number of regions')
    
    args = parser.parse_args()
    
    # Build wildcards
    wildcards = {
        'scenario': args.scenario,
        'model_type': args.model_type,
    }
    
    if args.rule == 'build_topology':
        # Only scenario needed for build_topology
        wildcards = {'scenario': args.scenario}
    
    # Create and run debugger
    debugger = SpatialJoinDebugger(args.rule, **wildcards)
    debugger.run_all_tests()


if __name__ == "__main__":
    main()