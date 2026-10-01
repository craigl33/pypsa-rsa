#!/usr/bin/env python3
"""
Comprehensive NetCDF file investigation script.
This will help diagnose why a 600MB+ NetCDF file appears empty in xarray.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

def investigate_netcdf_file(filepath):
    """
    Comprehensive investigation of a NetCDF file that appears empty.
    """
    filepath = Path(filepath)
    
    print("=" * 80)
    print(f"INVESTIGATING: {filepath}")
    print("=" * 80)
    
    # Basic file info
    print("\n1. BASIC FILE INFORMATION")
    print("-" * 40)
    try:
        stat = filepath.stat()
        print(f"File size: {stat.st_size:,} bytes ({stat.st_size/1024/1024:.1f} MB)")
        print(f"File exists: {filepath.exists()}")
        print(f"Is file: {filepath.is_file()}")
        print(f"Permissions: {oct(stat.st_mode)[-3:]}")
        print(f"Last modified: {pd.to_datetime(stat.st_mtime, unit='s')}")
    except Exception as e:
        print(f"Error getting file info: {e}")
        return
    
    # Check if it's actually a NetCDF file
    print("\n2. FILE FORMAT DETECTION")
    print("-" * 40)
    try:
        with open(filepath, 'rb') as f:
            header = f.read(100)
            print(f"First 20 bytes (hex): {header[:20].hex()}")
            print(f"First 20 bytes (ascii): {header[:20]}")
            
            # NetCDF magic numbers
            if header.startswith(b'CDF\x01'):
                print("✓ Classic NetCDF format detected")
            elif header.startswith(b'CDF\x02'):
                print("✓ 64-bit offset NetCDF format detected")
            elif header.startswith(b'\x89HDF'):
                print("✓ HDF5/NetCDF4 format detected")
            else:
                print("⚠ Unknown file format - may not be NetCDF")
                
    except Exception as e:
        print(f"Error reading file header: {e}")
    
    # Try different NetCDF libraries
    print("\n3. LIBRARY COMPATIBILITY TESTS")
    print("-" * 40)
    
    # Test with netCDF4 library directly
    try:
        import netCDF4
        print(f"netCDF4 version: {netCDF4.__version__}")
        
        with netCDF4.Dataset(filepath, 'r') as nc:
            print(f"✓ netCDF4.Dataset can open file")
            print(f"  Dimensions: {dict(nc.dimensions)}")
            print(f"  Variables: {list(nc.variables.keys())}")
            print(f"  Groups: {list(nc.groups.keys())}")
            print(f"  Global attributes: {list(nc.ncattrs())}")
            
            # Check each group
            if nc.groups:
                print(f"  Found {len(nc.groups)} groups:")
                for i, (group_name, group) in enumerate(nc.groups.items()):
                    if i < 5:  # Show first 5 groups
                        print(f"    {group_name}: {dict(group.dimensions)} dims, {list(group.variables.keys())} vars")
                    elif i == 5:
                        print(f"    ... and {len(nc.groups) - 5} more groups")
                        break
            
    except ImportError:
        print("❌ netCDF4 library not available")
    except Exception as e:
        print(f"❌ netCDF4 error: {e}")
    
    # Test with xarray
    try:
        import xarray as xr
        print(f"xarray version: {xr.__version__}")
        
        # Try different xarray backends
        backends = ['netcdf4', 'h5netcdf', 'scipy']
        
        for backend in backends:
            try:
                print(f"  Testing {backend} backend...")
                ds = xr.open_dataset(filepath, engine=backend)
                print(f"    ✓ {backend}: {len(ds.dims)} dims, {len(ds.data_vars)} vars, {len(ds.coords)} coords")
                print(f"      Dimensions: {dict(ds.dims)}")
                print(f"      Variables: {list(ds.data_vars.keys())}")
                
                # Check for groups
                if hasattr(ds, 'groups'):
                    print(f"      Groups: {list(ds.groups.keys())}")
                
                ds.close()
                break
                
            except Exception as e:
                print(f"    ❌ {backend}: {e}")
    
    except ImportError:
        print("❌ xarray not available")
    
    # Test with h5py (for HDF5/NetCDF4 files)
    try:
        import h5py
        print(f"h5py version: {h5py.__version__}")
        
        with h5py.File(filepath, 'r') as h5:
            print(f"✓ h5py can open file")
            print(f"  Root keys: {list(h5.keys())}")
            print(f"  Root attrs: {list(h5.attrs.keys())}")
            
            def print_h5_structure(name, obj):
                if isinstance(obj, h5py.Group):
                    print(f"    Group: {name} (keys: {list(obj.keys())})")
                elif isinstance(obj, h5py.Dataset):
                    print(f"    Dataset: {name} {obj.shape} {obj.dtype}")
            
            print("  Structure (first 10 items):")
            count = 0
            h5.visititems(lambda name, obj: print_h5_structure(name, obj) if count < 10 else None)
            
    except ImportError:
        print("❌ h5py not available")
    except Exception as e:
        print(f"❌ h5py error: {e}")
    
    # Try manual inspection of file structure
    print("\n4. MANUAL FILE INSPECTION")
    print("-" * 40)
    
    try:
        # Look for common NetCDF/HDF5 patterns in the file
        with open(filepath, 'rb') as f:
            content = f.read(10000)  # Read first 10KB
            
            # Look for dimension names
            common_dims = [b'time', b'lat', b'lon', b'bus', b'year', b'hour']
            found_dims = []
            for dim in common_dims:
                if dim in content:
                    found_dims.append(dim.decode())
            
            if found_dims:
                print(f"  Found dimension names: {found_dims}")
            
            # Look for variable names
            common_vars = [b'wind', b'solar', b'temperature', b'radiation']
            found_vars = []
            for var in common_vars:
                if var in content:
                    found_vars.append(var.decode())
            
            if found_vars:
                print(f"  Found variable names: {found_vars}")
            
            # Check for compression indicators
            if b'DEFLATE' in content or b'GZIP' in content:
                print("  ✓ File appears to use compression")
            
    except Exception as e:
        print(f"Error in manual inspection: {e}")
    
    print("\n5. RECOMMENDATIONS")
    print("-" * 40)
    print("Based on the investigation:")
    print("1. Check if file is corrupted or incomplete download")
    print("2. Try opening with different library backends")
    print("3. Verify the source/download process")
    print("4. Check if file requires specific xarray options")


def test_xarray_group_access(filepath):
    """
    Specifically test xarray group access patterns that might be used in your code.
    """
    print("\n" + "=" * 80)
    print("TESTING XARRAY GROUP ACCESS PATTERNS")
    print("=" * 80)
    
    import xarray as xr
    
    # Test patterns from your code
    test_patterns = [
        'solar_pv_10_era5',
        'wind_10_era5', 
        'solar_pv_rooftop_10_era5',
        'wind_onshore_10_era5',
        'solar_pv_fixed_era5',
        'wind_fixed_era5'
    ]
    
    for pattern in test_patterns:
        try:
            print(f"Testing group: {pattern}")
            data = xr.open_dataarray(filepath, group=pattern)
            print(f"  ✓ Success: {data.dims}, shape {data.shape}")
            data.close()
        except Exception as e:
            print(f"  ❌ Failed: {e}")


if __name__ == "__main__":
    filepath = "pre_processing/resource_processing/renewable_profiles_updated.nc"
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        # Try to find similar files
        import glob
        similar = glob.glob("**/*renewable*.nc", recursive=True)
        if similar:
            print(f"Found similar files: {similar}")
            filepath = similar[0]
        else:
            print("No renewable profile files found")
            sys.exit(1)
    
    investigate_netcdf_file(filepath)
    
    # Test specific group access if xarray is available
    try:
        test_xarray_group_access(filepath)
    except Exception as e:
        print(f"Could not test group access: {e}")
