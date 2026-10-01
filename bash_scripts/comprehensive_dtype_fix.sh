#!/bin/bash

echo "🔧 Applying comprehensive data type fix for NetCDF export..."

cd /home/clhar/showcase/pypsa-rsa

# Backup first
cp scripts/add_electricity.py scripts/add_electricity.py.backup_dtype

# Create a Python script to add the comprehensive fix
cat > add_dtype_fix.py << 'EOF'
# Read the file
with open('scripts/add_electricity.py', 'r') as f:
    content = f.read()

# Find the line "logging.info("Exporting network.")" and add the fix after it
export_line = 'logging.info("Exporting network.")'
if export_line in content:
    # Create the comprehensive data type fix
    dtype_fix = '''
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
    
    ensure_consistent_dtypes(n)'''
    
    # Insert the fix right after the "Exporting network." log line
    content = content.replace(
        export_line,
        export_line + dtype_fix
    )
    
    # Write back
    with open('scripts/add_electricity.py', 'w') as f:
        f.write(content)
    
    print("✅ Added comprehensive data type fix")
else:
    print("❌ Could not find the export section to add the fix")

EOF

python add_dtype_fix.py
rm add_dtype_fix.py

echo "✅ Applied comprehensive data type fix"

# Show what was added
echo ""
echo "Added comprehensive data type conversion function before NetCDF export."
echo "This ensures all time-series data has consistent float types."

echo ""
echo "🚀 Now try running: snakemake -R add_electricity"
echo "If it still fails, restore with: cp scripts/add_electricity.py.backup_dtype scripts/add_electricity.py"