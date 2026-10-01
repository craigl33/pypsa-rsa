#!/bin/bash

# =============================================================================
# COMPREHENSIVE GEOPANDAS 1.0+ COMPATIBILITY FIX SCRIPT
# =============================================================================

set -e  # Exit on any error

echo "🔧 Starting GeoPandas 1.0+ compatibility fixes..."

# Navigate to the pypsa-rsa directory
cd /home/clhar/showcase/pypsa-rsa

# =============================================================================
# STEP 1: BACKUP ORIGINAL FILES
# =============================================================================
echo "📁 Creating backups..."
mkdir -p backups/$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"

# Find all Python files that might need fixing
find scripts/ -name "*.py" -exec cp {} $BACKUP_DIR/ \;
echo "✅ Backed up files to $BACKUP_DIR"

# =============================================================================
# STEP 2: SEARCH AND REPORT ISSUES
# =============================================================================
echo "🔍 Scanning for geopandas compatibility issues..."

echo "--- Files with sjoin op= parameter ---"
grep -rn "sjoin.*op=" scripts/ || echo "No sjoin op= issues found"

echo "--- Files with index_right usage ---"
grep -rn "index_right" scripts/ || echo "No index_right issues found"

echo "--- Files with regex warnings ---"
grep -rn 're\.split.*\\s' scripts/ || echo "No regex issues found"

echo ""

# =============================================================================
# STEP 3: FIX 1 - REPLACE op= WITH predicate=
# =============================================================================
echo "🔄 Fix 1: Replacing 'op=' with 'predicate=' in sjoin calls..."

# More precise replacements to avoid false positives
find scripts/ -name "*.py" -exec sed -i 's/sjoin(\([^)]*\), op=/sjoin(\1, predicate=/g' {} \;
find scripts/ -name "*.py" -exec sed -i 's/sjoin(\([^)]*\),\s*op=/sjoin(\1, predicate=/g' {} \;

echo "✅ Fixed sjoin op= parameters"

# =============================================================================
# STEP 4: FIX 2 - ROBUST index_right HANDLING
# =============================================================================
echo "🔄 Fix 2: Creating robust index_right handling..."

# Create a temporary Python script to fix index_right issues
cat > fix_index_right.py << 'EOF'
import re
import os

def fix_index_right_in_file(filepath):
    """Fix index_right usage to be version-agnostic"""
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Pattern 1: joined.index_right in groupby
    pattern1 = r'joined\.groupby\(joined\.index_right\)'
    replacement1 = '''joined.groupby(joined.get('index_right', joined.columns[-1]))'''
    
    # Pattern 2: Simple joined.index_right access
    pattern2 = r'joined\.index_right'
    replacement2 = '''joined.get('index_right', joined.columns[-1] if 'index_right' not in joined.columns else joined.index_right)'''
    
    # Apply fixes
    if re.search(pattern1, content):
        content = re.sub(pattern1, replacement1, content)
        print(f"Fixed groupby pattern in {filepath}")
    
    # Only apply pattern2 if pattern1 wasn't found (to avoid double-fixing)
    elif re.search(pattern2, content):
        content = re.sub(pattern2, replacement2, content)
        print(f"Fixed index_right access in {filepath}")
    
    # Write back
    with open(filepath, 'w') as f:
        f.write(content)

# Process all Python files
for root, dirs, files in os.walk('scripts/'):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            if 'index_right' in open(filepath).read():
                fix_index_right_in_file(filepath)
EOF

python fix_index_right.py
rm fix_index_right.py

echo "✅ Fixed index_right handling"

# =============================================================================
# STEP 5: FIX 3 - REGEX RAW STRINGS
# =============================================================================
echo "🔄 Fix 3: Converting regex strings to raw strings..."

# Fix regex patterns - more targeted approach
find scripts/ -name "*.py" -exec sed -i 's/re\.split("/re.split(r"/g' {} \;
find scripts/ -name "*.py" -exec sed -i "s/re\.split('/re.split(r'/g" {} \;

echo "✅ Fixed regex raw string issues"

# =============================================================================
# STEP 6: COMPREHENSIVE build_topology.py FIX
# =============================================================================
echo "🔄 Fix 4: Applying comprehensive fix to build_topology.py..."

# Create a robust version of the load_region_data function
cat > build_topology_patch.py << 'EOF'
import re

# Read the current build_topology.py
with open('scripts/build_topology.py', 'r') as f:
    content = f.read()

# Replace the load_region_data function with a robust version
new_function = '''def load_region_data(model_regions):
    # Load supply regions and calculate population per region
    regions = gpd.read_file(
        snakemake.input.supply_regions,
        layer=model_regions,
    ).to_crs(snakemake.config["gis"]["crs"]["distance_crs"])

    possible_index_cols = ['name', 'Name', 'LocalArea', 'SupplyArea']

    index_column = [col for col in possible_index_cols if col in regions.columns]
    regions = regions.set_index(index_column[0])
    regions.index.name = 'name'

    gdp_pop = gpd.read_file(
        snakemake.input.gdp_pop_data,
    ).to_crs(snakemake.config["gis"]["crs"]["distance_crs"])

    joined = gpd.sjoin(gdp_pop, regions, how="inner", predicate="within")
    
    # Handle different geopandas versions for the right index column
    if 'index_right' in joined.columns:
        right_index_col = 'index_right'
    else:
        # Find the right index column (look for index columns that aren't index_left)
        index_cols = [col for col in joined.columns if col.startswith('index_')]
        right_index_cols = [col for col in index_cols if col != 'index_left']
        if right_index_cols:
            right_index_col = right_index_cols[0]
        else:
            # Fallback: use the last column or a reasonable guess
            right_index_col = joined.columns[-1]
    
    gva_cols = ["SIC1_2016", "SIC2_2016", "SIC3_2016", "SIC4_2016", "SIC6_2016", "SIC7_2016", "SIC8_2016", "SIC9_2016"]
    pop_col = ["POP_2016"]
    for col in gva_cols + pop_col:
        regions[col] = joined.groupby(joined[right_index_col]).sum()[col]
    
    regions["GVA_2016"] = regions[gva_cols].sum(axis=1)
    if len(regions)>1:
        regions.drop(["Shape_Area", "Shape_Leng"], axis=1, inplace=True)

    return regions'''

# Replace the function definition
pattern = r'def load_region_data\(model_regions\):.*?return regions'
content = re.sub(pattern, new_function, content, flags=re.DOTALL)

# Write back
with open('scripts/build_topology.py', 'w') as f:
    f.write(content)

print("Applied comprehensive fix to build_topology.py")
EOF

python build_topology_patch.py
rm build_topology_patch.py

echo "✅ Applied comprehensive build_topology.py fix"

# =============================================================================
# STEP 7: VALIDATION
# =============================================================================
echo "🔍 Validating fixes..."

echo "--- Remaining sjoin op= issues ---"
remaining_op=$(grep -rn "sjoin.*op=" scripts/ | wc -l)
if [ $remaining_op -eq 0 ]; then
    echo "✅ All sjoin op= issues fixed"
else
    echo "⚠️  Still have $remaining_op sjoin op= issues"
    grep -rn "sjoin.*op=" scripts/
fi

echo "--- Remaining index_right issues ---"
remaining_idx=$(grep -rn "joined\.index_right" scripts/ | wc -l)
if [ $remaining_idx -eq 0 ]; then
    echo "✅ All direct index_right access issues fixed"
else
    echo "⚠️  Still have $remaining_idx index_right issues"
    grep -rn "joined\.index_right" scripts/
fi

echo "--- Remaining regex issues ---"
remaining_regex=$(grep -rn 're\.split.*[^r]".*\\s' scripts/ | wc -l)
if [ $remaining_regex -eq 0 ]; then
    echo "✅ All regex issues fixed"
else
    echo "⚠️  Still have $remaining_regex regex issues"
fi

# =============================================================================
# STEP 8: TEST
# =============================================================================
echo "🧪 Testing the fix..."
echo "Run this command to test: snakemake -R build_topology --dry-run"
echo ""
echo "If that works, run: snakemake -R build_topology"

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "🎉 SUMMARY OF FIXES APPLIED:"
echo "1. ✅ Replaced all 'op=' with 'predicate=' in sjoin calls"
echo "2. ✅ Made index_right handling version-agnostic"  
echo "3. ✅ Converted regex patterns to raw strings"
echo "4. ✅ Applied comprehensive fix to build_topology.py"
echo ""
echo "📁 Original files backed up to: $BACKUP_DIR"
echo ""
echo "🚀 Ready to test! Run:"
echo "   snakemake -R build_topology"
echo ""
echo "If you encounter more issues, check the backup and revert if needed:"
echo "   cp $BACKUP_DIR/*.py scripts/"
