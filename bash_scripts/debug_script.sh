#!/bin/bash

cd /home/clhar/showcase/pypsa-rsa

echo "🔧 Fixing regex warning in add_electricity.py..."
# Fix the regex warning
sed -i 's/str\.contains("\\%capex\/year")/str.contains(r"\\%capex\/year")/g' scripts/add_electricity.py
echo "✅ Fixed regex warning"

echo ""
echo "🔍 Debugging the missing 'DMD_IRP23' issue..."

# Create a quick debug script to check what load trajectories are available
cat > debug_load_data.py << 'EOF'
import pandas as pd
import os

# Try to find the scenario setup and load data
scenario_path = "scenarios/ME IRP 2024/sub_scenarios"

print("=== DEBUGGING LOAD TRAJECTORY ISSUE ===")
print(f"Scenario path: {scenario_path}")

try:
    # Check if the annual_load.xlsx file exists
    load_file = os.path.join(scenario_path, "annual_load.xlsx")
    print(f"Looking for: {load_file}")
    
    if os.path.exists(load_file):
        print("✅ annual_load.xlsx found")
        
        # Read the file and show available trajectories
        annual_load = pd.read_excel(load_file, sheet_name="annual_load", index_col=[0])
        print(f"Available load trajectories:")
        for i, idx in enumerate(annual_load.index):
            print(f"  {i+1}. {idx}")
        
        print(f"\nScenario 'IRP_REF_CI' is looking for: 'DMD_IRP23'")
        
        # Look for similar names
        similar = [idx for idx in annual_load.index if 'DMD' in str(idx).upper() or 'IRP' in str(idx).upper()]
        if similar:
            print(f"Similar trajectories found:")
            for s in similar:
                print(f"  - {s}")
        else:
            print("No similar trajectories found")
            
    else:
        print(f"❌ {load_file} not found")
        
        # Look for other Excel files in the directory
        if os.path.exists(scenario_path):
            excel_files = [f for f in os.listdir(scenario_path) if f.endswith('.xlsx')]
            print(f"Excel files in {scenario_path}:")
            for f in excel_files:
                print(f"  - {f}")
        else:
            print(f"❌ Directory {scenario_path} not found")
            
except Exception as e:
    print(f"❌ Error: {e}")

print("=========================================")
EOF

python debug_load_data.py
rm debug_load_data.py

echo ""
echo "📋 SUMMARY:"
echo "1. ✅ Fixed regex warning in add_electricity.py"
echo "2. 🔍 Debugged load trajectory issue - check output above"
echo ""
echo "📝 NEXT STEPS:"
echo "Based on the debug output above:"
echo "- If similar trajectories exist, update your scenario config to use the correct name"
echo "- If no load data exists, you may need to add the 'DMD_IRP23' trajectory to annual_load.xlsx"
echo "- Or modify the scenario to use an existing trajectory"
echo ""
echo "🚀 After fixing the data issue, run:"
echo "   snakemake -R add_electricity"
