#!/bin/bash

echo "🌍 Quick system-wide fix for dd/mm/yyyy date format"
echo "================================================="

# =============================================================================
# RECOMMENDED: British English locale (best compromise)
# =============================================================================

echo "Setting up British English locale (en_GB.UTF-8)..."
echo "This gives you:"
echo "✅ dd/mm/yyyy date format"
echo "✅ Keep your US keyboard"
echo "✅ International compatibility" 
echo "✅ Works great with pandas"

# Check if en_GB is available
if locale -a 2>/dev/null | grep -q "en_GB"; then
    echo "✅ British English locale is available"
else
    echo "📦 Installing British English locale..."
    sudo locale-gen en_GB.UTF-8
    sudo update-locale
fi

# Set for current session
export LANG=en_GB.UTF-8
export LC_TIME=en_GB.UTF-8
export LC_NUMERIC=en_GB.UTF-8

# Make it permanent in your shell
if [[ -f ~/.bashrc ]]; then
    SHELL_RC="$HOME/.bashrc"
elif [[ -f ~/.zshrc ]]; then
    SHELL_RC="$HOME/.zshrc"
else
    SHELL_RC="$HOME/.profile"
fi

# Add to shell profile if not already there
if ! grep -q "LC_TIME.*en_GB" "$SHELL_RC" 2>/dev/null; then
    echo "" >> "$SHELL_RC"
    echo "# British English locale for dd/mm/yyyy dates" >> "$SHELL_RC"
    echo 'export LANG=en_GB.UTF-8' >> "$SHELL_RC"
    echo 'export LC_TIME=en_GB.UTF-8' >> "$SHELL_RC"
    echo 'export LC_NUMERIC=en_GB.UTF-8' >> "$SHELL_RC"
    echo "✅ Added to $SHELL_RC"
else
    echo "✅ Already configured in $SHELL_RC"
fi

# Configure conda environment
if [[ -n "$CONDA_PREFIX" ]] && command -v conda >/dev/null 2>&1; then
    echo "🐍 Configuring conda environment..."
    conda env config vars set LANG=en_GB.UTF-8 LC_TIME=en_GB.UTF-8 LC_NUMERIC=en_GB.UTF-8
    echo "✅ Conda environment configured"
    echo "⚠️  You'll need to reactivate: conda deactivate && conda activate pypsa-rsa"
else
    echo "ℹ️  Conda not detected or not in environment"
fi

# Test current settings
echo ""
echo "🔍 Current settings:"
echo "Date format: $(date '+%d/%m/%Y %H:%M')"
echo "Locale: $LANG"

# Test pandas
echo ""
echo "🐼 Testing pandas date parsing:"
python3 -c "
import pandas as pd
test_dates = ['13/04/2017 00:00', '01/12/2023 15:30']
for date_str in test_dates:
    try:
        result = pd.to_datetime(date_str)
        print(f'✅ {date_str} -> {result}')
    except Exception as e:
        print(f'❌ {date_str} -> Failed: {e}')
        try:
            result = pd.to_datetime(date_str, dayfirst=True)
            print(f'⚠️  {date_str} -> Works with dayfirst=True: {result}')
        except Exception as e2:
            print(f'❌ {date_str} -> Still fails: {e2}')
"

echo ""
echo "🎯 NEXT STEPS:"
echo "1. Restart your terminal: close and reopen"
echo "2. Reactivate conda: conda deactivate && conda activate pypsa-rsa"  
echo "3. Test again: python -c \"import pandas as pd; print(pd.to_datetime('13/04/2017'))\""
echo "4. Run PyPSA: snakemake -R add_electricity"
echo ""
echo "💡 This should fix the date parsing issue system-wide!"
echo "   Your US keyboard layout will remain unchanged."
