#!/bin/bash

# Component Usage Audit Script
echo "=== COMPONENT USAGE AUDIT ==="
echo ""

cd frontend/src

# Find all component files
components=$(find ./components -name "*.tsx" -o -name "*.ts" | sort)

echo "Checking component usage..."
echo ""

unused_count=0
used_count=0

for component in $components; do
    # Skip index files and test files
    if [[ $component == *"/index.ts"* ]] || [[ $component == *".test."* ]]; then
        continue
    fi
    
    # Get the component name without path and extension
    component_name=$(basename "$component" | sed 's/\.[^.]*$//')
    
    # Search for imports of this component (excluding the component file itself)
    import_count=$(grep -r "$component_name" . \
        --include="*.tsx" \
        --include="*.ts" \
        --exclude-dir="node_modules" \
        --exclude-dir="dist" \
        --exclude-dir="build" \
        | grep -v "$component" \
        | grep -E "import.*$component_name|from.*$component_name" \
        | wc -l)
    
    if [ $import_count -eq 0 ]; then
        echo "❌ UNUSED: $component"
        ((unused_count++))
    else
        echo "✅ USED ($import_count times): $component"
        ((used_count++))
    fi
done

echo ""
echo "=== SUMMARY ==="
echo "Total components: $((used_count + unused_count))"
echo "Used components: $used_count"
echo "Unused components: $unused_count"