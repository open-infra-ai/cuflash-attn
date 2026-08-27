#!/bin/bash
# Clean build artifacts and caches for cuflash
# Usage: ./scripts/clean.sh [--dry-run]

set -e

# Parse arguments
DRY_RUN=false
if [ "$1" = "--dry-run" ]; then
    DRY_RUN=true
    echo "🔍 Dry run mode - showing what would be removed..."
    echo ""
fi

# Remove build directory
if [ -d "build" ]; then
    if [ "$DRY_RUN" = true ]; then
        echo "  Would remove: build/"
    else
        rm -rf build/
        echo "  ✓ Removed build/"
    fi
fi

# Remove Python caches
if [ "$DRY_RUN" = true ]; then
    count=$(find . -name "__pycache__" -type d 2>/dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
        echo "  Would remove: $count __pycache__ directories"
    fi
    count=$(find . -name "*.pyc" 2>/dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
        echo "  Would remove: $count .pyc files"
    fi
else
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    echo "  ✓ Removed Python caches"
fi

# Remove CMake cache files
if [ "$DRY_RUN" = true ]; then
    count=$(find . -name "CMakeCache.txt" 2>/dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
        echo "  Would remove: $count CMakeCache.txt files"
    fi
    count=$(find . -name "CMakeFiles" -type d 2>/dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
        echo "  Would remove: $count CMakeFiles directories"
    fi
else
    find . -name "CMakeCache.txt" -delete 2>/dev/null || true
    find . -name "CMakeFiles" -type d -exec rm -rf {} + 2>/dev/null || true
    echo "  ✓ Removed CMake caches"
fi

# Remove compile_commands.json (will be regenerated)
if [ -f "compile_commands.json" ]; then
    if [ "$DRY_RUN" = true ]; then
        echo "  Would remove: compile_commands.json"
    else
        rm -f compile_commands.json 2>/dev/null || true
        echo "  ✓ Removed compile_commands.json"
    fi
fi

echo ""
if [ "$DRY_RUN" = true ]; then
    echo "✅ Dry run completed!"
else
    echo "✅ Clean completed!"
fi
echo ""
echo "To rebuild, run:"
echo "  cmake --preset release && cmake --build --preset release"
