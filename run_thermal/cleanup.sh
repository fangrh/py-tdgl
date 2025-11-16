#!/bin/bash
# Cleanup script for removing temporary and output files
# Usage: ./cleanup.sh [--dry-run]

DRY_RUN=false
if [ "$1" == "--dry-run" ]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE (no files will be deleted) ==="
fi

echo "Cleaning up current directory: $(pwd)"
echo "============================================"

# Count files before deletion
err_count=$(ls -1 *.err 2>/dev/null | wc -l)
out_count=$(ls -1 *.out 2>/dev/null | wc -l)
npz_count=$(ls -1 *.npz 2>/dev/null | wc -l)

echo "Found:"
echo "  - $err_count .err files"
echo "  - $out_count .out files"
echo "  - $npz_count .npz files"

# Find .h5 files with corresponding .h5.tmp files (incomplete/corrupted)
h5_with_tmp=()
for tmp_file in *.h5.tmp 2>/dev/null; do
    if [ -f "$tmp_file" ]; then
        h5_file="${tmp_file%.tmp}"
        if [ -f "$h5_file" ]; then
            h5_with_tmp+=("$h5_file")
        fi
    fi
done

echo "  - ${#h5_with_tmp[@]} .h5 files with .h5.tmp (incomplete)"

if [ ${#h5_with_tmp[@]} -gt 0 ]; then
    echo ""
    echo "Incomplete .h5 files to be deleted:"
    for f in "${h5_with_tmp[@]}"; do
        echo "    $f"
    done
fi

echo ""

if [ "$DRY_RUN" = false ]; then
    # Ask for confirmation
    read -p "Are you sure you want to delete these files? (y/N) " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Deleting files..."

        # Delete .err files
        if [ $err_count -gt 0 ]; then
            rm -f *.err
            echo "[OK] Deleted $err_count .err files"
        fi

        # Delete .out files
        if [ $out_count -gt 0 ]; then
            rm -f *.out
            echo "[OK] Deleted $out_count .out files"
        fi

        # Delete .npz files
        if [ $npz_count -gt 0 ]; then
            rm -f *.npz
            echo "[OK] Deleted $npz_count .npz files"
        fi

        # Delete .h5 files that have .h5.tmp counterparts (and their .tmp files)
        for h5_file in "${h5_with_tmp[@]}"; do
            rm -f "$h5_file" "${h5_file}.tmp"
            echo "[OK] Deleted $h5_file and ${h5_file}.tmp"
        done

        echo ""
        echo "Cleanup completed!"
    else
        echo "Aborted."
    fi
else
    echo "Dry run complete. No files were deleted."
    echo "Run without --dry-run to actually delete files."
fi
