"""Script to check the detailed structure of an HDF5 file."""
import h5py
import sys


def explore_group(group, indent=0):
    """Recursively explore an HDF5 group."""
    prefix = "  " * indent
    for key in group.keys():
        obj = group[key]
        if isinstance(obj, h5py.Group):
            print(f"{prefix}[Group] {key}")
            if obj.attrs:
                print(f"{prefix}   Attrs: {list(obj.attrs.keys())}")
            explore_group(obj, indent + 1)
        elif isinstance(obj, h5py.Dataset):
            print(f"{prefix}[Dataset] {key}")
            print(f"{prefix}   Shape: {obj.shape}, Dtype: {obj.dtype}")
            if obj.attrs:
                print(f"{prefix}   Attrs: {list(obj.attrs.keys())}")


def check_hdf5_detailed(file_path):
    """Check detailed structure of an HDF5 file."""
    print(f"\n{'='*60}")
    print(f"Checking HDF5 file structure")
    print(f"{'='*60}\n")

    try:
        with h5py.File(file_path, 'r') as f:
            print("Top-level keys:", list(f.keys()))
            print()

            explore_group(f)

            print(f"\n{'='*60}")
            print("Key checks:")
            print(f"  Has 'solution' group: {'solution' in f}")
            print(f"  Has 'data' group: {'data' in f}")
            print(f"  Has 'mesh' group: {'mesh' in f}")

            if 'data' in f:
                print(f"\n  Data group keys: {list(f['data'].keys())}")

            print(f"{'='*60}\n")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_hdf5_detailed.py <path_to_hdf5_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    check_hdf5_detailed(file_path)
