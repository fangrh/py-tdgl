"""Script to check the structure of an HDF5 file."""
import h5py
import sys


def print_structure(name, obj, indent=0):
    """Recursively print the structure of an HDF5 file."""
    prefix = "  " * indent
    if isinstance(obj, h5py.Group):
        print(f"{prefix}📁 Group: {name}")
        print(f"{prefix}   Attributes: {dict(obj.attrs)}")
    elif isinstance(obj, h5py.Dataset):
        print(f"{prefix}📄 Dataset: {name}")
        print(f"{prefix}   Shape: {obj.shape}, Dtype: {obj.dtype}")
        print(f"{prefix}   Attributes: {dict(obj.attrs)}")


def check_hdf5_structure(file_path):
    """Check and print the structure of an HDF5 file."""
    print(f"\n{'='*60}")
    print(f"Checking HDF5 file: {file_path}")
    print(f"{'='*60}\n")

    try:
        with h5py.File(file_path, 'r') as f:
            print("Top-level keys:")
            print(f"  {list(f.keys())}\n")

            print("Full structure:")
            f.visititems(print_structure)

            print(f"\n{'='*60}")
            print("Summary:")
            print(f"  Has 'solution' group: {'solution' in f}")
            print(f"  Has 'data' group: {'data' in f}")
            print(f"  Has 'mesh' group: {'mesh' in f}")
            print(f"{'='*60}\n")

    except Exception as e:
        print(f"Error reading file: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_hdf5_structure.py <path_to_hdf5_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    check_hdf5_structure(file_path)
