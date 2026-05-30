"""Per-Je HDF5 writer for TDGL simulation results.

Each Je value in a current sweep produces one independent HDF5 file.
Files are written atomically (write to .tmp, then rename) to prevent corruption.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np


def _je_filename(je_value: float) -> str:
    """Generate filename for a Je result: result_Je_{010.2f}.h5

    Uses zero-padded fixed-point notation to ensure lexicographic sorting
    corresponds to numeric sorting.
    """
    return f"result_Je_{je_value:010.2f}.h5"


def atomic_write_hdf5(target_path: Path, write_fn) -> Path:
    """Write an HDF5 file atomically.

    Writes to a .tmp file first, then renames to the target path.
    If the write fails, the .tmp file is cleaned up and no partial
    file remains at the target path.

    Args:
        target_path: Final file path.
        write_fn: Callable(h5py.File) that writes datasets/attrs.

    Returns:
        The target_path after successful write.
    """
    tmp_path = target_path.with_suffix(".h5.tmp")
    try:
        with h5py.File(tmp_path, "w") as f:
            write_fn(f)
        tmp_path.rename(target_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return target_path


def save_je_result(
    run_dir: Path,
    je_value: float,
    psi: np.ndarray,
    vector_potential: np.ndarray,
    superfluid_velocity: np.ndarray,
    electric_field: np.ndarray,
    current: np.ndarray,
    run_id: str,
    solver: str,
    completed_steps: int,
) -> Path:
    """Save a single Je simulation result to an HDF5 file.

    Args:
        run_dir: Directory for this simulation run.
        je_value: Applied current density (A/m^2).
        psi: Complex order parameter, shape (N_times, N_sites).
        vector_potential: Vector potential, shape (N_times, N_sites, 2).
        superfluid_velocity: Superfluid velocity, shape (N_times, N_sites, 2).
        electric_field: Electric field, shape (N_times, N_sites, 2).
        current: Total current density, shape (N_times, N_sites, 2).
        run_id: Unique run identifier.
        solver: Solver name ("py-tdgl" or "cpp-tdgl").
        completed_steps: Number of completed time steps.

    Returns:
        Path to the saved HDF5 file.
    """
    run_dir = Path(run_dir)
    target_path = run_dir / _je_filename(je_value)

    def _write(f: h5py.File):
        # Attributes
        f.attrs["Je"] = je_value
        f.attrs["run_id"] = run_id
        f.attrs["solver"] = solver
        f.attrs["completed_steps"] = completed_steps

        # Datasets — complex psi stored as-is, others as float64
        f.create_dataset("psi", data=psi)
        f.create_dataset("vector_potential", data=vector_potential)
        f.create_dataset("superfluid_velocity", data=superfluid_velocity)
        f.create_dataset("electric_field", data=electric_field)
        f.create_dataset("current", data=current)

    return atomic_write_hdf5(target_path, _write)
