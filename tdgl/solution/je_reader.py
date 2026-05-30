"""Per-Je HDF5 reader for TDGL simulation results.

Loads data from individual Je HDF5 files, including extracting
the final state for use as initial conditions in the next Je step.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np


def load_final_state(h5_path: Path) -> np.ndarray:
    """Load the final-time-step psi from a Je result file.

    This is used to set initial conditions for the next Je value
    in a serial sweep.

    Args:
        h5_path: Path to a result_Je_*.h5 file.

    Returns:
        Complex psi array of shape (N_sites,) from the last time step.
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        psi = f["psi"][:]
        return psi[-1]


def load_je_result(h5_path: Path) -> Dict[str, Any]:
    """Load all data from a Je result file.

    Args:
        h5_path: Path to a result_Je_*.h5 file.

    Returns:
        Dictionary with keys: je_value, run_id, solver, completed_steps,
        psi, vector_potential, superfluid_velocity, electric_field, current.
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        return {
            "je_value": float(f.attrs["Je"]),
            "run_id": str(f.attrs["run_id"]),
            "solver": str(f.attrs["solver"]),
            "completed_steps": int(f.attrs["completed_steps"]),
            "psi": f["psi"][:],
            "vector_potential": f["vector_potential"][:],
            "superfluid_velocity": f["superfluid_velocity"][:],
            "electric_field": f["electric_field"][:],
            "current": f["current"][:],
        }
