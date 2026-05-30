"""Tests for per-Je HDF5 writer."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from tdgl.solution.je_writer import save_je_result, atomic_write_hdf5


class TestSaveJeResult:
    def test_creates_hdf5_with_correct_structure(self, tmp_path):
        """A saved Je result must contain psi, vector_potential, and attrs."""
        run_dir = tmp_path / "run_001"
        run_dir.mkdir()
        n_times = 5
        n_sites = 100
        psi = np.random.randn(n_times, n_sites) + 1j * np.random.randn(n_times, n_sites)
        vector_potential = np.random.randn(n_times, n_sites, 2)
        superfluid_velocity = np.random.randn(n_times, n_sites, 2)
        electric_field = np.random.randn(n_times, n_sites, 2)
        current = np.random.randn(n_times, n_sites, 2)

        result_path = save_je_result(
            run_dir=run_dir,
            je_value=1.0e6,
            psi=psi,
            vector_potential=vector_potential,
            superfluid_velocity=superfluid_velocity,
            electric_field=electric_field,
            current=current,
            run_id="run_001",
            solver="py-tdgl",
            completed_steps=n_times,
        )

        assert result_path.exists()
        assert result_path.name == "result_Je_1000000.00.h5"

        with h5py.File(result_path, "r") as f:
            assert f.attrs["Je"] == 1.0e6
            assert f.attrs["run_id"] == "run_001"
            assert f.attrs["solver"] == "py-tdgl"
            assert f.attrs["completed_steps"] == n_times
            np.testing.assert_array_equal(f["psi"], psi)
            np.testing.assert_array_equal(f["vector_potential"], vector_potential)
            np.testing.assert_array_equal(f["superfluid_velocity"], superfluid_velocity)
            np.testing.assert_array_equal(f["electric_field"], electric_field)
            np.testing.assert_array_equal(f["current"], current)

    def test_uses_atomic_write(self, tmp_path):
        """If write is interrupted, no partial file should exist."""
        run_dir = tmp_path / "run_002"
        run_dir.mkdir()
        n_times, n_sites = 3, 10
        psi = np.zeros((n_times, n_sites), dtype=complex)
        va = np.zeros((n_times, n_sites, 2))

        result_path = save_je_result(
            run_dir=run_dir,
            je_value=2.5e5,
            psi=psi,
            vector_potential=va,
            superfluid_velocity=va,
            electric_field=va,
            current=va,
            run_id="run_002",
            solver="py-tdgl",
            completed_steps=n_times,
        )

        # No .tmp files should remain
        tmp_files = list(run_dir.glob("*.tmp"))
        assert len(tmp_files) == 0
        assert result_path.exists()

    def test_file_naming_sorts_correctly(self, tmp_path):
        """Files with different Je values must sort lexicographically by Je."""
        run_dir = tmp_path / "run_003"
        run_dir.mkdir()
        n_times, n_sites = 1, 5
        dummy = np.zeros((n_times, n_sites), dtype=complex)
        va = np.zeros((n_times, n_sites, 2))

        for je in [1.0e6, 2.0e6, 5.0e5, 1.5e6]:
            save_je_result(
                run_dir=run_dir, je_value=je,
                psi=dummy, vector_potential=va,
                superfluid_velocity=va, electric_field=va, current=va,
                run_id="run_003", solver="py-tdgl", completed_steps=n_times,
            )

        h5_files = sorted(run_dir.glob("result_Je_*.h5"))
        # Extract Je values from filenames
        je_from_name = lambda p: float(p.stem.split("_Je_")[1])
        je_values = [je_from_name(f) for f in h5_files]
        assert je_values == sorted(je_values), f"Files not sorted by Je: {je_values}"
