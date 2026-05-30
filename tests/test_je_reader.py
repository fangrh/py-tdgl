"""Tests for per-Je HDF5 reader (final state loading)."""
from pathlib import Path

import h5py
import numpy as np
import pytest

from tdgl.solution.je_writer import save_je_result
from tdgl.solution.je_reader import load_final_state, load_je_result


class TestLoadFinalState:
    def test_loads_last_timestep_psi(self, tmp_path):
        """load_final_state should return the psi from the last time step."""
        run_dir = tmp_path / "run_fs"
        run_dir.mkdir()
        n_times, n_sites = 10, 50
        rng = np.random.default_rng(42)
        psi = rng.standard_normal((n_times, n_sites)) + 1j * rng.standard_normal((n_times, n_sites))
        va = rng.standard_normal((n_times, n_sites, 2))

        path = save_je_result(
            run_dir=run_dir, je_value=1.0e6,
            psi=psi, vector_potential=va,
            superfluid_velocity=va, electric_field=va, current=va,
            run_id="run_fs", solver="py-tdgl", completed_steps=n_times,
        )

        final_psi = load_final_state(path)
        np.testing.assert_array_equal(final_psi, psi[-1])

    def test_load_je_result_returns_all_data(self, tmp_path):
        """load_je_result should return all datasets and attrs."""
        run_dir = tmp_path / "run_lr"
        run_dir.mkdir()
        n_times, n_sites = 5, 20
        rng = np.random.default_rng(43)
        psi = rng.standard_normal((n_times, n_sites)) + 1j * rng.standard_normal((n_times, n_sites))
        va = rng.standard_normal((n_times, n_sites, 2))
        sfv = rng.standard_normal((n_times, n_sites, 2))
        ef = rng.standard_normal((n_times, n_sites, 2))
        cur = rng.standard_normal((n_times, n_sites, 2))

        path = save_je_result(
            run_dir=run_dir, je_value=2.0e6,
            psi=psi, vector_potential=va,
            superfluid_velocity=sfv, electric_field=ef, current=cur,
            run_id="run_lr", solver="py-tdgl", completed_steps=n_times,
        )

        result = load_je_result(path)
        assert result["je_value"] == 2.0e6
        assert result["run_id"] == "run_lr"
        assert result["solver"] == "py-tdgl"
        assert result["completed_steps"] == n_times
        np.testing.assert_array_equal(result["psi"], psi)
        np.testing.assert_array_equal(result["current"], cur)
