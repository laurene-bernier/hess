#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optimized quantum simulation — FIXED physics for correct heat map
Key change:
- Replace placeholder toy formula with a physically consistent 2‑level model
  (Rabi formula for a detuned two‑level system) with proper ℏ and units.
- Vectorized on CPU/GPU for performance.

P(t, ΔU) = 1 - (4J^2 / (Δ^2 + 4J^2)) * sin^2( Ω t / 2 ),
where Δ = ΔU (in eV), J is coupling (eV), Ω = sqrt(Δ^2 + 4J^2) / ℏ.

This yields the expected interference / Rabi stripes in the heat map.
"""

import os
import gc
import time
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import psutil

# Optional GPU (CuPy). Falls back to NumPy.
try:
    import cupy as cp
    GPU_AVAILABLE = True
    xp_gpu = cp
    print("✅ CuPy detected - GPU acceleration enabled")
except Exception:
    GPU_AVAILABLE = False
    xp_gpu = None
    print("⚠️ CuPy not available - using CPU only")

# Matplotlib only for the final heatmap
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================
class Config:
    CPU_CORES = os.cpu_count() or 4
    RAM_GB = psutil.virtual_memory().total / (1024**3)

    # Grid sizing (kept from your script)
    FIXED_GRID_MODE = False
    FIXED_NU = 1000     # ΔU points (Y axis)
    FIXED_NT = 1000     # Δt points (X axis)

    TARGET_HOURS = 10.0
    MAX_TOTAL_POINTS = 10_000  # keep conservative for tests

    # Performance heuristics
    if GPU_AVAILABLE:
        POINTS_PER_SECOND = 500
        TARGET_NU = 100
        TARGET_NT = 100
    else:
        POINTS_PER_SECOND = 100
        TARGET_NU = 50
        TARGET_NT = 50

    # Physics constants / params
    HBAR_EVS = 6.582119569e-16  # eV·s

    # >>> Primary NEW parameter: tunnel coupling J (in meV) <<<
    J_COUPLING_MEV = 0.3  # adjust freely (0.1–1.0 meV typical)


# =============================================================================
# Helpers
# =============================================================================
def estimate_computation_time(n_U: int, n_T: int) -> float:
    base_time_per_point = 0.001 if GPU_AVAILABLE else 0.01  # seconds (very rough)
    total_points = n_U * n_T
    parallel_factor = min(Config.CPU_CORES, 8) if not GPU_AVAILABLE else 50
    estimated_time = (total_points * base_time_per_point) / parallel_factor
    print(f"📊 Estimation for {n_U}×{n_T} = {total_points:,} points:")
    print(f"⏱️  Estimated time: {estimated_time/3600:.1f} hours")
    print(f"🔥 Using: {'GPU acceleration' if GPU_AVAILABLE else f'{parallel_factor} CPU threads'}")
    return estimated_time


def optimize_grid_for_target_time(target_hours: float | None = None) -> tuple[int, int]:
    if target_hours is None:
        target_hours = Config.TARGET_HOURS
    target_seconds = target_hours * 3600
    performance = Config.POINTS_PER_SECOND
    max_points = int(target_seconds * performance)
    max_points = min(max_points, Config.MAX_TOTAL_POINTS)
    grid_size = int(np.sqrt(max_points))
    n_U = max(2, grid_size)
    n_T = max(2, max_points // grid_size)
    print(f"🎯 Optimized grid for {target_hours}h: {n_U}×{n_T} = {n_U*n_T:,} points")
    print(f"📊 Performance estimate: {performance} points/sec")
    print(f"🛡️  Limited by MAX_TOTAL_POINTS: {Config.MAX_TOTAL_POINTS:,}")
    return n_U, n_T


def get_grid_configuration() -> tuple[int, int]:
    if Config.FIXED_GRID_MODE:
        n_U, n_T = Config.FIXED_NU, Config.FIXED_NT
        total_points = n_U * n_T
        if total_points > Config.MAX_TOTAL_POINTS:
            print(f"⚠️  Fixed grid too large ({total_points:,} > {Config.MAX_TOTAL_POINTS:,}), shrinking…")
            ratio = np.sqrt(Config.MAX_TOTAL_POINTS / total_points)
            n_U = max(2, int(n_U * ratio))
            n_T = max(2, int(n_T * ratio))
        print(f"🔧 Fixed grid: {n_U}×{n_T} = {n_U*n_T:,} points")
    else:
        n_U, n_T = optimize_grid_for_target_time()
    return n_U, n_T


# =============================================================================
# Core physics — vectorized (CPU/GPU)
# =============================================================================

def two_level_fidelity_map(xp, delta_U_meV: np.ndarray, delta_t_s: np.ndarray,
                           J_meV: float, hbar_eVs: float) -> np.ndarray:
    """Compute P_LL(ΔU, t) for a detuned 2‑level Hamiltonian in closed form.

    Args:
        xp: module (numpy or cupy)
        delta_U_meV: (n_U,) detuning values in meV
        delta_t_s: (n_T,) time values in seconds
        J_meV: coupling (meV)
        hbar_eVs: ℏ in eV·s

    Returns:
        (n_U, n_T) array of P_LL in [0, 1]
    """
    # Convert to eV
    dU_eV = xp.asarray(delta_U_meV, dtype=xp.float64) * 1e-3
    t = xp.asarray(delta_t_s, dtype=xp.float64)
    J = xp.float64(J_meV * 1e-3)

    # Broadcast to (n_U, n_T)
    dU = dU_eV[:, None]          # (n_U, 1)
    tt = t[None, :]              # (1, n_T)

    denom = dU**2 + 4.0 * J**2   # (n_U, 1)
    # Guard against divide-by-zero (Δ=J=0): use tiny epsilon in denom
    denom = xp.maximum(denom, xp.finfo(xp.float64).tiny)

    Omega = xp.sqrt(denom) / hbar_eVs  # (n_U, 1) in rad/s

    amp = (4.0 * J**2) / denom         # (n_U, 1)
    P = 1.0 - amp * xp.sin(0.5 * Omega * tt)**2

    return P.astype(xp.float32)


# =============================================================================
# Compute + Save + Plot
# =============================================================================
class DataManager:
    def __init__(self, out_dir: str = "quantum_results"):
        self.output_dir = Path(out_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_npz(self, results: np.ndarray, delta_U_vals: np.ndarray, delta_t_vals: np.ndarray, timestamp: str) -> None:
        np.savez_compressed(
            self.output_dir / f"quantum_map_{timestamp}.npz",
            results=results,
            delta_U_vals=delta_U_vals,
            delta_t_vals=delta_t_vals,
        )

    def plot_heatmap(self, results: np.ndarray, delta_U_vals: np.ndarray, delta_t_vals: np.ndarray, timestamp: str) -> None:
        if plt is None:
            print("⚠️ Matplotlib not available, skipping plot")
            return
        plt.style.use('fast')
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        extent = [delta_t_vals[0]*1e9, delta_t_vals[-1]*1e9, delta_U_vals[0], delta_U_vals[-1]]
        im = ax.imshow(results, aspect='auto', origin='lower', extent=extent, cmap='viridis', interpolation='bilinear')
        plt.colorbar(im, label='Return probability P_LL')
        ax.set_xlabel('Δt (ns)')
        ax.set_ylabel('ΔU (meV)')
        ax.set_title(f'Two-level Fidelity Map ({results.shape[0]}×{results.shape[1]})')
        plt.tight_layout()
        fig.savefig(self.output_dir / f"heatmap_{timestamp}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    print("🚀 Starting Optimized Quantum Simulation — fixed physics")
    print(f"💻 System: {Config.CPU_CORES} cores, {Config.RAM_GB:.1f}GB RAM")
    print(f"🎮 GPU: {'✅ Available' if GPU_AVAILABLE else '❌ Not available'}")

    # Grid
    nU, nT = get_grid_configuration()
    delta_U_vals = np.linspace(0.0, 50.0, nU, dtype=np.float32)     # meV
    delta_t_vals = np.linspace(0.1e-9, 10e-9, nT, dtype=np.float32) # s

    estimate_computation_time(nU, nT)

    total_points = int(nU) * int(nT)
    if total_points > 500_000:
        try:
            ans = input(f"\n⚠️ Large run ({total_points:,} points). Continue? (y/N): ")
            if ans.strip().lower() != 'y':
                print("❌ Aborted by user.")
                return
        except EOFError:
            pass

    start = time.perf_counter()

    # Compute on GPU if available else CPU
    J = Config.J_COUPLING_MEV
    if GPU_AVAILABLE:
        dU_gpu = cp.asarray(delta_U_vals)
        dt_gpu = cp.asarray(delta_t_vals)
        results_gpu = two_level_fidelity_map(cp, dU_gpu, dt_gpu, J, Config.HBAR_EVS)
        results = cp.asnumpy(results_gpu)
        del dU_gpu, dt_gpu, results_gpu
        cp.get_default_memory_pool().free_all_blocks()
    else:
        results = two_level_fidelity_map(np, delta_U_vals, delta_t_vals, J, Config.HBAR_EVS)

    elapsed = time.perf_counter() - start

    # Save & plot
    dm = DataManager("quantum_results")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dm.save_npz(results, delta_U_vals, delta_t_vals, ts)
    dm.plot_heatmap(results, delta_U_vals, delta_t_vals, ts)

    print("\n✅ Done!")
    print(f"⏱️  Total time: {elapsed:.2f} s")
    print(f"🚀 Throughput: {results.size/elapsed:.1f} pts/s")
    print(f"📈 Grid size: {results.shape[0]}×{results.shape[1]} = {results.size:,} points")


if __name__ == "__main__":
    main()
