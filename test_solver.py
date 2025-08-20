#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
map_qubit_left_krylov_full.py

Version complète prête à l'emploi du pipeline "qubit gauche" utilisant
un propagateur Krylov (scipy.sparse.linalg.expm_multiply) pour accélérer
la propagation temporelle et écrire des lignes p_nochange incrémentales
avec sauvegarde simultanée de Δφ pour diagnostic.

Hypothèses / dépendances :
- modules projet : qutip_utils, analysis_qutip, U_t_2D_computing, param_simu
- qutip, numpy, scipy, matplotlib, tqdm

Usage : coller dans ton dossier de projet et lancer comme l'ancien script.
Remplace les imports de param_simu / qutip_utils si besoin.

Fais un test local sur une petite grille (ex: 10 x 20) avant de lancer tout.
"""

from __future__ import annotations
import os
import gc
import time
import math
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from numpy.lib.format import open_memmap
import scipy.constants as sc
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.ndimage import gaussian_filter

# try import qutip and project-specific modules
try:
    from qutip import sesolve, Options, Qobj
except Exception:
    # If qutip not available, define a minimal Qobj wrapper for vector -> Qobj conversion
    class Qobj:
        def __init__(self, arr, dims=None):
            self._arr = np.asarray(arr)
        def full(self):
            return np.asarray(self._arr)
        @property
        def dims(self):
            return None
        @property
        def shape(self):
            a = np.asarray(self._arr)
            return a.shape

# project imports (expected to exist in your project)
from qutip_utils import (
    hbar_eVs, all_occupations,
    _prepare_initial_state_qubits, build_spinful_hubbard_hamiltonian
)
from analysis_qutip import num_sites, psi0
from U_t_2D_computing import (
    pulse_U, potential_over_time,
    get_eigs, localize_with_fallback,
    t_from_orbitals, U_vector_from_orbitals
)
from param_simu import (init_sig, delta_U_vals_full, delta_t_vals_full,
                        n_electrons, t_imp, T_final, barrier_heights_meV,
                        well_depths_meV, well_width_nm, barrier_widths_nm,
                        a_meV_nm2, dot_x, sigma_x, sigma_y,
                        time_array, idx_t_imp, x, y, m_eff,
                        nbr_pts, basis_occ, logical_qubits, psi0_label, st_L, st_R)

# =================== Réglages spécifiques "qubit gauche" ===================
ROW_BASENAME        = "p_nochange_row"   # fichiers p_nochange_row_###.npy
FORCE_RECALC        = False
USE_PARALLEL        = True
MAX_WORKERS         = max(1, (os.cpu_count() or 2))
ESTIMATE_RUNTIME    = True
BENCH_FRAC_DT       = 0.10
BENCH_MIN_SAMPLES   = 20
UPSAMPLE_TO_HIGHRES = False

# Safety: if hbar_eVs not provided by qutip_utils, compute it
try:
    _ = float(hbar_eVs)
except Exception:
    # hbar (J s) / e (J/eV) -> eV*s
    hbar_eVs = sc.hbar / sc.e

# ========================= Utilitaires ===========================
def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in str(s).strip())

def state_signature(psi_list):
    vec = np.hstack([psi.full().ravel() for psi in psi_list]).view(np.complex128)
    return hashlib.sha1(vec.tobytes()).hexdigest()[:10]

def wrap_pi(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi

def phase_relative(a, b):
    if (np.abs(a) < 1e-15 and np.abs(b) < 1e-15):
        return 0.0
    return wrap_pi(np.angle(a) - np.angle(b))

# --- QUBIT GAUCHE UNIQUEMENT ---
def extract_qubit_L(final_state, logical_qubits):
    """
    Retourne (a, b) dans la base {|S>, |T0>} pour le qubit **gauche** uniquement.
    """
    qL = logical_qubits[0]  # LEFT
    a = qL["0"].overlap(final_state)
    b = qL["1"].overlap(final_state)
    return a, b

# ETA utils
def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else (f"{m:d}m{s:02d}s" if m else f"{s:d}s")

# =================== Fenêtre d'impulsion lissée ===================
def _smooth_window(t, t0, t1, tau):
    import numpy as _np
    return 0.5*(_np.tanh((t - t0)/tau) - _np.tanh((t - t1)/tau))

def _f_pulse(t, args):
    tau = args.get("tau", (args["t1"] - args["t0"]) / 20.0)
    return _smooth_window(t, args["t0"], args["t1"], tau)

def _f_base(t, args):
    return 1.0 - _f_pulse(t, args)

# =================== Conversion utilitaire -> csr ===================
def _to_csr_from_qobj_or_array(H):
    """Convertit Qobj (sparse/dense) ou array en scipy.sparse.csr_matrix."""
    try:
        data = getattr(H, "data", None)
        if data is not None:
            try:
                return data.tocsr().astype(np.complex128)
            except Exception:
                return csr_matrix(H.full(), dtype=np.complex128)
        if hasattr(H, "tocsr"):
            return H.tocsr().astype(np.complex128)
        return csr_matrix(np.asarray(H, dtype=np.complex128))
    except Exception:
        return csr_matrix(np.asarray(H, dtype=np.complex128))

# =================== Propagation rapide (Krylov) ===================
def qubits_impulsion_lastonly(num_sites, n_electrons,
                              H_base, H_pulse,
                              t_imp, Delta_t,
                              T_final, psi0_full,
                              nbr_pts=300,
                              max_substeps=6):
    """
    Propagation rapide par Krylov (expm_multiply) en approximation piecewise-constant.
    Retourne un Qobj (état final).
    """
    # Convertir psi0_full (Qobj) en vecteur numpy
    try:
        psi_vec = psi0_full.full().ravel().astype(np.complex128)
        dims = psi0_full.dims
        shape = psi0_full.shape
    except Exception:
        psi_vec = np.asarray(psi0_full, dtype=np.complex128).ravel()
        dims = None
        shape = (len(psi_vec), 1)

    # période totale (sim) en seconds via hbar_eVs
    t0 = float(t_imp / hbar_eVs)
    t1 = float((t_imp + Delta_t) / hbar_eVs)
    dt_total = max(1e-18, t1 - t0)

    args = {"t0": t0, "t1": t1, "tau": max((t1 - t0)/30.0, 1e-3*(t1 - t0))}

    n_sub = max(1, int(max_substeps))
    n_sub = min(n_sub, 12)

    psi = psi_vec
    # pre-convert bases to csr to save time
    Hb_csr = _to_csr_from_qobj_or_array(H_base)
    Hp_csr = _to_csr_from_qobj_or_array(H_pulse)

    for s in range(n_sub):
        ta = t0 + (s    ) * (dt_total / n_sub)
        tb = t0 + (s + 1) * (dt_total / n_sub)
        tmid = 0.5 * (ta + tb)

        f_p = _f_pulse(tmid, args)
        f_b = 1.0 - f_p

        H_mid = (f_b * Hb_csr) + (f_p * Hp_csr)

        dt_sub = (tb - ta)
        A = (-1j * dt_sub / hbar_eVs) * H_mid

        psi = expm_multiply(A, psi)

    try:
        final_state = Qobj(psi.reshape(shape), dims=dims)
    except Exception:
        final_state = Qobj(psi.reshape((len(psi), 1)))
    return final_state

# =================== Moyenne temporelle V(x,t) ===================
def _mean_Vx_over_window(pot_xt, start_idx, end_idx):
    if isinstance(pot_xt, np.ndarray):
        sl = pot_xt[start_idx:end_idx]
        if sl.ndim == 2:
            return sl.mean(axis=0)
        elif sl.ndim == 1:
            return sl
        else:
            raise ValueError(f"pot_xt ndarray de ndim inattendu: {sl.ndim}")
    else:
        window = pot_xt[start_idx:end_idx]
        if len(window) == 1:
            return np.array(window[0], copy=False)
        return np.mean(np.stack(window, axis=0), axis=0)

# =================== t(ΔU,Δt) & U(ΔU,Δt) via orbitales ===================
def compute_txU_for_pulse(delta_t, delta_U_meV, imp_start_idx):
    U_imp = pulse_U(
        time_array,
        t_start=t_imp,
        delta_t=delta_t,
        delta_U_eV=float(delta_U_meV) * 1e-3
    )

    pot_xt = potential_over_time(
        a_meV_nm2, U_imp, x, dot_x,
        well_depths_meV=well_depths_meV,
        well_width_nm=well_width_nm,
        barrier_heights_meV=barrier_heights_meV,
        barrier_widths_nm=barrier_widths_nm,
        strategy="central_only"
    )

    if float(delta_U_meV) == 0.0:
        idx_not_imp = max(0, imp_start_idx - 1)
        V_x = pot_xt[idx_not_imp]
    else:
        dt_sim  = T_final / len(time_array)
        steps   = max(1, int(np.ceil(delta_t / dt_sim)))
        imp_end = min(len(time_array), imp_start_idx + steps)
        V_x     = _mean_Vx_over_window(pot_xt, imp_start_idx, imp_end)

    _, orbs_raw = get_eigs(V_x, x, m_eff, num_states=4)
    orbitals    = localize_with_fallback(orbs_raw, x, dot_x, window_nm=20, thresh=0.80, max_iter=4)

    T_eV = t_from_orbitals(V_x, x, y, m_eff, sigma_y, orbitals)
    U_eV = U_vector_from_orbitals(orbitals, x, y, sigma_y, epsilon_r=11.7, a_soft=8e-9)

    T_eV = np.nan_to_num(T_eV, nan=0.0, posinf=0.0, neginf=0.0)
    U_eV = np.nan_to_num(U_eV, nan=0.0, posinf=0.0, neginf=0.0)
    # assouplir légèrement les clips pour éviter discontinuités trop fortes
    U_eV = np.clip(U_eV, 1e-8, 1.0)
    T_eV = np.clip(T_eV, -5e-3, 5e-3)

    return T_eV, U_eV

# =================== Baseline φ0(Δt) ===================
def compute_or_load_baseline(delta_t_vals, BASELINE_FILE,
                             imp_start_idx, num_sites, n_electrons, H_base, psi0_full, basis_occ, logical_qubits, nbr_pts):
    if (not FORCE_RECALC) and os.path.exists(BASELINE_FILE):
        try:
            data = np.load(BASELINE_FILE)
            if np.array_equal(data.get("delta_t_vals"), delta_t_vals):
                return data["phi0"]
        except Exception:
            pass
        print("⚠️ Baseline incompatible avec la grille actuelle. Recalcul…")

    print("🧭 Calcul baseline φ0(Δt) (ΔU=0)…")
    phi0 = np.zeros(len(delta_t_vals), dtype=np.float64)
    for j, delta_t in enumerate(tqdm(delta_t_vals, desc="Baseline Δt", unit="Δt")):
        T_eV, U_eV = compute_txU_for_pulse(delta_t, 0.0, imp_start_idx)
        H_pulse = build_spinful_hubbard_hamiltonian(num_sites, T_eV, U_eV, basis_occ)
        final_state = qubits_impulsion_lastonly(
            num_sites=num_sites, n_electrons=n_electrons,
            H_base=H_base, H_pulse=H_pulse,
            t_imp=t_imp, Delta_t=delta_t, T_final=T_final,
            psi0_full=psi0_full, nbr_pts=nbr_pts
        )
        a0, b0 = extract_qubit_L(final_state, logical_qubits)
        phi0[j] = phase_relative(a0, b0)

    np.savez(BASELINE_FILE, phi0=phi0, delta_t_vals=delta_t_vals)
    return phi0

# =================== Lignes p_nochange ===================
def row_path(base_dir: str, i: int) -> str:
    return os.path.join(base_dir, f"{ROW_BASENAME}_{i:03d}.npy")

def row_path_dphi(base_dir: str, i: int) -> str:
    return os.path.join(base_dir, f"{ROW_BASENAME}_dphi_{i:03d}.npy")


def _compute_row_for_deltaU(i, delta_U_meV, delta_t_vals, phi0, base_dir,
                            imp_start_idx, num_sites, n_electrons, H_base, psi0_full, basis_occ, logical_qubits, nbr_pts):
    t_line0 = time.perf_counter()
    out_file = row_path(base_dir, i)
    out_file_dphi = row_path_dphi(base_dir, i)

    if (not FORCE_RECALC) and os.path.exists(out_file):
        p_row_mm = open_memmap(out_file, mode='r+')
        if p_row_mm.shape != (len(delta_t_vals),):
            del p_row_mm
            os.remove(out_file)
            p_row_mm = open_memmap(out_file, mode='w+', dtype='float32', shape=(len(delta_t_vals),))
            p_row_mm[:] = np.nan
    else:
        p_row_mm = open_memmap(out_file, mode='w+', dtype='float32', shape=(len(delta_t_vals),))
        p_row_mm[:] = np.nan

    if (not FORCE_RECALC) and os.path.exists(out_file_dphi):
        dphi_mm = open_memmap(out_file_dphi, mode='r+')
        if dphi_mm.shape != (len(delta_t_vals),):
            del dphi_mm
            os.remove(out_file_dphi)
            dphi_mm = open_memmap(out_file_dphi, mode='w+', dtype='float32', shape=(len(delta_t_vals),))
            dphi_mm[:] = np.nan
    else:
        dphi_mm = open_memmap(out_file_dphi, mode='w+', dtype='float32', shape=(len(delta_t_vals),))
        dphi_mm[:] = np.nan

    for j, delta_t in enumerate(tqdm(delta_t_vals,
                                     desc=f"  → ΔU = {delta_U_meV:.3f} meV",
                                     unit="Δt", leave=False)):
        if np.isfinite(p_row_mm[j]) and np.isfinite(dphi_mm[j]):
            continue

        T_eV, U_eV = compute_txU_for_pulse(delta_t, delta_U_meV, imp_start_idx)
        H_pulse = build_spinful_hubbard_hamiltonian(num_sites, T_eV, U_eV, basis_occ)

        final_state = qubits_impulsion_lastonly(
            num_sites=num_sites, n_electrons=n_electrons,
            H_base=H_base, H_pulse=H_pulse,
            t_imp=t_imp, Delta_t=delta_t, T_final=T_final,
            psi0_full=psi0_full, nbr_pts=nbr_pts
        )

        aL, bL = extract_qubit_L(final_state, logical_qubits)
        dphi = wrap_pi(phase_relative(aL, bL) - phi0[j])
        pval = float(np.cos(0.5 * dphi)**2)

        p_row_mm[j] = pval
        dphi_mm[j] = float(dphi)

        if (j + 1) % 50 == 0:
            p_row_mm.flush()
            dphi_mm.flush()
            gc.collect()

    p_row_mm.flush(); dphi_mm.flush()
    del p_row_mm; del dphi_mm
    print(f"  ↳ ligne ΔU={delta_U_meV:.3f} meV faite en {_fmt_time(time.perf_counter() - t_line0)}")
    return out_file


def _estimate_per_line_time(delta_U_meV, delta_t_vals, phi0,
                            imp_start_idx, num_sites, n_electrons, H_base, psi0_full, basis_occ, logical_qubits, nbr_pts):
    nsamp = max(BENCH_MIN_SAMPLES, int(math.ceil(BENCH_FRAC_DT * len(delta_t_vals))))
    idxs  = np.linspace(0, len(delta_t_vals)-1, nsamp, dtype=int)
    dt_s  = delta_t_vals[idxs]

    start = time.perf_counter()
    for j, delta_t in enumerate(dt_s):
        T_eV, U_eV = compute_txU_for_pulse(delta_t, delta_U_meV, imp_start_idx)
        H_pulse = build_spinful_hubbard_hamiltonian(num_sites, T_eV, U_eV, basis_occ)
        final_state = qubits_impulsion_lastonly(
            num_sites=num_sites, n_electrons=n_electrons,
            H_base=H_base, H_pulse=H_pulse,
            t_imp=t_imp, Delta_t=delta_t, T_final=T_final,
            psi0_full=psi0_full, nbr_pts=nbr_pts
        )
        aL, bL = extract_qubit_L(final_state, logical_qubits)
        _ = np.cos(0.5 * wrap_pi(phase_relative(aL, bL) - phi0[idxs[j]]))**2
    elapsed = time.perf_counter() - start
    return elapsed * (len(delta_t_vals) / len(dt_s))


def _summarize_grid(vals, name="ΔU", unit="meV", max_items=10):
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    if n <= 2*max_items:
        s = ", ".join(f"{v:.3f}" for v in vals)
    else:
        head = ", ".join(f"{v:.3f}" for v in vals[:max_items])
        tail = ", ".join(f"{v:.3f}" for v in vals[-max_items:])
        s = f"{head}, …, {tail}"
    print(f"{name} grid ({unit}) [{n}]: {s}")


def build_p_nochange_rows(delta_U_vals, delta_t_vals, phi0, base_dir,
                          imp_start_idx, num_sites, n_electrons, H_base, psi0_full, basis_occ, logical_qubits, nbr_pts):
    print("🧪 Scan probas de non-changement de phase (cos²(Δφ/2)) — QUBIT GAUCHE…")
    _summarize_grid(delta_U_vals, name="ΔU", unit="meV")

    if ESTIMATE_RUNTIME:
        try:
            mid_idx  = len(delta_U_vals)//2
            probe_du = float(delta_U_vals[mid_idx])
            est_line = _estimate_per_line_time(probe_du, delta_t_vals, phi0,
                                               imp_start_idx, num_sites, n_electrons, H_base, psi0_full, basis_occ, logical_qubits, nbr_pts)
            n_lines  = sum(1 for i in range(len(delta_U_vals)) if (FORCE_RECALC or not os.path.exists(row_path(base_dir, i))))
            if n_lines:
                batches = math.ceil(n_lines / (MAX_WORKERS if (USE_PARALLEL and MAX_WORKERS>1) else 1))
                print(f"⏳ ETA totale ~{_fmt_time(batches*est_line)}  (~{_fmt_time(est_line)}/ligne, {n_lines} lignes)")
        except Exception as e:
            print(f"⚠️ ETA indisponible ({e}).")

    todo = [i for i in range(len(delta_U_vals)) if (FORCE_RECALC or not os.path.exists(row_path(base_dir, i)))]
    if not todo:
        print("✅ Toutes les lignes existent déjà — pas de recalcul.")
        return

    if USE_PARALLEL and len(todo) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        future_to_idx = {}
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex, \
             tqdm(total=len(todo), desc="ΔU rows (parallel)", unit="row") as pbar:
            for i in todo:
                future = ex.submit(_compute_row_for_deltaU, i, float(delta_U_vals[i]), delta_t_vals, phi0, base_dir,
                                   imp_start_idx, num_sites, n_electrons, H_base, psi0_full, basis_occ, logical_qubits, nbr_pts)
                future_to_idx[future] = i
            for fut in as_completed(future_to_idx):
                i = future_to_idx[fut]
                du = float(delta_U_vals[i])
                pbar.set_postfix_str(f"done ΔU={du:.3f} meV")
                pbar.update(1)
    else:
        with tqdm(total=len(todo), desc="ΔU (meV)", unit="ΔU") as pbar:
            for i in todo:
                du = float(delta_U_vals[i])
                pbar.set_postfix_str(f"ΔU={du:.3f}")
                _compute_row_for_deltaU(i, du, delta_t_vals, phi0, base_dir,
                                        imp_start_idx, num_sites, n_electrons, H_base, psi0_full, basis_occ, logical_qubits, nbr_pts)
                gc.collect()
                pbar.update(1)

# =================== Reconstruction / plotting ===================
def load_p_nochange_map(U_vals, T_vals, base_dir):
    rows = []
    for i in range(len(U_vals)):
        f = row_path(base_dir, i)
        if os.path.exists(f):
            arr = np.load(f, mmap_mode='r')
            nT = len(T_vals)
            if len(arr) == nT:
                rows.append(np.asarray(arr))
            elif len(arr) < nT:
                tmp = np.full(nT, np.nan, dtype=np.float32)
                tmp[:len(arr)] = np.asarray(arr)
                rows.append(tmp)
            else:
                rows.append(np.asarray(arr)[:nT])
        else:
            print(f"⚠️ Ligne manquante (remplie par NaN) : {f}")
            rows.append(np.full(len(T_vals), np.nan, dtype=np.float32))
    return np.vstack(rows)


def load_dphi_map(U_vals, T_vals, base_dir):
    rows = []
    for i in range(len(U_vals)):
        f = row_path_dphi(base_dir, i)
        if os.path.exists(f):
            arr = np.load(f, mmap_mode='r')
            nT = len(T_vals)
            if len(arr) == nT:
                rows.append(np.asarray(arr))
            elif len(arr) < nT:
                tmp = np.full(nT, np.nan, dtype=np.float32)
                tmp[:len(arr)] = np.asarray(arr)
                rows.append(tmp)
            else:
                rows.append(np.asarray(arr)[:nT])
        else:
            rows.append(np.full(len(T_vals), np.nan, dtype=np.float32))
    return np.vstack(rows)


def interpolate_to_full(coarse_map, Uc, Tc, Uf, Tf):
    try:
        from scipy.interpolate import RectBivariateSpline
        spl = RectBivariateSpline(Uc, Tc, coarse_map, kx=3, ky=3, s=0.0)
        return np.asarray(spl(Uf, Tf), dtype=np.float32)
    except Exception:
        tmp = np.empty((len(Uc), len(Tf)), dtype=np.float64)
        for i in range(len(Uc)):
            tmp[i, :] = np.interp(Tf, Tc, coarse_map[i, :])
        out = np.empty((len(Uf), len(Tf)), dtype=np.float64)
        for j in range(len(Tf)):
            out[:, j] = np.interp(Uf, Uc, tmp[:, j])
        return out.astype(np.float32)


def plot_p_nochange_map(p_map, U_vals, T_vals, psi0_label=None, out_dir=None,
                        gamma=0.6, smooth_sigma=1.2, pct_clip=(1,99), interp='bicubic'):
    p = np.asarray(p_map, dtype=np.float32)
    nan_mask = np.isnan(p)
    if nan_mask.any():
        p[nan_mask] = np.nanmedian(p)

    vmin = np.nanpercentile(p, pct_clip[0])
    vmax = np.nanpercentile(p, pct_clip[1])
    p = np.clip(p, vmin, vmax)

    p = (p - vmin) / (vmax - vmin + 1e-12)
    p = np.power(p, gamma)

    if smooth_sigma and smooth_sigma > 0:
        p = gaussian_filter(p, sigma=smooth_sigma)

    plt.figure(figsize=(8, 6))
    extent = [T_vals[0]*1e9, T_vals[-1]*1e9, U_vals[0], U_vals[-1]]
    im = plt.imshow(p, aspect='auto', origin='lower', extent=extent,
                    cmap='viridis', interpolation=interp, vmin=0.0, vmax=1.0)
    plt.colorbar(im, label="Proba phase non modifiée  (ajustée)")
    plt.xlabel("Δt (ns)"); plt.ylabel("ΔU (meV)")
    title = "Qubit : probabilité (lissée & ajustée)"
    if isinstance(psi0_label, str) and psi0_label.strip():
        title += f"\nConfiguration : {psi0_label}"
    plt.title(title); plt.tight_layout()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe = _slug(psi0_label) if isinstance(psi0_label, str) else "config"
        fname = f"p_nochange_map_qubit_{safe}_{len(U_vals)}x{len(T_vals)}_{stamp}.png"
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"🖼️ Figure sauvegardée : {out_path}")
    plt.show()


def plot_dphi_map(dphi_map, U_vals, T_vals, psi0_label=None, out_dir=None, deg=False):
    arr = np.asarray(dphi_map, dtype=np.float32)
    if deg:
        vmax = 180.0; vmin = -180.0
        arr_plot = (arr * 180.0 / np.pi)
        cmap = 'seismic'
    else:
        vmax = np.pi; vmin = -np.pi
        arr_plot = arr; cmap = 'seismic'

    plt.figure(figsize=(8, 6))
    extent = [T_vals[0]*1e9, T_vals[-1]*1e9, U_vals[0], U_vals[-1]]
    im = plt.imshow(arr_plot, aspect='auto', origin='lower', extent=extent,
                    cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.colorbar(im, label="Δφ (radians)" if not deg else "Δφ (°)")
    plt.xlabel("Δt (ns)"); plt.ylabel("ΔU (meV)")
    title = "Carte Δφ brute (wrap_pi)"
    if isinstance(psi0_label, str) and psi0_label.strip():
        title += f"\nConfiguration : {psi0_label}"
    plt.title(title); plt.tight_layout()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe = _slug(psi0_label) if isinstance(psi0_label, str) else "config"
        fname = f"dphi_map_qubit_{safe}_{len(U_vals)}x{len(T_vals)}_{stamp}.png"
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"🖼️ Δφ figure sauvegardée : {out_path}")
    plt.show()

# =============================== MAIN ===============================
def main_qubit_left(delta_U_vals_full, delta_t_vals_full):
    print("🚀 Démarrage map_qubit_left (Krylov pipeline)")
    t_total0 = time.perf_counter()

    if UPSAMPLE_TO_HIGHRES:
        COARSE_NU = 220; COARSE_NT = 220
        delta_U_vals = np.linspace(delta_U_vals_full.min(), delta_U_vals_full.max(), COARSE_NU)
        delta_t_vals = np.linspace(delta_t_vals_full.min(), delta_t_vals_full.max(), COARSE_NT)
    else:
        delta_U_vals = delta_U_vals_full
        delta_t_vals = delta_t_vals_full

    config_tag = _slug(psi0_label) if isinstance(psi0_label, str) and psi0_label else "config"
    INIT_SIG   = state_signature(init_sig)

    RESULTS_ROOT = "qubit_results"
    data_dir  = os.path.join(RESULTS_ROOT, f"{config_tag}__psi0_{INIT_SIG}")
    os.makedirs(data_dir, exist_ok=True)

    RES_TAG = f"{len(delta_U_vals_full)}x{len(delta_t_vals_full)}"
    image_dir = os.path.join(data_dir, "images", f"{RES_TAG}__{config_tag}__qubit")
    os.makedirs(image_dir, exist_ok=True)

    BASELINE_FILE = os.path.join(data_dir, "qubit_left_baseline_phi0.npz")

    nU, nT = len(delta_U_vals), len(delta_t_vals)

    def data_complete_for_grid(nU, nT):
        if not os.path.exists(BASELINE_FILE):
            return False
        try:
            b = np.load(BASELINE_FILE)
            dt_saved = b.get("delta_t_vals")
            if dt_saved is None or len(dt_saved) != nT:
                return False
        except Exception:
            return False
        for i in range(nU):
            f = row_path(data_dir, i)
            if not os.path.exists(f):
                return False
            try:
                arr = np.load(f, mmap_mode="r")
                if len(arr) != nT:
                    return False
            except Exception:
                return False
        return True

    if (not FORCE_RECALC) and data_complete_for_grid(nU, nT):
        print("📦 Données déjà présentes pour cette configuration et cette grille. Replot uniquement.")
        p_map_coarse = load_p_nochange_map(delta_U_vals, delta_t_vals, data_dir)
        print("p_map_coarse Nb NaN :", np.isnan(p_map_coarse).sum(), " / ", p_map_coarse.size)

        if UPSAMPLE_TO_HIGHRES and ((len(delta_U_vals_full) != nU) or (len(delta_t_vals_full) != nT)):
            print(f"🧩 Interpolation vers {len(delta_U_vals_full)}x{len(delta_t_vals_full)}…")
            p_map_full = interpolate_to_full(p_map_coarse, delta_U_vals, delta_t_vals,
                                             delta_U_vals_full, delta_t_vals_full)
            np.save(os.path.join(data_dir, f"p_nochange_map_{len(delta_U_vals_full)}x{len(delta_t_vals_full)}.npy"), p_map_full)
            plot_p_nochange_map(p_map_full, delta_U_vals_full, delta_t_vals_full, psi0_label=psi0_label, out_dir=image_dir)
        else:
            plot_p_nochange_map(p_map_coarse, delta_U_vals, delta_t_vals, psi0_label=psi0_label, out_dir=image_dir)
        print("✅ Replot terminé (aucun recalcul).")
        return

    t_step0 = time.perf_counter()
    psi0_full_local = _prepare_initial_state_qubits(num_sites, n_electrons, psi0, basis_occ, logical_qubits)[2]
    phi0 = compute_or_load_baseline(delta_t_vals, BASELINE_FILE,
                                    idx_t_imp, num_sites, n_electrons, None, psi0_full_local, basis_occ, logical_qubits, nbr_pts)
    print(f"⏱️ Baseline calculée en {_fmt_time(time.perf_counter() - t_step0)} ({len(phi0)} Δt).")

    if ESTIMATE_RUNTIME:
        try:
            mid_idx    = len(delta_U_vals)//2
            est_line   = _estimate_per_line_time(float(delta_U_vals[mid_idx]), delta_t_vals, phi0,
                                                 idx_t_imp, num_sites, n_electrons, None, psi0_full_local, basis_occ, logical_qubits, nbr_pts)
            n_lines_todo = sum(1 for i in range(len(delta_U_vals))
                               if (FORCE_RECALC or not os.path.exists(row_path(data_dir, i))))
            if n_lines_todo:
                batches = math.ceil(n_lines_todo / max(1, (MAX_WORKERS if USE_PARALLEL else 1)))
                print(f"⏳ ETA totale ~{_fmt_time(batches*est_line)}  (~{_fmt_time(est_line)}/ligne, {n_lines_todo} lignes)")
        except Exception as e:
            print(f"ETA indisponible ({e}).")

    t_step1 = time.perf_counter()
    # For H_base and psi0 use the baseline computation earlier
    T_base, U_base = compute_txU_for_pulse(delta_t=max(1e-12, float((T_final / len(time_array)))), delta_U_meV=0.0, imp_start_idx=idx_t_imp)
    H_base = build_spinful_hubbard_hamiltonian(num_sites, T_base, U_base, basis_occ)

    build_p_nochange_rows(delta_U_vals, delta_t_vals, phi0, data_dir,
                          idx_t_imp, num_sites, n_electrons, H_base, psi0_full_local, basis_occ, logical_qubits, nbr_pts)
    print(f"⏱️ Lignes ΔU calculées en {_fmt_time(time.perf_counter() - t_step1)}")

    t_step2 = time.perf_counter()
    p_map_coarse = load_p_nochange_map(delta_U_vals, delta_t_vals, data_dir)
    print("p_map_coarse Nb NaN :", np.isnan(p_map_coarse).sum(), " / ", p_map_coarse.size)

    if UPSAMPLE_TO_HIGHRES and ((len(delta_U_vals_full) != len(delta_U_vals)) or (len(delta_t_vals_full) != len(delta_t_vals))):
        print(f"🧩 Interpolation vers {len(delta_U_vals_full)}x{len(delta_t_vals_full)}…")
        p_map_full = interpolate_to_full(p_map_coarse, delta_U_vals, delta_t_vals,
                                         delta_U_vals_full, delta_t_vals_full)
        np.save(os.path.join(data_dir, f"p_nochange_map_{len(delta_U_vals_full)}x{len(delta_t_vals_full)}.npy"),
                p_map_full)
        print(f"✅ Temps total d'exécution : {_fmt_time(time.perf_counter() - t_total0)}")
        plot_p_nochange_map(p_map_full, delta_U_vals_full, delta_t_vals_full, psi0_label=psi0_label, out_dir=image_dir)
    else:
        print(f"✅ Temps total d'exécution : {_fmt_time(time.perf_counter() - t_total0)}")
        plot_p_nochange_map(p_map_coarse, delta_U_vals, delta_t_vals, psi0_label=psi0_label, out_dir=image_dir)
    print(f"⏱️ Reconstruction + plot en {_fmt_time(time.perf_counter() - t_step2)}")


if __name__ == "__main__":
    # prepare initial state/ H_base and run main
    _, _, psi0_full = _prepare_initial_state_qubits(num_sites, n_electrons, psi0, basis_occ, logical_qubits)

    T_base, U_base = compute_txU_for_pulse(delta_t=max(1e-12, float((T_final / len(time_array)))), delta_U_meV=0.0, imp_start_idx=idx_t_imp)
    T_base = np.clip(np.nan_to_num(T_base, nan=0.0, posinf=0.0, neginf=0.0), -1e-3, 1e-3)
    U_base = np.clip(np.nan_to_num(U_base, nan=0.0, posinf=0.0, neginf=0.0), 1e-8, 1.0)
    H_base = build_spinful_hubbard_hamiltonian(num_sites, T_base, U_base, basis_occ)

    print("map_qubit_L : t_matrix_not_pulse=", T_base)
    print("U_not_pul : ", U_base)

    main_qubit_left(delta_U_vals_full, delta_t_vals_full)
