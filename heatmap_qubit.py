#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
map_qubit_v2_unified — version alignée sur le nouveau pipeline U_t_2D_computing
Objectifs demandés :
- Un seul dossier **données** par configuration (qubit)
- Les **images** sont enregistrées dans un sous-dossier dont le nom contient
  la résolution demandée (TARGET_NU x TARGET_NT), la configuration (ex: singlet_triplet)
  et le mot **qubit**
- Si les données existent déjà pour cette configuration **et** ce nombre de points,
  le programme fait un **replot-only** (aucun recalcul)
- ETA global léger (benchmark partiel d’une seule ligne ΔU)

Le script conserve les noms/fonctions que tu utilises dans map_detection_v2 :
  pulse_U, potential_over_time, get_eigs, localize_with_fallback,
  t_from_orbitals, U_vector_from_orbitals, etc.
"""

import os
import gc
import time
import math
import hashlib
import scipy.constants as sc
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from U_t_2D_computing import build_adaptive_x_grid  # ajouter à l'import
from qutip import sesolve, Options, Qobj
from qutip_utils import (
    hbar_eVs, all_occupations, _build_logical_qubits,
    _prepare_initial_state_qubits, build_spinful_hubbard_hamiltonian
)
from analysis_qutip import num_sites
from analysis_qutip import psi0, psi0_label  # label humain optionnel
from analysis_qutip import st_L, st_R
from U_t_2D_computing import (
    pulse_U, potential_over_time,
    get_eigs, localize_with_fallback,
    t_from_orbitals, U_vector_from_orbitals,
    # paramètres partagés
    t_imp, T_final, barrier_heights_meV, well_depths_meV, a_meV_nm2
)
# Limiter le sur-threading BLAS (utile si on parallélise par process)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

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

# Qubit gauche en base {|S>, |T0>}
def extract_qubit_L(final_state, logical_qubits):
    q_L = logical_qubits[0]
    a_L = q_L["0"].overlap(final_state)  # |S>
    b_L = q_L["1"].overlap(final_state)  # |T0>
    return a_L, b_L

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

# =================== Évolution — dernier état seulement ===================

def _get_opt(opt, key, default):
    try:
        return opt.get(key, default) if isinstance(opt, dict) else getattr(opt, key, default)
    except Exception:
        return default

def qubits_impulsion_lastonly(num_sites, n_electrons,
                              H_base, H_pulse,
                              t_imp, Delta_t,
                              T_final, psi0_full,
                              nbr_pts=300):
    """Évolution TDSE lissée (on retourne l'état final)."""
    t0 = float(t_imp / hbar_eVs)
    t1 = float((t_imp + Delta_t) / hbar_eVs)
    args = {"t0": t0, "t1": t1, "tau": max((t1 - t0)/30.0, 1e-3*(t1 - t0))}

    H_td = [[H_base,  _f_base,  args],
            [H_pulse, _f_pulse, args]]

    times = np.linspace(0.0, T_final / hbar_eVs, int(nbr_pts))

    SE_OPTS_local = Options(
        store_states=False,
        store_final_state=True,
        progress_bar=None,
        rtol=max(_get_opt(SE_OPTS, "rtol", 1e-6), 5e-5),
        atol=max(_get_opt(SE_OPTS, "atol", 1e-8), 1e-7),
        nsteps=max(_get_opt(SE_OPTS, "nsteps", 10000), 10000),
        method="bdf"
    )

    return sesolve(H_td, psi0_full, times, e_ops=[], args=args, options=SE_OPTS_local).final_state

# =================== Moyenne temporelle V(x,t) ===================
def _mean_Vx_over_window(pot_xt, start_idx, end_idx):
    """Moyenne robuste sur [start_idx:end_idx] pour pot_xt (nt,nx) ou liste."""
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
    """Retourne (T_eV, U_eV) pendant l'impulsion pour (Δt, ΔU)."""
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

    # ... après avoir construit pot_xt, imp_end, etc.

    if float(delta_U_meV) == 0.0:
        # Aligner avec U_t_2D_computing : snapshot juste avant l’impulsion
        idx_not_imp = max(0, imp_start_idx - 1)
        V_x = pot_xt[idx_not_imp]
    else:
        # Pendant l’impulsion : moyenne robuste (inchangé)
        dt_sim  = T_final / len(time_array)
        steps   = max(1, int(np.ceil(delta_t / dt_sim)))
        imp_end = min(len(time_array), imp_start_idx + steps)
        V_x     = _mean_Vx_over_window(pot_xt, imp_start_idx, imp_end)


    # États propres + localisation robuste
    _, orbs_raw = get_eigs(V_x, x, m_eff, num_states=4)
    orbitals    = localize_with_fallback(orbs_raw, x, dot_x, window_nm=20, thresh=0.80, max_iter=4)

    # t_ij (eV) et U_i (eV)
    T_eV = t_from_orbitals(V_x, x, y, m_eff, sigma_y, orbitals)
    U_eV = U_vector_from_orbitals(orbitals, x, y, sigma_y, epsilon_r=11.7, a_soft=8e-9)

    # Nettoyage & bornes physiques raisonnables
    T_eV = np.nan_to_num(T_eV, nan=0.0, posinf=0.0, neginf=0.0)
    U_eV = np.nan_to_num(U_eV, nan=0.0, posinf=0.0, neginf=0.0)
    U_eV = np.clip(U_eV, 1e-5, 0.5)   # 0.01 meV .. 500 meV
    T_eV = np.clip(T_eV, -1e-3, 1e-3) # |t| <= 1 meV

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
    for j, delta_t in enumerate(tqdm(delta_t_vals, desc="Baseline Δt")):
        T_eV, U_eV = compute_txU_for_pulse(delta_t, 0.0, imp_start_idx)
        print("map_detection : t_matrix_pulse=", T_eV)
        print("U_pul_vec : ", U_eV)
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

def row_path(base_dir: str, i: int) -> str:
    return os.path.join(base_dir, f"{ROW_BASENAME}_{i:03d}.npy")

def _compute_row_for_deltaU(i, delta_U_meV, delta_t_vals, phi0, base_dir,
                            imp_start_idx, num_sites, n_electrons, H_base, psi0_full, basis_occ, logical_qubits, nbr_pts):
    t_line0 = time.perf_counter()
    out_file = row_path(base_dir, i)
    if (not FORCE_RECALC) and os.path.exists(out_file):
        return out_file

    p_row = np.zeros(len(delta_t_vals), dtype=np.float32)
    for j, delta_t in enumerate(tqdm(delta_t_vals, desc=f"  → ΔU = {delta_U_meV:.1f} meV", leave=False)):
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
        p_row[j] = np.cos(0.5 * dphi)**2

    np.save(out_file, p_row)
    print(f"  ↳ ligne ΔU={delta_U_meV:.1f} meV faite en {_fmt_time(time.perf_counter() - t_line0)}")
    return out_file

def _estimate_per_line_time(delta_U_meV, delta_t_vals, phi0,
                            imp_start_idx, num_sites, n_electrons, H_base, psi0_full, basis_occ, logical_qubits, nbr_pts):
    nsamp = max(BENCH_MIN_SAMPLES, int(math.ceil(BENCH_FRAC_DT * len(delta_t_vals))))
    idxs  = np.linspace(0, len(delta_t_vals)-1, nsamp, dtype=int)
    dt_s  = delta_t_vals[idxs]

    start = time.perf_counter()
    for j, delta_t in enumerate(tqdm(dt_s, desc=f"⏱️ Bench ΔU={delta_U_meV:.1f} meV", leave=False)):
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

def build_p_nochange_qubit_rows(delta_U_vals, delta_t_vals, phi0, base_dir,
                          imp_start_idx, num_sites, n_electrons, H_base, psi0_full, basis_occ, logical_qubits, nbr_pts):
    print("🧪 Scan probas de non-changement de phase (cos²(Δφ/2))…")

    # Estimation (mur global)
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
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = [
                ex.submit(_compute_row_for_deltaU, i, float(delta_U_vals[i]), delta_t_vals, phi0, base_dir,
                          imp_start_idx, num_sites, n_electrons, H_base, psi0_full, basis_occ, logical_qubits, nbr_pts)
                for i in todo
            ]
            start_all = time.perf_counter(); completed = 0
            for _ in tqdm(as_completed(futs), total=len(futs), desc="ΔU rows (parallel)"):
                completed += 1
                if ESTIMATE_RUNTIME and (completed % 5 == 0 or completed == len(futs)):
                    elapsed = time.perf_counter() - start_all
                    avg = elapsed / max(1, completed)
                    remaining = (len(futs) - completed) * avg
                    print(f"   ↪ progrès {completed}/{len(futs)} — écoulé: {_fmt_time(elapsed)} — ETA restante: {_fmt_time(remaining)}")
    else:
        start_all = time.perf_counter(); completed = 0
        for i in tqdm(todo, desc="Scan ΔU", position=0):
            _compute_row_for_deltaU(i, float(delta_U_vals[i]), delta_t_vals, phi0, base_dir,
                                    imp_start_idx, num_sites, n_electrons, H_base, psi0_full, basis_occ, logical_qubits, nbr_pts)
            gc.collect()
            completed += 1
            if ESTIMATE_RUNTIME and (completed % 5 == 0):
                elapsed = time.perf_counter() - start_all
                avg = elapsed / max(1, completed)
                remaining = (len(todo) - completed) * avg
                print(f"   ↪ progrès {completed}/{len(todo)} — écoulé: {_fmt_time(elapsed)} — ETA restante: {_fmt_time(remaining)}")

# =================== Reconstruction / plotting ===================

def load_p_nochange_qubit_map(U_vals, T_vals, base_dir):
    rows = []
    for i in range(len(U_vals)):
        f = row_path(base_dir, i)
        if os.path.exists(f):
            rows.append(np.load(f))
        else:
            print(f"⚠️ Ligne manquante (remplie par NaN) : {f}")
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

def plot_p_nochange_qubit_map(p_map, U_vals, T_vals, psi0_label=None, out_dir=None):
    plt.figure(figsize=(8, 6))
    extent = [T_vals[0]*1e9, T_vals[-1]*1e9, U_vals[0], U_vals[-1]]
    im = plt.imshow(p_map, aspect='auto', origin='lower', extent=extent,
                    cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar(im, label="Proba phase non modifiée  (cos²(Δφ/2))")
    plt.xlabel("Δt (ns)"); plt.ylabel("ΔU (meV)")
    title = "Qubits : probabilité que la phase du qubit n'ait pas changé"
    if isinstance(psi0_label, str) and psi0_label.strip():
        title += f"\nConfiguration : {psi0_label}"
    plt.title(title); plt.tight_layout()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe = _slug(psi0_label) if isinstance(psi0_label, str) else "config"
        fname = f"fidelity_map_qubit_{safe}_{len(U_vals)}x{len(T_vals)}_{stamp}.png"
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"🖼️ Figure sauvegardée : {out_path}")
    plt.show()

# =============================== MAIN ===============================

def main_qubit(delta_U_vals_full, delta_t_vals_full):
    print("🚀 Démarrage map_qubit (nouveau pipeline)")
    t_total0 = time.perf_counter()

    # Grille de calcul (coarse)
    if UPSAMPLE_TO_HIGHRES:
        COARSE_NU = 50
        COARSE_NT = 50 
        delta_U_vals = np.linspace(delta_U_vals_full.min(), delta_U_vals_full.max(), COARSE_NU)
        delta_t_vals = np.linspace(delta_t_vals_full.min(), delta_t_vals_full.max(), COARSE_NT)
    else:
        delta_U_vals = delta_U_vals_full
        delta_t_vals = delta_t_vals_full

    # ====================== Chemins et organisation ======================
    config_tag = _slug(psi0_label) if isinstance(psi0_label, str) and psi0_label else "config"
    INIT_SIG   = state_signature([st_L["S"], st_R["T0"]])

    RESULTS_ROOT = "qubit_results"
    data_dir  = os.path.join(RESULTS_ROOT, f"{config_tag}__psi0_{INIT_SIG}")
    os.makedirs(data_dir, exist_ok=True)

    RES_TAG = f"{len(delta_U_vals_full)}x{len(delta_t_vals_full)}"
    image_dir = os.path.join(data_dir, "images", f"{RES_TAG}__{config_tag}__qubit")
    os.makedirs(image_dir, exist_ok=True)

    BASELINE_FILE = os.path.join(data_dir, "qubit_baseline_phi0.npz")

    # 0) Replot-only si les données existent déjà pour cette grille (coarse)
    nU, nT = len(delta_U_vals), len(delta_t_vals)

    # === Construit un H_base cohérent ===
    _imp_start = idx_t_imp
    _baseline_dt = max(1e-12, float((T_final / len(time_array))))
    T_base, U_base = compute_txU_for_pulse(delta_t=_baseline_dt, delta_U_meV=0.0, imp_start_idx=_imp_start)
    T_base = np.clip(np.nan_to_num(T_base, nan=0.0, posinf=0.0, neginf=0.0), -1e-3, 1e-3)
    U_base = np.clip(np.nan_to_num(U_base, nan=0.0, posinf=0.0, neginf=0.0), 1e-5, 0.5)
    H_base = build_spinful_hubbard_hamiltonian(num_sites, T_base, U_base, basis_occ)
    print("map_detection : t_matrix_not_pulse=", T_base)
    print("U_not_pul : ", U_base)

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
        p_map_coarse = load_p_nochange_qubit_map(delta_U_vals, delta_t_vals, data_dir)
        print("p_map_coarse Nb NaN :", np.isnan(p_map_coarse).sum(), " / ", p_map_coarse.size)

        if UPSAMPLE_TO_HIGHRES and ((len(delta_U_vals_full) != nU) or (len(delta_t_vals_full) != nT)):
            print(f"🧩 Interpolation vers {len(delta_U_vals_full)}x{len(delta_t_vals_full)}…")
            p_map_full = interpolate_to_full(p_map_coarse, delta_U_vals, delta_t_vals,
                                             delta_U_vals_full, delta_t_vals_full)
            np.save(os.path.join(data_dir, f"p_nochange_qubit_map_{len(delta_U_vals_full)}x{len(delta_t_vals_full)}.npy"), p_map_full)
            plot_p_nochange_qubit_map(p_map_full, delta_U_vals_full, delta_t_vals_full, psi0_label=psi0_label, out_dir=image_dir)
        else:
            plot_p_nochange_qubit_map(p_map_coarse, delta_U_vals, delta_t_vals, psi0_label=psi0_label, out_dir=image_dir)
        print("✅ Replot terminé (aucun recalcul).")
        raise SystemExit(0)

    # 1) baseline φ0(Δt)
    t_step0 = time.perf_counter()
    phi0 = compute_or_load_baseline(delta_t_vals, BASELINE_FILE,
                                    idx_t_imp, num_sites, n_electrons, H_base, psi0_full, basis_occ, logical_qubits, nbr_pts)
    print(f"⏱️ Baseline calculée en {_fmt_time(time.perf_counter() - t_step0)} ({len(phi0)} Δt).")

    # 2) ETA globale (optionnelle)
    if ESTIMATE_RUNTIME:
        try:
            mid_idx    = len(delta_U_vals)//2
            est_line   = _estimate_per_line_time(float(delta_U_vals[mid_idx]), delta_t_vals, phi0,
                                                 idx_t_imp, num_sites, n_electrons, H_base, psi0_full, basis_occ, logical_qubits, nbr_pts)
            n_lines_todo = sum(1 for i in range(len(delta_U_vals))
                               if (FORCE_RECALC or not os.path.exists(row_path(data_dir, i))))
            if n_lines_todo:
                batches = math.ceil(n_lines_todo / max(1, (MAX_WORKERS if USE_PARALLEL else 1)))
                print(f"⏳ ETA totale ~{_fmt_time(batches*est_line)}  (~{_fmt_time(est_line)}/ligne, {n_lines_todo} lignes)")
        except Exception as e:
            print(f"ETA indisponible ({e}).")

    # 3) calcul des lignes ΔU
    t_step1 = time.perf_counter()
    build_p_nochange_qubit_rows(delta_U_vals, delta_t_vals, phi0, data_dir,
                          idx_t_imp, num_sites, n_electrons, H_base, psi0_full, basis_occ, logical_qubits, nbr_pts)
    print(f"⏱️ Lignes ΔU calculées en {_fmt_time(time.perf_counter() - t_step1)}")

    # 4) reconstruction et upsampling éventuel
    t_step2 = time.perf_counter()
    p_map_coarse = load_p_nochange_qubit_map(delta_U_vals, delta_t_vals, data_dir)
    print("Nb NaN :", np.isnan(p_map_coarse).sum(), " / ", p_map_coarse.size)

    if UPSAMPLE_TO_HIGHRES and ((len(delta_U_vals_full) != len(delta_U_vals)) or (len(delta_t_vals_full) != len(delta_t_vals))):
        print(f"🧩 Interpolation vers {len(delta_U_vals_full)}x{len(delta_t_vals_full)}…")
        p_map_full = interpolate_to_full(p_map_coarse, delta_U_vals, delta_t_vals,
                                         delta_U_vals_full, delta_t_vals_full)
        np.save(os.path.join(data_dir, f"p_nochange_qubit_map_{len(delta_U_vals_full)}x{len(delta_t_vals_full)}.npy"),
                p_map_full)
        print(f"✅ Temps total d'exécution : {_fmt_time(time.perf_counter() - t_total0)}")
        plot_p_nochange_qubit_map(p_map_full, delta_U_vals_full, delta_t_vals_full, psi0_label=psi0_label, out_dir=image_dir)
    else:
        print(f"✅ Temps total d'exécution : {_fmt_time(time.perf_counter() - t_total0)}")
        plot_p_nochange_qubit_map(p_map_coarse, delta_U_vals, delta_t_vals, psi0_label=psi0_label, out_dir=image_dir)
    print(f"⏱️ Reconstruction + plot en {_fmt_time(time.perf_counter() - t_step2)}")


if __name__ == "__main__":
    ROW_BASENAME = "p_nochange_qubit_row"
    # Domaine final (cible)
    TARGET_NU = 100  # ⟵ ajuste ici
    TARGET_NT = 100   # ⟵ ajuste ici
    # =========================== Réglages =============================
    USE_PARALLEL        = True
    MAX_WORKERS         = max(1, (os.cpu_count() or 2) - 1)
    ESTIMATE_RUNTIME    = True
    BENCH_FRAC_DT       = 0.10
    BENCH_MIN_SAMPLES   = 20
    UPSAMPLE_TO_HIGHRES = False
    FORCE_RECALC        = False

    m_eff   = 0.067 * sc.m_e
    sigma_y = 10e-9
    dot_x   = np.array([-75e-9, -25e-9, 25e-9, 75e-9])    # positions des 4 puits
    well_width_nm     = 23
    barrier_widths_nm = (15, 20, 15)
    sigma_x = 15e-9
    x = build_adaptive_x_grid(dot_x, sigma_x, well_width_nm, barrier_widths_nm,
                            safety_pts=16, span_sigma=5)   # comme U_t_2D_computing
    Ny = 40
    y  = np.linspace(-5*sigma_y, 5*sigma_y, Ny)

    # bornes ΔU (meV) et Δt (s)
    delta_U_vals_full = np.linspace(0.0, 47.0, TARGET_NU)
    delta_t_vals_full = np.linspace(0.1e-9, T_final - t_imp, TARGET_NT)

    # Temps & array de référence
    time_array = np.linspace(0.0, T_final, 200) # N_time = 200
    idx_t_imp = int(np.searchsorted(time_array, t_imp))

    # ========= Options solver =========
    SE_OPTS = Options(store_states=False, store_final_state=True, rtol=1e-6, atol=1e-8, progress_bar=None)

    # dimension, bases, état initial
    n_electrons    = 4
    nbr_pts        = 400
    basis_occ      = all_occupations(num_sites, n_electrons)
    logical_qubits = _build_logical_qubits(num_sites, basis_occ)
    _, _, psi0_full = _prepare_initial_state_qubits(num_sites, n_electrons, psi0, basis_occ, logical_qubits)

    main_qubit(delta_U_vals_full, delta_t_vals_full)