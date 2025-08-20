
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
map_qubit_v2_unified — version alignée sur le nouveau pipeline U_t_2D_computing
Objectifs :
- Un seul dossier **données** par configuration (qubit)
- Les **images** sont enregistrées dans un sous-dossier dont le nom contient
  la résolution demandée (TARGET_NU x TARGET_NT), la configuration (ex: singlet_triplet)
  et le mot **qubit**
- Si les données existent déjà pour cette configuration **et** ce nombre de points,
  le programme fait un **replot-only** (aucun recalcul)
- Écriture **incrémentale** par Δt (memmap .npy) → reprise sûre et RAM minimale
- ETA global léger (benchmark partiel d’une seule ligne ΔU)
- Affichage pratiques : résumé des ΔU, barre tqdm extérieure (lignes ΔU) + intérieure (Δt)
"""

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
from numpy.lib.format import open_memmap  # memmap pour écriture incrémentale
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from qutip import sesolve, Options, Qobj
from qutip_utils import (
    hbar_eVs, all_occupations, 
    _prepare_initial_state_qubits, build_spinful_hubbard_hamiltonian
)
from U_t_2D_computing import (
    pulse_U, potential_over_time,
    get_eigs, localize_with_fallback,
    t_from_orbitals, U_vector_from_orbitals
)
import scipy.constants as sc
from param_simu import (delta_U_vals_full, delta_t_vals_full, 
                        n_electrons, t_imp, T_final, barrier_heights_meV, 
                        well_depths_meV, well_width_nm, barrier_widths_nm, 
                        a_meV_nm2, dot_x, sigma_x, sigma_y,
                        time_array, idx_t_imp, x, y, m_eff,
                        nbr_pts, basis_occ, logical_qubits, nbr_pts, psi0_label, st_L, st_R,
                        num_sites, psi0)

ROW_BASENAME = "fidelity_detector_R_row"  # visible pour row_path() dans les workers
FORCE_RECALC = False  # défaut (sera surchargé si besoin dans __main__)
USE_PARALLEL        = True
MAX_WORKERS         = max(1, (os.cpu_count() or 2) - 1)
ESTIMATE_RUNTIME    = False
BENCH_FRAC_DT       = 0.10
BENCH_MIN_SAMPLES   = 20
UPSAMPLE_TO_HIGHRES = False
FORCE_RECALC        = False


# --- Globals visibles dans les sous-processus ---
SE_OPTS = {"rtol": 1e-6, "atol": 1e-8, "nsteps": 10000}  # défaut safe pour qubits_impulsion_lastonly


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

# --- QUBIT DROIT UNIQUEMENT ---
def extract_qubit_R(final_state, logical_qubits):
    """
    Retourne (a, b) dans la base {|S>, |T0>} pour le qubit droit uniquement.
    Convention du projet: logical_qubits[0] = gauche, logical_qubits[1] = droite.
    """
    qR = logical_qubits[1]  # RIGHT
    a = qR["0"].overlap(final_state)  # |S>
    b = qR["1"].overlap(final_state)  # |T0>
    return a, b

def right_qubit_metrics(final_state, logical_qubits, eps=1e-12, normalize=True):
    """
    Mesures sur le qubit droit dans la base {|S>, |T0>} :
    - rho (2x2): matrice densité projetée et normalisée sur ce sous-espace
    - coherence: |rho_ST| in [0,1] (1 = cohérence parfaite, 0 = totalement déphasé)
    - purity: Tr(rho^2) in [0,1] (1 = état pur)
    - weight: p = <ψ|P_R|ψ>, poids dans le sous-espace logique du qubit droit
    - a, b: amplitudes <S_R|ψ>, <T0_R|ψ>
    """
    # États logiques du qubit droit (Qobj kets dans l’espace complet)
    qR = logical_qubits[1]
    S_R = qR["0"]
    T_R = qR["1"]

    # Amplitudes dans la base {|S>,|T0>} du qubit droit
    a = complex(S_R.overlap(final_state))   # <S_R|ψ>
    b = complex(T_R.overlap(final_state))   # <T_R|ψ>

    # Matrice densité projetée non normalisée: P|ψ><ψ|P restreinte à {S,T0}
    rho_unn = np.array([[a*np.conjugate(a), a*np.conjugate(b)],
                        [b*np.conjugate(a), b*np.conjugate(b)]], dtype=np.complex128)

    # Poids dans le sous-espace du qubit droit
    weight = float((rho_unn[0,0] + rho_unn[1,1]).real)

    # Normalisation (pour une matrice densité de trace = 1 sur ce sous-espace)
    if normalize and weight > eps:
        rho = rho_unn / weight
    else:
        rho = rho_unn

    # Cohérence et pureté
    if normalize:
        coherence = float(np.abs(rho[0,1]))                 # ∈ [0,1]
        purity    = float(np.trace(rho @ rho).real)         # ∈ [0,1]
    else:
        # version "indépendante du poids": normalise l'off-diagonal par √(ρ_SS ρ_TT)
        denom = np.sqrt(max(rho[0,0].real,0.0)*max(rho[1,1].real,0.0)) + eps
        coherence = float(np.abs(rho[0,1]) / denom) if denom > eps else 0.0
        tr = float((rho[0,0] + rho[1,1]).real)
        purity = float(np.trace(rho @ rho).real / (tr*tr + eps))

    return {
        "rho": rho,                # 2x2 complex
        "coherence": coherence,    # [0,1]
        "purity": purity,          # [0,1]
        "weight": weight,          # [0,1] si pas de fuite
        "a": a, "b": b
    }



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
    tau = args.get("tau", (args["t1"] - args["t0"]) / 6)#20.0) # changed
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
        rtol=min(_get_opt(SE_OPTS, "rtol", 1e-6), 5e-5),
        atol=min(_get_opt(SE_OPTS, "atol", 1e-8), 1e-7), # changed
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
def compute_or_load_baseline_opti(delta_t_vals, BASELINE_FILE,
                             imp_start_idx, num_sites, n_electrons, H_base, psi0_full, basis_occ, logical_qubits, nbr_pts):
    if (not FORCE_RECALC) and os.path.exists(BASELINE_FILE):
        try:
            data = np.load(BASELINE_FILE)
            if np.array_equal(data.get("delta_t_vals"), delta_t_vals):
                return data["phi0"]
        except Exception:
            pass
        print("⚠️ Baseline incompatible avec la grille actuelle. Recalcul…")

    print("🧭 Calcul baseline φ0 (ΔU=0) — version rapide…")
    # 1 seul calcul suffit
    T_eV, U_eV = compute_txU_for_pulse(delta_t=float(delta_t_vals[0]), delta_U_meV=0.0, imp_start_idx=imp_start_idx)
    H_pulse = build_spinful_hubbard_hamiltonian(num_sites, T_eV, U_eV, basis_occ)
    final_state = qubits_impulsion_lastonly(
        num_sites=num_sites, n_electrons=n_electrons,
        H_base=H_base, H_pulse=H_pulse,
        t_imp=t_imp, Delta_t=float(delta_t_vals[0]), T_final=T_final,
        psi0_full=psi0_full, nbr_pts=nbr_pts
    )
    a0, b0 = extract_qubit_L(final_state, logical_qubits)
    phi0_val = phase_relative(a0, b0)
    phi0 = np.full(len(delta_t_vals), phi0_val, dtype=np.float64)

    np.savez(BASELINE_FILE, phi0=phi0, delta_t_vals=delta_t_vals)
    return phi0

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
        print("map_detection : t_matrix_pulse=", T_eV)
        print("U_pul_vec : ", U_eV)
        H_pulse = build_spinful_hubbard_hamiltonian(num_sites, T_eV, U_eV, basis_occ)
        final_state = qubits_impulsion_lastonly(
            num_sites=num_sites, n_electrons=n_electrons,
            H_base=H_base, H_pulse=H_pulse,
            t_imp=t_imp, Delta_t=delta_t, T_final=T_final,
            psi0_full=psi0_full, nbr_pts=nbr_pts
        )
        a0, b0 = extract_qubit_R(final_state, logical_qubits)

        phi0[j] = phase_relative(a0, b0)

    np.savez(BASELINE_FILE, phi0=phi0, delta_t_vals=delta_t_vals)
    return phi0

# =================== Lignes fidelity ===================

def row_path(base_dir: str, i: int) -> str:
    return os.path.join(base_dir, f"{ROW_BASENAME}_{i:03d}.npy")

def _compute_row_for_deltaU(i, delta_U_meV, delta_t_vals, phi0, base_dir,
                            imp_start_idx, num_sites, n_electrons, H_base, psi0_full, basis_occ, logical_qubits, nbr_pts):
    """Écrit **incrémentalement** la ligne i (ΔU_i) dans un memmap .npy, valeur par valeur (Δt_j)."""
    t_line0 = time.perf_counter()
    out_file = row_path(base_dir, i)

    # Ouvre/crée memmap
    if (not FORCE_RECALC) and os.path.exists(out_file):
        fidelity_row_mm = open_memmap(out_file, mode='r+')
        if fidelity_row_mm.shape != (len(delta_t_vals),):
            del fidelity_row_mm
            os.remove(out_file)
            fidelity_row_mm = open_memmap(out_file, mode='w+', dtype='float32', shape=(len(delta_t_vals),))
            fidelity_row_mm[:] = np.nan
    else:
        fidelity_row_mm = open_memmap(out_file, mode='w+', dtype='float32', shape=(len(delta_t_vals),))
        fidelity_row_mm[:] = np.nan

    # Calcul incrémental par Δt en sautant ce qui est déjà rempli
    for j, delta_t in enumerate(tqdm(delta_t_vals,
                                     desc=f"  → ΔU = {delta_U_meV:.3f} meV",
                                     unit="Δt", leave=False)):
        if np.isfinite(fidelity_row_mm[j]):
            continue

        T_eV, U_eV = compute_txU_for_pulse(delta_t, delta_U_meV, imp_start_idx)
        H_pulse = build_spinful_hubbard_hamiltonian(num_sites, T_eV, U_eV, basis_occ)
        print("map_detection : t_matrix_pulse=", T_eV)
        print("U_pul_vec : ", U_eV)
        final_state = qubits_impulsion_lastonly(
            num_sites=num_sites, n_electrons=n_electrons,
            H_base=H_base, H_pulse=H_pulse,
            t_imp=t_imp, Delta_t=delta_t, T_final=T_final,
            psi0_full=psi0_full, nbr_pts=nbr_pts
        )
        aR, bR = extract_qubit_R(final_state, logical_qubits)
        dphi = wrap_pi(phase_relative(aR, bR) - phi0[j])
        fidelity_row_mm[j] = float(np.cos(0.5 * dphi)**2)
        prob = float(np.cos(0.5 * dphi)**2)
        metrics = right_qubit_metrics(final_state, logical_qubits, normalize=True)
    # coh   = metrics["coherence"]   # |rho_ST|
    # pur   = metrics["purity"]      # Tr rho^2
    # wght  = metrics["weight"]      # masse dans le sous-espace (fuites si <<1)

    # # Exemple 1: map de cohérence (graduée 0..1)
    # fidelity_row_mm[j] = float(coh)
        print(f"ΔU={delta_U_meV:.2f} meV Δt={delta_t*1e9:.3f} ns  |a|={abs(aR):.3f} |b|={abs(bR):.3f}  dφ={dphi:.3f} rad  p={prob:.3f}")


        if (j + 1) % 50 == 0:
            fidelity_row_mm.flush()
            gc.collect()

    fidelity_row_mm.flush()
    del fidelity_row_mm
    print(f"  ↳ ligne ΔU={delta_U_meV:.3f} meV faite en {_fmt_time(time.perf_counter() - t_line0)}")
    return out_file

def _estimate_per_line_time(delta_U_meV, delta_t_vals, phi0,
                            imp_start_idx, num_sites, n_electrons, H_base, psi0_full, basis_occ, logical_qubits, nbr_pts):
    nsamp = max(BENCH_MIN_SAMPLES, int(math.ceil(BENCH_FRAC_DT * len(delta_t_vals))))
    idxs  = np.linspace(0, len(delta_t_vals)-1, nsamp, dtype=int)
    dt_s  = delta_t_vals[idxs]

    start = time.perf_counter()
    for j, delta_t in enumerate(tqdm(dt_s, desc=f"⏱️ Bench ΔU={delta_U_meV:.3f} meV", leave=False, unit="Δt")):
        T_eV, U_eV = compute_txU_for_pulse(delta_t, delta_U_meV, imp_start_idx)
        H_pulse = build_spinful_hubbard_hamiltonian(num_sites, T_eV, U_eV, basis_occ)
        final_state = qubits_impulsion_lastonly(
            num_sites=num_sites, n_electrons=n_electrons,
            H_base=H_base, H_pulse=H_pulse,
            t_imp=t_imp, Delta_t=delta_t, T_final=T_final,
            psi0_full=psi0_full, nbr_pts=nbr_pts
        )
        aL, bL = extract_qubit_R(final_state, logical_qubits)
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

def build_fidelity_detector_rows(delta_U_vals, delta_t_vals, phi0, base_dir,
                          imp_start_idx, num_sites, n_electrons, H_base, psi0_full, basis_occ, logical_qubits, nbr_pts):
    print("🧪 Scan probas de non-changement de phase (cos²(Δφ/2))…")
    _summarize_grid(delta_U_vals, name="ΔU", unit="meV")

    # ETA global
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
        # Séquentiel : barre de progression extérieure avec ΔU affiché
        with tqdm(total=len(todo), desc="ΔU (meV)", unit="ΔU") as pbar:
            for i in todo:
                du = float(delta_U_vals[i])
                pbar.set_postfix_str(f"ΔU={du:.3f}")
                _compute_row_for_deltaU(i, du, delta_t_vals, phi0, base_dir,
                                        imp_start_idx, num_sites, n_electrons, H_base, psi0_full, basis_occ, logical_qubits, nbr_pts)
                gc.collect()
                pbar.update(1)

# =================== Reconstruction / plotting ===================
def load_fidelity_detector_map(U_vals, T_vals, base_dir):
    rows = []
    for i in range(len(U_vals)):
        f = row_path(base_dir, i)
        if os.path.exists(f):
            arr = np.load(f, mmap_mode='r')  # lecture en memmap → faible RAM
            nT = len(T_vals)
            if len(arr) == nT:
                rows.append(np.asarray(arr))
            elif len(arr) < nT:  # ligne partielle
                tmp = np.full(nT, np.nan, dtype=np.float32)
                tmp[:len(arr)] = np.asarray(arr)
                rows.append(tmp)
            else:  # ligne plus longue que la grille actuelle
                rows.append(np.asarray(arr)[:nT])
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

def plot_fidelity_detector_map(p_map, U_vals, T_vals, psi0_label=None, out_dir=None):
    plt.figure(figsize=(8, 6))
    extent = [T_vals[0]*1e9, T_vals[-1]*1e9, U_vals[0], U_vals[-1]]
    im = plt.imshow(p_map, aspect='auto', origin='lower', extent=extent,
                    cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar(im, label="Proba phase non modifiée  (cos²(Δφ/2))")
    plt.xlabel("Δt (ns)"); plt.ylabel("ΔU (meV)")
    title = "Détecteur (qubit droit) : probabilité que la phase n'ait pas changé"
    if isinstance(psi0_label, str) and psi0_label.strip():
        title += f"\nConfiguration : {psi0_label}"
    plt.title(title); plt.tight_layout()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe = _slug(psi0_label) if isinstance(psi0_label, str) else "config"
        fname = f"fidelity_map_detector_{safe}_{len(U_vals)}x{len(T_vals)}_{stamp}.png"
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"🖼️ Figure sauvegardée : {out_path}")
    plt.show()

def main_detector(delta_U_vals_full, delta_t_vals_full):
    print("🚀 Démarrage map_detection (nouveau pipeline)")
    t_total0 = time.perf_counter()

    # Grille de calcul (coarse)
    if UPSAMPLE_TO_HIGHRES:
        COARSE_NU = 220
        COARSE_NT = 220
        delta_U_vals = np.linspace(delta_U_vals_full.min(), delta_U_vals_full.max(), COARSE_NU)
        delta_t_vals = np.linspace(delta_t_vals_full.min(), delta_t_vals_full.max(), COARSE_NT)
    else:
        delta_U_vals = delta_U_vals_full
        delta_t_vals = delta_t_vals_full

    # ====================== Chemins et organisation ======================
    config_tag = _slug(psi0_label) if isinstance(psi0_label, str) and psi0_label else "config"

    RESULTS_ROOT = "detector_results"
    RES_TAG = f"{len(delta_U_vals_full)}x{len(delta_t_vals_full)}"

    data_dir  = os.path.join(RESULTS_ROOT, f"{config_tag}__psi0_{RES_TAG}")
    os.makedirs(data_dir, exist_ok=True)

    image_dir = os.path.join(data_dir, "images", f"{RES_TAG}__{config_tag}__detector")
    os.makedirs(image_dir, exist_ok=True)

    BASELINE_FILE = os.path.join(data_dir, "detector_baseline_phi0.npz")

    # 0) Replot-only si les données existent déjà pour cette grille (coarse)
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
        fidelity_map_coarse = load_fidelity_detector_map(delta_U_vals, delta_t_vals, data_dir)
        print("fidelity_map_coarse coarse Nb NaN :", np.isnan(fidelity_map_coarse).sum(), " / ", fidelity_map_coarse.size)

        if UPSAMPLE_TO_HIGHRES and ((len(delta_U_vals_full) != nU) or (len(delta_t_vals_full) != nT)):
            print(f"🧩 Interpolation vers {len(delta_U_vals_full)}x{len(delta_t_vals_full)}…")
            fidelity_map_full = interpolate_to_full(fidelity_map_coarse, delta_U_vals, delta_t_vals,
                                             delta_U_vals_full, delta_t_vals_full)
            np.save(os.path.join(data_dir, f"fidelity_detector_map_{len(delta_U_vals_full)}x{len(delta_t_vals_full)}.npy"), fidelity_map_full)
            plot_fidelity_detector_map(fidelity_map_full, delta_U_vals_full, delta_t_vals_full, psi0_label=psi0_label, out_dir=image_dir)
        else:
            plot_fidelity_detector_map(fidelity_map_coarse, delta_U_vals, delta_t_vals, psi0_label=psi0_label, out_dir=image_dir)
        print("✅ Replot terminé (aucun recalcul).")
        raise SystemExit(0)

    # 1) baseline φ0(Δt)
    t_step0 = time.perf_counter()
    phi0 = compute_or_load_baseline_opti(delta_t_vals, BASELINE_FILE,
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

    # 3) calcul des lignes ΔU (écriture incrémentale par Δt) + barres tqdm ΔU
    t_step1 = time.perf_counter()
    build_fidelity_detector_rows(delta_U_vals, delta_t_vals, phi0, data_dir,
                          idx_t_imp, num_sites, n_electrons, H_base, psi0_full, basis_occ, logical_qubits, nbr_pts)
    print(f"⏱️ Lignes ΔU calculées en {_fmt_time(time.perf_counter() - t_step1)}")

    # 4) reconstruction et upsampling éventuel
    t_step2 = time.perf_counter()
    fidelity_map_coarse = load_fidelity_detector_map(delta_U_vals, delta_t_vals, data_dir)
    print("fidelity_map_coarse Nb NaN :", np.isnan(fidelity_map_coarse).sum(), " / ", fidelity_map_coarse.size)

    if UPSAMPLE_TO_HIGHRES and ((len(delta_U_vals_full) != len(delta_U_vals)) or (len(delta_t_vals_full) != len(delta_t_vals))):
        print(f"🧩 Interpolation vers {len(delta_U_vals_full)}x{len(delta_t_vals_full)}…")
        fidelity_map_full = interpolate_to_full(fidelity_map_coarse, delta_U_vals, delta_t_vals,
                                         delta_U_vals_full, delta_t_vals_full)
        np.save(os.path.join(data_dir, f"fidelity_detector_map_{len(delta_U_vals_full)}x{len(delta_t_vals_full)}.npy"),
                fidelity_map_full)
        print(f"✅ Temps total d'exécution : {_fmt_time(time.perf_counter() - t_total0)}")
        plot_fidelity_detector_map(fidelity_map_full, delta_U_vals_full, delta_t_vals_full, psi0_label=psi0_label, out_dir=image_dir)
    else:
        print(f"✅ Temps total d'exécution : {_fmt_time(time.perf_counter() - t_total0)}")
        plot_fidelity_detector_map(fidelity_map_coarse, delta_U_vals, delta_t_vals, psi0_label=psi0_label, out_dir=image_dir)
    print(f"⏱️ Reconstruction + plot en {_fmt_time(time.perf_counter() - t_step2)}")





# =============================== MAIN ===============================
if __name__ == "__main__":
    _, _, psi0_full = _prepare_initial_state_qubits(num_sites, n_electrons, psi0, basis_occ, logical_qubits)

    # === Construit un H_base cohérent ===
    _imp_start = idx_t_imp
    _baseline_dt = max(1e-12, float((T_final / len(time_array))))
    T_base, U_base = compute_txU_for_pulse(delta_t=_baseline_dt, delta_U_meV=0.0, imp_start_idx=_imp_start)
    T_base = np.clip(np.nan_to_num(T_base, nan=0.0, posinf=0.0, neginf=0.0), -1e-3, 1e-3)
    U_base = np.clip(np.nan_to_num(U_base, nan=0.0, posinf=0.0, neginf=0.0), 1e-5, 0.5)
    H_base = build_spinful_hubbard_hamiltonian(num_sites, T_base, U_base, basis_occ)

    print("map_detection : t_matrix_not_pulse=", T_base)
    print("U_not_pul : ", U_base)

    main_detector(delta_U_vals_full, delta_t_vals_full)

