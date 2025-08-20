#!/usr/bin/env python
# analysis_qutip.py  -----------------------------------------------
from math import e
import numpy as np
import time
from qutip_utils import (
    all_occupations,
    build_singlet_triplet_states,#tu voulait pas mettere celui la en signlet triplet ?
    _st_states_for_pair,
    top_hubbard_states_qubits,
    hbar_eVs, qubits_impulsion, qubits_double_impulsion,
    random_st_qubit_state
)

# === On ne garde qu'un seul chemin d'init pour t/U ===
# from U_t_2D_computing import params_sanitized, t_imp, Delta_t, T_final
# # 0) Récupère t/U (une seule fois) — pas de globals, pas de compute_static_demo
# t_base, U_base, t_pulse, U_pulse = params_sanitized()
import U_t_2D_computing as computing
from param_simu import(delta_U_meV, t_imp, Delta_t, T_final, 
                       num_sites, n_electrons, st_L, st_R, psi0_label,
                       spin_pair, cases)

t_matrix_not_pulse = None
t_matrix_pulse     = None
U_not_vec          = None
U_pul_vec          = None


# computing._compute_if_needed(delta_U_meV=delta_U_meV)

# t_base, U_base = computing.t_matrix_not_pulse, computing.U_not_vec
# t_pulse, U_pulse = computing.t_matrix_pulse, computing.U_pul_vec



# Garde-fous rapides

print("============= 2 qubits (4 sites)  —  impulsion =============")

# 0) Lance les pré-calculs (orbitales, t, U) UNE SEULE FOIS
#    (si compute_static_demo() imprime des infos comme "Delta_U_meV : 0", c'est normal)

# 1) Récupère puis assainit t/U (en eV)
# 0) Récupère t/U (lazy + sanitized) en eV


# Exemple d’utilisation :
left, right = cases[psi0_label]
psi0 = [spin_pair(st_L, left), spin_pair(st_R, right)]
print("left : ", left)
print("right : ", right)

# psi0_label = "singlet-triplet"             
# psi0 = [st_L["S"].unit(), -st_R["T0"].unit()]   
# Exemple : |ψ_k⟩ = a_k |S_k⟩ + b_k |T0_k⟩ (tu peux remplacer par random_st_qubit_state si tu veux)
aL, bL = 1/np.sqrt(2),  1/np.sqrt(2)
aR, bR = 1/np.sqrt(2),  1/np.sqrt(2)
# psiL   = (aL*st_L["S"]  + bL*st_L["T0"]).unit()
# psiR   = (aR*st_R["S"]  + bR*st_R["T0"]).unit()


if __name__=="__main__":
    t1 = time.time()
    t_base = np.zeros((num_sites, num_sites), dtype=float)
    t_base[0, 1] = t_base[1, 0] = 6e-5
    t_base[2, 3] = t_base[3, 2] = 2e-5# 4e-5
    t_base[1, 2] = t_base[2, 1] = 6e-5#4e-5

    U_base = np.array([3e-3, 3e-3, 1e-3, 1e-3])

    t_pulse = t_base.copy()
    print("t_puuuuuuuuuuuul : ", t_pulse)
    t_pulse[1, 2] = t_pulse[2, 1] = 0.6e-4 #4e-5    
    U_pulse=np.array([3e-3, 3e-3, 1e-3, 1e-3])
        

    print("analysis_qutip : t_base=", t_base)
    print("U_base=", U_base)
    print("t_pulse=", t_pulse)
    print("U_pulse=", U_pulse)

    # 2) Garde-fous stricts avant QuTiP
    assert t_base.shape  == (num_sites, num_sites),  f"t_base shape {t_base.shape} attendu {(num_sites, num_sites)}"
    assert t_pulse.shape == (num_sites, num_sites),  f"t_pulse shape {t_pulse.shape} attendu {(num_sites, num_sites)}"
    assert U_base.shape  == (num_sites,),            f"U_base shape {U_base.shape} attendu {(num_sites,)}"
    assert U_pulse.shape == (num_sites,),            f"U_pulse shape {U_pulse.shape} attendu {(num_sites,)}"

    for nm, arr in [("t_base", t_base), ("U_base", U_base),
                    ("t_pulse", t_pulse), ("U_pulse", U_pulse)]:
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{nm} contient des NaN/Inf après sanitisation.")
    if np.any(U_base <= 0) or np.any(U_pulse <= 0):
        raise ValueError("U ≤ 0 après sanitisation — vérifie compute_static_demo().")

    print("U_base : ", U_base)
    print("t_base :\n", t_base)
    print("t_pulse :\n", t_pulse)
    print("U_pulse : ", U_pulse)

    # ------------------------------------------------------------------
    # 3) États logiques initiaux  |ψ_L⟩ ⊗ |ψ_R⟩
    # ------------------------------------------------------------------

    # Variante random :
    # rng  = np.random.default_rng()
    # psiL = random_st_qubit_state(basis_occ, (0, 1), rng)
    # psiR = random_st_qubit_state(basis_occ, (2, 3), rng)

    print("Delta_U : ", delta_U_meV)
    t2_ = time.time()
    print(f"Temps de chargement : {t2_ - t1:.2f} secondes")

    # ------------------------------------------------------------------
    # 4) Appel du helper « impulsion »
    # ------------------------------------------------------------------
    print("⚙️  Lancement de la simulation, veuillez patienter...")
    print(psi0_label)

    t2, amps_list, coords_list = qubits_impulsion(
            num_sites, n_electrons,
            t_base, U_base,            # Hamiltonien de fond
            t_pulse, U_pulse,          # Hamiltonien pendant l’impulsion
            t_imp, Delta_t,            # fenêtre temporelle
            t_final=T_final,
            psi0=psi0,                 # liste « un sous-ket par qubit »
            display=True               # → deux sphères côte-à-côte
    )

    # ------------------------------------------------------------------
    # 5) Checks rapides (phases, dt, fidélités)
    # ------------------------------------------------------------------
    from math import pi
    hbar = 6.582119569e-16
    U_L = 0.5*(U_pulse[0] + U_pulse[1])
    U_R = 0.5*(U_pulse[2] + U_pulse[3])
    t12, t23 = float(t_pulse[0,1]), float(t_pulse[1,2])
    J_L = 4*t12**2 / U_L
    J_R = 4*t23**2 / U_R
    phi_L = J_L*Delta_t/hbar
    phi_R = J_R*Delta_t/hbar
    print(f"J_L={J_L:.3e} eV, J_R={J_R:.3e} eV,  φ_L={phi_L:.3f} rad, φ_R={phi_R:.3f} rad")

    time_array = np.array(t2)
    pulse_mask = (time_array >= t_imp) & (time_array < (t_imp + Delta_t))
    dt = float(time_array[1] - time_array[0])
    Delta_t_measured = float(pulse_mask.sum()) * dt
    print("Δt (s):", Delta_t, "mesuré:", Delta_t_measured)
    nz = t_pulse[np.nonzero(t_pulse)]
    if nz.size:
        print("t non nuls (eV): min", nz.min(), "max", nz.max())
    print("t12=", t12, "eV   t23=", t23, "eV")

    # Fidélité qubit gauche
    def try_get_left_amps(amps_list):
        import numpy as _np
        if amps_list is None:
            return None
        if isinstance(amps_list, dict):
            AL = amps_list.get("L", None)
            if AL is None:
                return None
            A = _np.asarray(AL)
            if A.ndim == 2:
                if A.shape[0] == 2: return A
                if A.shape[1] == 2: return A.T
            if isinstance(AL, (list, tuple)) and len(AL) == 2:
                aL = _np.asarray(AL[0]).reshape(1, -1)
                bL = _np.asarray(AL[1]).reshape(1, -1)
                return _np.vstack([aL, bL])
            return None
        if isinstance(amps_list, (list, tuple)):
            if len(amps_list) >= 2 and np.asarray(amps_list[0]).ndim == 1 and np.asarray(amps_list[1]).ndim == 1:
                aL = np.asarray(amps_list[0]).reshape(1, -1)
                bL = np.asarray(amps_list[1]).reshape(1, -1)
                return np.vstack([aL, bL])
            A0 = np.asarray(amps_list[0])
            if A0.ndim == 2:
                if A0.shape[0] == 2: return A0
                if A0.shape[1] == 2: return A0.T
        return None

    A_L = try_get_left_amps(amps_list)

    def amps_from_bloch(x, y, z):
        theta = np.arccos(np.clip(z, -1.0, 1.0))
        phi   = np.arctan2(y, x)
        alpha = np.cos(theta/2.0)
        beta  = np.exp(1j*phi) * np.sin(theta/2.0)
        v = np.array([[alpha],[beta]], dtype=complex)
        n = np.linalg.norm(v)
        return v/n if n > 0 else v

    if A_L is not None:
        k_end = int(np.searchsorted(time_array, t_imp + Delta_t, side='right') - 1)
        k_end = max(0, min(k_end, len(time_array)-1))
        psiL0  = A_L[:, 0].reshape(2, 1)
        psiLtf = A_L[:, k_end].reshape(2, 1)
    else:
        coords_L = np.array(coords_list[0])
        if coords_L.ndim == 2 and coords_L.shape[0] == 3:
            coords_L = coords_L.T
        x0, y0, z0 = coords_L[0]
        xT, yT, zT = coords_L[-1]
        psiL0  = amps_from_bloch(x0, y0, z0)
        psiLtf = amps_from_bloch(xT, yT, zT)

    g = psiLtf[0, 0] / abs(psiLtf[0, 0]) if abs(psiLtf[0, 0]) > 0 else 1.0
    psiLtf = psiLtf / g
    alpha0, beta0 = psiL0[0, 0], psiL0[1, 0]
    alphaT, betaT = psiLtf[0, 0], psiLtf[1, 0]
    fidelity_L = abs(np.conj(alpha0)*alphaT + np.conj(beta0)*betaT)**2
    print("Fidélité qubit gauche (fin impulsion) :", float(fidelity_L))
    t3 = time.time()
    print(f"⏱️ Reconstruction + plot en {t3-t2_:.2f}s")


