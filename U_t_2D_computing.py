#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
from scipy.constants import hbar, e, epsilon_0, pi
from scipy.sparse import diags
from scipy.linalg import eigh   # sert aussi pour les problèmes généralisés
from param_simu import (Delta_t, T_final, t_imp, delta_U_meV, 
                        dot_x, sigma_y, m_eff, a_meV_nm2, 
                        well_depths_meV, barrier_heights_meV, 
                        well_width_nm, barrier_widths_nm, build_adaptive_x_grid)

# =============================================================================
# Flags de validation
# =============================================================================
# Mode strict : U doit être fini et strictement > 0. Aucune substitution.
STRICT_U_SANITIZE = True
# (réservé si tu veux plus tard clipper/symétriser t de façon agressive)
ALLOW_T_CLIP = True

# =============================================================================
# Utilitaires généraux
# =============================================================================
def pulse_U(t_array, t_start=0.0, delta_t=1.0, delta_U_eV=5e-3):
    """Impulsion rectangulaire (amplitude en eV)."""
    return delta_U_eV * ((t_array >= t_start) & (t_array < t_start + delta_t))

def create_4dots_potential_with_custom_barriers(
    x_m,
    dot_positions_m,
    well_depths_meV=(40, 40, 40, 40),
    well_width_nm=10,
    barrier_12_meV=20, barrier_23_meV=80, barrier_34_meV=20,
    width_12_nm=5, width_23_nm=5, width_34_nm=5
):
    """Potentiel 1D en meV (puits gaussiens + barrières gaussiennes)."""
    V_meV = np.zeros_like(x_m)
    sig_w = (well_width_nm*1e-9) / 2.355  # FWHM -> sigma

    # puits
    for depth, x0 in zip(well_depths_meV, dot_positions_m):
        V_meV -= depth * np.exp(-(x_m - x0)**2 / (2*sig_w**2))

    # barrières
    for k, (b_meV, w_nm) in enumerate([(barrier_12_meV, width_12_nm),
                                       (barrier_23_meV, width_23_nm),
                                       (barrier_34_meV, width_34_nm)]):
        xc = 0.5*(dot_positions_m[k] + dot_positions_m[k+1])
        sig_b = (w_nm*1e-9) / (2*np.sqrt(2*np.log(2)))  # FWHM -> sigma
        V_meV += b_meV * np.exp(-(x_m - xc)**2 / (2*sig_b**2))

    return V_meV

def add_linear_tilt_eV(V_eV, x_m, tilt_total_meV=0.0):
    """Ajoute un très léger tilt linéaire (total = tilt_total_meV sur la longueur)."""
    if tilt_total_meV == 0.0:
        return V_eV
    x_nm = x_m*1e9
    L = x_nm.max() - x_nm.min()
    x0 = 0.5*(x_nm.max()+x_nm.min())
    tilt_meV = tilt_total_meV * (x_nm - x0) / L
    return V_eV + 1e-3*tilt_meV

def potential_over_time(
    a_meV_nm2, U_imp_eV, x_m, dot_positions_m,
    well_depths_meV=(40,40,40,40), well_width_nm=60,
    barrier_heights_meV=(300,400,300),
    barrier_widths_nm=(40,100,40),
    strategy="central_only"
):
    """
    Construit V(x,t) en eV. On module des barrières en meV via imp_meV = U_imp_eV*1e3.
    strategy:
      - "central_only": n'agit QUE sur la barrière centrale 2–3
                         B23 = max(1.0, b23 - imp_meV) ; B12=b12 ; B34=b34
    """
    out = []
    for imp_eV in U_imp_eV:
        imp_meV = float(imp_eV) * 1e3
        b12, b23, b34 = barrier_heights_meV
        w12, w23, w34 = barrier_widths_nm

        if strategy == "central_only":
            B12 = b12
            B23 = max(1.0, b23 - imp_meV)   # ne descend pas en-dessous de 1 meV
            B34 = b34
        else:
            B12, B23, B34 = b12, max(1.0, b23 - imp_meV), b34

        V_meV = create_4dots_potential_with_custom_barriers(
            x_m, dot_positions_m,
            well_depths_meV=well_depths_meV, well_width_nm=well_width_nm,
            barrier_12_meV=B12, barrier_23_meV=B23, barrier_34_meV=B34,
            width_12_nm=w12, width_23_nm=w23, width_34_nm=w34,
        )

        # Confinement harmonique (meV)
        x_nm = x_m * 1e9
        x0_nm = 0.5*(x_nm.min()+x_nm.max())
        V_harmo_meV = a_meV_nm2 * (x_nm - x0_nm)**2

        # Conversion en eV & murs externes doux
        V_eV = (V_meV + V_harmo_meV) * 1e-3
        V_ext_eV = 0.2e-3
        sigma_ext_nm = 20.0; offset_ext_nm = 25.0
        xl = x_nm[0] + offset_ext_nm; xr = x_nm[-1] - offset_ext_nm
        V_eV += V_ext_eV*np.exp(-(x_nm-xl)**2/(2*sigma_ext_nm**2))
        V_eV += V_ext_eV*np.exp(-(x_nm-xr)**2/(2*sigma_ext_nm**2))

        out.append(V_eV)
    return out


# =============================================================================
# Orbitales 1D (états propres)
# =============================================================================
def get_eigs(V_x_eV, x_m, m_eff, num_states=4):
    """Diag 1D → num_states premiers états. Retourne E (eV) et vecteurs colonnes φ(x)."""
    dx = x_m[1]-x_m[0]
    Nx = len(x_m)
    coeff = -(hbar**2)/(2*m_eff*dx**2)   # J
    lap = diags([np.ones(Nx-1), -2*np.ones(Nx), np.ones(Nx-1)], [-1,0,1]).toarray()
    H = coeff*lap + np.diag(V_x_eV*e)    # Joules
    E_J, vecs = eigh(H)                  # eigenvectors colonnes
    E_eV = E_J[:num_states]/e
    psi = vecs[:, :num_states]
    # normalise
    for k in range(psi.shape[1]):
        n = np.sqrt(np.trapezoid(np.abs(psi[:,k])**2, x_m))
        if n > 0:
            psi[:,k] /= n
    return E_eV, [psi[:,k].copy() for k in range(psi.shape[1])]

# =============================================================================
# Localisation robuste: diagonalisation de l'opérateur position X dans le sous-espace
# =============================================================================
def localize_by_x_operator(orbitals, x):
    """
    Diagonalise X_ij = <phi_i|x|phi_j> dans la base des états propres.
    Gère bien les quasi-dégénérescences → orbitales localisées gauche→droite.
    """
    n = len(orbitals)
    S = np.zeros((n,n), dtype=np.complex128)
    X = np.zeros((n,n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            S[i,j] = np.trapezoid(np.conj(orbitals[i])*orbitals[j], x)
            X[i,j] = np.trapezoid(np.conj(orbitals[i])*x*orbitals[j], x)
    # Problème généralisé: X c = lambda S c
    vals, vecs = eigh(X, S)
    # Construit les combos localisés et normalise
    loc = []
    for k in range(n):
        comb = sum(vecs[j,k]*orbitals[j] for j in range(n))
        comb /= np.sqrt(np.trapezoid(np.abs(comb)**2, x))
        loc.append(comb)
    # Trie gauche -> droite via <x>
    centers = [float(np.trapezoid(x*np.abs(p)**2, x)/np.trapezoid(np.abs(p)**2, x)) for p in loc]
    order = np.argsort(centers)
    loc = [loc[i] for i in order]
    centers = [centers[i] for i in order]
    return loc, centers

# (outil de reporting + rotation 2×2 déjà éprouvés)
def _norm(phi, x):  return np.sqrt(np.trapezoid(np.abs(phi)**2, x))
def _bary(phi, x):
    den = np.trapezoid(np.abs(phi)**2, x)
    return float(np.trapezoid(x*np.abs(phi)**2, x)/den) if den>0 else np.nan
def _mass_frac_window(phi, x, x0, w):
    win = (x >= x0 - w) & (x <= x0 + w)
    num = np.trapezoid(np.abs(phi[win])**2, x[win])
    den = np.trapezoid(np.abs(phi)**2, x)
    return float(num/den) if den>0 else 0.0

def assignment_report(orbitals, x, dot_x, window_nm=20, name="", thresh=None, delta_U_meV=20):
    w = window_nm*1e-9
    n, m = len(orbitals), len(dot_x)
    M = np.zeros((n,m))
    bary = [_bary(phi, x) for phi in orbitals]
    for i, phi in enumerate(orbitals):
        for j, xdot in enumerate(dot_x):
            M[i,j] = _mass_frac_window(phi, x, xdot, w)
    # print(f"\n=== Assignation report ({name}) ===")
    # for i in range(n):
    #     j_star = int(np.argmax(M[i])); second = float(np.sort(M[i])[-2]) if m>=2 else 0.0
    #     print(f"φ{i}: bary={bary[i]*1e9:7.2f} nm → dot{j_star} "
    #           f"(x={dot_x[j_star]*1e9:6.1f} nm)  mass@*={M[i,j_star]:.3f}  2nd={second:.3f}")
    # print("Confusion M[i,j]:\n", np.round(M,3))
    # if thresh is not None:
    #     bad = np.where(M.max(axis=1) < thresh)[0].tolist()
    #     if bad: print(f"⚠️  Orbitales < {thresh:.2f} :", bad)
    #     else:   print("✅ Toutes > seuil.")
    return M, bary

def rotate_pair_any(orbitals, x, dot_x, i, j, window_nm=20, ngrid=721):
    w = window_nm*1e-9
    pi = orbitals[i].copy(); pj = orbitals[j].copy()
    ni = _norm(pi, x)
    if ni>0: pi /= ni
    gamma = float(np.trapezoid(np.conj(pi)*pj, x).real)
    pj = pj - gamma*pi
    nj = _norm(pj, x)
    if nj>0: pj /= nj

    def mass_at(phi, x0): return _mass_frac_window(phi, x, x0, w)

    best = (-1.0, 0.0)
    thetas = np.linspace(0, np.pi/2, ngrid)
    for th in thetas:
        c, s = np.cos(th), np.sin(th)
        Ai, Aj = c*pi + s*pj, -s*pi + c*pj
        Ai /= _norm(Ai, x); Aj /= _norm(Aj, x)
        score = mass_at(Ai, dot_x[i]) + mass_at(Aj, dot_x[j])
        if score > best[0]: best = (score, th)

    th = best[1];  c, s = np.cos(th), np.sin(th)
    Ai, Aj = c*pi + s*pj, -s*pi + c*pj
    Ai /= _norm(Ai, x); Aj /= _norm(Aj, x)
    orbitals[i], orbitals[j] = Ai, Aj
    print(f"↪︎ rotation (φ{i},φ{j})  θ={th*180/np.pi:.2f}°  score={best[0]:.3f}")
    return orbitals

def localize_with_fallback(orbitals, x, dot_x, window_nm=20, thresh=0.85, max_iter=4, delta_U_meV=20):
    """Localise via opérateur X, puis raffine par rotations si besoin."""
    orbs, centers = localize_by_x_operator(orbitals, x)
    M, _ = assignment_report(orbs, x, dot_x, window_nm=window_nm, name="X-operator", thresh=thresh, delta_U_meV=delta_U_meV)
    if np.all(M.max(axis=1) >= thresh):
        return orbs
    # fallback: casse parité globale puis voisins
    for _ in range(max_iter):
        orbs = rotate_pair_any(orbs, x, dot_x, i=0, j=len(orbs)-1, window_nm=window_nm)
        for j in (0,1,2):
            orbs = rotate_pair_any(orbs, x, dot_x, i=j, j=j+1, window_nm=window_nm)
        M, _ = assignment_report(orbs, x, dot_x, window_nm=window_nm, name="refine", thresh=thresh, delta_U_meV=delta_U_meV)
        if np.all(M.max(axis=1) >= thresh):
            break
    return orbs

# =============================================================================
# t et U à partir d’orbitales localisées
# =============================================================================
def t_from_orbitals(V_x_eV, x_m, y_m, m_eff, sigma_y_m, orbitals):
    """Construit t_ij en eV entre voisins à partir des orbitales fournies."""
    dx = x_m[1]-x_m[0]; dy = y_m[1]-y[1] if False else y_m[1]-y_m[0]
    Nx, Ny = len(x_m), len(y_m)
    X, Y = np.meshgrid(x_m, y_m, indexing='ij')
    V2D_J = (V_x_eV*e)[:,None] * np.ones((Nx,Ny))
    g_y = (1/(np.pi*sigma_y_m**2))**0.25 * np.exp(-Y**2/(2*sigma_y_m**2))

    T = np.zeros((len(orbitals), len(orbitals)))
    for i in range(len(orbitals)-1):
        phi_i_2D = (orbitals[i][:,None]*g_y)
        phi_j_2D = (orbitals[i+1][:,None]*g_y)
        # normalisation 2D
        phi_i_2D /= np.sqrt(np.sum(np.abs(phi_i_2D)**2)*dx*dy)
        phi_j_2D /= np.sqrt(np.sum(np.abs(phi_j_2D)**2)*dx*dy)

        d2x = np.gradient(np.gradient(phi_j_2D, dx, axis=0), dx, axis=0)
        d2y = np.gradient(np.gradient(phi_j_2D, dy, axis=1), dy, axis=1)
        Hphi = -(hbar**2/(2*m_eff))*(d2x+d2y) + V2D_J*phi_j_2D  # Joules

        t_J = -np.sum(np.conj(phi_i_2D)*Hphi)*dx*dy
        T[i,i+1] = T[i+1,i] = abs(t_J/e)  # eV
    return T

def phi2D_from_phi_x(phi_x, x, y, sigma_y):
    Y = np.meshgrid(np.zeros_like(x), y, indexing='ij')[1]
    g = (1.0/(np.pi*sigma_y**2))**0.25 * np.exp(-Y**2/(2*sigma_y**2))
    psi = phi_x[:,None]*g
    dx = x[1]-x[0]; dy = y[1]-y[0]
    psi /= np.sqrt(np.sum(np.abs(psi)**2)*dx*dy)
    return psi

def U_soft_from_phi2D(phi2D, x, y, epsilon_r=11.7, a_soft=8e-9):
    dx = x[1]-x[0]; dy = y[1]-y[0]
    X, Y = np.meshgrid(x, y, indexing='ij')
    r = np.stack([X, Y], axis=-1).reshape(-1,2)
    rho = (np.abs(phi2D)**2).reshape(-1)
    d = r[:,None,:] - r[None,:,:]
    dist2 = np.einsum('ijk,ijk->ij', d, d) + a_soft**2
    K = 1.0/np.sqrt(dist2)
    pref = (e**2)/(4*np.pi*epsilon_0*epsilon_r)/e  # eV
    U = 0.5 * pref * (dx*dy)**2 * np.einsum('i,ij,j->', rho, K, rho)
    return float(U)

def U_soft_from_phi2D_chunked(phi2D, x, y, epsilon_r=11.7, a_soft=8e-9,
                              chunk_size=2000, use_float32=False):
    """
    Coulomb soft self-energy with O(N^2) FLOPs but O(chunk*N) memory.
    No full N×N allocation. Numériquement robuste (float64 par défaut).
    """
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])

    X, Y = np.meshgrid(x, y, indexing='ij')
    r = np.stack([X, Y], axis=-1).reshape(-1, 2)   # (N, 2)
    rho = (np.abs(phi2D)**2).reshape(-1)           # (N,)

    dtype = np.float32 if use_float32 else np.float64
    r   = r.astype(dtype, copy=False)
    rho = rho.astype(dtype, copy=False)

    N = r.shape[0]
    pref = (e**2) / (4*pi*epsilon_0*epsilon_r) / e  # eV
    acc = dtype(0.0)

    # Blocked accumulation
    for i0 in range(0, N, chunk_size):
        i1 = min(N, i0 + chunk_size)
        r_blk   = r[i0:i1]                # (m, 2)
        rho_blk = rho[i0:i1]              # (m,)

        d = r_blk[:, None, :] - r[None, :, :]
        dist2 = np.einsum('ijk,ijk->ij', d, d) + (a_soft**2)
        K = 1.0 / np.sqrt(dist2)          # (m,N)

        v = K @ rho                       # (m,)
        acc += np.dot(rho_blk, v)         # scalar

        del d, dist2, K, v

    U = 0.5 * pref * (dx*dy)**2 * float(acc)
    return float(U)

def U_vector_from_orbitals(orbitals, x, y, sigma_y, epsilon_r=11.7, a_soft=8e-9,
                           chunk_size=2000, use_float32=False):
    def phi2D_from_phi_x(phi_x, x, y, sigma_y):
        Y = np.meshgrid(np.zeros_like(x), y, indexing='ij')[1]
        g = (1.0/(np.pi*sigma_y**2))**0.25 * np.exp(-Y**2/(2*sigma_y**2))
        psi = phi_x[:, None] * g
        dx = x[1]-x[0]; dy = y[1]-y[0]
        psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx * dy)
        return psi

    Us = []
    for phi in orbitals:
        psi2D = phi2D_from_phi_x(phi, x, y, sigma_y)
        U = U_soft_from_phi2D_chunked(
            psi2D, x, y, epsilon_r=epsilon_r, a_soft=a_soft,
            chunk_size=chunk_size, use_float32=use_float32
        )
        Us.append(U)
    return np.array(Us, dtype=float)

# =============================================================================
# Plot helper
# =============================================================================
def plot_potential_with_orbitals(dot_x, V_x_eV, x_m, orbitals, title):
    plt.figure(figsize=(10,6))
    ax = plt.gca()
    plt.plot(x_m*1e9, V_x_eV*1e3, 'b', label='V(x) [meV]')
    colors = ['orange','green','red','purple','cyan','k']
    scale = 0.9*np.max(V_x_eV*1e3)
    for i,phi in enumerate(orbitals):
        plt.plot(x_m*1e9, scale*np.abs(phi)**2/np.max(np.abs(phi)**2),
                 color=colors[i%len(colors)], label=f"|φ_{i}|² (rescaled)")
    for i, xdot in enumerate(dot_x):
        ax.axvline(xdot*1e9, color='k', ls=':', lw=0.8)
        ax.text(xdot*1e9, ax.get_ylim()[1]*0.06, f"dot {i}", ha='center', fontsize=8)
    plt.xlabel("x [nm]"); plt.ylabel("Énergie [meV] / |φ|²")
    plt.title(title); plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()


# ============================ LAZY EXPORTS ===================================
# Public variables kept for backward compatibility (initially None)
t_matrix_not_pulse = None
t_matrix_pulse     = None
U_not_vec          = None
U_pul_vec          = None

# =============================================================================
# API "lazy" et "pure"
# =============================================================================
def _check_t_U_no_touch(t_mat, U_vec, name="(base)"):
    t = np.asarray(t_mat); U = np.asarray(U_vec)
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(U))):
        raise ValueError(f"[{name}] t/U contiennent NaN/Inf")
    # on vérifie SANS corriger
    if not np.allclose(t, t.T, atol=1e-12):
        raise ValueError(f"[{name}] t non hermitique (no-touch).")
    if np.any(U <= 0):
        raise ValueError(f"[{name}] U ≤ 0 (no-touch).")
    return t_mat, U_vec

def _compute_if_needed(x=None, y=None, time_array=None,
                       t_imp_override=None, Delta_t_override=None,
                       well_depths=None, barrier_heights=None,
                       a_parab_meV_nm2=None, delta_U_meV=None):
    """
    Calcule t/U si absent, puis remplit les globals.
    Utilise les fonctions déjà définies dans le module.
    """
    global t_matrix_not_pulse, t_matrix_pulse, U_not_vec, U_pul_vec
    if all(v is not None for v in (t_matrix_not_pulse, t_matrix_pulse, U_not_vec, U_pul_vec)):
        return

    # Defaults (reprennent tes constantes module)
    _t_imp   = t_imp if t_imp_override is None else t_imp_override
    _Delta_t = Delta_t if Delta_t_override is None else Delta_t_override
    _a_meV_nm2 = a_meV_nm2 if a_parab_meV_nm2 is None else a_parab_meV_nm2
    _well_depths = well_depths_meV if well_depths is None else tuple(well_depths)
    _barrier_heights = barrier_heights_meV if barrier_heights is None else tuple(barrier_heights)

    # Grilles par défaut si non fournies
    if x is None:
        sigma_x = 15e-9
        x = build_adaptive_x_grid(dot_x, sigma_x, well_width_nm, barrier_widths_nm,
                                  safety_pts=16, span_sigma=5)
    if y is None:
        Ny = 40
        y = np.linspace(-5*sigma_y, 5*sigma_y, Ny)
    if time_array is None:
        time_array = np.linspace(0.0, T_final, 200)

    # Potentiels hors/pdt impulsion
    U_imp0 = pulse_U(time_array, t_start=_t_imp, delta_t=_Delta_t, delta_U_eV=delta_U_meV*1e-3)
    pot_xt0 = potential_over_time(
        _a_meV_nm2, U_imp0, x, dot_x,
        well_depths_meV=_well_depths,
        well_width_nm=well_width_nm,
        barrier_heights_meV=_barrier_heights,
        barrier_widths_nm=barrier_widths_nm,
        strategy="central_only"
    )

    idx_t_imp     = np.searchsorted(time_array, _t_imp)
    idx_t_imp_end = np.searchsorted(time_array, _t_imp + _Delta_t)
    idx_not_imp   = max(0, idx_t_imp-1)
    idx_during    = min(idx_t_imp + 4, idx_t_imp_end-1)

    V_not = pot_xt0[idx_not_imp].copy()
    V_pul = pot_xt0[idx_during].copy()

    # Eigens + localisation
    _, orbs_not_raw = get_eigs(V_not, x, m_eff, num_states=4)
    _, orbs_pul_raw = get_eigs(V_pul, x, m_eff, num_states=4)
    orbs_not = localize_with_fallback(orbs_not_raw, x, dot_x, window_nm=20, thresh=0.80, max_iter=4)
    orbs_pul = localize_with_fallback(orbs_pul_raw, x, dot_x, window_nm=20, thresh=0.80, max_iter=4)

    # t et U (déjà en eV dans ce module)
    t_matrix_not = t_from_orbitals(V_not, x, y, m_eff, sigma_y, orbs_not)
    t_matrix_pul = t_from_orbitals(V_pul, x, y, m_eff, sigma_y, orbs_pul)
    U_not        = U_vector_from_orbitals(orbs_not, x, y, sigma_y, epsilon_r=11.7, a_soft=8e-9)
    U_pul        = U_vector_from_orbitals(orbs_pul, x, y, sigma_y, epsilon_r=11.7, a_soft=8e-9)

    # Validation stricte + assign
    t_matrix_not, U_not = _check_t_U_no_touch(t_matrix_not, U_not,  name="base")
    t_matrix_pul, U_pul = _check_t_U_no_touch(t_matrix_pul, U_pul,  name="pulse")

    t_matrix_not_pulse = t_matrix_not
    t_matrix_pulse     = t_matrix_pul
    U_not_vec          = U_not
    U_pul_vec          = U_pul

# =============================================================================
# ===============================  MAIN DEMO  =================================
# =============================================================================
if __name__ == "__main__":
    # Demo grid (example only; other scripts should set their own x,y,time_array)
    sigma_x = 15e-9
    x = build_adaptive_x_grid(dot_x, sigma_x, well_width_nm, barrier_widths_nm,
                              safety_pts=16, span_sigma=5)
    Ny = 40
    y = np.linspace(-5*sigma_y, 5*sigma_y, Ny)

    time_array = np.linspace(0.0, T_final, 200)
    U_imp = pulse_U(time_array, t_start=t_imp, delta_t=Delta_t, delta_U_eV=delta_U_meV*1e-3)

    pot_xt = potential_over_time(
        a_meV_nm2, U_imp, x, dot_x,
        well_depths_meV=well_depths_meV,
        well_width_nm=well_width_nm,
        barrier_heights_meV=barrier_heights_meV,
        barrier_widths_nm=barrier_widths_nm,
        strategy="central_only"
    )

    idx_t_imp     = np.searchsorted(time_array, t_imp)
    idx_t_imp_end = np.searchsorted(time_array, t_imp + Delta_t)
    idx_not_imp   = max(0, idx_t_imp-1)
    idx_during    = min(idx_t_imp + 4, idx_t_imp_end-1)

    V_not = pot_xt[idx_not_imp].copy()   # eV (hors impulsion)
    V_pul = pot_xt[idx_during].copy()    # eV (pendant impulsion)

    V_not = add_linear_tilt_eV(V_not, x, tilt_total_meV=0.0)

    # Eigens + localisation
    _, orbs_not_raw = get_eigs(V_not, x, m_eff, num_states=4)
    _, orbs_pul_raw = get_eigs(V_pul, x, m_eff, num_states=4)
    orbs_not = localize_with_fallback(orbs_not_raw, x, dot_x, window_nm=20, thresh=0.80, max_iter=4, delta_U_meV=delta_U_meV)
    orbs_pul = localize_with_fallback(orbs_pul_raw, x, dot_x, window_nm=20, thresh=0.80, max_iter=4, delta_U_meV=delta_U_meV)

    # t and U (demo)
    t_matrix_not_pulse = t_from_orbitals(V_not, x, y, m_eff, sigma_y, orbs_not)
    t_matrix_pulse     = t_from_orbitals(V_pul, x, y, m_eff, sigma_y, orbs_pul)
    print("\n t (not_imp) [eV]:\n", t_matrix_not_pulse)
    print("\n t (with_imp) [eV]:\n", t_matrix_pulse)

    U_not_vec = U_vector_from_orbitals(orbs_not, x, y, sigma_y, epsilon_r=11.7, a_soft=8e-9)
    U_pul_vec = U_vector_from_orbitals(orbs_pul, x, y, sigma_y, epsilon_r=11.7, a_soft=8e-9)
    print("\nU_not_imp (eV):", U_not_vec, "  (avg = %.3f meV)" % (1e3*np.mean(U_not_vec)))
    print("U_with_imp (eV):", U_pul_vec, " (avg = %.3f meV)" % (1e3*np.mean(U_pul_vec)))

    print("delta_U_meV : ", delta_U_meV)

    # Petite démo de phase
    hbar_eVs = sc.hbar/sc.e
    t12, t23, t34 = float(t_matrix_pulse[0,1]), float(t_matrix_pulse[1,2]), float(t_matrix_pulse[2,3])
    U_M = 0.5*(U_pul_vec[1] + U_pul_vec[2])
    J_M = 4*(t23**2)/U_M if U_M > 0 else 0.0
    phi_M = J_M*Delta_t/hbar_eVs
    print(f"\nt12={t12:.3e} eV   t23={t23:.3e} eV   t34={t34:.3e} eV")
    print(f"U_M={U_M:.3e} eV   J_M={J_M:.3e} eV   φ_M={phi_M:.3f} rad")

    # Plots (demo)
    plot_potential_with_orbitals(dot_x, V_not, x, orbs_not,
                                 title="V(x) + orbitales (hors impulsion)")
    plot_potential_with_orbitals(dot_x, V_pul, x, orbs_pul,
                                 title="V(x) + orbitales (pendant impulsion)")
