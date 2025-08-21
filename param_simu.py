# param_simu.py
import numpy as np
import scipy.constants as sc
from scipy.constants import hbar, e
from qutip_utils import all_occupations, _st_states_for_pair, _build_logical_qubits, _ud_states_for_pair_from_st


import numpy as np

# --- 1) rotation ST -> UD (et inverse), + check d’unitarité -----------------
def st_to_ud(a_S, b_T0):
    inv = 1/np.sqrt(2)
    c_ud = inv*(a_S + b_T0)
    c_du = inv*(-a_S + b_T0)
    return c_ud, c_du

def ud_to_st(c_ud, c_du):
    inv = 1/np.sqrt(2)
    a_S  = inv*(c_ud - c_du)
    b_T0 = inv*(c_ud + c_du)
    return a_S, b_T0

def check_ud_rotation():
    U = (1/np.sqrt(2))*np.array([[1, 1],
                                 [-1, 1]], dtype=complex)  # (S,T0) -> (UD,DU)
    M = U.conj().T @ U
    print("[UD CHECK] U†U =\n", M)  # doit afficher ~[[1,0],[0,1]]

# --- 2) coordonnées de Bloch depuis le spinor UD ----------------------------
def bloch_coords_from_ud(alpha, beta):
    # |ψ> = alpha |↑↓> + beta |↓↑>, normalisé
    nrm = np.hypot(abs(alpha), abs(beta))
    if nrm == 0:
        return 0.0, 0.0, 0.0
    a = alpha / nrm
    b = beta  / nrm
    x = 2*np.real(np.conj(a)*b)
    y = 2*np.imag(np.conj(a)*b)
    z = abs(a)**2 - abs(b)**2
    return float(x), float(y), float(z)

def relative_phase_ud(alpha, beta):
    # φ = arg(alpha * conj(beta)) ∈ [-π, π]
    return float(np.angle(alpha * np.conj(beta)))

# --- 3) DEBUG principal : imprime tout ce qu’il faut ------------------------
def ud_debug_print(final_state, logical_qubits, label=""):
    inv = 1/np.sqrt(2)
    # a) on essaie en priorité l’extracteur UD s’il existe
    try:
        c_ud, c_du, pop = right_qubit_spinor_ud_unit(final_state, logical_qubits)
        src = "UD extractor"
    except NameError:
        # b) fallback : extraction ST puis rotation ST->UD
        a_S, b_T0 = extract_qubit_R(final_state, logical_qubits)
        c_ud, c_du = st_to_ud(a_S, b_T0)
        pop = abs(c_ud)**2 + abs(c_du)**2
        src = "ST→UD rotation"

    # Normalisation douce (pour l’affichage)
    nrm = np.hypot(abs(c_ud), abs(c_du))
    c_ud_n = c_ud/nrm if nrm else c_ud
    c_du_n = c_du/nrm if nrm else c_du

    x, y, z = bloch_coords_from_ud(c_ud_n, c_du_n)
    phi = relative_phase_ud(c_ud_n, c_du_n)

    print(f"\n[UD DEBUG] {label} via {src}")
    print(f"  pop_in_subspace = {pop:.6f}   (idéal ≈ 1.000000)")
    print(f"  amplitudes (normées): c_ud={c_ud_n:.6f}   c_du={c_du_n:.6f}")
    print(f"  probs: |c_ud|^2={abs(c_ud_n)**2:.6f}   |c_du|^2={abs(c_du_n)**2:.6f}")
    print(f"  Bloch: x={x:.6f}   y={y:.6f}   z={z:.6f}")
    print(f"  relative phase φ_ud = {phi:.6f} rad  (φ=arg(c_ud * conj(c_du)))")

    # Comparaison ST (si extracteur ST dispo) pour valider la rotation
    try:
        a_S_dir, b_T0_dir = extract_qubit_R(final_state, logical_qubits)
        a_S_rot, b_T0_rot = ud_to_st(c_ud_n, c_du_n)
        dS  = abs(a_S_dir - a_S_rot)
        dT0 = abs(b_T0_dir - b_T0_rot)
        print(f"  ST check:  |⟨S|ψ⟩|^2={abs(a_S_rot)**2:.6f}  |⟨T0|ψ⟩|^2={abs(b_T0_rot)**2:.6f}")
        print(f"  diffs vs direct ST: ΔS={dS:.3e}  ΔT0={dT0:.3e}  (attendu ~ <1e-10)")
    except NameError:
        pass

    # Hints visuels en texte
    if z > 0.98:
        print("  >>> proche de |↑↓> (pôle nord)")
    elif z < -0.98:
        print("  >>> proche de |↓↑> (pôle sud)")
    elif abs(x) > 0.98 and abs(z) < 0.1:
        print("  >>> proche équateur (combinaisons ~|T0> ou ~|S>)")

# --- 4) petits tests “smoke” faciles à lancer --------------------------------
def ud_smoke_tests(st_L, st_R, ud_L=None, ud_R=None):
    print("\n[UD SMOKE] Unitarité de la rotation ST->UD")
    check_ud_rotation()

    # Vérifie que tes états UD (si construits via Option B) sont orthonormés
    if ud_L is not None:
        u, d = ud_L["ud"], ud_L["du"]
        o = (u.dag()*d).full()[0,0]
        print(f"[UD SMOKE] gauche: <ud|du>={o:.3e}   ||ud||={u.norm():.6f}   ||du||={d.norm():.6f}")
    if ud_R is not None:
        u, d = ud_R["ud"], ud_R["du"]
        o = (u.dag()*d).full()[0,0]
        print(f"[UD SMOKE] droite: <ud|du>={o:.3e}   ||ud||={u.norm():.6f}   ||du||={d.norm():.6f}")

    # Sanity: reconstruire ST depuis UD pur
    a_S, b_T0 = ud_to_st(1.0+0j, 0.0+0j)  # |↑↓>
    print(f"[UD SMOKE] |↑↓> → ST : |S|^2={abs(a_S)**2:.3f}, |T0|^2={abs(b_T0)**2:.3f} (attendu 0.5/0.5)")
    a_S, b_T0 = ud_to_st(0.0+0j, 1.0+0j)  # |↓↑>
    print(f"[UD SMOKE] |↓↑> → ST : |S|^2={abs(a_S)**2:.3f}, |T0|^2={abs(b_T0)**2:.3f} (attendu 0.5/0.5)")


# =============================== Matériau / système ===============================
num_sites   = 4
n_electrons = 4
m_eff = 0.067 * sc.m_e

# Positions des puits (4 dots)
dot_x = np.array([-75e-9, -25e-9, 25e-9, 75e-9])

# ============================ Potentiel 1D (x) par défaut =========================
# a_meV_nm2 = 6.5e-3
# well_depths_meV     = (30, 5, 5, 30)
# barrier_heights_meV = (35, 75, 35)
# well_width_nm       = 23
# barrier_widths_nm   = (15, 30, 25)  # (15, 30, 15 + asym_barrier_width) auparavant

# asym_well_depths = 10
# asym_barrier_width = 10
# asym_barrier_height = 15

a_meV_nm2 = 6.5e-3
well_depths_meV     = (30, 5, 20, 40)
barrier_heights_meV = (35, 80, 60)
well_width_nm       = 23
barrier_widths_nm   = (15, 40, 30)  # (15, 30, 15 + asym_barrier_width) auparavant
# ========================= Grille adaptative en x (comme avant) ===================
def build_adaptive_x_grid(dot_x, sigma_x_m, well_width_nm, barrier_widths_nm,
                          safety_pts=14, span_sigma=5):
    min_feature_nm = min([well_width_nm, *barrier_widths_nm])
    target_dx = (min_feature_nm*1e-9) / safety_pts
    L = (dot_x.max()-dot_x.min()) + 2*span_sigma*sigma_x_m
    Nx = int(np.ceil(L/target_dx))
    if Nx % 2 == 0:
        Nx += 1
    return np.linspace(dot_x.min()-span_sigma*sigma_x_m,
                       dot_x.max()+span_sigma*sigma_x_m, Nx)

    x = np.linspace(dot_x.min()-span_sigma*sigma_x_m,
                    dot_x.max()+span_sigma*sigma_x_m, Nx)
    return x


# Helper: construit l'état d'une paire (st = st_L ou st_R)
def spin_pair(st, psi0_label, triplet_kind="T0"):
    if psi0_label in ("s", "singlet"):     return st["S"].unit()
    if psi0_label in ("t0", "triplet0"):   return st["T0"].unit()
    if psi0_label in ("t+", "uu", "upup"): return st["T+"].unit()   # |↑↑>
    if psi0_label in ("t-", "dd", "downdown"): return st["T-"].unit() # |↓↓>
    if psi0_label in ("ud", "updown"):     return (st["T0"] + st["S"]).unit()  # |↑↓>
    if psi0_label in ("du", "downup"):     return (st["T0"] - st["S"]).unit()  # |↓↑>
    if psi0_label in ("t", "triplet"):     return st[triplet_kind].unit()      # par défaut T0
    raise ValueError(f"psi0_label inconnu: {psi0_label}")

# --- Tes 8 scénarios (interprétation 'gauche–droite') ---
# NB: triplet = T0 par défaut; change triplet_kind="T+" ou "T-" si besoin.
cases = {
    "up-up":            ("uu", "uu"),          # L=|↑↑>, R=|↑↑>
    "down-down":        ("dd", "dd"),          # L=|↓↓>, R=|↓↓>
    "up-down":          ("uu", "dd"),          # L=|↑↑>, R=|↓↓>
    "down-up":          ("dd", "uu"),          # L=|↓↓>, R=|↑↑>
    "singlet-triplet":  ("singlet", "triplet"),# L=S,    R=T0 (par défaut)
    "singlet-singlet":  ("singlet", "singlet"),# L=S,    R=S
    "triplet-triplet":  ("triplet", "triplet"),# L=T0,   R=T0 (par défaut)
    "triplet-singlet":  ("triplet", "singlet") # L=T0,   R=S
}


num_sites   = 4
n_electrons = 4

# =============================================================================
# ===============================  DEFAULTS  ==================================
# =============================================================================
m_eff = 0.067 * sc.m_e
sigma_y = 10e-9
dot_x = np.array([-75e-9, -25e-9, 25e-9, 75e-9])

asym_well_depths = 10
asym_barrier_width = 10
asym_barrier_height = 15
# Global conf for potentials
a_meV_nm2 = 6.5e-3
well_depths_meV = (30, 5, 5, 30 + asym_well_depths)
barrier_heights_meV = (35, 90, 35 + asym_barrier_height)
well_width_nm = 23
barrier_widths_nm = (15, 30, 15 + asym_barrier_width)



# géo heatmap :
sigma_x = 15e-9
x = build_adaptive_x_grid(dot_x, sigma_x, well_width_nm, barrier_widths_nm,
                          safety_pts=16, span_sigma=5)

# ========================= Confinement & grille en x (NOUVEAU) ====================
# --- Auto-sigma_x depuis la courbure locale du potentiel ---
import numpy as np
from scipy.constants import hbar, e

# =========================== Confinement & grille en y (NOUVEAU) ==================
def omega_y_from_meVnm2(a_meV_nm2, m_eff):
    """a [meV/nm^2] -> ω_y [rad/s] via 0.5 m ω^2 y^2 = a*1e-3*e*(y_nm)^2."""
    if a_meV_nm2 <= 0.0:
        return 0.0
    k_J_per_m2 = a_meV_nm2 * 1e-3 * e * 1e18  # meV/nm^2 -> J/m^2
    return np.sqrt(2.0 * k_J_per_m2 / m_eff)

def sigma_y_from_omega(omega_y, m_eff):
    """Largeur GS de l’oscillateur harmonique: σ = sqrt(ħ/(m ω))."""
    if omega_y <= 0.0:
        return None
    return np.sqrt(hbar / (m_eff * omega_y))

def sigma_y_from_meVnm2(a_meV_nm2, m_eff):
    """σ_y à partir de a [meV/nm^2]."""
    return sigma_y_from_omega(omega_y_from_meVnm2(a_meV_nm2, m_eff), m_eff)

def build_y_grid(sigma_y_m, span=5.0, Ny=121):
    """Ancienne API simple: Ny points uniformes sur ± span·σ_y."""
    return np.linspace(-span*sigma_y_m, span*sigma_y_m, Ny)

def build_adaptive_y_grid(sigma_y_m, safety_pts=16, span_sigma=5.0):
    """
    Grille 'y' calquée sur la logique x: cible un pas dy tel que la FWHM
    (2·sqrt(2 ln 2)·σ) soit résolue par ~ safety_pts points.
    """
    fwhm_m   = 2.0*np.sqrt(2.0*np.log(2.0))*sigma_y_m
    target_dy = fwhm_m / safety_pts
    L = 2.0*span_sigma*sigma_y_m
    Ny = int(np.ceil(L/target_dy))
    if Ny % 2 == 0:
        Ny += 1
    return np.linspace(-span_sigma*sigma_y_m, +span_sigma*sigma_y_m, Ny)

# Paramètres de confinement en y
y_harmo_meV_nm2 = 100e-3          # mets 0 pour pas de confinement harmonique
_sigma_y_nominal = 15e-9        # fallback si y_harmo = 0
_sigma_from_conf = sigma_y_from_meVnm2(y_harmo_meV_nm2, m_eff)
sigma_y_eff = _sigma_from_conf if (_sigma_from_conf is not None) else _sigma_y_nominal

# Grille y (nouvelle logique adaptative)
y = build_adaptive_y_grid(sigma_y_eff, safety_pts=16, span_sigma=5.0)

print("here is y grid dimension : ",len(y), y[0]*1e9, y[-1]*1e9, sigma_y_eff*1e9)
# Rétro-compatibilité: certains scripts s’attendent à trouver 'sigma_y'
sigma_y = sigma_y_eff

# =============================== Temps par défaut ================================
t_imp   = 0.1e-9
Delta_t = 1.2e-9
T_final = 2.0e-9
delta_U_meV = 40

N_time    = 300
time_array = np.linspace(0.0, T_final, N_time)
idx_t_imp     = np.searchsorted(time_array, t_imp)
idx_t_imp_end = np.searchsorted(time_array, t_imp + Delta_t)

# =================== Pré-calculs invariants (espaces QuTiP) ======================
basis_occ       = all_occupations(num_sites, n_electrons)
st_L            = _st_states_for_pair(basis_occ, (0,1))
st_R            = _st_states_for_pair(basis_occ, (2,3))

ud_L = _ud_states_for_pair_from_st(st_L)
ud_R = _ud_states_for_pair_from_st(st_R)

ud_smoke_tests(st_L, st_R, ud_L=ud_L, ud_R=ud_R)

logical_qubits  = _build_logical_qubits(num_sites, basis_occ)

# Heatmap (exemples; tes scripts map_* règlent leurs grilles finement de leur côté)
psi0_label = "singlet-triplet"
#init_sig = [st_L["S"], st_R["T0"]] # change1 et normalement cest bon si mais bon 
psi0 = [st_L["S"].unit(), -st_R["T0"].unit()]
#psi0 = [ud_L["ud"].unit(), -ud_R["ud"].unit()]


coarse_nu  = 50
coarse_nt  = 50
TARGET_NU  = 10
TARGET_NT  = 10
delta_U_vals_full = np.linspace(35, 60, TARGET_NU)
delta_t_vals_full = np.linspace(t_imp, T_final - t_imp, TARGET_NT)
