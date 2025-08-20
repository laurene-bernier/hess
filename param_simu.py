
import numpy as np
import scipy.constants as sc
from qutip_utils import all_occupations, _st_states_for_pair, _build_logical_qubits

    # ===== À RENSEIGNER AVANT LANCEMENT =====


# =============================================================================
# Grille adaptative (dx sûr quand on resserre/abaisse les barrières)
# ===========================================================$==================

def build_adaptive_x_grid(dot_x, sigma_x_m, well_width_nm, barrier_widths_nm,
                          safety_pts=14, span_sigma=5):
    """
    Choisit dx pour avoir ~safety_pts points au moins dans la FWHM la plus étroite.
    """
    min_feature_nm = min([well_width_nm, *barrier_widths_nm])
    target_dx = (min_feature_nm*1e-9) / safety_pts
    L = (dot_x.max()-dot_x.min()) + 2*span_sigma*sigma_x_m
    Nx = int(np.ceil(L/target_dx))
    if Nx % 2 == 0:
        Nx += 1
    x = np.linspace(dot_x.min()-span_sigma*sigma_x_m,
                    dot_x.max()+span_sigma*sigma_x_m, Nx)
    return x


# Helper: construit l'état d'une paire (st = st_L ou st_R)
def spin_pair(st, psi0_label, triplet_kind="T0"):
    psi0_label = psi0_label.lower()
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

asym_well_depths = 5
asym_barrier_width = 5
asym_barrier_height = 15
# Global conf for potentials
a_meV_nm2 = 6.5e-3
well_depths_meV = (30, 5, 5, 30 + asym_well_depths)
barrier_heights_meV = (50, 75, 50 + asym_barrier_height)
well_width_nm = 23
barrier_widths_nm = (15, 25, 15 + asym_barrier_width)



# géo heatmap :
sigma_x = 15e-9
x = build_adaptive_x_grid(dot_x, sigma_x, well_width_nm, barrier_widths_nm,
                        safety_pts=16, span_sigma=5)   # comme U_t_2D_computing
Ny = 40
y  = np.linspace(-5*sigma_y, 5*sigma_y, Ny)


# Default timing used by *examples* (your map code defines its own time grid)
t_imp   = 0.1e-9
Delta_t = 0.4e-9
T_final = 1.0e-9
delta_U_meV = 60 # 35-60 max

nbr_pts = 300

N_time    = 300 #200
time_array = np.linspace(0.0, T_final, N_time)
idx_t_imp  = int(np.searchsorted(time_array, t_imp))

# =================== Pré-calculs invariants (QuTiP) ===================
# dimension, bases, état initial
basis_occ = all_occupations(num_sites, n_electrons)
st_L = _st_states_for_pair(basis_occ, (0,1))
st_R = _st_states_for_pair(basis_occ, (2,3))
logical_qubits = _build_logical_qubits(num_sites, basis_occ)

# param heatmap
psi0_label = "singlet-triplet"       # change1 tu a plus le tchat ? ok x)je vais essayer de simuler pour ètre sur
#psi0 = [st_L["S"].unit(), -st_R["T0"].unit()]
init_sig = [st_L["S"], st_R["T0"]] # change1 et normalement cest bon si mais bon 
psi0 = [st_L["S"].unit(), -st_R["T0"].unit()]   

coarse_nu = 50
coarse_nt = 50

TARGET_NU = 10
TARGET_NT = 10

# bornes ΔU (meV) et Δt (s)
delta_U_vals_full = np.linspace(55, 60.0, TARGET_NU) #   change1 la ces bon je pence
delta_t_vals_full = np.linspace(0.4e-9, T_final - t_imp, TARGET_NT) #  





