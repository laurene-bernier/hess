# param_simu.py
import numpy as np
import scipy.constants as sc
from scipy.constants import hbar, e
from qutip_utils import all_occupations, _st_states_for_pair, _build_logical_qubits, _ud_states_for_pair_from_st


import numpy as np


# =============================== Matériau / système ===============================
num_sites   = 4
n_electrons = 4
m_eff = 0.067 * sc.m_e

# Positions des puits (4 dots)
dot_x = np.array([-75e-9, -25e-9, 25e-9, 75e-9])

# ============================ Potentiel 1D (x) par défaut =========================
a_meV_nm2 = 6.5e-3
well_depths_meV     = (30, 5, 5, 40)
barrier_heights_meV = (35, 90, 50)
well_width_nm       = 23
barrier_widths_nm   = (15, 30, 25)  # (15, 30, 15 + asym_barrier_width) auparavant

# asym_well_depths = 10
# asym_barrier_width = 10
# asym_barrier_height = 15

# a_meV_nm2 = 6.5e-3
# well_depths_meV     = (30, 5, 20, 40)
# barrier_heights_meV = (35, 80, 60)
# well_width_nm       = 23
# barrier_widths_nm   = (15, 40, 30)  # (15, 30, 15 + asym_barrier_width) auparavant
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
# a_meV_nm2 = 6.5e-3
# well_depths_meV = (30, 5, 5, 40)
# barrier_heights_meV = (35, 90, 50)
# well_width_nm = 23
# barrier_widths_nm = (15, 30, 25)


# a_meV_nm2 = 6.5e-3
# well_depths_meV = (30, 5, 5, 35)
# barrier_heights_meV = (35, 90, 35)
# well_width_nm = 23
# barrier_widths_nm = (15, 30, 15)

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
Delta_t = 1.6e-9
T_final = 2.0e-9
delta_U_meV = 57

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

LOGICAL_BASIS = "st"  # ou "st" ud
psi0_label = "singlet-triplet"


psi0 = [st_L["S"].unit(),   -st_R["T0"].unit()]   # |S>, |T0>
#psi0 = [ud_L["ud"].unit(), ud_R["du"].unit()]


if LOGICAL_BASIS == "ud":
    logical_qubits = {
        "L": {"ud": ud_L["ud"], "du": ud_L["du"]},
        "R": {"ud": ud_R["ud"], "du": ud_R["du"]},
    }
else:
    # fallback ST (comme aujourd'hui)
    logical_qubits  = _build_logical_qubits(num_sites, basis_occ)
print(f"[info] logical basis = {LOGICAL_BASIS}")

nbr_pts = 300

coarse_nu  = 50
coarse_nt  = 50
TARGET_NU  = 33
TARGET_NT  = 33
delta_U_vals_full = np.linspace(40, 60, TARGET_NU)
delta_t_vals_full = np.linspace(t_imp, T_final - t_imp, TARGET_NT)
