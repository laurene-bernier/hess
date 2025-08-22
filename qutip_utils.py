#!/usr/bin/env python
import numpy as np
from itertools import combinations
from qutip import Qobj, basis as q_basis, tensor as qtensor
from qutip import Bloch
import scipy.constants as sc
import numpy.random as npr
import matplotlib.pyplot as plt

import warnings
# silence upcoming deprecations from QuTiP ≥ 5.3
warnings.filterwarnings("ignore", category=FutureWarning, module="qutip")

# ---------------------------------------------------------------------
#  Physical constants
# ---------------------------------------------------------------------
hbar = sc.hbar              # Planck‑reduced constant (J·s)
hbar_eVs = hbar / sc.e      # ≈ 6.582 119 569 e‑16 eV·s


# ---------------------------------------------------------------------
#  Generic utilities for a spin‑full 1D Hubbard chain
# ---------------------------------------------------------------------
def all_occupations(num_sites: int, n_electrons: int):
    """Return every occupation vector (0/1 list) with `n_electrons` among the 2·num_sites spin‑orbitals."""
    dim = 2 * num_sites
    return [list(np.array([1 if i in c else 0 for i in range(dim)]))
            for c in combinations(range(dim), n_electrons)]


def get_dirac_label(occ):
    """Human‑readable Dirac ket from an occupation vector."""
    num_sites = len(occ) // 2
    lbl = []
    for s in range(num_sites):
        up, dn = occ[2*s], occ[2*s+1]
        if up and dn:
            lbl.append('↑↓')
        elif up:
            lbl.append('↑')
        elif dn:
            lbl.append('↓')
        else:
            lbl.append('0')
    return "|" + ",".join(lbl) + "⟩"


def project_on_basis(occ, basis_occ):
    """
    Embed a full Fock‑space occupation vector into the restricted basis.
    """
    try:
        idx = basis_occ.index(list(occ))
    except ValueError as err:
        raise ValueError(f"{occ} not in restricted basis") from err
    v = np.zeros(len(basis_occ))
    v[idx] = 1
    return Qobj(v)


def is_neighbor(i, j, num_sites):
    """True if site *j* is the nearest‑neighbor of site *i* on a 1D chain."""
    return abs(i - j) == 1 and 0 <= j < num_sites


def build_spinful_hubbard_hamiltonian(num_sites, t, U, basis_occ):
    """
    Hubbard Hamiltonian (spinful) avec :
      - t : flottant ou matrice (num_sites×num_sites) des amplitudes de saut
      - U : flottant ou vecteur de taille num_sites des énergies de répulsion sur chaque site
      - basis_occ : liste des configurations (list of lists) de longueur 2*num_sites
    Renvoie un Qobj de dimension dim×dim.
    """
    # Prépare t_matrix et U_array
    t_matrix = np.array(t)
    if t_matrix.ndim == 0:
        # scalaire → matrice avec la même valeur sur chaque bond
        t_matrix = t_matrix * np.ones((num_sites, num_sites))
    U_array = np.array(U)
    if U_array.ndim == 0:
        # scalaire → tableau constant
        U_array = U_array * np.ones(num_sites)

    dim = len(basis_occ)
    H = np.zeros((dim, dim), dtype=complex)

    # --- terme de saut (hopping) ---------------------------------------
    for i, occ in enumerate(basis_occ):
        for site_from in range(num_sites):
            for spin in (0,1):           # 0 → ↑, 1 → ↓
                idx_from = 2*site_from + spin
                if occ[idx_from] == 0:
                    continue

                # voisins périodiques ou ouverts selon is_neighbor
                for site_to in (site_from-1, site_from+1):
                    if not is_neighbor(site_from, site_to, num_sites):
                        continue
                    idx_to = 2*site_to + spin
                    if occ[idx_to] == 1:
                        continue

                    amp = t_matrix[site_from, site_to]
                    new_occ = list(occ)
                    new_occ[idx_from] = 0
                    new_occ[idx_to]   = 1
                    try:
                        j = basis_occ.index(new_occ)
                    except ValueError:
                        continue

                    # signe fermionique
                    lo, hi = sorted((idx_from, idx_to))
                    sign = (-1)**sum(occ[lo+1:hi])
                    H[i,j] -= amp * sign

    # --- terme de répulsion U ------------------------------------------
    for i, occ in enumerate(basis_occ):
        for site in range(num_sites):
            # si les deux spins sont occupés sur le site
            if occ[2*site] == 1 and occ[2*site+1] == 1:
                H[i,i] += U_array[site]

    # Hermitianisation pour être sûr
    H = (H + H.conjugate().T) / 2
    return Qobj(H)


def time_evolve_state(H: Qobj, psi0: Qobj, times):
    """Wrapper around `qutip.mesolve` that returns the list of states."""
    from qutip import mesolve
    return mesolve(H, psi0, times, [], []).states


def build_singlet_triplet_states(basis_occ):
    """
    Explicit {|S⟩, |T₀⟩, |T₊⟩, |T₋⟩} in the 6‑dim subspace of 2 dots / 2 e⁻.
    """
    idx = basis_occ.index  # local shortcut

    # occupation vectors (↑₀,↓₀,↑₁,↓₁)
    up0_dn1 = [1,0,0,1]
    dn0_up1 = [0,1,1,0]
    up0_up1 = [1,0,1,0]
    dn0_dn1 = [0,1,0,1]

    dim = len(basis_occ)
    ket = lambda k: Qobj(np.eye(dim)[:, idx(k)])

    v1, v2 = ket(up0_dn1), ket(dn0_up1)
    S  = (v1 - v2).unit() / np.sqrt(2)
    T0 = (v1 + v2).unit() / np.sqrt(2)
    Tplus  = ket(up0_up1)
    Tminus = ket(dn0_dn1)

    return {"Singlet": S, "Triplet 0": T0,
            "Triplet +": Tplus, "Triplet -": Tminus}


def build_effective_qubit_state(amp_dict):
    """
    From amplitude time‑traces {S(t),T0(t)} → list of 2‑component Qobj.
    Useful when a downstream routine expects an array of qubit kets.
    """
    S, T0 = amp_dict["Singlet"], amp_dict["Triplet 0"]
    states = []
    for a, b in zip(S, T0):
        vec = np.array([a, b], dtype=complex)
        if np.linalg.norm(vec) > 0:
            vec = vec / np.linalg.norm(vec)
        states.append(Qobj(vec))
    return states


# -----------------------------------------------------------------
#  Site isolé : |0>=|↑>, |1>=|↓>
# -----------------------------------------------------------------
def _single_site_states(basis_occ, site):
    num_sites = len(basis_occ[0]) // 2
    occ_up = [1 if i % 2 == 0 else 0 for i in range(2*num_sites)]
    occ_dn = occ_up.copy();  occ_dn[2*site] = 0;  occ_dn[2*site+1] = 1
    dim = len(basis_occ)
    ket = lambda occ: Qobj(np.eye(dim)[:, basis_occ.index(occ)]).unit()
    return {"0": ket(occ_up), "1": ket(occ_dn)}

# -----------------------------------------------------------------
#  Qubits logiques : paires ST + éventuel site isolé
# -----------------------------------------------------------------
def _build_logical_qubits(num_sites, basis_occ):
    q = []
    # paires singlet–triplet
    for k in range(num_sites // 2):
        st = _st_states_for_pair(basis_occ, (2*k, 2*k+1))
        q.append({"0": st["S"], "1": st["T0"]})
    # site isolé si num_sites est impair
    if num_sites % 2:
        lone = 2 * (num_sites // 2)
        q.append(_single_site_states(basis_occ, lone))
    return q



# ---------------------------------------------------------------------
#  Internal helper : logical “AND” of two kets expressed in the *same*
#  restricted basis  (element‑wise product of amplitudes)
# ---------------------------------------------------------------------


def _ket_and(k1: Qobj, k2: Qobj) -> Qobj:
    """
    Combine two kets that already live in the *global* restricted basis
    by multiplying their component amplitudes term‑by‑term.
    Useful to build product states of distinct ST pairs without leaving
    the original Hilbert space (avoids the dimension blow‑up caused by
    `qtensor`).

    The result is automatically re‑normalised by the caller.
    """
    vec = np.multiply(k1.full().ravel(), k2.full().ravel())
    return Qobj(vec)


def _auto_time_grid(H_or_list, t_final, nbr_pts=None):
    """
    Heuristic Nyquist‑safe grid.

    Parameters
    ----------
    H_or_list : Qobj or iterable of Qobj
        One Hamiltonian *or* a collection of Hamiltonians that will be
        active at different times (e.g. base + pulse).  The routine
        evaluates the largest spectral width among them and chooses a
        sampling rate that resolves the fastest oscillation.
    t_final : float
        End of the evolution window (seconds, *physical*).
    nbr_pts : int or None
        Forced number of points.  If given, it is returned unchanged.

    Returns
    -------
    numpy.ndarray
        1‑D array of length *nbr_pts* containing the time samples in
        QuTiP’s internal units (ħ = 1).
    """
    # User‑forced value always wins
    if nbr_pts is not None:
        return np.linspace(0, t_final / hbar_eVs, nbr_pts)

    # Accept either a single Qobj or any iterable of Qobj
    if isinstance(H_or_list, (list, tuple)):
        Hs = H_or_list
    else:
        Hs = [H_or_list]

    # Largest energy gap across *all* supplied Hamiltonians
    dw_max = 0.0
    for H in Hs:
        e = H.eigenenergies()
        dw = e.max() - e.min()
        if dw > dw_max:
            dw_max = dw

    # Fallback when every H is proportional to identity
    if dw_max == 0:
        nbr_pts = 1000
    else:
        omega_max = dw_max / hbar_eVs
        nbr_pts = int(max(100000, np.ceil(8 * omega_max * t_final / (2*np.pi))))

    return np.linspace(0, t_final / hbar_eVs, nbr_pts)

# ---------------------------------------------------------------------
#  N  singlet–triplet qubits  (num_sites = 2N, n_electrons = 2N)
# ---------------------------------------------------------------------

def _ud_states_for_pair_from_st(st_dict):
    """Transforme les états ST d'une paire en états UD normalisés."""
    S  = st_dict["S"]
    T0 = st_dict["T0"]
    inv_sqrt2 = 1/np.sqrt(2)
    ud = (T0 + S) * inv_sqrt2   # |↑↓>
    du = (T0 - S) * inv_sqrt2   # |↓↑>
    return {"ud": ud.unit(), "du": du.unit()}


def _st_states_for_pair(basis_occ, pair):
    """
    Return {|S⟩,|T0⟩,|T+⟩,|T-⟩} as Qobj vectors for the two sites given in
    `pair = (i,j)` (with i != j).  Works for a *half‑filled* chain
    (n_electrons = num_sites) where every site hosts exactly one electron.
    Convention: every *other* site outside the pair is taken ↑ by default.

    Parameters
    ----------
    basis_occ : list[list[int]]
        List of occupation vectors that defines the restricted basis
        (typically produced by `all_occupations(num_sites, num_sites)`).
    pair : tuple(int,int)
        Indices `(i,j)` of the two sites that form the logical qubit,
        0 ≤ i,j < num_sites.

    Returns
    -------
    dict
        {"S":Qobj, "T0":Qobj, "T+":Qobj, "T-":Qobj}
        Kets are *normalised*.
    """
    i, j = pair
    if i == j:
        raise ValueError("pair indices must be different")
    num_sites = len(basis_occ[0]) // 2
    if not (0 <= i < num_sites and 0 <= j < num_sites):
        raise ValueError(f"indices {pair} outside lattice size {num_sites}")

    # --- helper: build a reference occupation vector -------------------
    base = [0]*(2*num_sites)
    for s in range(num_sites):
        base[2*s] = 1          # put an ↑ electron on every site

    # Build the four configurations that differ only on sites i,j
    occ_up_i_dn_j = base.copy()
    occ_up_i_dn_j[2*j]   = 0   # remove ↑ on j
    occ_up_i_dn_j[2*j+1] = 1   # add ↓ on j

    occ_dn_i_up_j = base.copy()
    occ_dn_i_up_j[2*i]   = 0
    occ_dn_i_up_j[2*i+1] = 1

    occ_up_i_up_j = base.copy()            # both ↑   (T+)
    occ_dn_i_dn_j = base.copy()            # both ↓   (T-)
    occ_dn_i_dn_j[2*i]   = 0
    occ_dn_i_dn_j[2*i+1] = 1
    occ_dn_i_dn_j[2*j]   = 0
    occ_dn_i_dn_j[2*j+1] = 1

    dim = len(basis_occ)
    def ket(occ):
        return Qobj(np.eye(dim)[:, basis_occ.index(occ)])

    k1, k2 = ket(occ_up_i_dn_j), ket(occ_dn_i_up_j)
    k_upup = ket(occ_up_i_up_j)
    k_dndn = ket(occ_dn_i_dn_j)

    S  = (k1 - k2).unit() / np.sqrt(2)
    T0 = (k1 + k2).unit() / np.sqrt(2)
    Tplus  = k_upup.unit()
    Tminus = k_dndn.unit()

    return {"S": S, "T0": T0, "T+": Tplus, "T-": Tminus}

# ---------------------------------------------------------------------
#  Random ST‑qubit initial state
# ---------------------------------------------------------------------
def random_st_qubit_state(basis_occ, pair, seed=None, rng=None):
    """Draw a Haar‑random state  a|S⟩+b|T0⟩ for the ST pair `pair` (uses `rng` or a fresh default RNG)."""
    if rng is None:
        rng = np.random.default_rng(seed)

    st = _st_states_for_pair(basis_occ, pair)
    # Haar‑random 2‑vector (draw two complex Gaussians and normalise)
    z = rng.normal(size=2) + 1j * rng.normal(size=2)
    z /= np.linalg.norm(z)
    a, b = z
    return (a * st["S"] + b * st["T0"]).unit()

# ---------------------------------------------------------------------
#  Shared helpers for multi‑qubit routines
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
def _prepare_initial_state_qubits(num_sites, n_electrons, psi0,
                                  basis_occ, logical_qubits):
    """Build the global initial ket from per‑qubit specs (ST pairs + optional lone spin)."""
    # helper : return computational basis kets whatever the key scheme
    def _get_k0k1(q):
        if "0" in q and "1" in q:          # spin qubit _or_ ST already renumbered
            return q["0"], q["1"]
        elif "S" in q and "T0" in q:       # raw singlet–triplet pair
            return q["S"], q["T0"]         # map S→0 , T0→1
        else:
            raise KeyError("Unrecognised qubit dictionary keys.")

    # ----- nombre réel de qubits (paires ST + spin seul éventuel) -----
    Nq = len(logical_qubits)

    # ---------- |00…0⟩ par défaut ------------------------------------
    if psi0 is None:
        k0, _ = _get_k0k1(logical_qubits[0])
        psi0_full = k0
        for q in logical_qubits[1:]:
            k0, _ = _get_k0k1(q)
            psi0_full = _ket_and(psi0_full, k0)
        return Nq, [ _get_k0k1(q)[0] for q in logical_qubits ], psi0_full.unit()

    # ---------- ψ₀ fourni sous forme de ket global -------------------
    if isinstance(psi0, Qobj) and psi0.shape == (len(basis_occ), 1):
        return Nq, None, psi0.unit()

    # ---------- liste d’états/qubits fournie -------------------------
    if not isinstance(psi0, (list, tuple)) or len(psi0) != Nq:
        raise ValueError(f"psi0 must contain {Nq} sub-states (one per qubit).")

    local_kets = []
    for coeffs, q in zip(psi0, logical_qubits):
        k0, k1 = _get_k0k1(q)

        if isinstance(coeffs, Qobj) and coeffs.shape == (2, 1):
            a, b = coeffs.full().flatten()
        elif isinstance(coeffs, Qobj) and coeffs.shape == (len(basis_occ), 1):
            a = k0.overlap(coeffs)
            b = k1.overlap(coeffs)
        elif isinstance(coeffs, (list, tuple, np.ndarray)) and len(coeffs) == 2:
            a, b = coeffs
        else:
            raise ValueError("psi0 entries must be : 2×1 ket, full ket, or (a,b).")

        local_kets.append((a * k0 + b * k1).unit())

    # ---------- combinaison des kets locaux --------------------------
    psi0_full = local_kets[0]
    for ket in local_kets[1:]:
        psi0_full = _ket_and(psi0_full, ket)

    # si produit nul, passer à une somme cohérente
    if psi0_full.norm() == 0:
        psi0_full = sum(local_kets).unit()

    return Nq, local_kets, psi0_full.unit()


def _extract_qubit_traces(states, qubit_defs):
    """For each logical qubit, return time‑series of amplitudes on |0⟩/|1⟩ and their Bloch coords."""
    amps_list, coords_list = [], []

    for q in qubit_defs:
        # --- identify the basis kets & labels ------------------------
        if "0" in q and "1" in q:          # spin qubit *or* ST pair already renumbered
            k0, k1 = q["0"], q["1"]
            lbl0, lbl1 = "0", "1"
        elif "S" in q and "T0" in q:       # raw ST pair definition
            k0, k1 = q["S"], q["T0"]
            lbl0, lbl1 = "0", "1"          # map S→0 , T0→1 for consistency
        else:
            raise KeyError("Unrecognised qubit dictionary keys.")

        amps_k = {lbl0: [], lbl1: []}
        coords_k = []

        for s in states:
            a = k0.overlap(s)
            b = k1.overlap(s)
            pop = abs(a)**2 + abs(b)**2
            if pop > 0:
                a_n, b_n = a/np.sqrt(pop), b/np.sqrt(pop)
            else:
                a_n = b_n = 0

            amps_k[lbl0].append(a)
            amps_k[lbl1].append(b)

            # Bloch coordinates
            x = 2 * np.real(np.conjugate(a_n) * b_n)
            y = 2 * np.imag(np.conjugate(a_n) * b_n)
            z = abs(b_n)**2 - abs(a_n)**2
            coords_k.append((x, y, z))

        # numpy‑fy for convenience
        for key in amps_k:
            amps_k[key] = np.asarray(amps_k[key])

        amps_list.append(amps_k)
        coords_list.append(np.asarray(coords_k))

    return amps_list, coords_list

# ---------------------------------------------------------------------
#  Nicely format a qubit ket  a|0⟩ + b|1⟩  for console output
# ---------------------------------------------------------------------
def _format_ab(a, b, precision=3):
    """
    Return the string  'a|0⟩ + b|1⟩'  with complex coefficients rounded
    to *precision* decimals.
    """
    def _c(z):
        return f"{z.real:.{precision}f}{'+' if z.imag >= 0 else ''}{z.imag:.{precision}f}j"
    return f"{_c(a)}|0⟩ + {_c(b)}|1⟩"


def _plot_bloch_list(coords_list):
    """
    Render one Bloch sphere (with red trajectory) per qubit trajectory
    in `coords_list`.
    """
    import matplotlib.pyplot as plt
    from qutip import Bloch

    Nq = len(coords_list)
    ncols = min(3, Nq)
    nrows = int(np.ceil(Nq / ncols))
    fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
    for idx, coords in enumerate(coords_list):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')
        b = Bloch(axes=ax)
        b.add_points(coords.T, meth='s', colors=['r'])
        b.xlabel = ['$x$', '']
        b.ylabel = ['$y$', '']
        b.zlabel = ['$|1⟩$', '$|0⟩$']
        b.render()
        if idx == 0:
            ax.set_title('Qubit')
        else: ax.set_title('Detector')
    plt.tight_layout()


def top_hubbard_states_qubits(num_sites, n_electrons, t, U,
                                    t_final=1e-12, nbr_pts=None, psi0=None,
                                    display=False):
    """
    Gère num_sites = 2N (paires de points quantiques).
    - Chaque paire (2k, 2k+1) définit un qubit ST : |0>=|S>, |1>=|T0>.
    - Retour :
        times      : (N_t,)   secondes
        amps_list  : [ {'S':array, 'T0':array},  ...  ]  (longueur N)
        coords_list: [ (N_t,3) ndarray , ... ]            (longueur N)
    psi0 can be (i) a full ket, (ii) a list of per‑qubit kets (2×1 or full), or (iii) a list of (a,b) coefficient pairs.
    """
    if num_sites % 2 != 0:
        raise ValueError("num_sites doit être un multiple de 2.")
    if n_electrons != num_sites:
        raise ValueError("Pour l’instant on suppose un remplissage (1,1,1,1…).")

    Nq = num_sites // 2           # nombre de qubits logiques
    basis_occ = all_occupations(num_sites, n_electrons)
    H = build_spinful_hubbard_hamiltonian(num_sites, t, U, basis_occ)
    times = _auto_time_grid(H, t_final, nbr_pts)

    # -------------------- projecteurs ST pour chaque paire --------------------
    st_pairs = [ _st_states_for_pair(basis_occ, (2*k, 2*k+1))  for k in range(Nq) ]

    # -------------------- état initial ---------------------------------------
    if psi0 is None:
        # |00…0⟩ logique  (tous en singlet)
        psi0 = st_pairs[0]["S"]
        for k in range(1, Nq):
            psi0 = _ket_and(psi0, st_pairs[k]["S"])
        psi0 = psi0.unit()

    elif isinstance(psi0, Qobj) and psi0.shape == (len(basis_occ), 1):
        # l’utilisateur fournit déjà un ket dans la base complète
        pass

    elif isinstance(psi0, (list, tuple)):
        if len(psi0) != Nq:
            raise ValueError(f"psi0 doit contenir {Nq} sous-états (un par qubit).")

        local_kets = []
        for k, (coeffs, st) in enumerate(zip(psi0, st_pairs)):
            S_k, T0_k = st["S"], st["T0"]

            # --- décode le couple (a,b) ---
            if isinstance(coeffs, Qobj) and coeffs.shape == (2, 1):
                a, b = coeffs.full().flatten()
            elif isinstance(coeffs, Qobj) and coeffs.shape == (len(basis_occ), 1):
                # L’utilisateur fournit un ket déjà défini dans la base restreinte
                # → on extrait automatiquement les amplitudes sur |S⟩ et |T0⟩
                a = S_k.overlap(coeffs)
                b = T0_k.overlap(coeffs)
            elif isinstance(coeffs, (list, tuple, np.ndarray)) and len(coeffs) == 2:
                a, b = coeffs
            else:
                raise ValueError(
                    f"psi0[{k}] doit être un ket 2×1, un ket de dimension complète, "
                    "ou une séquence (a,b).")

            local_kets.append((a * S_k + b * T0_k).unit())

        # Combine the per‑qubit kets by **adding** them coherently rather than
        # performing an element‑wise product.  This ensures that the global
        # ket has non‑zero norm even when the local supports are disjoint.
        psi0 = local_kets[0].copy()
        for ket in local_kets[1:]:
            psi0 += ket
        psi0 = psi0.unit()

    else:
        raise ValueError("psi0: valeur non reconnue")

    # --- pretty‑print the logical state of every qubit ----------------
    for k, st in enumerate(st_pairs):
        a0 = st["S"].overlap(psi0)
        b0 = st["T0"].overlap(psi0)
        print(f"Qubit {k} : {_format_ab(a0, b0)}")

    print(f"[top_hubbard_states_qubits] État initial global : {psi0}")

    # -------------------- évolution temporelle --------------------------------
    states = time_evolve_state(H, psi0, times)

    amps_list, coords_list = [], []
    for k in range(Nq):
        S_k, T0_k = st_pairs[k]["S"], st_pairs[k]["T0"]
        amps_k = {"S": [], "T0": []}
        coords_k = []
        for s in states:
            a = S_k.overlap(s)
            b = T0_k.overlap(s)
            pop = abs(a)**2 + abs(b)**2
            a_n, b_n = (a/np.sqrt(pop), b/np.sqrt(pop)) if pop > 0 else (0, 0)
            amps_k["S"].append(a)
            amps_k["T0"].append(b)
            x = 2*np.real(np.conjugate(a_n)*b_n)
            y = 2*np.imag(np.conjugate(a_n)*b_n)
            z = abs(b_n)**2 - abs(a_n)**2
            coords_k.append((x, y, z))
        for key in amps_k:
            amps_k[key] = np.asarray(amps_k[key])
        amps_list.append(amps_k)
        coords_list.append(np.asarray(coords_k))

    # -------------------- affichage optionnel ---------------------------------
    if display:
        import matplotlib.pyplot as plt
        from qutip import Bloch
        ncols = min(3, Nq)                  # max 3 par ligne
        nrows = int(np.ceil(Nq / ncols))
        fig = plt.figure(figsize=(4*ncols, 4*nrows))
        for idx, coords in enumerate(coords_list):
            ax = fig.add_subplot(nrows, ncols, idx+1, projection='3d')
            b = Bloch(axes=ax)
            b.add_points(coords.T, meth='s', colors=['r'])
            b.xlabel = ['$x$',''];  b.ylabel = ['$y$','']
            b.zlabel = ['$|1⟩$','$|0⟩$']
            b.render()
            if idx == 0:
                ax.set_title('Qubit')
            else: ax.set_title('Detector')
        plt.tight_layout()

    return times * hbar_eVs, amps_list, coords_list

from qutip import expect, Qobj

def _extract_qubit_traces_v2(states, logical_qubits):
    """
    Extrait, à chaque instant, les états de chaque qubit logique
    sous forme de Qobj, et leurs coordonnées sur la sphère de Bloch.
    """
    amps_list = []
    coords_list = []

    for psi in states:  # boucle sur les états temporels
        qubits_at_t = []
        coords_at_t = []

        for q in logical_qubits:
            a = q["0"].overlap(psi)
            b = q["1"].overlap(psi)

            # Crée un état Qobj pour le qubit logique : |ψ⟩ = a|0⟩ + b|1⟩
            qobj = Qobj([[a], [b]], dims=[[2], [1]])  # colonne 2×1
            qubits_at_t.append(qobj)

            # Coordonnées sur la sphère de Bloch
            x = 2 * (a.conjugate() * b).real
            y = 2 * (a.conjugate() * b).imag
            z = (abs(a)**2 - abs(b)**2).real
            coords_at_t.append([x, y, z])

        amps_list.append(qubits_at_t)
        coords_list.append(np.array(coords_at_t))

    return amps_list, np.array(coords_list)


def qubits_impulsion(num_sites, n_electrons,
                                        t_base, U_base,
                                        t_pulse, U_pulse,
                                        t_imp, Delta_t,
                                        t_final=1e-12, nbr_pts=None,
                                        psi0=None, display=True):
    """
    Identique à top_hubbard_states_qubits, mais :
      – H_base :  t = t_base , U = U_base
      – H_pulse:  t = t_pulse, U = U_pulse   appliqué pendant [t_imp, t_imp+Δt]

    Paramètres supplémentaires
    --------------------------
    t_imp   : float (s)    début de la fenêtre
    Delta_t : float (s)    largeur de la fenêtre
    """
    # --------- vérifications rapides ----------------------------------
    print(f"🧪 t_imp = {t_imp:e}, Δt = {Delta_t:e}, T_final = {t_final:e}")

    if not (0 <= t_imp < t_final and 0 < Delta_t <= t_final-t_imp):
        raise ValueError("Fenêtre d’impulsion hors-plage.")

    Nq = num_sites//2
    basis_occ = all_occupations(num_sites, n_electrons)

    # juste après basis_occ = all_occupations(...)
    try:
        from param_simu import LOGICAL_BASIS, ud_L, ud_R
        if LOGICAL_BASIS == "ud":
            logical_qubits = [
                {"0": ud_L["ud"], "1": ud_L["du"]},   # qubit gauche : |0>=|↑↓>, |1>=|↓↑>
                {"0": ud_R["ud"], "1": ud_R["du"]},   # qubit droit  : |0>=|↑↓>, |1>=|↓↑>
            ]
        else:
            logical_qubits = _build_logical_qubits(num_sites, basis_occ)  # ST
    except Exception:
        logical_qubits = _build_logical_qubits(num_sites, basis_occ)      # ST fallback


    # --- deux Hamiltoniens “figés” ------------------------------------
    H_base  = build_spinful_hubbard_hamiltonian(num_sites, t_base,  U_base,  basis_occ)
    H_pulse = build_spinful_hubbard_hamiltonian(num_sites, t_pulse, U_pulse, basis_occ)

    # --- fonctions indicatrices (QuTiP accepte des lambdas) -----------
    t0, t1 = t_imp/hbar_eVs, (t_imp+Delta_t)/hbar_eVs   # -> unités QuTiP (ħ=1)
    f_pulse = lambda t, *_: float(t0 <= t < t1)
    f_base  = lambda t, *_: 1.0 - f_pulse(t)

    H_td = [[H_base,  f_base],
            [H_pulse, f_pulse]]

    # ---------------- grille temporelle -------------------------------
    times = _auto_time_grid([H_base, H_pulse], t_final, nbr_pts)

    # ---------------- état initial (copié de la version “qubits”) -----
    # → réutilise exactement la même routine que plus haut
    Nq, local_kets, psi0_full = _prepare_initial_state_qubits(
        num_sites, n_electrons, psi0, basis_occ, logical_qubits)
    for k, q in enumerate(logical_qubits):
        a0 = q["0"].overlap(psi0_full)
        b0 = q["1"].overlap(psi0_full)
        print(f"Qubit {k} : {_format_ab(a0, b0)}")

    # ------------------ évolution -------------------------------------
    from qutip import mesolve
    states = mesolve(H_td, psi0_full, times, [], []).states

    # -------------- extraction amplitudes, coords, etc. ---------------
    # on réutilise la même boucle que dans top_hubbard_states_qubits
    amps_list, coords_list = _extract_qubit_traces(states, logical_qubits)

    # -------------------- affichage optionnel ---------------------------------
    #print("coords_list shape:", [c.shape for c in coords_list])

    if display:
        import matplotlib.pyplot as plt
        from qutip import Bloch

        # repère la fenêtre d’impulsion dans l’index temporel
        pulse_mask = (times >= t0) & (times < t1)
        print("pulse_mask shape:", pulse_mask.shape)

        Nq = len(coords_list)
        ncols = min(3, Nq)
        nrows = int(np.ceil(Nq / ncols))
        fig = plt.figure(figsize=(4 * ncols, 4 * nrows))

        # save attempt
        # # sauvegarde propre
        # import os
        # from param_simu import delta_U_meV
        # out_dir = "sphere_bloch"
        # os.makedirs(out_dir, exist_ok=True)
        # fname = f"spheres_dU_{float(delta_U_meV):.3f}meV_dT_{float(Delta_t)*1e9:.3f}ns.png"
        # out_path = os.path.join(out_dir, fname)
        # fig.savefig(out_path, dpi=300, bbox_inches="tight")
        # print("🖼️ Image sauvegardée :", out_path)

        for idx, coords in enumerate(coords_list):
            if coords.shape[0] != pulse_mask.shape[0]:
                print(f"[WARN] Coordonnées du qubit {idx} incompatibles avec la grille temporelle. Ignoré.")
                continue
            ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')
            b = Bloch(axes=ax)

            # Trajectoire hors‑impulsion : rouge
            if (~pulse_mask).any():
                b.add_points(coords[~pulse_mask].T, meth='s', colors=['r'])

            # Trajectoire pendant l’impulsion : vert
            if pulse_mask.any():
                b.add_points(coords[pulse_mask].T, meth='s', colors=['g'])

            b.xlabel = ['$x$', '']
            b.ylabel = ['$y$', '']
            b.zlabel = ['$|1⟩$', '$|0⟩$']
            # -----------------------------------------------------------------------------------------------------------------------------
            try:
                from param_simu import LOGICAL_BASIS
            except Exception:
                LOGICAL_BASIS = "st"

            if LOGICAL_BASIS == "ud":
                b.zlabel = [r"$|\uparrow\downarrow\rangle$", r"$|\downarrow\uparrow\rangle$"]
                b.xlabel = [r"$|T_0\rangle$", r"$|S\rangle$"]
            else:
                b.zlabel = [r"$|T_0\rangle$", r"$|S\rangle$"]
                b.xlabel = [r"$|\uparrow\downarrow\rangle$", r"$|\downarrow\uparrow\rangle$"]
            #----------------------------------------------------------------------------------------------------------------------------------
            b.render()
            ax.set_title('Qubit' if idx == 0 else 'Detector')
        plt.tight_layout()
        plt.show()  # <= ⭐ C’EST ICI QU’IL MANQUAIT QUELQUE CHOSE

    return times*hbar_eVs, amps_list, coords_list


def qubits_double_impulsion(num_sites, n_electrons,
                                               t_base,  U_base,
                                               t_pulse, U_pulse,
                                               t_imp1,  Delta_t,
                                               t_imp2,
                                               t_final=1e-12, nbr_pts=None,
                                               psi0=None, display=False):
    """
    Deux fenêtres d’impulsion :
        – première :  [t_imp1, t_imp1+Δt[
        – seconde  :  [t_imp2, t_imp2+Δt[
    Tout le reste du comportement (entrée/sortie) est identique
    à `top_hubbard_states_qubits_impulsion`.
    """
    # -- vérifs basiques -------------------------------------------------
    if num_sites % 2 or n_electrons != num_sites:
        raise ValueError("Chaîne demi-remplie requise (num_sites = 2N, n_electrons = 2N).")
    if not (0 <= t_imp1 < t_final and 0 < Delta_t <= t_final - t_imp1):
        raise ValueError("Fenêtre d’impulsion #1 hors plage.")
    if not (0 <= t_imp2 < t_final and 0 < Delta_t <= t_final - t_imp2):
        raise ValueError("Fenêtre d’impulsion #2 hors plage.")
    if max(t_imp1, t_imp2) < min(t_imp1, t_imp2) + Delta_t:
        raise ValueError("Les deux impulsions se chevauchent ; séparez-les ou réduisez Δt.")

    # -- préparation du système -----------------------------------------
    basis_occ = all_occupations(num_sites, n_electrons)
    logical_qubits = _build_logical_qubits(num_sites, basis_occ)

    H_base   = build_spinful_hubbard_hamiltonian(num_sites, t_base,  U_base,  basis_occ)
    H_pulse  = build_spinful_hubbard_hamiltonian(num_sites, t_pulse, U_pulse, basis_occ)

    # instants (unités QuTiP : ħ = 1)
    t0, t1 = t_imp1 / hbar_eVs, (t_imp1 + Delta_t) / hbar_eVs
    t2, t3 = t_imp2 / hbar_eVs, (t_imp2 + Delta_t) / hbar_eVs

    f_pulse = lambda t, *_: float(t0 <= t < t1) + float(t2 <= t < t3)
    f_base  = lambda t, *_: 1.0 - f_pulse(t)

    H_td = [[H_base,  f_base],
            [H_pulse, f_pulse]]

    times = _auto_time_grid([H_base, H_pulse], t_final, nbr_pts)

    # -------- état initial (ré-utilise le helper commun) ---------------
    _, _, psi0_full = _prepare_initial_state_qubits(
        num_sites, n_electrons, psi0, basis_occ, logical_qubits)

    # -------- évolution ------------------------------------------------
    from qutip import mesolve
    states = mesolve(H_td, psi0_full, times, [], []).states

    # -------- extraction des traces -----------------------------------
    amps_list, coords_list = _extract_qubit_traces(states, logical_qubits)

    # -------- affichage optionnel -------------------------------------
    if display:
        import matplotlib.pyplot as plt
        from qutip import Bloch
        
        from param_simu import delta_U_meV

        pulse1_mask = (times >= t0) & (times < t1)
        pulse2_mask = (times >= t2) & (times < t3)
        base_mask   = ~(pulse1_mask | pulse2_mask)

        n_q = len(coords_list)
        ncols = min(3, n_q)
        nrows = int(np.ceil(n_q / ncols))
        fig = plt.figure(figsize=(4 * ncols, 4 * nrows))

        for idx, coords in enumerate(coords_list):
            ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')
            b  = Bloch(axes=ax)

            if base_mask.any():
                b.add_points(coords[base_mask].T,  meth='s', colors=['r'])  # rouge : fond
            if pulse1_mask.any():
                b.add_points(coords[pulse1_mask].T, meth='s', colors=['g'])  # vert  : 1ʳᵉ imp.
            if pulse2_mask.any():
                b.add_points(coords[pulse2_mask].T, meth='s', colors=['b'])  # bleu  : 2ᵉ  imp.

            b.xlabel = ['$x$', ''];  b.ylabel = ['$y$', '']
            b.zlabel = ['$|1⟩$', '$|0⟩$']
            b.render()
            ax.set_title(f"Qubit {idx}" if idx == 0 else f"Detector {idx}")

        plt.tight_layout()

    return times * hbar_eVs, amps_list, coords_list 


def qubit_impurities(num_sites, n_electrons,
                     t, U,
                     t_final=1e-12, nbr_pts=None,
                     psi0=None, display=False, seed=0, freq_GHz=1):
    """
    Version simplifiée : Hamiltonien constant avec impureté éventuelle (site impair).
    Affiche la sphère de Bloch Plotly automatiquement si display=True.
    Ne dépend que de t et U (constants), pas de t_base, t_pulse, etc.
    
    freq_GHz : float, optionnel
        Fréquence de changement du spin impurité (GHz). Par défaut 1 GHz.
    """
    basis_occ = all_occupations(num_sites, n_electrons)
    from param_simu import LOGICAL_BASIS, ud_L, ud_R
    if LOGICAL_BASIS == "ud":
        logical_qubits = {
                "L": {"ud": ud_L["ud"], "du": ud_L["du"]},
                "R": {"ud": ud_R["ud"], "du": ud_R["du"]},
            }
    else:
        # fallback ST (comme aujourd'hui)
        logical_qubits  = _build_logical_qubits(num_sites, basis_occ)
    print(f"[info] logical basis = {LOGICAL_BASIS}")

    Nq = len(logical_qubits)

    # Impureté : spin seul aléatoire si nécessaire
    has_impurity = (num_sites % 2 == 1)
    if has_impurity and isinstance(psi0, (list, tuple)):
        if len(psi0) == (n_electrons // 2):
            rng = npr.default_rng(seed)
            z = rng.normal(size=2) + 1j * rng.normal(size=2)
            z /= np.linalg.norm(z)
            psi0 = list(psi0) + [tuple(z)]
    elif has_impurity and psi0 is None:
        rng = npr.default_rng(seed)
        z = rng.normal(size=2) + 1j * rng.normal(size=2)
        z /= np.linalg.norm(z)
        psi0 = [tuple(z)]

    # Hamiltonien constant
    H = build_spinful_hubbard_hamiltonian(num_sites, t, U, basis_occ)
    times = _auto_time_grid(H, t_final, nbr_pts)

    # État initial
    Nq, local_kets, psi0_full = _prepare_initial_state_qubits(
        num_sites, n_electrons, psi0, basis_occ, logical_qubits)
    for k, q in enumerate(logical_qubits):
        a0 = q["0"].overlap(psi0_full)
        b0 = q["1"].overlap(psi0_full)
        print(f"Qubit {k} : {_format_ab(a0, b0)}")

    # Évolution par blocs consécutifs (spin impurity flips)
    from qutip import mesolve
    rng = npr.default_rng(seed)
    # Convertir temps en ns pour gestion des flips
    times_ns = times * hbar_eVs * 1e9
    T_flip = 1 / freq_GHz  # période de flip en ns


    flip_edges = np.arange(times_ns[0], times_ns[-1] + T_flip, T_flip)
    flip_idxs = np.searchsorted(times_ns, flip_edges)
    psi_t = psi0_full
    all_states = []
    imp_traj = []
    for k in range(len(flip_idxs) - 1):
        i0, i1 = flip_idxs[k], flip_idxs[k+1]
        if i1 <= i0:
            continue  # rien à faire
        spin_val = rng.choice([-1, 1])
        imp_traj += [spin_val] * (i1 - i0)
        t_block = times[i0:i1]
        if len(t_block) == 0:
            continue
        states_block = mesolve(H, psi_t, t_block, [], []).states
        all_states.extend(states_block)
        psi_t = states_block[-1]
    # Assure alignement spin_imp_traj <-> times
    spin_imp_traj = np.array(imp_traj)
    if len(spin_imp_traj) < len(times):
        spin_imp_traj = np.append(spin_imp_traj, spin_imp_traj[-1])
    elif len(spin_imp_traj) > len(times):
        spin_imp_traj = spin_imp_traj[:len(times)]

    # NEW → amplitudes « probabilité » formelles pour le spin impurité
    spin_amps = {
        "0": (spin_imp_traj == +1).astype(float),   # ↑  → état logique |0⟩
        "1": (spin_imp_traj == -1).astype(float)    # ↓  → état logique |1⟩
    }

    # Idem pour les états
    states = all_states[:len(times)]

    # Extraction amplitudes & coords
    amps_list, coords_list = _extract_qubit_traces(states, logical_qubits)
    # Séparation impureté si besoin
    has_impurity = (num_sites % 2 == 1)
    if has_impurity:
        spin_amps = spin_imp_traj           # tableau ±1
        amps_list  = amps_list[:-1]         # on retire l’impureté
        coords_list_no_imp = coords_list[:-1]
    else:
        spin_amps = None
        coords_list_no_imp = coords_list
        

    # Affichage matplotlib sphères de Bloch (ST) + impureté si besoin
    if display:
        n_bloch = Nq - 1 if has_impurity else Nq
        ncols = min(3, n_bloch)
        nrows = int(np.ceil(n_bloch / ncols)) if n_bloch > 0 else 1
        fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
        for idx, coords in enumerate(coords_list_no_imp):
            ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')
            b = Bloch(axes=ax)
            b.add_points(coords.T, meth='s', colors=['r'])
            b.xlabel = ['$x$', '']
            b.ylabel = ['$y$', '']
            b.zlabel = ['$|1⟩$', '$|0⟩$']
            b.render()
            ax.set_title('Qubit' if idx == 0 else f'Qubit {idx}')
            # Si impureté, sur le dernier subplot, ajouter la trajectoire stochastique du spin impurité
            if has_impurity and idx == n_bloch - 1:
                fig_imp, ax_imp = plt.subplots(figsize=(6, 3))
                ax_imp.step(times * hbar_eVs * 1e9, spin_imp_traj, where='mid', label="Spin impureté (±1)")
                ax_imp.set_xlabel("Temps (ns)")
                ax_imp.set_ylabel("Spin impureté")
                ax_imp.set_title("Site impureté : spin stochastique")
                ax_imp.set_yticks([-1, 1])
                ax_imp.legend(loc='upper right')
                plt.tight_layout()
        plt.tight_layout()
        # Appel plot_bloch_plotly sur les qubits logiques (hors impureté)
        try:
            plot_bloch_plotly(coords_list_no_imp)
        except Exception as e:
            print("[WARN] plot_bloch_plotly a échoué :", e)

    return times * hbar_eVs, amps_list, coords_list, spin_amps,all_states[:len(times)]



# ---------------------------------------------------------------------
#  Affichage interactif Plotly des sphères de Bloch
# ---------------------------------------------------------------------
def plot_bloch_plotly(coords_list, titles=None):
    """
    Affiche une sphère de Bloch interactive Plotly par trajectoire
    (compatible avec les sorties coords_list de toutes les fonctions du fichier).

    Paramètres
    ----------
    coords_list : list of ndarray (N,3)
        Liste de trajectoires sur la sphère de Bloch (N points 3D).
    titles : list of str, optionnel
        Titre de chaque sphère.

    Usage :
        _, _, coords_list = top_hubbard_states_qubits(...)
        plot_bloch_plotly(coords_list)
    """
    import plotly.graph_objs as go
    import numpy as np

    n = len(coords_list)
    for k, coords in enumerate(coords_list):
        x, y, z = coords[:,0], coords[:,1], coords[:,2]
        # Sphère de Bloch
        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 40)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones(np.size(u)), np.cos(v))
        sphere = go.Surface(x=xs, y=ys, z=zs,
                            opacity=0.15, showscale=False,
                            colorscale='Greys', hoverinfo='skip')

        traj = go.Scatter3d(x=x, y=y, z=z,
                            mode='lines+markers',
                            line=dict(width=5, color='red'),
                            marker=dict(size=3, color='red'),
                            name='Trajectoire')

        layout = go.Layout(
            title=(titles[k] if titles else f'Qubit {k}'),
            scene=dict(
                xaxis=dict(title='X', range=[-1.05, 1.05]),
                yaxis=dict(title='Y', range=[-1.05, 1.05]),
                zaxis=dict(title='Z', range=[-1.05, 1.05]),
                aspectmode='cube',
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        fig = go.Figure(data=[sphere, traj], layout=layout)
        fig.show()


# ------------------------------------------------------------------
# Rabi par segment – compatible dict ou tableau ±1
# ------------------------------------------------------------------
def rabi_freqs_per_segment(times, amps_list, spin_amps, all_states,
                           min_angle=np.pi/10, fft_factor=1.5):
    import numpy as np

    # --- format du spin impureté (tableau ±1 ou dict) -------------------
    if isinstance(spin_amps, dict):
        P_up = np.abs(spin_amps["0"])**2
        P_dn = np.abs(spin_amps["1"])**2
    else:
        spin_amps = np.asarray(spin_amps)
        P_up = (spin_amps > 0).astype(float)
        P_dn = 1.0 - P_up

    # --- indices des sauts de spin --------------------------------------
    spin_state = np.argmax(np.vstack([P_up, P_dn]).T, axis=1)  # 0=↑, 1=↓
    idx_flip = np.r_[0, np.where(np.diff(spin_state) != 0)[0] + 1, len(times) - 1]

    # --- boucle sur les qubits ------------------------------------------
    freqs = []

    for amp in amps_list:
        fq_k = []

        for i0, i1 in zip(idx_flip[:-1], idx_flip[1:]):
            if i1 <= i0 + 1:
                fq_k.append(None)
                continue

            psi0 = all_states[i0]
            psi1 = all_states[i1 - 1]

            # Angle sphère de Bloch
            overlap = abs(psi0.overlap(psi1))
            overlap = np.clip(overlap, -1, 1)
            angle = 2 * np.arccos(overlap)
            Δt = (times[i1] - times[i0])# s

            if Δt <= 0 or angle < min_angle:

                fq_k.append(None)
                continue

            # Estimation de la période
            T_est = 2 * np.pi * Δt / angle

            # --- Cas long : FFT
            if Δt > fft_factor * T_est:
                seg = np.abs(amp["1"][i0:i1])**2
                if len(seg) < 4:
                    fq_k.append(None)
                    print("zut")

                    continue
                print("fft")

                seg = (seg - np.mean(seg)) * np.hanning(len(seg))
                dt = (times[1] - times[0]) * 1e-9
                spec = np.abs(np.fft.rfft(seg))
                freqs_fft = np.fft.rfftfreq(len(seg), d=dt)
                spec[0] = 0
                freq_est = freqs_fft[np.argmax(spec)]
                fq_k.append(freq_est)
            else:
                # --- Cas court : estimation angulaire
                freq = angle / (2 * np.pi * Δt)
                fq_k.append(freq)

        freqs.append(fq_k)

    return freqs