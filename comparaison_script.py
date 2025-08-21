#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compare_qubit_detector_heatmap.py

But: Prendre 2 dossiers de données (QUANTUM QUBIT GAUCHE et DETECTOR / qubit droit)
     et générer une heatmap "rouge" où l'intensité est forte quand:
       - le détecteur est TRÈS déphasé (faible "same-phase prob"), ET
       - le qubit gauche est PEU déphasé (forte "same-phase prob").
     => score = (1 - p_same_detector) * (p_same_qubit)

Par défaut, le script cherche les cartes "overlap" écrites par tes scripts:
  - QUBIT:   p_qubit_overlap_row_###.npy
  - DETECTOR: p_detector_overlap_row_###.npy

Il récupère l'axe Δt depuis les fichiers baseline npz présents dans chaque dossier,
et l'axe ΔU depuis param_simu.delta_U_vals_full si disponible, sinon utilise des indices.

Exemples d'utilisation:
-----------------------
python compare_qubit_detector_heatmap.py \
  --qubit-dir "qubit_results/<config>__psi0_<NU>x<NT>" \
  --detector-dir "detector_results/<config>__psi0_<NU>x<NT>" \
  --out "comparison_heatmap.png"

Options utiles:
---------------
--metric overlap|nochange|auto   (défaut: auto → préfère 'overlap' si présent)
--normalize  (normalise le score sur [0,1] en le divisant par son max>0)
--save-npy   (sauvegarde également le tableau 'score' en .npy côté --out)
--dpi 200    (résolution de la figure)
"""
from __future__ import annotations

import os
import re
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional, Tuple, List

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def _find_rows(dir_path: str, prefix: str) -> List[str]:
    """
    Retourne la liste triée des chemins de fichiers p_<...>_row_###.npy présents,
    triés par index ### (zéro-fill). Ne parcourt pas récursivement.
    """
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Dossier introuvable: {dir_path}")
    pat = re.compile(rf"^{re.escape(prefix)}_(\d{{3}})\.npy$")
    files = []
    for name in os.listdir(dir_path):
        m = pat.match(name)
        if m:
            files.append((int(m.group(1)), os.path.join(dir_path, name)))
    files.sort(key=lambda t: t[0])
    return [p for _, p in files]

def _load_rows_as_map(dir_path: str, prefix: str) -> np.ndarray:
    """
    Charge toutes les lignes p_<...>_row_###.npy en un tableau 2D (nU, nT).
    Les lignes peuvent être memmap; on les lit en ndarray standard.
    """
    row_files = _find_rows(dir_path, prefix)
    if not row_files:
        raise FileNotFoundError(f"Aucun fichier '{prefix}_###.npy' trouvé dans: {dir_path}")

    rows = []
    nT_ref = None
    for f in row_files:
        arr = np.load(f, mmap_mode='r')
        arr = np.asarray(arr)  # copie si memmap
        if nT_ref is None:
            nT_ref = arr.shape[0]
        else:
            # Harmonise en tronquant/paddinant NaN si tailles différentes
            if arr.shape[0] < nT_ref:
                tmp = np.full(nT_ref, np.nan, dtype=arr.dtype)
                tmp[:arr.shape[0]] = arr
                arr = tmp
            elif arr.shape[0] > nT_ref:
                arr = arr[:nT_ref]
        rows.append(arr.astype(np.float32))

    return np.vstack(rows)

def _load_delta_t(dir_path: str) -> Optional[np.ndarray]:
    """
    Essaie de récupérer delta_t_vals depuis un des baseline *.npz connus
    dans le même dossier: ...baseline_spinor.npz (prioritaire) puis ...baseline_phi0.npz.
    """
    candidates = [
        os.path.join(dir_path, "qubit_left_baseline_spinor.npz"),
        os.path.join(dir_path, "detector_right_baseline_spinor.npz"),
        os.path.join(dir_path, "qubit_left_baseline_phi0.npz"),
        os.path.join(dir_path, "detector_baseline_phi0.npz"),
    ]
    for f in candidates:
        if os.path.exists(f):
            try:
                data = np.load(f)
                if "delta_t_vals" in data:
                    return data["delta_t_vals"]
            except Exception:
                pass
    return None

def _load_delta_u_from_param_simu() -> Optional[np.ndarray]:
    """
    Tente d'importer param_simu.delta_U_vals_full.
    """
    try:
        import importlib
        ps = importlib.import_module("param_simu")
        if hasattr(ps, "delta_U_vals_full"):
            return np.asarray(getattr(ps, "delta_U_vals_full"), dtype=float)
    except Exception:
        return None
    return None

def _extent_for_axes(delta_t_vals: np.ndarray, delta_u_vals: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Calcul l'extent pour imshow: [xmin, xmax, ymin, ymax]
      - x: Δt en ns
      - y: ΔU en meV (valeurs directes)
    """
    return float(delta_t_vals[0])*1e9, float(delta_t_vals[-1])*1e9, float(delta_u_vals[0]), float(delta_u_vals[-1])

# -----------------------------------------------------------------------------
# Core
# -----------------------------------------------------------------------------
def make_comparison_heatmap(
    qubit_dir: str,
    detector_dir: str,
    metric: str = "auto",
    normalize: bool = False,
    dpi: int = 200,
    out_path: Optional[str] = None,
    save_npy: bool = False,
) -> str:
    """
    Construit la heatmap de score = (1 - p_detector) * p_qubit.

    - qubit_dir: dossier contenant p_qubit_overlap_row_###.npy (ou p_nochange_row_###.npy)
    - detector_dir: dossier contenant p_detector_overlap_row_###.npy (ou fidelity_detector_row_###.npy)

    - metric: "overlap" (|<q0|q>|^2), "nochange" (cos^2(Δφ/2)) ou "auto"
    - normalize: si True, divise le score par son max>0 pour l'étirer sur [0,1]
    - dpi: résolution de la figure
    - out_path: chemin de sauvegarde .png (si None, construit auto)
    - save_npy: enregistre aussi le tableau score en .npy (à côté du png)

    Retourne le chemin du fichier image créé.
    """
    # 1) Choix des préfixes en fonction du metric
    metric = metric.lower().strip()
    if metric not in ("overlap", "nochange", "auto"):
        raise ValueError("--metric doit être 'overlap', 'nochange' ou 'auto'")

    prefixes = {
        "overlap": ("p_qubit_overlap_row", "p_detector_overlap_row"),
        "nochange": ("p_nochange_row", "fidelity_detector_row"),
    }

    if metric == "auto":
        # Préfère "overlap" si présent côté qubit ET detector; sinon bascule sur "nochange"
        try:
            if _find_rows(qubit_dir, prefixes["overlap"][0]) and _find_rows(detector_dir, prefixes["overlap"][1]):
                metric = "overlap"
            else:
                metric = "nochange"
        except Exception:
            metric = "nochange"

    q_prefix, d_prefix = prefixes[metric]
    # 2) Charge les cartes (nU, nT)
    q_map = _load_rows_as_map(qubit_dir, q_prefix)     # p_same_qubit
    d_map = _load_rows_as_map(detector_dir, d_prefix)  # p_same_detector

    # Harmonisation dimensionnelle si nécessaire
    nU = min(q_map.shape[0], d_map.shape[0])
    nT = min(q_map.shape[1], d_map.shape[1])
    if (q_map.shape[0], q_map.shape[1]) != (nU, nT):
        q_map = q_map[:nU, :nT]
    if (d_map.shape[0], d_map.shape[1]) != (nU, nT):
        d_map = d_map[:nU, :nT]

    # 3) Score rouge: fort si détecteur déphase (1 - pD) ET qubit stable (pQ élevé)
    p_qubit    = q_map
    p_detector = d_map
    score = (1.0 - p_detector) * p_qubit  # ∈ [0,1] si entrées ∈ [0,1]

    # 4) Axes Δt, ΔU
    # Δt: on essaie de le récupérer des baselines. Priorité au dossier QUANTUM, sinon DETECTOR.
    delta_t_vals = _load_delta_t(qubit_dir)
    if delta_t_vals is None:
        delta_t_vals = _load_delta_t(detector_dir)
    if delta_t_vals is None:
        # fallback: indices [0..nT-1] (en s)
        delta_t_vals = np.arange(nT, dtype=float)

    # ΔU: on essaie depuis param_simu (plus fiable). Sinon on prend des indices [0..nU-1].
    delta_u_vals = _load_delta_u_from_param_simu()
    if delta_u_vals is None or len(delta_u_vals) != nU:
        delta_u_vals = np.arange(nU, dtype=float)

    # 5) Normalisation optionnelle
    if normalize:
        finite_max = np.nanmax(score)
        if np.isfinite(finite_max) and finite_max > 0:
            score = score / finite_max

    # 6) Plot
    fig = plt.figure(figsize=(8.4, 6.2))
    ax = plt.gca()
    extent = _extent_for_axes(delta_t_vals, delta_u_vals)
    im = ax.imshow(score, origin='lower', aspect='auto', extent=extent,
                   cmap='Reds', interpolation='nearest', vmin=0.0, vmax=1.0)
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("Hot spot = détecteur déphasé & qubit stable\nscore = (1 − p$_{det}$) × p$_{qubit}$")

    ax.set_xlabel("Δt (ns)" if delta_t_vals is not None else "Δt index")
    ax.set_ylabel("ΔU (meV)" if delta_u_vals is not None else "ΔU index")

    # Titre compact avec infos utiles
    tag_q = os.path.basename(os.path.normpath(qubit_dir))
    tag_d = os.path.basename(os.path.normpath(detector_dir))
    ax.set_title(f"Comparaison déphasage (rouge) — metric={metric}\nqubit: {tag_q}   |   detector: {tag_d}")

    plt.tight_layout()

    # 7) Sauvegarde
    if out_path is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = os.path.join(os.path.commonpath([qubit_dir, detector_dir]) if os.path.commonprefix([qubit_dir, detector_dir]) else ".", "comparison_results")
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception:
            out_dir = "."
        out_path = os.path.join(out_dir, f"heatmap_comparison_{metric}_{nU}x_{nT}_{stamp}.png")

    try:
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    except Exception as e:
        print(f"⚠️ Échec de la sauvegarde de la figure ({e}), tentative dans le dossier courant…")
        out_path = os.path.basename(out_path)
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")

    if save_npy:
        base, _ = os.path.splitext(out_path)
        np.save(base + ".npy", score.astype(np.float32))

    print(f"🖼️ Image créée: {out_path}")
    return out_path

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Comparer qubit & détecteur en une heatmap rouge.")
    p.add_argument("--qubit-dir", required=True, help="Dossier des lignes du QUBIT (ex: .../qubit_results/<config>__psi0_... )")
    p.add_argument("--detector-dir", required=True, help="Dossier des lignes du DETECTOR (ex: .../detector_results/<config>__psi0_... )")
    p.add_argument("--metric", default="auto", choices=["auto", "overlap", "nochange"], help="Type de probas à utiliser (défaut: auto)")
    p.add_argument("--normalize", action="store_true", help="Normalise le score par son max pour étirer l'échelle sur [0,1]")
    p.add_argument("--out", default=None, help="Chemin .png de sortie (défaut: auto)")
    p.add_argument("--dpi", type=int, default=200, help="DPI de la figure (défaut: 200)")
    p.add_argument("--save-npy", action="store_true", help="Sauvegarde aussi le tableau numpy du score")
    return p

def main(argv: Optional[list] = None) -> int:
    ap = _build_argparser()
    args = ap.parse_args(argv)

    try:
        out_path = make_comparison_heatmap(
            qubit_dir=args.qubit_dir,
            detector_dir=args.detector_dir,
            metric=args.metric,
            normalize=args.normalize,
            dpi=args.dpi,
            out_path=args.out,
            save_npy=args.save_npy,
        )
        print(out_path)
    except Exception as e:
        print(f"Erreur: {e}", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

