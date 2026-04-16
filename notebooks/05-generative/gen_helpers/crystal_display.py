"""Structure display helpers for the crystal diffusion notebook."""

from __future__ import annotations

import html
import io
import math
from pathlib import Path

import matplotlib.pyplot as plt
from IPython.display import HTML, Image as IPythonImage
from ase.visualize.plot import plot_atoms
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter

try:
    import py3Dmol  # type: ignore
except Exception:
    py3Dmol = None


_ASE_ADAPTOR = AseAtomsAdaptor()


def structure_to_ase_atoms(structure: Structure):
    return _ASE_ADAPTOR.get_atoms(structure)


def structure_to_cif_string(structure: Structure) -> str:
    return str(CifWriter(structure))


def show_structures_ase(structures, labels=None, columns=2, rotation="20x,30y,0z", dpi=180):
    labels = labels or [f"structure {i}" for i in range(len(structures))]
    if len(structures) == 0:
        raise ValueError("Need at least one structure to preview.")

    columns = max(1, min(int(columns), len(structures)))
    rows = int(math.ceil(len(structures) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.5 * columns, 3.8 * rows),
        squeeze=False,
        facecolor="white",
    )
    axes = axes.ravel()

    for ax in axes[len(structures) :]:
        ax.set_axis_off()

    for ax, structure, label in zip(axes, structures, labels):
        ax.set_facecolor("white")
        try:
            atoms = structure_to_ase_atoms(structure)
            plot_atoms(atoms, ax, rotation=rotation, radii=0.35, show_unit_cell=2)
        except Exception as exc:
            ax.text(
                0.5,
                0.5,
                f"Preview failed\n{type(exc).__name__}: {exc}",
                ha="center",
                va="center",
                fontsize=9,
                wrap=True,
            )
        ax.set_title(str(label), fontsize=9, wrap=True, pad=8)
        ax.set_axis_off()

    plt.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return IPythonImage(data=buffer.getvalue(), format="png")


def show_structures_py3dmol(structures, labels=None, columns=2, width=360, height=300, py3dmol_module=None):
    viewer_module = py3Dmol if py3dmol_module is None else py3dmol_module
    if viewer_module is None:
        raise RuntimeError("py3Dmol is not installed in this Python session.")

    labels = labels or [f"structure {i}" for i in range(len(structures))]
    if len(structures) == 0:
        raise ValueError("Need at least one structure to preview.")

    cards = []
    for structure, label in zip(structures, labels):
        viewer = viewer_module.view(width=int(width), height=int(height))
        viewer.addModel(structure_to_cif_string(structure), "cif")
        viewer.setStyle(
            {
                "sphere": {"scale": 0.32, "colorscheme": "Jmol"},
                "stick": {"radius": 0.14, "colorscheme": "Jmol"},
            }
        )
        viewer.addUnitCell()
        viewer.setBackgroundColor("white")
        viewer.zoomTo()
        cards.append(
            '<div style="border:1px solid #ddd;border-radius:10px;padding:8px;background:white;">'
            f'<div style="font-size:13px;font-weight:600;margin-bottom:6px;">{html.escape(str(label))}</div>'
            f"{viewer._make_html()}"
            "</div>"
        )

    columns = max(1, min(int(columns), len(structures)))
    return HTML(
        '<div style="display:grid;'
        f"grid-template-columns:repeat({columns}, minmax({int(width)}px, 1fr));"
        'gap:14px;align-items:start;">'
        + "".join(cards)
        + "</div>"
    )


def show_structures(
    structures,
    labels=None,
    columns=2,
    rotation="20x,30y,0z",
    dpi=180,
    viewer_mode="static",
    py3dmol_width=360,
    py3dmol_height=300,
    py3dmol_module=None,
):
    if viewer_mode == "py3Dmol":
        viewer_module = py3Dmol if py3dmol_module is None else py3dmol_module
        if viewer_module is None:
            print("py3Dmol is unavailable in this session, so the notebook is falling back to static ASE previews.")
        else:
            try:
                return show_structures_py3dmol(
                    structures,
                    labels=labels,
                    columns=columns,
                    width=py3dmol_width,
                    height=py3dmol_height,
                    py3dmol_module=viewer_module,
                )
            except Exception as exc:
                print(f"Interactive viewer failed ({type(exc).__name__}: {exc}); falling back to static ASE previews.")

    return show_structures_ase(
        structures,
        labels=labels,
        columns=columns,
        rotation=rotation,
        dpi=dpi,
    )


def save_structures_as_cifs(structures, out_dir: str | Path):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for i, structure in enumerate(structures):
        file_path = out_path / f"sample_{i:03d}.cif"
        CifWriter(structure).write_file(str(file_path))
    return out_path


__all__ = [
    "save_structures_as_cifs",
    "show_structures",
    "show_structures_ase",
    "show_structures_py3dmol",
    "structure_to_ase_atoms",
    "structure_to_cif_string",
]

