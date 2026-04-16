"""Optional downstream validation helpers for the Day 5 generative notebooks."""

from __future__ import annotations

import importlib.util
import itertools
import json
import site
from math import gcd
from pathlib import Path


def _find_smact_data_dir() -> Path | None:
    candidate_roots = []
    spec = importlib.util.find_spec("smact")
    if spec is not None and spec.submodule_search_locations:
        candidate_roots.extend(Path(path) for path in spec.submodule_search_locations)

    for root in site.getsitepackages() + [site.getusersitepackages()]:
        candidate_roots.append(Path(root) / "smact")

    for root in candidate_roots:
        data_dir = root / "data"
        if data_dir.exists():
            return data_dir.resolve()
    return None


def _parse_oxidation_state_text(path: Path) -> dict[str, list[int]]:
    oxidation_states = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        oxidation_states[parts[0]] = [int(value) for value in parts[1:]]
    return oxidation_states


def _load_oxidation_state_map(commonality: str = "medium", consensus: int = 3) -> dict[str, list[int]]:
    data_dir = _find_smact_data_dir()
    if data_dir is None:
        raise FileNotFoundError("Could not locate the installed SMACT data directory.")

    counts_path = data_dir / "oxidation_states_icsd24_counts.json"
    if counts_path.exists():
        threshold_map = {"low": 0.0, "medium": 10.0, "high": 50.0, "main": None}
        threshold = threshold_map.get(commonality)
        if commonality not in threshold_map:
            raise ValueError("commonality must be one of: 'low', 'medium', 'high', or 'main'.")

        rows = json.loads(counts_path.read_text(encoding="utf-8"))
        grouped = {}
        for row in rows:
            element = str(row["element"])
            oxidation_state = int(row["oxidation_state"])
            results_count = int(row["results_count"])
            if oxidation_state == 0 or results_count < int(consensus):
                continue
            grouped.setdefault(element, []).append((oxidation_state, results_count))

        oxidation_states = {}
        for element, states in grouped.items():
            total = sum(results_count for _, results_count in states)
            if total <= 0:
                continue
            if commonality == "main":
                max_count = max(results_count for _, results_count in states)
                selected = [oxidation_state for oxidation_state, results_count in states if results_count == max_count]
            else:
                selected = [
                    oxidation_state
                    for oxidation_state, results_count in states
                    if (100.0 * results_count / total) >= float(threshold)
                ]
            oxidation_states[element] = sorted(set(selected))
        return oxidation_states

    fallback_map = {
        "low": data_dir / "oxidation_states_icsd24_filtered.txt",
        "medium": data_dir / "oxidation_states_icsd24_common.txt",
        "high": data_dir / "oxidation_states_icsd24_common.txt",
        "main": data_dir / "oxidation_states_icsd24_common.txt",
    }
    fallback_path = fallback_map.get(commonality)
    if fallback_path is None or not fallback_path.exists():
        raise FileNotFoundError("Could not locate a suitable oxidation-state data file for screening.")
    return _parse_oxidation_state_text(fallback_path)


def _has_charge_neutral_assignment(amounts: list[int], oxidation_state_choices: list[list[int]]) -> bool:
    for oxidation_states in itertools.product(*oxidation_state_choices):
        total_charge = sum(amount * oxidation_state for amount, oxidation_state in zip(amounts, oxidation_states, strict=True))
        if total_charge == 0 and any(value > 0 for value in oxidation_states) and any(value < 0 for value in oxidation_states):
            return True
    return False


def _reduce_formula_counts(counts: list[int]) -> list[int]:
    divisor = 0
    for value in counts:
        divisor = gcd(divisor, int(value))
    divisor = max(divisor, 1)
    return [int(value) // divisor for value in counts]


def _format_formula(elements: list[str], counts: list[int]) -> str:
    formula_parts = []
    for element, count in zip(elements, counts, strict=True):
        if count <= 0:
            continue
        formula_parts.append(element if count == 1 else f"{element}{count}")
    return "".join(formula_parts)


def get_valid_compositions(
    elements: list[str],
    max_stoichiometry: int = 5,
    commonality: str = "medium",
):
    """Screen simple compositions using oxidation-state balance.

    This is intentionally notebook-friendly: it relies on the oxidation-state
    tables shipped with SMACT but avoids importing heavyweight stacks such as
    `pymatgen` during basic shortlist generation.
    """
    elements = [str(element).strip() for element in elements if str(element).strip()]
    if not elements:
        raise ValueError("Provide at least one element symbol.")

    oxidation_state_map = _load_oxidation_state_map(commonality=commonality, consensus=3)
    missing = [element for element in elements if not oxidation_state_map.get(element)]
    if missing:
        raise ValueError(
            "Could not find oxidation-state data for: " + ", ".join(missing)
        )

    valid_rows = []
    seen_formulas = set()
    for amounts in itertools.product(range(1, int(max_stoichiometry) + 1), repeat=len(elements)):
        oxidation_state_choices = [oxidation_state_map[element] for element in elements]
        if not _has_charge_neutral_assignment(list(amounts), oxidation_state_choices):
            continue
        reduced_amounts = _reduce_formula_counts(list(amounts))
        formula = _format_formula(elements, reduced_amounts)
        if formula in seen_formulas:
            continue
        seen_formulas.add(formula)
        valid_rows.append(
            {
                "formula": formula,
                "elements": list(elements),
                "counts": reduced_amounts,
                "num_elements": len(elements),
                "num_atoms_in_reduced_formula": int(sum(reduced_amounts)),
            }
        )

    valid_rows.sort(key=lambda row: (row["num_atoms_in_reduced_formula"], row["formula"]))
    return valid_rows


def relax_atoms_with_mace(
    atoms_list,
    *,
    device: str = "cpu",
    model_size: str = "small",
):
    """Relax ASE atoms with a MACE foundation model through TorchSim."""
    import torch_sim as ts
    from mace.calculators.foundations_models import mace_mp
    from torch_sim.models.mace import MaceModel

    atoms_list = list(atoms_list)
    if not atoms_list:
        raise ValueError("Need at least one structure to relax.")

    mace = mace_mp(model=model_size, return_raw_model=True)
    mace_model = MaceModel(model=mace, device=device)
    relaxed_state = ts.optimize(
        system=atoms_list,
        model=mace_model,
        optimizer=ts.optimizers.frechet_cell_fire,
    )
    relaxed_atoms = relaxed_state.to_atoms()
    energies = [float(x) for x in relaxed_state.energy]
    return {
        "relaxed_state": relaxed_state,
        "relaxed_atoms": relaxed_atoms,
        "energies": energies,
        "device": device,
        "model_size": model_size,
    }


def build_relaxation_rows(before_atoms_list, after_atoms_list, energies, labels=None):
    before_atoms_list = list(before_atoms_list)
    after_atoms_list = list(after_atoms_list)
    energies = list(energies)
    labels = list(labels) if labels is not None else [f"candidate_{idx}" for idx in range(len(before_atoms_list))]

    rows = []
    for idx, (label, before, after, energy) in enumerate(
        zip(labels, before_atoms_list, after_atoms_list, energies, strict=True)
    ):
        before_volume = float(before.get_volume())
        after_volume = float(after.get_volume())
        rows.append(
            {
                "sample_id": idx,
                "label": str(label),
                "formula_before": before.get_chemical_formula(),
                "formula_after": after.get_chemical_formula(),
                "n_sites": len(after),
                "energy_eV": float(energy),
                "volume_before": before_volume,
                "volume_after": after_volume,
                "delta_volume": after_volume - before_volume,
            }
        )
    return rows


def show_before_after_relaxation(
    before_atoms_list,
    after_atoms_list,
    labels,
    title: str,
    *,
    save_path: Path | None = None,
):
    import matplotlib.pyplot as plt
    from ase.visualize.plot import plot_atoms

    before_atoms_list = list(before_atoms_list)
    after_atoms_list = list(after_atoms_list)
    labels = list(labels)
    if not before_atoms_list:
        raise ValueError("Need at least one structure to visualize.")

    fig, axes = plt.subplots(
        len(before_atoms_list),
        2,
        figsize=(10, 4 * len(before_atoms_list)),
        squeeze=False,
        facecolor="white",
    )
    for row_idx, (label, before, after) in enumerate(zip(labels, before_atoms_list, after_atoms_list, strict=True)):
        for ax, atoms, subtitle in [
            (axes[row_idx, 0], before, "Before relaxation"),
            (axes[row_idx, 1], after, "After relaxation"),
        ]:
            plot_atoms(atoms, ax=ax, rotation="20x,30y,0z", radii=0.35, show_unit_cell=2)
            ax.set_axis_off()
            ax.set_title(f"{label}\n{subtitle}", fontsize=10)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.show()
    return fig


def fetch_mp_entries_for_chemsys(
    elements,
    api_key: str,
    *,
    thermo_types=("GGA_GGA+U",),
):
    from mp_api.client import MPRester
    try:
        from pymatgen.analysis.phase_diagram import PDEntry
    except Exception as exc:
        raise ImportError(
            "Phase-diagram analysis requires a working pymatgen installation. "
            "In this environment the optional pymatgen stack may be unavailable "
            "(for example because the Python build is missing `_bz2`)."
        ) from exc

    with MPRester(api_key=api_key) as mpr:
        raw_entries = mpr.get_entries_in_chemsys(
            elements=list(elements),
            additional_criteria={"thermo_types": list(thermo_types)},
        )
    return [PDEntry(entry.composition, entry.uncorrected_energy) for entry in raw_entries]


def build_generated_pd_entries(atoms_list, energies):
    try:
        from pymatgen.analysis.phase_diagram import PDEntry
        from pymatgen.core import Composition
    except Exception as exc:
        raise ImportError(
            "Building phase-diagram entries requires a working pymatgen installation. "
            "In this environment the optional pymatgen stack may be unavailable "
            "(for example because the Python build is missing `_bz2`)."
        ) from exc

    entries = []
    for atoms, energy in zip(atoms_list, energies, strict=True):
        entries.append(
            PDEntry(
                composition=Composition(atoms.get_chemical_formula()),
                energy=float(energy),
            )
        )
    return entries


def summarize_generated_stability(
    mp_entries,
    generated_entries,
    *,
    labels=None,
    stable_cutoff: float = 0.10,
):
    try:
        from pymatgen.analysis.phase_diagram import PhaseDiagram
    except Exception as exc:
        raise ImportError(
            "Phase-diagram stability analysis requires a working pymatgen installation. "
            "In this environment the optional pymatgen stack may be unavailable "
            "(for example because the Python build is missing `_bz2`)."
        ) from exc

    labels = list(labels) if labels is not None else [f"candidate_{idx}" for idx in range(len(generated_entries))]
    phase_diagram = PhaseDiagram(list(mp_entries))
    rows = []
    for label, entry in zip(labels, generated_entries, strict=True):
        e_hull = float(phase_diagram.get_e_above_hull(entry, allow_negative=True))
        rows.append(
            {
                "label": str(label),
                "formula": entry.composition.reduced_formula,
                "energy_above_hull_eV_per_atom": e_hull,
                "is_stable_or_close": e_hull < stable_cutoff,
            }
        )
    rows.sort(key=lambda row: row["energy_above_hull_eV_per_atom"])
    return {"rows": rows, "phase_diagram": phase_diagram}


def plot_phase_diagram_with_generated(mp_entries, generated_entries):
    try:
        from pymatgen.analysis.phase_diagram import PDPlotter, PhaseDiagram
    except Exception as exc:
        raise ImportError(
            "Plotting the phase diagram requires a working pymatgen installation. "
            "In this environment the optional pymatgen stack may be unavailable "
            "(for example because the Python build is missing `_bz2`)."
        ) from exc

    phase_diagram = PhaseDiagram(list(mp_entries) + list(generated_entries))
    plotter = PDPlotter(phase_diagram)
    return plotter.get_plot()


__all__ = [
    "build_generated_pd_entries",
    "build_relaxation_rows",
    "fetch_mp_entries_for_chemsys",
    "get_valid_compositions",
    "plot_phase_diagram_with_generated",
    "relax_atoms_with_mace",
    "show_before_after_relaxation",
    "summarize_generated_stability",
]
