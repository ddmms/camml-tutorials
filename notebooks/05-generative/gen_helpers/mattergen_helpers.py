"""Helper utilities for the MatterGen teaching notebook."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections import Counter
from html import escape
from pathlib import Path
from statistics import mean

from IPython.display import HTML, display


TUTORIAL_REPO = "tutorials"
TUTORIAL_REMOTE = "https://gitlab.com/cam-ml/tutorials.git"
TUTORIAL_COLAB_DIR_CANDIDATES = (
    Path("/content") / "tutorials",
    Path("/content") / "cam_ml_tutorials",
    Path("/content") / "camml-tutorials",
)
NOTEBOOK_FILENAME = "mattergen-crystals.ipynb"
MATTERGEN_REMOTE = "https://github.com/microsoft/mattergen.git"
MATTERGEN_COMMIT = "a245cf2b7538eea6d873e6430b0e30c56d26c60e"

def find_notebook_root(notebook_filename: str = NOTEBOOK_FILENAME) -> Path:
    module_candidate = Path(__file__).resolve().parent.parent
    if (module_candidate / notebook_filename).exists() and (module_candidate / "README.md").exists():
        return module_candidate.resolve()

    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / notebook_filename).exists() and (candidate / "README.md").exists():
            return candidate.resolve()

    for repo_dir in TUTORIAL_COLAB_DIR_CANDIDATES:
        content_candidate = repo_dir / "notebooks" / "05-generative"
        if (content_candidate / notebook_filename).exists():
            return content_candidate.resolve()

    raise FileNotFoundError("Could not locate the notebooks/05-generative directory for this tutorial repo.")


def ensure_colab_tutorial_repo() -> Path:
    for repo_dir in TUTORIAL_COLAB_DIR_CANDIDATES:
        notebook_root = repo_dir / "notebooks" / "05-generative"
        if notebook_root.exists():
            return repo_dir.resolve()

    for repo_dir in TUTORIAL_COLAB_DIR_CANDIDATES:
        if repo_dir.exists():
            continue
        print(f"Cloning {TUTORIAL_REMOTE} into {repo_dir}...")
        subprocess.run(["git", "clone", "--depth", "1", "--branch", "main", TUTORIAL_REMOTE, str(repo_dir)], check=True)
        return repo_dir.resolve()

    raise FileNotFoundError(
        "Could not find or clone the Day 5 tutorial repo in /content for this Colab session."
    )


def repo_has_commit(path: Path, commit: str) -> bool:
    return (
        subprocess.run(
            ["git", "-C", str(path), "cat-file", "-e", f"{commit}^{{commit}}"],
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )


def setup_mattergen_environment(
    notebook_filename: str = NOTEBOOK_FILENAME,
    mattergen_commit: str = MATTERGEN_COMMIT,
):
    if "google.colab" in sys.modules:
        ensure_colab_tutorial_repo()

    notebook_root = find_notebook_root(notebook_filename)
    os.chdir(notebook_root)
    print("Notebook root:", notebook_root)

    repo_dir = notebook_root / "mattergen_repo"
    venv_dir = repo_dir / ".venv"
    cache_dir = Path("/tmp/uv-cache")
    python_dir = Path("/tmp/uv-python")
    cache_dir.mkdir(parents=True, exist_ok=True)
    python_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["UV_CACHE_DIR"] = str(cache_dir)
    env["UV_PYTHON_INSTALL_DIR"] = str(python_dir)

    uv_bin = shutil.which("uv")
    if uv_bin is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)
        uv_bin = shutil.which("uv") or str(Path.home() / ".local/bin/uv")

    if not repo_dir.exists():
        subprocess.run(["git", "clone", MATTERGEN_REMOTE, str(repo_dir)], check=True)

    if not repo_has_commit(repo_dir, mattergen_commit):
        subprocess.run(["git", "-C", str(repo_dir), "fetch", "--depth", "1", "origin", mattergen_commit], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "switch", "--detach", mattergen_commit], check=True)

    if not venv_dir.exists():
        subprocess.run([uv_bin, "venv", str(venv_dir), "--python", "3.10"], check=True, env=env)

    print("Pinned MatterGen commit:", mattergen_commit)
    print("Installing MatterGen into:", venv_dir.resolve())
    subprocess.run(
        [
            uv_bin,
            "pip",
            "install",
            "--python",
            str(venv_dir / "bin" / "python"),
            "-e",
            str(repo_dir),
        ],
        check=True,
        env=env,
    )

    subprocess.run(
        [str(venv_dir / "bin" / "python"), "-c", "import mattergen; print(mattergen.__file__)"],
        check=True,
        env=env,
    )

    return {
        "notebook_root": notebook_root,
        "repo_dir": repo_dir.resolve(),
        "venv_dir": venv_dir.resolve(),
        "mattergen_bin": (venv_dir / "bin" / "mattergen-generate").resolve(),
        "pinned_commit": mattergen_commit,
        "uv_bin": uv_bin,
        "env": env,
    }


def run_mattergen_generation(
    mattergen_env,
    output_dir,
    *,
    pretrained_name,
    batch_size,
    num_batches,
    record_trajectories=False,
    properties=None,
    guidance=None,
    force=False,
):
    from ase.io import read as ase_read

    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = mattergen_env["notebook_root"] / output_dir
    output_dir = output_dir.resolve()
    extxyz_path = output_dir / "generated_crystals.extxyz"
    expected_samples = batch_size * num_batches
    if extxyz_path.exists() and not force:
        try:
            existing_samples = len(ase_read(extxyz_path, index=":"))
        except Exception:
            existing_samples = None
        if existing_samples == expected_samples:
            print(f"Reusing existing MatterGen outputs in {output_dir} ({existing_samples} samples)")
            return output_dir
        print(f"Found {existing_samples} existing samples in {output_dir}; regenerating for {expected_samples} samples.")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(mattergen_env["mattergen_bin"]),
        str(output_dir),
        f"--pretrained-name={pretrained_name}",
        f"--batch_size={batch_size}",
        f"--num_batches={num_batches}",
        f"--record_trajectories={record_trajectories}",
    ]
    if properties is not None:
        cmd.append(f"--properties_to_condition_on={properties}")
    if guidance is not None:
        cmd.append(f"--diffusion_guidance_factor={guidance}")

    env = os.environ.copy()
    env["MPLCONFIGDIR"] = env.get("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    return output_dir


def load_extxyz_frames(run_dir: Path):
    from ase.io import read

    extxyz_path = run_dir / "generated_crystals.extxyz"
    if not extxyz_path.exists():
        raise FileNotFoundError(f"Could not find MatterGen outputs in {run_dir}")
    return extxyz_path, read(extxyz_path, index=":")


def inspect_generated_cif_archive(run_dir: Path, preview_index: int = 0, preview_lines: int = 30):
    import zipfile

    archive_path = Path(run_dir) / "generated_crystals_cif.zip"
    if not archive_path.exists():
        raise FileNotFoundError(f"Could not find MatterGen CIF archive in {run_dir}")

    with zipfile.ZipFile(archive_path) as zf:
        cif_names = [name for name in zf.namelist() if name.lower().endswith(".cif")]
        if not cif_names:
            raise FileNotFoundError(f"Archive {archive_path.name} did not contain any CIF files.")
        preview_index = max(0, min(int(preview_index), len(cif_names) - 1))
        preview_name = cif_names[preview_index]
        preview_text = zf.read(preview_name).decode("utf-8", errors="replace")

    return {
        "archive_path": archive_path,
        "cif_names": cif_names,
        "preview_name": preview_name,
        "preview_text": preview_text,
        "preview_excerpt": "\n".join(preview_text.splitlines()[:preview_lines]),
    }


def mattergen_row(label, atoms, local_index, mattergen_run_info):
    volume = float(atoms.get_volume())
    mass = float(atoms.get_masses().sum())
    chemical_symbols = atoms.get_chemical_symbols()
    return {
        "label": label,
        "display_name": mattergen_run_info[label]["display_name"],
        "target_label": mattergen_run_info[label]["target_label"],
        "condition_kind": mattergen_run_info[label]["condition_kind"],
        "local_index": local_index,
        "formula": atoms.get_chemical_formula(),
        "elements": "-".join(sorted(set(chemical_symbols))),
        "n_sites": len(atoms),
        "volume": volume,
        "density": mass / max(volume, 1e-12),
    }


def collect_mattergen_runs(mattergen_run_info):
    mattergen_atoms = {}
    mattergen_rows = []
    summary_rows = []

    for label, info in mattergen_run_info.items():
        extxyz_path, atoms_list = load_extxyz_frames(Path(info["output_dir"]))
        mattergen_atoms[label] = atoms_list
        run_rows = [mattergen_row(label, atoms, i, mattergen_run_info) for i, atoms in enumerate(atoms_list)]
        mattergen_rows.extend(run_rows)

        formula_counter = Counter(row["formula"] for row in run_rows)
        summary_rows.append(
            {
                "run": info["display_name"],
                "target": info["target_label"],
                "samples": len(run_rows),
                "mean_n_sites": f"{mean(row['n_sites'] for row in run_rows):.2f}",
                "mean_volume": f"{mean(row['volume'] for row in run_rows):.2f}",
                "unique_formulas": len(formula_counter),
                "top_formula": formula_counter.most_common(1)[0][0],
            }
        )
        print(f"Loaded {len(run_rows)} structures for {info['display_name']} from {extxyz_path}")

    return mattergen_atoms, mattergen_rows, summary_rows


def render_summary_table(rows):
    headers = ["run", "target", "samples", "mean_n_sites", "mean_volume", "unique_formulas", "top_formula"]
    th_style = "border:1px solid #ddd; padding:4px 8px; text-align:left; background:#eff6ff;"
    td_style = "border:1px solid #ddd; padding:4px 8px; text-align:left;"
    body = []
    for row in rows:
        body.append(
            "<tr>" + "".join(f'<td style="{td_style}">{escape(str(row[h]))}</td>' for h in headers) + "</tr>"
        )
    html = f"""
    <table style="border-collapse:collapse; min-width:760px; margin:0.75rem 0 1rem 0;">
      <thead><tr>{''.join(f'<th style="{th_style}">{escape(h)}</th>' for h in headers)}</tr></thead>
      <tbody>{''.join(body)}</tbody>
    </table>
    """
    display(HTML(html))


def render_rank_table(title, rows, headers):
    th_style = "border:1px solid #ddd; padding:4px 8px; text-align:left; background:#eff6ff;"
    td_style = "border:1px solid #ddd; padding:4px 8px; text-align:left;"

    def fmt(value):
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    body = []
    for row in rows:
        body.append("<tr>" + "".join(f'<td style="{td_style}">{escape(fmt(row[h]))}</td>' for h in headers) + "</tr>")
    html = f"""<div style="margin:0.5rem 0 0.25rem 0; font-weight:600;">{escape(title)}</div>
    <table style="border-collapse:collapse; min-width:760px; margin:0.35rem 0 1rem 0;">
      <thead><tr>{''.join(f'<th style="{th_style}">{escape(h)}</th>' for h in headers)}</tr></thead>
      <tbody>{''.join(body)}</tbody>
    </table>"""
    display(HTML(html))


def plot_mattergen_diagnostics(mattergen_run_info, mattergen_rows, palette=None):
    import matplotlib.pyplot as plt
    import numpy as np

    palette = dict(palette or {})
    default_palette = {
        "unconditional": "#4c78a8",
        "low_band_gap": "#54a24b",
        "high_band_gap": "#e45756",
    }
    for key, value in default_palette.items():
        palette.setdefault(key, value)

    label_order = list(mattergen_run_info.keys())
    rows_by_label = {label: [row for row in mattergen_rows if row["label"] == label] for label in label_order}

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), facecolor="white")
    rng = np.random.default_rng(0)
    baseline_label = "unconditional" if "unconditional" in rows_by_label else label_order[0]
    baseline_rows = rows_by_label[baseline_label]
    baseline_mean_n_sites = float(np.mean([row["n_sites"] for row in baseline_rows]))
    baseline_mean_volume = float(np.mean([row["volume"] for row in baseline_rows]))
    shift_rows = []

    for x, label in enumerate(label_order, start=1):
        rows = rows_by_label[label]
        volumes = [row["volume"] for row in rows]
        n_sites = [row["n_sites"] for row in rows]
        jitter = rng.normal(0.0, 0.04, size=len(rows)) if rows else np.array([])

        axes[0, 0].scatter(
            np.full(len(rows), x) + jitter,
            n_sites,
            s=60,
            alpha=0.85,
            color=palette.get(label, "#666666"),
            edgecolors="black",
            linewidths=0.3,
        )
        axes[0, 1].scatter(
            np.full(len(rows), x) + jitter,
            volumes,
            s=60,
            alpha=0.85,
            color=palette.get(label, "#666666"),
            edgecolors="black",
            linewidths=0.3,
        )
        axes[1, 0].scatter(
            n_sites,
            volumes,
            s=90,
            alpha=0.85,
            label=mattergen_run_info[label]["display_name"],
            color=palette.get(label, "#666666"),
            edgecolors="black",
            linewidths=0.3,
        )
        axes[1, 1].hist(
            volumes,
            bins=min(6, max(3, len(rows))),
            alpha=0.55,
            color=palette.get(label, "#666666"),
            label=mattergen_run_info[label]["display_name"],
        )
        shift_rows.append(
            {
                "run": mattergen_run_info[label]["display_name"],
                "target": mattergen_run_info[label]["target_label"],
                "mean_n_sites": float(np.mean(n_sites)),
                "delta_mean_n_sites": float(np.mean(n_sites) - baseline_mean_n_sites),
                "mean_volume": float(np.mean(volumes)),
                "delta_mean_volume": float(np.mean(volumes) - baseline_mean_volume),
                "unique_formulas": len({row["formula"] for row in rows}),
            }
        )

    for ax, ylabel, title in [
        (axes[0, 0], "n_sites", "Atom counts by MatterGen run"),
        (axes[0, 1], "volume (A^3)", "Cell volumes by MatterGen run"),
    ]:
        ax.set_xticks(range(1, len(label_order) + 1))
        ax.set_xticklabels([mattergen_run_info[label]["display_name"] for label in label_order], rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    axes[1, 0].set_title("Did conditioning move the generator into different size regimes?")
    axes[1, 0].set_xlabel("n_sites")
    axes[1, 0].set_ylabel("volume (A^3)")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].set_title("Volume distributions across unconditional and conditioned runs")
    axes[1, 1].set_xlabel("volume (A^3)")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].legend(frameon=False)

    plt.tight_layout()
    plt.show()

    print("Per-run shifts relative to the unconditional baseline:")
    for row in shift_rows:
        print(
            f"- {row['run']}: "
            f"mean n_sites={row['mean_n_sites']:.2f} "
            f"(delta={row['delta_mean_n_sites']:+.2f}), "
            f"mean volume={row['mean_volume']:.2f} A^3 "
            f"(delta={row['delta_mean_volume']:+.2f}), "
            f"unique formulas={row['unique_formulas']}"
        )

    sorted_by_volume = sorted(mattergen_rows, key=lambda row: row["volume"])
    representative_rows = []
    for label in label_order:
        rows = rows_by_label[label]
        mean_n_sites = float(np.mean([row["n_sites"] for row in rows]))
        mean_volume = float(np.mean([row["volume"] for row in rows]))
        representative_rows.extend(
            sorted(
                rows,
                key=lambda row: (
                    abs(row["n_sites"] - mean_n_sites),
                    abs(row["volume"] - mean_volume),
                    row["local_index"],
                ),
            )[:2]
        )

    return {
        "label_order": label_order,
        "rows_by_label": rows_by_label,
        "representative_rows": representative_rows,
        "shift_rows": shift_rows,
        "sorted_by_volume": sorted_by_volume,
    }


def show_mattergen_gallery(mattergen_atoms, selected_rows, title, *, columns=3):
    import matplotlib.pyplot as plt
    import numpy as np
    from ase.visualize.plot import plot_atoms

    selected_rows = list(selected_rows)
    if not selected_rows:
        print(f"No MatterGen structures available for {title}")
        return
    cols = min(columns, len(selected_rows))
    nrows = int(np.ceil(len(selected_rows) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(4 * cols, 4 * nrows), squeeze=False, facecolor="white")
    for ax in axes.ravel():
        ax.axis("off")
    for ax, row in zip(axes.ravel(), selected_rows):
        atoms = mattergen_atoms[row["label"]][row["local_index"]]
        plot_atoms(atoms, ax=ax, rotation="20x,30y,0z", radii=0.35, show_unit_cell=2)
        ax.set_title(
            f"{row['display_name']}\n{row['formula']} | {row['n_sites']} atoms\nvolume={row['volume']:.1f} A^3",
            fontsize=8,
        )
        ax.set_axis_off()
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


def extract_mattergen_trajectory_preview(mattergen_run_info, notebook_root, *, output_dir_name="unzipped_trajectories"):
    import zipfile

    import matplotlib.pyplot as plt
    import numpy as np
    from ase.io import read, write
    from ase.visualize.plot import plot_atoms

    trajectory_dir = Path(notebook_root) / output_dir_name
    trajectory_dir = trajectory_dir.resolve()
    if trajectory_dir.exists():
        shutil.rmtree(trajectory_dir)
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    zip_candidates = [
        Path(mattergen_run_info["unconditional"]["output_dir"]) / "generated_trajectories.zip",
        Path(notebook_root) / "results_unconditional" / "generated_trajectories.zip",
    ]
    zip_file_path = next((p for p in zip_candidates if p.exists()), None)
    if zip_file_path is None:
        raise FileNotFoundError(
            "Could not find a MatterGen trajectory archive. Run the unconditional generation cell with record_trajectories=True."
        )

    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(trajectory_dir)

    trajectory_files = sorted(trajectory_dir.glob("gen_*.extxyz"))
    if not trajectory_files:
        trajectory_files = sorted(trajectory_dir.glob("*.extxyz"))
    if not trajectory_files:
        raise FileNotFoundError(f"Archive {zip_file_path} did not contain any .extxyz trajectory files.")

    selected_trajectory = trajectory_files[0]
    trajectory_frames = read(selected_trajectory, index=":")
    if len(trajectory_frames) < 2:
        raise ValueError(f"Trajectory file {selected_trajectory.name} did not contain multiple frames.")

    frame_ids = np.unique(np.linspace(0, len(trajectory_frames) - 1, num=min(5, len(trajectory_frames)), dtype=int))
    preview_path = trajectory_dir / f"{selected_trajectory.stem}_preview.png"
    gif_path = trajectory_dir / f"{selected_trajectory.stem}.gif"

    fig, axes = plt.subplots(1, len(frame_ids), figsize=(4 * len(frame_ids), 4), squeeze=False, facecolor="white")
    for ax, frame_idx in zip(axes[0], frame_ids):
        plot_atoms(trajectory_frames[frame_idx], ax, rotation="20x,30y,0z", radii=0.35, show_unit_cell=2)
        ax.set_title(f"frame {frame_idx + 1}/{len(trajectory_frames)}", fontsize=9)
        ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(preview_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    write(str(gif_path), trajectory_frames[:: max(1, len(trajectory_frames) // 80)])

    print(f"Extracted {len(trajectory_files)} trajectory files from {zip_file_path.name}")
    print(f"Selected one trajectory: {selected_trajectory.name}")
    print(f"Loaded {len(trajectory_frames)} frames from that single reverse-diffusion path")
    print("First-frame formula:", trajectory_frames[0].get_chemical_formula())
    print("Last-frame formula:", trajectory_frames[-1].get_chemical_formula())

    return {
        "trajectory_dir": trajectory_dir,
        "zip_file_path": zip_file_path,
        "selected_trajectory": selected_trajectory,
        "trajectory_frames": trajectory_frames,
        "preview_path": preview_path,
        "gif_path": gif_path,
    }


__all__ = [
    "MATTERGEN_COMMIT",
    "MATTERGEN_REMOTE",
    "TUTORIAL_COLAB_DIR_CANDIDATES",
    "TUTORIAL_REMOTE",
    "TUTORIAL_REPO",
    "collect_mattergen_runs",
    "ensure_colab_tutorial_repo",
    "extract_mattergen_trajectory_preview",
    "find_notebook_root",
    "inspect_generated_cif_archive",
    "plot_mattergen_diagnostics",
    "render_rank_table",
    "render_summary_table",
    "repo_has_commit",
    "run_mattergen_generation",
    "setup_mattergen_environment",
    "show_mattergen_gallery",
]
