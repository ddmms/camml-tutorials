"""Helper utilities for the Chemeleon-DNG teaching notebook."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from html import escape
from pathlib import Path

from IPython.display import HTML, display


TUTORIAL_REPO = "tutorials"
TUTORIAL_REMOTE = "https://gitlab.com/cam-ml/tutorials.git"
TUTORIAL_COLAB_DIR_CANDIDATES = (
    Path("/content") / "tutorials",
    Path("/content") / "cam_ml_tutorials",
    Path("/content") / "camml-tutorials",
)
NOTEBOOK_FILENAME = "chemeleon-crystals.ipynb"
CHEMELEON_DNG_REMOTE = "https://github.com/hspark1212/chemeleon-dng.git"
CHEMELEON_DNG_COMMIT = "0d8da3a82a0c2211245a1b1394b599ca0545883c"

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


def repo_is_valid(path: Path) -> bool:
    return (path / "pyproject.toml").exists() and (path / ".git").exists()


def repo_has_commit(path: Path, commit: str) -> bool:
    return (
        subprocess.run(
            ["git", "-C", str(path), "cat-file", "-e", f"{commit}^{{commit}}"],
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )


def setup_chemeleon_dng_environment(
    notebook_filename: str = NOTEBOOK_FILENAME,
    chemeleon_commit: str = CHEMELEON_DNG_COMMIT,
    device: str = "cuda",
    demo_timesteps: int = 32,
):
    print("Python:", sys.version)
    print("Working dir:", os.getcwd())

    if "google.colab" in sys.modules:
        ensure_colab_tutorial_repo()

    notebook_root = find_notebook_root(notebook_filename)
    os.chdir(notebook_root)
    print("Notebook root:", notebook_root)

    repo_dir = (notebook_root / "chemeleon_dng_repo").resolve()
    backup_dir = (notebook_root / "chemeleon_dng_repo_incomplete_backup").resolve()
    output_dir = (notebook_root / "results_chemeleon_dng").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tmp_root = Path(tempfile.gettempdir()).resolve()
    venv_dir = tmp_root / "chemeleon-dng-venv"
    python_path = venv_dir / "bin" / "python"
    cli_path = venv_dir / "bin" / "chemeleon-dng"
    uv_cache_dir = tmp_root / "uv-cache-chemeleon-dng"
    uv_cache_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_cache_dir = tmp_root / "matplotlib-cache-chemeleon-dng"
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)

    base_env = os.environ.copy()
    base_env["UV_CACHE_DIR"] = str(uv_cache_dir)
    base_env["UV_LINK_MODE"] = "copy"
    base_env["PYTHONNOUSERSITE"] = "1"
    base_env["MPLBACKEND"] = "Agg"
    base_env["MPLCONFIGDIR"] = str(matplotlib_cache_dir)

    uv_bin = shutil.which("uv")
    if uv_bin is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)
        uv_bin = shutil.which("uv")
    if uv_bin is None:
        raise RuntimeError("Could not find uv after installation.")

    if repo_dir.exists() and not repo_is_valid(repo_dir):
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        repo_dir.rename(backup_dir)
        print("Moved incomplete Chemeleon-DNG directory to:", backup_dir)
    if not repo_is_valid(repo_dir):
        subprocess.run(["git", "clone", CHEMELEON_DNG_REMOTE, str(repo_dir)], check=True)
    else:
        print("Using existing Chemeleon-DNG clone:", repo_dir)

    if not repo_has_commit(repo_dir, chemeleon_commit):
        subprocess.run(["git", "-C", str(repo_dir), "fetch", "--depth", "1", "origin", chemeleon_commit], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "switch", "--detach", chemeleon_commit], check=True)

    def import_check() -> bool:
        if not python_path.exists():
            return False
        result = subprocess.run(
            [
                str(python_path),
                "-c",
                "import chemeleon_dng, numpy, torch; "
                "print(chemeleon_dng.__file__); print(numpy.__version__); print(torch.__version__)",
            ],
            cwd=str(repo_dir),
            env=base_env,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(result.stdout.strip())
            return True
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return False

    if not import_check():
        if venv_dir.exists():
            shutil.rmtree(venv_dir)
        subprocess.run([uv_bin, "venv", str(venv_dir), "--python", "3.11"], check=True, env=base_env)
        subprocess.run(
            [
                uv_bin,
                "pip",
                "install",
                "--python",
                str(python_path),
                "-e",
                str(repo_dir),
            ],
            check=True,
            env=base_env,
        )
        if not import_check():
            raise RuntimeError("Chemeleon-DNG environment was created but the import check still failed.")

    commit = subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"], text=True).strip()
    print("Chemeleon-DNG repo:", repo_dir)
    print("Chemeleon-DNG commit:", commit)
    print("Pinned Chemeleon-DNG commit:", chemeleon_commit)
    print("Chemeleon-DNG venv:", venv_dir)
    print("Chemeleon-DNG Python:", python_path)
    print("Chemeleon-DNG CLI:", cli_path)
    print("Default notebook device:", device)
    print("Notebook demo timesteps:", demo_timesteps)

    return {
        "notebook_root": notebook_root,
        "repo_dir": repo_dir,
        "backup_dir": backup_dir,
        "output_dir": output_dir,
        "venv_dir": venv_dir,
        "python_path": python_path,
        "cli_path": cli_path,
        "device": device,
        "demo_timesteps": demo_timesteps,
        "matplotlib_cache_dir": matplotlib_cache_dir,
        "base_env": base_env,
        "pinned_commit": chemeleon_commit,
    }


def run_chemeleon_dng_python(env, code: str, *, device: str | None = None):
    run_env = env["base_env"].copy()
    run_env["MPLBACKEND"] = "Agg"
    run_env["MPLCONFIGDIR"] = str(env["matplotlib_cache_dir"])
    if (device or env["device"]) == "cpu":
        run_env["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run([str(env["python_path"]), "-c", code], cwd=str(env["repo_dir"]), env=run_env, check=True)


def sample_chemeleon_dng(
    env,
    *,
    task: str,
    output_dir: Path,
    device: str | None = None,
    num_samples: int = 2,
    batch_size: int | None = None,
    formulas: list[str] | None = None,
    num_atom_distribution: str | list[int] | None = "mp-20",
    reuse_existing: bool = True,
    demo_num_timesteps: int | None = None,
    model_path: str | None = None,
):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_count = num_samples if task == "dng" else num_samples * len(formulas or [])
    existing = sorted(output_dir.glob("sample_*.cif"))
    if reuse_existing and expected_count and len(existing) >= expected_count:
        print(f"Reusing {len(existing)} existing CIF files from {output_dir}")
        return existing[:expected_count]

    for old in output_dir.glob("sample_*.cif"):
        old.unlink()
    json_path = output_dir / "generated_structures.json.gz"
    if json_path.exists():
        json_path.unlink()

    resolved_device = device or env["device"]
    resolved_timesteps = env["demo_timesteps"] if demo_num_timesteps is None else int(demo_num_timesteps)

    code_lines = [
        "from pathlib import Path",
        "from chemeleon_dng.diffusion.diffusion_module import DiffusionModule",
        "from chemeleon_dng.download_util import ensure_checkpoints_downloaded",
        "from chemeleon_dng.sample import DEFAULT_MODEL_PATH, get_checkpoint_path, sample_csp, sample_dng",
        f"task = {json.dumps(task)}",
        f"device = {json.dumps(resolved_device)}",
        f"output_dir = Path({json.dumps(str(output_dir))})",
        "output_dir.mkdir(parents=True, exist_ok=True)",
        f"demo_num_timesteps = {resolved_timesteps}",
        f"batch_size = {int(batch_size if batch_size is not None else num_samples)}",
        f"num_samples = {int(num_samples)}",
        f"model_path = {repr(model_path)}",
        "if model_path is None:",
        "    model_path = get_checkpoint_path(task, DEFAULT_MODEL_PATH)",
        "else:",
        "    model_path = Path(model_path)",
        "    if not model_path.is_absolute():",
        "        model_path = (Path.cwd() / model_path).resolve()",
        "    if not model_path.exists():",
        '        print(f\"Checkpoint {model_path} not found locally; downloading the upstream Chemeleon-DNG checkpoints bundle...\")',
        "        ensure_checkpoints_downloaded(str(model_path.parent))",
        "    if not model_path.exists():",
        '        raise FileNotFoundError(f\"Checkpoint not found after download attempt: {model_path}\")',
        "    model_path = str(model_path)",
        'print(f"Using checkpoint path: {model_path}")',
        "dm = DiffusionModule.load_from_checkpoint(model_path, map_location=device, weights_only=False)",
        'print(f"Original timesteps: {dm.num_timesteps}")',
        "dm.num_timesteps = demo_num_timesteps",
        'print(f"Notebook demo timesteps: {dm.num_timesteps}")',
    ]
    if formulas is not None:
        code_lines.append(f"formulas = {json.dumps(list(formulas))}")
    if num_atom_distribution is not None:
        code_lines.append(f"num_atom_distribution = {json.dumps(num_atom_distribution)}")
    code_lines.extend(
        [
            'if task == "dng":',
            "    sample_dng(dm=dm, num_atom_distribution=num_atom_distribution, num_samples=num_samples, batch_size=batch_size, output_path=output_dir)",
            'elif task == "csp":',
            "    sample_csp(dm=dm, formulas=formulas, num_samples=num_samples, batch_size=batch_size, output_path=output_dir)",
            "else:",
            '    raise ValueError(f"Unsupported task: {task}")',
            'print("Generated CIF files:", len(list(output_dir.glob("sample_*.cif"))))',
        ]
    )
    run_chemeleon_dng_python(env, "\n".join(code_lines), device=resolved_device)
    cif_paths = sorted(output_dir.glob("sample_*.cif"))
    if not cif_paths:
        raise FileNotFoundError(f"Chemeleon-DNG sampling finished but wrote no CIF files in {output_dir}")
    return cif_paths


def show_atoms_gallery(atoms_list, title: str, save_path: Path, *, columns: int = 4, subtitles=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from ase.visualize.plot import plot_atoms

    atoms_list = list(atoms_list)
    subtitles = list(subtitles) if subtitles is not None else None
    if not atoms_list:
        print(f"No structures to display for {title}")
        return None
    cols = min(columns, len(atoms_list))
    rows = int(np.ceil(len(atoms_list) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False, facecolor="white")
    for ax in axes.ravel():
        ax.axis("off")
    for idx, (ax, atoms) in enumerate(zip(axes.ravel(), atoms_list)):
        plot_atoms(atoms, ax=ax, radii=0.35, rotation=("10x,15y,0z"))
        ax.set_axis_off()
        subtitle = subtitles[idx] if subtitles and idx < len(subtitles) else f"{atoms.get_chemical_formula()} | {len(atoms)} atoms"
        ax.set_title(subtitle, fontsize=9)
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.show()
    return fig


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


def analyze_dng_runs(dng_run_atoms, dng_run_titles, output_dir: Path, colors=None):
    import matplotlib.pyplot as plt
    import numpy as np

    default_palette = ["#7aa2f7", "#9ece6a", "#f7768e", "#e0af68"]
    colors = dict(colors or {})
    for idx, label in enumerate(dng_run_atoms):
        colors.setdefault(label, default_palette[idx % len(default_palette)])

    dng_rows = []
    for label, atoms_list in dng_run_atoms.items():
        for i, atoms in enumerate(atoms_list):
            volume = float(atoms.get_volume())
            dng_rows.append(
                {
                    "label": label,
                    "display_name": dng_run_titles[label],
                    "sample_id": i,
                    "formula": atoms.get_chemical_formula(),
                    "n_sites": len(atoms),
                    "volume": volume,
                    "density": float(atoms.get_masses().sum() / max(volume, 1e-12)),
                }
            )

    print(f"Loaded {len(dng_rows)} DNG structures across {len(dng_run_atoms)} steering settings")
    baseline_label = "baseline_prior" if "baseline_prior" in dng_run_atoms else next(iter(dng_run_atoms))
    baseline_rows = [row for row in dng_rows if row["label"] == baseline_label]
    baseline_mean_n_sites = float(np.mean([row["n_sites"] for row in baseline_rows]))
    baseline_mean_volume = float(np.mean([row["volume"] for row in baseline_rows]))
    shift_rows = []

    for label in dng_run_atoms:
        rows = [row for row in dng_rows if row["label"] == label]
        mean_n_sites = float(np.mean([row["n_sites"] for row in rows]))
        mean_volume = float(np.mean([row["volume"] for row in rows]))
        shift_rows.append(
            {
                "setting": dng_run_titles[label],
                "mean_n_sites": mean_n_sites,
                "delta_mean_n_sites": mean_n_sites - baseline_mean_n_sites,
                "mean_volume": mean_volume,
                "delta_mean_volume": mean_volume - baseline_mean_volume,
                "unique_formulas": len({row["formula"] for row in rows}),
            }
        )
        print(
            f"- {dng_run_titles[label]}: mean n_sites={mean_n_sites:.2f}, "
            f"mean volume={mean_volume:.2f}, "
            f"unique formulas={len({row['formula'] for row in rows})}"
        )

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), facecolor="white")
    for x, label in enumerate(dng_run_atoms, start=1):
        rows = [row for row in dng_rows if row["label"] == label]
        jitter = np.linspace(-0.08, 0.08, num=len(rows)) if len(rows) > 1 else np.array([0.0])
        axes[0, 0].scatter(
            np.full(len(rows), x) + jitter,
            [row["n_sites"] for row in rows],
            s=65,
            color=colors[label],
            edgecolors="black",
            linewidths=0.3,
        )
        axes[0, 1].scatter(
            np.full(len(rows), x) + jitter,
            [row["volume"] for row in rows],
            s=65,
            color=colors[label],
            edgecolors="black",
            linewidths=0.3,
        )
        axes[1, 0].scatter(
            [row["n_sites"] for row in rows],
            [row["volume"] for row in rows],
            s=95,
            color=colors[label],
            label=dng_run_titles[label],
            edgecolors="black",
            linewidths=0.3,
        )
        axes[1, 1].hist(
            [row["n_sites"] for row in rows],
            bins=np.arange(5.5, 17.6, 1.0),
            alpha=0.5,
            color=colors[label],
            label=dng_run_titles[label],
        )

    for ax, ylabel, title in [
        (axes[0, 0], "n_sites", "Atom-count control in DNG"),
        (axes[0, 1], "volume (A^3)", "Volume by DNG steering setting"),
    ]:
        ax.set_xticks(range(1, len(dng_run_atoms) + 1))
        ax.set_xticklabels([dng_run_titles[label] for label in dng_run_atoms], rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    axes[1, 0].set_title("Do the steering schedules move the generator into different size regimes?")
    axes[1, 0].set_xlabel("n_sites")
    axes[1, 0].set_ylabel("volume (A^3)")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].set_title("Atom-count distributions across DNG steering settings")
    axes[1, 1].set_xlabel("n_sites")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].legend(frameon=False)

    plt.tight_layout()
    plt.show()

    dng_sorted_by_n_sites = sorted(dng_rows, key=lambda row: (row["n_sites"], row["volume"], row["sample_id"]))
    render_rank_table(
        "Smallest generated cells across all DNG steering settings",
        dng_sorted_by_n_sites[:4],
        ["display_name", "formula", "n_sites", "volume"],
    )
    render_rank_table(
        "Largest generated cells across all DNG steering settings",
        list(reversed(dng_sorted_by_n_sites[-4:])),
        ["display_name", "formula", "n_sites", "volume"],
    )

    output_dir = Path(output_dir)
    if "small_cells" in dng_run_atoms:
        show_atoms_gallery(
            dng_run_atoms["small_cells"][:3],
            "Chemeleon-DNG small-cell DNG steering",
            output_dir / "dng_small_cells_gallery.png",
            subtitles=[
                f"{atoms.get_chemical_formula()} | {len(atoms)} atoms | volume={float(atoms.get_volume()):.1f} A^3"
                for atoms in dng_run_atoms["small_cells"][:3]
            ],
            columns=3,
        )
    if "large_cells" in dng_run_atoms:
        show_atoms_gallery(
            dng_run_atoms["large_cells"][:3],
            "Chemeleon-DNG large-cell DNG steering",
            output_dir / "dng_large_cells_gallery.png",
            subtitles=[
                f"{atoms.get_chemical_formula()} | {len(atoms)} atoms | volume={float(atoms.get_volume()):.1f} A^3"
                for atoms in dng_run_atoms["large_cells"][:3]
            ],
            columns=3,
        )

    show_atoms_gallery(
        [dng_run_atoms[row["label"]][row["sample_id"]] for row in dng_sorted_by_n_sites[:3]],
        "Smallest generated cells across all steering settings",
        output_dir / "dng_smallest_cells_gallery.png",
        subtitles=[f"{row['display_name']} | {row['formula']} | {row['n_sites']} atoms" for row in dng_sorted_by_n_sites[:3]],
        columns=3,
    )
    show_atoms_gallery(
        [dng_run_atoms[row["label"]][row["sample_id"]] for row in list(reversed(dng_sorted_by_n_sites[-3:]))],
        "Largest generated cells across all steering settings",
        output_dir / "dng_largest_cells_gallery.png",
        subtitles=[
            f"{row['display_name']} | {row['formula']} | {row['n_sites']} atoms"
            for row in list(reversed(dng_sorted_by_n_sites[-3:]))
        ],
        columns=3,
    )

    return {"rows": dng_rows, "shift_rows": shift_rows, "sorted_by_n_sites": dng_sorted_by_n_sites}


def analyze_csp_samples(csp_samples, csp_targets, output_dir: Path, colors=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from ase.formula import Formula

    default_palette = ["#7aa2f7", "#f7768e", "#9ece6a", "#e0af68"]
    colors = dict(colors or {})
    for idx, formula in enumerate(csp_targets):
        colors.setdefault(formula, default_palette[idx % len(default_palette)])

    csp_rows = []
    for formula, atoms_list in csp_samples.items():
        target_reduced = Formula(formula).reduce()[0].format("hill")
        for i, atoms in enumerate(atoms_list):
            volume = float(atoms.get_volume())
            generated_reduced = Formula(atoms.get_chemical_formula()).reduce()[0].format("hill")
            csp_rows.append(
                {
                    "formula_target": formula,
                    "sample_id": i,
                    "generated_formula": atoms.get_chemical_formula(),
                    "generated_reduced_formula": generated_reduced,
                    "formula_match": generated_reduced == target_reduced,
                    "n_sites": len(atoms),
                    "volume": volume,
                    "density": float(atoms.get_masses().sum() / max(volume, 1e-12)),
                    "atoms": atoms,
                }
            )

    print(f"Loaded {len(csp_rows)} CSP structures across {len(csp_samples)} formulas")
    summary_rows = []
    for formula in csp_targets:
        subset = [row for row in csp_rows if row["formula_target"] == formula]
        mean_volume = float(np.mean([row["volume"] for row in subset]))
        mean_n_sites = float(np.mean([row["n_sites"] for row in subset]))
        match_fraction = float(np.mean([row["formula_match"] for row in subset]))
        summary_rows.append(
            {
                "formula_target": formula,
                "mean_volume": mean_volume,
                "mean_n_sites": mean_n_sites,
                "formula_match_fraction": match_fraction,
            }
        )
        print(
            f"- {formula}: mean volume={mean_volume:.2f}, "
            f"mean n_sites={mean_n_sites:.2f}, "
            f"formula-match fraction={match_fraction:.2f}"
        )

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), facecolor="white")
    for x, formula in enumerate(csp_targets, start=1):
        subset = [row for row in csp_rows if row["formula_target"] == formula]
        jitter = np.linspace(-0.06, 0.06, num=len(subset)) if len(subset) > 1 else np.array([0.0])
        axes[0, 0].scatter(
            np.full(len(subset), x) + jitter,
            [row["volume"] for row in subset],
            s=65,
            color=colors[formula],
            edgecolors="black",
            linewidths=0.3,
        )
        axes[0, 1].scatter(
            np.full(len(subset), x) + jitter,
            [row["n_sites"] for row in subset],
            s=65,
            color=colors[formula],
            edgecolors="black",
            linewidths=0.3,
        )
        axes[1, 0].scatter(
            [row["volume"] for row in subset],
            [row["n_sites"] for row in subset],
            label=formula,
            s=90,
            color=colors[formula],
            edgecolors="black",
            linewidths=0.3,
        )
        axes[1, 1].bar(x, np.mean([row["formula_match"] for row in subset]), color=colors[formula], width=0.6)

    for ax, ylabel, title in [
        (axes[0, 0], "volume (A^3)", "Volume by CSP target formula"),
        (axes[0, 1], "n_sites", "Atom count by CSP target formula"),
    ]:
        ax.set_xticks(range(1, len(csp_targets) + 1))
        ax.set_xticklabels(csp_targets)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    axes[1, 0].set_title("Did different formulas land in different structural size regimes?")
    axes[1, 0].set_xlabel("volume")
    axes[1, 0].set_ylabel("n_sites")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].set_xticks(range(1, len(csp_targets) + 1))
    axes[1, 1].set_xticklabels(csp_targets)
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_ylabel("formula-match fraction")
    axes[1, 1].set_title("Did the generated stoichiometry match the target?")

    plt.tight_layout()
    plt.show()

    csp_matching_rows = [row for row in csp_rows if row["formula_match"]]
    csp_mismatched_rows = [row for row in csp_rows if not row["formula_match"]]
    csp_matching_rows = sorted(csp_matching_rows, key=lambda row: (row["formula_target"], row["volume"], row["sample_id"]))
    csp_mismatched_rows = sorted(csp_mismatched_rows, key=lambda row: (row["formula_target"], row["volume"], row["sample_id"]))
    render_rank_table(
        "Formula-matching CSP candidates",
        csp_matching_rows[:4],
        ["formula_target", "generated_formula", "n_sites", "volume"],
    )
    if csp_mismatched_rows:
        render_rank_table(
            "Off-target CSP candidates",
            csp_mismatched_rows[:4],
            ["formula_target", "generated_formula", "n_sites", "volume"],
        )

    output_dir = Path(output_dir)
    if csp_matching_rows:
        show_atoms_gallery(
            [row["atoms"] for row in csp_matching_rows[:2]],
            "Formula-matching CSP candidates",
            output_dir / "csp_matching_gallery.png",
            subtitles=[
                f"{row['formula_target']} target | {row['generated_formula']} | volume={row['volume']:.1f} A^3"
                for row in csp_matching_rows[:2]
            ],
            columns=2,
        )
    if csp_mismatched_rows:
        show_atoms_gallery(
            [row["atoms"] for row in csp_mismatched_rows[:2]],
            "Off-target CSP candidates",
            output_dir / "csp_mismatched_gallery.png",
            subtitles=[
                f"{row['formula_target']} target | {row['generated_formula']} | volume={row['volume']:.1f} A^3"
                for row in csp_mismatched_rows[:2]
            ],
            columns=2,
        )

    return {
        "rows": csp_rows,
        "summary_rows": summary_rows,
        "matching_rows": csp_matching_rows,
        "mismatched_rows": csp_mismatched_rows,
    }


__all__ = [
    "CHEMELEON_DNG_COMMIT",
    "CHEMELEON_DNG_REMOTE",
    "NOTEBOOK_FILENAME",
    "TUTORIAL_COLAB_DIR_CANDIDATES",
    "TUTORIAL_REMOTE",
    "TUTORIAL_REPO",
    "analyze_csp_samples",
    "analyze_dng_runs",
    "ensure_colab_tutorial_repo",
    "find_notebook_root",
    "render_rank_table",
    "repo_has_commit",
    "repo_is_valid",
    "run_chemeleon_dng_python",
    "sample_chemeleon_dng",
    "setup_chemeleon_dng_environment",
    "show_atoms_gallery",
]
