"""Dataset and geometry helpers for the crystal diffusion notebook."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from pathlib import Path
import pandas as pd

import numpy as np
import requests
import torch
from monty.serialization import loadfn
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm.auto import tqdm

try:
    from mp_api.client import MPRester
except Exception:
    MPRester = None


VAL_FRACTION = 0.1

GEOM_LOG_LENGTH_CLAMP = (-0.5, 2.5)
GEOM_COS_CLAMP = (-0.98, 0.98)
GEOM_MAX_DIST = 12.0

MIN_DECODED_VOLUME = 5e-2
MIN_DECODED_LENGTH = 2.0
MAX_DECODED_LENGTH = 20.0
MIN_DECODED_ANGLE = 35.0
MAX_DECODED_ANGLE = 145.0

DEFAULT_MP_CACHE_DIR = Path("mp_curated_cache")
DEFAULT_MP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

BUNDLED_MP_FALLBACK_PATH = DEFAULT_MP_CACHE_DIR / "bundled_real_mp_2_to_4_elements_small.json.gz"
BUNDLED_MP_FALLBACK_CONFIG = {
    "name": "bundled_real_mp_2_to_4_elements_small",
    "chemistry_scope": "2_to_4_elements",
    "min_num_elements": 2,
    "max_num_elements": 4,
    "max_atoms": 20,
    "max_structures": 128,
    "energy_above_hull_max": 0.05,
    "is_stable_only": True,
    "include_theoretical": False,
    "exclude_elements": ["H"],
}

CHEMISTRY_SCOPE_TO_NUM_ELEMENTS = {
    "2_elements": (2, 2),
    "3_elements": (3, 3),
    "2_to_4_elements": (2, 4),
    "2_to_5_elements": (2, 5),
}

DEFAULT_SEED = 7
_LAT_MEAN = None
_LAT_STD = None


def set_lattice_feature_stats(lat_mean, lat_std) -> None:
    """Store lattice normalization stats for helper functions that decode features."""
    global _LAT_MEAN, _LAT_STD
    _LAT_MEAN = np.asarray(lat_mean, dtype=np.float32)
    _LAT_STD = np.asarray(lat_std, dtype=np.float32)


def _require_lattice_feature_stats() -> tuple[np.ndarray, np.ndarray]:
    if _LAT_MEAN is None or _LAT_STD is None:
        raise RuntimeError(
            "Lattice feature statistics are unset. Run `set_lattice_feature_stats(lat_mean, lat_std)` first."
        )
    return _LAT_MEAN, _LAT_STD


def normalized_structure_anonymous_formula(structure: Structure) -> str:
    try:
        return str(structure.composition.anonymized_formula).replace(" ", "")
    except Exception:
        return ""


def canonicalize_config_for_cache(cfg: dict) -> dict:
    return json.loads(json.dumps(cfg, sort_keys=True))


def dataset_cache_path_from_config(cfg: dict) -> Path:
    canonical = canonicalize_config_for_cache(cfg)
    digest = hashlib.md5(json.dumps(canonical, sort_keys=True).encode()).hexdigest()[:10]
    safe_name = (
        re.sub(r"[^a-zA-Z0-9_-]+", "_", str(cfg.get("name", "mp_general_dataset"))).strip("_")
        or "mp_general_dataset"
    )
    return DEFAULT_MP_CACHE_DIR / f"{safe_name}_{digest}.json.gz"


def make_general_dataset_config(
    chemistry_scope: str = "2_to_4_elements",
    max_structures: int = 96,
    max_atoms: int = 16,
    force_refresh: bool = False,
):
    if chemistry_scope not in CHEMISTRY_SCOPE_TO_NUM_ELEMENTS:
        raise ValueError(f"Unknown chemistry_scope: {chemistry_scope}")
    min_el, max_el = CHEMISTRY_SCOPE_TO_NUM_ELEMENTS[chemistry_scope]
    target = int(max_structures)
    chunk_size = 250
    num_chunks = max(2, math.ceil((target * 3) / chunk_size))
    return {
        "name": f"general_{chemistry_scope}",
        "chemistry_scope": chemistry_scope,
        "min_num_elements": int(min_el),
        "max_num_elements": int(max_el),
        "max_atoms": int(max_atoms),
        "max_structures": int(max_structures),
        "energy_above_hull_max": 0.05,
        "is_stable_only": True,
        "include_theoretical": False,
        "exclude_elements": ["H"],
        "force_refresh": bool(force_refresh),
        "query_chunk_size": int(chunk_size),
        "query_num_chunks": int(num_chunks),
    }


def build_general_mp_query_from_config(cfg: dict):
    min_el = int(cfg["min_num_elements"])
    max_el = int(cfg["max_num_elements"])
    num_elements_filter = min_el if min_el == max_el else (min_el, max_el)

    return dict(
        num_sites=(2, int(cfg["max_atoms"])),
        num_elements=num_elements_filter,
        energy_above_hull=(0.0, float(cfg.get("energy_above_hull_max", 0.05))),
        is_stable=bool(cfg.get("is_stable_only", True)),
        theoretical=bool(cfg.get("include_theoretical", False)),
        exclude_elements=list(cfg.get("exclude_elements", ["H"])),
        num_chunks=int(cfg.get("query_num_chunks", 2)),
        chunk_size=int(cfg.get("query_chunk_size", 250)),
        all_fields=False,
        fields=[
            "material_id",
            "formula_pretty",
            "density",
            "structure",
            "energy_above_hull",
            "band_gap",
            "is_stable",
            "theoretical",
        ],
    )


def mp_doc_get(doc, key: str, default=None):
    if isinstance(doc, dict):
        return doc.get(key, default)
    return getattr(doc, key, default)


def structure_from_mp_doc(doc):
    structure = mp_doc_get(doc, "structure", None)
    if structure is None:
        return None
    if isinstance(structure, Structure):
        return structure
    if isinstance(structure, dict):
        return Structure.from_dict(structure)
    raise TypeError(f"Unsupported structure payload type: {type(structure)!r}")


def build_general_mp_rest_params(query: dict) -> dict:
    params = {}

    num_sites = query.get("num_sites")
    if num_sites is not None:
        if isinstance(num_sites, (tuple, list)):
            params["nsites_min"] = int(num_sites[0])
            params["nsites_max"] = int(num_sites[1])
        else:
            params["nsites_min"] = int(num_sites)
            params["nsites_max"] = int(num_sites)

    num_elements = query.get("num_elements")
    if num_elements is not None:
        if isinstance(num_elements, (tuple, list)):
            params["nelements_min"] = int(num_elements[0])
            params["nelements_max"] = int(num_elements[1])
        else:
            params["nelements_min"] = int(num_elements)
            params["nelements_max"] = int(num_elements)

    energy_above_hull = query.get("energy_above_hull")
    if energy_above_hull is not None:
        params["energy_above_hull_min"] = float(energy_above_hull[0])
        params["energy_above_hull_max"] = float(energy_above_hull[1])

    if query.get("is_stable") is not None:
        params["is_stable"] = bool(query["is_stable"])
    if query.get("theoretical") is not None:
        params["theoretical"] = bool(query["theoretical"])

    exclude_elements = query.get("exclude_elements")
    if exclude_elements:
        params["exclude_elements"] = ",".join(str(el) for el in exclude_elements)

    fields = query.get("fields")
    if fields:
        params["_fields"] = ",".join(fields)
    params["_all_fields"] = False
    return params


def fetch_general_summary_docs_via_rest(api_key: str, cfg: dict):
    query = build_general_mp_query_from_config(cfg)
    base_params = build_general_mp_rest_params(query)
    url = "https://api.materialsproject.org/materials/summary/"
    headers = {"X-API-KEY": api_key}
    docs = []
    seen = set()

    num_chunks = int(query.get("num_chunks", 2))
    chunk_size = int(query.get("chunk_size", 250))

    for chunk_idx in range(num_chunks):
        params = dict(base_params)
        params["_limit"] = chunk_size
        params["_skip"] = chunk_idx * chunk_size

        response = requests.get(url, params=params, headers=headers, timeout=60)
        response.raise_for_status()
        payload = response.json()
        page_docs = payload.get("data", [])
        if not page_docs:
            break

        for doc in page_docs:
            mpid = str(mp_doc_get(doc, "material_id", ""))
            if mpid in seen:
                continue
            seen.add(mpid)
            docs.append(doc)

        if len(page_docs) < chunk_size:
            break

    query = dict(query)
    query["transport"] = "rest"
    return docs, query


def is_teaching_friendly_structure(structure: Structure) -> bool:
    elements = getattr(structure.composition, "elements", [])
    if any(getattr(el, "is_radioactive", False) for el in elements):
        return False
    if any(getattr(el, "is_noble_gas", False) for el in elements):
        return False
    return True


def build_synthetic_records(cfg: dict, seed: int = DEFAULT_SEED):
    rng = np.random.default_rng(seed)
    prototypes = [
        ("NaCl", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]], 5.0),
        ("CsCl", Lattice.cubic(4.12), ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]], 4.4),
        ("Si", Lattice.cubic(5.43), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]], 1.1),
        ("ZnS", Lattice.cubic(5.41), ["Zn", "S"], [[0, 0, 0], [0.25, 0.25, 0.25]], 3.2),
        ("MgO", Lattice.cubic(4.21), ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]], 7.4),
        (
            "SrTiO3",
            Lattice.cubic(3.905),
            ["Sr", "Ti", "O", "O", "O"],
            [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
            2.1,
        ),
        (
            "CaTiO3",
            Lattice.orthorhombic(5.44, 7.64, 5.38),
            ["Ca", "Ti", "O", "O", "O"],
            [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0.25]],
            3.0,
        ),
        (
            "LiFePO4",
            Lattice.orthorhombic(10.33, 6.01, 4.69),
            ["Li", "Fe", "P", "O", "O", "O", "O", "O", "O", "O"],
            [
                [0.1, 0.1, 0.1],
                [0.3, 0.3, 0.3],
                [0.5, 0.5, 0.5],
                [0.05, 0.15, 0.25],
                [0.15, 0.25, 0.35],
                [0.25, 0.35, 0.45],
                [0.35, 0.45, 0.55],
                [0.45, 0.55, 0.65],
                [0.55, 0.65, 0.75],
                [0.65, 0.75, 0.85],
            ],
            3.4,
        ),
        (
            "Na2CO3",
            Lattice.monoclinic(7.1, 5.8, 6.4, 110),
            ["Na", "Na", "C", "O", "O", "O"],
            [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]],
            5.8,
        ),
        (
            "BaTiO3",
            Lattice.cubic(4.00),
            ["Ba", "Ti", "O", "O", "O"],
            [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
            2.3,
        ),
    ]

    target = int(cfg.get("max_structures", 24))
    min_el = int(cfg.get("min_num_elements", 2))
    max_el = int(cfg.get("max_num_elements", 5))
    max_atoms = int(cfg.get("max_atoms", 16))

    records = []
    skipped = {"synthetic": 0}
    i = 0
    while len(records) < target:
        name, lattice, species, frac_coords, base_gap = prototypes[i % len(prototypes)]
        scale = float(rng.uniform(0.96, 1.08))
        structure = Structure(Lattice(lattice.matrix * scale), species, frac_coords)
        num_elements = len({str(el) for el in structure.composition.elements})
        if not (min_el <= num_elements <= max_el) or len(structure) > max_atoms:
            i += 1
            skipped["synthetic"] += 1
            continue
        try:
            sga = SpacegroupAnalyzer(structure, symprec=0.15)
            sg_num = int(sga.get_space_group_number())
            sg_symbol = str(sga.get_space_group_symbol())
        except Exception:
            sg_num = 1
            sg_symbol = "P1"
        records.append(
            {
                "name": f"{name}_{len(records):03d}",
                "formula": structure.composition.reduced_formula,
                "chemsys": "-".join(sorted({str(el) for el in structure.composition.elements})),
                "structure": structure,
                "atomic_numbers": [int(site.specie.Z) for site in structure],
                "num_atoms": int(len(structure)),
                "frac0": wrap_centered_np(np.asarray(structure.frac_coords, dtype=np.float32)).astype(np.float32),
                "lattice0": lattice_to_features(structure).astype(np.float32),
                "density": float(structure.density),
                "band_gap": float(max(0.0, base_gap + rng.normal(0.0, 0.2))),
                "spacegroup_number": sg_num,
                "spacegroup_symbol": sg_symbol,
            }
        )
        i += 1

    return records, skipped, {"source": "synthetic_last_resort", "target_size": target}


def fetch_general_summary_docs(api_key: str, cfg: dict):
    query = build_general_mp_query_from_config(cfg)

    if MPRester is not None:
        try:
            docs = []
            seen = set()
            with MPRester(api_key) as mpr:
                for doc in mpr.materials.summary.search(**query):
                    mpid = str(mp_doc_get(doc, "material_id", ""))
                    if mpid in seen:
                        continue
                    docs.append(doc)
                    seen.add(mpid)
            query = dict(query)
            query["transport"] = "mp_api"
            return docs, query
        except Exception as exc:
            print("`mp_api` import/query failed in this environment; retrying via direct REST requests to Materials Project.")
            print(f"Reason: {exc.__class__.__name__}: {exc}")

    return fetch_general_summary_docs_via_rest(api_key=api_key, cfg=cfg)


def select_evenly_spaced_records(records, target_size: int):
    records = list(records)
    if len(records) <= target_size:
        return records
    records = sorted(records, key=lambda r: (float(r["density"]), float(r["band_gap"]), r["formula"]))
    idxs = np.linspace(0, len(records) - 1, target_size).round().astype(int).tolist()
    dedup = []
    seen = set()
    for idx in idxs:
        if idx not in seen:
            dedup.append(records[idx])
            seen.add(idx)
    return dedup


def cache_payload_source_label(cache_payload, records) -> str:
    if isinstance(cache_payload, dict):
        mp_meta = cache_payload.get("mp_query_used", {}) or {}
        if mp_meta.get("source"):
            return str(mp_meta["source"])
    if records:
        record_source = records[0].get("source")
        if record_source:
            return str(record_source)
    return "unknown"


def cache_payload_is_real_mp(cache_payload, records) -> bool:
    source = cache_payload_source_label(cache_payload, records).lower()
    return "materials_project" in source or source in {
        "bundled_materials_project_fallback",
        "materials_project_live_query",
    }


def ensure_record_feature_fields(records):
    if not records:
        return records
    for r in records:
        structure = r.get("structure")
        if structure is None:
            continue
        if "lattice0" not in r:
            r["lattice0"] = lattice_to_features(structure).astype(np.float32)
        if "frac0" not in r:
            r["frac0"] = wrap_centered_np(np.asarray(structure.frac_coords, dtype=np.float32)).astype(np.float32)
        if "num_atoms" not in r:
            r["num_atoms"] = int(len(structure))
        if "formula" not in r:
            r["formula"] = structure.composition.reduced_formula
        if "chemsys" not in r:
            r["chemsys"] = "-".join(sorted({el.symbol for el in structure.composition.elements}))
        if "num_elements" not in r:
            r["num_elements"] = int(len(structure.composition.elements))
        if "spacegroup_number" not in r or "spacegroup_symbol" not in r:
            try:
                sga = SpacegroupAnalyzer(structure, symprec=0.15)
                r.setdefault("spacegroup_number", int(sga.get_space_group_number()))
                r.setdefault("spacegroup_symbol", str(sga.get_space_group_symbol()))
            except Exception:
                r.setdefault("spacegroup_number", 1)
                r.setdefault("spacegroup_symbol", "P1")
    return records


def filter_records_for_config(records, cfg: dict):
    min_el = int(cfg.get("min_num_elements", 2))
    max_el = int(cfg.get("max_num_elements", 5))
    max_atoms = int(cfg.get("max_atoms", 20))
    target = int(cfg.get("max_structures", len(records)))

    filtered = []
    for r in records:
        structure = r.get("structure")
        num_atoms = int(r.get("num_atoms", len(structure) if structure is not None else 0))
        num_elements = int(r.get("num_elements", len(structure.composition.elements) if structure is not None else 0))
        if structure is not None and not is_teaching_friendly_structure(structure):
            continue
        if num_atoms > max_atoms:
            continue
        if num_elements < min_el or num_elements > max_el:
            continue
        filtered.append(r)

    return select_evenly_spaced_records(filtered, min(target, len(filtered)))


def load_bundled_fallback_records(cfg: dict):
    if not BUNDLED_MP_FALLBACK_PATH.exists():
        raise FileNotFoundError(f"Bundled fallback dataset is missing: {BUNDLED_MP_FALLBACK_PATH}")

    payload = loadfn(BUNDLED_MP_FALLBACK_PATH)
    if isinstance(payload, dict) and "records" in payload:
        raw_records = payload["records"]
        skipped = payload.get("skipped", {})
        meta = dict(payload.get("mp_query_used", {}) or {})
    else:
        raw_records = payload
        skipped = {}
        meta = {}

    raw_records = ensure_record_feature_fields(list(raw_records))
    records = filter_records_for_config(raw_records, cfg)
    meta["source"] = "bundled_materials_project_fallback"
    meta["bundled_fallback_path"] = BUNDLED_MP_FALLBACK_PATH.name
    meta["bundled_record_count"] = len(raw_records)
    meta["adapted_record_count"] = len(records)
    return records, skipped, meta


def probe_materials_project_connection(api_key: str):
    if MPRester is not None:
        try:
            with MPRester(api_key) as mpr:
                docs = list(
                    mpr.materials.summary.search(
                        material_ids=["mp-149"],
                        all_fields=False,
                        fields=["material_id"],
                    )
                )
            return len(docs)
        except Exception:
            pass

    response = requests.get(
        "https://api.materialsproject.org/materials/summary/",
        params={
            "material_ids": "mp-149",
            "_fields": "material_id",
            "_all_fields": False,
            "_limit": 1,
        },
        headers={"X-API-KEY": api_key},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return len(payload.get("data", []))


def curate_general_summary_docs(docs, cfg: dict, seed: int = DEFAULT_SEED):
    rng = random.Random(seed)
    docs = list(docs)
    rng.shuffle(docs)

    records = []
    skipped = {
        "missing_structure": 0,
        "missing_band_gap": 0,
        "parse_error": 0,
        "too_many_atoms": 0,
        "too_few_elements": 0,
        "too_many_elements": 0,
        "unfriendly_chemistry": 0,
    }

    min_el = int(cfg["min_num_elements"])
    max_el = int(cfg["max_num_elements"])
    max_atoms = int(cfg["max_atoms"])

    for doc in tqdm(docs, desc="Curating MP structures"):
        try:
            structure = structure_from_mp_doc(doc)
            if structure is None:
                skipped["missing_structure"] += 1
                continue

            structure = structure.copy()
            try:
                structure = structure.get_primitive_structure()
            except Exception:
                pass

            if len(structure) > max_atoms:
                skipped["too_many_atoms"] += 1
                continue

            num_elements = len(structure.composition.elements)
            if num_elements < min_el:
                skipped["too_few_elements"] += 1
                continue
            if num_elements > max_el:
                skipped["too_many_elements"] += 1
                continue
            if not is_teaching_friendly_structure(structure):
                skipped["unfriendly_chemistry"] += 1
                continue

            band_gap = mp_doc_get(doc, "band_gap", None)
            if band_gap is None or not np.isfinite(float(band_gap)):
                skipped["missing_band_gap"] += 1
                continue

            try:
                sga = SpacegroupAnalyzer(structure, symprec=0.15)
                sg_num = int(sga.get_space_group_number())
                sg_symbol = str(sga.get_space_group_symbol())
            except Exception:
                sg_num = 1
                sg_symbol = "P1"

            atomic_numbers = np.array([site.specie.Z for site in structure], dtype=np.int64)
            frac0 = wrap_centered_np(np.asarray(structure.frac_coords, dtype=np.float32)).astype(np.float32)
            lattice0 = lattice_to_features(structure).astype(np.float32)
            density_value = mp_doc_get(doc, "density", None)
            density = float(density_value if density_value is not None else structure.density)
            band_gap = max(0.0, float(band_gap))
            chemsys = "-".join(sorted({el.symbol for el in structure.composition.elements}))

            records.append(
                {
                    "source": "materials_project",
                    "name": str(mp_doc_get(doc, "material_id", "unknown")),
                    "mpid": str(mp_doc_get(doc, "material_id", "unknown")),
                    "formula": str(mp_doc_get(doc, "formula_pretty", None) or structure.composition.reduced_formula),
                    "anonymous_formula": normalized_structure_anonymous_formula(structure),
                    "chemsys": chemsys,
                    "num_elements": int(num_elements),
                    "num_atoms": int(len(structure)),
                    "atomic_numbers": atomic_numbers,
                    "frac0": frac0,
                    "lattice0": lattice0,
                    "density": density,
                    "band_gap": band_gap,
                    "energy_above_hull": (
                        float(mp_doc_get(doc, "energy_above_hull"))
                        if mp_doc_get(doc, "energy_above_hull", None) is not None
                        else np.nan
                    ),
                    "spacegroup_number": sg_num,
                    "spacegroup_symbol": sg_symbol,
                    "structure": structure,
                }
            )
        except Exception:
            skipped["parse_error"] += 1
            continue

    records = select_evenly_spaced_records(records, int(cfg["max_structures"]))
    return records, skipped


def wrap_centered_np(frac: np.ndarray) -> np.ndarray:
    return ((frac + 0.5) % 1.0) - 0.5


def wrap_centered_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x + 0.5, 1.0) - 0.5


def lattice_to_features(structure: Structure) -> np.ndarray:
    a, b, c = structure.lattice.abc
    alpha, beta, gamma = structure.lattice.angles
    return np.array(
        [
            np.log(a),
            np.log(b),
            np.log(c),
            np.cos(np.deg2rad(alpha)),
            np.cos(np.deg2rad(beta)),
            np.cos(np.deg2rad(gamma)),
        ],
        dtype=np.float32,
    )


def features_to_lattice(features: np.ndarray) -> Lattice:
    features = np.asarray(features, dtype=np.float64)

    log_lengths = np.clip(features[:3], *GEOM_LOG_LENGTH_CLAMP)
    lengths = np.exp(log_lengths)
    lengths = np.clip(lengths, MIN_DECODED_LENGTH, MAX_DECODED_LENGTH)

    cos_angles = np.clip(features[3:], *GEOM_COS_CLAMP)
    angles = np.rad2deg(np.arccos(cos_angles))
    angles = np.clip(angles, MIN_DECODED_ANGLE, MAX_DECODED_ANGLE)

    try:
        lattice = Lattice.from_parameters(*lengths.tolist(), *angles.tolist())
    except Exception:
        side = float(np.clip(np.mean(lengths), MIN_DECODED_LENGTH, MAX_DECODED_LENGTH))
        lattice = Lattice.cubic(side)

    volume = float(getattr(lattice, "volume", np.nan))
    if (not np.isfinite(volume)) or volume <= MIN_DECODED_VOLUME:
        side = float(
            np.clip(
                np.cbrt(max(float(np.prod(lengths)), MIN_DECODED_VOLUME)),
                MIN_DECODED_LENGTH,
                MAX_DECODED_LENGTH,
            )
        )
        lattice = Lattice.cubic(side)

    return lattice


def denormalize_lattice_features_torch(features_norm: torch.Tensor) -> torch.Tensor:
    lat_mean, lat_std = _require_lattice_feature_stats()
    lat_mean_t = torch.as_tensor(lat_mean, dtype=features_norm.dtype, device=features_norm.device)
    lat_std_t = torch.as_tensor(lat_std, dtype=features_norm.dtype, device=features_norm.device)
    return features_norm * lat_std_t + lat_mean_t


def clamp_lattice_features_norm_torch(features_norm: torch.Tensor) -> torch.Tensor:
    lat_mean, lat_std = _require_lattice_feature_stats()
    features = denormalize_lattice_features_torch(features_norm).clone()

    features[:, :3] = torch.clamp(
        features[:, :3],
        min=float(np.log(MIN_DECODED_LENGTH)),
        max=float(np.log(MAX_DECODED_LENGTH)),
    )

    cos_min = float(np.cos(np.deg2rad(MAX_DECODED_ANGLE)))
    cos_max = float(np.cos(np.deg2rad(MIN_DECODED_ANGLE)))
    features[:, 3:] = torch.clamp(features[:, 3:], min=cos_min, max=cos_max)

    lat_mean_t = torch.as_tensor(lat_mean, dtype=features_norm.dtype, device=features_norm.device)
    lat_std_t = torch.as_tensor(lat_std, dtype=features_norm.dtype, device=features_norm.device)
    return (features - lat_mean_t) / torch.clamp(lat_std_t, min=1e-8)


def lattice_matrix_from_features_torch(features: torch.Tensor) -> torch.Tensor:
    log_lengths = torch.clamp(features[:, :3], min=GEOM_LOG_LENGTH_CLAMP[0], max=GEOM_LOG_LENGTH_CLAMP[1])
    lengths = torch.exp(log_lengths)
    cos_angles = torch.clamp(features[:, 3:], min=GEOM_COS_CLAMP[0], max=GEOM_COS_CLAMP[1])
    angles = torch.arccos(cos_angles)

    a, b, c = lengths[:, 0], lengths[:, 1], lengths[:, 2]
    alpha, beta, gamma = angles[:, 0], angles[:, 1], angles[:, 2]

    va = torch.stack([a, torch.zeros_like(a), torch.zeros_like(a)], dim=-1)
    vb = torch.stack([b * torch.cos(gamma), b * torch.sin(gamma), torch.zeros_like(b)], dim=-1)

    cx = c * torch.cos(beta)
    cy = c * (torch.cos(alpha) - torch.cos(beta) * torch.cos(gamma)) / torch.clamp(torch.sin(gamma), min=1e-6)
    cz_sq = torch.clamp(c**2 - cx**2 - cy**2, min=1e-8)
    vz = torch.stack([cx, cy, torch.sqrt(cz_sq)], dim=-1)

    return torch.stack([va, vb, vz], dim=1)


def safe_structure_density(structure: Structure) -> float:
    try:
        volume = float(structure.volume)
        if (not np.isfinite(volume)) or volume <= 1e-8:
            return float("nan")
        density = float(structure.density)
        return density if np.isfinite(density) else float("nan")
    except Exception:
        return float("nan")

def safe_spacegroup_info(structure: Structure):
    try:
        sga = SpacegroupAnalyzer(structure, symprec=0.1)
        return int(sga.get_space_group_number()), str(sga.get_space_group_symbol())
    except Exception:
        return np.nan, "unknown"

def safe_min_pair_distance(structure: Structure) -> float:
    try:
        dm = np.asarray(structure.distance_matrix, dtype=float)
    except Exception:
        return float("nan")
    if dm.ndim != 2 or dm.shape[0] == 0:
        return float("nan")
    mask = np.isfinite(dm) & (dm > 1e-8)
    if not mask.any():
        return float("nan")
    return float(dm[mask].min())


def lightweight_validity_dict(structure: Structure) -> dict:
    try:
        num_atoms = int(len(structure))
    except Exception:
        num_atoms = 0

    try:
        volume = float(structure.volume)
    except Exception:
        volume = float("nan")

    density = safe_structure_density(structure)
    min_pair_distance = safe_min_pair_distance(structure)

    try:
        lengths = np.asarray(structure.lattice.abc, dtype=float)
        angles = np.asarray(structure.lattice.angles, dtype=float)
    except Exception:
        lengths = np.array([np.nan, np.nan, np.nan], dtype=float)
        angles = np.array([np.nan, np.nan, np.nan], dtype=float)

    valid_volume = bool(np.isfinite(volume) and volume > 1e-3)
    valid_density = bool(np.isfinite(density) and 0.2 <= density <= 25.0)
    distance_ok = bool(np.isfinite(min_pair_distance) and min_pair_distance >= 0.6)
    lengths_ok = bool(np.all(np.isfinite(lengths)) and np.all((lengths >= 2.0) & (lengths <= 25.0)))
    angles_ok = bool(np.all(np.isfinite(angles)) and np.all((angles >= 20.0) & (angles <= 160.0)))
    atom_count_ok = bool(num_atoms >= 2)

    reasons = []
    if not atom_count_ok:
        reasons.append("too few atoms")
    if not valid_volume:
        reasons.append("bad cell volume")
    if not valid_density:
        reasons.append("bad density")
    if not distance_ok:
        reasons.append("atoms too close")
    if not lengths_ok:
        reasons.append("bad lattice lengths")
    if not angles_ok:
        reasons.append("bad lattice angles")

    volume_per_atom = float(volume / max(num_atoms, 1)) if np.isfinite(volume) else float("nan")
    lightweight_valid = len(reasons) == 0

    return {
        "min_pair_distance": min_pair_distance,
        "volume_per_atom": volume_per_atom,
        "valid_volume": valid_volume,
        "valid_density": valid_density,
        "distance_ok": distance_ok,
        "lattice_lengths_ok": lengths_ok,
        "lattice_angles_ok": angles_ok,
        "lightweight_valid": lightweight_valid,
        "failure_reason": "ok" if lightweight_valid else "; ".join(reasons),
    }


def summarize_structures(structures):
    rows = []
    for idx, s in enumerate(structures):
        try:
            formula = s.composition.reduced_formula
        except Exception:
            formula = "INVALID"

        try:
            volume = float(s.volume)
        except Exception:
            volume = float("nan")

        density = safe_structure_density(s)
        sg_num, sg_symbol = safe_spacegroup_info(s)
        validity = lightweight_validity_dict(s)
        rows.append(
            {
                "sample_id": idx,
                "formula": formula,
                "num_atoms": len(s),
                "density": density,
                "volume": volume,
                "spacegroup_number": sg_num,
                "spacegroup_symbol": sg_symbol,
                **validity,
            }
        )
    return pd.DataFrame(rows)


def validity_report(summary_df: pd.DataFrame, label: str) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(
            [{
                "label": label,
                "n_samples": 0,
                "valid_count": 0,
                "valid_fraction": float("nan"),
                "median_density": float("nan"),
                "median_min_pair_distance": float("nan"),
                "most_common_issue": "no samples",
            }]
        )

    invalid_df = summary_df.loc[~summary_df["lightweight_valid"]]
    if invalid_df.empty:
        most_common_issue = "all passed"
    else:
        most_common_issue = invalid_df["failure_reason"].value_counts().idxmax()

    return pd.DataFrame(
        [{
            "label": label,
            "n_samples": int(len(summary_df)),
            "valid_count": int(summary_df["lightweight_valid"].sum()),
            "valid_fraction": float(summary_df["lightweight_valid"].mean()),
            "median_density": float(summary_df["density"].median()),
            "median_min_pair_distance": float(summary_df["min_pair_distance"].median()),
            "most_common_issue": most_common_issue,
        }]
    )


__all__ = [
    "BUNDLED_MP_FALLBACK_CONFIG",
    "BUNDLED_MP_FALLBACK_PATH",
    "CHEMISTRY_SCOPE_TO_NUM_ELEMENTS",
    "DEFAULT_MP_CACHE_DIR",
    "GEOM_COS_CLAMP",
    "GEOM_LOG_LENGTH_CLAMP",
    "GEOM_MAX_DIST",
    "MAX_DECODED_ANGLE",
    "MAX_DECODED_LENGTH",
    "MIN_DECODED_ANGLE",
    "MIN_DECODED_LENGTH",
    "MIN_DECODED_VOLUME",
    "VAL_FRACTION",
    "build_general_mp_query_from_config",
    "build_general_mp_rest_params",
    "build_synthetic_records",
    "cache_payload_is_real_mp",
    "cache_payload_source_label",
    "canonicalize_config_for_cache",
    "clamp_lattice_features_norm_torch",
    "curate_general_summary_docs",
    "dataset_cache_path_from_config",
    "denormalize_lattice_features_torch",
    "ensure_record_feature_fields",
    "features_to_lattice",
    "fetch_general_summary_docs",
    "fetch_general_summary_docs_via_rest",
    "filter_records_for_config",
    "is_teaching_friendly_structure",
    "lattice_matrix_from_features_torch",
    "lattice_to_features",
    "load_bundled_fallback_records",
    "make_general_dataset_config",
    "mp_doc_get",
    "normalized_structure_anonymous_formula",
    "probe_materials_project_connection",
    "select_evenly_spaced_records",
    "set_lattice_feature_stats",
    "structure_from_mp_doc",
    "wrap_centered_np",
    "wrap_centered_torch",
    "summarize_structures",
    "validity_report",
    "lightweight_validity_dict",
    "safe_structure_density",
    "safe_spacegroup_info",
    "safe_min_pair_distance",
]

