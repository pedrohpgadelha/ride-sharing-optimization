"""
File system and data manipulation utilities.
"""

import csv
import json
import tempfile
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
import numpy as np
import yaml


def to_path(path: str | Path) -> Path:
    """
    Convert a string or Path to a Path object.
    """
    return path if isinstance(path, Path) else Path(path)


def ensure_dir(directory: str | Path) -> Path:
    """
    Ensure a directory exists, creating it if it doesn't.
    """
    directory_path = to_path(directory)
    directory_path.mkdir(parents=True, exist_ok=True)
    return directory_path


def ensure_parent_dir(path: str | Path) -> None:
    """
    Ensure a parent directory exists, creating it if it doesn't.
    """
    to_path(path).parent.mkdir(parents=True, exist_ok=True)


def now_utc_iso() -> str:
    """
    Get the current UTC time in ISO format.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_text(path: str | Path) -> str:
    """
    Read a text file and return its contents as a str.
    """
    return to_path(path).read_text(encoding="utf-8")


def write_text(path: str | Path, text: str) -> None:
    """
    Write a str to a text file.
    """
    path = to_path(path)
    ensure_parent_dir(path)
    _atomic_write_bytes(path, text.encode("utf-8"))


def read_yaml(path: str | Path) -> dict:
    """
    Read a YAML file and return its contents as a dict.
    """
    with to_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: str | Path, data: dict) -> None:
    """
    Write a dict to a YAML file.
    """
    path_ = to_path(path)
    ensure_parent_dir(path_)
    tmp = yaml.safe_dump(data, sort_keys=True, allow_unicode=True)
    _atomic_write_bytes(path_, tmp.encode("utf-8"))


def read_json(path: str | Path) -> Any:
    """
    Read a JSON file and return its contents as a dict.
    """
    with to_path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: str | Path, data: Any, *, indent: int = 2) -> None:
    """
    Write a dict to a JSON file.
    """
    path_ = to_path(path)
    ensure_parent_dir(path_)
    tmp = json.dumps(data, ensure_ascii=False, indent=indent)
    _atomic_write_bytes(path_, tmp.encode("utf-8"))


def write_csv_rows(
    path: str | Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    """
    Write a list of dicts to a CSV file.
    """
    path_ = to_path(path)
    ensure_parent_dir(path_)
    if not rows and not fieldnames:
        raise ValueError("rows is empty and fieldnames not provided")
    fns = fieldnames or list(rows[0].keys())
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        delete=False,
        dir=str(path_.parent)) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=fns)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fns})
        tmp_path = Path(tmp.name)
    tmp_path.replace(path_)


def save_npz(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    """
    Save a dictionary of numpy arrays to a .npz file.
    """
    path_ = to_path(path)
    ensure_parent_dir(path_)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(path_.parent)) as tmp:
        np.savez_compressed(tmp, **arrays)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path_)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """
    Load a .npz file and return its contents as a dict of numpy arrays.
    """
    with np.load(to_path(path)) as data:
        return {k: data[k] for k in data.files}


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """
    Compute the SHA-256 hash of a file.
    """
    hash = sha256()
    with to_path(path).open("rb") as file:
        while True:
            bytes = file.read(chunk_size)
            if not bytes:
                break
            hash.update(bytes)
    return hash.hexdigest()


def bytes_sha256(bytes: bytes) -> str:
    """
    Compute the SHA-256 hash of a bytes object.
    """
    return sha256(bytes).hexdigest()


def make_run_id(
    scenario: str,
    seed: int | str,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Make a run ID from a scenario, seed, and parameters of type
    scenario__seed<seed>__<param1><value1>__<param2><value2>... Used for storing
    solutions and metrics.
    """
    base = f"{_normalize_str(scenario)}__seed{seed}"
    if not params:
        return base
    kv = []
    for k in sorted(params.keys()):
        v = params[k]
        kv.append(f"{_normalize_str(k)}{_format_value(v)}")
    return base + "__" + "__".join(kv)


def write_manifest(path: str | Path, meta: dict[str, Any]) -> None:
    """
    Write a manifest to a JSON file with the current UTC time and metadata.
    """
    write_json(path, {"created_at_utc": now_utc_iso(), **meta})


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    """
    Write bytes to a temporary file and later replace it. Used for avoiding
    corromped files.
    """
    with tempfile.NamedTemporaryFile(
        "wb", delete=False, dir=str(path.parent)
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _normalize_str(s: Any) -> str:
    t = str(s)
    out = []
    for ch in t:
        if ch.isalnum():
            out.append(ch.lower())
        elif ch in ("-", "_"):
            out.append(ch)
        elif ch.isspace():
            out.append("_")
    return "".join(out).strip("_") or "x"


def _format_value(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6g}".rstrip("0").rstrip(".")
    if isinstance(v, bool):
        return "1" if v else "0"
    return str(v)
