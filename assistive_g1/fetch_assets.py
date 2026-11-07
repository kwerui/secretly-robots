"""Download the G1 model, meshes and GR00T policies (binary files, kept out of git).

Source: NVlabs/GR00T-WholeBodyControl (Apache-2.0 code, NVIDIA Open Model License for the
policy weights), decoupled_wbc/sim2mujoco/resources/robots/g1. Files are Git-LFS objects,
fetched from GitHub's media host so you don't need git-lfs or a Hugging Face login.

    python fetch_assets.py
"""

from __future__ import annotations

import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = ("https://media.githubusercontent.com/media/NVlabs/GR00T-WholeBodyControl/main/"
        "decoupled_wbc/sim2mujoco/resources/robots/g1")
RAW = ("https://raw.githubusercontent.com/NVlabs/GR00T-WholeBodyControl/main/"
       "decoupled_wbc/sim2mujoco/resources/robots/g1")
HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"


def get(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 1000:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as r:
        data = r.read()
    if data.startswith(b"version https://git-lfs"):
        raise RuntimeError(f"got an LFS pointer instead of the file for {url}")
    dest.write_bytes(data)


def main() -> None:
    xml_dest = ASSETS / "g1" / "g1.xml"
    if not xml_dest.exists():
        for base in (RAW, BASE):
            try:
                get(f"{base}/g1_gear_wbc.xml", xml_dest)
                break
            except Exception as e:  # noqa: BLE001
                print("retrying xml:", e)
    meshes = sorted(set(re.findall(r'file="([^"]+)"', xml_dest.read_text())))
    jobs = [(f"{BASE}/meshes/{m}", ASSETS / "g1" / "meshes" / m) for m in meshes]
    jobs += [(f"{BASE}/policy/{p}", ASSETS / "policy" / p)
             for p in ("GR00T-WholeBodyControl-Balance.onnx", "GR00T-WholeBodyControl-Walk.onnx")]
    with ThreadPoolExecutor(8) as ex:
        for (url, dest), err in zip(jobs, ex.map(lambda j: _safe(*j), jobs)):
            print(("OK  " if err is None else "ERR ") + dest.relative_to(HERE).as_posix() + ("" if err is None else f"  {err}"))
    missing = [d for _, d in jobs if not d.exists()]
    if missing:
        sys.exit(f"{len(missing)} files missing")
    print("assets ready")


def _safe(url: str, dest: Path):
    try:
        get(url, dest)
        return None
    except Exception as e:  # noqa: BLE001
        return e


if __name__ == "__main__":
    main()
