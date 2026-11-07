"""Procedural texture generators for photorealistic MuJoCo house rendering.

Generates seamless high-resolution textures for:
- Oak parquet flooring (living room & bedroom)
- Ceramic slate kitchen tiles (kitchen)
- Soft woven loop carpet (study)
- Wall plaster / architectural eggshell paint
- Natural wood grain for furniture, tables, and cabinets
- Polished Carrara marble for kitchen countertops
- Upholstery fabric for sofas and cushions
- Brushed stainless steel for appliances, drawer handles, and push plates
- Modern geometric living room area rug
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
TEXTURES_DIR = HERE / "assets" / "textures"


def make_parquet_texture(size: int = 512) -> Image.Image:
    """Generate realistic warm oak plank flooring with subtle grain and seams."""
    arr = np.zeros((size, size, 3), dtype=np.float32)
    base_color = np.array([195.0, 155.0, 108.0], dtype=np.float32)  # warm oak
    n_planks_y = 8
    plank_h = size // n_planks_y
    n_planks_x = 4
    plank_w = size // n_planks_x

    rng = np.random.default_rng(42)
    for row in range(n_planks_y):
        y0, y1 = row * plank_h, (row + 1) * plank_h
        offset = (row * (plank_w // 2)) % plank_w
        for col in range(-1, n_planks_x + 1):
            x0 = col * plank_w + offset
            x1 = x0 + plank_w
            # Tone variation per plank
            tone_factor = rng.uniform(0.90, 1.10)
            plank_color = np.clip(base_color * tone_factor, 0, 255)
            # Clip bounds
            bx0, bx1 = max(0, x0), min(size, x1)
            if bx0 < bx1:
                arr[y0:y1, bx0:bx1] = plank_color

    # Add fine wood grain stripes (horizontal)
    y_coords = np.arange(size)[:, None]
    grain = np.sin(y_coords * 0.8) * 8.0 + rng.normal(0, 4.0, size=(size, size))
    arr += grain[..., None]

    # Draw dark seam lines between planks
    for row in range(n_planks_y + 1):
        y = min(row * plank_h, size - 1)
        arr[max(0, y - 1):min(size, y + 2), :] *= 0.65  # groove shadow

    for row in range(n_planks_y):
        y0, y1 = row * plank_h, (row + 1) * plank_h
        offset = (row * (plank_w // 2)) % plank_w
        for col in range(-1, n_planks_x + 2):
            x = col * plank_w + offset
            if 0 <= x < size:
                arr[y0:y1, max(0, x - 1):min(size, x + 2)] *= 0.65

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def make_tile_texture(size: int = 512) -> Image.Image:
    """Generate modern ceramic porcelain tiles with neat grout lines."""
    arr = np.zeros((size, size, 3), dtype=np.float32)
    tile_color = np.array([215.0, 218.0, 222.0], dtype=np.float32)  # clean slate/porcelain
    grout_color = np.array([120.0, 122.0, 125.0], dtype=np.float32)

    n_tiles = 4
    step = size // n_tiles
    grout_w = 3

    arr[:] = tile_color

    # Subtle stone noise
    rng = np.random.default_rng(101)
    noise = rng.normal(0, 3.5, size=(size, size, 1))
    arr += noise

    # Grout lines
    for i in range(n_tiles + 1):
        coord = min(i * step, size - 1)
        arr[max(0, coord - grout_w):min(size, coord + grout_w), :] = grout_color
        arr[:, max(0, coord - grout_w):min(size, coord + grout_w)] = grout_color

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def make_carpet_texture(size: int = 512) -> Image.Image:
    """Generate soft woven neutral loop-pile carpet texture."""
    base = np.array([178.0, 185.0, 172.0], dtype=np.float32)  # subtle sage/taupe
    rng = np.random.default_rng(202)
    # Loop weave pattern
    x = np.arange(size)[None, :]
    y = np.arange(size)[:, None]
    weave = (np.sin(x * 0.5) * np.cos(y * 0.5) * 12.0).astype(np.float32)
    fiber_noise = rng.normal(0, 6.0, size=(size, size)).astype(np.float32)
    arr = base + (weave + fiber_noise)[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def make_wood_grain_texture(size: int = 512) -> Image.Image:
    """Generate rich walnut / natural oak grain for furniture."""
    base = np.array([145.0, 98.0, 58.0], dtype=np.float32)
    rng = np.random.default_rng(303)
    y = np.arange(size)[:, None]
    x = np.arange(size)[None, :]
    rings = np.sin(x * 0.08 + np.sin(y * 0.03) * 3.0) * 22.0
    fine_grain = rng.normal(0, 5.0, size=(size, size))
    arr = base + (rings + fine_grain)[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def make_marble_texture(size: int = 512) -> Image.Image:
    """Generate elegant white polished quartz / Carrara marble with grey veins."""
    base = np.array([242.0, 244.0, 246.0], dtype=np.float32)
    x = np.linspace(0, 4 * math.pi, size)[None, :]
    y = np.linspace(0, 4 * math.pi, size)[:, None]
    rng = np.random.default_rng(404)

    # Multi-frequency turbulent marble veining
    v1 = np.sin(x * 2.0 + np.sin(y * 3.0) * 1.5)
    v2 = np.sin(y * 4.0 + np.cos(x * 2.5) * 1.2)
    veins = np.abs(v1 + 0.6 * v2)
    vein_mask = np.exp(-veins * 3.5)  # narrow sharp veins
    vein_color = np.array([120.0, 130.0, 140.0], dtype=np.float32)

    arr = base * (1.0 - 0.45 * vein_mask[..., None]) + vein_color * (0.45 * vein_mask[..., None])
    arr += rng.normal(0, 2.0, size=(size, size, 1))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def make_fabric_texture(size: int = 512) -> Image.Image:
    """Generate textured woven sofa upholstery fabric (slate blue/charcoal)."""
    base = np.array([90.0, 115.0, 145.0], dtype=np.float32)  # contemporary muted blue
    x = np.arange(size)[None, :]
    y = np.arange(size)[:, None]
    weave = (np.sin(x * 0.7) * 8.0 + np.sin(y * 0.7) * 8.0).astype(np.float32)
    rng = np.random.default_rng(505)
    noise = rng.normal(0, 5.0, size=(size, size)).astype(np.float32)
    arr = base + (weave + noise)[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def make_plaster_texture(size: int = 512) -> Image.Image:
    """Generate subtle eggshell wall plaster with architectural matte finish."""
    base = np.array([238.0, 235.0, 230.0], dtype=np.float32)  # warm architectural linen
    rng = np.random.default_rng(606)
    noise = rng.normal(0, 3.0, size=(size, size, 1))
    arr = base + noise
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def make_brushed_metal_texture(size: int = 512) -> Image.Image:
    """Generate directional brushed stainless steel texture."""
    base = np.array([205.0, 208.0, 212.0], dtype=np.float32)
    rng = np.random.default_rng(707)
    # Horizontal streaks
    streaks = rng.normal(0, 12.0, size=(size, 1))
    streaks = np.repeat(streaks, size, axis=1)
    fine = rng.normal(0, 3.0, size=(size, size))
    arr = base + (streaks + fine)[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def make_rug_texture(size: int = 512) -> Image.Image:
    """Generate contemporary Scandinavian geometric diamond area rug."""
    base = np.array([235.0, 230.0, 222.0], dtype=np.float32)
    pattern_color = np.array([75.0, 80.0, 92.0], dtype=np.float32)
    arr = np.repeat(base[None, None, :], size, axis=0)
    arr = np.repeat(arr, size, axis=1)

    # Diamond grid
    cell = 64
    x = np.arange(size)[None, :]
    y = np.arange(size)[:, None]
    dist = (np.abs((x % cell) - cell / 2) + np.abs((y % cell) - cell / 2))
    line_mask = (dist < 4) | ((dist >= cell / 2 - 3) & (dist <= cell / 2 + 1))
    arr[line_mask] = pattern_color

    rng = np.random.default_rng(808)
    arr += rng.normal(0, 4.0, size=(size, size, 1))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def ensure_texture_assets(out_dir: Path | None = None) -> dict[str, Path]:
    """Generate and save all photorealistic procedural textures."""
    target_dir = Path(out_dir) if out_dir else TEXTURES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    generators = {
        "floor_parquet.png": make_parquet_texture,
        "floor_tile.png": make_tile_texture,
        "floor_carpet.png": make_carpet_texture,
        "wood_grain.png": make_wood_grain_texture,
        "marble_counter.png": make_marble_texture,
        "fabric_sofa.png": make_fabric_texture,
        "wall_plaster.png": make_plaster_texture,
        "metal_brushed.png": make_brushed_metal_texture,
        "rug_geometric.png": make_rug_texture,
    }

    results = {}
    for filename, fn in generators.items():
        path = target_dir / filename
        img = fn(size=512)
        img.save(path)
        results[filename] = path

    return results


if __name__ == "__main__":
    generated = ensure_texture_assets()
    print(f"Generated {len(generated)} textures in {TEXTURES_DIR}")
