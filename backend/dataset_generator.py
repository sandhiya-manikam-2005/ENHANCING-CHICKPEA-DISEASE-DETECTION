"""
Synthetic Dataset Generator for Chickpea Disease Detection
─────────────────────────────────────────────────────────
Generates realistic-looking synthetic leaf images for each disease class.
Used for training/testing when real images are not yet available.

Classes: Healthy | Blight | Rust | Wilt
"""

import os
import random
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import json


CLASSES   = ["Healthy", "Blight", "Rust", "Wilt"]
IMG_SIZE  = 256
N_PER_CLASS = 100   # images per class (increase for better training)

random.seed(42)
np.random.seed(42)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def draw_leaf_shape(draw: ImageDraw.ImageDraw, cx: int, cy: int,
                    rx: int = 90, ry: int = 55, color=(50, 140, 50)):
    """Draw an elliptical leaf on the canvas."""
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=color)
    # Midrib
    draw.line([(cx - rx + 10, cy), (cx + rx - 10, cy)], fill=(30, 100, 30), width=2)
    # Veins
    for angle in [-30, -15, 15, 30]:
        rad  = math.radians(angle)
        x2   = int(cx + rx * 0.6 * math.cos(rad))
        y2   = int(cy + ry * 0.8 * math.sin(rad))
        draw.line([(cx, cy), (x2, y2)], fill=(30, 100, 30), width=1)


def add_noise(img: Image.Image, intensity: float = 0.05) -> Image.Image:
    arr   = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, intensity * 255, arr.shape)
    arr   = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ─── Per-class generators ─────────────────────────────────────────────────────

def generate_healthy() -> Image.Image:
    """Bright green, uniform, no lesions."""
    img  = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(210, 230, 190))
    draw = ImageDraw.Draw(img)
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    green = (
        random.randint(50, 80),
        random.randint(130, 170),
        random.randint(40, 70)
    )
    draw_leaf_shape(draw, cx, cy, color=green)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    img = add_noise(img, 0.03)
    return img


def generate_blight() -> Image.Image:
    """
    Ascochyta Blight: water-soaked, tan/brown irregular lesions with
    dark edges on a pale-green leaf.
    """
    img  = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(210, 220, 195))
    draw = ImageDraw.Draw(img)
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    draw_leaf_shape(draw, cx, cy, color=(100, 150, 80))

    n_spots = random.randint(4, 10)
    for _ in range(n_spots):
        sx = random.randint(cx - 75, cx + 75)
        sy = random.randint(cy - 40, cy + 40)
        rx = random.randint(8, 22)
        ry = random.randint(6, 16)
        # Tan lesion interior
        draw.ellipse([sx - rx, sy - ry, sx + rx, sy + ry],
                     fill=(160, 120, 70))
        # Dark brown border
        draw.ellipse([sx - rx - 2, sy - ry - 2, sx + rx + 2, sy + ry + 2],
                     outline=(80, 50, 20), width=2)

    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
    img = add_noise(img, 0.04)
    return img


def generate_rust() -> Image.Image:
    """
    Chickpea Rust: orange/reddish-brown small pustules on yellow-green leaf.
    """
    img  = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(215, 225, 185))
    draw = ImageDraw.Draw(img)
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    draw_leaf_shape(draw, cx, cy, color=(130, 155, 60))

    n_pustules = random.randint(15, 35)
    for _ in range(n_pustules):
        px = random.randint(cx - 80, cx + 80)
        py = random.randint(cy - 45, cy + 45)
        r  = random.randint(3, 8)
        rust_color = (
            random.randint(170, 200),
            random.randint(80, 110),
            random.randint(20, 50)
        )
        draw.ellipse([px - r, py - r, px + r, py + r], fill=rust_color)

    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    img = add_noise(img, 0.04)
    return img


def generate_wilt() -> Image.Image:
    """
    Fusarium Wilt: yellowing, browning at margins, curled/droopy appearance.
    """
    img  = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(220, 215, 175))
    draw = ImageDraw.Draw(img)
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2

    # Yellowish leaf
    draw_leaf_shape(draw, cx, cy, color=(160, 150, 50))

    # Brown necrotic margins
    for _ in range(6):
        mx = random.randint(cx - 90, cx + 90)
        my = random.randint(cy - 55, cy + 55)
        r  = random.randint(10, 30)
        draw.ellipse([mx - r, my - r, mx + r, my + r],
                     fill=(120, 80, 30))

    # Random pale yellow streaks (vascular discoloration)
    for _ in range(4):
        x1 = random.randint(cx - 60, cx + 60)
        y1 = random.randint(cy - 30, cy + 30)
        x2 = x1 + random.randint(-20, 20)
        y2 = y1 + random.randint(-20, 20)
        draw.line([(x1, y1), (x2, y2)], fill=(200, 190, 100), width=2)

    img = img.filter(ImageFilter.GaussianBlur(radius=0.9))
    img = add_noise(img, 0.05)
    return img


GENERATORS = {
    "Healthy": generate_healthy,
    "Blight":  generate_blight,
    "Rust":    generate_rust,
    "Wilt":    generate_wilt,
}


# ─── Dataset builder ──────────────────────────────────────────────────────────

def build_dataset(base_dir: str, n_per_class: int = N_PER_CLASS,
                  splits: dict = None):
    """
    Generate synthetic images and save them in:
        base_dir/
            train/
                Healthy/ Blight/ Rust/ Wilt/
            val/
                Healthy/ Blight/ Rust/ Wilt/
            test/
                Healthy/ Blight/ Rust/ Wilt/

    splits – proportions dict, e.g. {'train': 0.7, 'val': 0.15, 'test': 0.15}
    """
    if splits is None:
        splits = {"train": 0.70, "val": 0.15, "test": 0.15}

    assert abs(sum(splits.values()) - 1.0) < 1e-6, "Splits must sum to 1"

    summary = {}

    for cls in CLASSES:
        images   = [GENERATORS[cls]() for _ in range(n_per_class)]
        indices  = list(range(n_per_class))
        random.shuffle(indices)

        cutoffs  = {}
        start    = 0
        for split_name, ratio in splits.items():
            end = start + int(ratio * n_per_class)
            cutoffs[split_name] = indices[start:end]
            start = end
        # Edge: give remainder to last split
        last_key = list(splits.keys())[-1]
        cutoffs[last_key].extend(indices[start:])

        summary[cls] = {}
        for split_name, idxs in cutoffs.items():
            split_dir = os.path.join(base_dir, split_name, cls)
            os.makedirs(split_dir, exist_ok=True)
            for i, idx in enumerate(idxs):
                fname = f"{cls.lower()}_{i:04d}.png"
                images[idx].save(os.path.join(split_dir, fname))
            summary[cls][split_name] = len(idxs)
            print(f"  [{split_name:5s}] {cls:8s}: {len(idxs)} images → {split_dir}")

    # Save class info
    meta = {
        "classes":       CLASSES,
        "num_classes":   len(CLASSES),
        "class_to_idx":  {c: i for i, c in enumerate(CLASSES)},
        "splits_summary": summary
    }
    with open(os.path.join(base_dir, "dataset_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Dataset saved to: {base_dir}")
    return meta


if __name__ == "__main__":
    import sys
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    n           = int(sys.argv[2]) if len(sys.argv) > 2 else N_PER_CLASS
    print(f"Generating synthetic chickpea dataset ({n} images/class)…")
    meta = build_dataset(dataset_dir, n_per_class=n)
    print("\nDataset summary:")
    for cls, splits in meta["splits_summary"].items():
        total = sum(splits.values())
        print(f"  {cls}: {total} total ({splits})")
