"""
Image Preprocessing Pipeline
Implements Algorithm 1 from the research paper:
- Noise Removal (Gaussian Filter)
- Contrast Enhancement (Histogram Equalization / CLAHE)
- Segmentation (Otsu's Thresholding)
- Normalization [0, 1]
"""

import cv2
import numpy as np
from PIL import Image
import io
import base64
import torchvision.transforms as transforms


# ─── Constants ────────────────────────────────────────────────────────────────
IMAGE_SIZE   = 128   # resize target (pixels)
MEAN         = [0.485, 0.456, 0.406]   # ImageNet-style normalization
STD          = [0.229, 0.224, 0.225]


# ─── Step-by-step preprocessing (mirrors the paper's Algorithm 1) ─────────────

def remove_noise(img_bgr: np.ndarray) -> np.ndarray:
    """
    Step 2 – Noise Removal
    Apply Gaussian Filter: G(x,y) = 1/(2πσ²) · exp(-(x²+y²)/(2σ²))
    """
    return cv2.GaussianBlur(img_bgr, (5, 5), sigmaX=1.0)


def enhance_contrast(img_bgr: np.ndarray) -> np.ndarray:
    """
    Step 3 – Contrast Enhancement
    Use CLAHE (Contrast Limited Adaptive Histogram Equalization)
    on the L-channel of LAB space — better than global HE for leaf images.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq  = clahe.apply(l)

    merged = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def segment_leaf(img_bgr: np.ndarray):
    """
    Step 4 – Segmentation
    Apply Otsu's Thresholding to isolate diseased regions from healthy areas.
    σ²_B(t) = ω₁(t)·ω₂(t)·[μ₁(t) − μ₂(t)]²  →  maximise inter-class variance
    Returns the segmented image and the binary mask.
    """
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask     = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask     = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    # Apply mask to original image
    segmented = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    return segmented, mask


def normalize_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    Step 5 – Normalization
    Scale pixel values to [0, 1]: I_norm = (I − I_min) / (I_max − I_min)
    """
    img_float = img_bgr.astype(np.float32)
    mn, mx    = img_float.min(), img_float.max()
    if mx - mn > 0:
        img_float = (img_float - mn) / (mx - mn)
    return (img_float * 255).astype(np.uint8)


# ─── Full pipeline ─────────────────────────────────────────────────────────────

def preprocess_image_cv2(img_bgr: np.ndarray) -> np.ndarray:
    """
    Complete preprocessing pipeline (Algorithm 1):
    Raw image → denoised → contrast-enhanced → segmented → normalized
    Returns a BGR uint8 image ready for further processing.
    """
    denoised    = remove_noise(img_bgr)
    enhanced    = enhance_contrast(denoised)
    segmented, _= segment_leaf(enhanced)
    normalized  = normalize_image(segmented)
    resized     = cv2.resize(normalized, (IMAGE_SIZE, IMAGE_SIZE))
    return resized


def get_inference_transform():
    """
    Returns torchvision transform for model inference.
    Converts preprocessed image → normalised tensor.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])


def get_training_transform():
    """
    Returns torchvision transform with data augmentation for training.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 20, IMAGE_SIZE + 20)),
        transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])


# ─── PIL / bytes helpers ───────────────────────────────────────────────────────

def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL RGB image → OpenCV BGR array."""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def cv2_to_pil(img_bgr: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR array → PIL RGB image."""
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    """Convert raw bytes → PIL image."""
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def pil_to_base64(pil_img: Image.Image, fmt: str = "PNG") -> str:
    """Convert PIL image → base64 string (for API responses)."""
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def preprocess_for_inference(image_bytes: bytes):
    """
    Full end-to-end preprocessing for a single uploaded image.
    Returns:
        tensor       – shape (1, 3, H, W) ready for model input
        preprocessed – PIL image of the preprocessed leaf (for display)
        original     – PIL image of the original upload
    """
    import torch

    pil_orig  = bytes_to_pil(image_bytes)
    bgr_orig  = pil_to_cv2(pil_orig)

    # Algorithm 1 preprocessing
    bgr_proc  = preprocess_image_cv2(bgr_orig)
    pil_proc  = cv2_to_pil(bgr_proc)

    # Convert to model tensor
    transform = get_inference_transform()
    tensor    = transform(pil_proc).unsqueeze(0)   # (1, 3, H, W)

    return tensor, pil_proc, pil_orig


if __name__ == "__main__":
    # Smoke-test with a synthetic image
    dummy_bgr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    result    = preprocess_image_cv2(dummy_bgr)
    print(f"Preprocessing output shape: {result.shape}")  # (128, 128, 3)
    print("Preprocessing pipeline OK ✓")
