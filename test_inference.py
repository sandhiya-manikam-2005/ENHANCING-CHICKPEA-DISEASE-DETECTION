"""
Quick Test Script — Test the trained model with a single image
Usage:
    python test_inference.py                    # test with a synthetic image
    python test_inference.py path/to/image.jpg  # test with your own image
"""

import sys
import os
import io
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
from model import get_model, DISEASE_CLASSES, CLASS_DESCRIPTIONS, CLASS_TREATMENTS
from preprocessing import preprocess_for_inference, IMAGE_SIZE

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model.pth")


def run_test(image_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load model
    if not os.path.exists(MODEL_PATH):
        print("❌ No trained model found! Run train.py first.")
        return

    ckpt  = torch.load(MODEL_PATH, map_location=device)
    model = get_model(num_classes=4, image_size=IMAGE_SIZE)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    print(f"✓ Model loaded (val_acc={ckpt.get('val_acc', 0):.2%})")

    # Get image bytes
    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        print(f"✓ Loaded image: {image_path}")
    else:
        # Generate a synthetic test image
        from dataset_generator import GENERATORS
        import random
        cls = random.choice(list(GENERATORS.keys()))
        pil = GENERATORS[cls]()
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        print(f"✓ Using synthetic {cls} image for testing")

    # Preprocess + Inference
    tensor, pil_proc, pil_orig = preprocess_for_inference(image_bytes)
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)[0]
        pred   = probs.argmax().item()

    # Display results
    pred_label = DISEASE_CLASSES[pred]
    confidence = probs[pred].item() * 100

    print("\n" + "═" * 50)
    print(f"  PREDICTION: {pred_label}")
    print(f"  CONFIDENCE: {confidence:.1f}%")
    print("═" * 50)
    print(f"\n  {CLASS_DESCRIPTIONS[pred_label]}")

    print("\n  Class Probabilities:")
    sorted_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    for i, p in sorted_probs:
        bar  = "█" * int(p.item() * 30)
        mark = " ◄" if i == pred else ""
        print(f"    {DISEASE_CLASSES[i]:10s} │ {bar:<30} {p.item()*100:5.1f}%{mark}")

    print(f"\n  Recommended Treatments:")
    for t in CLASS_TREATMENTS[pred_label]:
        print(f"    • {t}")

    print()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    run_test(path)
