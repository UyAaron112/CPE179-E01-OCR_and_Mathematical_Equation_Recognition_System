# make_noisy.py
# Generates noisy variants of captured math equation images
# and evaluates EasyOCR baseline performance on each variant.
# Designed for Raspberry Pi — images sourced from USB/SD card.

import cv2
import numpy as np
import os
import easyocr

# ─────────────────────────────────────────
# CONFIG — adjust these to match your setup
# ─────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
INPUT_DIR   = os.path.join(PROJECT_DIR, "input_images")

# Output directory for noisy variants
PROJECT_DIR = os.path.dirname(BASE_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, "noisy_output")

# Ground truth LaTeX strings for each input image
# Key = filename without extension, Value = expected LaTeX
GROUND_TRUTH = {
    "img1": r"x̄ = (x_1 + x_2 + ... + x_n) / n = Σ(i=1 to n) x_i / n",
    "img2": r"μ = Σ(i=1 to N) x_i * f(x_i) = Σ(i=1 to N) x_i / N",
    "img3": r"s^2 = Σ(i=1 to n)(x_i - x̄)^2 / (n - 1)",
    "img4": r"Σ(i=1 to 8)(x_i - x̄)^2 = 1.60",
    "img5": r"s = √0.2286 = 0.48 pounds",
    "img6": r"s^2 = Σ(i=1 to n)(x_i - x̄)^2 / (n - 1)",
    "img7": r"Σ(i=1 to n)(x_i^2 + x̄^2 - 2*x̄*x_i) / (n - 1)",
    "img8": r"o^2 = Σ(i=1 to N)(x_i - μ)^2 / N",
    "img9": r"n < 8 or 10",
    "img10": r"r_xy = Σ(i=1 to n) y_i(x_i - x̄) / [ Σ(i=1 to n)(y_i - ȳ)^2 * Σ(i=1 to n)(x_i - x̄)^2 ]^(1/2",
    "img11": r"(j - 0.5) / n = P(Z ≤ z_j) = Φ",
    "img12": r"(j - 0.5) / n = 0.05, Φ(z_j)",
    "img13": r"Σ(i=1 to n) (x_i - a)^2",
    "img14": r"y_i = a + b*x_i",
    "img15": r"(j - 0.5) / 10",
}

# ─────────────────────────────────────────
# NOISE FUNCTIONS
# ─────────────────────────────────────────

def salt_pepper(image, prob=0.02):
    """Add salt-and-pepper noise to an image."""
    noisy = image.copy()
    h, w = noisy.shape[:2]

    num_salt = int(prob * h * w)
    sy = np.random.randint(0, h, num_salt)
    sx = np.random.randint(0, w, num_salt)
    noisy[sy, sx] = 255

    num_pepper = int(prob * h * w)
    py = np.random.randint(0, h, num_pepper)
    px = np.random.randint(0, w, num_pepper)
    noisy[py, px] = 0

    return noisy


def apply_noise_variants(img, base, output_dir):
    """Apply Gaussian blur, salt-and-pepper, and low contrast to an image."""
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite(os.path.join(output_dir, f"{base}_blur.png"), blur)

    sp = salt_pepper(img)
    cv2.imwrite(os.path.join(output_dir, f"{base}_sp.png"), sp)

    low = cv2.convertScaleAbs(img, alpha=0.5, beta=0)
    cv2.imwrite(os.path.join(output_dir, f"{base}_contrast.png"), low)

    print(f"  Saved: {base}_blur.png | {base}_sp.png | {base}_contrast.png")


# ─────────────────────────────────────────
# PREPROCESSING — applied before OCR
# Mimics what the Pi will do on real captures
# ─────────────────────────────────────────

def preprocess_for_ocr(image):
    """
    Convert to grayscale, denoise, and threshold.
    This improves EasyOCR accuracy on equation images.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=30)
    # Adaptive threshold works better than global for uneven lighting
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return thresh


# ─────────────────────────────────────────
# OCR HELPERS
# ─────────────────────────────────────────

def run_ocr(reader, image_path):
    """Run EasyOCR on an image file. Returns (text, confidence)."""
    # Load and preprocess before passing to OCR
    img = cv2.imread(image_path)
    if img is None:
        return "", 0.0
    processed = preprocess_for_ocr(img)
    results = reader.readtext(processed)
    if not results:
        return "", 0.0
    text = " ".join(r[1] for r in results)
    conf = float(np.mean([r[2] for r in results]))
    return text, round(conf, 3)


def classify_error(prediction, ground_truth, confidence):
    """Classify the type of OCR error relative to ground truth."""
    if not prediction or confidence == 0.0:
        return "Detection failure"
    if prediction.strip() == ground_truth.strip():
        return "None"
    if len(prediction) < len(ground_truth) * 0.6:
        return "Missed characters"
    return "Character substitution"


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    print("=" * 60)
    print(" Math Equation Image — Noise Generation & OCR Evaluation")
    print("=" * 60)

    # Collect input images from INPUT_DIR matching GROUND_TRUTH keys
    images = []
    for key in GROUND_TRUTH:
        path = os.path.join(INPUT_DIR, f"{key}.jpg")
        if os.path.exists(path):
            images.append((key, path))
        else:
            print(f"[WARNING] Image not found: {path}. Skipping.")

    if not images:
        print("[ERROR] No valid input images found. Check INPUT_DIR path.")
        return

    print(f"\nGenerating noisy variants for {len(images)} image(s)...\n")
    for base, path in images:
        img = cv2.imread(path)
        if img is None:
            print(f"[WARNING] Cannot read {path}. Skipping.")
            continue
        print(f"Processing {base}.png:")
        apply_noise_variants(img, base, OUTPUT_DIR)

    print(f"\nAll noisy images saved to: {OUTPUT_DIR}")

    # Build full evaluation list: clean + 3 noisy variants per image
    all_images = []
    for base, clean_path in images:
        gt = GROUND_TRUTH[base]
        all_images += [
            ("Clean Captured",  clean_path,                                        gt),
            ("Gaussian Blur",   os.path.join(OUTPUT_DIR, f"{base}_blur.png"),     gt),
            ("Salt-and-Pepper", os.path.join(OUTPUT_DIR, f"{base}_sp.png"),       gt),
            ("Low Contrast",    os.path.join(OUTPUT_DIR, f"{base}_contrast.png"), gt),
        ]

    print("\nLoading EasyOCR model (this may take a moment on Raspberry Pi)...")
    # gpu=False — Raspberry Pi has no CUDA GPU
    reader = easyocr.Reader(['en'], gpu=False)
    print("EasyOCR ready.\n")

    print(f"{'No.':<4} {'Image Type':<18} {'Ground Truth (LaTeX)':<30} {'Prediction':<20} {'Conf':>5}  {'Error Type'}")
    print("-" * 100)

    results_data = []
    for idx, (img_type, img_path, gt) in enumerate(all_images, 1):
        prediction, conf = run_ocr(reader, img_path)
        error = classify_error(prediction, gt, conf)
        print(f"{idx:<4} {img_type:<18} {gt:<30} {prediction:<20} {conf:>5.2f}  {error}")
        results_data.append((img_type, gt, prediction, conf, error))

    print("\n" + "=" * 60)
    print("AVERAGE CONFIDENCE PER CONDITION")
    print("=" * 60)
    for condition in ["Clean Captured", "Gaussian Blur", "Salt-and-Pepper", "Low Contrast"]:
        scores = [c for (t, _, _, c, _) in results_data if t == condition]
        avg = round(sum(scores) / len(scores), 3) if scores else 0.0
        print(f"  {condition:<20} →  {avg}")

    print("\nDone!")


if __name__ == "__main__":
    main()