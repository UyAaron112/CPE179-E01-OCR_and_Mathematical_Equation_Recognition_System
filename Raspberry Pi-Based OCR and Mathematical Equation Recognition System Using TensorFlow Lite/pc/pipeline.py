# pipeline.py
# OCR + Math Equation Recognition Pipeline
# Input:  image captured from phone, transferred via USB/SD card
# Output: recognized plain text (TrOCR) + LaTeX equation string (pix2tex)
# Target: Raspberry Pi (CPU only, no GPU)

import os
import string
import warnings
warnings.filterwarnings("ignore")

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import cv2
import numpy as np

# Download NLTK data if not already present
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

stop_words = set(stopwords.words("english"))

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
INPUT_DIR   = os.path.join(PROJECT_DIR, "input_images")

# Ground truth LaTeX for evaluation (fill in per image)
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
# PREPROCESSING
# ─────────────────────────────────────────

def preprocess_image(image_path):
    """
    Load image from path, apply preprocessing to improve OCR accuracy.
    Returns a PIL Image (RGB) ready for TrOCR.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise — helpful for phone camera captures
    denoised = cv2.fastNlMeansDenoising(gray, h=30)

    # Adaptive threshold — handles uneven lighting from phone flash
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Convert back to RGB PIL image for TrOCR
    pil_img = Image.fromarray(thresh).convert("RGB")
    return pil_img

# ─────────────────────────────────────────
# TROCR — PRINTED TEXT OCR
# ─────────────────────────────────────────

def load_trocr_printed():
    """Load TrOCR printed model. Heavy on first run — cached after."""
    print("Loading microsoft/trocr-base-printed...")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    print("TrOCR ready.\n")
    return processor, model


def run_trocr(processor, model, pil_image):
    """Run TrOCR inference on a PIL image. Returns recognized string."""
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, max_new_tokens=64)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# ─────────────────────────────────────────
# LATEX OCR — MATH EQUATION RECOGNITION
# Uses pix2tex (LaTeX-OCR) for equation images
# ─────────────────────────────────────────

def load_latex_ocr():
    """
    Load pix2tex LatexOCR model.
    Install: pip install pix2tex[gui]
    On Raspberry Pi: pip install pix2tex (no GUI needed)
    """
    try:
        from pix2tex.cli import LatexOCR
        print("Loading pix2tex LaTeX-OCR model...")
        model = LatexOCR()
        print("LaTeX-OCR ready.\n")
        return model
    except ImportError:
        print("[WARNING] pix2tex not installed. LaTeX output will be skipped.")
        print("          Install with: pip install pix2tex")
        return None


def run_latex_ocr(latex_model, pil_image):
    """Run LaTeX OCR on a PIL image. Returns LaTeX string."""
    if latex_model is None:
        return "pix2tex not available"
    try:
        return latex_model(pil_image)
    except Exception as e:
        return f"Error: {e}"

# ─────────────────────────────────────────
# NLTK PREPROCESSING
# ─────────────────────────────────────────

def preprocess_text(text):
    """Tokenize, lowercase, remove punctuation and stopwords."""
    tokens = word_tokenize(text)
    tokens = [t.lower() for t in tokens]
    tokens = [t for t in tokens if t not in string.punctuation]
    tokens = [t for t in tokens if t not in stop_words]
    return tokens

# ─────────────────────────────────────────
# EVALUATION HELPERS
# ─────────────────────────────────────────

def classify_latex_error(prediction, ground_truth):
    """Simple error classification for LaTeX output."""
    if not prediction or "Error" in prediction or "not available" in prediction:
        return "Recognition failure"
    if prediction.strip() == ground_truth.strip():
        return "None"
    if len(prediction) < len(ground_truth) * 0.5:
        return "Incomplete recognition"
    return "Symbol substitution"

# ─────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Raspberry Pi OCR + Math Equation Recognition Pipeline")
    print("  Input: phone-captured images via USB/SD card")
    print("=" * 70)

    # Collect images
    samples = []
    for key, gt in GROUND_TRUTH.items():
        path = os.path.join(INPUT_DIR, f"{key}.jpg")
        if os.path.exists(path):
            samples.append((key, gt, path))
        else:
            print(f"[WARNING] {path} not found. Skipping.")

    if not samples:
        print("[ERROR] No images found in INPUT_DIR. Check your USB/SD mount path.")
        return

    # Load models
    trocr_processor, trocr_model = load_trocr_printed()
    latex_model = load_latex_ocr()

    # ── Section 1: TrOCR Plain Text Recognition ──────────────────────────
    print("=" * 70)
    print("SECTION 1: TrOCR — Plain Text Recognition")
    print("=" * 70)
    print(f"\n{'Image':<10} {'Ground Truth (LaTeX)':<30} {'TrOCR Output'}")
    print("-" * 70)

    trocr_results = []
    for name, gt, path in samples:
        pil_img = preprocess_image(path)
        trocr_out = run_trocr(trocr_processor, trocr_model, pil_img)
        print(f"{name:<10} {gt:<30} {trocr_out}")
        trocr_results.append((name, gt, trocr_out, path))

    # ── Section 2: LaTeX Equation Recognition ────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 2: LaTeX Equation Recognition (pix2tex)")
    print("=" * 70)
    print(f"\n{'Image':<10} {'Ground Truth (LaTeX)':<30} {'LaTeX OCR Output':<35} {'Error Type'}")
    print("-" * 90)

    latex_results = []
    for name, gt, trocr_out, path in trocr_results:
        pil_img = preprocess_image(path)
        latex_out = run_latex_ocr(latex_model, pil_img)
        error = classify_latex_error(latex_out, gt)
        print(f"{name:<10} {gt:<30} {latex_out:<35} {error}")
        latex_results.append((name, gt, trocr_out, latex_out, error))

    # ── Section 3: NLTK Token Preprocessing ──────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 3: NLTK Token Preprocessing (TrOCR output)")
    print("=" * 70)
    print(f"\n{'Image':<10} {'Ground Truth Tokens':<30} {'TrOCR Tokens'}")
    print("-" * 70)

    for name, gt, trocr_out, latex_out, error in latex_results:
        gt_tokens = preprocess_text(gt)
        trocr_tokens = preprocess_text(trocr_out)
        print(f"{name:<10} {str(gt_tokens):<30} {str(trocr_tokens)}")

# ── Section 4: Final Comparison Table ────────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 4: Final Comparison Table")
    print("=" * 70)
    print(f"\n{'Image':<10} {'Ground Truth':<30} {'TrOCR Output':<25} {'LaTeX OCR Output':<35} {'Error'}")
    print("-" * 110)

    for name, gt, trocr_out, latex_out, error in latex_results:
        print(f"{name:<10} {gt:<30} {trocr_out:<25} {latex_out:<35} {error}")

    # ── Save Results to File ──────────────────────────────────────────────
    RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, "pipeline_results.txt")

    with open(results_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("SECTION 1: TrOCR — Plain Text Recognition\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Image':<10} {'Ground Truth (LaTeX)':<50} {'TrOCR Output'}\n")
        f.write("-" * 70 + "\n")
        for name, gt, trocr_out, path in trocr_results:
            f.write(f"{name:<10} {gt:<50} {trocr_out}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("SECTION 2: LaTeX Equation Recognition (pix2tex)\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Image':<10} {'Ground Truth (LaTeX)':<50} {'LaTeX OCR Output':<35} {'Error Type'}\n")
        f.write("-" * 90 + "\n")
        for name, gt, trocr_out, latex_out, error in latex_results:
            f.write(f"{name:<10} {gt:<50} {latex_out:<35} {error}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("SECTION 4: Final Comparison Table\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Image':<10} {'Ground Truth':<50} {'TrOCR Output':<25} {'LaTeX OCR Output':<35} {'Error'}\n")
        f.write("-" * 110 + "\n")
        for name, gt, trocr_out, latex_out, error in latex_results:
            f.write(f"{name:<10} {gt:<50} {trocr_out:<25} {latex_out:<35} {error}\n")

    print(f"\nResults saved to: {results_path}")

    print("\n" + "=" * 70)
    print("Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()