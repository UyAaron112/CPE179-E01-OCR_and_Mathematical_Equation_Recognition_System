import os
import argparse
import time
import string
import cv2
import nltk
import pytesseract
from PIL import Image as PILImage
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

stop_words = set(stopwords.words("english"))

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img = cv2.fastNlMeansDenoising(img, h=30)

    img = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )

    img = cv2.resize(
        img,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC,
    )

    return img


def postprocess(text):
    tokens = word_tokenize(text)
    tokens = [t.lower() for t in tokens]
    tokens = [t for t in tokens if t not in string.punctuation]
    tokens = [t for t in tokens if t not in stop_words]
    return tokens

def save_result(name, text, tokens, ms):
    path = os.path.join(RESULTS_DIR, f"{name}_result.txt")

    with open(path, "w") as f:
        f.write(f"Output : {text}\n")
        f.write(f"Tokens : {tokens}\n")
        f.write(f"Time   : {ms} ms\n")

    print(f"[INFO] Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image",
    )

    args = parser.parse_args()

    print("=" * 50)
    print(" Raspberry Pi OCR Pipeline (Tesseract)")
    print("=" * 50)

    print(f"[INFO] Image: {args.image}")

    img = preprocess_image(args.image)
    pil_img = PILImage.fromarray(img)

    print("[INFO] Running Tesseract OCR...")
    start = time.time()

    custom_config = r"--psm 6 --oem 3"

    text = pytesseract.image_to_string(
        pil_img,
        config=custom_config,
    ).strip()

    ms = round((time.time() - start) * 1000, 2)

    data = pytesseract.image_to_data(
        pil_img,
        config=custom_config,
        output_type=pytesseract.Output.DICT,
    )

    confs = [
        int(c)
        for c in data["conf"]
        if str(c).isdigit() and int(c) >= 0
    ]

    confidence = (
        round(sum(confs) / len(confs), 2)
        if confs
        else 0.0
    )

    tokens = postprocess(text)

    print("\n──────── RESULT ────────")
    print("TEXT      :", text)
    print("CONFIDENCE:", confidence, "%")
    print("TOKENS    :", tokens)
    print("TIME      :", ms, "ms")

    save_result(
        os.path.splitext(
            os.path.basename(args.image)
        )[0],
        text,
        tokens,
        ms,
    )

    print("\nDONE")


if __name__ == "__main__":
    main()