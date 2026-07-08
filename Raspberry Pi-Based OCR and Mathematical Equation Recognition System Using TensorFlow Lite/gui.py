# ui.py
# Mapúa University — CPE179P Group 6
# Raspberry Pi OCR & Math Equation Recognition System
# Tkinter UI integrated with Tesseract OCR backend

import os
import time
import string
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import pytesseract
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import platform
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────
# COLOR PALETTE
# ─────────────────────────────────────────
MAPUA_RED    = "#9E1B32"
MAPUA_GOLD   = "#F3A900"
BG_PAGE      = "#F0F4F8"
BG_CARD      = "#FFFFFF"
TEXT_MAIN    = "#1E293B"
TEXT_MUTED   = "#64748B"
BORDER_COLOR = "#CBD5E1"
COLOR_GREEN  = "#10B981"
COLOR_BLUE   = "#0284C7"
COLOR_TEAL   = "#0D9488"

# ─────────────────────────────────────────
# OCR BACKEND — from pipeline_pi.py
# ─────────────────────────────────────────

stop_words = set(stopwords.words("english"))

def preprocess_image(image_path):
    """Grayscale → denoise → adaptive threshold → upscale."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.fastNlMeansDenoising(img, h=30)
    img = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return img


def run_tesseract(image_path):
    """
    Run Tesseract OCR on the image.
    Returns: (text, confidence, tokens, inference_ms)
    """
    img        = preprocess_image(image_path)
    pil_img    = Image.fromarray(img)
    config     = r"--psm 6 --oem 3"

    start = time.time()
    text  = pytesseract.image_to_string(pil_img, config=config).strip()
    ms    = round((time.time() - start) * 1000, 2)

    data  = pytesseract.image_to_data(
        pil_img, config=config,
        output_type=pytesseract.Output.DICT
    )
    confs = [
        int(c) for c in data["conf"]
        if str(c).isdigit() and int(c) >= 0
    ]
    confidence = round(sum(confs) / len(confs), 2) if confs else 0.0

    tokens = word_tokenize(text)
    tokens = [t.lower() for t in tokens if t not in string.punctuation and t not in stop_words]

    return text, confidence, tokens, ms


def save_result(image_path, text, confidence, tokens, ms):
    """Save result to results/ folder."""
    name = os.path.splitext(os.path.basename(image_path))[0]
    path = os.path.join(RESULTS_DIR, f"{name}_result.txt")
    with open(path, "w") as f:
        f.write(f"Image      : {image_path}\n")
        f.write(f"Output     : {text}\n")
        f.write(f"Confidence : {confidence}%\n")
        f.write(f"Tokens     : {tokens}\n")
        f.write(f"Time       : {ms} ms\n")
    return path


# ─────────────────────────────────────────
# GUI
# ─────────────────────────────────────────

class OcrGuiApp:
    def __init__(self, root):
        self.root       = root
        self.image_path = None

        self.root.title("Mapúa OCR & Equation Recognition System")
        self.root.geometry("940x640")
        self.root.configure(bg=BG_PAGE)
        self.root.resizable(False, False)

        self._build_header()
        self._build_main()
        self._build_footer()

    # ── HEADER ───────────────────────────
    def _build_header(self):
        hf = tk.Frame(self.root, bg=MAPUA_RED, height=75)
        hf.pack(fill="x", side="top")
        hf.pack_propagate(False)
        tk.Label(hf, text="MAPÚA UNIVERSITY",
                 font=("Helvetica", 18, "bold"),
                 bg=MAPUA_RED, fg=MAPUA_GOLD).pack(pady=(10, 0))
        tk.Label(hf,
                 text="CPE179P  •  Raspberry Pi OCR & Mathematical Equation Recognition System",
                 font=("Helvetica", 10),
                 bg=MAPUA_RED, fg="#FFFFFF").pack()

    # ── MAIN ─────────────────────────────
    def _build_main(self):
        container = tk.Frame(self.root, bg=BG_PAGE)
        container.pack(fill="both", expand=True, padx=20, pady=15)

        self._build_left(container)
        self._build_right(container)

    # ── LEFT CARD (Input) ─────────────────
    def _build_left(self, parent):
        card = tk.Frame(parent, bg=BG_CARD, bd=1, relief="solid",
                        highlightbackground=BORDER_COLOR)
        card.pack(side="left", fill="both", expand=True, padx=(0, 10))

        tk.Label(card, text="1. IMAGE SOURCE",
                 font=("Helvetica", 11, "bold"),
                 bg=BG_CARD, fg=TEXT_MAIN).pack(anchor="w", padx=20, pady=(15, 10))

        # Image preview frame
        self.img_frame = tk.Frame(card, width=400, height=290, bg="#E2E8F0")
        self.img_frame.pack(padx=20, pady=5)
        self.img_frame.pack_propagate(False)

        self.img_label = tk.Label(
            self.img_frame,
            text="No Image Uploaded\n(Supported: JPG, PNG)",
            bg="#E2E8F0", fg=TEXT_MUTED, font=("Helvetica", 10)
        )
        self.img_label.pack(expand=True)

        # Buttons
        btn_row = tk.Frame(card, bg=BG_CARD)
        btn_row.pack(pady=(15, 20))

        self.upload_btn = tk.Button(
            btn_row, text="Upload Image",
            font=("Helvetica", 10, "bold"),
            bg=TEXT_MAIN, fg="white",
            activebackground="#334155", activeforeground="white",
            relief="flat", padx=15, pady=8,
            cursor="hand2", command=self.upload_image
        )
        self.upload_btn.pack(side="left", padx=6)

        self.process_btn = tk.Button(
            btn_row, text="Run OCR ▶",
            font=("Helvetica", 10, "bold"),
            bg="#E2E8F0", fg="#94A3B8",
            relief="flat", padx=15, pady=8,
            state=tk.DISABLED, command=self.process_image
        )
        self.process_btn.pack(side="left", padx=6)

        self.clear_btn = tk.Button(
            btn_row, text="✖ Clear",
            font=("Helvetica", 10, "bold"),
            bg="#E2E8F0", fg="#94A3B8",
            relief="flat", padx=10, pady=8,
            state=tk.DISABLED, command=self.clear_ui
        )
        self.clear_btn.pack(side="left", padx=6)

    # ── RIGHT CARD (Output) ───────────────
    def _build_right(self, parent):
        card = tk.Frame(parent, bg=BG_CARD, bd=1, relief="solid",
                        highlightbackground=BORDER_COLOR)
        card.pack(side="right", fill="both", expand=True, padx=(10, 0))

        # Header row
        right_header = tk.Frame(card, bg=BG_CARD)
        right_header.pack(fill="x", padx=20, pady=(15, 5))

        tk.Label(right_header, text="2. RECOGNITION OUTPUT",
                 font=("Helvetica", 11, "bold"),
                 bg=BG_CARD, fg=TEXT_MAIN).pack(side="left")

        self.status_label = tk.Label(
            right_header, text="Status: Waiting for image...",
            font=("Helvetica", 9, "italic"),
            bg=BG_CARD, fg=TEXT_MUTED
        )
        self.status_label.pack(side="right")

        # Metrics row — confidence + time
        metrics_row = tk.Frame(card, bg=BG_CARD)
        metrics_row.pack(fill="x", padx=20, pady=(0, 8))

        self.conf_label = tk.Label(
            metrics_row, text="Confidence: —",
            font=("Helvetica", 10, "bold"),
            bg=BG_CARD, fg=TEXT_MUTED
        )
        self.conf_label.pack(side="left", padx=(0, 20))

        self.time_label = tk.Label(
            metrics_row, text="Inference Time: —",
            font=("Helvetica", 10, "bold"),
            bg=BG_CARD, fg=TEXT_MUTED
        )
        self.time_label.pack(side="left")

        # Scrollable text output
        output_frame = tk.Frame(card, bg=BG_CARD, bd=1, relief="solid",
                                highlightbackground=BORDER_COLOR)
        output_frame.pack(fill="both", expand=True, padx=20, pady=5)

        scrollbar = tk.Scrollbar(output_frame)
        scrollbar.pack(side="right", fill="y")

        self.output_text = tk.Text(
            output_frame, wrap="word",
            font=("Consolas", 11),
            bg="#F8FAFC", fg=TEXT_MAIN,
            bd=0, padx=15, pady=12,
            yscrollcommand=scrollbar.set
        )
        self.output_text.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.output_text.yview)

        # Export button
        self.save_btn = tk.Button(
            card, text="⭳ Export Output to .TXT",
            font=("Helvetica", 10, "bold"),
            bg="#E2E8F0", fg="#94A3B8",
            relief="flat", padx=20, pady=8,
            state=tk.DISABLED, command=self.export_result
        )
        self.save_btn.pack(pady=(10, 20))

    # ── FOOTER ───────────────────────────
    def _build_footer(self):
        tk.Label(
            self.root,
            text="Computer Engineering Dept.  •  Mapúa University  •  CPE179P Group 6",
            bg=BG_PAGE, fg=TEXT_MUTED, font=("Helvetica", 8)
        ).pack(side="bottom", pady=(0, 8))

    # ─────────────────────────────────────────
    # ACTIONS
    # ─────────────────────────────────────────

def upload_image(self):
    """Open file dialog and display selected image."""
    path = filedialog.askopenfilename(
        filetypes=[
            ("Image Files", "*.png *.jpg *.jpeg *.bmp"),
            ("PNG files", "*.png"),
            ("JPG files", "*.jpg *.jpeg"),
            ("All files", "*.*")
        ]
    )
    if not path:
        return

        self.image_path = path

        # Display preview
        img    = Image.open(path)
        img.thumbnail((400, 290))
        img_tk = ImageTk.PhotoImage(img)
        self.img_label.configure(image=img_tk, text="")
        self.img_label.image = img_tk

        # Reset output area
        self.output_text.delete(1.0, tk.END)
        self.conf_label.config(text="Confidence: —", fg=TEXT_MUTED)
        self.time_label.config(text="Inference Time: —", fg=TEXT_MUTED)

        # Enable buttons
        self.process_btn.config(state=tk.NORMAL, bg=MAPUA_RED, fg="white", cursor="hand2")
        self.clear_btn.config(state=tk.NORMAL, bg="#94A3B8", fg="white", cursor="hand2")
        self.save_btn.config(state=tk.DISABLED, bg="#E2E8F0", fg="#94A3B8", cursor="arrow")
        self.status_label.config(text="Status: Image loaded. Ready.", fg=COLOR_BLUE)

    def process_image(self):
        """Run Tesseract OCR and display results."""
        if not self.image_path:
            return

        self.status_label.config(text="Status: Running OCR...", fg=MAPUA_GOLD)
        self.output_text.delete(1.0, tk.END)
        self.root.update()

        try:
            text, confidence, tokens, ms = run_tesseract(self.image_path)

            # Save result automatically
            saved_path = save_result(self.image_path, text, confidence, tokens, ms)

            # Update metrics
            self.conf_label.config(
                text=f"Confidence: {confidence}%",
                fg=COLOR_GREEN if confidence >= 50 else MAPUA_RED
            )
            self.time_label.config(
                text=f"Inference Time: {ms} ms",
                fg=TEXT_MAIN
            )

            # Build output display
            output = (
                f"─── RECOGNIZED EQUATION / TEXT ───\n"
                f"{text if text else '[No text detected]'}\n\n"
                f"─── CONFIDENCE SCORE ───\n"
                f"{confidence}%\n\n"
                f"─── INFERENCE TIME ───\n"
                f"{ms} ms\n\n"
                f"─── CLEANED TOKENS ───\n"
                f"{tokens}\n\n"
                f"─── RESULT SAVED TO ───\n"
                f"{saved_path}"
            )
            self.output_text.insert(tk.END, output)

            # Enable export
            self.save_btn.config(state=tk.NORMAL, bg=COLOR_TEAL, fg="white", cursor="hand2")
            self.status_label.config(text="Status: Inference complete.", fg=COLOR_GREEN)

        except Exception as e:
            self.status_label.config(text="Status: Error occurred.", fg=MAPUA_RED)
            messagebox.showerror("OCR Error", f"Failed to process image:\n{str(e)}")

    def export_result(self):
        """Export output text to a .txt file chosen by user."""
        content = self.output_text.get(1.0, tk.END).strip()
        if not content:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Document", "*.txt")],
            title="Export Output"
        )
        if path:
            with open(path, "w") as f:
                f.write(content)
            messagebox.showinfo("Export Success", f"Saved to:\n{path}")

    def clear_ui(self):
        """Reset the UI to its initial state."""
        self.image_path = None

        self.img_label.configure(
            image="",
            text="No Image Uploaded\n(Supported: JPG, PNG)"
        )
        self.img_label.image = None

        self.output_text.delete(1.0, tk.END)
        self.conf_label.config(text="Confidence: —", fg=TEXT_MUTED)
        self.time_label.config(text="Inference Time: —", fg=TEXT_MUTED)
        self.status_label.config(text="Status: Waiting for image...", fg=TEXT_MUTED)

        self.process_btn.config(state=tk.DISABLED, bg="#E2E8F0", fg="#94A3B8", cursor="arrow")
        self.save_btn.config(state=tk.DISABLED, bg="#E2E8F0", fg="#94A3B8", cursor="arrow")
        self.clear_btn.config(state=tk.DISABLED, bg="#E2E8F0", fg="#94A3B8", cursor="arrow")


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = OcrGuiApp(root)
    root.mainloop()
