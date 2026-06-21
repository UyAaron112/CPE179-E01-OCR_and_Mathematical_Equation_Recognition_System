import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

# ==========================================
# 🎨 PROFESSIONAL MAPÚA COLOR PALETTE 
# ==========================================
MAPUA_RED    = "#9E1B32"  # Official Mapúa Cardinal Red
MAPUA_GOLD   = "#F3A900"  # Official Mapúa Gold accent
BG_PAGE      = "#F0F4F8"  # Soft cool-gray page background
BG_CARD      = "#FFFFFF"  # Pure white card backgrounds
TEXT_MAIN    = "#1E293B"  # Dark slate (softer & more pro than pure black)
TEXT_MUTED   = "#64748B"  # Muted gray for secondary labels
BORDER_COLOR = "#CBD5E1"  # Clean light-gray for card borders

# ==========================================
# PLACEHOLDERS FOR LOGIC LAYER
# ==========================================
class ImageProcessor:
    def preprocess(self, image_path): pass
    def segment(self, image_path): pass

class OCRRecognizer:
    def recognizeText(self, image_path):
        return "Mapua University\nSchool of Electrical, Electronics and Computer Engineering\nCPE179P - Design Project"

class EqnRecognizer:
    def recognizeEquation(self, image_path):
        return "f(x) = \\int_{a}^{b} x^2 \\,dx + \\sum_{i=1}^{n} y_i"

class TensorFlowLite:
    def loadModel(self): pass
    def runInference(self, processed_image): pass

# ==========================================
# PRESENTATION LAYER (GUI)
# ==========================================
class OcrGuiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mapúa OCR & Equation Recognition System")
        self.root.geometry("900x620")
        self.root.configure(bg=BG_PAGE)
        
        self.image_path = None
        
        # ----------------------------------------------------
        # 1. TOP BRANDING BANNER
        # ----------------------------------------------------
        header_frame = tk.Frame(root, bg=MAPUA_RED, height=75)
        header_frame.pack(fill="x", side="top")
        header_frame.pack_propagate(False) 

        title_label = tk.Label(
            header_frame, 
            text="MAPÚA UNIVERSITY", 
            font=("Helvetica", 18, "bold"), 
            bg=MAPUA_RED, fg=MAPUA_GOLD
        )
        title_label.pack(pady=(10, 0))

        sub_label = tk.Label(
            header_frame, 
            text="CPE179P • Raspberry Pi OCR & Mathematical Equation Recognition System", 
            font=("Helvetica", 10), 
            bg=MAPUA_RED, fg="#FFFFFF"
        )
        sub_label.pack()

        # ----------------------------------------------------
        # 2. MAIN DASHBOARD CONTAINER (2 Columns)
        # ----------------------------------------------------
        main_container = tk.Frame(root, bg=BG_PAGE)
        main_container.pack(fill="both", expand=True, padx=20, pady=15)

        # ====================================================
        # LEFT COLUMN: INPUT CARD
        # ====================================================
        left_card = tk.Frame(main_container, bg=BG_CARD, bd=1, relief="solid", highlightbackground=BORDER_COLOR)
        left_card.pack(side="left", fill="both", expand=True, padx=(0, 10))

        left_title = tk.Label(left_card, text="1. IMAGE SOURCE", font=("Helvetica", 11, "bold"), bg=BG_CARD, fg=TEXT_MAIN)
        left_title.pack(anchor="w", padx=20, pady=(15, 10))

        # Image Display Canvas
        self.image_frame = tk.Frame(left_card, width=380, height=280, bg="#E2E8F0")
        self.image_frame.pack(padx=20, pady=5)
        self.image_frame.pack_propagate(False)

        self.image_label = tk.Label(self.image_frame, text="No Image Uploaded\n(Supported: JPG, PNG)", bg="#E2E8F0", fg=TEXT_MUTED, font=("Helvetica", 10))
        self.image_label.pack(expand=True)

        # Action Buttons inside Left Card
        btn_container = tk.Frame(left_card, bg=BG_CARD)
        btn_container.pack(pady=(15, 20))

        self.upload_btn = tk.Button(
            btn_container, text="Upload Image", font=("Helvetica", 10, "bold"),
            bg=TEXT_MAIN, fg="white", activebackground="#334155", activeforeground="white",
            relief="flat", padx=15, pady=8, cursor="hand2", command=self.upload_image
        )
        self.upload_btn.pack(side="left", padx=6)

        self.process_btn = tk.Button(
            btn_container, text="Run Inference ▶", font=("Helvetica", 10, "bold"),
            bg=MAPUA_RED, fg="white", activebackground="#7A1527", activeforeground="white",
            relief="flat", padx=15, pady=8, cursor="hand2", state=tk.DISABLED, command=self.process_image
        )
        self.process_btn.pack(side="left", padx=6)

        # ====================================================
        # RIGHT COLUMN: OUTPUT CARD
        # ====================================================
        right_card = tk.Frame(main_container, bg=BG_CARD, bd=1, relief="solid", highlightbackground=BORDER_COLOR)
        right_card.pack(side="right", fill="both", expand=True, padx=(10, 0))

        right_title = tk.Label(right_card, text="2. TFLITE RECOGNITION OUTPUT", font=("Helvetica", 11, "bold"), bg=BG_CARD, fg=TEXT_MAIN)
        right_title.pack(anchor="w", padx=20, pady=(15, 10))

        # Monospaced Text box for mathematical alignment
        self.output_text = tk.Text(
            right_card, wrap="word", font=("Consolas", 11), 
            bg="#F8FAFC", fg=TEXT_MAIN, bd=1, relief="solid", highlightbackground=BORDER_COLOR,
            padx=15, pady=12
        )
        self.output_text.pack(fill="both", expand=True, padx=20, pady=5)

        # Export Button at bottom right
        self.save_btn = tk.Button(
            right_card, text="⭳ Export Output to .TXT", font=("Helvetica", 10, "bold"),
            bg="#0D9488", fg="white", activebackground="#0F766E", activeforeground="white",
            relief="flat", padx=20, pady=8, cursor="hand2", state=tk.DISABLED, command=self.save_result
        )
        self.save_btn.pack(pady=(15, 20))

        # ----------------------------------------------------
        # 3. FOOTER
        # ----------------------------------------------------
        footer = tk.Label(root, text="Computer Engineering Dept. • Mapúa University • Group 6", bg=BG_PAGE, fg=TEXT_MUTED, font=("Helvetica", 8))
        footer.pack(side="bottom", pady=(0, 8))

    # --- Use Case Implementations ---

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            self.image_path = file_path
            
            img = Image.open(self.image_path)
            img.thumbnail((380, 280)) 
            img_tk = ImageTk.PhotoImage(img)
            
            self.image_label.configure(image=img_tk, text="")
            self.image_label.image = img_tk 
            
            self.process_btn.config(state=tk.NORMAL)
            self.output_text.delete(1.0, tk.END)

    def process_image(self):
        if not self.image_path: return
            
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "Initializing TFLite Runtime...\nRunning local inference...")
        self.root.update()
        
        try:
            img_processor = ImageProcessor()
            ocr_engine    = OCRRecognizer()
            eqn_engine    = EqnRecognizer()
            tflite_model  = TensorFlowLite()
            
            img_processor.preprocess(self.image_path)
            tflite_model.loadModel()
            
            text_result = ocr_engine.recognizeText(self.image_path)
            eqn_result  = eqn_engine.recognizeEquation(self.image_path)
            
            final_output = f"--- DETECTED STANDARD TEXT ---\n{text_result}\n\n\n--- DETECTED MATHEMATICAL EQUATION ---\n{eqn_result}"
            
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, final_output)
            self.save_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Inference Error", f"Failed to execute model:\n{str(e)}")

    def save_result(self):
        result_content = self.output_text.get(1.0, tk.END).strip()
        if not result_content: return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Document", "*.txt")],
            title="Export TFLite Output"
        )
        if file_path:
            with open(file_path, "w") as file:
                file.write(result_content)
            messagebox.showinfo("Export Success", f"File saved successfully to:\n{file_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = OcrGuiApp(root)
    root.mainloop()