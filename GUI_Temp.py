import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

# -----------------------------------------------------------------------------
# 🔌 TEAMMATE "PLUG-IN" INSTRUCTIONS:
# When your teammate gives you their finished OCR file, drop it in the same folder, 
# delete the "MOCK BACKEND" section below, and uncomment the real import here:
#
# from ocr_engine import OCREngine 
# -----------------------------------------------------------------------------

# =============================================================================
# 1. MOCK BACKEND (Lets you test the GUI standalone without their code)
# =============================================================================
class MockOCREngine:
    def extract_text(self, image_path):
        return "MAPÚA UNIVERSITY\nIntramuros, Manila\n[Simulated OCR Output from Teammate's Engine]\n\n(Scrolling Test Line 1)\n(Scrolling Test Line 2)\n(Scrolling Test Line 3)"

class MockEqnEngine:
    def recognizeEquation(self, image_path):
        return "f(x) = \\int_{a}^{b} x^2 \\,dx = \\frac{b^3 - a^3}{3}"

class MockTFLiteModel:
    def loadModel(self): pass
    def runInference(self): pass


# =============================================================================
# 2. PROFESSIONAL MAPÚA COLOR PALETTE 
# =============================================================================
MAPUA_RED    = "#9E1B32"  # Official Mapúa Cardinal Red
MAPUA_GOLD   = "#F3A900"  # Official Mapúa Gold accent
BG_PAGE      = "#F0F4F8"  
BG_CARD      = "#FFFFFF"  
TEXT_MAIN    = "#1E293B"  
TEXT_MUTED   = "#64748B"  
BORDER_COLOR = "#CBD5E1"  

# =============================================================================
# 3. PRESENTATION LAYER (GUI)
# =============================================================================
class OcrGuiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mapúa OCR & Equation Recognition System")
        self.root.geometry("900x620")
        self.root.configure(bg=BG_PAGE)
        
        self.image_path = None
        
        # --- Top Banner ---
        header_frame = tk.Frame(root, bg=MAPUA_RED, height=75)
        header_frame.pack(fill="x", side="top")
        header_frame.pack_propagate(False) 

        tk.Label(header_frame, text="MAPÚA UNIVERSITY", font=("Helvetica", 18, "bold"), bg=MAPUA_RED, fg=MAPUA_GOLD).pack(pady=(10, 0))
        tk.Label(header_frame, text="CPE179P • Raspberry Pi OCR & Mathematical Equation Recognition System", font=("Helvetica", 10), bg=MAPUA_RED, fg="#FFFFFF").pack()

        # --- Main Dashboard ---
        main_container = tk.Frame(root, bg=BG_PAGE)
        main_container.pack(fill="both", expand=True, padx=20, pady=15)

        # ==========================================
        # LEFT COLUMN (Input)
        # ==========================================
        left_card = tk.Frame(main_container, bg=BG_CARD, bd=1, relief="solid", highlightbackground=BORDER_COLOR)
        left_card.pack(side="left", fill="both", expand=True, padx=(0, 10))

        tk.Label(left_card, text="1. IMAGE SOURCE", font=("Helvetica", 11, "bold"), bg=BG_CARD, fg=TEXT_MAIN).pack(anchor="w", padx=20, pady=(15, 10))

        self.image_frame = tk.Frame(left_card, width=380, height=280, bg="#E2E8F0")
        self.image_frame.pack(padx=20, pady=5)
        self.image_frame.pack_propagate(False)

        self.image_label = tk.Label(self.image_frame, text="No Image Uploaded\n(Supported: JPG, PNG)", bg="#E2E8F0", fg=TEXT_MUTED, font=("Helvetica", 10))
        self.image_label.pack(expand=True)

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
            bg="#E2E8F0", fg="#94A3B8", activebackground="#7A1527", activeforeground="white",
            relief="flat", padx=15, pady=8, state=tk.DISABLED, command=self.process_image
        )
        self.process_btn.pack(side="left", padx=6)

        # [NEW] Clear Button
        self.clear_btn = tk.Button(
            btn_container, text="✖ Clear", font=("Helvetica", 10, "bold"),
            bg="#E2E8F0", fg="#94A3B8", activebackground="#CBD5E1", activeforeground="#1E293B",
            relief="flat", padx=10, pady=8, state=tk.DISABLED, command=self.clear_ui
        )
        self.clear_btn.pack(side="left", padx=6)

        # ==========================================
        # RIGHT COLUMN (Output)
        # ==========================================
        right_card = tk.Frame(main_container, bg=BG_CARD, bd=1, relief="solid", highlightbackground=BORDER_COLOR)
        right_card.pack(side="right", fill="both", expand=True, padx=(10, 0))

        # [NEW] Header Frame for Title and Status
        right_header = tk.Frame(right_card, bg=BG_CARD)
        right_header.pack(fill="x", padx=20, pady=(15, 10))

        tk.Label(right_header, text="2. TFLITE RECOGNITION OUTPUT", font=("Helvetica", 11, "bold"), bg=BG_CARD, fg=TEXT_MAIN).pack(side="left")
        
        # [NEW] Dynamic Status Label
        self.status_label = tk.Label(right_header, text="Status: Waiting for Image...", font=("Helvetica", 9, "italic"), bg=BG_CARD, fg=TEXT_MUTED)
        self.status_label.pack(side="right")

        # [NEW] Scrollable Output Frame
        output_frame = tk.Frame(right_card, bg=BG_CARD, bd=1, relief="solid", highlightbackground=BORDER_COLOR)
        output_frame.pack(fill="both", expand=True, padx=20, pady=5)

        self.scrollbar = tk.Scrollbar(output_frame)
        self.scrollbar.pack(side="right", fill="y")

        self.output_text = tk.Text(
            output_frame, wrap="word", font=("Consolas", 11), 
            bg="#F8FAFC", fg=TEXT_MAIN, bd=0, padx=15, pady=12, yscrollcommand=self.scrollbar.set
        )
        self.output_text.pack(side="left", fill="both", expand=True)
        self.scrollbar.config(command=self.output_text.yview)

        self.save_btn = tk.Button(
            right_card, text="⭳ Export Output to .TXT", font=("Helvetica", 10, "bold"),
            bg="#E2E8F0", fg="#94A3B8", activebackground="#0F766E", activeforeground="white",
            relief="flat", padx=20, pady=8, state=tk.DISABLED, command=self.save_result
        )
        self.save_btn.pack(pady=(15, 20))

        # --- Footer ---
        tk.Label(root, text="Computer Engineering Dept. • Mapúa University • Group 6", bg=BG_PAGE, fg=TEXT_MUTED, font=("Helvetica", 8)).pack(side="bottom", pady=(0, 8))

    # ==========================================
    # ACTIONS & LOGIC
    # ==========================================
    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            self.image_path = file_path
            img = Image.open(self.image_path)
            img.thumbnail((380, 280)) 
            img_tk = ImageTk.PhotoImage(img)
            
            self.image_label.configure(image=img_tk, text="")
            self.image_label.image = img_tk 
            
            # Wake up Process & Clear buttons, update Status
            self.process_btn.config(state=tk.NORMAL, bg=MAPUA_RED, fg="white", cursor="hand2")
            self.clear_btn.config(state=tk.NORMAL, bg="#94A3B8", fg="white", cursor="hand2") 
            self.status_label.config(text="Status: Image Loaded. Ready.", fg="#0284C7") # Blue
            
            self.output_text.delete(1.0, tk.END)

    def process_image(self):
        if not self.image_path: return
            
        self.output_text.delete(1.0, tk.END)
        self.status_label.config(text="Status: Running Inference...", fg=MAPUA_GOLD) # Gold
        self.root.update()
        
        try:
            # INSTANTIATE THE MOCKS (Your teammate will swap these 3 names to their class names)
            ocr_engine   = MockOCREngine()
            eqn_engine   = MockEqnEngine()
            tflite_model = MockTFLiteModel()
            
            tflite_model.loadModel()
            text_result = ocr_engine.extract_text(self.image_path)
            eqn_result  = eqn_engine.recognizeEquation(self.image_path)
            
            final_output = f"--- DETECTED STANDARD TEXT ---\n{text_result}\n\n\n--- DETECTED MATHEMATICAL EQUATION ---\n{eqn_result}"
            
            self.output_text.insert(tk.END, final_output)
            
            # Wake up Save button, update Status
            self.save_btn.config(state=tk.NORMAL, bg="#0D9488", fg="white", cursor="hand2")
            self.status_label.config(text="Status: Inference Complete.", fg="#10B981") # Green
            
        except Exception as e:
            self.status_label.config(text="Status: Error Occurred.", fg=MAPUA_RED)
            messagebox.showerror("Engine Error", f"Backend failed to respond:\n{str(e)}")

    def save_result(self):
        result_content = self.output_text.get(1.0, tk.END).strip()
        if not result_content: return
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Document", "*.txt")], title="Export Output")
        if file_path:
            with open(file_path, "w") as f: f.write(result_content)
            messagebox.showinfo("Export Success", "Saved to Data Access Layer.")

    # [NEW] Logic to reset the UI
    def clear_ui(self):
        """ Resets the GUI back to its initial state """
        self.image_path = None
        self.image_label.configure(image="", text="No Image Uploaded\n(Supported: JPG, PNG)")
        self.image_label.image = None
        self.output_text.delete(1.0, tk.END)
        
        self.status_label.config(text="Status: Waiting for Image...", fg=TEXT_MUTED)
        
        # Put buttons back to sleep
        self.process_btn.config(state=tk.DISABLED, bg="#E2E8F0", fg="#94A3B8", cursor="arrow")
        self.save_btn.config(state=tk.DISABLED, bg="#E2E8F0", fg="#94A3B8", cursor="arrow")
        self.clear_btn.config(state=tk.DISABLED, bg="#E2E8F0", fg="#94A3B8", cursor="arrow")

if __name__ == "__main__":
    root = tk.Tk()
    app = OcrGuiApp(root)
    root.mainloop()