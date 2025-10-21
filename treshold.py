import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext
from PIL import Image, ImageTk
import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ImageAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analisis Citra - Histogram, Threshold, Equalization")
        self.root.geometry("1400x900")

        # --- Variabel utama ---
        self.cv_image = None
        self.gray_image = None

        # --- Main Frame ---
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Frame Kiri (Kontrol dan Citra) ---
        left_frame = tk.Frame(main_frame, width=800)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        control_frame = tk.Frame(left_frame)
        control_frame.pack(pady=10)
        self.btn_upload = tk.Button(control_frame, text="📤 Upload Citra", font=("Arial", 12, "bold"), command=self.upload_image)
        self.btn_upload.pack()

        image_notebook = ttk.Notebook(left_frame)
        image_notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_original = ttk.Frame(image_notebook)
        self.tab_biner = ttk.Frame(image_notebook)
        self.tab_equalized = ttk.Frame(image_notebook)

        image_notebook.add(self.tab_original, text='Citra Asli & Grayscale')
        image_notebook.add(self.tab_biner, text='Citra Biner')
        image_notebook.add(self.tab_equalized, text='Citra Hasil Equalization')

        self.lbl_original = tk.Label(self.tab_original)
        self.lbl_original.pack(side=tk.LEFT, padx=5, expand=True)
        self.lbl_grayscale = tk.Label(self.tab_original)
        self.lbl_grayscale.pack(side=tk.LEFT, padx=5, expand=True)
        self.lbl_biner = tk.Label(self.tab_biner)
        self.lbl_biner.pack(padx=5, expand=True)
        self.lbl_equalized = tk.Label(self.tab_equalized)
        self.lbl_equalized.pack(padx=5, expand=True)

        # --- Frame Kanan (Info dan Histogram) ---
        right_frame = tk.Frame(main_frame, width=600)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

        info_frame = ttk.LabelFrame(right_frame, text="Informasi")
        info_frame.pack(fill=tk.X, pady=5)
        self.lbl_threshold = tk.Label(info_frame, text="Nilai Threshold Otsu: -", font=("Arial", 12))
        self.lbl_threshold.pack(pady=5, padx=10, anchor='w')

        hist_notebook = ttk.Notebook(right_frame)
        hist_notebook.pack(fill=tk.BOTH, expand=True)
        
        self.tab_hist_plot = ttk.Frame(hist_notebook)
        self.tab_hist_data = ttk.Frame(hist_notebook)
        hist_notebook.add(self.tab_hist_plot, text='Grafik Histogram')
        hist_notebook.add(self.tab_hist_data, text='Data Angka Histogram')
        
        # Area untuk plot
        self.fig_rgb = Figure(figsize=(6, 4), dpi=100)
        self.ax_rgb = self.fig_rgb.add_subplot(111)
        self.canvas_rgb = FigureCanvasTkAgg(self.fig_rgb, master=self.tab_hist_plot)
        
        self.fig_gray = Figure(figsize=(6, 4), dpi=100)
        self.ax_gray = self.fig_gray.add_subplot(111)
        self.canvas_gray = FigureCanvasTkAgg(self.fig_gray, master=self.tab_hist_plot)
        
        self.canvas_rgb.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_gray.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Area untuk data angka
        self.txt_hist_data = scrolledtext.ScrolledText(self.tab_hist_data, wrap=tk.WORD, font=("Courier", 10))
        self.txt_hist_data.pack(fill=tk.BOTH, expand=True)

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        
        self.cv_image = cv2.imread(file_path)
        self.gray_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        
        self.display_images()
        self.calculate_and_show_histograms()
        self.binarize_image()
        self.equalize_image()

    def display_images(self):
        # Tampilkan Citra Asli
        self.display_image_on_label(self.cv_image, self.lbl_original, 400)
        # Tampilkan Citra Grayscale
        self.display_image_on_label(self.gray_image, self.lbl_grayscale, 400)
        
    def calculate_and_show_histograms(self):
        # Kosongkan data sebelumnya
        self.txt_hist_data.delete('1.0', tk.END)
        self.ax_rgb.clear()
        self.ax_gray.clear()

        # 1. Histogram RGB
        colors = ('b', 'g', 'r')
        self.ax_rgb.set_title("Histogram RGB")
        self.txt_hist_data.insert(tk.END, "--- Data Histogram RGB ---\n")
        for i, color in enumerate(colors):
            hist = cv2.calcHist([self.cv_image], [i], None, [256], [0, 256])
            self.ax_rgb.plot(hist, color=color)
            self.txt_hist_data.insert(tk.END, f"\nChannel {color.upper()}:\n")
            self.txt_hist_data.insert(tk.END, np.array2string(hist.flatten(), max_line_width=200, separator=', '))
        self.canvas_rgb.draw()
        
        # 2. Histogram Grayscale
        hist_gray = cv2.calcHist([self.gray_image], [0], None, [256], [0, 256])
        self.ax_gray.set_title("Histogram Grayscale")
        self.ax_gray.plot(hist_gray, color='gray')
        self.canvas_gray.draw()
        
        self.txt_hist_data.insert(tk.END, "\n\n--- Data Histogram Grayscale ---\n")
        self.txt_hist_data.insert(tk.END, np.array2string(hist_gray.flatten(), max_line_width=200, separator=', '))
        
    def binarize_image(self):
        # Metode Otsu secara otomatis menemukan threshold di antara dua puncak
        threshold_val, binary_image = cv2.threshold(self.gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.lbl_threshold.config(text=f"Nilai Threshold Otsu: {int(threshold_val)}")
        self.display_image_on_label(binary_image, self.lbl_biner, 800)

    def equalize_image(self):
        equalized_img = cv2.equalizeHist(self.gray_image)
        self.display_image_on_label(equalized_img, self.lbl_equalized, 800)
    
    def display_image_on_label(self, cv_img, label, max_size):
        if len(cv_img.shape) == 2: # Grayscale
            img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        else: # Warna (BGR)
            img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            
        pil_img = Image.fromarray(img_rgb)
        pil_img.thumbnail((max_size, max_size))
        tk_img = ImageTk.PhotoImage(pil_img)
        
        label.config(image=tk_img)
        label.image = tk_img

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAnalyzerApp(root)
    root.mainloop()